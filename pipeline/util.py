"""Shared helpers: logging, subprocess wrappers, ffmpeg/ffprobe access."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_T0 = time.time()
_LOG_PREFIX = "[pipeline]"


def log(msg: str, *, level: str = "info") -> None:
    if level == "error":
        tag = "ERROR"
    elif level == "warn":
        tag = "warn "
    elif level == "debug":
        tag = "debug"
    else:
        tag = " info"
    sys.stderr.write(f"{_LOG_PREFIX} [{time.time() - _T0:6.1f}s] {tag} {msg}\n")
    sys.stderr.flush()


class PipelineError(RuntimeError):
    pass


class ToolError(PipelineError):
    pass


# --------------------------------------------------------------- filesystem

def require_tools() -> dict[str, str]:
    found = {}
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        if not path:
            raise PipelineError(
                f"'{name}' not found on PATH. Install with: brew install ffmpeg"
            )
        found[name] = path
    return found


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(round(seconds)), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"
    return f"{m}:{s:02d}"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- subprocess

def run(cmd: list[str], *, timeout: int = 3600, check: bool = True,
        quiet: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-25:]
        raise ToolError(
            "command failed: " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else "")
            + "\n  " + "\n  ".join(tail)
        )
    return proc


def have_filter(name: str) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == name:
            return True
    return False


# ------------------------------------------------------------------ ffprobe

@dataclass
class MediaInfo:
    path: Path
    kind: str                      # "photo" | "video"
    duration: float = 0.0          # seconds (video only)
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    rotation: int = 0
    codec: str = ""
    pix_fmt: str = ""
    # EXIF / filesystem
    captured: float = 0.0          # unix timestamp (true UTC); 0 if unknown
    captured_source: str = "mtime"
    # Seconds east of UTC for the place the photo was taken, from the camera's
    # own OffsetTimeOriginal. Day-based chaptering needs this: without it every
    # timestamp is treated as UTC, a shot taken at 23:30 in London lands on the
    # following day, and chapters split in the wrong place. 0 when unknown,
    # which is correct for cameras that record UTC.
    utc_offset: int = 0
    latitude: float = 0.0
    longitude: float = 0.0
    camera: str = ""
    filesize: int = 0
    # EXIF Orientation tag (1-8) for photos: the transform a viewer must apply
    # to the stored pixels. Kept because the render path needs it to turn a
    # sideways photo upright, and the tag is stripped once the image is loaded.
    exif_orientation: int = 1

    @property
    def display_size(self) -> tuple[int, int]:
        """Width and height as the frames will actually be seen.

        Deliberately *not* a re-application of the rotation. probe() has
        already resolved any display matrix against what ffmpeg really decodes
        (see decoded_size), and the render path leans on ffmpeg's own
        autorotate, so swapping here on top of that would transpose the frames
        a second time.
        """
        return self.width, self.height

    @property
    def aspect(self) -> float:
        w, h = self.display_size
        return (w / h) if h else 0.0

    @property
    def orientation(self) -> str:
        a = self.aspect
        if a > 1.15:
            return "landscape"
        if a < 0.87:
            return "portrait"
        return "square"


def _parse_fps(text: str | None) -> float:
    if not text or text in ("0/0", "N/A"):
        return 0.0
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(text)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: Path, kind: str) -> MediaInfo:
    """Read real stream properties + rotation sidecar."""
    info = MediaInfo(path=path, kind=kind)
    try:
        info.filesize = path.stat().st_size
    except OSError:
        pass

    proc = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ], check=False)
    if proc.returncode != 0:
        raise ToolError(f"ffprobe failed on {path.name}: {proc.stderr.strip()[:200]}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ToolError(f"ffprobe returned invalid JSON for {path.name}") from exc

    fmt = data.get("format", {}) or {}
    if kind == "video":
        try:
            info.duration = float(fmt.get("duration") or 0.0)
        except (TypeError, ValueError):
            info.duration = 0.0

    for stream in data.get("streams", []) or []:
        ctype = stream.get("codec_type")
        if ctype == "video" and not info.width:
            info.width = int(stream.get("width") or 0)
            info.height = int(stream.get("height") or 0)
            info.fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(
                stream.get("r_frame_rate")
            )
            info.codec = stream.get("codec_name") or ""
            info.pix_fmt = stream.get("pix_fmt") or ""
            rot = 0
            for sd in stream.get("side_data_list", []) or []:
                if "rotation" in sd:
                    try:
                        rot = int(sd["rotation"])
                    except (TypeError, ValueError):
                        rot = 0
            if not rot and stream.get("tags", {}).get("rotate"):
                try:
                    rot = int(stream["tags"]["rotate"])
                except (TypeError, ValueError):
                    rot = 0
            info.rotation = rot
        elif ctype == "audio" and not info.has_audio:
            info.has_audio = True

    if kind == "video" and info.duration <= 0:
        info.duration = float(fmt.get("duration") or 0.0)

    # ffmpeg auto-rotates on decode, and the render path deliberately relies on
    # that rather than applying a transform of its own. The only thing that
    # needs care is agreeing with it about which way up the frames are, so when
    # a display matrix is present and the two answers could differ, ask ffmpeg
    # what it will actually decode rather than guessing from the metadata.
    #
    # Only a quarter turn can change the reported orientation. A 180-degree
    # matrix leaves width and height untouched, so there is nothing to
    # reconcile and the decode probe must be skipped - it is pure cost.
    if kind == "video" and rot % 180:
        true_w, true_h = decoded_size(path)
        if true_w and true_h:
            info.width, info.height = true_w, true_h
    return info


def decoded_size(path: Path) -> tuple[int, int]:
    """Dimensions ffmpeg will actually produce when decoding this file.

    ffprobe's ``width``/``height`` are the post-rotation size on current
    builds, but older ones report the coded size and leave the rotation in a
    side data entry. Reading the two as equivalent silently transposes half the
    library, so this decodes a single frame and reports the truth. Cheap enough
    to call only in the ambiguous case.

    No scaling filter: a resized frame reports the size it was scaled *to*, not
    the size of the source, which is the one number this function exists to
    find. An earlier version used ``scale=320:-2`` and cheerfully reported a
    1920x1080 clip as 320x180, which then failed the 320px quality floor and
    threw the clip away as unusably small.
    """
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "frame.png"
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
             "-y", str(out)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not out.exists():
            return (0, 0)
        try:
            with Image.open(out) as im:
                return im.size
        except Exception:
            return (0, 0)


# ------------------------------------------------------------------- exiftool

_EXIFTOOL_BATCH = 8192


def exiftool_metadata(paths: list[Path]) -> dict[str, dict]:
    """Batch-read capture time / GPS / camera. Never raises.

    Returns a dict keyed by resolved path string.
    """
    if not paths or not shutil.which("exiftool"):
        return {}
    out: dict[str, dict] = {}
    wanted = [
        "DateTimeOriginal", "CreateDate", "MediaCreateDate", "TrackCreateDate",
        # The UTC offset the camera recorded alongside the wall-clock time.
        # Without it a photo taken at 00:30 in London reads as the previous day
        # and the trip gets split in the wrong place.
        "OffsetTimeOriginal", "OffsetTime", "OffsetTimeDigitized",
        "GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef",
        "Make", "Model", "ImageWidth", "ImageHeight", "Rotation",
    ]
    args = ["exiftool", "-json", "-n", "-charset", "filename=utf8"]
    for w in wanted:
        args += ["-" + w]

    # Filenames go in an argument file rather than on the command line. A real
    # trip library is thousands of files with long paths, and the combined
    # length blew past the OS argument limit with OSError: Argument list too
    # long - which is a Python-level exception, not an exiftool error, so it
    # escaped every error path here and killed the run outright.
    with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False,
                                     encoding="utf-8") as fh:
        for p in paths:
            # exiftool's argument-file format is one argument per line, with
            # quotes and backslashes escaped. Paths really do contain spaces,
            # and apostrophes are common in filenames from phones.
            # exiftool -@ reads filenames one per line, one argument per line. Writing
            # the raw path avoids having to escape quotes and backslashes, which
            # is what a shell-quoted argument file would need - and getting that
            # subtly wrong makes exiftool report "File not found" for every file.
            fh.write(str(p) + "\n")
        argfile = fh.name
    try:
        proc = run(args + ["-@", argfile], timeout=3600, check=False)
    finally:
        try:
            os.unlink(argfile)
        except OSError:
            pass
    # A non-zero exit here is normal, not exceptional: exiftool reports a
    # problem for any single unreadable file (an interrupted copy from a
    # phone, a half-synced iCloud download) and still emits a complete row for
    # every file it could read. Treating the exit code as fatal and retrying
    # with just the first path threw away the metadata for the entire library,
    # which silently demoted every real capture time to a file modified time
    # and lost all GPS. So parse stdout first and only give up if it is
    # genuinely empty or unparseable.
    try:
        rows = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        rows = []
    if not isinstance(rows, list) or not rows:
        if proc.returncode != 0:
            log(f"exiftool failed on {len(paths)} file(s): "
                f"{proc.stderr.strip()[:160]}", level="warn")
        else:
            log("exiftool produced invalid JSON; ignoring metadata",
                level="warn")
        return {}
    if proc.returncode != 0:
        log(f"exiftool flagged {len(paths)} file(s) but returned metadata for "
            f"{len(rows)}; continuing with what could be read", level="warn")

    for row in rows:
        src = row.get("SourceFile") or row.get("FileName") or ""
        if not src:
            continue
        key = str(Path(src).resolve())
        out[key] = row
    return out


_DATE_FORMATS = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y:%m:%d %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y:%m:%d",
    "%Y-%m-%d",
)


def parse_utc_offset(value) -> int:
    """Seconds east of UTC from an EXIF offset string like ``+01:00``.

    Also accepts the numeric form exiftool emits with ``-n`` (minutes) and
    ``Z``. Returns 0 when there is nothing usable, which is the right answer
    for a camera that recorded UTC or for a file with no offset at all.
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        # With -n exiftool reports OffsetTime* in minutes. Someone's library
        # will contain both forms; minutes is the only sane reading of a bare
        # number here, since a raw seconds value would be absurd for an offset.
        return int(round(float(value) * 60))
    text = str(value).strip()
    if not text or text.upper() in ("Z", "UTC", "GMT"):
        return 0
    sign = -1 if text[0] == "-" else 1
    body = text.lstrip("+-")
    try:
        if ":" in body:
            hh, mm = body.split(":", 1)
            return sign * (int(hh) * 3600 + int(mm) * 60)
        return sign * int(float(body)) * 3600
    except ValueError:
        return 0


def parse_exif_datetime(value, utc_offset: int = 0) -> float:
    """Convert an EXIF date string to a unix timestamp. 0.0 if unparseable.

    EXIF wall-clock times are local to where the photo was taken, not UTC. The
    camera usually records the offset alongside them, so pass it here to get a
    real instant in time. With no offset the value is read as UTC, which is what
    the tag means on cameras that do not record one.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        # exiftool -n returns unix epoch seconds for date fields.
        v = float(value)
        return v if v > 100_000_000 else 0.0
    text = str(value).strip()
    if not text or text in ("0000:00:00 00:00:00",):
        return 0.0
    from datetime import datetime, timedelta, timezone
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.year < 1990:
                return 0.0
            # Treat the wall clock as local to where the shot was taken, then
            # convert to a real instant. UTC is the fallback for cameras that
            # record no offset of their own.
            tz = timezone(timedelta(seconds=utc_offset))
            return dt.replace(tzinfo=tz).timestamp()
        except ValueError:
            continue
    return 0.0


def _gps(row: dict) -> tuple[float, float]:
    lat, lon = row.get("GPSLatitude"), row.get("GPSLongitude")
    try:
        if lat is None or lon is None:
            return 0.0, 0.0
        lat = float(lat)
        lon = float(lon)
        if str(row.get("GPSLatitudeRef", "N")).upper().startswith("S"):
            lat = -abs(lat)
        if str(row.get("GPSLongitudeRef", "E")).upper().startswith("W"):
            lon = -abs(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return 0.0, 0.0
        return lat, lon
    except (TypeError, ValueError):
        return 0.0, 0.0


# ---------------------------------------------------------------------- misc

def default_jobs() -> int:
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu - 1))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
