"""Stage 1 - ingest.

Walk the media folder, work out what every file actually is, pull real capture
timestamps from EXIF, and reject anything unusable before we spend time
scoring it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import AUDIO_EXT, PHOTO_EXT, VIDEO_EXT, Pipeline
from .util import (
    MediaInfo, PipelineError, ToolError, exiftool_metadata, log, parse_exif_datetime,
    _gps, probe,
)

# Files that are almost never useful in a recap.
_JUNK_STEMS = {
    "screenshot", "screen shot", "screencapture", "snap", "img",
    "untitled", "wechat", "zalo", "line", "messenger", "telegram",
}
_JUNK_EXACT = {".ds_store", ".nomedia", "thumbs.db"}


@dataclass
class Library:
    photos: list[MediaInfo] = field(default_factory=list)
    videos: list[MediaInfo] = field(default_factory=list)
    audio: list[Path] = field(default_factory=list)
    rejected: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def items(self) -> list[MediaInfo]:
        return self.photos + self.videos

    def summary(self) -> str:
        return (f"{len(self.photos)} photos, {len(self.videos)} videos, "
                f"{len(self.audio)} audio, {len(self.rejected)} skipped")


def _classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in PHOTO_EXT:
        return "photo"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    return "unknown"


def _looks_like_junk(path: Path, info: MediaInfo | None) -> bool:
    stem = path.stem.lower()
    if any(j in stem for j in _JUNK_STEMS) and "screenshot" in stem:
        return True
    if path.name.lower() in _JUNK_EXACT:
        return True
    if info is None:
        return False
    # Too small to be real content.
    if info.kind == "photo" and (info.width < 320 or info.height < 320):
        return True
    if info.kind == "video":
        if info.duration < 0.4:
            return True
        if info.width < 320 or info.height < 320:
            return True
    return False


def scan(root: Path, *, extra_audio_dirs: list[Path] | None = None) -> Library:
    """Build a Library from a folder of mixed media."""
    if not root.exists():
        raise PipelineError(
            f"media folder not found: {root}\n"
            f"Create it and drop your photos and videos in."
        )

    files = [p for p in sorted(root.rglob("*"))
             if p.is_file() and not p.name.startswith(".")
             and not any(part.startswith(".") for part in p.parts)]
    if not files:
        raise PipelineError(f"no files found in {root}")

    photo_paths: list[Path] = []
    video_paths: list[Path] = []
    audio_paths: list[Path] = []
    for p in files:
        kind = _classify(p)
        if kind == "photo":
            photo_paths.append(p)
        elif kind == "video":
            video_paths.append(p)
        elif kind == "audio":
            audio_paths.append(p)

    log(f"found {len(photo_paths)} photos, {len(video_paths)} videos, "
        f"{len(audio_paths)} audio files")

    for extra in (extra_audio_dirs or []):
        if extra.exists():
            for p in sorted(extra.glob("*")):
                if p.is_file() and p.suffix.lower() in AUDIO_EXT:
                    audio_paths.append(p)

    # ---- EXIF in one batch (much faster than per-file)
    meta = exiftool_metadata(photo_paths + video_paths)
    if meta:
        log(f"read EXIF for {len(meta)} files")
    else:
        log("no EXIF available; falling back to file modified times", level="warn")

    lib = Library()

    def finish(info: MediaInfo) -> MediaInfo:
        row = meta.get(str(info.path.resolve())) or {}
        info.captured = (
            parse_exif_datetime(row.get("DateTimeOriginal"))
            or parse_exif_datetime(row.get("CreateDate"))
            or parse_exif_datetime(row.get("MediaCreateDate"))
            or parse_exif_datetime(row.get("TrackCreateDate"))
        )
        if info.captured:
            info.captured_source = "exif"
        else:
            try:
                info.captured = info.path.stat().st_mtime
                info.captured_source = "mtime"
            except OSError:
                info.captured = 0.0
                info.captured_source = "none"
        info.latitude, info.longitude = _gps(row)
        make = str(row.get("Make") or "").strip()
        model = str(row.get("Model") or "").strip()
        info.camera = f"{make} {model}".strip()
        return info

    for p in photo_paths:
        if _looks_like_junk(p, None):
            lib.rejected.append((p, "screenshot/suspicious name"))
            continue
        try:
            info = probe(p, "photo")
        except ToolError as exc:
            lib.rejected.append((p, f"probe failed: {exc}"))
            continue
        if _looks_like_junk(p, info):
            lib.rejected.append((p, "too small to use"))
            continue
        lib.photos.append(finish(info))

    for p in video_paths:
        if _looks_like_junk(p, None):
            lib.rejected.append((p, "screenshot/suspicious name"))
            continue
        try:
            info = probe(p, "video")
        except ToolError as exc:
            lib.rejected.append((p, f"probe failed: {exc}"))
            continue
        if _looks_like_junk(p, info):
            lib.rejected.append((p, "too small or too short"))
            continue
        lib.videos.append(finish(info))

    lib.audio = [p for p in audio_paths if p.stat().st_size > 1024]

    if not lib.items:
        reasons = "; ".join(f"{p.name}: {why}" for p, why in lib.rejected[:5])
        raise PipelineError(
            "none of the files in the media folder are usable.\n  " + reasons
        )

    # Chronological is the default backbone of the final edit.
    lib.photos.sort(key=lambda i: (i.captured, i.path.name))
    lib.videos.sort(key=lambda i: (i.captured, i.path.name))
    return lib


def capture_span(lib: Library) -> tuple[float, float, int]:
    """(earliest, latest, distinct capture days) across everything with a date."""
    stamps = [i.captured for i in lib.items if i.captured]
    if not stamps:
        return 0.0, 0.0, 0
    from datetime import datetime, timezone
    days = {
        datetime.fromtimestamp(t, tz=timezone.utc).date()
        for t in stamps
    }
    return min(stamps), max(stamps), len(days)


def day_index(stamp: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d")
