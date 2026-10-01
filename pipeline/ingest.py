"""Stage 1 - ingest.

Walk the media folder, work out what every file actually is, pull real capture
timestamps from EXIF, and reject anything unusable before we spend time
scoring it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

from . import livephoto
from .config import AUDIO_EXT, PHOTO_EXT, VIDEO_EXT
from .util import (
    MediaInfo,
    PipelineError,
    ToolError,
    _gps,
    exiftool_metadata,
    log,
    parse_exif_datetime,
    parse_utc_offset,
    probe,
    probe_jobs,
    progress,
    progress_done,
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


def _unreadable(path: Path) -> str | None:
    """Catch files that never finished arriving, before ffprobe gets confused.

    An interrupted copy from a phone, or an iCloud download that gave up,
    leaves a 0-byte file. ffprobe reports these as "moov atom not found",
    which reads like a corrupt library rather than "this file is empty".
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"cannot read file ({exc.strerror or 'access denied'})"
    if size == 0:
        return "empty file (0 bytes) - looks like an interrupted copy"
    if size < 1024:
        return f"suspiciously small ({size} bytes) - looks truncated"
    return None


def _probe_reason(path: Path, exc: Exception) -> str:
    """Turn a raw ffprobe error into something a person can act on."""
    text = str(exc).lower()
    size = path.stat().st_size if path.exists() else 0
    if "moov atom" in text or "invalid data" in text:
        return (f"not a readable {path.suffix.lstrip('.').upper()} file "
                f"({size} bytes) - truncated or the wrong extension")
    return f"could not be read ({str(exc)[:80]})"


def scan(root: Path, *, extra_audio_dirs: list[Path] | None = None,
         live_photos: bool = True) -> Library:
    """Build a Library from a folder of mixed media.

    `live_photos` off leaves each still and its motion clip as two unrelated
    files, which is what this did before pairing existed - useful for
    comparing, and wrong for a finished film.
    """
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
        # Cameras record the UTC offset on photos and usually not on video, so
        # this is routinely a minority. Worth knowing: a file without one has
        # its wall clock read as UTC, which can put a late-night shot on the
        # wrong calendar day and split a chapter in two.
        withoff = sum(1 for r in meta.values()
                      if parse_utc_offset(r.get("OffsetTimeOriginal"))
                      or parse_utc_offset(r.get("OffsetTimeDigitized"))
                      or parse_utc_offset(r.get("OffsetTime")))
        if withoff < len(meta):
            log(f"{len(meta) - withoff} of {len(meta)} files record no UTC "
                f"offset; their times are read as UTC")
    else:
        log("no EXIF available; falling back to file modified times", level="warn")

    lib = Library()

    def finish(info: MediaInfo) -> MediaInfo:
        row = meta.get(str(info.path.resolve())) or {}
        # Prefer the offset recorded with the frame; fall back to any offset the
        # file carries, and finally to UTC for cameras that record neither.
        info.utc_offset = (
            parse_utc_offset(row.get("OffsetTimeOriginal"))
            or parse_utc_offset(row.get("OffsetTimeDigitized"))
            or parse_utc_offset(row.get("OffsetTime"))
        )
        off = info.utc_offset
        info.captured = (
            parse_exif_datetime(row.get("DateTimeOriginal"), off)
            or parse_exif_datetime(row.get("CreateDate"), off)
            or parse_exif_datetime(row.get("MediaCreateDate"), off)
            or parse_exif_datetime(row.get("TrackCreateDate"), off)
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
        _true_photo_size(info, row)
        return info

    # Probing is an ffprobe/ffmpeg subprocess per file, so it releases the GIL
    # for essentially all of its cost and threads scale. This was the single
    # slowest step in the whole pipeline on a real library: 11,293 files at
    # ~0.15s each is 27 minutes of waiting before selection could begin, all
    # of it serial subprocess latency. Order is restored afterwards so the
    # library is deterministic regardless of which probe finished first.
    jobs = probe_jobs()
    log(f"probing {len(photo_paths) + len(video_paths)} files with {jobs} workers")
    work = ([(p, "photo") for p in photo_paths]
            + [(p, "video") for p in video_paths])
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # as_completed rather than pool.map, so the wait can be reported. The
        # results are written back by index because the order still matters to
        # the caller: pool.map was what guaranteed work[i] went with results[i],
        # and losing that would scramble which classification belongs to which
        # file. This phase is minutes of silence on a real library otherwise,
        # which is indistinguishable from a hang.
        futures = {pool.submit(classify, p, k): i for i, (p, k) in enumerate(work)}
        results = [None] * len(work)
        done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            progress("probing files", done, len(work))
        progress_done("probing files")

    for (p, _kind), (path, info, why) in zip(work, results):
        if why:
            lib.rejected.append((path, why))
        elif _kind == "photo":
            lib.photos.append(finish(info))
        else:
            lib.videos.append(finish(info))

    lib.audio = [p for p in audio_paths if p.stat().st_size > 1024]

    if live_photos:
        _attach_live_photos(lib)

    if not lib.items:
        reasons = "; ".join(f"{p.name}: {why}" for p, why in lib.rejected[:5])
        raise PipelineError(
            "none of the files in the media folder are usable.\n  " + reasons
        )

    # Chronological is the default backbone of the final edit.
    lib.photos.sort(key=lambda i: (i.captured, i.path.name))
    lib.videos.sort(key=lambda i: (i.captured, i.path.name))
    _borrow_offsets(lib)
    _report_skips(lib)
    return lib


def _attach_live_photos(lib: Library) -> None:
    """Fold each Live Photo's motion clip into the still it belongs to.

    Called after the pools are built and before anything downstream sees them,
    so the two halves of one moment are a single asset from here on. The clips
    that turn out to be Live Photo halves leave the video pool entirely: they
    are not footage competing for screen time, and leaving them in would let the
    same instant be queued twice.
    """
    if not lib.photos or not lib.videos:
        return
    motion, consumed = livephoto.pair_library(lib.photos, lib.videos)
    if not motion:
        return
    for photo in lib.photos:
        clip = motion.get(photo.path)
        if clip is not None:
            photo.live_motion = clip
    # Re-sort: removing entries from a list that was sorted by capture time is
    # safe, but keeping the invariant explicit here means the caller never has
    # to remember which pool entries came out of a pairing pass.
    lib.videos = [v for v in lib.videos if v.path not in consumed]


def _true_photo_size(info: MediaInfo, row: dict) -> None:
    """Replace ffprobe's photo dimensions with the ones exiftool read.

    ffprobe reports the dimensions of whatever stream it finds first, and for a
    HEIC that is the embedded thumbnail rather than the image: a 4032x3024
    iPhone photo came back as 512x512 and classified as square, another as
    640x896 and classified portrait. 6,087 of this library's 6,972 photos are
    HEIC, so orientation was effectively a coin flip.

    Nothing downstream acted on it - MediaInfo.orientation was dead code - so
    this never damaged a finished render. It did make the "too small to be real
    content" junk check in classify() meaningless for HEICs, since a 200x200
    photo still carries a 512x512 thumbnail, and it would mislead the first
    thing that did care about orientation.

    exiftool already reads ImageWidth/ImageHeight for every file as part of the
    same batch call, so the correct dimensions were on hand and unused.

    A quarter-turn orientation (EXIF 5-8) means the stored axes are transposed
    relative to how the photo displays, so the swap is applied here; 180 and
    mirror values leave the axes alone.

    It reads the Orientation tag, not Rotation. exiftool's Rotation is a
    derived, inconsistently-reported column - under -n it came back as 3 for a
    photo whose Orientation was plainly 6 - so the swap it guarded never fired,
    and the 100 Orientation-6 photos in the finished edit were reported as the
    4:3 landscape they are stored as rather than the portrait they display as.
    That is what let the renderer centre-crop them instead of giving them the
    blur treatment, on top of showing them lying on their side.

    The renderer applies its own transpose for display; this only corrects the
    reported size, and hands the tag on so it does not have to ask again.
    """
    if info.kind != "photo":
        return
    try:
        o = int(row.get("Orientation") or 1)
    except (TypeError, ValueError):
        o = 1
    info.exif_orientation = o if 1 <= o <= 8 else 1
    try:
        w = int(row.get("ImageWidth") or 0)
        h = int(row.get("ImageHeight") or 0)
    except (TypeError, ValueError):
        return
    if w <= 0 or h <= 0:
        return
    if info.exif_orientation in (5, 6, 7, 8):
        w, h = h, w
    info.width, info.height = w, h


def _borrow_offsets(lib: Library) -> None:
    """Give files that record no UTC offset the one their neighbours used.

    iPhones write OffsetTimeOriginal onto stills and leave it off video, so in a
    real library roughly half the files have no offset of their own. Reading
    those as UTC is not a small error: a clip shot at 00:30 in Zermatt is
    stamped two hours late and lands on the previous calendar day, which splits
    a chapter and starts the next one at the wrong hour.

    Borrowing from the nearest file that does record one keeps a trip's
    timezone changes intact - an England morning next to a Swiss afternoon
    keeps its own +01:00 or +02:00 - because the neighbour is from the same
    place and the same day, not from whatever offset dominates the library.
    """
    timed = sorted((i for i in lib.items if i.captured), key=lambda i: i.captured)
    if not timed:
        return
    known = [i for i in timed if i.utc_offset]
    if not known:
        return
    filled = 0
    for info in timed:
        if info.utc_offset:
            continue
        best = min(known, key=lambda k: abs(k.captured - info.captured))
        # Only trust a neighbour taken within a few hours: someone flying
        # across the Atlantic mid-trip really did change timezone, and a
        # distant borrow would silently apply the wrong one.
        if abs(best.captured - info.captured) > 6 * 3600:
            continue
        # The timestamp itself has to move, not just the offset we report. A
        # file with no offset was read as UTC, so its stored time is really a
        # wall clock. Recording the offset and leaving the number alone would
        # still put it two hours from a photo taken in the same minute, and
        # sorting by capture time would interleave the two files wrongly.
        info.captured -= best.utc_offset
        info.utc_offset = best.utc_offset
        filled += 1
    if filled:
        log(f"borrowed the UTC offset from nearby files for {filled} "
            f"file(s) that record none")
    # Sorting again: the correction moves files across each other.
    lib.photos.sort(key=lambda i: (i.captured, i.path.name))
    lib.videos.sort(key=lambda i: (i.captured, i.path.name))


def classify(p: Path, kind: str) -> tuple[Path, MediaInfo | None, str]:
    """Junk-name, then unreadable, then probe, then size.

    One file in, one verdict out, no shared state - so scan() can run it across
    a thread pool. Kept at module level rather than nested in scan() precisely
    so a test can compare it against a serial implementation and prove the
    threading did not change which files survive.
    """
    if _looks_like_junk(p, None):
        return (p, None, "screenshot/suspicious name")
    why = _unreadable(p)
    if why:
        return (p, None, why)
    try:
        info = probe(p, kind)
    except ToolError as exc:
        return (p, None, _probe_reason(p, exc))
    if _looks_like_junk(p, info):
        return (p, None,
                "too small to use" if kind == "photo"
                else "too small or too short")
    return (p, info, "")


def _report_skips(lib: Library, limit: int = 8) -> None:
    """Name the files we left out, and why.

    A silent skip is the worst outcome here: the video comes out fine and the
    user has no idea two photos are missing. Broken-and-empty files get their
    own warning, because those are worth fixing rather than ignoring.
    """
    if not lib.rejected:
        return
    broken = [(p, w) for p, w in lib.rejected
              if "interrupted" in w or "truncated" in w or "not a readable" in w]
    for p, why in lib.rejected[:limit]:
        log(f"  skipped {p.name}: {why}", level="warn")
    if len(lib.rejected) > limit:
        log(f"  ... and {len(lib.rejected) - limit} more skipped file(s); "
            "the full list is in the report", level="warn")
    if broken:
        log(f"{len(broken)} file(s) look damaged - re-copy them from the "
            "original device to recover them", level="warn")


def capture_span(lib: Library) -> tuple[float, float, int]:
    """(earliest, latest, distinct capture days) across everything with a date."""
    stamps = [i.captured for i in lib.items if i.captured]
    if not stamps:
        return 0.0, 0.0, 0
    from datetime import datetime
    days = {
        datetime.fromtimestamp(t, tz=UTC).date()
        for t in stamps
    }
    return min(stamps), max(stamps), len(days)


def day_index(stamp: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(stamp, tz=UTC).strftime("%Y-%m-%d")
