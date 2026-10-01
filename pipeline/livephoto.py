"""Pairing iPhone Live Photos with their motion clip.

A Live Photo is two files that share a base name: a still image (usually HEIC
on a modern iPhone) and a short MOV holding the few seconds of motion and sound
that were recorded alongside it. In the Photos app the two are shown as one
picture that happens to move.

The pipeline was treating them as two unrelated files. That produced three
distinct faults, all of them visible in a finished cut:

  * the MOV was treated as ordinary video and given the reserved video share of
    the timeline, so a 2.4-second wiggle of a still competed for screen time
    against real handheld footage;
  * the still was shown as a flat pan with no sound, when the real ambient audio
    of that exact moment was sitting on disk next to it (measured around -37 dB
    mean, quiet but genuine room tone);
  * nothing stopped both halves of the same moment from being selected, so the
    same instant could appear twice.

Pairing them fixes all three. The MOV stops being a video in its own right and
becomes an optional motion track for the still it belongs to.
"""
from __future__ import annotations

from pathlib import Path

from .util import log

# Extensions an iPhone writes the motion half to, in the order we prefer them.
_MOTION_EXT = (".MOV", ".mov")


def _siblings(photo: Path) -> dict[str, Path]:
    """Every file in the photo's folder sharing its base name, by extension."""
    found: dict[str, Path] = {}
    # Case-insensitive scan: the extension's capitalisation varies by device and
    # by whether the phone or a desktop wrote the file, and a miss here is
    # silent - the photo simply loses its motion and nobody can tell why.
    try:
        for entry in photo.parent.iterdir():
            if not entry.is_file():
                continue
            stem = entry.stem
            # IMG_1234.HEIC and IMG_1234.mov -> "IMG_1234"
            if stem.casefold() == photo.stem.casefold():
                found[entry.suffix.lower()] = entry
    except OSError:
        return found
    return found


def find_motion(photo: Path) -> Path | None:
    """The Live Photo motion clip belonging to this still, if there is one."""
    for ext in _MOTION_EXT:
        found = _siblings(photo).get(ext.lower())
        if found is not None:
            return found
    return None


def motion_duration(motion: Path) -> float:
    """Length of a motion clip, cheap: trust the file size, do not probe.

    A Live Photo clip is 2-3 seconds and there may be thousands of them, so
    probing each one during scan would add minutes to a full-library run. This
    is only used to tell "a real Live Photo" from "some unrelated video that
    happens to share a name", and a wrong answer there is harmless: the real
    duration is read later, for the handful of clips actually selected.
    """
    try:
        size = motion.stat().st_size
    except OSError:
        return 0.0
    # A genuine Live Photo MOV carries video plus audio and is never tiny.
    return 20_000.0 if size > 20_000 else 0.0


def pair_library(photos: list, videos: list) -> tuple[dict, set]:
    """Attach motion clips to the stills they belong to.

    Returns ``(motion_by_photo, consumed_motions)``: a mapping from each photo
    path to its motion clip, and the set of MOV paths that are Live Photo
    halves rather than standalone footage. The caller drops the latter from the
    video pool so a moment is never queued twice.
    """
    # A Live Photo's two halves always sit in the same folder, so the index is
    # keyed by (folder, base name) rather than base name alone. On a real
    # library this matters: 1,041 base names are reused across folders because
    # the same iPhone roll was copied into several places, and a flat index
    # pairs those to each other's clips - the wrong motion on the wrong photo,
    # which looks like a plausible edit rather than an obvious failure.
    by_key: dict[tuple[str, str], Path] = {}
    for v in videos:
        by_key.setdefault((str(v.path.parent).casefold(),
                           v.path.stem.casefold()), v.path)

    motion: dict[Path, Path] = {}
    consumed: set[Path] = set()
    # Pixel counts, taken from the scan's own probe, so resolving a duplicate
    # costs nothing and does not re-open the file.
    pixels: dict[Path, int] = {}
    for photo in photos:
        key = (str(photo.path.parent).casefold(), photo.path.stem.casefold())
        candidate = by_key.get(key)
        if candidate is None:
            continue
        if not motion_duration(candidate):
            continue
        motion[photo.path] = candidate
        consumed.add(candidate)
        pixels[photo.path] = int(photo.width or 0) * int(photo.height or 0)

    # One moment can be on disk twice. Saving a Live Photo and then also
    # exporting it as a JPEG leaves IMG_0044.HEIC and IMG_0044.JPG beside a
    # single IMG_0044.mov, and both stills match it - the real library has
    # exactly this collision. Left alone, the same three seconds of motion
    # would be attached to both stills and, if selection happened to keep both,
    # the moment would play twice in a row.
    #
    # The clip belongs to the better copy of the photo: HEIF before JPEG,
    # because the JPEG is a re-encode of it, then the larger one. Both orderings
    # are deterministic, because a pairing that changed between runs would be a
    # bad thing to have to explain in a bug report.
    collisions: dict[Path, list[Path]] = {}
    for photo, clip in motion.items():
        collisions.setdefault(clip, []).append(photo)
    contested = {c: p for c, p in collisions.items() if len(p) > 1}
    for clip, claimants in contested.items():
        winner = _better_still(claimants, pixels)
        for photo in claimants:
            if photo != winner:
                del motion[photo]
    if contested:
        log(f"live photos: {len(contested)} clip(s) matched more than one copy "
            f"of the same photo; kept the best copy of each, so the moment is "
            f"not shown twice")

    if motion:
        log(f"live photos: {len(motion)} stills have motion, "
            f"{len(consumed)} clips are their halves")
    return motion, consumed


# Lower is better. HEIF first because a JPEG beside it is a re-encode of the
# same moment; the rest are a guess at how much was thrown away.
_FORMAT_RANK = {".heic": 0, ".heif": 0, ".tif": 1, ".tiff": 1, ".png": 2,
                ".jpg": 3, ".jpeg": 3, ".webp": 4}


def _better_still(candidates: list[Path], pixels: dict[Path, int]) -> Path:
    """Pick which copy of a duplicated photo keeps its Live Photo clip.

    The widest copy wins a tie within a format, and the path breaks a tie
    between two identical files, so the answer never depends on scan order.
    """
    return min(candidates, key=lambda p: (
        _FORMAT_RANK.get(p.suffix.lower(), 5),
        -pixels.get(p, 0),
        str(p)))
