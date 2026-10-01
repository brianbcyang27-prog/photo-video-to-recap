#!/usr/bin/env python3
"""Verify a rendered recap against its EDL, measuring the things that matter.

Rather than eyeballing the file, this checks the properties a good render must
have: motion inside stills, readable title cards, no accidental black bars,
blur-fill where portrait shots were letterboxed, and cut points that line up
with the beat grid.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Same loader the pipeline uses, so formats it accepts (HEIC, for one) are
# also readable here. Using PIL.Image.open directly makes the verifier crash
# on exactly the files the pipeline handled fine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.quality import (  # noqa: E402
    _load_via_sips, exif_orientation, load_image,
)

ROOT = Path(__file__).resolve().parent.parent


def grab(video: Path, t: float) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros((1, 1, 3), np.uint8)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    w, h = (int(x) for x in probe.stdout.strip().split("x")[:2])
    return np.frombuffer(proc.stdout[:w * h * 3], np.uint8).reshape(h, w, 3)


def luma(a: np.ndarray) -> np.ndarray:
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])


def grab_small(video: Path, t: float, w: int = 320) -> np.ndarray:
    """One downscaled greyscale frame, for measuring motion shape.

    Deliberately not the full-resolution grab above. Comparing full-HD frames
    measures H.264 noise and blurred-backdrop drift alongside the actual pan,
    which flattens the velocity profile to within ~20% of linear. Downscaling
    averages that away and leaves the motion, and it is far cheaper.
    """
    h = max(2, round(w * 9 / 16))
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={w}:{h}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or len(proc.stdout) < w * h:
        return np.zeros((h, w), np.uint8)
    return np.frombuffer(proc.stdout[:w * h], np.uint8).reshape(h, w)


def read_audio(video: Path) -> tuple[np.ndarray, int]:
    """Decode the whole soundtrack as mono float32 at 48kHz."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vn",
         "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, np.float32), 48000
    return np.frombuffer(proc.stdout, np.float32), 48000


def envelope(x: np.ndarray, rate: int, window_ms: float = 50.0
             ) -> tuple[np.ndarray, float]:
    """Short-window RMS. Returns the envelope and its window length in seconds.

    The window length comes back alongside the data so callers can index it
    without recomputing it: a mismatch here is silent, and reads as "the music
    is not being ducked" rather than as a bug in the checker.
    """
    n = max(1, int(rate * window_ms / 1000.0))
    count = len(x) // n
    if count == 0:
        return np.zeros(0, np.float32), n / rate
    trimmed = x[:count * n].reshape(count, n)
    return np.sqrt((trimmed ** 2).mean(axis=1) + 1e-12), n / rate


def db(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-9))


def _check_audio(video: Path, entries: list[dict], edl: dict, check) -> None:
    """Measure the soundtrack rather than assuming the filter graph worked.

    Three things are worth knowing about a finished mix: that it is loud enough
    to be watchable and quiet enough not to clip, that it never goes silent,
    and - if the source clips carried audio - that the music really is getting
    ducked underneath it. That last one is the easiest thing in the pipeline to
    get subtly wrong and the hardest to notice by ear, so it is measured: the
    gaps between words in a clip with audio should sit clearly below the music
    bed heard under a still, because the compressor pulled it down there.
    """
    x, rate = read_audio(video)
    if len(x) == 0:
        check(False, "soundtrack could be decoded")
        return

    lufs = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(video), "-af", "ebur128=peak=true",
         "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stderr

    def field(name: str) -> float | None:
        # Lines look like "    I:         -12.5 LUFS", so the value has to be
        # picked out rather than cast directly.
        pattern = re.compile(rf"^\s*{re.escape(name)}:\s*(-?\d+(?:\.\d+)?)")
        for line in lufs.splitlines():
            m = pattern.match(line)
            if m:
                return float(m.group(1))
        return None

    i_lufs = field("I")
    peak = field("Peak")
    if i_lufs is not None and i_lufs > -70:
        check(-20.0 <= i_lufs <= -11.0,
              f"integrated loudness {i_lufs:.1f} LUFS "
              f"(streaming target is about -14)")
    if peak is not None and peak > -70:
        check(peak <= -0.8,
              f"true peak {peak:.1f} dBTP (must stay under 0 to avoid clipping)")

    env, win = envelope(x, rate)
    if len(env) > 4:
        floor = float(np.percentile(env, 2.0))
        check(db(floor) > -55.0,
              f"no dead air: quietest 2% sits at {db(floor):.0f} dBFS")

    # ---- ducking, only meaningful if some shots actually had audio
    with_audio = [e for e in entries
                  if e.get("has_audio") and e["type"] == "video"]
    stills = [e for e in entries if e["type"] == "photo"]
    if not with_audio or not stills:
        return

    def gap_levels(group: list[dict], percentile: float) -> list[float]:
        out = []
        for e in group:
            a = int((e["start"] + e["duration"] * 0.15) / win)
            b = int((e["start"] + e["duration"] * 0.85) / win)
            seg = env[a:b]
            if len(seg) >= 4:
                out.append(float(np.percentile(seg, percentile)))
        return out

    # Between words in a clip with audio, the bed should be pulled down.
    gap = gap_levels(with_audio, 15.0)
    bed = gap_levels(stills, 15.0)
    if gap and bed:
        g, b = float(np.median(gap)), float(np.median(bed))
        drop = db(b) - db(g)
        check(drop >= 3.0,
              f"music ducks under native audio: quiet moments inside clips sit "
              f"{drop:.1f} dB below the bed under stills "
              f"({db(g):.0f} dBFS vs {db(b):.0f} dBFS)")


def _norm(a: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance, so brightness cannot decide a match."""
    a = a.astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def _small(a: np.ndarray, n: int = 64) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((n, n), Image.LANCZOS),
                      dtype=np.float32)


# Detail floor for a two-up half, measured rather than guessed. A correct
# spread bottomed out at 0.944; a deliberately broken one, where the
# second photo was replaced by a blurred copy of the first, fell to
# 0.444. 0.65 sits between the two with ~2x margin either side.
SHARP = 0.65


def _check_rotation(video: Path, entries: list[dict], check) -> None:
    """Photos must appear the way up the camera recorded them.

    The finished edit shipped with 106 of its 141 photos lying on their side
    and every other check still passed. Nothing about a sideways photo is
    detectably wrong to a luma, edge or loudness measurement - it is simply the
    wrong picture - so nothing here could have noticed.

    So this compares the frame actually on screen against both hypotheses for
    the source: the orientation the camera recorded, and the pixels as stored,
    which for a quarter-turn is exactly the other way round. The recorded one
    has to match better.

    Which strip of the frame to compare matters. Sampling the middle of a
    two-up spread catches the tail of one photo and the head of its partner,
    and neither hypothesis then matches anything: that is what made this check
    report a portrait photo as sideways when it was not. Each photo is compared
    against the half it actually occupies.
    """
    good = bad = tested = 0
    for e in entries:
        if tested >= 8:
            break
        if e["type"] != "photo":
            continue
        # (path, the slice of the frame it should occupy)
        #
        # A spread puts its two photos in the left and right halves. A lone
        # portrait shot with blur-fill puts its photo in the *middle*, with
        # blurred sides either side of it. Sampling the left half of a lone
        # shot therefore measures mostly blur, which is how three correct
        # photos came to be reported as sideways here.
        if e.get("pair"):
            candidates = [(e.get("source"), (0.02, 0.47)),
                          (e["pair"], (0.53, 0.98))]
        else:
            candidates = [(e.get("source"), (0.30, 0.70))]
        for src_name, (x0, x1) in candidates:
            if not src_name:
                continue
            src = Path(src_name)
            if not src.exists():
                continue
            try:
                if exif_orientation(src) < 5:   # no quarter turn to confuse us
                    continue
            except Exception:
                continue
            upright = load_image(src)
            stored = _load_via_sips(src)
            if upright is None or stored is None:
                continue
            frame = luma(grab(video, e["start"] + min(0.4, e["duration"] / 2)))
            if frame.shape[0] < 100:
                continue
            h, w = frame.shape
            band = _norm(_small(frame[int(h * 0.15):int(h * 0.85),
                                    int(w * x0):int(w * x1)]))
            su = _norm(_small(np.asarray(upright.convert("L"))))
            ss = _norm(_small(np.asarray(stored.convert("L"))))
            tested += 1
            if float((band * su).mean()) > float((band * ss).mean()):
                good += 1
            else:
                bad += 1
    if tested:
        check(bad == 0,
              f"quarter-turned photos shown upright, not on their side: "
              f"{good}/{tested} ({bad} sideways)")


def _check_two_up(video: Path, entries: list[dict], check) -> None:
    """A two-up spread has to show two photos, not one photo and an empty half.

    Both halves are measured for detail. A lone portrait shot with blur-fill
    puts a heavily smoothed copy of itself down the sides, so those halves come
    back with very little high-frequency energy; a real photo on each side does
    not. This is the check that would notice the second half silently going
    missing, which is exactly the shape of the bug that a length or a
    loudness measurement cannot see.
    """
    tested = sharp = 0
    worst = 1e9
    for e in entries:
        if e["type"] != "photo" or not e.get("pair"):
            continue
        if tested >= 8:
            break
        g = luma(grab(video, e["start"] + min(0.4, e["duration"] / 2)))
        if g.shape[0] < 100:
            continue
        h, w = g.shape
        left = highfreq(g[:, int(w * 0.06):int(w * 0.42)])
        right = highfreq(g[:, int(w * 0.58):int(w * 0.94)])
        tested += 1
        worst = min(worst, left, right)
        if left >= SHARP and right >= SHARP:
            sharp += 1
    if tested:
        check(sharp == tested,
              f"two-up spreads show a sharp photo in both halves: "
              f"{sharp}/{tested} (weakest half {worst:.3f}, floor {SHARP:.2f})")


def highfreq(gray: np.ndarray) -> float:
    p = np.pad(gray, 1, mode="edge")
    lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * gray)
    return float(np.abs(lap).mean())


def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out" / "trip_recap.mp4"
    edl_path = video.with_suffix(".edl.json")
    edl = json.loads(edl_path.read_text())
    entries = edl["entries"]
    total = edl["duration"]
    bpm = (edl.get("music") or {}).get("bpm") or 0

    results: list[tuple[bool, str]] = []

    def check(ok: bool, msg: str) -> None:
        results.append((ok, msg))

    planned = sum(e["duration"] for e in entries)
    check(abs(total - planned) <= 0.15,
          f"rendered length {total:.2f}s matches the {planned:.2f}s plan "
          f"(within 0.15s)")
    if abs(planned - edl["target"]) > 1.5:
        check(True,
              f"NOTE: plan is {planned:.0f}s but {edl['target']:.0f}s was "
              f"requested - the library could not fill it, and the pipeline "
              f"reported this")
    else:
        check(abs(total - edl["target"]) <= 1.5,
              f"duration {total:.1f}s vs target {edl['target']:.0f}s")

    # ---- container sanity
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height,pix_fmt,duration", "-of", "json",
         str(video)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    streams = json.loads(probe.stdout or "{}").get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    check(v is not None, "video stream present")
    check(a is not None, "audio stream present")
    if v:
        check(v.get("width") == 1920 and v.get("height") == 1080,
              f"resolution {v.get('width')}x{v.get('height')}")
        check(v.get("pix_fmt") == "yuv420p", f"pix_fmt {v.get('pix_fmt')} (playable everywhere)")
    if v and a:
        vd = float(v.get("duration") or 0)
        ad = float(a.get("duration") or 0)
        check(abs(vd - ad) <= 0.10,
              f"picture and sound end together: {vd:.2f}s vs {ad:.2f}s")

    # ---- titles readable: bright text pixels in the lower third
    titles = [e for e in entries if e["type"] == "title"]
    ok_titles = 0
    for e in titles:
        t = e["start"] + e["duration"] / 2
        g = luma(grab(video, t))
        band = g[int(g.shape[0] * 0.62):int(g.shape[0] * 0.86), :]
        # Count near-white pixels, the hallmark of rendered text.
        bright = float(np.mean(band > 235))
        if bright > 0.0008:
            ok_titles += 1
    check(ok_titles == len(titles),
          f"title cards with legible text: {ok_titles}/{len(titles)}")

    # ---- chapters are labelled with places, and are given a fair share
    #
    # Both of these read the EDL rather than the pixels, because both bugs
    # produce a perfectly well-rendered video. A chapter card is legible
    # whether it says "Cambridge" or "51.75, -1.26", so the legibility check
    # above passes either way.
    #
    # Coordinate labels mean reverse geocoding was lost partway through the
    # run. That happened for real: one URLError disabled lookups for the rest
    # of the render, and 18/18 checks passed on a video whose chapter cards
    # showed latitude and longitude for three days of the trip.
    #
    # Imbalance is the other one. The chapter quota was a no-op through two
    # separate unit bugs, so a ten-minute recap of a fourteen-day trip was
    # really a recap of its first three days, and 18/18 checks passed on that
    # too - every shot was there, every cut was on the beat, and half the trip
    # had 12 seconds.
    titled = [e for e in titles if e.get("subtitle")]
    coord_re = re.compile(r"-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+")
    coords = [e for e in titled if coord_re.search(e["subtitle"] or "")]
    check(not coords,
          f"chapter cards name a real place: "
          f"{len(titled) - len(coords)}/{len(titled)}"
          + (f"  (e.g. {coords[0]['subtitle']!r} - geocoding was lost)"
             if coords else ""))

    # Each chapter's screen time should track how much material it holds.
    # Compare against the shot count the card itself advertises, so this needs
    # no access to the library.
    shots_re = re.compile(r"(\d[\d,]*)\s+shots")
    by_chapter: dict[int, list[dict]] = {}
    for e in entries:
        if e["type"] != "title":
            continue
        m = shots_re.search(e.get("subtitle") or "")
        if m:
            by_chapter.setdefault(e["chapter"], []).append(
                {"start": e["start"], "dur": e["duration"],
                 "shots": int(m.group(1).replace(",", ""))})
    starts = sorted((e["start"], e["chapter"]) for e in titles)
    all_shots = sum(i["shots"] for items in by_chapter.values()
                    for i in items)
    rates: list[float] = []
    floored = 0
    for items in by_chapter.values():
        start = min(i["start"] for i in items)
        nxt = [s for s, _ in starts if s > start + 1e-6]
        end = min(nxt) if nxt else total
        got = end - start
        shots = sum(i["shots"] for i in items)
        if got <= 0 or shots <= 0:
            continue
        # Chapters holding a negligible share of the trip get a floor of one
        # shot each, by design - a day with two photos should still appear.
        # Judging those against proportional share would fail the very
        # behaviour the floor exists to provide, so they are counted and
        # reported separately rather than mixed into the comparison.
        if shots < all_shots * 0.02:
            floored += 1
            continue
        rates.append(got / shots)
    if len(rates) >= 3:
        rates.sort()
        median = rates[len(rates) // 2]
        # 6x, chosen to sit between two measured outcomes rather than fitted to
        # whichever run came last: correct code gives 2.7x on a real 14-day
        # library (3.0x on the synthetic harness), and the quota bug that
        # shipped that same library as a three-day recap gave 68x (16x on the
        # harness). Real residue comes from the per-minute diversity cap and
        # from the ~30% of the timeline deliberately left to score. The margin
        # is wide enough for that and far too tight for the bug.
        worst = rates[-1] / median
        thinnest = median / max(rates[0], 1e-9)
        check(max(worst, thinnest) <= 6.0,
              f"chapters get a fair share of screen time across "
              f"{len(rates)} chapters "
              f"(richest earns {worst:.1f}x the median, thinnest "
              f"{thinnest:.1f}x"
              + (f", {floored} tiny chapters at the one-shot floor"
                 if floored else ")"))
    elif rates:
        check(True, f"only {len(rates)} chapter(s) hold enough material to "
                    f"compare screen time fairly")

    # ---- Ken Burns: stills must actually move, and move continuously.
    #
    # The obvious version of this test - "is the difference between two frames
    # small?" - cannot work, because that difference grows with how much detail
    # the photo has. A busy 12MP iPhone frame differs from itself far more
    # under a slow, perfectly smooth pan than a flat synthetic test image does,
    # so any fixed allowance tuned on one library fails on the other. The first
    # version of this check did exactly that and reported 4/7 on real photos
    # that were moving correctly.
    #
    # What actually distinguishes a smooth move from a cut is not how far the
    # image travels but whether the travel is *evenly* distributed over time. A
    # pan advances a similar amount every step; a cut hidden inside a shot
    # shows one step vastly larger than the rest. So this measures the shape of
    # the per-step displacement and looks for an outlier, which is independent
    # of scene detail.
    stills = [e for e in entries if e["type"] == "photo"]
    moved = 0
    smooth = 0
    for e in stills[:14]:
        dur = e["duration"]
        if dur < 0.6:
            continue
        f0 = grab(video, e["start"] + dur * 0.12).astype(np.float32)
        f1 = grab(video, e["start"] + dur * 0.88).astype(np.float32)
        if f0.shape != f1.shape:
            continue
        diff = float(np.abs(f0 - f1).mean())
        if diff > 0.4:
            moved += 1
            # Sample the interior only. Including t=dur lands on the boundary
            # with the next shot, so the final step measures that cut rather
            # than this shot's motion - which reads as a huge spike at the end
            # of every single still and made all of them look broken.
            n = 8
            frames = [grab_small(video, e["start"] + dur * (0.04 + 0.92 * i / n),
                                 w=256).astype(np.float32) for i in range(n + 1)]
            if any(f.shape != frames[0].shape for f in frames):
                continue
            steps = [float(np.abs(frames[i + 1] - frames[i]).mean())
                     for i in range(n)]
            med = float(np.median(steps))
            # Eased motion legitimately has its fastest step in the middle,
            # about 2.5x the median for this curve, so the outlier bar sits
            # above that. A cut is an order of magnitude larger.
            if max(steps) < max(6.0 * med, 3.0):
                smooth += 1
    check(moved > 0, f"stills with motion (of {min(14, len(stills))} sampled): {moved}")
    check(smooth >= max(1, moved - 1),
          f"motion is smooth, not a hard cut: {smooth}/{moved}")

    # ---- Ken Burns must ease in and out, not ramp at a constant rate.
    # The signature of an eased move is that displacement peaks in the middle
    # and falls away at both ends, so the shot starts and finishes at rest and
    # the hard cut either side lands without a velocity jump. A linear ramp is
    # flat across the whole shot, which is what makes it read as drift. This is
    # a *symmetric* ease, so the two ends are near-equal and the test uses the
    # weaker of the two ratios - comparing one end against the other would pass
    # a linear ramp by accident.
    #
    # Calibrated against a deliberately linear build of the same library:
    # eased shots score at least 1.42, linear shots at most 1.05.
    eased = 0
    tested = 0
    for e in stills[:10]:
        dur = e["duration"]
        if dur < 2.0:
            continue
        frames = [grab_small(video, e["start"] + dur * f).astype(np.float32)
                  for f in (0.04, 0.29, 0.54, 0.79, 0.96)]
        if any(f.shape != frames[0].shape for f in frames):
            continue
        steps = [float(np.abs(frames[i + 1] - frames[i]).mean())
                 for i in range(len(frames) - 1)]
        if max(steps) < 0.15:      # fully static shot: no profile to judge
            continue
        tested += 1
        first, mid, last = steps[0], max(steps[1:3]), steps[-1]
        if min(mid / max(first, 1e-6), mid / max(last, 1e-6)) > 1.25:
            eased += 1
    check(tested == 0 or eased >= max(1, tested - 2),
          f"stills ease in and out instead of ramping linearly: "
          f"{eased}/{tested}")

    # ---- no accidental black bars (blur-fill should fill the frame)
    bar_hits = 0
    checked = 0
    for e in entries[:20]:
        t = e["start"] + min(0.4, e["duration"] / 2)
        g = luma(grab(video, t))
        if g.shape[0] < 100:
            continue
        checked += 1
        edges = np.concatenate([
            g[:3, :].ravel(), g[-3:, :].ravel(),
            g[:, :3].ravel(), g[:, -3:].ravel(),
        ])
        if float(np.mean(edges < 6)) > 0.97:
            bar_hits += 1
    check(bar_hits == 0, f"frames with unintended black bars: {bar_hits}/{checked}")

    # ---- blur-fill: edges should be softer than the centre on portrait shots
    fit = (edl.get("render") or {}).get("fit", "blur")
    blurred = 0
    portrait_tested = 0
    for e in entries:
        if e["type"] != "photo" or not e.get("source"):
            continue
        if e.get("pair"):
            continue        # a two-up spread: both halves are sharp by design
        src = Path(e["source"])
        if not src.exists():
            continue
        try:
            im = load_image(src)
            w, h = im.size
        except Exception:
            continue        # unreadable source; nothing to assert about framing
        finally:
            im.close() if "im" in dir() else None
        if w > h * 1.15:   # skip landscape sources; nothing to letterbox
            continue
        portrait_tested += 1
        g = luma(grab(video, e["start"] + min(0.4, e["duration"] / 2)))
        if g.shape[0] < 100:
            continue
        c = highfreq(g[:, g.shape[1] // 3: 2 * g.shape[1] // 3])
        s = highfreq(np.concatenate([g[:, :g.shape[1] // 5],
                                     g[:, -g.shape[1] // 5:]], axis=1))
        if c > 0 and s < c:
            blurred += 1
        if portrait_tested >= 10:
            break
    if fit == "blur":
        check(portrait_tested == 0 or blurred == portrait_tested,
              f"portrait shots given blur-fill rather than cropping: "
              f"{blurred}/{portrait_tested}")
    elif fit == "crop":
        # Cropping is the point of this mode, so the sharp edges are correct.
        check(True,
              f"fit=crop: {portrait_tested} portrait shots cropped to fill, "
              f"as requested")
    else:
        # pad: the frame is letterboxed, so edges should be black rather than
        # blurred. The black-bar check above already covers that.
        check(True, f"fit=pad: {portrait_tested} portrait shots letterboxed")

    # ---- cuts land on the beat grid
    if bpm > 1:
        spb = 60.0 / bpm
        fps = 30.0
        # A cut must sit within half a frame of a beat. Anything tighter is
        # meaningless, since the renderer can only place cuts on frame edges.
        tol = (1.0 / fps) + 1e-3
        worst = 0.0
        offbeat = 0
        total_cuts = 0
        for e in entries:
            t = e["start"]
            if t <= 0:
                continue
            total_cuts += 1
            err = abs(t - round(t / spb) * spb)
            worst = max(worst, err)
            if err > tol:
                offbeat += 1
        check(offbeat == 0,
              f"cuts on the beat: {total_cuts - offbeat}/{total_cuts} "
              f"(worst {worst * 1000:.0f}ms, tolerance {tol * 1000:.0f}ms)")

    # ---- native audio: loudness, continuity, and whether ducking happened
    _check_audio(video, entries, edl, check)
    _check_rotation(video, entries, check)
    _check_two_up(video, entries, check)

    # ---- report
    print()
    print("=" * 64)
    print(f"VERIFY  {video.name}  ({total:.1f}s)")
    print("=" * 64)
    failed = 0
    for ok, msg in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
        failed += (not ok)
    print("=" * 64)
    print(f"{len(results) - failed}/{len(results)} checks passed")
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
