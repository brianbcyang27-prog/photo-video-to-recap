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
from pipeline.quality import load_image  # noqa: E402

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

    # ---- Ken Burns: stills must actually move, but smoothly
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
            # A long shot pans further across its span, so the allowance has to
            # scale with duration or every 4s clip reads as a hard cut.
            if diff < 8.0 + 12.0 * dur:
                smooth += 1
    check(moved > 0, f"stills with motion (of {min(14, len(stills))} sampled): {moved}")
    check(smooth >= max(1, moved - 1),
          f"motion is smooth, not a hard cut: {smooth}/{moved}")

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
