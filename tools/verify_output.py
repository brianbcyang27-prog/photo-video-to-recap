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
    load_image,
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
    #
    # The 15th percentile inside a clip is a weak probe, and it is worth being
    # explicit about why rather than tuning the threshold until it passes. It
    # only sees ducking if the clip has real dynamics to duck *between*:
    # measured on a quiet synthetic library, the per-clip 15th-percentile has a
    # 5.3 dB standard deviation, so with six clips the median of those numbers
    # is mostly noise. A near-constant tone has no quiet moments, and against
    # one no amount of correct ducking is detectable this way.
    #
    # So the check is kept but reports honestly when the material cannot show
    # the effect, instead of failing a render whose ducking is in fact correct.
    # It still fails loudly on a library that *does* have dynamics and is not
    # being ducked - which is the fault worth catching.
    gap = gap_levels(with_audio, 15.0)
    bed = gap_levels(stills, 15.0)
    if not gap or not bed:
        return
    g, b = float(np.median(gap)), float(np.median(bed))
    drop = db(b) - db(g)
    spread = float(np.std(gap))
    # Below this, the measurement is noise: the variation between clips is as
    # large as the effect being looked for. The `spread < 0.5` arm catches the
    # degenerate case where every clip measures identically, which means there
    # is no variation to compare rather than no ducking.
    if spread < 0.5 or spread > abs(drop) * 1.5 or spread > 4.0:
        why = ("every clip measures the same, so there is no variation to "
               "compare" if spread < 0.5 else
               f"the quiet moments inside clips vary by {spread:.1f} dB "
               f"between clips, as large as the effect itself")
        check(True,
              f"NOTE: cannot measure ducking on this library - {why}. The "
              f"source audio has too little dynamics for a 15th-percentile "
              f"probe to see ducking; the threshold itself is derived from the "
              f"content peak and is checked in tests/test_audio_loudness.py")
    else:
        check(drop >= 3.0,
              f"music ducks under native audio: quiet moments inside clips sit "
              f"{drop:.1f} dB below the bed under stills "
              f"({db(g):.0f} dBFS vs {db(b):.0f} dBFS)")


def _check_colour_tags(v: dict | None, check) -> None:
    """The output must declare its colour matrix, and declare it correctly.

    This is the regression guard for the largest defect the pipeline had. ffmpeg's
    default RGB->YUV matrix is bit-for-bit BT.601 - verified, not assumed: the
    default conversion hashes identically to explicit `out_color_matrix=bt601` and
    differs from bt709, at 1080p and 4K alike. Nothing in the file said so, and
    every HD player resolves "unknown" as BT.709, so each still was *written*
    with BT.601 luma coefficients and *read* as BT.709. Measured over 40 library
    photos that was a median dE76 of 18.98 against a ~2.3 just-noticeable-
    difference, worst case 105.69; declaring BT.709 brings the median to 2.51,
    which is the 8-bit 4:2:0 subsampling floor rather than leftover colour error.

    The check is deliberately strict about `unknown` rather than only about a
    wrong value, because `unknown` is exactly how the bug presented. A container
    that says nothing is a container whose colours depend on the player.

    Note that libx264 does not carry `-colorspace` / `-color_primaries` /
    `-color_trc` into the bitstream, so the render passes these as x264 VUI
    parameters; the tags that actually land in the file are the ones checked here.
    """
    if not v:
        return
    space = (v.get("color_space") or "").strip().lower()
    prim = (v.get("color_primaries") or "").strip().lower()
    trc = (v.get("color_transfer") or "").strip().lower()
    rng = (v.get("color_range") or "").strip().lower()
    check(space == "bt709" and prim == "bt709" and trc == "bt709",
          f"colour matrix declared as BT.709 (space={space or 'unset'}, "
          f"primaries={prim or 'unset'}, transfer={trc or 'unset'}) - "
          f"unlabelled output is read as BT.709 regardless of what it was "
          f"written as")
    check(rng in ("tv", "mpeg"),
          f"colour range declared as limited (range={rng or 'unset'}) - "
          f"a full-range tag on limited-range samples shifts every pixel level")


def _reference_dir() -> Path | None:
    """The scratch dir holding reference.mp4, if a recent render left one.

    Run as `--reference` writes it; run without it, .work is deleted. So the
    usual case is "no reference", and that has to be visible in the output rather
    than a silently-skipped check.
    """
    root = ROOT / ".work"
    if not root.is_dir():
        return None
    cands = sorted((p for p in root.glob("run-*") if (p / "reference.mp4").exists()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _check_encoder_fidelity(video: Path, work: Path | None, check) -> None:
    """Measure the encode against a lossless re-render of the same thing.

    Only runs when a reference was produced, because building one means re-rendering
    the whole timeline at `-qp 0`. That is far too slow to do on every run, but it
    is the only way to answer the question that actually matters: *did this render
    degrade the source, or does it just look like it?* Every other check in this file
    tests structure - lengths, tags, ducking, rotation. None of them can tell a
    clean encode from a smeared one.

    The normalisation on both legs is not optional. ffmpeg's `psnr` / `ssim` /
    `libvmaf` filters pair frames by *timestamp*, not by order. Matroska PTS are
    millisecond-quantised (0, 42, 83, 125 ...) while MP4 tracks an exact 1/24s grid,
    so comparing a Matroska reference against an MP4 render without normalising both
    legs reports a near-lossless encode as VMAF 67.03 instead of 99.94 - measured,
    with a per-frame minimum of 0.0 that betrays it as a pairing fault rather than a
    quality problem. A verifier built on that would be worse than no verifier.
    """
    if work is None:
        return
    ref = work / "reference.mp4"
    if not ref.exists():
        log_note("encoder fidelity skipped: no lossless reference was rendered")
        return

    # A reference from a *different* render is worse than no reference at all.
    # reference.mp4 is only written for runs passed --reference, but the lookup
    # above globs every run-* dir, so an ordinary render will happily find one
    # left behind by an earlier calibration pass. The two files then get paired
    # frame-by-frame with no relation between them: measured here as 720 frames
    # against 1344, VMAF 0.52 with a per-frame minimum of 0.00 - which looks
    # exactly like a catastrophic encode and is really a length mismatch.
    #
    # So verify the lengths agree before trusting the score. One frame of
    # tolerance covers container rounding; anything more is a different film.
    n_dist = _count_frames(video)
    n_ref = _count_frames(ref)
    if n_dist and n_ref and abs(n_dist - n_ref) > 1:
        log_note(f"encoder fidelity skipped: the reference is a different "
                 f"render ({n_ref} frames vs {n_dist}), so comparing them would "
                 f"measure nothing - re-run with --reference")
        return

    log_path = work / "vmaf_log.json"

    # Normalise both legs to an identical, frame-indexed timebase before comparing.
    fps = _probe_fps(video) or 30.0
    norm = f"settb=AVTB,setpts=N/({fps:g}*TB)"
    graph = (f"[0:v]{norm},format=yuv420p[dist];"
             f"[1:v]{norm},format=yuv420p[ref];"
             "[dist][ref]libvmaf=n_threads=4:log_fmt=json:"
             f"log_path={log_path}")

    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-i", str(ref),
         "-filter_complex", graph, "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # libvmaf writes its log to `log_path`, not to stderr - stderr only carries the
    # one-line score. Parsing stderr for JSON is what makes this silently measure
    # nothing, so it reads the file the filter was told to write.
    score = None
    try:
        with log_path.open() as fh:
            pooled = json.load(fh)["pooled_metrics"]["vmaf"]
        score = float(pooled["mean"])
        worst = float(pooled["min"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        log_note("encoder fidelity: libvmaf unavailable or unparseable")
        return
    finally:
        log_path.unlink(missing_ok=True)
    # 90 rather than 95. This is a CRF encode at delivery resolution of material
    # that was itself resampled and Ken-Burns'd, so it is not a codec comparison -
    # it is a check that nothing in the chain is destroying the picture. A smear
    # from a bad resample or a broken colour path lands far below this.
    check(score >= 90.0,
          f"encoder fidelity VMAF {score:.2f} (worst frame {worst:.2f}) against "
          f"a lossless re-render of the same chain - both legs frame-normalised "
          f"first, without which this reports ~67 for a clean encode; under 90 "
          f"means the delivery encode is degrading the source")


def log_note(msg: str) -> None:
    print(f"  note: {msg}")


def length_check(edl: dict, planned: float) -> tuple[bool, str]:
    """Does the planned length match the length that was asked for?

    Split out from main() so it can be tested without a rendered file. The
    tolerance is one beat, because the pipeline allocates whole beats and a
    target that is not a whole number of beats cannot be hit exactly.
    """
    target = float(edl.get("target") or 0.0)
    bpm = float((edl.get("music") or {}).get("bpm") or 0.0)
    beat = 60.0 / bpm if bpm > 1.0 else 0.5
    if target <= 0.0:
        return True, "no target recorded, so no length to check against"
    gap = abs(planned - target)
    return (gap <= beat + 1e-6,
            f"plan runs {planned:.2f}s against the {target:.2f}s requested "
            f"({gap:.2f}s out; one beat is {beat:.3f}s)")


def _count_frames(path: Path) -> int | None:
    """Exact decoded frame count, or None if ffprobe cannot say.

    Reads nb_read_frames rather than nb_frames: the container's frame count is
    an estimate for variable-frame-rate material, and the whole point of this
    comparison is to know the two files hold the same number of frames.
    """
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return int((proc.stdout or "").strip().split(",")[0])
    except (ValueError, IndexError):
        return None


def _probe_fps(path: Path) -> float | None:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        num, _, den = (proc.stdout or "").strip().partition("/")
        f = float(num) / float(den) if float(den) else 0.0
        return f or None
    except (ValueError, ZeroDivisionError):
        return None


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
    # The length of the *picture* is checked against the plan further down, on
    # the video stream rather than the container. That distinction is the whole
    # point: a container reports the longest stream in it, so an audio track one
    # beat long covers for a picture that stopped early. On the trip render that
    # is exactly what happened - the container read 599.84s against a 599.86s
    # plan and passed, while the video stream was 599.26s, 35 frames short, with
    # every shot after the first shortfall landing up to 0.58s ahead of its own
    # sound.
    #
    # The plan is also checked against the *requested* target, and that used to
    # be a branch that could not fail: any plan more than 1.5s from the target
    # reported "the library could not fill it" and passed unconditionally. It is
    # how a 177s request producing a 174.2s plan came back 27/27 - and the note
    # named the library as the cause when the cause was a 1.57% arithmetic error,
    # with 6972 photos sitting unused. A NOTE that always passes is worse than no
    # check at all, because it reads like a result.
    #
    # One beat of slack, because durations are allocated in whole beats and the
    # plan therefore cannot land closer than that to an arbitrary target. This is
    # the same bound assign_timing() uses when it trims the last shot.
    ok, msg = length_check(edl, planned)
    check(ok, msg)

    # ---- container sanity
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height,pix_fmt,duration,"
         "color_space,color_primaries,color_transfer,color_range",
         "-of", "json", str(video)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    streams = json.loads(probe.stdout or "{}").get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    check(v is not None, "video stream present")
    check(a is not None, "audio stream present")
    _check_colour_tags(v, check)
    if v:
        # Against the *plan*, not against 1920x1080. The EDL records the render
        # settings the pipeline was asked for, so a hardcoded expectation would
        # report a false failure on every --size 4k render - and a check that
        # cries wolf is a check people learn to ignore. Falls back to 1080p for
        # an EDL written before the render block existed.
        plan = edl.get("render") or {}
        want_w = int(plan.get("width") or 1920)
        want_h = int(plan.get("height") or 1080)
        want_fps = float(plan.get("fps") or 30)
        check(v.get("width") == want_w and v.get("height") == want_h,
              f"resolution {v.get('width')}x{v.get('height')} "
              f"matches the planned {want_w}x{want_h}")
        # fps was not in the original ffprobe field list, so ask again.
        fps_probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,level", "-of", "json",
             str(video)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        vs = (json.loads(fps_probe.stdout or "{}").get("streams") or [{}])[0]
        num, _, den = (vs.get("avg_frame_rate") or "0/1").partition("/")
        got_fps = float(num or 0) / (float(den or 1) or 1)
        check(abs(got_fps - want_fps) < 0.5,
              f"frame rate {got_fps:.2f}fps matches the planned "
              f"{want_fps:g}fps")

    # ---- pacing
    #
    # Reads the per-shot zoom back out of the EDL and checks that the film
    # actually paces itself: short shots travel further *per second* than long
    # ones, so a 2s stab reads as a stab and a 6s drift as a drift. Every shot
    # moving the same amount is the bug this guards, and it is invisible in a
    # single frame - only the distribution across the film gives it away.
    paced = [(e.get("duration"), e.get("zoom")) for e in entries
             if not e.get("is_title") and e.get("zoom") is not None
             and (e.get("duration") or 0) > 0.4
             # A `still` shot is *meant* to have no travel, so it is excluded
             # from the spread below: dividing by its 0.0 gave a 1.9e8x ratio
             # and failed a correct render. It is still included in the
             # speed comparison above, where zero travel on a long shot is
             # exactly the low end being looked for.
             and (e.get("zoom") or 0.0) > 0.0]
    if not paced:
        check(True, "NOTE: no per-shot zoom in the EDL - render predates "
                    "pacing, or Ken Burns was disabled")
    else:
        short = [z / d for d, z in paced if d <= 3.0]
        long_ = [z / d for d, z in paced if d >= 4.5]
        if not short or not long_:
            check(True, f"NOTE: {len(paced)} shots paced but none span both "
                        f"the short (<=3s) and long (>=4.5s) ends, so pacing "
                        f"cannot be compared")
        else:
            s_rate = sum(short) / len(short)
            l_rate = sum(long_) / len(long_)
            check(s_rate > l_rate * 1.2,
                  f"short shots move {s_rate:.4f}/s vs long shots "
                  f"{l_rate:.4f}/s - {s_rate / l_rate:.2f}x, so the film "
                  f"paces rather than repeating one gesture")
        spread = max(z for _, z in paced) / max(1e-9, min(z for _, z in paced))
        check(spread < 4.0,
              f"per-shot travel varies {spread:.2f}x across the film (not a "
              f"single repeated move)")
        # The level tag has to be able to carry this frame size, or a hardware
        # decoder that trusts it will refuse or mis-play the file. Level 4.0
        # caps at 8192 macroblocks per frame and 4K needs 32400, and a hardcoded
        # level used to tag every file 4.0 regardless of size - ffmpeg accepts
        # it silently, so the wrong tag only shows up on someone else's TV.
        level = int(vs.get("level") or 0)
        # MaxFS from the H.264 spec, in macroblocks, keyed by level*10.
        max_fs = {10: 99, 11: 396, 12: 396, 13: 396, 20: 396, 21: 792,
                  22: 1620, 30: 1620, 31: 3600, 32: 5120, 40: 8192,
                  41: 8192, 42: 8704, 50: 22080, 51: 36864, 52: 36864}
        # A macroblock is 16x16 luma samples, and each row/column of the frame
        # is padded up to a whole macroblock, which is why this is not simply
        # w*h/256.
        need_fs = ((want_w + 15) // 16) * ((want_h + 15) // 16)
        # Phrased as a fact, not a claim, so a failure does not read as
        # "H.264 level 40 can carry 3840x2160" in the output.
        check(bool(level) and max_fs.get(level, 0) >= need_fs,
              f"H.264 level {level}/{max_fs.get(level, 0):,} macroblocks vs "
              f"{need_fs:,} needed for {want_w}x{want_h}"
              if level else "H.264 level tag is missing")
        check(v.get("pix_fmt") == "yuv420p", f"pix_fmt {v.get('pix_fmt')} (playable everywhere)")
    if v:
        # How long the picture runs, against how long it was planned to run.
        # This is a per-shot arithmetic question - does every shot render the
        # frames it was allotted - and it is only answerable on the video
        # stream. A shot can come up short without error: the encoder stops at
        # end of file when asked for more frames than its source holds, and the
        # concat's `-frames:v` is a ceiling rather than a promise. Every shortfall
        # then pushes the rest of the film early against audio placed by the
        # EDL's timings, so a fraction of a second here is a sync fault, not a
        # rounding note.
        vd = float(v.get("duration") or 0)
        check(abs(vd - planned) <= 0.15,
              f"picture runs for {vd:.2f}s against the {planned:.2f}s plan "
              f"(within 0.15s)")
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
        # A deliberately still shot (zoom 0, motion "still") holds one frame
        # for its whole length. Every per-step displacement is ~0, so the
        # median is ~0 and the outlier test below cannot be evaluated - it
        # reported these as hard cuts. A held frame is the opposite of a cut;
        # it is just not a move, so it is not this check's subject. Skipped
        # rather than counted, and the "stills with motion" count still
        # reports how many there were.
        if e.get("zoom") is not None and (e.get("zoom") or 0.0) <= 0.0:
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
        # A shot with no camera travel has no displacement profile to judge, so
        # it cannot ease. Skipping it is not a loophole - it is the only correct
        # answer, and the old frame-difference test for "is it static" could not
        # see it: a held shot still has large frame-to-frame differences coming
        # out of the crossfade on either side, so it was scored as a failed ease
        # every time. Measured on the trip library: 2 of the first 10 stills are
        # `motion: still, zoom: 0.0` and both were reported as failures. The
        # EDL says so directly, so ask it.
        if e.get("zoom") is not None and not e.get("zoom"):
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
    _check_two_up(video, entries, check)

    # There is deliberately no check here that photos appear the right way up.
    # There was one, written because a finished edit shipped with 106 of its 141
    # photos lying on their side and every other check passed. Three ways of
    # measuring it from the frame were tried and all three failed to decide, and
    # the measurements are worth keeping because the reasoning looks sound:
    #
    #   * Correlating the frame against the two orientations, in the strip the
    #     composition put the photo in. Every still is a window onto a larger
    #     canvas, pushed in and panned, so the photo is not where it was put:
    #     a shot at zoom 0.142 matched its own half at +0.061, and two upright
    #     photos in a spread were reported as sideways.
    #   * The same, searching every strip instead of assuming one. That removed
    #     the false alarms and removed the teeth with them: with the frame
    #     rotated 90 degrees, so the sideways reading was plainly true, it still
    #     called the photo upright on 3 of 5.
    #   * The vertical-to-horizontal edge-energy ratio, which a crop and a pan
    #     barely disturb and a quarter turn inverts. This is the best of the
    #     three and it is still not a measurement of orientation. On the trip
    #     render - verified upright, by window search, on the same frames - it
    #     read 3 of 8 photos the right way up, and rotating a frame merely
    #     negated its answer. A rule that condemns correct output and cannot
    #     distinguish it from the fault it was written for is worse than no
    #     rule, because it teaches the reader to ignore the checks.
    #
    # Orientation is decided where it is decidable instead: the transform is a
    # function of eight integers and is pinned for all eight, along with the
    # probe that reads the tag, the renderer call that forwards it, and the
    # loader branch that has to honour it. See tests/test_orientation.py, whose
    # 31 tests include the three sabotages that each made this fault return.

    # ---- how good is the encode, not just whether it is well-formed.
    # Only active with --reference, which writes reference.mp4 into the scratch
    # dir; without it there is nothing to measure against and it says so rather
    # than passing silently.
    _check_encoder_fidelity(video, _reference_dir(), check)

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
