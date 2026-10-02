"""A Live Photo is much shorter than the slot it has to fill, and the gap used
to be a frozen frame.

The bug this file exists for: `render_live` played the clip and then held its
last frame with `tpad=stop_mode=clone` for whatever was left of the beat. On the
real trip library that gap is not small. Live Photo clips there run 0.70s to
2.71s against a 2.80s slot, so most Live Photo shots were mostly a still image
with no movement at all - entry 18 held 1.92s of its 2.80s, entry 40 held
2.10s. Measured on entry 18's actual clip: 0 of 54 tail frames moved. Three
quarters of the picture standing still, in a film whose whole complaint was
that the motion felt cheap.

The fix is that the tail is not a hold but a slow Ken Burns push on the frame
the clip ended on, so the shot keeps drifting to its cut.

The two halves are tested separately on purpose. `_live_split` is where the
arithmetic lives and is a plain comparison between two numbers, so it is
checked without an encoder. The end-to-end test is what would have caught the
original bug and it needs ffmpeg, because the failure mode is a *picture* that
stops moving, which no assertion about filter strings can see.
"""
from __future__ import annotations

import importlib
import inspect
import subprocess

import numpy as np
import pytest

from pipeline.select import Entry, Item
from pipeline.util import MediaInfo

# `from pipeline import render` binds the function the package re-exports.
render = importlib.import_module("pipeline.render")

# A real shot length from the trip library, and a real clip length from it.
SLOT = 2.80
SHORT_CLIP = 0.88
LONG_CLIP = 2.71


# --------------------------------------------------------------- the arithmetic


def test_a_clip_long_enough_for_its_slot_needs_no_push():
    play, tail = render._live_split(SLOT, LONG_CLIP)
    assert play == pytest.approx(LONG_CLIP)
    # 2.71s of clip in a 2.80s slot is a 0.09s shortfall - three frames. The
    # split still reports it honestly, but render_live has to decline to spend
    # two extra encodes and a concat on a gap that short.
    assert tail == pytest.approx(SLOT - LONG_CLIP)
    assert tail < render.TAIL_MIN, (
        f"a {tail:.2f}s gap is {tail * 30:.0f} frames - not a visible freeze, "
        f"but TAIL_MIN is {render.TAIL_MIN}s so it would trigger the two-pass path"
    )


def test_a_gap_worth_fixing_is_actually_worth_fixing():
    """The other direction: the real cases must clear the bar, or nothing changes."""
    for clip in (0.70, 0.88, 1.75, 2.05):
        _, tail = render._live_split(SLOT, clip)
        assert tail > render.TAIL_MIN, (
            f"a {clip}s clip in a {SLOT}s slot leaves {tail:.2f}s frozen and "
            f"would not be rescued"
        )


def test_a_short_clip_leaves_a_tail_that_is_exactly_the_shortfall():
    play, tail = render._live_split(SLOT, SHORT_CLIP)
    assert play == pytest.approx(SHORT_CLIP)
    assert tail == pytest.approx(SLOT - SHORT_CLIP)
    assert play + tail == pytest.approx(SLOT), "the split must account for all of it"


def test_the_clip_is_never_stretched_to_fill_the_slot():
    """The push fills the gap. The footage keeps its own pace.

    Stretching 0.88s of a person's hair moving across three seconds turns a
    gesture into a slow-motion artifact, which is worse than the freeze it
    replaced. So the played length is capped at the clip's real length.
    """
    for clip in (0.4, 0.88, 1.5, 2.1, 2.71):
        play, _ = render._live_split(SLOT, clip)
        assert play <= clip + 1e-6, f"a {clip}s clip was played as {play:.2f}s"


def test_an_unprobeable_clip_plays_the_whole_slot_rather_than_guessing():
    play, tail = render._live_split(SLOT, 0.0)
    assert (play, tail) == (SLOT, 0.0)


def test_a_clip_longer_than_its_slot_is_trimmed_not_extended():
    play, tail = render._live_split(1.0, 3.0)
    assert play == pytest.approx(1.0)
    assert tail == 0.0


def test_a_sub_frame_shortfall_is_not_worth_two_encodes():
    """Just under the bar the single-pass path is kept, just over it is not."""
    _, tiny = render._live_split(SLOT, SLOT - render.TAIL_MIN / 2)
    _, small = render._live_split(SLOT, SLOT - render.TAIL_MIN * 2)
    assert tiny < render.TAIL_MIN
    assert small > render.TAIL_MIN


# ------------------------------------------------------------- the end-to-end


def _moving_clip(path, seconds=1.0, size="640x360"):
    """A short clip that genuinely moves, so a frozen tail is measurable.

    `testsrc` scrolls its own pattern, which is what makes the difference
    between "the tail is drifting" and "the tail is a still" visible as a
    number rather than a matter of opinion.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=size={size}:rate=30:duration={seconds}",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def _entry(mov, duration):
    info = MediaInfo(path=mov.with_suffix(".HEIC"), kind="photo",
                     width=4032, height=3024, live_motion=mov)
    return Entry(item=Item(info=info, kind="photo"), duration=duration)


def _tail_is_frozen(path, clip_seconds, size=(96, 54)):
    """Per-frame movement over the part of the shot after the clip ends."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "gray", "-s", f"{size[0]}x{size[1]}", "-"],
        check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8)
    n = len(frames) // (size[0] * size[1])
    frames = frames[:n * size[0] * size[1]].reshape(n, size[1], size[0])
    steps = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2))
    # Start two frames in: the join itself is a cut, and judging a push by the
    # size of the cut it follows would measure the wrong thing.
    tail = steps[int(clip_seconds * 30) + 2:]
    return float(tail.max()) if len(tail) else 0.0


def test_a_short_live_photo_keeps_moving_after_its_clip_ends(tmp_path, monkeypatch):
    """The regression. Fails on the old `tpad=stop_mode=clone` render.

    A 1s clip in a 3s slot is the same shape as entry 18: the clip runs out
    with two thirds of the shot left to fill. Before the fix that remainder was
    a cloned frame and this measurement was 0.0.
    """
    monkeypatch.setattr(render, "FPS", 30)
    monkeypatch.setattr(render, "ZOOM_W", 640)
    monkeypatch.setattr(render, "ZOOM_H", 360)
    monkeypatch.setattr(render, "ZOOM_SS", 1)

    mov = _moving_clip(tmp_path / "IMG_0001.MOV", seconds=1.0)
    out = tmp_path / "seg.mp4"
    render.render_live(_entry(mov, 3.0), render.Pipeline(), out,
                       sw=640, sh=360)

    assert out.exists(), "render_live wrote nothing"
    assert _tail_is_frozen(out, 1.0) > 0.3, (
        "the tail of a short Live Photo is a frozen frame again - 0 of the "
        "frames after the clip ends show any movement"
    )


def test_the_shot_still_comes_out_the_right_length(tmp_path, monkeypatch):
    """Three seconds in, three seconds out - the split must not lose frames."""
    monkeypatch.setattr(render, "FPS", 30)
    monkeypatch.setattr(render, "ZOOM_W", 640)
    monkeypatch.setattr(render, "ZOOM_H", 360)
    monkeypatch.setattr(render, "ZOOM_SS", 1)

    mov = _moving_clip(tmp_path / "IMG_0002.MOV", seconds=1.0)
    out = tmp_path / "seg.mp4"
    render.render_live(_entry(mov, 3.0), render.Pipeline(), out,
                       sw=640, sh=360)

    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)],
        check=True, capture_output=True, text=True).stdout.strip())
    assert abs(dur - 3.0) < 0.1, f"a 3.00s shot rendered as {dur:.2f}s"


def test_a_live_photo_long_enough_for_its_slot_renders_in_one_pass(tmp_path,
                                                                   monkeypatch):
    """No tail means no second encode. Guards against paying for it every time."""
    monkeypatch.setattr(render, "FPS", 30)
    monkeypatch.setattr(render, "ZOOM_W", 640)
    monkeypatch.setattr(render, "ZOOM_H", 360)
    monkeypatch.setattr(render, "ZOOM_SS", 1)

    mov = _moving_clip(tmp_path / "IMG_0003.MOV", seconds=2.0)
    calls: list[list[str]] = []
    real_run = render.run

    def counting(cmd, **kw):
        calls.append(cmd)
        return real_run(cmd, **kw)

    monkeypatch.setattr(render, "run", counting)
    render.render_live(_entry(mov, 2.0), render.Pipeline(),
                       tmp_path / "seg.mp4", sw=640, sh=360)
    assert len(calls) == 1, (
        f"a 2s clip in a 2s slot needed {len(calls)} ffmpeg calls; there is no "
        "tail to fill, so it should be a single render"
    )


def test_the_temporary_parts_are_cleaned_up(tmp_path, monkeypatch):
    """Three encodes write three files next to the segment. Two are junk."""
    monkeypatch.setattr(render, "FPS", 30)
    monkeypatch.setattr(render, "ZOOM_W", 640)
    monkeypatch.setattr(render, "ZOOM_H", 360)
    monkeypatch.setattr(render, "ZOOM_SS", 1)

    mov = _moving_clip(tmp_path / "IMG_0004.MOV", seconds=1.0)
    out = tmp_path / "seg.mp4"
    render.render_live(_entry(mov, 3.0), render.Pipeline(), out,
                       sw=640, sh=360)

    leftovers = sorted(p.name for p in tmp_path.iterdir()
                       if p.name.startswith("seg_") and p != out)
    assert leftovers == [], f"render_live left {leftovers} behind"


def test_the_push_continues_from_the_clips_last_frame_not_the_still(tmp_path,
                                                                    monkeypatch):
    """Taking the frame from the clip is the difference between a settle and a jump.

    The still and the clip are the same instant, so this is hard to see by
    eye - but the push has to start where the motion stopped, and rebuilding it
    from the still would restart the shot from a slightly different frame.
    Asserted structurally: the tail frame is read out of the clip, and
    `-sseof` is the only way to ask for its last frame.
    """
    src = inspect.getsource(render._push_from_last_frame)
    assert "-sseof" in src, (
        "the push must be built from the end of the clip, not from the still"
    )
    assert "orientation=1" in src, (
        "video frames are already upright; applying the still's EXIF rotation "
        "would tilt the push"
    )
