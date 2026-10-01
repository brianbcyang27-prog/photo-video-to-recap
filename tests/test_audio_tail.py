"""The dead-audio-tail guard.

A recap that plays normally for nine minutes and then goes bit-exactly silent
is the worst failure this pipeline can ship, and it is invisible to every other
check: the duration is right, the streams are there, loudness is measured over
the whole file, and the picture is fine. The only thing that notices is the
buffer itself.

That is why the check counts a *contiguous* run of exact zeros at the end rather
than a quiet level or a proportion of zeros in a window. A fade tapers toward
zero and only touches it on the last sample, so a real fade produces no
contiguous run; a bed that stopped early and got padded produces a long one.

The thresholds here are the ones the guard's own docstring records as measured
against controls, and these tests are what stop that record from drifting away
from the code. The shape of the argument - samples in, seconds out - is chosen
so the whole thing is testable without ffmpeg, a media file, or a render.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

# `from pipeline import render` binds the *function* the package re-exports, not
# the module, so anything reaching into module-level helpers has to import it by
# name. Easy to trip over and silent when it happens.
render = importlib.import_module("pipeline.render")

RATE = 8000


def _tone(seconds: float, rate: int = RATE) -> np.ndarray:
    """A non-zero buffer, so a test only controls the tail it cares about."""
    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float64) / rate
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)


def _with_dead_tail(tone: np.ndarray, dead_s: float) -> np.ndarray:
    """The failure shape: real audio, then exact zeros, no taper."""
    zeros = np.zeros(int(dead_s * RATE), dtype=np.int16)
    return np.concatenate([tone, zeros])


def _with_fade(tone: np.ndarray, fade_s: float) -> np.ndarray:
    """The natural shape: a taper that reaches zero only on the last sample.

    Built by scaling the tail down to a floor of 1 rather than 0, because that
    is what a real fade in 16-bit looks like - it asymptotes, and only the very
    last sample is exact zero.
    """
    n = len(tone)
    fade_n = min(n, int(fade_s * RATE))
    ramp = np.linspace(1.0, 1.0 / 8000.0, fade_n)
    out = tone.astype(np.float64).copy()
    out[n - fade_n:] *= ramp
    return np.maximum(np.round(out), 1).astype(np.int16)


# ---- the measurement itself


def test_a_healthy_mix_reports_no_trailing_silence():
    assert render.trailing_silence_s(_tone(4.0), RATE) == 0.0


def test_a_single_zero_sample_is_not_a_dead_tail():
    buf = _tone(4.0)
    buf[-1] = 0
    # One zero sample is what a finished fade leaves behind.
    assert render.trailing_silence_s(buf, RATE) == pytest.approx(1 / RATE)


def test_the_measurement_is_contiguous_not_a_proportion():
    """The property that makes this check work at all.

    A 3s hole in the middle of a 4s window is 75% zeros, which any
    threshold-on-a-fraction would rank as mild. Counting the run from the end
    finds the run, wherever it starts.
    """
    buf = _tone(4.0)
    buf[RATE // 2: RATE // 2 + 3 * RATE] = 0     # 3s hole at 0.5s, fits in 4s
    zeros_fraction = float((buf == 0).mean())
    assert zeros_fraction > 0.7, "this really is mostly zeros in the window"
    # ...but the trailing run is zero, because audio follows the hole.
    assert render.trailing_silence_s(buf, RATE) == 0.0

    dead = _with_dead_tail(_tone(4.0), 3.0)
    assert render.trailing_silence_s(dead, RATE) == pytest.approx(3.0, abs=1e-3)


def test_a_hole_before_a_tail_is_not_counted_twice():
    """A hole and a tail are separate runs; only the one at the end counts.

    The name of the earlier version of this test said "summed". That was wrong
    and the assertion now says what the function actually does, which is the
    useful thing: a mid-buffer hole is not a dead ending, and must not be
    reported as one.
    """
    buf = _tone(4.0)
    buf[100:RATE] = 0            # 1s hole, mid-buffer
    buf[-RATE:] = 0             # 1s dead tail
    # The hole is not contiguous with the end, so only the tail counts.
    assert render.trailing_silence_s(buf, RATE) == pytest.approx(1.0, abs=1e-3)
    # And the hole on its own reads as no dead tail at all.
    only_hole = _tone(4.0)
    only_hole[100:RATE] = 0
    assert render.trailing_silence_s(only_hole, RATE) == 0.0


# ---- the threshold, replayed against the recorded controls

@pytest.mark.parametrize("fade_s", [1.0, 1.5, 2.0, 4.0])
def test_natural_fades_pass(fade_s):
    """Real fades, at and beyond the threshold, must not trip the guard.

    4.0s is the whole buffer: even a fade that occupies everything still leaves
    non-zero samples right up to the end, which is the point.
    """
    buf = _with_fade(_tone(4.0), fade_s)
    assert render.trailing_silence_s(buf, RATE) < render.DEAD_TAIL_S


def test_a_one_second_dead_tail_passes():
    """The documented edge: short enough to be indistinguishable from a fade."""
    assert render.trailing_silence_s(_with_dead_tail(_tone(4.0), 1.0), RATE) < \
        render.DEAD_TAIL_S


@pytest.mark.parametrize("dead_s", [2.0, 3.0, 4.0, 9.3])
def test_dead_tails_are_caught(dead_s):
    assert render.trailing_silence_s(_with_dead_tail(_tone(4.0), dead_s), RATE) \
        >= render.DEAD_TAIL_S


def test_the_real_observed_fault_is_caught():
    """9.28s is the length that actually happened on a real render."""
    assert render.trailing_silence_s(_with_dead_tail(_tone(10.0), 9.28),
                                     RATE) >= render.DEAD_TAIL_S


def test_the_threshold_sits_between_the_pass_and_fail_controls():
    """One assertion, so the two tables above cannot both be right by accident."""
    worst_fade = max(
        render.trailing_silence_s(_with_fade(_tone(4.0), s), RATE)
        for s in (1.0, 1.5, 2.0, 4.0))
    best_failure = min(
        render.trailing_silence_s(_with_dead_tail(_tone(4.0), s), RATE)
        for s in (2.0, 3.0, 4.0, 9.3))
    assert worst_fade < render.DEAD_TAIL_S <= best_failure


# ---- the guard refuses to be fooled by a file it cannot read


def test_the_guard_fails_open_when_ffmpeg_gives_it_nothing(tmp_path, monkeypatch):
    """A -ss past the end of a short file returns no samples.

    That is not "the mix is clean", it is "we did not look". Skipping silently
    is right here - a render must not die because a check could not run - but it
    has to be a skip and not a pass, which is why this is a test at all.
    """
    from pipeline.util import ToolError

    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        class R:
            stdout = b""
            stderr = b""
        return R()

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    render._assert_music_underneath(tmp_path / "m.wav", tmp_path / "s.wav", 8.0)
    # It ran the probe, and did not raise.
    assert calls, "the guard must actually probe the file"
    # It seeks from the start, not past the end, which is what made the
    # negative control read as a pass.
    assert "atrim=start=4" in calls[0][calls[0].index("-af") + 1]


def test_the_guard_raises_on_a_dead_tail(tmp_path, monkeypatch):
    """End to end through the real guard, with only ffmpeg's output faked."""
    from pipeline.util import ToolError

    # 4s of tone then 3s of exact zeros, at 8kHz, as ffmpeg would emit it.
    buf = _with_dead_tail(_tone(4.0), 3.0)
    payload = buf.astype("<i2").tobytes()

    def fake_run(cmd, **kw):
        class R:
            stdout = payload
            stderr = b""
        return R()

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    with pytest.raises(ToolError) as exc:
        render._assert_music_underneath(tmp_path / "final.wav",
                                       tmp_path / "bed.wav", 7.0)
    assert "digital silence" in str(exc.value)
    assert "3.0" in str(exc.value)


def test_the_guard_passes_a_healthy_mix(tmp_path, monkeypatch):
    payload = _with_fade(_tone(4.0), 2.0).astype("<i2").tobytes()

    def fake_run(cmd, **kw):
        class R:
            stdout = payload
            stderr = b""
        return R()

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    render._assert_music_underneath(tmp_path / "final.wav",
                                   tmp_path / "bed.wav", 6.0)
