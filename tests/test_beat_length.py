"""The timeline has to reach the length that was asked for.

`assign_timing` decides how many beats the film contains, and then places every
cut at `round(cumulative_beats * seconds_per_beat * fps)`. Those two steps have
to agree about how long a beat is, and they did not.

The beat *count* was derived from a rounded per-beat frame count -
`round(seconds_per_beat * fps)` - while the cuts are placed from the unrounded
product. At 126.05 BPM and 60fps a beat is 28.5601 frames, which rounds to 29.
Dividing the target by 29 instead of by 28.5601 under-counts the beats by 1.57%,
so a 177s request came back 174.2s: 2.8s short, and the pipeline blamed the
library for it ("not enough distinct shots in the library"). The library had
6972 photos in it. Nothing was wrong with the library; the arithmetic was.

This matters beyond tidiness. The verifier compares picture length against the
*plan*, not the target, so a plan that is itself 1.57% short passes every
length check it is given. Only a test that measures against the requested
target can see this class of bug at all.

Whole-beat allocation leaves up to half a beat of rounding over, so the film
cannot land on the target by construction alone. The last shot absorbs it -
that cut is the end of the film rather than an edit, and lengthening a shot by
a fraction of a beat moves nothing on screen.
"""
from __future__ import annotations

import importlib

import pytest

from pipeline import config as config_mod
from pipeline.analysis import Item
from pipeline.music import MusicResult
from pipeline.util import MediaInfo

select = importlib.import_module("pipeline.select")


def _cfg(target: float, *, fps: int = 60) -> config_mod.Pipeline:
    cfg = config_mod.Pipeline(target_seconds=target)
    cfg.render.fps = fps
    cfg.render.title_card = True
    return cfg


def _photo(name: str) -> Item:
    info = MediaInfo(path=f"/lib/{name}", kind="photo", width=4032, height=3024,
                     duration=None, fps=None, has_audio=False)
    return Item(info=info, kind="photo", start=0.0, end=3.0, score=0.5)


def _entries(n_titles: int, n_photos: int) -> list[select.Entry]:
    """Cards interleaved with shots, as build_cutlist() emits them.

    The photo count is deliberately generous. photo_min/photo_max bound each
    shot, so a fixture with too few photos cannot reach the target however
    correct the arithmetic is - and then the test is measuring the fixture.
    """
    out: list[select.Entry] = []
    for c in range(n_titles):
        for k in range(n_photos // n_titles):
            out.append(select.Entry(item=_photo(f"{c}_{k}.jpg")))
        out.append(select.Entry(is_title=True, title=f"Day {c}", subtitle="Rome"))
    return out


def _music(bpm: float) -> MusicResult:
    spb = 60.0 / bpm
    return MusicResult(path="/music/x.wav", duration=10_000.0, bpm=bpm,
                       beat_times=[i * spb for i in range(10_000)])


def _frames(entries: list[select.Entry], fps: int) -> int:
    return round(sum(e.duration for e in entries) * fps)


# ---- the discriminating test


def test_length_reaches_target_when_a_beat_is_a_fractional_frame():
    """126.05 BPM at 60fps: a beat is 28.5601 frames, not 29.

    This is the real 177s render's tempo and length, and the shape that
    produced a 2:54 film. 45 content shots against photo_max 4.6s can cover
    207s and photo_min 2.6s needs only 117s, so the allocation is free to hit
    177s and nothing is clamped: any shortfall is arithmetic, not a dry library.
    """
    entries = _entries(21, 45)
    select.assign_timing(entries, 177.0, _music(126.05), _cfg(177.0))

    assert _frames(entries, 60) == 177 * 60


def test_total_beats_is_measured_with_the_same_beat_length_as_the_cuts():
    """Pins the root cause directly rather than the symptom.

    The cuts are placed from `seconds_per_beat`, so the count that decides how
    many beats there are has to come from the same number. A test on the sum of
    durations alone would also be satisfied by any other arithmetic that happens
    to sum correctly; this cannot drift away from the placement code without
    failing.
    """
    bpm, target, fps = 126.05, 177.0, 60
    entries = _entries(21, 45)
    select.assign_timing(entries, target, _music(bpm), _cfg(target, fps=fps))

    spb = 60.0 / bpm
    assert sum(e.beats for e in entries) == round(target / spb)


# ---- the opposite direction, so the fix cannot overshoot


@pytest.mark.parametrize("bpm", [120.0, 100.0, 90.0, 60.0])
def test_exact_beat_lengths_are_left_alone(bpm):
    """At 120 BPM and 60fps a beat is exactly 30 frames, so there is no rounding
    to disagree about and the old arithmetic was already right. A fix that
    rescaled everything would show up here as an over- or undershoot, which is
    the other half of what a fix like this gets wrong.
    """
    target = 180.0
    entries = _entries(6, 45)
    select.assign_timing(entries, target, _music(bpm), _cfg(target))

    assert _frames(entries, 60) == target * 60


def test_a_beat_longer_than_a_frame_is_not_divided_by_one():
    """The other direction of the same rounding. 40 BPM at 60fps is a beat of
    90 frames, which also divides exactly, so this case agrees too - it is here
    to show the fix is a single consistent substitution rather than a fudge that
    only happens to work at 126 BPM.
    """
    target = 180.0
    entries = _entries(6, 45)
    select.assign_timing(entries, target, _music(40.0), _cfg(target))

    assert _frames(entries, 60) == target * 60


@pytest.mark.parametrize("fps", [24, 30, 50, 60])
def test_length_holds_across_frame_rates(fps):
    """A beat is a whole number of frames at very few tempo/frame-rate pairs,
    so this is the general form of the bug: everything else was quietly short.
    """
    target = 150.0
    entries = _entries(6, 40)
    select.assign_timing(entries, target, _music(126.05), _cfg(target, fps=fps))

    assert _frames(entries, fps) == target * fps


# ---- the rounding residue, and what must not absorb it


def test_the_last_shot_takes_up_the_whole_beat_rounding():
    """A whole-beat count cannot land on an arbitrary target, and at 126.05 BPM
    it missed by 4 frames. The film should still end on the target: the final
    shot's end is the end of the film, not an edit, so adjusting it disturbs no
    cut on the beat.
    """
    entries = _entries(21, 45)
    select.assign_timing(entries, 177.0, _music(126.05), _cfg(177.0))

    spb = 60.0 / 126.05
    beats = sum(e.beats for e in entries)
    residue = 177 * 60 - round(beats * spb * 60)
    assert residue != 0, "expected whole-beat rounding to leave a residue"
    assert -0.5 * spb * 60 <= residue <= 0.5 * spb * 60


def test_snapping_the_end_does_not_move_any_cut():
    """The residue lands on the last shot, so every cut that is actually an edit
    must still sit on its beat. Checks the cumulative beat positions before the
    final entry against the true beat times.
    """
    fps = 60
    bpm = 126.05
    entries = _entries(21, 45)
    select.assign_timing(entries, 177.0, _music(bpm), _cfg(177.0))
    spb = 60.0 / bpm

    cum = 0.0
    prev_frame = 0
    for e in entries[:-1]:
        cum += e.beats
        frame = int(round(cum * spb * fps))
        assert e.duration == pytest.approx((frame - prev_frame) / fps,
                                           abs=1e-9)
        prev_frame = frame


def test_a_library_that_cannot_fill_the_target_is_not_silently_stretched():
    """The snap exists to absorb rounding, not to hide a genuine shortfall.

    Six shots against a 3-minute target cannot reach it however the arithmetic
    is done, and a film that stops 40s short is worth reporting rather than
    padding the last shot to cover the gap.
    """
    entries = _entries(6, 6)
    select.assign_timing(entries, 180.0, _music(126.05), _cfg(180.0))

    assert _frames(entries, 60) < 180 * 60
