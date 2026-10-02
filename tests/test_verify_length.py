"""The verifier has to be able to fail.

`verify_output.py` checked the rendered picture against the plan, and the plan
against the target - but the second comparison was a branch that returned True
whenever the two disagreed by more than 1.5s, with a message saying the library
could not fill the target. So the one check that could have caught a 177s request
coming back as 174.2s was incapable of failing, and it named the library as the
cause when 6972 photos were sitting unused.

A verifier that cannot fail is worse than no verifier, because a green run reads
as a result. These tests pin that the length check discriminates.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "verify_output", _ROOT / "tools" / "verify_output.py")
vo = importlib.util.module_from_spec(_spec)
sys.modules["verify_output"] = vo
_spec.loader.exec_module(vo)


def edl(target: float, bpm: float) -> dict:
    return {"target": target, "duration": target, "music": {"bpm": bpm}}


# ---- the bug this exists for


def test_a_plan_short_of_the_target_fails():
    """The real render: target 177s, plan 174.22s at 126.05 BPM.

    2.78s out against a beat of 0.476s. The old code passed this unconditionally
    and printed that the library could not fill the target.
    """
    ok, msg = vo.length_check(edl(177.0, 126.05), 174.219667)
    assert not ok, msg
    assert "2.78s out" in msg


def test_a_plan_long_of_the_target_also_fails():
    """The same branch caught overshoot too, and had to: a film that runs long
    is just as wrong as one that runs short, and a check written only for the
    shortfall would pass it.
    """
    ok, msg = vo.length_check(edl(177.0, 126.05), 180.4)
    assert not ok, msg


def test_the_message_does_not_blame_the_library():
    """It used to say "the library could not fill it" as though that were a
    finding. It is not knowable from the EDL, and on the real render it was
    wrong: the cause was a 1.57% arithmetic error, not a shortage of footage.
    """
    _, msg = vo.length_check(edl(177.0, 126.05), 150.0)
    assert "library" not in msg.lower()


# ---- a plan that is right must still pass, or the check is useless


def test_a_plan_on_the_target_passes():
    ok, msg = vo.length_check(edl(177.0, 126.05), 177.0)
    assert ok, msg


def test_a_plan_within_one_beat_passes():
    """Whole-beat allocation cannot land closer than a beat on a target that is
    not a whole number of beats, so anything inside that is correct, not a
    near-miss worth failing a render over.
    """
    beat = 60.0 / 126.05
    ok, _ = vo.length_check(edl(177.0, 126.05), 177.0 - beat * 0.99)
    assert ok
    ok, _ = vo.length_check(edl(177.0, 126.05), 177.0 + beat * 0.99)
    assert ok


def test_just_past_one_beat_fails():
    """The other side of the boundary. A tolerance with no sharp edge is a
    tolerance that will be widened the next time something annoys someone.
    """
    beat = 60.0 / 126.05
    ok, _ = vo.length_check(edl(177.0, 126.05), 177.0 - beat * 1.01)
    assert not ok


# ---- the bound has to track the tempo, not sit at a fixed number


def test_the_tolerance_tracks_the_tempo():
    """At 60 BPM a beat is a second, so a plan a second out is within tolerance;
    at 200 BPM the same 0.9s gap is nearly two beats and is not. A fixed 1.5s
    slack would pass both, and would fail the 200 BPM one at a gap where the
    arithmetic is plainly wrong.
    """
    assert vo.length_check(edl(177.0, 60.0), 176.1)[0]
    assert not vo.length_check(edl(177.0, 200.0), 176.1)[0]


def test_no_tempo_falls_back_to_a_half_second():
    """A render with no music has no beat grid. The fallback has to be tight
    enough to catch a real shortfall, since that is the case where there is no
    beat-lock excuse for being out.
    """
    assert vo.length_check({"target": 177.0}, 177.0)[0]
    assert not vo.length_check({"target": 177.0}, 174.2)[0]


def test_a_missing_target_is_not_a_failure():
    """Nothing was requested, so there is nothing to have missed.
    """
    ok, msg = vo.length_check({}, 174.2)
    assert ok
    assert "no target" in msg


@pytest.mark.parametrize("target", [30.0, 60.0, 177.0, 240.0, 600.0])
def test_plans_landing_on_target_pass_across_lengths(target):
    """The property the check exists to protect, stated directly: when the
    library can fill the request, the plan is the request.
    """
    for bpm in (90.0, 105.0, 126.05, 140.0):
        assert vo.length_check(edl(target, bpm), target)[0], (target, bpm)
