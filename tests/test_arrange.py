"""Arrangement: does a long piece actually change?

The bed used to loop one four-bar progression to fill the target, which at the
default tempo is 240 bars of the same ten seconds in a ten-minute recap. These
tests are about the shape of the plan rather than the audio, because the audio
is the slow thing to generate and the shape is where the bug would live.

The two properties that matter are that a plan covers its target exactly - a
short plan leaves silence at the end of the film - and that no section runs so
long that the piece stops changing, since stretching the sections instead of
repeating them made a ten-minute plan monotonous and a thirty-minute one worse.
"""
from __future__ import annotations

import pytest

from pipeline.arrange import plan, section_at

BPM = 96
BAR = 60 / BPM * 4


def _bars_for(seconds: float) -> int:
    return int(seconds / BAR) + 1


@pytest.mark.parametrize("target", [15, 30, 45, 90, 180, 300, 600, 1200, 1800, 3000])
def test_plan_covers_the_target_exactly(target):
    """A plan that falls short leaves the end of the film silent."""
    n = _bars_for(target)
    assert sum(s.bars for s in plan(n)) == n


@pytest.mark.parametrize("target", [90, 180, 600, 1800, 5400])
def test_no_section_outlasts_its_cap(target):
    n = _bars_for(target)
    longest = max(s.bars for s in plan(n))
    assert longest * BAR <= 95, (
        f"{target}s gave a {longest * BAR:.0f}s section; past ~95s the listener "
        "stops hearing a change"
    )


def test_middle_sections_are_evenly_sized():
    """Otherwise one section absorbs the rounding and reads as an outlier."""
    n = _bars_for(1200)
    mid = [s.bars for s in plan(n)[1:-1]]
    assert max(mid) - min(mid) <= 1, f"uneven middle sections: {mid}"


def test_long_pieces_do_not_get_monotonically_wider_sections():
    """The bug: sections stretched to fill, so longer meant less varied."""
    def longest_secs(target):
        n = _bars_for(target)
        return max(s.bars for s in plan(n)) * BAR
    # Ten minutes and thirty minutes should be about the same, because the
    # thirty-minute one repeats the arrangement rather than diluting it.
    assert longest_secs(1800) <= longest_secs(600) * 1.15


def test_short_pieces_get_few_sections():
    """A 15-second clip with nine sections changes key every other second."""
    assert len(plan(_bars_for(15))) <= 3


def test_the_arrangement_actually_changes_over_a_long_film():
    n = _bars_for(600)
    p = plan(n)
    names = [section_at(p, b).name for b in range(n)]
    changes = sum(1 for i in range(1, len(names)) if names[i] != names[i - 1])
    assert changes >= 5, f"only {changes} section changes in 10 minutes"


def test_sections_differ_in_what_they_play():
    """A plan whose sections are all the same is not an arrangement."""
    secs = plan(_bars_for(600))
    assert len({s.drums for s in secs}) > 1
    assert len({s.arp for s in secs}) > 1
    assert min(s.drums for s in secs) < 0.6, (
        "a section with no drums is what gives a long piece somewhere to rest"
    )


def test_bookends_are_fixed_regardless_of_length():
    for target in (30, 600, 3000):
        p = plan(_bars_for(target))
        assert p[0].name == "open"
        assert p[-1].name == "close"
        assert p[0].bars <= 2, "the opening should not scale with runtime"
        assert p[-1].bars <= 6, "nor should the outro"


def test_degenerate_input_is_safe():
    assert plan(0) == []
    assert section_at([], 5).name == "groove"


def test_cap_is_honoured_when_passed_explicitly():
    n = _bars_for(600)
    assert max(s.bars for s in plan(n, max_bars=8)) <= 8
