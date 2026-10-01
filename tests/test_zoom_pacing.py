"""Per-shot camera-move pacing.

`cfg.render.zoom_amount` used to be the amount for *every* shot on the
timeline. That is not one decision, it is the same decision repeated two
hundred times, and a viewer reads the repetition as a lack of intent: a 2s stab
that moves as far as a 6s drift has the same travel at three times the speed,
and a film that does that throughout stops feeling edited.

_zoom_for fixes the pacing. The tests here pin the property that actually makes
it work - shorter shots move further *per second* than long ones - and
deliberately do not pin the exact numbers, because the numbers are taste and
will be retuned; the ordering is the thing that must survive.
"""
from __future__ import annotations

import importlib

import pytest

select = importlib.import_module("pipeline.select")

BASE = 0.10


def _z(duration, index=0, motion="in", base=BASE, energy=1.0):
    return select._zoom_for(index, duration, motion, base, energy)


# ---- the property the whole change exists for


def _mean_amount(duration, spread=200):
    """Average amount across indices, so the per-index jitter cannot help.

    Averages rather than single samples, because the tie-breaking jitter is
    worth keeping - it is what stops a run of equal-length beats from rendering
    as a run of identical moves. But a *mean* is what isolates the pacing law
    from that jitter. Taking one sample per index here was the mistake: a
    reverted-to-constant implementation still passes a single-sample comparison,
    because the jitter alone makes any two samples differ.
    """
    return sum(_z(duration, index=i) for i in range(spread)) / spread


def test_short_shots_travel_further_than_long_ones():
    """The discriminating test.

    Note what this is *not* asserting: that a short shot covers more ground per
    second. A constant amount already does that, since speed is just
    amount/duration and dividing a constant by a smaller number gives a larger
    one. A speed-based test therefore passes on the unfixed code and pins
    nothing. Travel is the thing that has to change: two shots of 6s and 2s
    with identical amounts are the same gesture, however fast one of them
    happens to be.
    """
    short = _mean_amount(2.0)
    long = _mean_amount(6.0)
    assert short > long * 1.3, (
        f"a 2s shot averages {short:.4f} travel and a 6s shot averages "
        f"{long:.4f}; the length scaling is missing"
    )


def test_short_shots_also_cover_ground_faster_per_second():
    """Kept because it is the property a viewer actually perceives, even
    though it does not discriminate. Speed = travel / duration, and it has to
    fall as shots lengthen or the pacing reads as a machine winding down."""
    short_speed = _mean_amount(2.0) / 2.0
    long_speed = _mean_amount(6.0) / 6.0
    assert short_speed > long_speed * 1.4, (
        f"a 2s shot averages {short_speed:.4f}/s but a 6s shot averages "
        f"{long_speed:.4f}/s; short shots must read as clearly faster"
    )


def test_travel_falls_monotonically_across_the_whole_duration_range():
    """Every step, not just 2s vs 6s, because that is what 300 shots means.

    Averaged per duration so the jitter cannot mask a missing length term, and
    spaced with a margin so the ordering is a real separation rather than
    floating-point noise that could flip on a different machine.
    """
    durations = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    travel = [_mean_amount(d) for d in durations]
    for d1, d2, t1, t2 in zip(durations, durations[1:], travel, travel[1:]):
        assert t1 > t2, (
            f"{d1}s averages {t1:.4f} travel and {d2}s averages {t2:.4f}; "
            f"travel must fall as shots get longer, not wobble"
        )


def test_the_spread_is_bounded_so_pacing_never_becomes_a_spectacle():
    """Square-root, not linear, and not a step function.

    A linear law would give a 2s shot 3x the travel of a 6s shot, which reads
    as the short shots being broken rather than emphasised. The bound keeps the
    spread near 1.7x so the variation is felt without being pointed at.
    """
    short = _mean_amount(2.0)
    long = _mean_amount(6.0)
    assert short > long, "a 2s shot must travel further than a 6s one"
    ratio = short / long
    assert 1.4 < ratio < 2.2, f"2s-to-6s travel spread is {ratio:.2f}x"


# ---- the still case, and the clamps


def test_a_still_frame_cannot_travel():
    """`still` is a deliberate no-move; a number here would be a zoom anyway."""
    assert _z(3.0, motion="still") == 0.0
    assert _z(3.0, motion="still", energy=1.3) == 0.0


def test_amounts_stay_inside_the_camera_range_at_every_duration():
    """Nothing escapes into a number the frame can express."""
    for d in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 60.0):
        z = _z(d)
        assert select.ZOOM_MIN <= z <= select.ZOOM_MAX, (
            f"{d}s produced {z:.4f}, outside "
            f"[{select.ZOOM_MIN}, {select.ZOOM_MAX}]"
        )


def test_extreme_energies_are_clamped_not_trusted():
    """A crescendo must not push the frame past where a move stops reading."""
    quiet = _z(3.0, energy=0.1)
    loud = _z(3.0, energy=9.0)
    assert quiet >= select.ZOOM_MIN
    assert loud <= select.ZOOM_MAX
    assert loud > quiet, "energy is being clamped into uselessness"


def test_a_zero_or_negative_duration_does_not_explode():
    """Durations come from beat arithmetic, and arithmetic has edge cases."""
    for d in (0.0, -1.0, -100.0):
        z = _z(d)
        assert 0.0 <= z <= select.ZOOM_MAX, f"{d}s produced {z}"


# ---- determinism, which the rest of the suite depends on


def test_the_same_inputs_always_give_the_same_amount():
    """No RNG anywhere in the cut.

    A jitter term that used a random source would make every render of the same
    library a different film, which would break the reproducibility the tests,
    the EDL and the verification all assume.
    """
    first = [_z(3.0, index=i) for i in range(50)]
    second = [_z(3.0, index=i) for i in range(50)]
    assert first == second


def test_neighbouring_shots_do_not_all_get_the_same_amount():
    """The jitter exists to break ties between equal-length neighbours.

    Without it, a run of same-length beats - which is most of a beat-aligned
    cut - would produce a run of identical moves, and the variation would only
    appear wherever the durations happened to differ.
    """
    amounts = [_z(3.0, index=i) for i in range(40)]
    assert len(set(amounts)) > 5, (
        f"only {len(set(amounts))} distinct amounts across 40 shots; the "
        f"jitter is not reaching the output"
    )


def test_jitter_is_a_narrow_spread_not_a_scatter():
    """A wide jitter would swamp the pacing signal it is meant to break up."""
    amounts = [_z(3.0, index=i) for i in range(200)]
    ratio = max(amounts) / min(amounts)
    assert ratio < 1.35, f"jitter spread is {ratio:.2f}x, too wide to be subtle"


def test_different_indices_give_different_amounts():
    """The hash has to depend on the index, not just on its own constants."""
    assert _z(3.0, index=0) != _z(3.0, index=1)


# ---- the music coupling, which is the part that was silently dead


class _FakeMusic:
    def __init__(self, bpm=120.0):
        self.bpm = bpm
        self.path = None
        self.label = ""

    @property
    def seconds_per_beat(self):
        return 60.0 / self.bpm

    @property
    def bar_seconds(self):
        return self.seconds_per_beat * 4.0


def test_the_energy_curve_is_normalised_to_a_mean_of_one():
    """A loud film must not come out systematically faster than a quiet one.

    Only the relative shape of the arrangement is meaningful here; its absolute
    level is already set by the generator. So the curve is divided by its own
    mean, which puts the average at 1.0 whatever the arrangement is.
    """
    curve = select._energy_curve(_FakeMusic(), total=600.0)
    assert len(curve) > 1
    assert sum(curve) / len(curve) == pytest.approx(1.0, abs=0.01)


def test_the_energy_curve_actually_varies():
    """A flat curve would make the whole mechanism a no-op with extra steps."""
    curve = select._energy_curve(_FakeMusic(), total=600.0)
    assert max(curve) - min(curve) > 0.15, (
        f"energy range is only {max(curve) - min(curve):.3f}; the curve is "
        f"flat, so pacing will not follow the music"
    )


def test_no_music_yields_a_flat_curve_rather_than_an_error():
    """Music generation can be disabled; the cut must still build."""
    assert select._energy_curve(None, total=600.0) == [1.0]
    assert select._energy_curve(_FakeMusic(bpm=0.0), total=600.0) == [1.0]


def test_a_zero_length_film_yields_a_flat_curve():
    assert select._energy_curve(_FakeMusic(), total=0.0) == [1.0]


def test_energy_lookup_clamps_outside_the_film():
    curve = [0.5, 1.0, 1.5]
    bar = 1.0
    assert select._energy_at(curve, -5.0, bar) == 0.5
    assert select._energy_at(curve, 1.0, bar) == 1.0
    assert select._energy_at(curve, 99.0, bar) == 1.5
    assert select._energy_at([], 3.0, bar) == 1.0


def test_energy_at_zero_bar_never_divides_by_zero():
    """A degenerate bpm must not turn into ZeroDivisionError mid-cut."""
    assert select._energy_at([1.0, 2.0], 5.0, 0.0) in (1.0, 2.0)


def test_energy_reaches_the_cut_rather_than_sitting_in_a_signature():
    """The regression, and the reason it is pinned at all.

    `energy` was a documented parameter of _zoom_for that no caller ever
    passed, so the music coupling existed only in a docstring while the tests
    exercised it directly. Testing the helper therefore passed the whole time
    the feature was off. This greps the assignment site, which is the only
    place that can tell the difference.
    """
    import inspect

    src = inspect.getsource(select.build_cutlist)
    assert "_energy_curve" in src, (
        "build_cutlist never builds the energy curve, so every shot is paced "
        "against 1.0 and the music does not affect the picture"
    )
    assert "_energy_at" in src, "build_cutlist never samples the curve"
    assert "_energy_at(curve" in src, "the sampled value is not passed anywhere"


def test_energy_actually_changes_a_shot_amount():
    """End to end through the helper, at fixed duration and index."""
    quiet = _z(3.0, energy=0.8)
    peak = _z(3.0, energy=1.3)
    assert peak > quiet * 1.3, (
        f"a peak section only scales the move to {peak / quiet:.2f}x"
    )


# ---- it has to actually reach the encoder


def test_the_configured_amount_is_still_the_default():
    """base is the anchor: a caller passing the configured value gets the
    configured scale, so the tuning knob keeps meaning what it says."""
    amounts = [_z(4.0, index=i, base=0.20) for i in range(20)]
    mids = [_z(4.0, index=i, base=0.10) for i in range(20)]
    assert all(a > m for a, m in zip(amounts, mids)), (
        "doubling the configured zoom did not double the output"
    )


def test_render_still_accepts_and_uses_a_per_shot_amount():
    """The call site, not just the helper.

    Tested at render_still rather than only in _zoom_for, because a helper that
    computes the right number into a variable nothing reads is a helper that
    fixed nothing - which is exactly the failure the call-site greps in
    test_render.py exist to catch.
    """
    render = importlib.import_module("pipeline.render")
    import inspect

    sig = inspect.signature(render.render_still)
    assert "zoom" in sig.parameters, (
        "render_still has no per-shot zoom parameter, so _zoom_for's output "
        "cannot reach the filter graph"
    )
    assert sig.parameters["zoom"].default is None, (
        "zoom must default to None so existing callers keep the global setting"
    )

    # And the value is used, not accepted and dropped.
    src = inspect.getsource(render.render_still)
    assert "amount" in src and "cfg.render.zoom_amount if zoom is None" in src


def test_the_entry_carries_a_zoom_only_when_it_was_computed():
    """None means 'unset' and must not be confused with a real amount."""
    e = select.Entry()
    assert e.zoom is None
    e.zoom = 0.12
    assert e.zoom == 0.12


def test_zero_is_a_real_zoom_and_not_the_same_as_unset():
    """A still shot legitimately wants exactly no travel.

    Collapsing that to None would silently give it the global 0.10 default and
    animate a frame that is supposed to hold still.
    """
    assert _z(3.0, motion="still") == 0.0
    assert select.Entry(zoom=0.0).zoom is not None
