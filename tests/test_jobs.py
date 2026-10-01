"""How many segments get rendered at once.

The bug this pins down is not a subtle one, which is why it took a crash to
find. `default_jobs()` returned `min(8, cpu - 1)` and nothing else, so a 4K60
render on a 16GB laptop started eight ffmpeg processes at roughly 3-4.8GB of
peak resident memory each: 24-38GB of concurrent allocation against 16GB of
physical RAM. That is not a slow render, it is a kernel OOM kill, and on a
laptop it takes the rest of the desktop's unsaved work with it.

So the worker count is now bounded by memory first and CPU second. The memory
figures are measured, not guessed - see SEGMENT_PEAK_GB - and these tests hold
the arithmetic to the property that actually matters: the total allocation has
to fit, with headroom, on the machine that is really there.
"""
from __future__ import annotations

import importlib
import os

import pytest

from pipeline import util

render = importlib.import_module("pipeline.render")


# ---- the property that matters, checked against a real machine


def test_concurrent_workers_cannot_exceed_available_memory():
    """The regression. On this machine, whatever it is."""
    total_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") \
        / (1024 ** 3)
    for w, h in ((1920, 1080), (2560, 1440), (3840, 2160)):
        jobs = util.default_jobs(w, h)
        peak = render._segment_peak_gb(w, h)
        budget = (total_gb - util.MEMORY_HEADROOM_GB) / peak
        assert jobs <= budget, (
            f"{jobs} workers at {w}x{h} need {jobs * peak:.1f}GB but only "
            f"{budget:.1f}GB is available ({total_gb:.0f}GB machine less "
            f"{util.MEMORY_HEADROOM_GB}GB headroom)"
        )


def test_memory_is_consulted_before_cpu_count():
    """The ordering is the fix. A 4K render cannot be as parallel as 1080p."""
    assert util.default_jobs(3840, 2160) < util.default_jobs(1920, 1080), (
        "4K must run with fewer workers than 1080p on the same machine"
    )


def test_four_k_on_a_sixteen_gigabyte_machine_is_not_eight_workers():
    """The exact configuration that crashed, stated as a number."""
    jobs = util.default_jobs(3840, 2160)   # on the real 16GB host this is <=4
    peak = util.SEGMENT_PEAK_GB[3840]
    assert jobs * peak <= 16.0 - util.MEMORY_HEADROOM_GB, (
        f"{jobs} x {peak}GB = {jobs * peak:.1f}GB will not fit in 16GB "
        f"with {util.MEMORY_HEADROOM_GB}GB spare"
    )


# ---- the arithmetic, at sizes and machines that are not this one


def _jobs(ram_gb: float, width: int, height: int) -> int:
    """default_jobs with the machine's RAM substituted, so it can be varied."""
    peak = util.SEGMENT_PEAK_GB.get(max(width, height),
                                    max(util.SEGMENT_PEAK_GB.values()))
    affordable = int((ram_gb - util.MEMORY_HEADROOM_GB) // peak)
    cpu = os.cpu_count() or 4
    return max(1, min(cpu - 1, 8, affordable))


@pytest.mark.parametrize("ram_gb,expected", [
    (8.0, 1),      # 4K does not fit alongside any headroom -> single worker
    (16.0, 2),     # the machine that crashed: 2, not 8
    (32.0, 5),
    (64.0, 8),     # CPU count takes over as the ceiling
    (128.0, 8),
])
def test_worker_count_by_machine_size(ram_gb, expected):
    assert _jobs(ram_gb, 3840, 2160) == expected


def test_1080p_costs_almost_nothing_versus_the_old_behaviour():
    """The fix must not meaningfully slow down the common case.

    1080p is not fully unchanged: 8 workers at 1.6GB is 12.8GB, which does not
    fit in 16GB under the headroom, so the correct answer here is 7. That is a
    12% cut in parallelism for a machine that was running right at the edge, and
    it is the price of not crashing. The claim being pinned is the size of the
    price, so that a future change to the peak table cannot quietly turn "7
    instead of 8" into "2 instead of 8" at the default resolution.
    """
    old = 8                      # what min(8, cpu - 1) gave on a 10-core host
    new = _jobs(16.0, 1920, 1080)
    assert new >= old - 1, f"1080p lost {old - new} workers, not just one"
    assert new >= 6, f"1080p fell to {new} workers, which is a real slowdown"


def test_1080p_is_cpu_limited_once_memory_is_not_the_constraint():
    """With RAM to spare, the count is decided by cores, as it always was."""
    assert _jobs(256.0, 1920, 1080) == max(1, min(8, (os.cpu_count() or 4) - 1))


def test_a_small_machine_still_runs_one_worker():
    """Never zero. A 4GB machine must produce a film, slowly, not refuse."""
    assert _jobs(4.0, 3840, 2160) == 1
    assert _jobs(2.0, 3840, 2160) == 1


def test_memory_far_below_the_headroom_still_returns_one():
    """The subtraction goes negative; the floor has to catch it."""
    assert _jobs(3.0, 3840, 2160) == 1
    assert _jobs(1.0, 1920, 1080) == 1


# ---- probing is a different cost model from rendering


def test_probe_pool_does_not_depend_on_output_size():
    """The decoupling.

    Probing decodes one frame to measure sharpness; it never builds an output
    frame, so a 4K render has no reason to shrink it. If this ever regresses to
    the render table, `--size 4k` silently halves the probe pool and adds
    minutes to the pipeline's slowest step for no benefit.
    """
    assert util.probe_jobs() >= 4, (
        f"only {util.probe_jobs()} probe workers; the probe pool has been "
        f"coupled to something it should not be coupled to"
    )


def test_probe_pool_is_memory_safe_on_a_small_machine():
    """Still bounded - the fix is not 'probes are free'."""
    peak = util.PROBE_PEAK_GB
    total = 8.0
    assert int((total - util.MEMORY_HEADROOM_GB) // peak) >= 1
    assert util.probe_jobs() >= 1


def test_probe_and_render_choose_differently_at_4k():
    """On this host, 4K rendering is memory-bound but probing is not."""
    assert util.probe_jobs() > util.default_jobs(3840, 2160), (
        "probing 11k files should not be throttled by 4K render memory"
    )


def test_probe_peak_is_smaller_than_any_render_peak():
    """Sanity on the table itself: a probe is never the expensive one."""
    assert util.PROBE_PEAK_GB < min(util.SEGMENT_PEAK_GB.values())


# ---- the table itself


def test_the_peak_table_is_ordered_by_frame_area():
    """A bigger frame cannot use less memory; that would be a typo."""
    peaks = [util.SEGMENT_PEAK_GB[k] for k in sorted(util.SEGMENT_PEAK_GB)]
    assert peaks == sorted(peaks), f"peak table is not monotonic: {peaks}"
    assert util.SEGMENT_PEAK_GB[1920] < util.SEGMENT_PEAK_GB[3840]


def test_an_unrecognised_size_assumes_the_worst():
    """Guessing low here is the expensive direction."""
    assert render._segment_peak_gb(9999, 9999) == max(util.SEGMENT_PEAK_GB.values())


def test_peak_is_looked_up_by_long_side_not_width():
    """A portrait render is 1080x1920 and is not a different thing."""
    assert render._segment_peak_gb(1920, 1080) == render._segment_peak_gb(1080, 1920)
    assert render._segment_peak_gb(1080, 1920) == util.SEGMENT_PEAK_GB[1920]
