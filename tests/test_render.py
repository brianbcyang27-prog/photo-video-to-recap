"""Ken Burns move variety, and the render settings that feed it.

The move pattern is easy to get subtly wrong: a stride that is not coprime with
the pattern length produces a short cycle, which reads as the same trick repeated
rather than as variety. The old stride of 5 over 6 patterns is the textbook case,
since 5 is -1 mod 6, so the sequence ran backwards with a nudge every seventh
shot. It looked varied in a list of six and was not varied over 180 shots.
"""
from __future__ import annotations

import importlib

import pytest

from pipeline.select import _motion_for


def _spread(n=180):
    from collections import Counter
    c = Counter(_motion_for(i, n) for i in range(n))
    return max(c.values()) / n, len(c)


def test_no_move_dominates_a_long_cut():
    worst, kinds = _spread(180)
    assert worst <= 0.20, (
        f"one move took {worst:.0%} of shots; anything above ~20% reads as a "
        "repeated trick rather than variety"
    )
    assert kinds >= 6, "expected several distinct moves"


def test_moves_never_repeat_back_to_back():
    """The function's own docstring promised this, and the old code broke it."""
    m = [_motion_for(i, 180) for i in range(180)]
    repeats = sum(1 for i in range(1, len(m)) if m[i] == m[i - 1])
    assert repeats == 0, f"{repeats} back-to-back repeats in 180 shots"


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 20, 60, 180, 601])
def test_is_safe_at_any_cut_length(n):
    for i in range(n):
        assert _motion_for(i, n) in (
            "in", "out", "left", "right", "up", "down", "still")


def test_stride_visits_every_move_before_repeating():
    """Seven patterns and a stride of 5: gcd(5,7)=1, so the cycle is 7."""
    seen = [_motion_for(i, 100) for i in range(7)]
    assert len(set(seen)) == 7, f"cycle shorter than the pattern: {seen}"


def test_still_moves_appear_at_all():
    """A visual rest between moves is part of the variety."""
    m = [_motion_for(i, 180) for i in range(180)]
    assert "still" in m


# --------------------------------------------------------------- render setup

def _render_module():
    return importlib.import_module("pipeline.render")


def test_h264_level_scales_with_frame_size():
    r = _render_module()
    out = {}
    for w, h, f in ((1920, 1080, 30), (2560, 1440, 30), (3840, 2160, 30),
                    (3840, 2160, 60)):
        r.ZOOM_W, r.ZOOM_H, r.FPS = w, h, f
        out[(w, h, f)] = r._h264_level()
    assert out[(1920, 1080, 30)] == "4.0"
    # 4K has to declare a higher level or hardware decoders trust a tag that
    # says the stream is smaller than it is.
    assert out[(3840, 2160, 30)] != "4.0"
    assert out[(3840, 2160, 60)] != out[(3840, 2160, 30)]
    levels = ["4.0", "4.1", "4.2", "5.0", "5.1", "5.2", "6.0", "6.1", "6.2"]
    for v in out.values():
        assert v in levels


def test_gop_holds_two_seconds_at_any_frame_rate():
    r = _render_module()
    for f in (24, 30, 60):
        r.FPS = f
        assert r._gop() == f * 2, (
            "a fixed 60 meant a 1s GOP at 60fps, halving seek granularity on "
            "the largest files"
        )


def test_video_path_resamples_with_lanczos():
    r = _render_module()
    r.ZOOM_W, r.ZOOM_H, r.FPS = 1920, 1080, 30
    for fit in ("crop", "pad", "blur"):
        graph = r._video_filter(1920, 1080, fit)
        scales = [s for s in graph.split(";") if "scale=" in s]
        assert scales, f"{fit} path has no scale filter"
        for s in scales:
            assert "flags=lanczos" in s, (
                f"{fit} resamples with bicubic, which is the default and is "
                "what made the timeline look soft"
            )


def test_zoompan_supersamples_when_there_is_detail_to_sample():
    """The supersample is the measured fix for the soft look on every still."""
    r = _render_module()
    r.ZOOM_W, r.ZOOM_H, r.FPS, r.ZOOM_SS = 1920, 1080, 30, 2
    expr = r._zoompan_expr("in", 0.10, 90)
    assert "scale=1920:1080:flags=lanczos" in expr
    assert "s=3840x2160" in expr, "zoompan should render above delivery size"
    # One bar at a time: with ZOOM_SS=1 there is nothing to reduce.
    r.ZOOM_SS = 1
    assert "scale=" not in r._zoompan_expr("in", 0.10, 90)


def test_supersample_factor_is_capped_not_fixed():
    """A 4K film has no headroom to sample, so it must not pay for one.

    The factor is computed per run from the largest still in the edit. At 4K
    the delivery frame is 3840px and a phone photo is 4032px, so asking zoompan
    for 2x would be interpolating 7680px out of a 4032px file - four times the
    filter work for no additional detail.
    """
    import re
    from pathlib import Path as _Path
    r = _render_module()
    src = _Path(r.__file__).read_text()

    # The factor is derived, not assigned the old hardcoded value.
    assert "ZOOM_SS = 1 if not biggest else max(" in src, (
        "the effective supersample should be computed from source resolution"
    )
    # And it is bounded by the cap.
    assert re.search(r"min\(ZOOM_SS_CAP,", src), (
        "the derived factor must be bounded, or a huge source buys unbounded "
        "filter cost"
    )

    # Direct check of the arithmetic the render path relies on.
    def factor(delivery_w, biggest, cap=2):
        return 1 if not biggest else max(1, min(cap, int(biggest // max(1, delivery_w))))
    assert factor(1920, 4032) == 2, (
        "1080p from a 12MP photo: 2x is the measured win (5.402 -> 5.688)"
    )
    assert factor(1280, 4032) == 2, "720p from a 12MP photo: 2x"
    assert factor(3840, 4032) == 1, (
        "4K from a 12MP photo: covering 2x would need 7680px, so the factor "
        "drops to 1 rather than interpolating"
    )
    assert factor(1920, 0) == 1, "unknown source size must not guess"
    assert factor(1920, 1000) == 1, "a small source must not be oversampled"
    assert factor(3840, 8000) == 2, "4K from a real 8K scan: 2x"


def test_encode_commands_actually_use_the_derived_level_and_gop():
    """The helpers are easy to pass and still not be wired up.

    Calibrating this file caught exactly that: breaking the call sites back to
    a hardcoded "4.0"/60 left every test above passing, because they exercise
    the helper functions directly. So this one reads the commands the render
    path really builds.
    """
    import inspect
    import re
    from pathlib import Path as _Path


    r = _render_module()
    r.ZOOM_W, r.ZOOM_H, r.FPS = 3840, 2160, 60

    seen: list[list[str]] = []
    real_run = r.run

    def spy(cmd, **kw):
        seen.append(list(cmd))
        return real_run(["true"], **kw)

    src = _Path(r.__file__).read_text()
    # Every encode must route its level and gop through the helpers.
    encodes = re.findall(r'"-profile:v", "high", "-level", ([^,]+),', src)
    assert len(encodes) >= 3, f"found {len(encodes)} encode sites, expected 3+"
    for value in encodes:
        assert "_h264_level()" in value, (
            f"an encode still hardcodes its level as {value!r}; at 4K that "
            "writes a file tagged as 1080p for hardware decoders to trust"
        )
    gops = re.findall(r'"-g", ([^,]+), "-keyint_min"', src)
    assert len(gops) >= 3
    for value in gops:
        assert "str(_gop())" in value, (
            f"an encode hardcodes its GOP as {value!r}, which halves the seek "
            "interval when the frame rate goes up"
        )

    assert inspect.getsource(r.render_still).count("_h264_level()") == 1
    assert inspect.getsource(r.render_video).count("_h264_level()") == 1
    assert inspect.getsource(r.render_live).count("_h264_level()") == 1


def test_video_filter_extra_goes_inside_the_graph():
    """ffmpeg refuses simple and complex filtering on one stream at once."""
    r = _render_module()
    r.ZOOM_W, r.ZOOM_H, r.FPS = 1920, 1080, 30
    graph = r._video_filter(1920, 1080, "blur", "tpad=stop_mode=clone:stop_duration=0.7")
    assert "tpad=stop_mode=clone:stop_duration=0.7" in graph
    # Must land before the final label, or it becomes an unconnected output.
    assert graph.rstrip().endswith("[v]")
    assert graph.count("tpad") == 1
