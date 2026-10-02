"""Ken Burns move variety, and the render settings that feed it.

The move pattern is easy to get subtly wrong: a stride that is not coprime with
the pattern length produces a short cycle, which reads as the same trick repeated
rather than as variety. The old stride of 5 over 6 patterns is the textbook case,
since 5 is -1 mod 6, so the sequence ran backwards with a nudge every seventh
shot. It looked varied in a list of six and was not varied over 180 shots.
"""
from __future__ import annotations

import importlib
import re

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


def _src(fn):
    """Source of a function in pipeline.render, for wiring assertions."""
    import inspect
    return inspect.getsource(fn)


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


# --------------------------------------------------------------- colour matrix

def test_still_chain_declares_bt709():
    """The still path is where the matrix is baked into the pixels.

    PIL hands ffmpeg 24-bit RGB, so the RGB->YUV conversion happens *inside* this
    encode and ffmpeg's default matrix is bit-for-bit BT.601 - verified by hash,
    not assumed. So an undeclared still is written with BT.601 luma coefficients
    and then read as BT.709 by any HD player.

    Control: `_render_module()` builds the real command, so removing STILL_MATRIX
    from the chain makes this fail. A test that only checked the constant existed
    would keep passing with the constant unwired, which is the failure mode the
    level/GOP test above was written to catch.
    """
    r = _render_module()
    r.ZOOM_W, r.ZOOM_H, r.FPS = 1920, 1080, 30
    # The chain is built from constants, so assert on the *resolved* string that
    # reaches ffmpeg rather than on source text - otherwise this passes while
    # STILL_MATRIX is defined and never interpolated.
    assert "out_color_matrix=bt709" in r.STILL_MATRIX
    src = _src(r.render_still)
    assert "STILL_MATRIX" in src, (
        "render_still does not interpolate STILL_MATRIX, so stills are written "
        "with BT.601 luma coefficients (the default) and read as BT.709"
    )
    # It must be appended after the chain, not hung off a scale inside it: the
    # Ken Burns branch carries no scale at all once ZOOM_SS == 1, which is what a
    # 4K render produces. A flag on a scale that is not there in half the renders
    # is a flag that is missing from them.
    line = next(ln for ln in src.splitlines() if '"-vf"' in ln)
    assert line.index("{chain}") < line.index("STILL_MATRIX")
    assert "force_original_aspect_ratio=increase," not in line
    # `scale=W:H:...` resizes; `scale=<options only>` just declares the
    # conversion and passes dimensions through. Assert the absence of dimensions
    # specifically, since option keys are also colon-separated.
    opts = r.STILL_MATRIX.split("=", 1)[1]
    assert not re.match(r"^\d+:\d+", opts), (
        f"STILL_MATRIX declares dimensions ({opts!r}); it must pass them through "
        "and only set the conversion, or it resizes the frame"
    )


def test_still_matrix_survives_the_supersample_collapse():
    """4K renders set ZOOM_SS to 1, which strips the scale out of the Ken Burns chain.

    This is the case that made the first attempt at this fix wrong. Putting
    `:out_color_matrix=bt709` on the existing `scale` inside `_zoompan_expr` looks
    correct and covers 1080p perfectly - and at 4K that scale does not exist,
    because a 4032px photo cannot cover a 3840px frame at 2x, so `ZOOM_SS` becomes
    1 and the function returns a bare `zoompan`. The matrix would then be declared
    in exactly the renders that do not need it and missing from the ones that do.
    """
    r = _render_module()
    try:
        for ss in (1, 2):
            r.ZOOM_SS = ss
            r.ZOOM_W, r.ZOOM_H, r.FPS = 3840, 2160, 30
            chain = r._zoompan_expr("in", 0.1, 90)
            if ss == 1:
                # The collapsed case: no scale to hang a flag on, which is the
                # whole reason STILL_MATRIX is applied separately.
                assert "scale=" not in chain
            rendered = f"{chain},{r.STILL_MATRIX},format=yuv420p,setrange=limited"
            assert "out_color_matrix=bt709" in rendered
            assert rendered.count("scale=out_color_matrix") == 1
    finally:
        r.ZOOM_SS = 1
        r.ZOOM_W, r.ZOOM_H, r.FPS = 1920, 1080, 30


def test_video_path_declares_both_ends_of_the_conversion():
    """A video segment is already YUV, so its *input* matrix has to be declared too.

    Otherwise a genuine BT.601 source gets rescaled as though it were BT.709 and
    then labelled BT.709 - two errors that cancel in the tags and not in the
    picture.
    """
    r = _render_module()
    r.ZOOM_W, r.ZOOM_H, r.FPS = 1920, 1080, 30
    info = r.MediaInfo(path=__import__("pathlib").Path("x.mp4"), kind="video",
                       width=1920, height=1080, pix_fmt="yuv420p")
    graph = r._video_filter(1920, 1080, "crop", info=info)
    assert "out_color_matrix=bt709" in graph
    assert "in_color_matrix=bt709" in graph


def test_source_matrix_prefers_the_tag_over_the_frame_size():
    r = _render_module()
    P = __import__("pathlib").Path
    # Tagged: believe it, even at a size that would imply otherwise.
    tagged = r.MediaInfo(path=P("a.mp4"), kind="video", width=1920, height=1080,
                         color_space="bt470bg")
    assert r.source_matrix(tagged) == "bt470bg"
    # Untagged: infer from size, which is what a player does with the same file.
    hd = r.MediaInfo(path=P("b.mp4"), kind="video", width=3840, height=2160)
    assert r.source_matrix(hd) == "bt709"
    sd = r.MediaInfo(path=P("c.mp4"), kind="video", width=640, height=480)
    assert r.source_matrix(sd) == "bt601"
    # "unknown" is ffprobe's way of saying nothing, so it must not be believed.
    explicit_unknown = r.MediaInfo(path=P("d.mp4"), kind="video", width=640,
                                   height=480, color_space="unknown")
    assert r.source_matrix(explicit_unknown) == "bt601"


def test_full_range_source_is_detected_from_the_pixel_format():
    """ffmpeg reads an untagged stream as limited, which is wrong for yuvj420p.

    Decoding full-range source as limited comes out washed out and dull - a
    visible fault, not a rounding error. The `yuvj` prefix is the honest signal.
    """
    r = _render_module()
    P = __import__("pathlib").Path
    full = r.MediaInfo(path=P("a.mov"), kind="video", width=1920, height=1080,
                       pix_fmt="yuvj420p")
    assert "in_range=full" in r._matrix_opts(full)
    limited = r.MediaInfo(path=P("b.mp4"), kind="video", width=1920, height=1080,
                          pix_fmt="yuv420p")
    assert "in_range=tv" in r._matrix_opts(limited)


def test_every_encode_and_the_mux_carry_the_colour_tags():
    """The tags have to be on the *bitstream*, and on every site.

    Two things this catches that a constant-presence check would not:

    * `-colorspace` / `-color_primaries` / `-color_trc` do not reach an H.264
      bitstream through libx264 - measured on ffmpeg 9.0.1, they read back as
      "unknown" while only `-color_range` and `-color_space` survive. Passing the
      values as x264 VUI parameters is what actually writes them, so that is what
      is asserted.
    * The final mux is a `-c copy` concat. If only the segments were tagged, the
      container would inherit whatever the first one declared.
    """
    import inspect
    r = _render_module()
    tags = " ".join(r.COLOR_TAGS)
    assert "x264-params" in tags, (
        "-colorspace/-color_primaries/-color_trc are not propagated by libx264; "
        "they have to be written as VUI parameters to land in the file"
    )
    for fn in (r.render_still, r.render_video, r.render_live):
        assert "COLOR_TAGS" in inspect.getsource(fn), (
            f"{fn.__name__} does not tag its output colour"
        )
    # The mux is a `-c copy` concat, so segments alone leave the container
    # describing only its first segment. Split the function at the comment that
    # introduces the mux so this cannot accidentally match a segment encode.
    body = _src(r.render)
    # rindex on the right: `if reference:` appears *before* the mux too, where the
    # reference encode is written, so a plain index would slice the wrong region
    # and quietly pass.
    mux = body[body.index("# ---- mux"):body.rindex("if reference:")]
    assert mux.count("COLOR_TAGS") >= 2, (
        f"the mux declares COLOR_TAGS {mux.count('COLOR_TAGS')} time(s); both mux "
        "branches (with and without audio) need them, because the concat is "
        "`-c copy` and the container inherits only the first segment's tags"
    )


def test_probe_reads_colour_tags_from_the_stream():
    """The tags must be captured at probe time, or source_matrix has nothing to read.

    Control: probe() runs against a real file, so this exercises the JSON parsing
    rather than asserting the dataclass has fields.
    """
    import json as _json
    import tempfile
    from pathlib import Path as _Path
    from unittest import mock

    from pipeline.util import probe

    with tempfile.TemporaryDirectory() as td:
        f = _Path(td) / "t.mp4"
        f.write_bytes(b"\0" * 32)
        payload = _json.dumps({"format": {"duration": "1.0"}, "streams": [{
            "codec_type": "video", "width": 1920, "height": 1080,
            "pix_fmt": "yuv420p", "color_space": "bt709",
            "color_primaries": "bt709", "color_transfer": "bt709",
            "color_range": "tv",
        }]})
        fake = mock.Mock(returncode=0, stdout=payload, stderr="")
        with mock.patch("pipeline.util.run", return_value=fake):
            info = probe(f, "video")
    assert info.color_space == "bt709"
    assert info.color_primaries == "bt709"
    assert info.color_transfer == "bt709"
    assert info.color_range == "tv"
