"""Stage 7 - render.

Strategy: every timeline entry becomes a small, uniformly-encoded intermediate
clip, then they are concatenated and mixed. Chaining hundreds of inputs into
one giant filter graph is what makes naive ffmpeg slideshow scripts eat 50 GB
of RAM; this stays flat and scales to full-length timelines.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .config import Pipeline
from .music import MusicResult
from .select import CutList, Entry
from .util import (
    SEGMENT_PEAK_GB,
    MediaInfo,
    ToolError,
    default_jobs,
    ensure_dir,
    human_duration,
    log,
    progress,
    progress_done,
    run,
)

AUDIO_RATE = 48000
# Extra resolution fed to zoompan so the pan/zoom does not soften the image.
HEADROOM = 1.20

# Ceiling on how much bigger than the delivery frame zoompan renders, before a
# Lanczos reduction. 1 disables the reduction. Measured on a real 4032x3024
# phone photo: 5.402 mean edge energy at 1x, 5.688 at 2x, 5.700 at 4x - so 2x
# takes essentially all of the available gain at 40% of the cost, because the
# source resolution caps what more can recover. Costs ~17x the filter time per
# still over 1x, so it is a knob rather than a constant; see the note in
# _zoompan_expr for the measurement this came from, and the note in render()
# for why the effective factor is computed per run rather than fixed here.
ZOOM_SS_CAP = 2
ZOOM_SS = 1

# Resampling kernel for the video path. Lanczos rather than ffmpeg's default
# bicubic; see the note in _video_filter for the measurement.
SCALE_FLAGS = "lanczos"

# ============================================================ colour matrix
#
# This is the largest quality defect the pipeline had, and it was invisible.
#
# ffmpeg's default RGB->YUV matrix is bit-for-bit BT.601 - verified, not assumed:
# the SHA-256 of a default conversion is identical to explicit
# `out_color_matrix=bt601` and differs from `bt709`, at 1080p and at 4K alike.
# It then writes the file with no colour tags at all, and every HD player
# resolves "unknown" as BT.709.
#
# So a still was *written* with BT.601 luma coefficients and *read* as BT.709.
# Measured across the 40 photos in the library: per-photo median dE76 18.98
# against a ~2.3 just-noticeable-difference, worst case 105.69 - and declaring
# BT.709 brings the median to 2.51, which is the 8-bit 4:2:0 subsampling floor
# rather than leftover colour error. 0 of 40 photos got worse.
#
# See docs/research/filter-quality-and-colour-harmonisation.md section 2.
#
# Three things follow, and all three are needed:
#
#   * declare the matrix on every encode, not just the stills - the segments are
#     concatenated with `-c copy`, so a file assembled from mixed signalling
#     carries signalling that describes none of its own frames;
#   * declare the *input* matrix too on the video path, so genuinely BT.601
#     source is converted properly instead of rescaled and then mislabelled;
#   * tag the container, so correctness stops depending on the reader guessing.

COLOR_MATRIX = "bt709"

# libx264 does NOT propagate `-colorspace` / `-color_primaries` / `-color_trc`
# into the bitstream. Measured on ffmpeg 9.0.1: those three flags leave
# color_transfer and color_primaries reading back as "unknown" in ffprobe,
# while `-color_range` and `-color_space` survive. Passing the same values as
# x264 VUI parameters writes all four correctly, and it is the bitstream that a
# player reads - container-level tags on an H.264 stream are not consulted in
# preference to VUI. So the encoder parameters are the ones that matter here.
COLOR_TAGS = (
    "-color_range", "tv",
    "-x264-params",
    (f"colorprim={COLOR_MATRIX}:transfer={COLOR_MATRIX}"
     f":colormatrix={COLOR_MATRIX}:range=tv"),
)

# The matrix is appended as its own `scale` on the still path rather than hung off
# whichever `scale` a given chain happens to contain. Two reasons, and the second
# is the one that matters:
#
#   * `_zoompan_expr` returns a bare `zoompan` when ZOOM_SS == 1 - which is exactly
#     what a 4K render does, because a 4032px photo cannot cover a 3840px frame at
#     2x. So the Ken Burns branch has a scale to put a flag on at 1080p and has none
#     at 4K. A flag hung on the existing scale is silently absent from half the
#     renders it looks present in.
#   * With no `w:h` arguments this scale passes dimensions through and only
#     declares the conversion, so it cannot change the frame size - and ffmpeg's
#     RGB->YUV conversion is performed by swscale either way, so declaring it here
#     changes the matrix and nothing else.
STILL_MATRIX = f"scale=out_color_matrix={COLOR_MATRIX}:out_range=tv"


def _infer_matrix(width: int, height: int) -> str:
    """BT.709 at HD and above, BT.601 below - the rule players themselves use."""
    return "bt709" if max(int(width), int(height)) >= 1280 else "bt601"


def tags_for(info: MediaInfo) -> tuple[str, ...]:
    """Bitstream tags for one segment, from what that source actually is.

    A still is RGB on the way in, so its output matrix is simply BT.709. A video
    segment is already YUV and has to be *converted*, so its tag has to describe
    the conversion that was actually performed. Since the conversion is declared
    as BT.709 out on every segment, that is the tag every segment carries - which
    is also what makes the `-c copy` concat below coherent.
    """
    return COLOR_TAGS


def source_matrix(info: MediaInfo) -> str:
    """The matrix a source clip really is: its own tag if it has one, else inferred.

    Phone footage almost never carries a tag - every clip in test_media reports
    "unknown" for all four fields - so in practice this is inference rather than
    knowledge. But it is inference from the frame size, which is precisely what a
    player does with the same file, so the result is the picture a viewer would
    have seen anyway. Declaring it is what stops the guess from being *silent*.
    """
    tag = (info.color_space or "").strip().lower()
    if tag and tag not in ("unknown", "unspecified", "reserved"):
        return tag
    return _infer_matrix(info.width, info.height)


def _source_range(info: MediaInfo) -> str:
    """`full` only when the pixel format says so, otherwise `tv`.

    ffmpeg reads an untagged stream as limited, which is right for phone footage
    but wrong for a `yuvj420p` source - and full-range source decoded as limited
    comes out washed out and dull, which is a visible fault and not a rounding
    error. The `yuvj` prefix is the honest signal that the stream is full-range.
    """
    return "full" if (info.pix_fmt or "").lower().startswith("yuvj") else "tv"


def _matrix_opts(info: MediaInfo) -> str:
    """swscale options for the video path: declare both ends of the conversion."""
    return (f":in_color_matrix={source_matrix(info)}"
            f":in_range={_source_range(info)}"
            f":out_color_matrix={COLOR_MATRIX}:out_range=tv")

# Why a Live Photo clip is played from its first frame.
#
# It is not a skipped optimisation, it is a measured decision. Matching
# each HEIC against every frame of its own clip put the still at a median of
# 54% of the way through, but with a +/-26% spread and a runner-up only 1.00x
# worse than the winner on every pair - meaning the match was not identifying
# anything, just picking noise. A planted-frame control confirmed the weakness:
# it scored the same test 8.9x on frames re-read from the same clip, so the
# failure is that a HEIC and a HEVC decode of the same instant genuinely
# differ, not that the method is broken. With the offset unknown within a
# range that wide, any fixed trim would be a guess - and a wrong one skips
# past the moment or starts the motion before it. So the clip plays whole.
#
# This is also why the audio path does no trimming of its own: picture and
# sound have to start at the same instant, and trimming them separately would
# silently slide the ambience against the movement.


# ============================================================= still frames

_FONT_CANDIDATES = (
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
    ("/System/Library/Fonts/Helvetica.ttc", 0),
    ("/Library/Fonts/Arial.ttf", 0),
    ("/System/Library/Fonts/SFNS.ttf", 0),
)


def _font(size: int, bold: bool = True):
    order = list(_FONT_CANDIDATES)
    if not bold:
        regular = [c for c in order if "Bold" not in c[0]]
        order = regular + order
    for path, index in order:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=index)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    scale = max(w / img.width, h / img.height)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


def _contain(img: Image.Image, w: int, h: int) -> Image.Image:
    scale = min(w / img.width, h / img.height)
    return img.resize((max(1, round(img.width * scale)),
                       max(1, round(img.height * scale))), Image.LANCZOS)


def fit_for(aspect: float, cfg: Pipeline) -> str:
    """How this one shot should be arranged inside the frame.

    A 16:9 frame and a 4:3 landscape photo disagree about 25% of the width.
    Centre-cropping that costs a strip of sky and foreground and reads as a
    deliberate crop; blur-filling it puts visible bars down both sides of every
    single shot and reads as a bug. So a landscape shot is filled edge-to-edge.

    A portrait shot disagrees about far more - cropping a 9:16 phone photo to
    16:9 discards most of the picture - so it keeps the blur treatment and
    nothing is cut. `fit` still decides what happens inside each case, so
    "crop" and "pad" behave exactly as before; only "blur" is refined.
    """
    if not cfg.render.fit_per_shot or cfg.render.fit != "blur":
        return cfg.render.fit
    frame = cfg.render.width / max(1, cfg.render.height)
    # Wider than the frame, or wide enough that cropping costs little.
    return "crop" if aspect >= frame * cfg.render.fit_crop_min_aspect else "blur"


def compose_pair(left: Image.Image, right: Image.Image,
                 w: int, h: int) -> Image.Image:
    """Two photos side by side, filling a w x h frame between them.

    Each half gets the same treatment a lone portrait shot would - fitted to
    the half without being cropped, over a blurred copy of itself - so a spread
    reads as a photo album rather than as two photos floating in a void. The
    gap between them is a hairline rather than a fat border, so the pair still
    looks like one frame.
    """
    gap = max(2, w // 320)
    half = (w - gap) // 2
    a = compose(left, half, h, "blur")
    b = compose(right, w - gap - half, h, "blur")
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (half + gap, 0))
    return canvas


def compose(img: Image.Image, w: int, h: int, fit: str = "blur") -> Image.Image:
    """Normalise any aspect ratio into exactly w x h without losing content."""
    if fit == "crop":
        return _cover(img, w, h)
    if fit == "pad":
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        fg = _contain(img, w, h)
        canvas.paste(fg, ((w - fg.width) // 2, (h - fg.height) // 2))
        return canvas

    # "blur": fill the frame with a softened copy of the same photo, then lay
    # the uncropped photo on top. Works for portrait phone shots in a
    # landscape timeline without cutting anyone's head off.
    bg = _cover(img, w, h)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=max(12, int(w / 42))))
    bg = ImageEnhance.Brightness(bg).enhance(0.72)
    fg = _contain(img, w, h)
    bg.paste(fg, ((w - fg.width) // 2, (h - fg.height) // 2))
    return bg


# ============================================================ title cards

def title_card(entry: Entry, cfg: Pipeline, bg_source: Image.Image | None,
               out_png: Path) -> None:
    w, h = cfg.render.width, cfg.render.height

    if bg_source is not None:
        canvas = _cover(bg_source, w, h)
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=max(22, int(w / 20))))
        canvas = ImageEnhance.Brightness(canvas).enhance(0.44)
        canvas = ImageEnhance.Color(canvas).enhance(0.80)
    else:
        canvas = Image.new("RGB", (w, h))
        d = ImageDraw.Draw(canvas)
        for y in range(h):
            t = y / h
            d.line([(0, y), (w, y)],
                   fill=(int(18 + 26 * t), int(22 + 30 * t), int(34 + 44 * t)))

    # Scrim so text always reads.
    scrim = Image.new("L", (w, h), 0)
    sd = ImageDraw.Draw(scrim)
    sd.rectangle([0, int(h * 0.42), w, h], fill=190)
    scrim = scrim.filter(ImageFilter.GaussianBlur(radius=h // 9))
    canvas = Image.composite(Image.new("RGB", (w, h), (6, 8, 12)), canvas, scrim)

    draw = ImageDraw.Draw(canvas)
    margin = int(w * 0.085)
    title = (entry.title or "").strip()
    sub = (entry.subtitle or "").strip()

    tsize = max(34, int(h * (0.105 if len(title) < 26 else 0.078)))
    tfont = _font(tsize)
    while tfont and draw.textlength(title, font=tfont) > w - 2 * margin:
        tsize = int(tsize * 0.92)
        if tsize <= 28:
            break
        tfont = _font(tsize)

    # Baseline block sits in the lower third.
    sub_font = _font(max(20, int(h * 0.036)), bold=False)
    sub_h = int(h * 0.052) if sub else 0
    block_h = tsize + sub_h
    y = int(h * 0.70) - block_h // 2

    if title:
        draw.text((margin, y), title, font=tfont, fill=(255, 255, 255))
        y += tsize + int(h * 0.012)
    if sub:
        draw.text((margin, y), sub, font=sub_font, fill=(228, 232, 240))

    # Accent rule above the title.
    rule_y = y - tsize - int(h * 0.030) if title else y - int(h * 0.02)
    draw.rectangle([margin, rule_y, margin + int(w * 0.085), rule_y + 5],
                   fill=(255, 214, 130))

    canvas.save(out_png, quality=94)


# =============================================================== ken burns

def _zoompan_expr(motion: str, amount: float, frames: int) -> str:
    """Build the zoompan filter string for one move.

    The progress term is eased, not linear. A constant-rate ramp is at full
    speed right up to the cut, and the next shot then starts from rest, so
    every cut carries a velocity discontinuity - it reads as drift rather than
    as a camera move that settled. This uses Perlin's smootherstep, whose first
    *and* second derivatives are both zero at the ends, so the shot starts and
    finishes genuinely at rest and cuts land without a visible jolt. Following
    the same reasoning as the CSS curve table: motion already on screen wants
    ease-in-out; linear is for constant motion like a spinner or a marquee.
    """
    a = max(0.01, min(0.45, amount))
    last = max(1, frames - 1)
    t = f"(min(on,{last})/{last})"
    # smootherstep(t) = t^3 (6t^2 - 15t + 10)
    p = f"({t})*({t})*({t})*(({t})*(6*({t})-15)+10)"
    cx = "iw/2-(iw/zoom/2)"
    cy = "ih/2-(ih/zoom/2)"

    if motion == "in":
        zoom, x, y = f"1+{a}*{p}", cx, cy
    elif motion == "out":
        zoom, x, y = f"1+{a}*(1-{p})", cx, cy
    elif motion == "left":
        zoom, x, y = f"1+{a}", f"(iw-iw/zoom)*(1-{p})", cy
    elif motion == "right":
        zoom, x, y = f"1+{a}", f"(iw-iw/zoom)*{p}", cy
    elif motion == "up":
        zoom, x, y = f"1+{a}", cx, f"(ih-ih/zoom)*(1-{p})"
    elif motion == "down":
        zoom, x, y = f"1+{a}", cx, f"(ih-ih/zoom)*{p}"
    else:  # "still"
        zoom, x, y = "1.0", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"

    # zoompan is asked for a frame several times the delivery size and the
    # result is scaled back down. zoompan samples the source itself with a
    # nearest-neighbour kernel, so asking it for a 1:1 1080p frame loses about
    # a tenth of the visible edge detail - measured on a real 12MP photo at
    # 5.252 -> 4.727 mean edge energy against a straight Lanczos downscale of
    # the same frame. Rendering the same move at 4x and reducing afterwards
    # recovers most of it (5.009, -4.6%), because each output pixel is then
    # chosen from four times as many source pixels. That softening was the
    # single largest cause of the picture looking low-resolution.
    ss_w, ss_h = ZOOM_W * ZOOM_SS, ZOOM_H * ZOOM_SS
    zoompan = (f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:"
               f"s={ss_w}x{ss_h}:fps={FPS}")
    if ZOOM_SS == 1:
        return zoompan
    return (f"{zoompan},scale={ZOOM_W}:{ZOOM_H}:flags=lanczos,"
            f"setsar=1")


# Per-render globals set by prepare_render(); kept module-level so the ffmpeg
# filter strings above stay readable.
FPS = 30
ZOOM_W = 1920
ZOOM_H = 1080


def _h264_level() -> str:
    """The smallest H.264 level that actually carries this frame size.

    Level 4.0 tops out at 2048x1080 at 30fps, so hardcoding it made every
    encode above 1080p illegal - x264 either refuses outright or the output
    plays back at the wrong rate on hardware that trusts the level rather than
    the stream. Picked from the macroblock rate rather than a lookup table,
    because "is this level big enough" is exactly the question a fixed string
    gets wrong when --size and --fps are both free.
    """
    # Level caps, in macroblocks per second: 4.1 = 245760, 5.1 = 983040,
    # 5.2 = 2073600, 6.0 = 4177920.
    mb_w = (ZOOM_W + 15) // 16
    mb_h = (ZOOM_H + 15) // 16
    rate = mb_w * mb_h * max(1, FPS)
    for level, cap in (("4.0", 245_760), ("4.1", 245_760), ("4.2", 522_240),
                       ("5.0", 589_824), ("5.1", 983_040), ("5.2", 2_073_600),
                       ("6.0", 4_177_920), ("6.1", 8_355_840),
                       ("6.2", 16_711_680)):
        if rate <= cap:
            return level
    return "6.2"


def _gop() -> int:
    """Keyframe interval, held at about two seconds whatever the frame rate.

    Hardcoded at 60 it meant two seconds at 30fps and one at 60, which halves
    the seek granularity exactly when the file is already the biggest one being
    produced - the wrong direction.
    """
    return max(2, FPS * 2)


def render_still(png: Path, out_mp4: Path, cfg: Pipeline, duration: float,
                 motion: str, zoom: float | None = None) -> None:
    """Animate one still into a clip of the requested length.

    `zoom` overrides the configured amount for this shot only. It is how a 2s
    stab moves further than a 6s drift without the caller having to rewrite
    cfg.render.zoom_amount for every entry; None keeps the global default, so
    every other call site is unaffected.
    """
    frames = max(2, int(round(duration * FPS)))
    amount = cfg.render.zoom_amount if zoom is None else zoom
    if cfg.render.ken_burns:
        chain = _zoompan_expr(motion, amount, frames)
    else:
        chain = (f"scale={ZOOM_W}:{ZOOM_H}:force_original_aspect_ratio=increase,"
                 f"crop={ZOOM_W}:{ZOOM_H},setsar=1")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-loop", "1", "-framerate", str(FPS), "-i", str(png),
        # STILL_MATRIX is appended here for *both* branches rather than hung off
        # whichever scale a branch happens to contain, because the Ken Burns
        # branch has no scale at all once ZOOM_SS == 1 - which is exactly what a
        # 4K render does. See the STILL_MATRIX comment.
        #
        # setrange=limited stays: it is a no-op on a PNG input, but it is the
        # guard against a yuvj420p segment reaching the concat demuxer and
        # propagating its full-range signalling to the whole film.
        "-vf", f"{chain},{STILL_MATRIX},format=yuv420p,setrange=limited",
        "-frames:v", str(frames),
        "-an",
        "-c:v", "libx264", "-crf", str(cfg.render.crf),
        "-preset", cfg.render.preset,
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-profile:v", "high", "-level", _h264_level(),
        "-g", str(_gop()), "-keyint_min", str(_gop()), "-sc_threshold", "0",
        "-video_track_timescale", "90000",
        *COLOR_TAGS,
        str(out_mp4),
    ]
    run(cmd, timeout=900)


def _video_filter(width: int, height: int, fit: str, extra: str = "",
                  info: MediaInfo | None = None) -> str:
    """Scale one source clip to the working frame size.

    Always returns a fully labelled graph ending in ``[v]``. Crop and pad look
    like a single unlabelled filter, which is fine for ``-vf`` but not for
    ``-filter_complex``: without a label ffmpeg expects an unlabelled input
    stream to feed the chain's input pad, and since the caller maps the source
    stream directly there is nothing left to bind, so the render fails.

    `extra` is appended inside the graph, before the colour handling. It has to
    go here rather than in a -vf of its own: ffmpeg will not apply simple and
    complex filtering to the same stream, so anything added alongside one of
    these chains has to be part of it.

    `info` is the source's probe result, used to declare what colour the incoming
    frames actually are. Without it the scale runs on ffmpeg's assumption, which
    is BT.601 - correct only for SD, and for a still it silently bakes the wrong
    luma coefficients into the pixels. See the colour matrix note above.
    """
    tail = f",{extra}" if extra else ""
    # Lanczos, not ffmpeg's default bicubic. This is the *video* path, and
    # unlike the stills - which already go through PIL's Lanczos in _cover and
    # _contain - every frame of every clip was being resampled with bicubic
    # here. Against the same source that measures 5.6% less edge energy, and on
    # a 4K frame the shorter effective kernel is what puts a soft look on the
    # whole timeline.
    #
    # No `info` means no honest matrix is available, so nothing is declared and
    # ffmpeg's BT.601 default stands. Every real caller passes it; the default
    # exists for tests and for a caller that genuinely has nothing to probe, and
    # the verifier check on colour_space is what catches a real render that
    # somehow took this path.
    mat = _matrix_opts(info) if info is not None else ""
    sc = f":flags={SCALE_FLAGS}{mat}"
    if fit == "crop":
        return (f"[0:v]scale={width}:{height}:"
                f"force_original_aspect_ratio=increase{sc},"
                f"crop={width}:{height},setsar=1{tail},fps={FPS},"
                f"format=yuv420p,setrange=limited[v]")
    if fit == "pad":
        return (f"[0:v]scale={width}:{height}:"
                f"force_original_aspect_ratio=decrease{sc},"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1{tail},fps={FPS},format=yuv420p,setrange=limited[v]")
    # blur
    return (
        f"[0:v]split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={width}:{height}:force_original_aspect_ratio=increase{sc},"
        f"crop={width}:{height},boxblur=18:2,eq=brightness=-0.10[bg];"
        f"[fgsrc]scale={width}:{height}:force_original_aspect_ratio=decrease{sc}[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{tail},fps={FPS},"
        f"format=yuv420p,setrange=limited[v]"
    )


def render_live(entry: Entry, cfg: Pipeline, out_mp4: Path, *,
                sw: int | None = None, sh: int | None = None) -> None:
    """Play a Live Photo's motion clip as this shot.

    The still and its clip are one picture in the Photos app, and rendering
    them as one picture here is the whole point of pairing them. Three details
    decide whether it looks right:

    * The clip plays whole. A Live Photo MOV begins with roughly half a second
      recorded *before* the shutter, so starting at zero shows the moment
      assembling rather than happening. Skipping that lead-in would be better,
      but the offset could not be measured reliably (see the note on HEADROOM),
      so the whole clip plays rather than a guessed cut.
    * Motion is kept at its own pace. Stretching 2s of footage across a 4s slot
      to fill the beat would slow a walk to a crawl and make the movement look
      wrong, so the clip plays once at its own speed.
    * The tail keeps drifting rather than stopping. A Live Photo is usually much
      shorter than the slot it has to fill - measured across this trip's library,
      0.70s to 2.71s of motion against a 2.80s slot - so most of a Live Photo
      shot used to be a cloned last frame: entry 18 held 1.92s of its 2.80s and
      entry 40 held 2.10s, three quarters of the shot dead. Measured on entry
      18's real clip: 0 of 54 tail frames moved. That is what a freeze-frame bug
      looks like, and it was the largest single source of the "the motion feels
      cheap" complaint. Holding is right in the Photos app because you are the
      one holding it and you expect that; nobody asked for it in a cut film. So
      the clip's last frame gets a slow eased Ken Burns push for the remainder,
      which is the documentary idiom for settling on a moment.
    * Ken Burns is not panned on top of the clip itself. The clip already moves,
      so panning during playback gives two competing motions and reads as a
      wobble. The push only begins once the clip has ended.

    `sw`/`sh` are the supersampled canvas the stills are prepared at, needed so
    the push samples the same pixels the rest of the stills do. Left optional
    so a caller that does not have them still gets a correct push, just at
    delivery resolution.
    """
    assert entry.item is not None
    src = entry.item.info.live_motion
    assert src is not None
    dur = max(0.08, entry.duration)
    width, height = ZOOM_W, ZOOM_H
    frames = max(2, int(round(dur * FPS)))

    info = entry.item.info
    aspect = (info.width / info.height) if (info.width and info.height) else 0.0
    fit = fit_for(aspect, cfg) if aspect else cfg.render.fit

    play, tail = _live_split(dur, _probe_motion(src))

    def encode(frames_n: int, vf: str) -> list[str]:
        return [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(src),
            "-filter_complex", vf,
            "-map", "[v]",
            "-frames:v", str(frames_n),
            "-an",
            "-c:v", "libx264", "-crf", str(cfg.render.crf),
            "-preset", cfg.render.preset,
            "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-profile:v", "high", "-level", _h264_level(),
            "-g", str(_gop()), "-keyint_min", str(_gop()), "-sc_threshold", "0",
            "-video_track_timescale", "90000",
            *COLOR_TAGS,
        ]

    if tail <= TAIL_MIN:
        # The clip fills its slot, so there is no held frame to rescue.
        run(encode(frames, _video_filter(width, height, fit, "", info))
            + [str(out_mp4)], timeout=900)
        return

    clip_mp4 = out_mp4.with_name(f"{out_mp4.stem}_clip.mp4")
    push_mp4 = out_mp4.with_name(f"{out_mp4.stem}_push.mp4")
    tail_png = out_mp4.with_name(f"{out_mp4.stem}_tail.png")
    try:
        # Part one: the clip, at its own pace, trimmed to the point it ends.
        run(encode(max(2, int(round(play * FPS))),
                   _video_filter(width, height, fit, "", info))
            + [str(clip_mp4)], timeout=900)

        # Part two: a slow push on the frame the clip ended on. Taken from the
        # clip rather than from the still, so the push continues from exactly
        # where the motion left off instead of jumping back to the shutter.
        _push_from_last_frame(src, fit, tail_png, sw or width, sh or height)
        render_still(tail_png, push_mp4, cfg, tail, "in", TAIL_PUSH)

        run([
            "ffmpeg", "-y", "-v", "error",
            "-i", str(clip_mp4), "-i", str(push_mp4),
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]",
            "-frames:v", str(frames),
            "-an",
            "-c:v", "libx264", "-crf", str(cfg.render.crf),
            "-preset", cfg.render.preset,
            "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-profile:v", "high", "-level", _h264_level(),
            "-g", str(_gop()), "-keyint_min", str(_gop()), "-sc_threshold", "0",
            "-video_track_timescale", "90000",
            *COLOR_TAGS,
            str(out_mp4),
        ], timeout=900)
    finally:
        for tmp in (clip_mp4, push_mp4, tail_png):
            try:
                tmp.unlink()
            except OSError:
                pass


def _push_from_last_frame(src: Path, fit: str, out_png: Path,
                          w: int, h: int) -> None:
    """Write the final frame of a Live Photo clip as a prepared canvas PNG.

    A tenth of a second back from the end rather than exactly at it: seeking to
    the very last timestamp can land past the final frame on some decoders and
    return nothing. Orientation is left alone because video frames are already
    upright - the EXIF tag that matters for a HEIC has no meaning here.
    """
    from .quality import load_image
    run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.1", "-i", str(src),
         "-frames:v", "1", "-update", "1", str(out_png)], timeout=300)
    img = load_image(out_png, max_long_side=0, orientation=1)
    if img is None:
        raise ToolError(f"could not read the last frame of {src.name}")
    compose(img, w, h, fit).save(out_png)


# How far the held tail of a Live Photo pushes in, and the shortest tail worth
# rescuing. Small on purpose: this is a settle, not a second camera move, and it
# has to read as the same shot continuing rather than as a cut into a new one.
#
# TAIL_MIN is where a hold stops being invisible. The shortfall has to be worth
# two extra encodes and a concat, and a couple of dozen frames is not: 0.09s is
# three frames at 30fps, which nobody sees, and it was being paid for on every
# Live Photo whose clip ran slightly long. The real cases were 1.0s to 2.1s.
# A quarter of a second is about seven frames - under that, a brief stillness
# reads as a pause between beats rather than as a freeze.
TAIL_PUSH = 0.055
TAIL_MIN = 0.25


def _live_split(dur: float, clip: float) -> tuple[float, float]:
    """Split a Live Photo shot into (seconds of clip, seconds of push).

    Separate from ``render_live`` so the decision is a comparison between two
    numbers and can be tested without an encoder. The clip plays at its own
    speed and is never stretched - only made up, by a push on its last frame.
    """
    if not clip or clip <= 0.0:
        return dur, 0.0
    play = max(0.3, min(dur, clip))
    return play, max(0.0, dur - play)


_MOTION_DURATION: dict[Path, float] = {}


def _probe_motion(src: Path) -> float:
    """Length of a Live Photo clip, probed once and remembered.

    Only the handful of Live Photos that make the cut are ever probed here, so
    the cache stays tiny; it exists because render_live may be called for the
    same source twice when a still is retried.
    """
    if src in _MOTION_DURATION:
        return _MOTION_DURATION[src]
    dur = 0.0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(src)],
            capture_output=True, text=True, timeout=60).stdout.strip()
        dur = float(out)
    except (ValueError, OSError, subprocess.SubprocessError):
        dur = 0.0
    _MOTION_DURATION[src] = dur
    return dur


def render_video(entry: Entry, cfg: Pipeline, out_mp4: Path) -> None:
    """Trim one segment of one source clip to the assigned duration.

    The frame count is pinned explicitly so the encoded length matches the
    beat arithmetic exactly; relying on -t alone can land a frame short.
    """
    assert entry.item is not None
    src = entry.item.info.path
    start = max(0.0, entry.src_in)
    dur = max(0.08, entry.duration)
    frames = max(2, int(round(dur * FPS)))
    width, height = ZOOM_W, ZOOM_H

    # Same per-shot decision as stills. For video the aspect comes from
    # MediaInfo.display_size, which probe() has already reconciled against what
    # ffmpeg really decodes - 35 of the 61 clips in the real edit carry a -90
    # display matrix and are genuinely portrait once upright. An unknown size
    # falls back to `fit` unchanged rather than guessing.
    info = entry.item.info
    aspect = (info.width / info.height) if (info.width and info.height) else 0.0
    fit = fit_for(aspect, cfg) if aspect else cfg.render.fit

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-i", str(src),
        "-t", f"{dur + 0.5:.3f}",
        "-an",
        "-filter_complex", _video_filter(width, height, fit, info=info),
        "-map", "[v]",
        "-frames:v", str(frames),
        "-c:v", "libx264", "-crf", str(cfg.render.crf),
        "-preset", cfg.render.preset,
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-profile:v", "high", "-level", _h264_level(),
        "-g", str(_gop()), "-keyint_min", str(_gop()), "-sc_threshold", "0",
        "-video_track_timescale", "90000",
        *COLOR_TAGS,
        str(out_mp4),
    ]
    run(cmd, timeout=900)


# ==================================================================== audio

def extract_segment_audio(entry: Entry, cfg: Pipeline, out_wav: Path) -> None:
    """Pull the segment's native audio at its own level. Silent for stills.

    There is deliberately no `loudnorm` here. It used to run on every segment,
    which was wrong three separate ways:

    * With no `measured_*` inputs it is single-pass *dynamic* mode, which re-gains
      as it goes - textbook pumping, measured at ~0.19s of drift over 120s.
    * It normalised each shot to the same -18 LUFS, so a quiet market and a loud
      market both arrived at -18 and the difference between them was gone. Shot
      to shot loudness relationships are part of what makes a sequence feel like
      a place rather than a contact sheet.
    * Most of this project's shots are room tone measured around -37 dB mean.
      Taking that to -18 LUFS is roughly a +19 dB lift on a noise floor, and
      amplifying and gating a noise floor is how you get pumping hiss.

    The programme is normalised once, as a whole, in `_normalise_programme`.
    """
    dur = max(0.08, entry.duration)
    item = entry.item

    fade = "afade=t=in:st=0:d=0.06,areverse,afade=t=in:st=0:d=0.10,areverse"
    chain = f"{fade},aformat=sample_fmts=s16:sample_rates={AUDIO_RATE}:channel_layouts=stereo"

    def silence() -> None:
        run(["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i",
             f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
             "-t", f"{dur:.3f}", "-c:a", "pcm_s16le", str(out_wav)])

    # A Live Photo still is silent by definition, but the clip recorded
    # alongside it carries the real room sound of that moment - measured around
    # -37 dB mean, quiet but genuine. Reading it here is what turns a silent
    # film into one with sound in it, and it is the same audio already used to
    # decide what the shot sounds like, so nothing extra is probed.
    live = None
    if item is not None and item.kind == "photo":
        live = item.info.live_motion

    if item is None or item.kind != "video":
        if live is None:
            silence()
            return
        proc = run([
            "ffmpeg", "-y", "-v", "error",
            "-i", str(live),
            "-t", f"{dur:.3f}",
            "-vn", "-map", "0:a:0",
            "-af", chain,
            "-c:a", "pcm_s16le", str(out_wav),
        ], check=False)
        if proc.returncode != 0 or not out_wav.exists():
            silence()
        return

    if not item.info.has_audio:
        silence()
        return

    proc = run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{entry.src_in:.3f}", "-i", str(item.info.path),
        "-t", f"{dur:.3f}",
        "-vn", "-map", "0:a:0",
        "-af", chain,
        "-c:a", "pcm_s16le", str(out_wav),
    ], check=False)
    if proc.returncode != 0 or not out_wav.exists():
        run(["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i",
             f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
             "-t", f"{dur:.3f}", "-c:a", "pcm_s16le", str(out_wav)])


def build_audio_track(entries: list[Path], cfg: Pipeline, music_path: Path,
                      total: float, work: Path, *,
                      music: MusicResult | None = None) -> Path:
    """Concat per-segment audio, then duck the music underneath it."""
    content_wav = work / "content_audio.wav"
    mixed_wav = work / "mixed_audio.wav"

    list_file = work / "audio_concat.txt"
    list_file.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in entries), encoding="utf-8"
    )
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c:a", "pcm_s16le", str(content_wav)])

    # The ducking threshold has to be relative to the content it is ducking
    # under, not a constant. It used to be a hardcoded 0.035 (-29.1 dBFS),
    # which is a bet that the source audio is louder than -29 dBFS - and that
    # bet is wrong often enough to matter. Measured on a quiet library whose
    # clips peak at -20.9 dBFS, the compressor never engaged at all: the
    # verifier reported the music bed only 1.4 dB below the clips' own audio,
    # which is not ducking.
    #
    # A real phone library spans far more than that - clips of a quiet room and
    # clips of a concert are 20 dB apart - so a single constant cannot be
    # right for both. Measuring the assembled content and setting the trigger
    # relative to it makes the ratio mean the same thing whatever the source
    # was recorded at, which is the only way a fixed 6:1 is meaningful.
    duck_threshold = _duck_threshold_for(content_wav)

    # sidechaincompress holds frames internally to compute its release ramp, so
    # it emits slightly less than it is given - measured at ~0.19s short of
    # 120s here. That shortfall used to reach the mux, where -shortest settled
    # it by cutting the *picture* to match the sound. Both inputs are therefore
    # extended by a lookahead tail so the filter can flush, and the result is
    # trimmed back to the exact planned length.
    release_ms = 380
    tail = release_ms / 1000.0 + 0.2
    end = total + tail

    # Start the music on beat zero of the grid.
    chain = ["[1:a]"]
    if music is not None:
        phase = max(0.0, float(getattr(music, "phase", 0.0)))
        if phase > 0.01:
            chain.append(f"atrim=start={phase:.4f},asetpts=PTS-STARTPTS,")
        stretch = float(getattr(music, "stretch", 1.0) or 1.0)
        if abs(stretch - 1.0) > 1e-6:
            chain.append(f"atempo={stretch:.6f},")

    # A deliberate fade to silence at the very end is normal for a recap, and
    # the fade-in at the start is too. Both are part of the shape of the piece,
    # not dead air, so they are left as they are. (Pulling the fade short of the
    # end to satisfy the verifier was the wrong fix: it bends the output to fit
    # the check rather than teaching the check what a fade looks like.)
    fade_out = max(1.5, total * 0.05)
    chain.append("aformat=sample_fmts=fltp:"
                 f"sample_rates={AUDIO_RATE}:channel_layouts=stereo,")
    # Pad before trimming so a track shorter than the video still reaches length.
    chain.append(f"apad,atrim=end={end:.6f},asetpts=N/SR/TB,")
    chain.append(f"volume={cfg.render.music_volume:.3f},")
    chain.append(f"afade=t=in:st=0:d={min(2.5, total * 0.05):.2f},")
    chain.append(f"afade=t=out:st={max(0.0, total - fade_out):.2f}:"
                 f"d={fade_out:.2f}[mus]")
    music_filter = "".join(chain)

    # Per-segment audio is cut to a whole sample count, so the concatenated
    # track can land a few milliseconds under the picture. Pin it exactly.
    content_fmt = (
        f"aformat=sample_fmts=fltp:sample_rates={AUDIO_RATE}:"
        f"channel_layouts=stereo,apad,atrim=end={end:.6f},"
        f"asetpts=N/SR/TB[cont]"
    )
    # Final guarantee: whatever the mixer did, the track is exactly `total`.
    pin = f"apad,atrim=end={total:.6f},asetpts=N/SR/TB"

    if cfg.music.duck:
        # knee=6, not ffmpeg's unset default of 2.82843. A 2.83 knee is nearly a
        # hard knee, and a hard knee on a music bed at 6:1 is an audible step
        # rather than a transition. 6 is in the recommended band for a sidechain
        # of this kind and turns the step back into a ramp.
        mix = (
            f"{music_filter};[0:a]{content_fmt};"
            f"[mus][cont]sidechaincompress=threshold={duck_threshold:.6f}:"
            f"ratio=6:knee=6:"
            f"attack=18:release={release_ms}:makeup=1[ducked];"
            # sidechaincompress has no ceiling of its own, so a loud mix (a
            # driving 'energetic' score under already-hot source audio) can land
            # just over 0 dBTP. The amix branch below already limits; this one
            # needs the same guard, or true peak ends up depending on the mood.
            f"[ducked]alimiter=limit=0.891:level=0,"
        )
    else:
        mix = (
            f"{music_filter};[0:a]{content_fmt};"
            f"[cont][mus]amix=inputs=2:duration=first:dropout_transition=0:"
            f"weights=1.0 1.0,alimiter=limit=0.97,"
        )
    mix += f"{pin}[mixed]"

    run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(content_wav),
        "-stream_loop", "-1", "-i", str(music_path),
        "-t", f"{total:.3f}",
        "-filter_complex", mix,
        "-map", "[mixed]",
        "-c:a", "pcm_s16le",
        str(mixed_wav),
    ], timeout=1800)

    # Normalise the assembled programme, once, two-pass. See
    # _normalise_programme for why this is a separate stage rather than a filter
    # in the graph above.
    final_wav = _normalise_programme(mixed_wav, cfg)
    _assert_music_underneath(final_wav, music_path, total)
    return final_wav


# Ducking is triggered relative to the content's own peak. Measured in dB, not
# linear, because everything a listener hears about level is on a log scale.
#
#   DUCK_BELOW_PEAK_DB   how far under the content's peak the trigger sits. A
#                        small number means the compressor is already working on
#                        the loudest passages; a large one means it waits for
#                        something close to the peak before reacting. 18 dB was
#                        chosen as roughly "the part of the clip a listener
#                        would call speech" rather than its quietest room tone.
#   DUCK_THRESHOLD_FLOOR the linear floor, so a silent library cannot drive the
#                        trigger to zero and make the compressor engage on
#                        numerical noise. -45 dBFS.
DUCK_BELOW_PEAK_DB = 18.0
DUCK_THRESHOLD_FLOOR = 0.0056      # -45 dBFS


def _duck_threshold_db(peak_db: float | None) -> float:
    """The trigger level, in dBFS, for content peaking at `peak_db`.

    Split out from the ffmpeg call so the arithmetic can be tested directly:
    the bug this replaces was a threshold that sat *above* the material it was
    supposed to react to, and that is a comparison between two numbers, so it
    should not need an encoder to test.
    """
    if peak_db is None or peak_db < -60.0:
        # Nothing measurable. Fall back to a threshold low enough to trigger on
        # ordinary speech-level material rather than one tuned to silence.
        return -40.0
    return max(peak_db - DUCK_BELOW_PEAK_DB, -60.0)


def _duck_threshold_for(content_wav: Path) -> float:
    """Measure the assembled content bus and derive the ducking trigger.

    Peaks the *content* bus rather than individual clips: after concat the bus
    is what the compressor actually sees as its sidechain key, and it is
    normalised to a common level by the concat, so one measurement describes
    the whole programme.
    """
    proc = run(["ffmpeg", "-nostats", "-hide_banner", "-i", str(content_wav),
                "-af", "volumedetect", "-f", "null", "-"], check=False)
    peak_db = None
    for line in (proc.stderr or "").splitlines():
        if "max_volume:" in line:
            try:
                peak_db = float(line.split("max_volume:")[1].split("dB")[0])
            except (IndexError, ValueError):
                pass
    threshold_db = _duck_threshold_db(peak_db)
    return max(DUCK_THRESHOLD_FLOOR, 10.0 ** (threshold_db / 20.0))


def _measure_loudness(path: Path) -> dict[str, float] | None:
    """Pass 1 of loudnorm: ask ffmpeg what the programme actually measures."""
    proc = run(["ffmpeg", "-nostats", "-hide_banner", "-i", str(path),
                "-af", "loudnorm=I=-18:TP=-1.5:LRA=11:print_format=json",
                "-f", "null", "-"], check=False)
    blob = proc.stderr or ""
    start = blob.rfind("{")
    end = blob.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(blob[start:end + 1])
    except json.JSONDecodeError:
        return None
    out: dict[str, float] = {}
    for key in ("input_i", "input_lra", "input_tp", "input_thresh",
                "target_offset"):
        try:
            out[key] = float(data[key])
        except (KeyError, TypeError, ValueError):
            continue
    return out or None


def _normalise_programme(mixed_wav: Path, cfg: Pipeline) -> Path:
    """Loudness-normalise the finished programme, as a whole, in two passes.

    This is the stage that the per-segment `loudnorm` used to be, and moving it
    here changes three things at once:

    * Shot-to-shot loudness relationships survive. Normalising each segment
      separately made a quiet market and a loud market both arrive at -18 LUFS,
      which is part of why the edit felt flat: the *differences between places*
      were being deleted before anyone heard them.
    * It is a static gain rather than a moving one. Feeding the measured values
      back in puts loudnorm on its linear path, so the gain is computed once from
      the whole programme and applied evenly - no re-gaining between frames.
    * Room tone is no longer amplified on its own terms. Most shots here are
      ambience around -37 dB mean; as part of a programme it gets the same
      treatment as everything else instead of a ~19 dB lift.

    Two passes because one pass in dynamic mode is exactly the pumping that was
    removed. If the measurement pass cannot be parsed, the mix is left as it is:
    a slightly hot or quiet recap is recoverable, a truncated one is not.
    """
    target_i, target_tp, target_lra = -18.0, -1.5, 11.0
    measured = _measure_loudness(mixed_wav)
    if measured is None:
        log("could not measure programme loudness; leaving the mix un-normalised",
            level="warn")
        return mixed_wav

    applied = (
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        f":measured_I={measured['input_i']:.2f}"
        f":measured_LRA={measured['input_lra']:.2f}"
        f":measured_TP={measured['input_tp']:.2f}"
        f":measured_thresh={measured['input_thresh']:.2f}"
        f":offset={measured.get('target_offset', 0.0):.2f}"
        ":linear=true:print_format=summary"
    )
    final = mixed_wav.with_name("final_audio.wav")
    proc = run(["ffmpeg", "-y", "-v", "info", "-i", str(mixed_wav),
                "-af", applied, "-c:a", "pcm_s16le", str(final)],
               check=False)
    if proc.returncode != 0 or not final.exists():
        log("programme normalisation failed; using the un-normalised mix",
            level="warn")
        return mixed_wav
    return final


def measure_true_peak(path: Path) -> float | None:
    """Measured true peak in dBTP, from the encoded file.

    `alimiter` is a *sample*-peak limiter with no oversampling, so the -1 dBTP
    that EBU R128, ATSC A/85, AES TD1008 and Netflix all specify was never
    actually guaranteed - and AAC encoding creates inter-sample peaks above the
    sample peak on top of that. So rather than trust the graph, measure the file
    that will actually ship. Returns None when it cannot be read, so callers can
    decide whether an unreadable measurement is fatal.
    """
    proc = run(["ffmpeg", "-v", "info", "-i", str(path), "-af", "ebur128=peak=true",
                "-f", "null", "-"], check=False)
    best: float | None = None
    for line in (proc.stderr or "").splitlines():
        m = re.search(r"^\s*Peak:\s*(-?\d+(?:\.\d+)?)", line)
        if m:
            val = float(m.group(1))
            # ebur128 prints a running maximum; the final value is the answer.
            best = val if best is None else max(best, val)
    return best


DEAD_TAIL_S = 1.5
"""Contiguous bit-exact zeros at the tail that mean the bed died, not that it faded.

A natural fade tapers toward zero and only reaches it at the last sample, so
real fades produce no contiguous run at all. These thresholds were set against
controls rather than guessed: fades of 1s, 1.5s, 2s and 4s all pass, a 1s dead
tail passes, and dead tails of 2s, 3s, 4s and 9.3s are all refused. The fault
seen on real media measured 9.28s.
"""


def trailing_silence_s(samples: np.ndarray, rate: int) -> float:
    """Seconds of *contiguous* bit-exact zeros ending the buffer.

    Contiguity is the whole point. Counting the fraction of zeros in a window
    dilutes the signal - a 3s hole inside a 4s window is only 75% zeros, which
    reads as less bad than the 1.5s tail it contains - and a bed that stops
    early is precisely a hole, not a uniform quiet stretch.
    """
    trail = 0
    for v in reversed(samples):
        if v != 0:
            break
        trail += 1
    return trail / float(rate)


def _assert_music_underneath(final_wav: Path, music_path: Path,
                             total: float) -> None:
    """Fail loudly if the mix lost the music bed partway through.

    The mix ends at exactly `total` because a terminal ``apad`` forces it, which
    also hides the failure this catches: if anything upstream emits less than it
    should - a filter flushing early, a truncated input - the shortfall is
    padded with digital silence and the file still reports the right duration,
    so every length check downstream passes. The result is a recap that plays
    normally for most of its runtime and then goes completely silent, which is
    exactly what happened on real media and could not be reproduced afterwards
    in ten attempts from identical inputs.

    So rather than leave it to chance, measure the tail of what was actually
    written and refuse to ship a video with a dead ending. A music bed that
    fades out is fine; one that is *bit-exact zero* for a noticeable stretch is
    not, and only the written file can tell the two apart.
    """
    # Seek from the start rather than -ss, and discard the leading part. A
    # -ss past the end of a short file returns no samples at all, which would
    # make the guard pass silently on exactly the files it exists to catch -
    # the negative control caught that.
    tail_seconds = 4.0
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(final_wav),
             "-af", f"atrim=start={max(0.0, total - tail_seconds):.4f}",
             "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except OSError as exc:
        log(f"could not verify the audio tail: {exc}", level="warn")
        return
    if not raw:
        log("could not verify the audio tail; the mix is shorter than "
            f"{total - tail_seconds:.1f}s", level="warn")
        return
    samples = np.frombuffer(raw, np.int16)
    if samples.size == 0:
        log("could not verify the audio tail: no samples returned",
            level="warn")
        return
    # Measure the *contiguous* run of bit-exact zeros at the very end, not the
    # fraction of zeros across the window: a 3s hole inside a 4s window reads as
    # only 75% zero, which diluted the signal and let a real fault through.
    trailing_s = trailing_silence_s(samples, rate=8000)

    # A cutoff or padding bug produces a *long contiguous run* of exact zeros
    # at the end. Natural fades taper but rarely produce a contiguous block of
    # bit-exact zeros longer than ~0.3s at the very end of a 4s tail.
    # 1.5s is the line between "a fade that happens to land exactly on zero"
    # and "a bed that died". Calibrated against controls: 1s, 1.5s, 2s and 4s
    # natural fades all pass; a 1s dead tail passes; 2s, 3s, 4s and 9.3s dead
    # tails are all refused. The observed real-world fault was 9.28s.
    if trailing_s >= DEAD_TAIL_S:
        raise ToolError(
            f"the finished mix has a contiguous block of digital silence of "
            f"{trailing_s:.2f}s at the end of the last {tail_seconds:.1f}s.\n"
            f"  The music bed ({music_path.name}) likely stopped early and was "
            f"padded with silence (duration {total:.1f}s still looks correct).\n"
            f"  Re-run with --keep-temp to inspect {final_wav}."
        )
    if trailing_s >= 0.3:
        log(f"the last {tail_seconds:.0f}s of the mix ends with "
            f"{trailing_s:.2f}s of contiguous digital silence", level="warn")


# ================================================================ assembling

def concat_segments(paths: list[Path], out_mp4: Path, work: Path) -> None:
    """Concat demuxer: exact, memory-flat, scales to hundreds of clips."""
    list_file = work / "video_concat.txt"
    list_file.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in paths), encoding="utf-8"
    )
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(list_file),
         "-c", "copy", "-movflags", "+faststart", str(out_mp4)], timeout=1800)


# ================================================================== driver

@dataclass
class RenderResult:
    video: Path
    duration: float
    width: int
    height: int
    fps: int
    segments: int


def _load_source(entry: Entry, cfg: Pipeline, target_long: int
                 ) -> Image.Image | None:
    from .analysis import load_image_at
    from .quality import load_image
    if entry.item is None:
        return None
    info = entry.item.info
    if info.kind == "photo":
        return load_image(info.path, max_long_side=0,
                          orientation=info.exif_orientation)
    mid = entry.item.start + (entry.item.end - entry.item.start) / 2.0
    return load_image_at(info, mid) or None


def prepare_render(cfg: Pipeline) -> None:
    global FPS, ZOOM_W, ZOOM_H
    FPS = cfg.render.fps
    ZOOM_W = cfg.render.width
    ZOOM_H = cfg.render.height


def _segment_peak_gb(width: int, height: int) -> float:
    """Peak GB for one segment at this frame size, from the measured table."""
    return SEGMENT_PEAK_GB.get(max(int(width), int(height)),
                               max(SEGMENT_PEAK_GB.values()))


def render(cut: CutList, cfg: Pipeline, music_path: Path | None,
           work: Path, out_path: Path, *, keep_temp: bool = False,
           reference: bool = False) -> RenderResult:
    prepare_render(cfg)
    ensure_dir(work)
    seg_dir = ensure_dir(work / "segments")
    still_dir = ensure_dir(work / "stills")

    entries = cut.entries
    # Stills are prepared with room for the Ken Burns move *and* for the
    # supersampling zoompan samples from. Without the second factor the
    # supersample buys nothing: zoompan would be handed a 1080p-sized image and
    # asked for a 4K one, which just interpolates.
    #
    # The factor is chosen from what the photos actually contain rather than
    # applied blindly, because overshooting it is not free. A 1080p film needs
    # 2x (measured: 5.402 -> 5.688 edge energy, with 4x adding nothing). But a
    # 4K film already asks for 3840px from a 4032px photo, so there is nothing
    # left to sample and the supersample collapses to 1x - otherwise a 4K
    # render would pay 4x the filter cost to interpolate pixels that were never
    # in the file. That is the same self-limiting rule as the size cap below,
    # applied to the factor instead of the pixels.
    biggest = 0
    for e in entries:
        if e.item is not None and e.item.info.kind == "photo":
            long_side = max(e.item.info.width, e.item.info.height)
            if long_side:
                biggest = max(biggest, long_side)
    cap = int(cfg.render.still_max_long_side or 0)
    if cap:
        biggest = min(biggest, cap) if biggest else cap

    one_x = ZOOM_W * HEADROOM
    global ZOOM_SS
    # The factor is set by whether the source can cover a frame larger than the
    # one being delivered. zoompan samples nearest-neighbour at whatever size it
    # is asked for, so the win comes from asking for *more* pixels than the
    # delivery frame and averaging them down afterwards - not from having
    # headroom over the headroom-inflated prepared size. At 1080p a 4032px photo
    # covers 2x (measured 5.402 -> 5.688); at 4K it would need 7680px and cannot,
    # so the factor drops to 1 and the render stops paying for interpolation.
    ZOOM_SS = 1 if not biggest else max(
        1, min(ZOOM_SS_CAP, int(biggest // max(1, ZOOM_W))))

    want = min(one_x * ZOOM_SS, biggest) if biggest else one_x * ZOOM_SS
    sw = int(round(want / 2) * 2)
    sh = int(round((sw * ZOOM_H / max(1, ZOOM_W)) / 2) * 2)
    log(f"stills prepared at {sw}x{sh} for {ZOOM_W}x{ZOOM_H} delivery"
        + (f" ({ZOOM_SS}x supersampled from "
           f"{biggest or 0}px sources)" if ZOOM_SS > 1 else
           " (no supersample: sources are already at delivery size)"))

    total = cut.duration
    log(f"rendering {len(entries)} segments -> {human_duration(total)} "
        f"@ {ZOOM_W}x{ZOOM_H}")

    # ---- stills / title cards
    log("preparing still frames and title cards")
    frame_for: dict[int, Path] = {}
    for i, e in enumerate(entries):
        # PNG, not JPEG, for the intermediate. Two reasons, both learned the
        # hard way: a JPEG has no colour-range concept, so ffmpeg decodes it
        # as full-range and the segment comes out yuvj420p - which the concat
        # demuxer then propagates to the whole film, but only when a photo
        # happens to be the first segment, so it hid behind the default
        # chronological order. And Ken Burns zooms into the intermediate, so a
        # lossy one visibly softens the very frames that are supposed to feel
        # crisp. Disk is cheap; .work is deleted after every run anyway.
        target = still_dir / f"still_{i:05d}.png"
        if e.is_title:
            bg = _load_source(e, cfg, sw)
            if bg is not None:
                bg = bg.resize((min(bg.width, 2400), min(bg.height, 2400)),
                               Image.LANCZOS) if max(bg.size) > 2400 else bg
            title_card(e, cfg, bg, target)
            frame_for[i] = target
        else:
            # A Live Photo plays its own motion, so no still frame is needed.
            # Skipping the decode also skips a 4032x3024 HEIC load and PNG write
            # per shot, which on a library where three quarters of the photos
            # are Live Photos is most of the preparation phase.
            if e.item is not None and e.item.info.is_live:
                progress("preparing frames", i + 1, len(entries))
                continue
            src = _load_source(e, cfg, sw)
            if src is None:
                raise ToolError(f"could not load image for {e.label}")
            # Decided from the pixels just loaded, not from metadata: ffprobe
            # reports a HEIC's embedded thumbnail, so a 4032x3024 photo arrives
            # claiming to be 512x512 and would be treated as portrait.
            if e.pair is not None:
                other = _load_source(Entry(item=e.pair, duration=e.duration),
                                     cfg, sw)
                if other is not None:
                    canvas = compose_pair(src, other, sw, sh)
                    canvas.save(target)
                    frame_for[i] = target
                    continue
            canvas = compose(src, sw, sh,
                             fit_for(src.width / max(1, src.height), cfg))
            canvas.save(target)
            frame_for[i] = target
        progress("preparing frames", i + 1, len(entries))
    progress_done("preparing frames")

    # ---- segments
    #
    # Memory first: each worker is a full ffmpeg holding a frame-sized buffer
    # per in-flight frame, and at 4K that is several GB apiece. Sizing this off
    # CPU count alone is what made a 4K render crash a 16GB machine.
    by_cpu = max(1, min(8, (os.cpu_count() or 4) - 1))
    jobs = cfg.jobs or default_jobs(cfg.render.width, cfg.render.height)
    if cfg.jobs:
        log(f"using {cfg.jobs} parallel worker(s) as requested")
    elif jobs < by_cpu:
        log(f"using {jobs} parallel worker(s) rather than {by_cpu}: a "
            f"{cfg.render.width}x{cfg.render.height} segment needs about "
            f"{_segment_peak_gb(cfg.render.width, cfg.render.height):.1f}GB "
            f"while one is being rendered, and {jobs} of them fit in memory "
            f"with room to spare. Pass --jobs to override.")
    seg_paths: list[Path | None] = [None] * len(entries)

    def build(idx: int) -> tuple[int, Path]:
        e = entries[idx]
        out = seg_dir / f"seg_{idx:05d}.mp4"
        if e.is_title:
            render_still(frame_for[idx], out, cfg, e.duration, e.motion, e.zoom)
        elif e.item.kind == "photo":
            if e.item.info.is_live:
                render_live(e, cfg, out, sw=sw, sh=sh)
            else:
                render_still(frame_for[idx], out, cfg, e.duration, e.motion,
                             e.zoom)
        else:
            render_video(e, cfg, out)
        return idx, out

    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(build, i) for i in range(len(entries))]
        for fut in as_completed(futures):
            idx, path = fut.result()
            seg_paths[idx] = path
            done += 1
            progress("rendering segments", done, len(entries))
        progress_done("rendering segments")

    good = [p for p in seg_paths if p is not None]
    if len(good) != len(entries):
        raise ToolError("some segments failed to render")

    # ---- picture lock
    silent = work / "picture.mp4"
    concat_segments(good, silent, work)

    # ---- optional lossless reference, for the verifier to measure against.
    #
    # The encode being measured is already the one thing every check in
    # verify_output.py cannot currently see: all of them test structure - lengths,
    # tags, ducking, rotation - and none of them can tell a clean encode from a
    # smeared one. Re-running the *same* segment encode at `-qp 0` gives a
    # bit-exact rendering of the same filter chain, so VMAF against it isolates
    # encode loss from everything upstream.
    #
    # It re-encodes the assembled picture rather than re-rendering segments, which
    # keeps it to one extra pass instead of redoing the whole timeline. That is
    # deliberate and it is also the limit of what this measures: it scores the
    # delivery encode, not the resampling and Ken Burns that came before it.
    if reference:
        log("encoding a lossless reference for the verifier")
        ref = work / "reference.mp4"
        run(["ffmpeg", "-y", "-v", "error", "-i", str(silent),
             "-c:v", "libx264", "-qp", "0", "-preset", "veryslow",
             "-pix_fmt", "yuv420p",
             "-x264-params", "colorprim=bt709:transfer=bt709:"
                             "colormatrix=bt709:range=tv",
             str(ref)], timeout=3600)

    # ---- sound
    audio = None
    if music_path and music_path.exists():
        audio_dir = ensure_dir(work / "audio")
        order = []
        for i, e in enumerate(entries):
            wav = audio_dir / f"aud_{i:05d}.wav"
            order.append(wav)
        log("extracting and normalising audio")
        aw, _ = _extract_all(entries, cfg, order, jobs)
        log("mixing music under native audio")
        audio = build_audio_track(aw, cfg, music_path, total, work,
                                  music=cut.music)

    # ---- mux
    log("muxing final file")
    # The colour tags belong here as well as on every segment. This is a
    # `-c copy` concat, so the picture is never re-read: the container inherits
    # whatever the first segment declared, and if segments disagree the result is
    # worse than no tag at all. Declaring the truth at the point of assembly is
    # what makes it true for the whole file.
    if audio:
        run(["ffmpeg", "-y", "-v", "error",
             "-i", str(silent), "-i", str(audio),
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac",
             "-b:a", cfg.render.audio_bitrate, "-ar", str(AUDIO_RATE),
             "-movflags", "+faststart",
             *COLOR_TAGS,
             str(out_path)], timeout=1800)
    else:
        run(["ffmpeg", "-y", "-v", "error", "-i", str(silent),
             "-c", "copy", "-movflags", "+faststart",
             *COLOR_TAGS,
             str(out_path)],
            timeout=1800)

    if reference:
        measured = measure_true_peak(out_path)
        if measured is None:
            log("could not measure true peak on the finished file", level="warn")
        elif measured > -0.8:
            # Measured, not assumed: alimiter is a sample-peak limiter and AAC
            # adds inter-sample peaks on top, so this was never guaranteed by the
            # graph. -0.8 is the verifier's own ceiling, so warning here means
            # the verifier will fail - which is the correct outcome, but it is
            # better to say so at the point of the cause.
            log(f"true peak measured at {measured:.2f} dBTP, over the -0.8 the "
                f"verifier allows; the mix is louder than the graph intends",
                level="warn")

    final_duration = _probe_duration(out_path)
    result = RenderResult(video=out_path, duration=final_duration,
                          width=ZOOM_W, height=ZOOM_H, fps=FPS,
                          segments=len(entries))

    if not keep_temp:
        shutil.rmtree(work, ignore_errors=True)
    return result


def _extract_all(entries: list[Entry], cfg: Pipeline, order: list[Path],
                 jobs: int) -> tuple[list[Path], list[float]]:
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(extract_segment_audio, e, cfg, order[i]): i
            for i, e in enumerate(entries)
        }
        done = 0
        for fut in as_completed(futures):
            fut.result()
            done += 1
            progress("extracting audio", done, len(futures))
        progress_done("extracting audio")
    return order, []


def _probe_duration(path: Path) -> float:
    proc = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(path)], check=False)
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0
