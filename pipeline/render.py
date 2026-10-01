"""Stage 7 - render.

Strategy: every timeline entry becomes a small, uniformly-encoded intermediate
clip, then they are concatenated and mixed. Chaining hundreds of inputs into
one giant filter graph is what makes naive ffmpeg slideshow scripts eat 50 GB
of RAM; this stays flat and scales to full-length timelines.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .config import Pipeline
from .music import MusicResult
from .select import CutList, Entry
from .util import ToolError, default_jobs, ensure_dir, human_duration, log, run

AUDIO_RATE = 48000
# Extra resolution fed to zoompan so the pan/zoom does not soften the image.
HEADROOM = 1.20


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

    return f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:s={ZOOM_W}x{ZOOM_H}:fps={FPS}"


# Per-render globals set by prepare_render(); kept module-level so the ffmpeg
# filter strings above stay readable.
FPS = 30
ZOOM_W = 1920
ZOOM_H = 1080


def render_still(png: Path, out_mp4: Path, cfg: Pipeline, duration: float,
                 motion: str) -> None:
    """Animate one still into a clip of the requested length."""
    frames = max(2, int(round(duration * FPS)))
    if cfg.render.ken_burns:
        chain = _zoompan_expr(motion, cfg.render.zoom_amount, frames)
    else:
        chain = (f"scale={ZOOM_W}:{ZOOM_H}:force_original_aspect_ratio=increase,"
                 f"crop={ZOOM_W}:{ZOOM_H},setsar=1")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-loop", "1", "-framerate", str(FPS), "-i", str(png),
        "-vf", f"{chain},format=yuv420p,setrange=limited",
        "-frames:v", str(frames),
        "-an",
        "-c:v", "libx264", "-crf", str(cfg.render.crf),
        "-preset", cfg.render.preset,
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-profile:v", "high", "-level", "4.0",
        "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
        "-video_track_timescale", "90000",
        str(out_mp4),
    ]
    run(cmd, timeout=900)


def _video_filter(width: int, height: int, fit: str) -> str:
    """Scale one source clip to the working frame size.

    Always returns a fully labelled graph ending in ``[v]``. Crop and pad look
    like a single unlabelled filter, which is fine for ``-vf`` but not for
    ``-filter_complex``: without a label ffmpeg expects an unlabelled input
    stream to feed the chain's input pad, and since the caller maps the source
    stream directly there is nothing left to bind, so the render fails.
    """
    if fit == "crop":
        return (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setsar=1,fps={FPS},format=yuv420p,setrange=limited[v]")
    if fit == "pad":
        return (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1,fps={FPS},format=yuv420p,setrange=limited[v]")
    # blur
    return (
        f"[0:v]split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},boxblur=18:2,eq=brightness=-0.10[bg];"
        f"[fgsrc]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={FPS},format=yuv420p,setrange=limited[v]"
    )


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
        "-filter_complex", _video_filter(width, height, fit),
        "-map", "[v]",
        "-frames:v", str(frames),
        "-c:v", "libx264", "-crf", str(cfg.render.crf),
        "-preset", cfg.render.preset,
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-profile:v", "high", "-level", "4.0",
        "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
        "-video_track_timescale", "90000",
        str(out_mp4),
    ]
    run(cmd, timeout=900)


# ==================================================================== audio

def extract_segment_audio(entry: Entry, cfg: Pipeline, out_wav: Path) -> None:
    """Pull the segment's native audio, level-matched. Silent for stills."""
    dur = max(0.08, entry.duration)
    item = entry.item

    fade = f"afade=t=in:st=0:d=0.06,areverse,afade=t=in:st=0:d=0.10,areverse"
    chain = f"loudnorm=I=-18:TP=-1.5:LRA=11,{fade},aformat=sample_fmts=s16:sample_rates={AUDIO_RATE}:channel_layouts=stereo"

    if item is None or item.kind != "video":
        run(["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i",
             f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
             "-t", f"{dur:.3f}", "-c:a", "pcm_s16le", str(out_wav)])
        return

    if not item.info.has_audio:
        run(["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i",
             f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
             "-t", f"{dur:.3f}", "-c:a", "pcm_s16le", str(out_wav)])
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
    final_wav = work / "final_audio.wav"

    list_file = work / "audio_concat.txt"
    list_file.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in entries), encoding="utf-8"
    )
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c:a", "pcm_s16le", str(content_wav)])

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
        mix = (
            f"{music_filter};[0:a]{content_fmt};"
            f"[mus][cont]sidechaincompress=threshold=0.035:ratio=6:"
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
        str(final_wav),
    ], timeout=1800)
    _assert_music_underneath(final_wav, music_path, total)
    return final_wav


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
    trail = 0
    for v in reversed(samples):
        if v == 0:
            trail += 1
        else:
            break
    trailing_s = trail / 8000.0  # 8kHz

    # A cutoff or padding bug produces a *long contiguous run* of exact zeros
    # at the end. Natural fades taper but rarely produce a contiguous block of
    # bit-exact zeros longer than ~0.3s at the very end of a 4s tail.
    # 1.5s is the line between "a fade that happens to land exactly on zero"
    # and "a bed that died". Calibrated against controls: 1s, 1.5s, 2s and 4s
    # natural fades all pass; a 1s dead tail passes; 2s, 3s, 4s and 9.3s dead
    # tails are all refused. The observed real-world fault was 9.28s.
    if trailing_s >= 1.5:  # contiguous zero run of >= 1.5s at the tail end
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


def render(cut: CutList, cfg: Pipeline, music_path: Path | None,
           work: Path, out_path: Path, *, keep_temp: bool = False) -> RenderResult:
    prepare_render(cfg)
    ensure_dir(work)
    seg_dir = ensure_dir(work / "segments")
    still_dir = ensure_dir(work / "stills")

    entries = cut.entries
    sw = int(round(ZOOM_W * HEADROOM / 2) * 2)
    sh = int(round(ZOOM_H * HEADROOM / 2) * 2)

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
            src = _load_source(e, cfg, sw)
            if src is None:
                raise ToolError(f"could not load image for {e.label}")
            # Decided from the pixels just loaded, not from metadata: ffprobe
            # reports a HEIC's embedded thumbnail, so a 4032x3024 photo arrives
            # claiming to be 512x512 and would be treated as portrait.
            canvas = compose(src, sw, sh,
                             fit_for(src.width / max(1, src.height), cfg))
            canvas.save(target)
            frame_for[i] = target

    # ---- segments
    jobs = cfg.jobs or default_jobs()
    seg_paths: list[Path | None] = [None] * len(entries)

    def build(idx: int) -> tuple[int, Path]:
        e = entries[idx]
        out = seg_dir / f"seg_{idx:05d}.mp4"
        if e.is_title:
            render_still(frame_for[idx], out, cfg, e.duration, e.motion)
        elif e.item.kind == "photo":
            render_still(frame_for[idx], out, cfg, e.duration, e.motion)
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
            if done % 5 == 0 or done == len(entries):
                log(f"  segments {done}/{len(entries)}")

    good = [p for p in seg_paths if p is not None]
    if len(good) != len(entries):
        raise ToolError("some segments failed to render")

    # ---- picture lock
    silent = work / "picture.mp4"
    concat_segments(good, silent, work)

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
    if audio:
        run(["ffmpeg", "-y", "-v", "error",
             "-i", str(silent), "-i", str(audio),
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac",
             "-b:a", cfg.render.audio_bitrate, "-ar", str(AUDIO_RATE),
             "-movflags", "+faststart",
             str(out_path)], timeout=1800)
    else:
        run(["ffmpeg", "-y", "-v", "error", "-i", str(silent),
             "-c", "copy", "-movflags", "+faststart", str(out_path)],
            timeout=1800)

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
        for fut in as_completed(futures):
            fut.result()
    return order, []


def _probe_duration(path: Path) -> float:
    proc = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(path)], check=False)
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0
