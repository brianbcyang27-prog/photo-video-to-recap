"""Stage 2 - analysis.

Photos get scored whole. Videos get decoded into sampled frames, split into
shots at detected cuts, and each shot is scored so we can keep only the good
parts of a long clip.
"""
from __future__ import annotations

import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .cache import (
    AnalysisCache,
    config_fingerprint,
    file_key,
    item_to_record,
    record_to_item,
)
from .config import Analysis
from .ingest import Library
from .quality import FrameMetrics, dhash, load_image, measure
from .util import (
    MediaInfo,
    ToolError,
    default_jobs,
    log,
    progress,
    progress_done,
)

# Calibration constants, derived from typical phone-camera output measured
# at the shared analysis resolution (see Analysis.analysis_long_side).
SHARP_SAT = 2000.0     # Laplacian variance treated as "fully sharp"
MOTION_GOOD = 12.0     # mean abs grey delta that reads as pleasant movement
MOTION_SHAKE = 45.0    # above this we assume camera shake, not subject motion


def _norm_sharpness(value: float) -> float:
    if value <= 0:
        return 0.0
    return float(np.clip(math.log1p(value) / math.log1p(SHARP_SAT), 0.0, 1.0))


def _norm_motion(value: float) -> float:
    """Reward moderate movement; punish handheld shake."""
    if value <= 0:
        return 0.0
    if value <= MOTION_GOOD:
        return float(np.clip(value / MOTION_GOOD, 0.0, 1.0))
    over = (value - MOTION_GOOD) / (MOTION_SHAKE - MOTION_GOOD)
    return float(np.clip(1.0 - over, 0.0, 1.0))


def _norm_faces(m: FrameMetrics) -> float:
    if m.faces <= 0:
        return 0.0
    return float(np.clip(0.60 + 0.12 * min(m.faces, 3) + 0.28 * m.eyes, 0.0, 1.0))


@dataclass
class Item:
    """One selected thing to put on screen: a whole photo, or a slice of video."""

    info: MediaInfo
    kind: str                     # "photo" | "video"
    start: float = 0.0            # seconds into source (video only)
    end: float = 0.0
    score: float = 0.0
    metrics: FrameMetrics = field(default_factory=FrameMetrics)
    reject: str | None = None
    hash: int = 0
    # Provenance so the report can explain the decision.
    note: str = ""

    @property
    def duration(self) -> float:
        if self.kind == "video":
            return max(0.05, self.end - self.start)
        return 0.0

    @property
    def key(self) -> str:
        if self.kind == "photo":
            return str(self.info.path)
        return f"{self.info.path}@{self.start:.2f}-{self.end:.2f}"

    @property
    def label(self) -> str:
        return self.info.path.name


# ----------------------------------------------------------------- stills

def analyse_photo(info: MediaInfo, cfg: Analysis) -> Item:
    item = Item(info=info, kind="photo")
    img = load_image(info.path, max_long_side=cfg.analysis_long_side,
                     orientation=info.exif_orientation)
    if img is None:
        item.reject = "could not decode image"
        return item
    rgb = np.asarray(img)
    if rgb.size == 0:
        item.reject = "empty image"
        return item
    m = measure(rgb, want_hash=True, hash_image=img)
    item.metrics = m
    item.hash = dhash(img)
    item.reject = m.hard_fail(cfg.min_sharpness, cfg.max_clipped, cfg.max_black)
    item.score = composite(m, cfg, is_video=False)
    return item


# ----------------------------------------------------------------- video

def _analysis_geometry(info: MediaInfo, long_side: int) -> tuple[int, int]:
    """Output dims for the sampled-frame pipe, long side pinned to long_side.

    Must mirror exactly what ffmpeg's scale filter will produce, since we read
    the raw stream back without any per-frame headers.
    """
    w, h = info.display_size
    if w <= 0 or h <= 0:
        return (2, 2)
    if w >= h:
        ow = long_side
        oh = max(2, round(long_side * h / w))
    else:
        oh = long_side
        ow = max(2, round(long_side * w / h))
    ow = max(2, ow - (ow % 2))
    oh = max(2, oh - (oh % 2))
    return (ow, oh)


def _sample_frames(info: MediaInfo, cfg: Analysis):
    """Decode evenly-spaced RGB frames. Yields (time_seconds, ndarray)."""
    width, height = _analysis_geometry(info, cfg.analysis_long_side)
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(info.path),
        "-vf", f"fps={cfg.video_sample_fps},scale={width}:{height}:flags=bicubic",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    step = 1.0 / cfg.video_sample_fps
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            yield idx * step, np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            idx += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
        if proc.stderr:
            proc.stderr.close()


def _find_cuts(motion: list[float], cfg: Analysis) -> list[int]:
    """Indices where a new shot begins, using an adaptive spike threshold."""
    n = len(motion)
    if n < 3:
        return []
    arr = np.asarray(motion, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) or 0.0
    # Robust threshold: median + k*MAD, with a floor and a relative ceiling.
    thr = max(cfg.scene_threshold * 255.0 * 0.35, med + 6.0 * (1.4826 * mad))
    ceiling = float(np.percentile(arr, 99.0))
    thr = min(thr, max(thr, ceiling * cfg.scene_threshold * 2.2))

    cuts = [0]
    min_gap = max(1, int(cfg.min_shot_seconds / (1.0 / cfg.video_sample_fps)))
    last = 0
    for i in range(1, n):
        if arr[i] >= thr and (i - last) >= min_gap:
            cuts.append(i)
            last = i
    return cuts


def _split_long(shots: list[tuple[int, int]], total: int, cfg: Analysis) -> list[tuple[int, int]]:
    """Cap shot length so one long clip cannot dominate the timeline."""
    max_frames = max(1, int(cfg.max_shot_seconds * cfg.video_sample_fps))
    out: list[tuple[int, int]] = []
    for a, b in shots:
        if b - a <= max_frames:
            out.append((a, b))
            continue
        pieces = int(math.ceil((b - a) / max_frames))
        span = (b - a) / pieces
        for k in range(pieces):
            s = a + int(round(k * span))
            e = a + int(round((k + 1) * span))
            if e - s >= 1:
                out.append((s, e))
    return out


def analyse_video(info: MediaInfo, cfg: Analysis) -> list[Item]:
    """Split a clip into shots and score each one."""
    items: list[Item] = []
    try:
        sampled = list(_sample_frames(info, cfg))
    except (ToolError, OSError, ValueError) as exc:
        log(f"  ! {info.path.name}: decode failed ({exc})", level="warn")
        return [Item(info=info, kind="video", reject=f"decode failed: {exc}")]

    if len(sampled) < 2:
        return [Item(info=info, kind="video", reject="too short to analyse")]

    step = 1.0 / cfg.video_sample_fps
    gray_prev: np.ndarray | None = None
    metrics: list[FrameMetrics] = []
    motions: list[float] = []
    for t, rgb in sampled:
        m = measure(rgb, prev_gray=gray_prev, want_faces=True)
        if t == 0.0:
            m.motion = 0.0
        metrics.append(m)
        motions.append(m.motion)
        gray_prev = np.asarray(
            rgb.astype(np.float32) * np.array([0.299, 0.587, 0.114], np.float32)
        ).sum(axis=2)

    cuts = _find_cuts(motions, cfg)
    shots = list(zip(cuts, cuts[1:] + [len(metrics)]))
    shots = _split_long(shots, len(metrics), cfg)

    duration = info.duration or (len(metrics) * step)

    for a, b in shots:
        if b <= a:
            continue
        chunk = metrics[a:b]
        start = a * step
        end = min(duration, b * step)
        if end - start < cfg.min_shot_seconds:
            continue

        # Average the per-frame metrics over the shot.
        avg = FrameMetrics(
            sharpness=float(np.mean([m.sharpness for m in chunk])),
            tenengrad=float(np.mean([m.tenengrad for m in chunk])),
            exposure=float(np.mean([m.exposure for m in chunk])),
            clipped=float(np.max([m.clipped for m in chunk])),
            black=float(np.max([m.black for m in chunk])),
            contrast=float(np.mean([m.contrast for m in chunk])),
            colour=float(np.mean([m.colour for m in chunk])),
            faces=int(round(np.mean([m.faces for m in chunk]))),
            eyes=float(np.mean([m.eyes for m in chunk])),
            motion=float(np.mean([m.motion for m in chunk])),
            luma=float(np.mean([m.luma for m in chunk])),
        )
        item = Item(info=info, kind="video", start=start, end=end, metrics=avg)
        item.reject = avg.hard_fail(cfg.min_sharpness, cfg.max_clipped, cfg.max_black)
        item.score = composite(avg, cfg, is_video=True)

        # Hash the mid-shot frame so video stills dedupe against photos.
        try:
            mid = np.asarray(
                load_image_at(info, start + (end - start) / 2.0) or Image.new("RGB", (1, 1))
            )
            item.hash = dhash(Image.fromarray(mid))
        except Exception:
            item.hash = 0

        total_shots = len(shots)
        if total_shots > 1:
            item.note = f"shot {shots.index((a, b)) + 1}/{total_shots} of this clip"
        items.append(item)

    return items or [Item(info=info, kind="video", reject="no usable shots")]


def load_image_at(info: MediaInfo, t: float) -> Image.Image | None:
    """Grab a single full-quality frame from a video at time t."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(info.path),
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    import io
    try:
        img = Image.open(io.BytesIO(proc.stdout))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


# ----------------------------------------------------------------- scoring

def composite(m: FrameMetrics, cfg: Analysis, *, is_video: bool) -> float:
    """Weighted blend of the normalised metrics, returned in 0..1."""
    sharp = _norm_sharpness(m.sharpness)
    face = _norm_faces(m)
    colour = float(np.clip(m.colour, 0.0, 1.0))
    contrast = float(np.clip(m.contrast, 0.0, 1.0))
    motion = _norm_motion(m.motion) if is_video else 0.0

    parts: list[tuple[float, float]] = [
        (cfg.w_sharpness, sharp),
        (cfg.w_exposure, m.exposure),
        (cfg.w_faces, face),
        (cfg.w_color, colour),
        (cfg.w_contrast, contrast),
    ]
    if is_video:
        parts.append((cfg.w_motion, motion))

    total_w = sum(w for w, _ in parts) or 1.0
    return float(sum(w * v for w, v in parts) / total_w)


# ------------------------------------------------------------------ driver

def analyse_library(lib: Library, cfg: Analysis,
                    cache_dir: Path | None = None) -> list[Item]:
    """Score everything. Returns photos plus every surviving video shot.

    With a cache_dir, results are reused across runs. Decoding every file is
    the dominant cost on a real library (~50 min for 12,193 files) and depends
    only on the bytes and the config, so re-running to change the *edit* should
    not pay for it twice. See cache.py for how the key is built to avoid
    returning stale results that look fresh.
    """
    items: list[Item] = []
    jobs = default_jobs()

    cache = None
    if cache_dir is not None:
        cache = AnalysisCache(cache_dir / "analysis.jsonl", config_fingerprint(cfg))

    def cached_photo(info) -> Item | None:
        if cache is None:
            return None
        recs = cache.get(file_key(info, cache.cfg_fp))
        if recs is None:
            return None
        rebuilt = [record_to_item(r, info) for r in recs]
        if rebuilt:
            return rebuilt[0]
        return None

    def cached_video(info) -> list[Item] | None:
        if cache is None:
            return None
        recs = cache.get(file_key(info, cache.cfg_fp))
        if recs is None:
            return None
        return [record_to_item(r, info) for r in recs]

    def run_photos(pool_items):
        """Photos: one Item each, some may be cached."""
        hits = 0
        out: list[Item] = []
        todo = []
        for info in pool_items:
            hit = cached_photo(info)
            if hit is not None:
                out.append(hit)
                hits += 1
            else:
                todo.append(info)
        if hits:
            log(f"  {hits} photo scores reused from cache, "
                f"{len(todo)} to analyse")
        if todo:
            done = 0
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(analyse_photo, info, cfg): info
                           for info in todo}
                for fut in as_completed(futures):
                    it = fut.result()
                    out.append(it)
                    if cache is not None:
                        cache.put(file_key(futures[fut], cache.cfg_fp),
                                  [item_to_record(it)])
                    done += 1
                    progress("scoring photos", done, len(todo))
            progress_done("scoring photos")
        return out

    def run_videos(pool_items):
        """Videos: several Items per clip, all cached together."""
        hits = 0
        out: list[Item] = []
        todo = []
        for info in pool_items:
            got = cached_video(info)
            if got is not None:
                out.extend(got)
                hits += 1
            else:
                todo.append(info)
        if hits:
            log(f"  {hits} clip analyses reused from cache, "
                f"{len(todo)} to analyse")
        if todo:
            done = 0
            # Parallel for the same reason photos are: each clip is an independent
            # ffmpeg decode in a subprocess, so the GIL is released for nearly the
            # whole of it and threads scale. A real trip library holds thousands of
            # clips, and doing them one at a time put a 4,750-clip library at ~50
            # minutes of decoding before selection even started.
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(analyse_video, info, cfg): info
                           for info in todo}
                for fut in as_completed(futures):
                    got = fut.result()
                    out.extend(got)
                    if cache is not None:
                        cache.put(file_key(futures[fut], cache.cfg_fp),
                                  [item_to_record(i) for i in got])
                    done += 1
                    progress("scoring videos", done, len(todo))
            progress_done("scoring videos")
        return out

    if lib.photos:
        log(f"scoring {len(lib.photos)} photos with {jobs} workers")
        items.extend(run_photos(lib.photos))

    if lib.videos:
        log(f"analysing {len(lib.videos)} videos with {jobs} workers")
        items.extend(run_videos(lib.videos))

    if cache is not None:
        cache.flush()

    for it in items:
        if it.kind == "photo":
            it.score *= cfg.photo_bias

    items.sort(key=lambda i: i.score, reverse=True)
    good = [i for i in items if not i.reject]
    rejected = [i for i in items if i.reject]

    log(f"analysis: {len(good)} usable, {len(rejected)} rejected")
    if rejected:
        from collections import Counter
        why = Counter(r.reject.split(" (")[0] for r in rejected)
        for reason, count in why.most_common(5):
            log(f"  rejected {count}x {reason}")
    return items
