"""Per-frame quality metrics.

Shared by stills and by sampled video frames so photos and video compete on
the same scale. Everything is pure numpy/scipy; cv2 is used only for faces
and is optional.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ------------------------------------------------------------------ optional

try:
    import cv2  # type: ignore
    HAVE_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    HAVE_CV2 = False

Image.MAX_IMAGE_PIXELS = 400_000_000


def _cascade(name: str):
    """Load a Haar cascade, or None. Resilient across cv2 builds."""
    if not HAVE_CV2:
        return None
    try:
        path = cv2.data.haarcascades + name
        c = cv2.CascadeClassifier(path)
        return None if c.empty() else c
    except Exception:
        return None


_FACE = None
_FACE_INIT = False


def _face_cascade():
    global _FACE, _FACE_INIT
    if not _FACE_INIT:
        _FACE = _cascade("haarcascade_frontalface_default.xml")
        _FACE_INIT = True
    return _FACE


# ------------------------------------------------------------------- loading

def _load_via_sips(path: Path) -> Image.Image | None:
    """macOS native converter. Handles HEIC/HEIF and other odd formats."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "conv.jpg"
        proc = subprocess.run(
            ["sips", "-s", "format", "jpeg", str(path), "--out", str(out)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
            try:
                return Image.open(out).convert("RGB")
            except Exception:
                return None
    return None


def _load_via_ffmpeg(path: Path) -> np.ndarray | None:
    """Last-resort decode: ask ffmpeg for a raw RGB frame."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        w, h = (int(x) for x in probe.stdout.strip().split("x")[:2])
    except (ValueError, IndexError):
        return None
    need = w * h * 3
    if len(proc.stdout) < need:
        return None
    return np.frombuffer(proc.stdout[:need], dtype=np.uint8).reshape(h, w, 3)


def apply_exif_orientation(img: Image.Image, o: int) -> Image.Image:
    """Turn stored pixels the way the EXIF Orientation tag says to display them.

    Orientation is the transform a viewer must apply to the stored image, on the
    standard 1-8 scale. Values 5-8 also transpose the axes, which is what makes
    a portrait shot legible as portrait rather than as a sideways landscape.
    """
    if o == 2:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if o == 3:
        return img.transpose(Image.ROTATE_180)
    if o == 4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if o == 5:
        return img.transpose(Image.TRANSPOSE)
    if o == 6:
        return img.rotate(-90, expand=True)
    if o == 7:
        return img.transpose(Image.TRANSVERSE)
    if o == 8:
        return img.rotate(90, expand=True)
    return img


def exif_orientation(path: Path) -> int:
    """Read the EXIF Orientation tag, cached per file+size+mtime.

    Only used when the caller has no MediaInfo to hand (the verifier, and any
    one-off inspection). Ingest already fetches this for every file in one
    batched exiftool call, so the hot paths pass it in instead of paying for a
    subprocess per image.
    """
    key = str(path)
    try:
        st = path.stat()
        key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        pass
    cached = _ORIENT_CACHE.get(key)
    if cached is not None:
        return cached
    value = 1
    try:
        out = subprocess.run(
            ["exiftool", "-s3", "-n", "-Orientation", str(path)],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        if out and out[0].strip().isdigit():
            v = int(out[0].strip())
            if 1 <= v <= 8:
                value = v
    except Exception:
        pass
    if len(_ORIENT_CACHE) > 20000:                       # keep the map bounded
        _ORIENT_CACHE.clear()
    _ORIENT_CACHE[key] = value
    return value


_ORIENT_CACHE: dict[str, int] = {}


def load_image(path: Path, max_long_side: int = 0,
               orientation: int | None = None) -> Image.Image | None:
    """Robust image load: PIL -> sips -> ffmpeg. EXIF orientation applied.

    The orientation has to be applied on whichever branch won, because the
    three loaders disagree about it: PIL only rotates if it can open the file at
    all, and it cannot open HEIC (6,087 of this library's 6,972 photos), so
    those photos arrived here already carrying a quarter-turn tag and were
    handed back lying on their side. ffmpeg autorotates on its own; sips, the
    fast native path, quietly ignores the tag. So the sips branch is the one
    that has to be corrected by hand.

    `orientation` is the EXIF Orientation tag (1-8) if the caller already knows
    it, which avoids a subprocess per image.
    """
    img: Image.Image | None = None
    needs_rotate = False
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        img = None

    if img is not None:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        img = img.convert("RGB")
    else:
        img = _load_via_sips(path)
        needs_rotate = img is not None          # sips ignores Orientation
        if img is None:
            arr = _load_via_ffmpeg(path)         # ffmpeg already autorotates
            if arr is not None:
                img = Image.fromarray(arr)

    if img is None:
        return None
    if needs_rotate:
        o = exif_orientation(path) if orientation is None else orientation
        img = apply_exif_orientation(img, o)
    if max_long_side:
        img = resize_long_side(img, max_long_side)
    return img


def resize_long_side(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= target or longest == 0:
        return img
    scale = target / longest
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


# ------------------------------------------------------------------- metrics

def _gray(rgb: np.ndarray) -> np.ndarray:
    # Rec.601 luma, uint8 out.
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return (0.299 * r + 0.587 * g + 0.114 * b)


def laplacian_var(gray: np.ndarray) -> float:
    """Variance of the Laplacian - the standard reference-free blur metric."""
    p = np.pad(gray, 1, mode="edge")
    lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
           - 4.0 * gray)
    return float(np.var(lap))


def tenengrad(gray: np.ndarray) -> float:
    """Mean Sobel gradient magnitude - complements the Laplacian."""
    p = np.pad(gray, 1, mode="edge")
    gx = (p[:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:]
          - p[:-2, :-2] - 2.0 * p[1:-1, :-2] - p[2:, :-2])
    gy = (p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]
          - p[:-2, :-2] - 2.0 * p[:-2, 1:-1] - p[:-2, 2:])
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def colourfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk colourfulness, squashed to 0..1."""
    arr = rgb.astype(np.float32)
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    std_root = float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2))
    mean_root = float(np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))
    return float(np.clip((std_root + 0.3 * mean_root) / 110.0, 0.0, 1.0))


def face_count(rgb: np.ndarray) -> int:
    if not HAVE_CV2:
        return 0
    cascade = _face_cascade()
    if cascade is None:
        return 0
    try:
        h, w = rgb.shape[:2]
        scale = 480.0 / max(1, min(h, w))
        if scale < 1.0:
            small = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = rgb
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = cascade.detectMultiScale(
            gray, scaleFactor=1.12, minNeighbors=5,
            minSize=(max(20, int(0.05 * min(gray.shape))),)*2,
        )
        return 0 if boxes is None else len(boxes)
    except Exception:
        return 0


def faces_are_eyes_open(rgb: np.ndarray) -> float:
    """Cheap closed-eye proxy: 1.0 = looks alert, 0.0 = eyes shut/looking away.

    Uses eye-region brightness contrast rather than a landmark model, so it
    has no extra dependency. Deliberately gentle - it nudges ranking, it does
    not reject photos on its own.
    """
    if not HAVE_CV2:
        return 0.5
    cascade = _face_cascade()
    if cascade is None:
        return 0.5
    try:
        h, w = rgb.shape[:2]
        scale = 480.0 / max(1, min(h, w))
        small = rgb if scale >= 1.0 else cv2.resize(
            rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5)
        if boxes is None or len(boxes) == 0:
            return 0.5
        bh, bw = max(boxes, key=lambda b: b[2] * b[3])
        fh = int(bh * 0.30)
        if fh < 3:
            return 0.5
        eye = gray[bh: bh + fh, bw: bw + bw]
        if eye.size == 0:
            return 0.5
        # Open eyes give strong dark-pupil/light-sclera contrast.
        return float(np.clip(eye.std() / 42.0, 0.0, 1.0))
    except Exception:
        return 0.5


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """64-bit difference hash. Near-identical images -> small Hamming distance."""
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(small, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ------------------------------------------------------------------ aggregate

@dataclass
class FrameMetrics:
    sharpness: float = 0.0
    tenengrad: float = 0.0
    exposure: float = 0.0        # 0..1, 1 = well exposed
    clipped: float = 0.0         # fraction of blown pixels
    black: float = 0.0           # fraction of crushed pixels
    contrast: float = 0.0        # 0..1
    colour: float = 0.0          # 0..1
    faces: int = 0
    eyes: float = 0.5
    motion: float = 0.0          # mean abs diff vs previous frame (video only)
    luma: float = 128.0
    hash: int = 0

    def hard_fail(self, min_sharpness: float, max_clipped: float,
                  max_black: float) -> str | None:
        if self.sharpness < min_sharpness:
            return f"blur (sharpness {self.sharpness:.0f})"
        if self.clipped > max_clipped:
            return f"overexposed ({self.clipped:.0%} blown)"
        if self.black > max_black:
            return f"underexposed ({self.black:.0%} crushed)"
        return None


def measure(rgb: np.ndarray, *, prev_gray: np.ndarray | None = None,
            want_hash: bool = False, want_faces: bool = True,
            hash_image: Image.Image | None = None) -> FrameMetrics:
    m = FrameMetrics()
    gray = _gray(rgb)
    m.luma = float(np.mean(gray))
    m.sharpness = laplacian_var(gray)
    m.tenengrad = tenengrad(gray)

    m.clipped = float(np.mean(gray > 247))
    m.black = float(np.mean(gray < 8))

    spread = float(np.std(gray))
    m.contrast = float(np.clip(spread / 70.0, 0.0, 1.0))
    m.colour = colourfulness(rgb)

    # Exposure quality: peak at mid grey, penalised by clipping either end.
    darkness = abs(m.luma - 128.0) / 128.0
    exposure = 1.0 - (darkness ** 1.7)
    exposure -= 1.6 * m.clipped + 0.8 * m.black
    m.exposure = float(np.clip(exposure, 0.0, 1.0))

    if want_faces:
        m.faces = face_count(rgb)
        m.eyes = faces_are_eyes_open(rgb) if m.faces else 0.5

    if prev_gray is not None and prev_gray.shape == gray.shape:
        m.motion = float(np.mean(np.abs(gray - prev_gray)))

    if want_hash and hash_image is not None:
        m.hash = dhash(hash_image)
    return m
