#!/usr/bin/env python3
"""Generate a synthetic trip library for testing the pipeline end to end.

Produces photos and video clips with realistic variety: blurred shots, dark
shots, burst duplicates, portrait/landscape mixes, GPS clusters across three
days, and a couple of genuinely bad frames that should be rejected.
"""
from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
# Deliberately not media/ - that folder belongs to the user's real photos, and
# a test library sitting in it would quietly end up in their recap.
OUT = ROOT / "test_media"
FPS = 24

PLACES = [
    # (name, lat, lon)
    ("Kyoto", 35.0116, 135.7681),
    ("Nara", 34.6851, 135.8048),
    ("Osaka", 34.6937, 135.5023),
]
DAY_STARTS = [datetime(2026, 4, 12, 9, 0, tzinfo=timezone.utc),
              datetime(2026, 4, 13, 9, 30, tzinfo=timezone.utc),
              datetime(2026, 4, 14, 17, 0, tzinfo=timezone.utc)]


def palette(rng: random.Random) -> tuple[tuple, tuple]:
    h = rng.random()
    import colorsys
    r1, g1, b1 = colorsys.hsv_to_rgb(h, 0.65, 0.85)
    r2, g2, b2 = colorsys.hsv_to_rgb((h + 0.12) % 1.0, 0.85, 0.35)
    return ((int(r1 * 255), int(g1 * 255), int(b1 * 255)),
            (int(r2 * 255), int(g2 * 255), int(b2 * 255)))


def scene(w: int, h: int, rng: random.Random, *, portrait: bool,
          seed: int) -> Image.Image:
    """A plausible-looking frame: gradient sky, hills, sun, some structure."""
    top, bottom = palette(rng)
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        d.line([(0, y), (w, y)],
               fill=(int(top[0] + (bottom[0] - top[0]) * t),
                     int(top[1] + (bottom[1] - top[1]) * t),
                     int(top[2] + (bottom[2] - top[2]) * t)))

    # Sun / moon
    sx = rng.randint(int(w * 0.1), int(w * 0.9))
    sy = rng.randint(int(h * 0.08), int(h * 0.4))
    r = rng.randint(int(w * 0.04), int(w * 0.10))
    d.ellipse([sx - r, sy - r, sx + r, sy + r],
              fill=(min(255, top[0] + 70), min(255, top[1] + 60),
                    min(255, top[2] + 30)))

    # Layered hills
    layers = rng.randint(2, 4)
    for i in range(layers):
        frac = 0.45 + 0.16 * i
        amp = rng.uniform(0.03, 0.12) * h
        phase = rng.uniform(0, 6.28)
        freq = rng.uniform(1.2, 3.4)
        pts = [(x, int(h * frac + amp * math.sin(freq * x / w * 6.28 + phase)))
               for x in range(0, w + 1, max(1, w // 60))]
        pts += [(w, h), (0, h)]
        shade = 0.45 + 0.14 * i
        d.polygon(pts, fill=(int(top[0] * shade), int(top[1] * shade),
                             int(top[2] * shade * 0.95)))

    # Foreground blocks (buildings / walls) to give edges for sharpness tests
    for _ in range(rng.randint(2, 6)):
        bx = rng.randint(0, w - 1)
        bw = rng.randint(int(w * 0.05), int(w * 0.22))
        by = rng.randint(int(h * 0.35), int(h * 0.8))
        bh = rng.randint(int(h * 0.08), int(h * 0.3))
        v = rng.uniform(0.25, 0.75)
        d.rectangle([bx, by, min(w - 1, bx + bw), min(h - 1, by + bh)],
                    fill=(int(top[0] * v), int(top[1] * v), int(top[2] * v * 1.1)))
        for wy in range(by + 4, min(h - 2, by + bh), 9):
            for wx in range(bx + 4, min(w - 2, bx + bw), 9):
                if rng.random() < 0.45:
                    d.rectangle([wx, wy, wx + 3, wy + 4], fill=(250, 235, 190))

    # Fine texture so sharpness metrics have something to bite on
    rs = np.random.default_rng(seed)
    noise = rs.normal(0, 9, (h, w, 1)).repeat(3, axis=2)
    arr = np.clip(np.asarray(img, np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def degrade(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(3.2, 5.5)))
    if kind == "dark":
        return Image.fromarray(
            (np.asarray(img, np.float32) * rng.uniform(0.10, 0.18)).astype(np.uint8))
    if kind == "blown":
        return Image.fromarray(
            np.clip(np.asarray(img, np.float32) * 1.9 + 90, 0, 255).astype(np.uint8))
    if kind == "flat":
        g = img.convert("L").convert("RGB")
        return g.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.4, 2.2)))
    return img


def stamp(path: Path, when: datetime, lat: float, lon: float,
          camera: str = "iPhone 16 Pro") -> None:
    cmd = [
        "exiftool", "-overwrite_original", "-q",
        f"-DateTimeOriginal={when.strftime('%Y:%m:%d %H:%M:%S')}",
        f"-CreateDate={when.strftime('%Y:%m:%d %H:%M:%S')}",
        f"-GPSLatitude={lat}", f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
        f"-GPSLongitude={lon}", f"-GPSLongitudeRef={'E' if lon >= 0 else 'W'}",
        f"-Make=Apple", f"-Model={camera}",
        str(path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_video(path: Path, seconds: float, seed: int, portrait: bool) -> None:
    rng = random.Random(seed)
    w, h = (1080, 1920) if portrait else (1920, 1080)
    aw, ah = (360, 640) if portrait else (640, 360)
    n = int(seconds * FPS)
    base = scene(aw, ah, rng, portrait=portrait, seed=seed)
    pan = rng.uniform(0.4, 1.8)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{aw}x{ah}", "-r", str(FPS), "-i", "-",
         "-vf", f"scale={w}:{h}:flags=bicubic,format=yuv420p",
         "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
         "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for i in range(n):
        t = i / n
        # Slow drift plus a gentle brightness sweep, so motion is detectable
        # and shots differ from one another.
        dx = int((pan * t) * aw * 0.10)
        dy = int(math.sin(t * 6.28) * ah * 0.03)
        ox = max(0, min(aw - 1, dx))
        oy = max(0, min(ah - 1, dy))
        crop = base.crop((ox, oy, min(aw, ox + aw), min(ah, oy + ah)))
        arr = np.asarray(crop, np.float32) * (0.92 + 0.16 * math.sin(t * 6.28))
        proc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()


def main() -> int:
    rng = random.Random(20260412)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()

    n_photo = 0
    n_video = 0
    video_meta: list[tuple[Path, datetime, float, float]] = []

    for day_i, day_start in enumerate(DAY_STARTS):
        place_i = min(day_i, len(PLACES) - 1)
        name, lat, lon = PLACES[place_i]
        clock = day_start

        # 6 video clips per day, spread through the day
        for k in range(6):
            clock += timedelta(minutes=rng.randint(25, 130))
            portrait = rng.random() < 0.45
            secs = rng.uniform(3.5, 9.0)
            fname = f"video_day{day_i + 1}_{k + 1:02d}.mp4"
            p = OUT / fname
            make_video(p, secs, seed=rng.randint(1, 10 ** 6), portrait=portrait)
            video_meta.append((p, clock, lat + rng.uniform(-0.004, 0.004),
                               lon + rng.uniform(-0.004, 0.004)))
            n_video += 1

        # ~14 photos per day: mostly good, with a burst and some duds
        for k in range(14):
            clock += timedelta(seconds=rng.randint(20, 400))
            portrait = rng.random() < 0.45
            w, h = (1170, 2532) if portrait else (2532, 1170)
            img = scene(w // 2, h // 2, rng, portrait=portrait,
                        seed=rng.randint(1, 10 ** 6))

            roll = rng.random()
            if roll < 0.10:
                img = degrade(img, "blur", rng)
            elif roll < 0.16:
                img = degrade(img, "dark", rng)
            elif roll < 0.20:
                img = degrade(img, "blown", rng)
            elif roll < 0.26:
                img = degrade(img, "flat", rng)
            elif roll < 0.40:
                # Burst: nearly identical to the previous frame.
                img = img.rotate(rng.uniform(-0.7, 0.7), resample=Image.BICUBIC)

            fname = f"IMG_day{day_i + 1}_{k + 1:04d}.jpg"
            p = OUT / fname
            img.save(p, quality=90)
            la = lat + rng.uniform(-0.003, 0.003)
            lo = lon + rng.uniform(-0.003, 0.003)
            stamp(p, clock, la, lo)
            os.utime(p, (clock.timestamp(), clock.timestamp()))
            n_photo += 1

    for p, when, la, lo in video_meta:
        stamp(p, when, la, lo)
        os.utime(p, (when.timestamp(), when.timestamp()))

    print(f"created {n_photo} photos and {n_video} videos in {OUT}")
    print(f"  3 days across {len(PLACES)} locations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
