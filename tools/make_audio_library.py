#!/usr/bin/env python3
"""Build a copy of the test library whose video clips actually carry audio.

The synthetic trip is generated silent, which leaves the whole native-audio
path untested: per-shot audio extraction, level matching, the fades, and the
sidechain compressor that ducks the music under speech. This mints a variant
library with speech-like audio in every clip, loudness varying per clip so the
normaliser and the ducking threshold both have something to do.

    .venv/bin/python tools/make_audio_library.py media media_audio
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
RATE = 48000


def speech_like(duration: float, seed: int) -> np.ndarray:
    """A crude stand-in for a talking clip.

    Not speech, but the right *shape* for testing: bursts of voiced sound with
    pauses between them, so the ducking has transients to react to rather than a
    steady tone that would only ever produce one gain-reduction value.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * RATE)
    t = np.arange(n) / RATE
    out = np.zeros(n, dtype=np.float32)

    # Syllable rate around 4Hz, jittered.
    pos = 0.0
    while pos < duration:
        pos += float(rng.uniform(0.12, 0.30))
        length = float(rng.uniform(0.08, 0.22))
        a, b = int(pos * RATE), int(min(duration, pos + length) * RATE)
        if b <= a:
            continue
        seg_t = t[a:b] - pos
        # A fundamental with a couple of harmonics and a form-ish envelope.
        f0 = float(rng.uniform(95, 210))
        env = np.sin(np.pi * np.clip(seg_t / max(length, 1e-6), 0, 1)) ** 1.5
        tone = sum(
            (1.0 / h) * np.sin(2 * np.pi * f0 * h * seg_t + rng.uniform(0, 6.28))
            for h in (1, 2, 3, 4)
        )
        out[a:b] += env * tone.astype(np.float32) * 0.5

    # Room tone so it is never true digital silence between words.
    out += rng.normal(0.0, 0.0015, n).astype(np.float32)
    peak = float(np.abs(out).max())
    return (out / peak * 0.9).astype(np.float32) if peak > 0 else out


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "test_media"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "test_media_audio"
    if not src.is_dir():
        print(f"no such folder: {src}", file=sys.stderr)
        return 1

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    photos = [p for p in sorted(src.iterdir())
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".tif",
                                      ".tiff", ".webp"}]
    clips = [p for p in sorted(src.iterdir())
             if p.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}]

    for p in photos:
        shutil.copy2(p, dst / p.name)

    tmp = ROOT / ".work_audio_src"
    tmp.mkdir(exist_ok=True)
    made = 0
    for i, clip in enumerate(clips):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(clip)],
            capture_output=True, text=True, check=True).stdout.strip()
        dur = float(probe)

        # Deliberately uneven levels: quiet, loud, and in-between.
        gain = (0.10, 0.55, 0.25, 0.85)[i % 4]
        wav = tmp / f"a{i:03d}.wav"
        sf.write(wav, speech_like(dur, seed=1000 + i) * gain, RATE)

        out = dst / clip.name
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-i", str(clip), "-i", str(wav),
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
             "-shortest", str(out)],
            check=True)
        made += 1
        print(f"  {clip.name}  {dur:5.1f}s  audio @ {gain:.2f} gain")
        wav.unlink()

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{made} clips given audio, {len(photos)} photos copied -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
