"""Stage 5 - music.

Two paths:

* ``generate``  - synthesise an original score. Because we create the audio we
  also know its beat grid *exactly*, so cuts land on real beats with no
  detection error. Also completely licence-free.
* ``track``     - use a track you supply. Then we have to find the beats by
  listening, via spectral-flux onset detection and an Ellis-style dynamic
  programming beat tracker (implemented here in numpy, no heavy deps).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import arrange
from .util import log

SR = 44100

# --------------------------------------------------------------------- result

@dataclass
class MusicResult:
    path: Path
    duration: float
    bpm: float
    beat_times: list[float] = field(default_factory=list)
    downbeat_times: list[float] = field(default_factory=list)
    generated: bool = False
    label: str = ""
    # Audio timestamp of the first beat; the grid is offset so this lands at 0.
    phase: float = 0.0
    # atempo factor that brings the audio onto the quantised grid (1.0 = exact).
    stretch: float = 1.0

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm if self.bpm > 0 else 1.0

    def beats_in(self, start: float, end: float) -> list[float]:
        return [b for b in self.beat_times if start <= b < end]


def quantise_tempo(bpm: float, fps: int) -> tuple[float, float]:
    """Snap a tempo to the video frame grid.

    A shot's length has to be a whole number of *frames*, so a tempo that does
    not land on whole frames cannot be represented exactly - and the rounding
    error accumulates across every cut until the last shot falls visibly off
    the beat.

    Rounding to the nearest whole frame per beat is the closest representable
    tempo, and costs up to ~2% at slow tempi. That is worth paying only for
    music we synthesise ourselves, where the tempo is ours to choose; for a
    supplied track the error is applied to the timeline instead of the audio.
    """
    if bpm <= 0:
        return 120.0, 1.0
    frames_per_beat = max(1, int(round((60.0 / bpm) * fps)))
    bpm_eff = 60.0 * fps / frames_per_beat
    return bpm_eff, bpm_eff / bpm


def _regular_grid(bpm: float, duration: float, phase: float) -> list[float]:
    spb = 60.0 / bpm
    n = int(duration / spb) + 1
    return [phase + i * spb for i in range(n)]


# =========================================================== beat detection

def _read_mono(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    import soundfile as sf
    try:
        data, native_sr = sf.read(str(path), always_2d=True)
    except Exception:
        # Transcode anything exotic (m4a/aac) via ffmpeg first.
        tmp = path.with_suffix(".decoded.wav")
        from .util import run
        run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
             "-ac", "1", "-ar", str(sr), str(tmp)])
        data, native_sr = sf.read(str(tmp), always_2d=True)
        tmp.unlink(missing_ok=True)
    mono = data.mean(axis=1)
    if native_sr != sr:
        n = int(round(len(mono) * sr / native_sr))
        mono = np.interp(
            np.linspace(0, len(mono) - 1, n),
            np.arange(len(mono)),
            mono,
        ).astype(np.float32)
    return mono.astype(np.float32), sr


def _onset_envelope(x: np.ndarray, sr: int,
                    n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Log-magnitude spectral flux, mean-removed and half-wave rectified."""
    if len(x) < n_fft * 2:
        return np.zeros(1, dtype=np.float64)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(n_fft)[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1))
    logm = np.log1p(100.0 * mag)

    diff = np.diff(logm, axis=0)
    flux = np.sum(np.maximum(diff, 0.0), axis=1)
    flux = np.concatenate([[0.0], flux])

    # Remove local mean so sustained tones don't dominate.
    kernel = np.hanning(15)
    kernel /= kernel.sum()
    padded = np.pad(flux, (7, 7), mode="edge")
    local = np.convolve(padded, kernel, mode="valid")[:len(flux)]
    env = np.maximum(flux - local, 0.0)

    std = env.std()
    return (env / std).astype(np.float64) if std > 1e-9 else env


def _estimate_tempo(env: np.ndarray, sr: int, hop: int,
                    bpm_range: tuple[float, float] = (60.0, 185.0)) -> float:
    """Autocorrelate the onset envelope, with a log-normal prior at 120 BPM."""
    fps = sr / hop
    n = len(env)
    if n < 8:
        return 120.0
    centred = env - env.mean()
    ac = np.correlate(centred, centred, mode="full")[n - 1:]
    ac /= (ac[0] + 1e-9)

    min_lag = max(1, int(round(60.0 / bpm_range[1] * fps)))
    max_lag = min(n - 1, int(round(60.0 / bpm_range[0] * fps)))
    if max_lag <= min_lag:
        return 120.0

    lags = np.arange(min_lag, max_lag)
    scores = ac[min_lag:max_lag].copy()
    bpms = 60.0 * fps / lags
    # Prefer tempi near 120, mildly.
    prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)
    scores *= prior

    best = lags[int(np.argmax(scores))]
    bpm = float(np.clip(60.0 * fps / best, *bpm_range))

    # Octave check: is twice or half the tempo a better fit?
    for factor in (0.5, 2.0):
        alt = bpm * factor
        if not (bpm_range[0] <= alt <= bpm_range[1]):
            continue
        lag = 60.0 * fps / alt
        lo, hi = int(max(1, lag - 1)), int(min(len(ac) - 1, lag + 1)) + 1
        if hi <= lo:
            continue
        if float(ac[lo:hi].max()) > float(ac[best - 1:best + 2].max()) * 1.06:
            bpm = float(np.clip(alt, *bpm_range))
            best = int(lag)
    return bpm


def _dp_beat_track(env: np.ndarray, bpm: float, sr: int, hop: int,
                   tightness: float = 100.0) -> np.ndarray:
    """Ellis-style dynamic programming beat tracker."""
    fps = sr / hop
    n = len(env)
    period = 60.0 * fps / bpm
    if period <= 1 or n < int(period * 2):
        return np.array([], dtype=int)

    # Transition cost: penalise beats that land far from a musically correct
    # lag behind the previous one. This is a correlation of the onset envelope
    # with the log-Gaussian transition weight, so compute it in one shot.
    width = int(np.ceil(period * 2)) + 1
    lags = np.arange(1, width + 2, dtype=np.float64)
    w = -tightness * (np.log(lags / period) ** 2)
    full = np.convolve(env, w, mode="full")
    txcost = full[width: width + n]

    score = np.zeros(n)
    back = np.zeros(n, dtype=int)
    score[0] = txcost[0]
    search = int(np.ceil(period * 2))
    for t in range(1, n):
        lo = max(0, t - search)
        cand = score[lo:t] + txcost[t]
        k = int(np.argmax(cand))
        score[t] = cand[k]
        back[t] = lo + k

    beats = [n - 1]
    t = n - 1
    while t > 0:
        t = back[t]
        beats.append(t)
    beats = np.array(sorted(set(beats)), dtype=int)
    return beats


def detect_beats(path: Path) -> MusicResult:
    log(f"analysing beats in {path.name}")
    x, sr = _read_mono(path)
    duration = len(x) / sr
    hop = 256
    env = _onset_envelope(x, sr, hop=hop)
    bpm = _estimate_tempo(env, sr, hop)
    beats = _dp_beat_track(env, bpm, sr, hop)
    fps = sr / hop
    beat_times = (beats / fps).tolist()
    # Downbeats: every 4th beat, phase-chosen to maximise onset energy.
    beats_per_bar = 4
    downbeats: list[float] = []
    if len(beat_times) >= beats_per_bar:
        offsets = list(range(beats_per_bar))
        energies = []
        for o in offsets:
            sel = beat_times[o::beats_per_bar]
            e = sum(env[min(len(env) - 1, int(b * fps))] for b in sel)
            energies.append(e)
        phase = int(np.argmax(energies))
        downbeats = beat_times[phase::beats_per_bar]
    log(f"detected {bpm:.1f} BPM, {len(beat_times)} beats")
    return MusicResult(path=path, duration=duration, bpm=bpm,
                       beat_times=beat_times, downbeat_times=downbeats,
                       generated=False, label=path.stem)


# ============================================================ synthesis

_NOTE = 440.0
_MIDI_A4 = 69

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]

MAJ = (0, 4, 7)
MIN = (0, 3, 7)
DOM7 = (0, 4, 7, 10)
MAJ7 = (0, 4, 7, 11)
MIN7 = (0, 3, 7, 10)
SUS2 = (0, 2, 7)
SUS4 = (0, 5, 7)

# (root offset from key root, chord tone set)
PROGRESSIONS = {
    "uplifting": [(0, MAJ), (7, MAJ), (9, MIN), (5, MAJ)],
    "energetic": [(0, MAJ), (5, MAJ), (7, MAJ), (9, MIN)],
    "cinematic": [(9, MIN7), (5, MAJ), (0, MAJ), (7, DOM7)],
    "warm":      [(0, MAJ7), (9, MIN7), (5, MAJ7), (7, DOM7)],
    "calm":      [(0, MAJ7), (5, MAJ7), (9, MIN7), (7, DOM7)],
}
MODES = {
    "uplifting": (MAJOR, 0), "energetic": (MAJOR, 0), "cinematic": (MINOR, 2),
    "warm": (MAJOR, 5), "calm": (MAJOR, 5),
}
BPM_RANGE = {
    "uplifting": (116, 126), "energetic": (128, 140), "cinematic": (78, 90),
    "warm": (96, 106), "calm": (82, 94),
}


def _hz(semitone: float) -> float:
    return _NOTE * (2.0 ** ((semitone - _MIDI_A4) / 12.0))


def _adsr(n: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    """Simple ADSR envelope, lengths in seconds, s = sustain level 0..1."""
    na, nd = int(a * sr), int(d * sr)
    nr = int(r * sr)
    ns = max(0, n - na - nd - nr)
    env = np.concatenate([
        np.linspace(0.0, 1.0, na, endpoint=False) if na > 0 else np.zeros(0),
        np.linspace(1.0, s, nd, endpoint=False) if nd > 0 else np.zeros(0),
        np.full(ns, s),
        np.linspace(s, 0.0, nr) if nr > 0 else np.zeros(0),
    ])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def _pluck(freq: float, dur: float, sr: int, *, bright: float = 0.5) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    # Triangle-ish: fundamental plus decaying harmonics.
    sig = np.zeros(n)
    for h, amp in ((1, 1.0), (2, 0.42 * bright), (3, 0.22 * bright),
                   (4, 0.11 * bright), (5, 0.06 * bright)):
        sig += amp * np.sin(2 * np.pi * freq * h * t)
    sig /= 1.85
    decay = np.exp(-t * (4.0 + 2.0 * (1.0 - bright)))
    attack = np.clip(t / 0.006, 0.0, 1.0)
    return sig * decay * attack


def _pad(freqs: list[float], dur: float, sr: int, *, detune: float = 0.004
         ) -> tuple[np.ndarray, np.ndarray]:
    """Soft detuned pad. Returns (left, right)."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    left = np.zeros(n)
    right = np.zeros(n)
    for f in freqs:
        for sign, dest in ((-1, "l"), (1, "r")):
            ph = np.random.uniform(0, 2 * np.pi)
            sig = np.zeros(n)
            for h, amp in ((1, 1.0), (2, 0.30), (3, 0.13), (4, 0.06)):
                sig += amp * np.sin(2 * np.pi * f * h * t + ph
                                    + sign * detune * 2 * np.pi * f * t)
            sig /= 1.5
            if dest == "l":
                left += sig
            else:
                right += sig
    env = _adsr(n, sr, a=min(0.9, dur * 0.28), d=0.3, s=0.72,
                r=min(1.4, dur * 0.35))
    return left * env, right * env


def _bass(freq: float, dur: float, sr: int) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    sig = (np.sin(2 * np.pi * freq * t)
           + 0.28 * np.sin(2 * np.pi * freq * 2 * t)
           + 0.10 * np.sin(2 * np.pi * freq * 3 * t))
    sig = np.tanh(sig * 1.25) * 0.72
    env = _adsr(n, sr, a=0.008, d=0.12, s=0.78, r=0.12)
    return sig * env


def _kick(sr: int) -> np.ndarray:
    n = int(0.32 * sr)
    t = np.arange(n) / sr
    sweep = 118.0 * np.exp(-t * 34.0) + 44.0
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    body = np.sin(phase) * np.exp(-t * 11.0)
    click = np.random.default_rng(1).normal(0, 1, n) * np.exp(-t * 260.0) * 0.25
    return np.tanh((body + click) * 1.3)


def _snare(sr: int) -> np.ndarray:
    rng = np.random.default_rng(2)
    n = int(0.22 * sr)
    t = np.arange(n) / sr
    noise = rng.normal(0, 1, n)
    # Crude high-pass: difference the noise.
    noise = np.diff(noise, prepend=0.0)
    noise /= (np.abs(noise).max() + 1e-9)
    tone = (np.sin(2 * np.pi * 186 * t) + np.sin(2 * np.pi * 331 * t)) * 0.35
    env = np.exp(-t * 26.0)
    return (noise * 0.8 + tone) * env * 0.75


def _hat(sr: int, *, open_: bool = False) -> np.ndarray:
    rng = np.random.default_rng(3)
    n = int((0.16 if open_ else 0.055) * sr)
    t = np.arange(n) / sr
    noise = rng.normal(0, 1, n)
    noise = np.diff(noise, prepend=0.0)
    noise = np.diff(noise, prepend=0.0)
    noise /= (np.abs(noise).max() + 1e-9)
    env = np.exp(-t * (16.0 if open_ else 55.0))
    return noise * env * 0.30


def _reverb(x: np.ndarray, sr: int, amount: float = 0.20,
            decay: float = 1.9) -> np.ndarray:
    """Cheap diffuse reverb: a handful of decorrelated delay taps."""
    if amount <= 0:
        return x
    taps = [0.013, 0.021, 0.037, 0.049, 0.071, 0.097, 0.131]
    out = x.copy()
    rng = np.random.default_rng(7)
    for i, d in enumerate(taps):
        g = int(d * sr)
        w = 0.5 ** (i * decay / 2.0)
        sig = rng.normal(0, 1, len(x) - g) * w
        # Smooth the noise tail so it reads as a room, not hiss.
        k = np.hanning(min(g, 512)) if g >= 4 else np.ones(max(1, g))
        sig = np.convolve(sig, k / k.sum(), mode="same")
        out[g:] += sig * amount * 0.6
    return out


def generate_music(target_seconds: float, mood: str = "uplifting",
                   bpm: int = 0, out_path: Path | None = None,
                   seed: int = 0) -> MusicResult:
    """Synthesise an original bed. Fully licence-free."""
    mood = mood if mood in PROGRESSIONS else "uplifting"
    scale, tonic_pc = MODES[mood]
    prog = PROGRESSIONS[mood]
    lo, hi = BPM_RANGE[mood]
    bpm = float(bpm or int(round((lo + hi) / 2)))

    # Pick a pleasant absolute key (avoid harsh keys near our tonic choices).
    # Seeded from the mood, bpm and target rather than hash(). Python randomises
    # string hashing per process, so hash() here gave every run a different
    # score - the same library produced visibly different edits on each run, and
    # any bug that only shows up for certain musical content became impossible
    # to reproduce. zlib.crc32 is stable across processes and runs.
    np.random.seed(seed or zlib.crc32(
        f"{mood}|{bpm:.4f}|{target_seconds:.3f}".encode()) % (2 ** 31))
    key_root = tonic_pc + np.random.choice([0, 2, 4, 5, 7, 9, 11])

    spb = 60.0 / bpm
    beats_per_bar = 4
    bar = spb * beats_per_bar
    n_bars = int(math.ceil(target_seconds / bar)) + 1
    total = n_bars * bar

    n = int(total * SR)
    pad_l = np.zeros(n)
    pad_r = np.zeros(n)
    bass_buf = np.zeros(n)
    arp_buf_l = np.zeros(n)
    arp_buf_r = np.zeros(n)
    kick_buf = np.zeros(n)
    snare_buf = np.zeros(n)
    hat_buf = np.zeros(n)

    intro_bars = 2
    drums_on = intro_bars * bar

    # The arrangement, so a long piece has a shape instead of one looped bar.
    # See pipeline.arrange - the short version is that this used to repeat the
    # same four bars to fill the target, which at ten minutes meant hearing the
    # same ten seconds sixty times.
    shape = arrange.plan(n_bars, max_bars=max(2, int(90.0 / bar)))

    for b in range(n_bars):
        sec = arrange.section_at(shape, b)
        deg, tones = prog[b % len(prog)]
        root_pc = (key_root + deg) % 12
        t0 = b * bar
        start = int(t0 * SR)

        # ---- pad: whole bar, one chord
        chord_freqs = [_hz(key_root + 12 * 3 + root_pc + iv) for iv in tones]
        chord_freqs += [_hz(key_root + 12 * 3 + root_pc + tones[0] + 12)]
        chord_l, chord_r = _pad(chord_freqs, bar * 1.02, SR)
        end = min(n, start + len(chord_l))
        pad_l[start:end] += chord_l[:end - start] * sec.pad
        pad_r[start:end] += chord_r[:end - start] * sec.pad

        # ---- bass: root on 1 and 3, fifth on the "and" of 3
        for beat, mul in ((0, 1.0), (2, 1.0), (2.5, 1.1892)):
            s = int((t0 + beat * spb) * SR)
            dur = spb * (1.6 if beat == 0 else 0.9)
            sig = _bass(_hz(key_root + root_pc + 12), dur, SR)
            e = min(n, s + len(sig))
            if s < e:
                bass_buf[s:e] += (sig[:e - s] * mul * sec.bass
                                  * (0.9 if b >= intro_bars else 0.6))

        # ---- arpeggio: sixteenths through the chord, in stereo
        if sec.arp > 0.02 and (mood not in ("cinematic",) or b >= intro_bars):
            steps = 8
            arp_notes = [tones[k % len(tones)] + 12 * (2 + (k // len(tones)))
                         for k in range(len(tones) + 1)]
            for s_i in range(steps):
                beat_pos = s_i * (beats_per_bar / steps)
                s = int((t0 + beat_pos * spb) * SR)
                if s >= n:
                    continue
                pc = arp_notes[s_i % len(arp_notes)]
                f = _hz(key_root + root_pc + pc + 24)
                bright = (0.65 if mood == "energetic" else 0.4) * sec.brightness
                sig = _pluck(f, spb * 0.7, SR, bright=bright)
                sig *= (0.20 if b < intro_bars else 0.26) * sec.arp
                e = min(n, s + len(sig))
                if s >= e:
                    continue
                # Alternate slightly for width.
                arp_buf_l[s:e] += sig[:e - s]
                arp_buf_r[s:e] += sig[:e - s] * 0.88

        if t0 < drums_on or sec.drums < 0.02:
            continue

        # ---- drums
        for beat in range(beats_per_bar):
            s = int((t0 + beat * spb) * SR)
            if s >= n:
                continue
            global_beat = b * beats_per_bar + beat
            if beat in (0, 2):
                k = _kick(SR)
                e = min(n, s + len(k))
                kick_buf[s:e] += k[:e - s] * sec.drums
            if beat in (1, 3):
                sn = _snare(SR)
                e = min(n, s + len(sn))
                snare_buf[s:e] += sn[:e - s] * sec.drums
            # Offbeat hats
            for off in (0.5, 1.5, 2.5, 3.5):
                hs = int((t0 + off * spb) * SR)
                if hs >= n:
                    continue
                hh = _hat(SR, open_=(off == 3.5 and mood == "cinematic"))
                e = min(n, hs + len(hh))
                hat_buf[hs:e] += hh[:e - hs] * sec.drums
            # Extra 16th hat pickup on every other bar for forward motion.
            if mood in ("energetic", "uplifting") and global_beat % 4 == 3:
                hs = int((t0 + 3.75 * spb) * SR)
                if hs < n:
                    hh = _hat(SR)
                    e = min(n, hs + len(hh))
                    hat_buf[hs:e] += hh[:e - hs] * sec.drums

    # ---- sidechain pump the melodic layers on each kick
    duck = np.ones(n, dtype=np.float64)
    step = int(spb * SR)
    for i in range(0, n, max(1, step)):
        length = int(0.16 * SR)
        seg = np.exp(-np.arange(length) / (0.055 * SR))
        e = min(n, i + length)
        duck[i:e] = np.minimum(duck[i:e], 1.0 - 0.32 * seg[:e - i])

    pad_l *= duck
    pad_r *= duck
    arp_buf_l *= duck
    arp_buf_r *= duck
    bass_buf *= duck

    # ---- reverb on pad + arp
    pad_l = _reverb(pad_l, SR, amount=0.26 if mood != "energetic" else 0.14)
    pad_r = _reverb(pad_r, SR, amount=0.26 if mood != "energetic" else 0.14)
    arp_buf_l = _reverb(arp_buf_l, SR, amount=0.14)
    arp_buf_r = _reverb(arp_buf_r, SR, amount=0.14)

    mix_l = (pad_l * 0.30 + bass_buf * 0.40 + arp_buf_l * 0.55
             + kick_buf * 0.72 + snare_buf * 0.26 + hat_buf * 0.34)
    mix_r = (pad_r * 0.30 + bass_buf * 0.40 + arp_buf_r * 0.55
             + kick_buf * 0.72 + snare_buf * 0.26 + hat_buf * 0.34)

    # ---- master: soft clip then normalise
    def _master(x: np.ndarray, ceiling: float = 0.92) -> np.ndarray:
        x = np.tanh(x * 1.05)
        peak = float(np.abs(x).max())
        return x * (ceiling / peak) if peak > 1e-9 else x

    mix_l = _master(mix_l)
    mix_r = _master(mix_r)

    stereo = np.stack([mix_l, mix_r], axis=1).astype(np.float32)
    fade = int(0.5 * SR)
    env = np.ones(n)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    stereo *= env[:, None]

    out_path = out_path or Path("music_generated.wav")
    import soundfile as sf
    sf.write(str(out_path), stereo, SR, subtype="PCM_16")
    duration = n / SR

    beat_times = [i * spb for i in range(int(duration / spb) + 1)]
    downbeats = [i * spb for i in range(0, len(beat_times), beats_per_bar)]
    log(f"generated '{mood}' score at {bpm:.0f} BPM "
        f"({n_bars} bars, {duration:.1f}s) -> {out_path.name}")
    return MusicResult(path=out_path, duration=duration, bpm=bpm,
                       beat_times=beat_times, downbeat_times=downbeats,
                       generated=True, label=f"{mood} {bpm:.0f}bpm")


# ------------------------------------------------------------------ driver

def prepare_music(cfg, out_dir: Path, *, library_audio: list[Path] | None = None,
                  target_seconds: float = 90.0, prefer: str | None = None,
                  fps: int = 30) -> MusicResult:
    """Resolve the music for this render, snapped to the frame grid."""
    mode = cfg.music.mode

    # 1. explicit user preference
    if prefer:
        mode = "track"

    if mode == "track":
        candidates: list[Path] = []
        if cfg.music.track_path:
            candidates.append(Path(cfg.music.track_path).expanduser())
        candidates.extend(library_audio or [])
        for c in candidates:
            if c and c.exists():
                m = detect_beats(c)
                return _finalise(m, fps, target_seconds, on_grid=False)
        log("no usable track supplied; generating instead", level="warn")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Generated music can be rendered straight onto the grid, so pick the
    # quantised tempo first and synthesise at that exact value.
    lo, hi = BPM_RANGE.get(cfg.music.mood, BPM_RANGE["uplifting"])
    want = float(cfg.music.bpm or int(round((lo + hi) / 2)))
    want = min(max(want, min(lo, hi)), max(lo, hi))
    exact, _ = quantise_tempo(want, fps)
    # Name the score after what determines its content, not a per-process hash of
    # the mood. The old form left a different score file behind on every run,
    # so a work directory slowly filled with orphaned WAVs that nothing
    # referred to.
    tag = zlib.crc32(f"{cfg.music.mood}|{exact:.4f}|{target_seconds:.3f}"
                     .encode()) % 100000
    path = out_dir / f"score_{cfg.music.mood}_{tag}.wav"
    m = generate_music(target_seconds, mood=cfg.music.mood, bpm=exact,
                       out_path=path)
    return _finalise(m, fps, target_seconds, on_grid=True)


def _finalise(m: MusicResult, fps: int, target_seconds: float, *,
              on_grid: bool) -> MusicResult:
    """Decide the final tempo and beat grid for this render.

    Generated scores are put *exactly* on the frame grid, because we choose the
    tempo and synthesise to it - every beat becomes a whole number of frames
    and the cuts are then mathematically exact with no audio altered at all.

    A supplied track keeps its own tempo. Snapping the tempo to the frame grid
    would mean resampling the music by up to ~2%, and hearing a track played
    2% fast is far more obvious than the thing it buys: cuts that sit within
    half a frame (17ms at 30fps) of the beat, which is below the threshold
    anyone can hear. So we keep the real tempo and let the timeline snap to the
    nearest frame per cut instead.
    """
    if on_grid:
        bpm, _ = quantise_tempo(m.bpm, fps)
        phase = 0.0
        beats = _regular_grid(bpm, target_seconds, phase)
    else:
        bpm = m.bpm
        phase = min((b for b in m.beat_times if b >= 0.0), default=0.0)
        spb = 60.0 / bpm
        n = max(1, int(max(0.0, m.duration - phase) / spb))
        beats = [phase + i * spb for i in range(n + 1)]

    downbeats = beats[::4] if beats else []
    return MusicResult(
        path=m.path, duration=m.duration, bpm=bpm,
        beat_times=beats, downbeat_times=downbeats,
        generated=m.generated, label=m.label,
        phase=phase, stretch=1.0,
    )
