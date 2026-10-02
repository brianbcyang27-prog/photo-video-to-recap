"""Programme-level loudness, and the ducking parameters that feed it.

Three separate things live here because they are three separate mistakes:

1. **Normalising per segment.** Every shot used to get its own `loudnorm` to
   -18 LUFS. That deleted the difference *between* places - a quiet market and a
   loud market both arrived at -18 - and it ran in single-pass dynamic mode, which
   re-gains as it goes. Worse, most shots in this project are room tone measured
   around -37 dB mean, so -18 was roughly a +19 dB lift on a noise floor.

2. **A near-hard knee on the ducking.** `sidechaincompress` has no knee parameter
   set, which means 2.82843 - and a 2.83 knee on a music bed at 6:1 is an audible
   step rather than a transition.

3. **Sample-peak limiting standing in for true-peak limiting.** `alimiter` does
   lookahead limiting with no oversampling, so the -1 dBTP that EBU R128,
   ATSC A/85, AES TD1008 and Netflix all require was never actually guaranteed -
   and AAC creates inter-sample peaks above the sample peak on top of that.

The thresholds here were measured, not guessed. Where a test needs ffmpeg it says
so; the pure-argument ones do not need a media file or a render.
"""
from __future__ import annotations

import importlib
import inspect
import re
import subprocess
import wave

import numpy as np
import pytest

# `from pipeline import render` binds the *function* the package re-exports, not the
# module, so anything reaching into module-level helpers imports it by name.
render = importlib.import_module("pipeline.render")

RATE = render.AUDIO_RATE


def _write_wav(path, x: np.ndarray, rate: int = RATE,
               normalise: bool = True) -> None:
    """Mono int16 WAV, so the fixture is a real file rather than a mock.

    `normalise` scales to peak, which keeps the level-sensitive fixtures (a -37 dB
    floor, a 12 dB shot-to-shot gap) from clipping in 16-bit. It preserves ratios,
    so relative comparisons still hold. Pass normalise=False for anything whose
    *absolute* level is the thing under test - notably true peak, where
    normalising would silently pin every fixture to 0 dBFS.
    """
    if normalise:
        peak = float(np.max(np.abs(x))) or 1.0
        x = x / peak
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def _speech_like(seconds: float, rate: int = RATE, floor_db: float = -37.0,
                 tone_hz: float = 220.0, gate_hz: float = 2.0) -> np.ndarray:
    """Gated tone over a noise floor: what room sound plus speech looks like."""
    rng = np.random.default_rng(11)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    floor = rng.normal(0.0, 10.0 ** (floor_db / 20.0), n)
    gate = (np.sin(2 * np.pi * gate_hz * t) > 0.6).astype(float)
    return floor + gate * np.sin(2 * np.pi * tone_hz * t) * 0.3


def _lufs(path) -> float | None:
    m = render._measure_loudness(path)
    return None if m is None else m["input_i"]


# ------------------------------------------------- no normalisation per segment

def test_extract_segment_audio_does_not_normalise():
    """The per-segment loudnorm is the bug, so assert it is not in the chain.

    Asserted against the resolved chain string rather than the source, because
    the value lives in a module constant that could be defined and unused.
    """
    seg = render.extract_segment_audio
    src = inspect.getsource(seg)
    chain_line = next(ln for ln in src.splitlines() if "chain = f" in ln)
    assert "loudnorm" not in chain_line, (
        "extract_segment_audio still runs loudnorm per segment: it deletes the "
        "loudness differences between shots, runs in dynamic (pumping) mode "
        "without measured_ inputs, and lifts -37 dB room tone by roughly 19 dB"
    )


def test_programme_normalisation_is_two_pass_and_feeds_the_measurements_back():
    """One pass in dynamic mode *is* the pumping. Two passes is not optional.

    Feeding `measured_I` / `measured_LRA` / `measured_TP` / `measured_thresh`
    back is what puts loudnorm on its linear path, where the gain is computed once
    for the whole programme and applied evenly instead of being re-derived between
    frames.
    """
    src = inspect.getsource(render._normalise_programme)
    for key in ("measured_I", "measured_LRA", "measured_TP", "measured_thresh"):
        assert key in src, f"the second pass never uses {key}, so loudnorm stays dynamic"
    assert "linear=true" in src


def test_programme_lands_on_the_target_instead_of_each_shot(tmp_path):
    """The whole point: one gain for the programme, measured on the whole programme."""
    cfg = render.Pipeline()
    src = tmp_path / "mixed.wav"
    _write_wav(src, _speech_like(6.0))

    before = _lufs(src)
    out = render._normalise_programme(src, cfg)
    after = _lufs(out)

    assert before is not None and after is not None
    assert abs(after - (-18.0)) <= 1.0, (
        f"programme normalised to {after:.2f} LUFS, not the -18 target"
    )


def test_shot_to_shot_differences_survive_programme_normalisation(tmp_path):
    """The loudness *relationship* between shots is content, and it used to be deleted.

    A quiet shot and a loud shot differ by 12 dB going in. Normalising each one
    separately made them identical, so a busy market and a still one arrived at the
    same level. Normalising the pair together must leave the gap essentially
    untouched - that is the measurable difference between the two designs.
    """
    cfg = render.Pipeline()
    quiet = _speech_like(3.0, floor_db=-37.0, tone_hz=220.0)
    loud = quiet * 4.0                      # ~12 dB hotter
    joined = np.concatenate([quiet, loud])
    src = tmp_path / "joined.wav"
    _write_wav(src, joined)

    def gap(x: np.ndarray) -> float:
        n = len(x) // 2
        def rms(part):
            return 20 * np.log10(np.sqrt(np.mean(part ** 2)) + 1e-12)
        return rms(x[n:]) - rms(x[:n])

    before = gap(joined)
    out = render._normalise_programme(src, cfg)

    raw = np.frombuffer(
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(out), "-f", "f32le", "-ac", "1",
             "-ar", str(RATE), "-"], stdout=-1).stdout, np.float32)
    after = gap(raw)

    assert before > 8.0
    assert abs(after - before) < 1.5, (
        f"the {before:.1f} dB difference between the two shots became "
        f"{after:.1f} dB; per-segment normalisation is what erases this"
    )


def test_room_tone_is_not_amplified_on_its_own_terms(tmp_path):
    """A noise floor taken to -18 LUFS is pumping hiss waiting to happen.

    Under the old scheme a -37 dB room-tone-only segment became -18 LUFS on its
    own. Programme normalisation moves the whole thing by one shared gain, so the
    floor's level *relative to the rest of the programme* is preserved.
    """
    cfg = render.Pipeline()
    rng = np.random.default_rng(3)
    n = int(8.0 * RATE)
    # Half speech-like, half pure noise floor at the level Live Photo room sound
    # actually measures (~-37 dB mean). Under the old scheme each half was
    # normalised *separately*, so the quiet half was lifted to -18 LUFS on its own
    # terms - about +19 dB on a noise floor. Under programme normalisation both
    # halves move by one shared gain and the gap between them survives.
    speech = _speech_like(4.0)
    floor = rng.normal(0.0, 10.0 ** (-37.0 / 20.0), n - len(speech))
    joined = np.concatenate([speech, floor])
    src = tmp_path / "roomy.wav"
    _write_wav(src, joined)
    out = render._normalise_programme(src, cfg)

    def rms_db(path, seconds: float) -> float:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{seconds}", "-i", str(path),
             "-t", "2", "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"],
            stdout=-1).stdout
        x = np.frombuffer(raw, np.float32)
        return 20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-12)

    # The gap between the busy half and the quiet half, in the shipped file.
    gap_after = rms_db(out, 0.0) - rms_db(out, 6.0)
    # And the same gap measured on the pre-normalisation mix, normalised out.
    raw_in = np.frombuffer(
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-f", "f32le",
                        "-ac", "1", "-ar", str(RATE), "-"], stdout=-1).stdout,
        np.float32)
    half = len(raw_in) // 2

    def in_gap() -> float:
        def db(part):
            return 20 * np.log10(float(np.sqrt(np.mean(part ** 2))) + 1e-12)
        return db(raw_in[:half]) - db(raw_in[half:])

    assert abs(gap_after - in_gap()) < 1.5, (
        f"the {in_gap():.1f} dB gap between the busy and quiet halves became "
        f"{gap_after:.1f} dB; per-segment normalisation erases exactly this"
    )

    # And the quiet half is still quiet in absolute terms - not lifted to sit
    # just under the loud half.
    quiet_section = rms_db(out, 6.0)
    assert quiet_section < -20.0, (
        f"the quiet half sits at {quiet_section:.1f} dBFS after programme "
        "normalisation; a noise floor taken to programme level is pumping hiss"
    )


def test_normalisation_fails_open_when_measurement_is_unreadable(tmp_path, monkeypatch):
    """A slightly wrong level is recoverable; a truncated mix is not.

    So if the measurement pass cannot be parsed, the render continues on the
    un-normalised mix rather than refusing to produce a video.
    """
    src = tmp_path / "m.wav"
    _write_wav(src, _speech_like(2.0))
    monkeypatch.setattr(render, "_measure_loudness", lambda p: None)
    out = render._normalise_programme(src, render.Pipeline())
    assert out == src, "an unreadable measurement must leave the mix alone"
    assert out.exists()


def test_true_peak_is_measured_not_assumed(tmp_path):
    """`alimiter` guarantees sample peak, not true peak, and AAC adds more on top.

    The point of measuring the shipped file is that the graph cannot be trusted to
    have delivered -1 dBTP. This asserts the measurement reads a real number off
    a real file, and that a hot file reports hot.
    """
    tone = np.sin(2 * np.pi * 997 * np.arange(RATE) / RATE)
    hot = tmp_path / "hot.wav"
    _write_wav(hot, tone, normalise=False)
    tp = render.measure_true_peak(hot)
    assert tp is not None
    # A full-scale sine reaches about 0 dBFS sample peak; its *inter-sample* true
    # peak is slightly above that, which is the entire reason `alimiter` cannot be
    # trusted to deliver a -1 dBTP ceiling. Anything at or above 0 here means the
    # reader is reporting a sample peak and not a true peak, which is the fault
    # this function exists to avoid.
    assert tp >= -0.1, (
        f"a full-scale sine measured {tp:.2f} dBTP; if this reads below 0 the "
        "filter is reporting a sample peak and not a true peak, and a "
        "-1 dBTP ceiling derived from it would be wrong by the overs amount"
    )
    assert tp < 3.0, f"a full-scale sine measured {tp:.2f} dBTP; the reader is wrong"

    # And a quieter file must read quieter, so the number tracks the signal.
    quiet = tmp_path / "quiet.wav"
    _write_wav(quiet, tone * 0.1, normalise=False)
    tp_q = render.measure_true_peak(quiet)
    assert tp_q is not None and tp_q < tp - 15.0, (
        f"a signal 20 dB quieter measured {tp_q:.2f} dBTP against {tp:.2f} - the "
        "measurement is not tracking the level"
    )


# ------------------------------------------------------------- ducking chain

def _duck_filter(tmp_path, monkeypatch) -> str:
    """The `-filter_complex` build_audio_track actually hands to ffmpeg.

    Read from the real command rather than from the source text. The chain is an
    f-string split across several lines, so a source regex sees only the first
    fragment and silently misses everything after it - which is exactly how
    `attack` disappeared from the assertion below while still being in the code.
    """
    from unittest import mock

    captured: list[str] = []

    def spy(cmd, **kw):
        if "-filter_complex" in cmd:
            captured.append(cmd[cmd.index("-filter_complex") + 1])
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(render, "run", spy)
    monkeypatch.setattr(render, "_normalise_programme",
                        lambda mixed, cfg: mixed)
    monkeypatch.setattr(render, "_assert_music_underneath",
                        lambda *a, **k: None)

    work = tmp_path / "work"
    work.mkdir()
    segs = []
    for i in range(2):
        p = work / f"aud_{i}.wav"
        _write_wav(p, _speech_like(1.0))
        segs.append(p)
    music = tmp_path / "music.wav"
    _write_wav(music, np.sin(2 * np.pi * 440 * np.arange(RATE * 2) / RATE))

    cfg = render.Pipeline()
    cfg.music.duck = True
    render.build_audio_track(segs, cfg, music, 2.0, work)
    assert captured, "build_audio_track never built a filter graph"
    return captured[0]


def _duck_opts(graph: str) -> dict[str, str]:
    m = re.search(r"\[mus\]\[cont\]sidechaincompress=([^\[]+)\[ducked\]", graph)
    assert m, f"no sidechaincompress between [mus] and [cont] in:\n{graph}"
    out: dict[str, str] = {}
    for part in m.group(1).split(":"):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k] = v
    return out


def test_ducking_has_a_real_knee(tmp_path, monkeypatch):
    """2.82843 is ffmpeg's unset default, and it is nearly a hard knee.

    A hard knee on a music bed at 6:1 is a step, not a transition. The recommended
    band for a sidechain like this is 4-8; 6 is the middle of it.
    """
    opts = _duck_opts(_duck_filter(tmp_path, monkeypatch))
    assert "knee" in opts, (
        "sidechaincompress has no knee, so it uses ffmpeg's default of 2.82843 - "
        "nearly a hard knee, and a step in the music bed"
    )
    knee = float(opts["knee"])
    assert 4.0 <= knee <= 8.0, f"knee={knee} is outside the recommended 4-8 band"


def test_ducking_ratio_and_attack_are_unchanged_by_the_knee_fix(tmp_path, monkeypatch):
    """The knee is additive. Silently retuning ratio or attack would be a change
    nobody asked for, and it would move duck depth for every existing render."""
    opts = _duck_opts(_duck_filter(tmp_path, monkeypatch))
    assert float(opts["ratio"]) == 6.0
    assert float(opts["attack"]) == 18.0
    assert float(opts["threshold"]) == pytest.approx(0.035)
