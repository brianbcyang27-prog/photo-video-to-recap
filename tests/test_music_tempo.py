"""Tempo detection: the half-time octave trap.

A supplied track is beat-locked, not resampled, so the tempo `detect_beats`
reports *is* the tempo the cut list is built on. Shot lengths are quantised to
`60/bpm`, so a tempo that is wrong by a factor of two is not a rounding
detail: every shot is twice as long as it should be and every cut lands
between beats instead of on them.

The failure mode is octave confusion, and it is not hypothetical. On Avicii's
"The Nights" (126 BPM, verified against published tempo data) `_estimate_tempo`
returned 63.02, and the mechanism is specific:

1. The onset envelope is autocorrelated and scored against a log-normal prior
   centred at 120 BPM. That correctly prefers 126 over 63 - measured
   0.4890 vs 0.3130.
2. The octave check that runs afterwards compares **raw** autocorrelation
   against a 6% margin, ignoring the prior that produced the candidate.
3. On half-time house the kick lands on beats 1 and 3, so the envelope
   genuinely repeats every two beats. Raw ac at the half-tempo lag came out
   0.5332 against 0.4905 at the full-tempo lag - a ratio of 1.087 - which
   cleared the 1.06 margin and overrode the prior.

The check therefore used a different objective from the selection it was
correcting, and could only ever pull the answer toward whatever the raw
autocorrelation liked. The fix is to score both candidates the same way.

These tests build the envelope directly rather than synthesising audio, because
routing a signal through spectral flux smears a two-beat alternation and the
window is narrow: at amplitude 0.50 the ratio drops to 0.955 and the bug
disappears. The envelopes here are the same structure the detector sees, with
the beat amplitude alternation the real track implies.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

# `from pipeline import music` is fine here: music re-exports functions, not
# itself, but _estimate_tempo is private so import the module by name.
music = importlib.import_module("pipeline.music")

SR = 48000
HOP = 256
FPS = SR / HOP


def _half_time_envelope(bpm: float = 126.0, seconds: float = 45.0,
                        loud: float = 1.0, quiet: float = 0.40) -> np.ndarray:
    """Onset envelope of a half-time 4/4 pattern at `bpm`.

    Beats 1 and 3 (indices 0 and 2) get `loud`, beats 2 and 4 get `quiet`, so
    the envelope repeats every two beats but not every beat. At quiet=0.40 the
    half-tempo autocorrelation peak is ~11% above the full-tempo one, which
    is what real half-time house measures.
    """
    n = int(seconds * FPS)
    env = np.zeros(n)
    beat = (60.0 / bpm) * FPS
    k = 0
    while k * beat < n:
        env[int(k * beat)] += loud if (k % 4) in (0, 2) else quiet
        k += 1
    return env


def test_half_time_groove_is_not_reported_at_half_tempo() -> None:
    """The reported tempo must be the full tempo, not the groove's half."""
    got = music._estimate_tempo(_half_time_envelope(), SR, HOP)
    assert abs(got - 126.0) < 4.0, (
        f"half-time groove reported as {got:.2f} BPM; expected ~126. "
        "Shot lengths are quantised to 60/bpm, so the octave error makes every "
        "shot twice as long and puts every cut between beats.")


def test_prior_still_breaks_a_tie_the_raw_correlation_would_not() -> None:
    """The fix is prior-awareness, not simply dropping the octave check.

    A near-equal alternation is genuinely ambiguous in raw autocorrelation
    (the two peaks are within 4%), and here the prior is the only thing that
    can prefer the full tempo. If the octave check still switched on raw ac,
    this would report the half tempo.
    """
    got = music._estimate_tempo(_half_time_envelope(quiet=0.55), SR, HOP)
    assert abs(got - 126.0) < 4.0, (
        f"ambiguous alternation resolved to {got:.2f} BPM; the 120 BPM prior "
        "should decide when raw autocorrelation is within a few percent.")


def test_genuinely_slow_track_still_reports_its_own_tempo() -> None:
    """Guard the opposite case: the fix must not over-correct upward.

    A kick on beat 1 only, with no off-beat material at all, is a real 63 BPM
    track rather than a half-time 126 BPM one. The raw autocorrelation strongly
    prefers 63 there, so that is what must be reported - doubling it would be a
    different bug, and a fix that always prefers the faster octave would cause
    exactly that.
    """
    n = int(45.0 * FPS)
    env = np.zeros(n)
    beat = (60.0 / 63.0) * FPS
    k = 0
    while k * beat < n:
        env[int(k * beat)] += 1.0
        k += 1
    got = music._estimate_tempo(env, SR, HOP)
    assert abs(got - 63.0) < 3.0, (
        f"a genuine 63 BPM pulse reported as {got:.2f} BPM; the octave check "
        "must not promote a clearly slow track to double tempo.")


def test_octave_check_scores_candidates_the_same_way() -> None:
    """The structural invariant: one objective, used for both decisions.

    Selecting a tempo with a prior and then re-judging it on raw
    autocorrelation guarantees the second stage can undo the first. Both
    stages must use the prior-weighted score.
    """
    env = _half_time_envelope()
    centred = env - env.mean()
    ac = np.correlate(centred, centred, mode="full")[len(env) - 1:]
    ac /= ac[0] + 1e-9
    fps = FPS

    def weighted(bpm: float) -> float:
        lag = 60.0 * fps / bpm
        lo = int(max(1, lag - 1))
        hi = int(min(len(ac) - 1, lag + 1)) + 1
        raw = float(ac[lo:hi].max())
        prior = float(np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.9) ** 2))
        return raw * prior

    half, full = 63.0, 126.0
    assert weighted(half) < weighted(full), (
        "fixture no longer discriminates: the prior must prefer 126 for this "
        "envelope, otherwise the test proves nothing")
    assert music._estimate_tempo(env, SR, HOP) == pytest.approx(126.0, abs=4.0)
