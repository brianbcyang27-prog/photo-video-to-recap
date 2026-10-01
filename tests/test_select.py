from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import select
from pipeline.analysis import Item
from pipeline.config import Analysis, Pipeline
from pipeline.util import MediaInfo


def make_item(name: str, *, kind: str = "photo", score: float = 1.0,
              start: float = 0.0, end: float = 3.0, hash_: int = 0,
              ts: float | None = None) -> Item:
    """A minimal Item carrying only what the functions under test read."""
    info = MediaInfo(path=Path(name), kind=kind, width=1920, height=1080,
                     duration=10.0, fps=30.0, has_audio=False)
    if ts is None:
        ts = 1_700_000_000.0
    info.captured = ts
    item = Item(info=info, kind=kind, start=start, end=end, score=score, hash=hash_)
    return item


def cfg(**analysis) -> Pipeline:
    return Pipeline(target_seconds=90, analysis=Analysis(**analysis))


# ---------------------------------------------------------------- dedupe

def test_dedupe_drops_near_identical_hashes():
    """Burst shots share almost every pixel, so the hashes are a few bits apart."""
    a = make_item("a.jpg", hash_=0b10101010101010101010101010101010)
    b = make_item("b.jpg", hash_=0b10101010101010101010101010101011)   # 2 bits
    c = make_item("c.jpg", hash_=0b11111111000000001111111100000000)

    kept, dropped = select._dedupe([a, b, c], cfg(dedupe_threshold=10))

    assert dropped == 1
    assert [i.label for i in kept] == ["a.jpg", "c.jpg"]


def test_dedupe_keeps_genuinely_different_shots():
    """Completely different content is never a duplicate.

    Hashes here avoid zero, which the deduper treats as "never computed".
    """
    kept, dropped = select._dedupe(
        [make_item("a.jpg", hash_=0b00001111), make_item("b.jpg", hash_=0xFFFF0000)],
        cfg(dedupe_threshold=10),
    )
    assert dropped == 0
    assert len(kept) == 2


def test_dedupe_ignores_rejected_and_hashed_zero():
    """A zero hash means "not computed", and must never be treated as a match."""
    rejected = make_item("bad.jpg", hash_=1234)
    rejected.reject = "too dark"
    unhashed = make_item("nohash.jpg", hash_=0)

    kept, dropped = select._dedupe([rejected, unhashed], cfg())

    assert kept == []
    assert dropped == 0


def test_dedupe_threshold_controls_how_close_is_too_close():
    # Six bits apart: duplicates at a threshold of 6, distinct at 5.
    a = make_item("a.jpg", hash_=0b000111)
    b = make_item("b.jpg", hash_=0b111000)
    assert select._dedupe([a, b], cfg(dedupe_threshold=6))[1] == 1
    assert select._dedupe([a, b], cfg(dedupe_threshold=5))[1] == 0


# ------------------------------------------------------- temporal spread

def test_temporal_spread_caps_bursts():
    """Ten shots inside one minute must not all reach the timeline."""
    base = 1_700_000_000.0
    items = [make_item(f"p{i}.jpg", ts=base + i) for i in range(10)]
    picked = select._temporal_spread(items, cfg(max_per_minute=2))
    assert 0 < len(picked) < 10


def test_temporal_spread_respects_an_explicit_cap():
    items = [make_item(f"p{i}.jpg", ts=1_700_000_000.0 + i) for i in range(10)]
    assert len(select._temporal_spread(items, cfg(), cap=99)) == 10
    assert len(select._temporal_spread(items, cfg(), cap=3)) == 3


# ------------------------------------------------------------- quantising

def test_quantised_rounds_up_to_whole_beats():
    """Durations round up, so a shot is never cut off mid-beat."""
    spb = 0.5
    assert select._quantised(2.10, spb) == pytest.approx(2.5)
    assert select._quantised(2.01, spb) == pytest.approx(2.5)
    assert select._quantised(2.00, spb) == pytest.approx(2.0)


def test_quantised_without_a_tempo_is_a_no_op():
    assert select._quantised(2.37, None) == pytest.approx(2.37)
    assert select._quantised(2.37, 0.0) == pytest.approx(2.37)


def test_quantised_never_shortens_and_never_overruns_a_beat():
    spb = 0.45
    for raw in (1.0, 1.1, 2.24, 3.99, 7.5):
        got = select._quantised(raw, spb)
        assert got >= raw - 1e-9, "rounding up must not shorten a shot"
        assert got < raw + spb, "must not overshoot by a whole beat"
