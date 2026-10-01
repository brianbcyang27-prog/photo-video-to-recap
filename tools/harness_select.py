#!/usr/bin/env python3
"""Exercise chapter fairness without decoding or rendering anything.

The selection logic is where a 14-day trip becomes a 10-minute edit, and it is
the part that was quietly broken twice. Re-rendering the real library to check
it costs an hour; this rebuilds the same chapter shape as synthetic items and
runs the real functions, so a fairness change can be measured in seconds and
compared against the real numbers it has to reproduce.

The shape (chapter sizes, photo/clip mix) is taken from the real library so a
pass here means something for that library rather than for a convenient one.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.analysis import Item  # noqa: E402
from pipeline.config import Analysis, Pipeline  # noqa: E402
from pipeline.util import MediaInfo  # noqa: E402

# From the real library: (usable shots, fraction that are video clips).
REAL_SHAPE = [
    ("Jun 11 Hong Kong",      194, 0.15),
    ("Jun 11 London",          44, 0.10),
    ("Jun 12 Westminster",   1001, 0.11),
    ("Jun 13 Oxford",         479, 0.33),
    ("Jun 14 Cambridge",     1810, 0.27),
    ("Jun 15 Flims",          377, 0.00),
    ("Jun 16 Flims",         1369, 0.18),
    ("Jun 17 Flims",          838, 0.23),
    ("Jun 18 Zermatt",       1014, 0.52),
    ("Jun 19 Zermatt",        858, 0.29),
    ("Jun 20 Thun",           377, 0.14),
    ("Jun 21 Grindelwald",   1379, 0.56),
    ("Jun 22 Gundlischwand",  543, 0.33),
    ("Jun 23 Lauterbrunnen",  700, 0.00),
    ("Jun 24 Lauterbrunnen",  972, 0.58),
    ("Jun 25 Lauterbrunnen",  574, 0.00),
    ("Jun 26 Zurich",         648, 0.00),
    ("Jun 27 Kloten",         185, 0.00),
]
TARGET_CONTENT = 545.0

# How far the richest and thinnest chapters may sit either side of the median
# before this is a bug rather than variation.
#
# Not fitted to make a run pass. The two measurements that bracket it: a quota
# that does nothing starves whole chapters and measures 16x here and 68x on the
# real library; correct code measures 3.0x here and 2.7x on the real library,
# the residue coming from the per-minute diversity cap (bursty days lose far
# more candidates than steady ones) and from the ~30% of the timeline that is
# deliberately left to be filled by score alone. 6x sits clear of both with
# room on each side, and is still tight enough that reintroducing either bug
# fails loudly.
FAIRNESS_LIMIT = 6.0


@dataclass
class FakeStop:
    index: int = 0
    name: str = ""


@dataclass
class FakeChapterPlan:
    chapter: FakeStop
    items: list[Item] = field(default_factory=list)

    @property
    def chapter_index(self) -> int:
        return self.chapter.index


@dataclass
class FakeChapter:
    index: int
    start: float = 0.0


@dataclass
class FakeCtx:
    chapters: list[FakeChapterPlan]
    of_item: dict[int, int] = field(default_factory=dict)
    _by_index: dict[int, FakeChapterPlan] = field(default_factory=dict)

    def chapter_for(self, item: Item) -> FakeChapterPlan | None:
        return self._by_index.get(self.of_item.get(id(item), -1))

    def __post_init__(self):
        self._by_index = {c.chapter.index: c for c in self.chapters}


def _mix(n: int) -> int:
    """splitmix64 - well-spread 64-bit values, so nothing looks like a dupe.

    The obvious constructions are not good enough here: _dedupe drops any pair
    within 10 bits of Hamming distance, so a counter-based hash (or an
    arithmetic sequence like k*7919) makes thousands of unrelated shots look
    identical and silently guts the library. That failure is invisible unless
    the item counts are printed, which is why this harness prints them.
    """
    x = (n + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def build(shape=REAL_SHAPE) -> tuple[FakeCtx, list[Item]]:
    """One item per 'usable shot', each tagged with the chapter it came from."""
    chapters: list[FakeChapterPlan] = []
    items: list[Item] = []
    of_item: dict[int, int] = {}
    for ci, (name, count, clip_frac) in enumerate(shape):
        chapters.append(FakeChapterPlan(FakeStop(ci, name)))
        for k in range(count):
            is_video = (k / max(1, count)) < clip_frac
            info = MediaInfo(path=Path(f"c{ci}_{k}.jpg" if not is_video
                                       else f"c{ci}_{k}.mov"),
                             kind="video" if is_video else "photo",
                             duration=8.0 if is_video else 0.0)
            # Spread within the chapter so the per-minute diversity cap sees a
            # realistic burst pattern rather than one huge pile.
            info.captured = 1_767_225_600.0 + ci * 86_400 + k * 3.7
            it = Item(info=info, kind=info.kind, hash=_mix(ci * 100_000 + k),
                      score=0.4 + 0.5 * ((k * 7) % 11) / 10.0)
            if is_video:
                # A real usable slice. Without start/end, _source_available
                # reports nothing and every clip is discarded before selection.
                it.start, it.end = 0.0, 8.0
            of_item[id(it)] = ci
            items.append(it)
    return FakeCtx(chapters, of_item), items


def measure(chosen: list[Item], ctx: FakeCtx, shape=REAL_SHAPE) -> dict:
    """Screen time per chapter, judged against the material each one holds.

    Measured against the same minimum slot lengths selection charges for, so
    these numbers are comparable with the real EDL's seconds-per-shot.
    """
    from pipeline.select import _quantised
    cfg = Pipeline()
    min_photo = _quantised(cfg.photo_min, None)
    min_video = _quantised(cfg.video_min, None)
    spent: dict[int, float] = {}
    picked: dict[int, int] = {}
    for it in chosen:
        ci = ctx.of_item.get(id(it), -1)
        need = min_video if it.kind == "video" else min_photo
        spent[ci] = spent.get(ci, 0.0) + need
        picked[ci] = picked.get(ci, 0) + 1
    rows = []
    for ci, (name, count, _f) in enumerate(shape):
        rows.append((name, count, picked.get(ci, 0), spent.get(ci, 0.0)))
    return {"rows": rows, "spent": spent, "picked": picked}


def main() -> int:
    from pipeline.select import select_items
    cfg = Pipeline()
    cfg.analysis = Analysis()
    cfg.target_seconds = 600
    ctx, items = build()
    chosen, _notes = select_items(items, ctx, cfg, TARGET_CONTENT, None)
    data = measure(chosen, ctx)

    shape = REAL_SHAPE
    total_shots = sum(s for _n, s, _f in shape)
    # Only chapters holding a real share are judged; the rest sit on the
    # one-shot floor by design, exactly as in the real edit.
    big = [(n, s, p, t) for n, s, p, t in data["rows"]
           if s >= total_shots * 0.02]
    rates = sorted(t / s for _n, s, _p, t in big)
    med = rates[len(rates) // 2]

    print(f"  target {TARGET_CONTENT:.0f}s of content, "
          f"{len(chosen)} items chosen, "
          f"{sum(t for _n,_s,_p,t in data['rows']):.0f}s allocated\n")
    print(f"  {'chapter':<24} {'shots':>6} {'picked':>7} {'screen':>7} {'s/shot':>8}")
    for name, shots, picked, screen in data["rows"]:
        flag = "" if shots >= total_shots * 0.02 else "  (floor)"
        print(f"  {name:<24} {shots:>6} {picked:>7} {screen:>7.1f} "
              f"{screen/max(1,shots):>8.3f}{flag}")
    print(f"\n  median {med:.3f} s/shot, "
          f"richest {rates[-1]/med:.2f}x, thinnest {med/max(rates[0],1e-9):.2f}x")
    worst = max(rates[-1] / med, med / max(rates[0], 1e-9))
    print(f"  fairness limit {FAIRNESS_LIMIT:.2f}x: "
          f"{'PASS' if worst <= FAIRNESS_LIMIT else '*** FAIL ***'}")
    print("  (a no-op quota starves whole chapters and measures 16x+; the")
    print("   real 14-day library measures 2.7x with correct code, 68x broken)")
    return 0 if worst <= FAIRNESS_LIMIT else 1


if __name__ == "__main__":
    raise SystemExit(main())
