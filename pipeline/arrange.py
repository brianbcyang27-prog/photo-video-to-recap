"""Arrangement: give a long piece a shape instead of one repeated bar.

The bed used to be a single 4-bar progression, looped to fill the target. At the
default tempo a 90-second recap is 34 bars and a 10-minute one is 240 - so the
"arrangement" heard at 0:40 is byte-for-byte the same 10 seconds heard at 6:20,
repeated 60 times. Ten minutes of that is not a score, it is a test tone, and
listeners stop hearing the music at all, which is the complaint that prompted
this module.

The fix is to treat the piece as having sections the way a record does: an
opening, a build, a lift, a drop back, and an ending. Each section changes what
is actually playing - which instruments are present, how loud they are, how
full the chord is, how busy the rhythm is - so the listener has something to
notice roughly every 30-45 seconds.

The sections are also tied to the film's length rather than to a fixed clock,
so a 90-second recap gets a compressed version of the same shape instead of an
intro that eats half its own runtime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """One stretch of the piece.

    `weight` is 0.0-1.0 and scales the parts that are always present; the
    booleans turn individual layers on or off. Keeping everything on a single
    continuous scale rather than a set of named presets is what lets the
    sections blend into each other instead of switching abruptly.
    """
    name: str
    bars: int
    drums: float = 1.0
    bass: float = 1.0
    arp: float = 1.0
    pad: float = 1.0
    brightness: float = 1.0


# The shapes below are relative lengths in bars; they are scaled to fit whatever
# target is asked for. `energy` is the single number that moves the arrangement
# from here to there, and each entry only overrides the parts it changes.
_SHAPES: tuple[tuple[str, dict], ...] = (
    # A bare start: pads and bass only, so the film can open on a picture
    # before the rhythm arrives.
    ("open", dict(drums=0.0, arp=0.0, bass=0.55, pad=1.15, brightness=0.7)),
    # The main groove arrives and settles.
    ("groove", dict(drums=1.0, bass=1.0, arp=0.85, pad=1.0)),
    # Lift: everything opens up, brighter and busier.
    ("lift", dict(drums=1.0, bass=1.0, arp=1.0, pad=1.2, brightness=1.35)),
    # Drop back: drums out, arp out. The quiet stretch that keeps a long piece
    # from being exhausting and gives the ear somewhere to rest.
    ("drop", dict(drums=0.0, arp=0.0, bass=0.5, pad=1.1, brightness=0.6)),
    # Back in, brighter than the first time - so the return is recognisably a
    # return rather than a restart.
    ("return", dict(drums=1.0, bass=1.0, arp=1.0, pad=1.1, brightness=1.2)),
    # Closing: drums fade, pad holds.
    ("close", dict(drums=0.55, bass=0.7, arp=0.5, pad=1.25, brightness=0.8)),
)


def plan(n_bars: int, max_bars: int = 32) -> list[Section]:
    """Lay sections across `n_bars`, keeping the open and close fixed.

    The first and last sections are the ones the viewer notices as the film
    starting and stopping, so they are not scaled - they stay short whatever the
    runtime.

    `max_bars` caps how long any one section may run. Without it the middle
    shapes simply stretch to fill: a 10-minute plan came out with 34-bar
    sections and a 30-minute one with 102-bar sections, so the piece got more
    monotonous the longer it got - exactly backwards. Capping the length means
    a long film repeats the middle of the arrangement instead, which is what a
    record does and what keeps something changing to listen to. At the default
    tempo the cap is about 80 seconds.

    A short film is not a long film played fast. At 34 bars the full nine-shape
    plan would give each section four bars, which reads as an exercise changing
    key every four seconds. So the number of sections scales with the length:
    up to about a minute only the opening and the groove are used, and the
    longer forms come in as there is room for them.
    """
    if n_bars <= 0:
        return []

    def build(names: tuple[str, ...], opening: int, closing: int) -> list[Section]:
        by_name = dict(_SHAPES)
        head = _section(names[0], opening, **by_name[names[0]])
        tail = _section(names[-1], closing, **by_name[names[-1]])
        spare = max(1, n_bars - opening - closing)

        if len(names) == 2:
            # No middle shapes named, so the groove carries it. The cap does
            # not apply here: a 90-second recap is open / groove / close, and
            # splitting a single section would leave nothing to change into.
            return [head, _section("groove", spare, **by_name["groove"]), tail]

        # Repeat the middle shapes until the film is full, without any single
        # one exceeding the cap. Repeating the list rather than stretching its
        # entries keeps the arrangement's shape identical at 3 minutes and 30.
        cycle = names[1:-1]
        # Aim for sections of about `per` bars, then take as many as are needed
        # to cover the film. `base` is the floor and `slack` is divided evenly,
        # so every section ends up within one bar of every other and the plan
        # sums to exactly `spare`. Doing it in one pass matters: capping first
        # and patching afterwards left plans 15 bars short, with the shortfall
        # collecting in a short section at the end.
        per = max(1, min(max_bars, spare // len(cycle)))
        count = max(1, min(96, -(-spare // per)))
        names_mid = [cycle[i % len(cycle)] for i in range(count)]
        base = spare // count
        slack = spare - base * count
        counts = [base + (1 if i < slack else 0) for i in range(count)]

        out = [head]
        for name, bars in zip(names_mid, counts):
            out.append(_section(name, bars, **by_name[name]))
        out.append(tail)
        return out

    if n_bars < 48:                      # under ~90s
        return build(("open", "groove", "close"), 2, 4)
    if n_bars < 100:                     # under ~3min
        return build(("open", "groove", "drop", "groove", "close"), 2, 4)
    if n_bars < 170:                     # under ~6min
        return build(("open", "groove", "lift", "drop", "groove", "close"), 2, 4)
    # Ten minutes and up: the full arc, with the middle shape repeated so the
    # lift is heard twice rather than once and then forgotten.
    return build(("open", "groove", "lift", "drop", "return", "drop",
                  "lift", "groove", "close"), 2, 4)


def _section(name: str, bars: int, **kw) -> Section:
    return Section(name=name, bars=max(1, int(bars)),
                   drums=kw.get("drums", 1.0), bass=kw.get("bass", 1.0),
                   arp=kw.get("arp", 1.0), pad=kw.get("pad", 1.0),
                   brightness=kw.get("brightness", 1.0))


def _replace(s: Section, bars: int) -> Section:
    return Section(s.name, bars, s.drums, s.bass, s.arp, s.pad, s.brightness)


def section_at(plan_: list[Section], bar_index: int) -> Section:
    """Which section bar `bar_index` falls in. Clamps past the end."""
    start = 0
    for s in plan_:
        if bar_index < start + s.bars:
            return s
        start += s.bars
    return plan_[-1] if plan_ else Section("groove", 1)
