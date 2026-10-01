"""Trip context - the map of the whole journey.

Everything downstream (chapter titles, pacing, how much screen time each day
earns) reads from this, so the edit follows the real trip rather than the
order files happened to be written in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import math
from pathlib import Path

from .analysis import Item
from .config import Pipeline
from .geo import (
    Chapter, Geocoder, Stop, build_chapters, cluster_stops, light_score,
    name_stops,
)
from .ingest import Library
from .util import log, human_duration


@dataclass
class ChapterPlan:
    """A chapter plus how much screen time it has earned."""

    chapter: Chapter
    items: list[Item] = field(default_factory=list)
    weight: float = 0.0        # relative share of the timeline
    seconds: float = 0.0       # assigned screen time
    title: str = ""
    # Where this chapter happened. Kept apart from the title so the card can
    # show "Mon Jun 15" up top and "Zermatt" underneath.
    place_label: str = ""

    @property
    def label(self) -> str:
        return self.title or self.chapter.day_label


@dataclass
class TripContext:
    stops: list[Stop] = field(default_factory=list)
    chapters: list[ChapterPlan] = field(default_factory=list)
    has_gps: bool = False
    day_count: int = 0
    span_seconds: float = 0.0
    geocoded: bool = False

    def chapter_for(self, item: Item) -> ChapterPlan | None:
        ts = item.info.captured
        if not ts:
            return None
        for cp in self.chapters:
            if cp.chapter.start - 1 <= ts <= cp.chapter.end + 1:
                return cp
        return self.chapters[0] if self.chapters else None

    def stats(self) -> dict:
        return {
            "days": self.day_count,
            "stops": len([s for s in self.stops if s.has_gps()]),
            "chapters": len(self.chapters),
            "span": self.span_seconds,
            "geocoded": self.geocoded,
        }


def _entries(lib: Library) -> list[tuple[float, float, float]]:
    out = []
    for info in lib.items:
        if info.captured:
            out.append((info.latitude, info.longitude, info.captured))
    return out


def _offsets(lib: Library) -> dict[float, int]:
    """ts -> utc_offset, so day boundaries follow the camera's own clock."""
    return {i.captured: i.utc_offset
            for i in lib.items if i.captured and i.utc_offset}


def local_day(ts: float, offsets: dict[float, int]) -> date:
    return datetime.fromtimestamp(
        ts + offsets.get(ts, 0), tz=timezone.utc).date()


def build_context(lib: Library, cfg: Pipeline, cache_dir: Path,
                  *, geocode: bool = True) -> TripContext:
    entries = _entries(lib)
    offsets = _offsets(lib)
    ctx = TripContext()

    if entries:
        ctx.span_seconds = max(e[2] for e in entries) - min(e[2] for e in entries)
        ctx.day_count = len({local_day(e[2], offsets) for e in entries})

    ctx.has_gps = any(e[0] or e[1] for e in entries)
    if not ctx.has_gps:
        log("no GPS in any file - will organise by time only", level="warn")

    ctx.stops = cluster_stops(entries) if entries else []
    if geocode and ctx.has_gps:
        geo = Geocoder(cache_dir)
        name_stops(ctx.stops, geo)
        ctx.geocoded = any(s.name for s in ctx.stops)
        if ctx.geocoded:
            log(f"resolved {sum(1 for s in ctx.stops if s.name)}/{len(ctx.stops)} "
                f"place names")
        else:
            log("place names unavailable (offline?) - using coordinates",
                level="warn")

    chapters = (build_chapters(entries, ctx.stops, offsets=offsets,
                               split_on_move=cfg.chapter_on_move)
               if entries else [])
    ctx.chapters = [ChapterPlan(chapter=c) for c in chapters]
    # Date-led editing: the card leads with the day, and the place it was spent
    # in becomes the subtitle. A chapter that spans two towns names both, since
    # splitting on distance is off in this mode.
    for cp in ctx.chapters:
        cp.title = cp.chapter.day_label
        if cp.chapter.stop:
            cp.place_label = cp.chapter.stop.label

    # Attach items to chapters.
    for item in _items_of(lib, cfg):
        cp = ctx.chapter_for(item)
        if cp:
            cp.items.append(item)

    if not any(cp.items for cp in ctx.chapters) and ctx.chapters:
        # Everything shares one timestamp or has none - keep it simple.
        ctx.chapters[0].items = list(_items_of(lib, cfg))

    _assign_weights(ctx)
    return ctx


def _items_of(lib: Library, cfg: Pipeline) -> list[Item]:
    """Placeholder list; real Items are attached by the caller after analysis.

    This keeps build_context usable before analysis has run (for titles) and
    after (for weighting). See attach_items().
    """
    return []


def attach_items(ctx: TripContext, items: list[Item]) -> None:
    for cp in ctx.chapters:
        cp.items = []
    for item in items:
        cp = ctx.chapter_for(item)
        if cp:
            cp.items.append(item)
    if not any(cp.items for cp in ctx.chapters) and ctx.chapters:
        ctx.chapters[0].items = list(items)
    _assign_weights(ctx)


def _assign_weights(ctx: TripContext) -> None:
    """Weight chapters by how much usable material they actually contain.

    Dense days earn more screen time, but a quiet day never disappears
    entirely - we floor the share so a single-beat day still gets a look.
    """
    total = sum(len(cp.items) for cp in ctx.chapters)
    if total <= 0:
        for cp in ctx.chapters:
            cp.weight = 1.0 if len(ctx.chapters) == 1 else 0.0
        return

    raw = [len(cp.items) for cp in ctx.chapters]
    floor = 0.55 if len(ctx.chapters) > 1 else 0.0
    boosted = [v + floor * (total / max(1, len(ctx.chapters))) for v in raw]
    scale = sum(boosted) or 1.0
    for cp, w in zip(ctx.chapters, boosted):
        cp.weight = w / scale

    # How many usable shots per chapter, for the report.
    log(f"trip context: {ctx.day_count} days, "
        f"{len([s for s in ctx.stops if s.has_gps()])} located stops, "
        f"{len(ctx.chapters)} chapters over {human_duration(ctx.span_seconds)}")


def allocate_seconds(ctx: TripContext, items: list[Item], target: float,
                     *, title_seconds: float = 0.0) -> None:
    """Hand out screen time per chapter, then per item.

    Title cards are subtracted up front so the content lands exactly on target.
    """
    active = [cp for cp in ctx.chapters if cp.items]
    if not active:
        return

    content = max(5.0, target - title_seconds * len(active))
    for cp in active:
        cp.seconds = content * cp.weight

    log("screen time: " + ", ".join(
        f"{cp.label} {cp.seconds:.0f}s/{len(cp.items)}" for cp in active))


def light_for(item: Item) -> float:
    """Light quality for one item, using its GPS when available."""
    ts = item.info.captured
    if not ts:
        return 0.6
    lat, lon = item.info.latitude, item.info.longitude
    if not (lat or lon):
        # No GPS: approximate from latitude and day of year.
        day = datetime.fromtimestamp(ts, tz=timezone.utc)
        lat = max(-66.0, min(66.0, 15.0 * _daylight_offset(day.timetuple().tm_yday)))
    try:
        return light_score(ts, lat, lon)
    except Exception:
        return 0.6


def _daylight_offset(day_of_year: int) -> float:
    """Rough latitude proxy from day of year: +1 midsummer, -1 midwinter.

    Only used when a shot has no GPS at all, where a seasonal guess is a much
    better guess than assuming the equator.
    """
    return math.cos((day_of_year - 172) / 365.25 * 2 * math.pi)
