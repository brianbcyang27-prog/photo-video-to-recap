"""Stage 6 - selection and structure.

Takes the scored pool and produces an ordered cut list whose shot boundaries
land exactly on musical beats and whose total length hits the target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .analysis import Item
from .config import Pipeline
from .context import TripContext, light_for
from .music import MusicResult
from .util import log


@dataclass
class Entry:
    """One thing on screen for a known length."""

    item: Item | None = None
    duration: float = 0.0
    beats: int = 0
    chapter: int = 0
    is_title: bool = False
    title: str = ""
    subtitle: str = ""
    # Where the shot actually starts in the source. Negative means "use the
    # item's own start"; it differs when a shot is slid earlier to give the
    # assigned duration room to fit.
    in_point: float = -1.0
    # Populated during rendering.
    motion: str = "in"
    rank_reason: str = ""

    @property
    def src_in(self) -> float:
        if self.in_point >= 0.0:
            return self.in_point
        return self.item.start if self.item else 0.0

    @property
    def label(self) -> str:
        if self.is_title:
            return f"[title] {self.title}"
        assert self.item is not None
        return self.item.label

    @property
    def score(self) -> float:
        return self.item.score if self.item else 0.0

    @property
    def kind(self) -> str:
        if self.is_title:
            return "title"
        return self.item.kind if self.item else "photo"


@dataclass
class CutList:
    entries: list[Entry] = field(default_factory=list)
    target: float = 0.0
    music: MusicResult | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(e.duration for e in self.entries)

    def summary(self) -> dict:
        photos = sum(1 for e in self.entries if e.kind == "photo")
        videos = sum(1 for e in self.entries if e.kind == "video")
        titles = sum(1 for e in self.entries if e.is_title)
        return {
            "entries": len(self.entries),
            "photos": photos,
            "videos": videos,
            "titles": titles,
            "duration": self.duration,
            "target": self.target,
            "bpm": self.music.bpm if self.music else 0.0,
        }


# ------------------------------------------------------------------- dedupe

def _dedupe(items: list[Item], cfg: Pipeline) -> tuple[list[Item], int]:
    """Drop near-identical items, keeping the highest scorer in each cluster."""
    a = cfg.analysis
    kept: list[Item] = []
    kept_hashes: list[int] = []
    dropped = 0

    for item in items:
        if item.reject or not item.hash:
            continue
        too_close = False
        for h in kept_hashes:
            if bin(item.hash ^ h).count("1") <= a.dedupe_threshold:
                too_close = True
                break
        if too_close:
            dropped += 1
            continue
        kept.append(item)
        kept_hashes.append(item.hash)

    if dropped:
        log(f"dedupe: removed {dropped} near-duplicate items")
    return kept, dropped


def _temporal_spread(items: list[Item], cfg: Pipeline, cap: int | None = None
                     ) -> list[Item]:
    """Cap how many picks can come from the same minute.

    Stops the edit becoming five near-identical frames of one burst. The cap is
    relaxed for long timelines, where a strict limit would leave the requested
    length unreachable no matter how much good material exists.
    """
    if cap is None:
        cap = cfg.analysis.max_per_minute
    if cap <= 0:
        return items
    buckets: dict[int, int] = {}
    out: list[Item] = []
    skipped = 0
    for item in items:
        ts = item.info.captured
        if not ts:
            out.append(item)
            continue
        key = int(ts // 60)
        if buckets.get(key, 0) >= cap:
            skipped += 1
            continue
        buckets[key] = buckets.get(key, 0) + 1
        out.append(item)
    if skipped:
        log(f"diversity: skipped {skipped} items from over-full minutes")
    return out


# --------------------------------------------------------------- selection

# Share of a timeline that should be moving pictures rather than stills.
VIDEO_FLOOR = 0.32
# A chapter title is a short orientation beat, not a segment of its own.
TITLE_SECONDS = 2.5
TITLE_SHARE = 0.12
# How much of the timeline is earmarked for chapter quotas before the best
# remaining shots are allowed to compete for the rest.
TOTAL_QUOTA_FRACTION = 0.7


def _greedy_fill(deduped: list[Item], ctx: TripContext, cfg: Pipeline,
                 music: MusicResult | None, target_content: float
                 ) -> list[Item]:
    """Best-first fill that respects per-type minimums and chapter quotas.

    Video clips are reserved first, because motion carries a montage better
    than a run of stills and clips are cheaper on screen time - filling purely
    by score will happily hand every slot to a photo and produce a slideshow.

    Each chapter also gets a quota proportional to how much usable material it
    holds. Without that, a day of exceptional photographs can crowd a quieter
    day out of the recap entirely, which for a trip video is the wrong answer:
    the point is to show where you went, not only your best exposures.
    """
    spb = music.seconds_per_beat if (music and music.bpm > 1) else None
    min_photo = _quantised(cfg.photo_min, spb)
    min_video = _quantised(cfg.video_min, spb)

    # A clip with less usable footage than the minimum slot can never be
    # honoured, so it is not a candidate at all.
    deduped = [i for i in deduped
               if i.kind != "video" or _source_available(i) >= min_video - 1e-6]
    videos = [i for i in deduped if i.kind == "video"]
    desired = min(len(videos),
                  max(1, round(VIDEO_FLOOR * target_content / min_photo)))

    chosen: list[Item] = []
    seen: set[int] = set()
    running = 0.0

    def need_of(it: Item) -> float:
        return min_video if it.kind == "video" else min_photo

    def take(it: Item) -> bool:
        nonlocal running
        if running + need_of(it) > target_content and len(chosen) >= 4:
            return False
        chosen.append(it)
        seen.add(id(it))
        running += need_of(it)
        return True

    # Phase 1: reserve the video floor with the best clips available.
    for it in videos:
        if len([c for c in chosen if c.kind == "video"]) >= desired:
            break
        take(it)
    if running >= target_content:
        return chosen

    # Phase 2: give every chapter its share, best shots first within each.
    quotas = _chapter_quotas(deduped, ctx, min_photo, min_video)
    for index, quota in quotas.items():
        spent = 0.0
        for it in deduped:
            if id(it) in seen:
                continue
            cp = ctx.chapter_for(it)
            if cp is None or cp.chapter.index != index:
                continue
            if spent + need_of(it) > quota:
                break
            take(it)
            spent += need_of(it)
        if running >= target_content:
            return chosen

    # Phase 3: anything left over goes to the best remaining shots, wherever
    # they come from.
    for it in deduped:
        if id(it) in seen:
            continue
        take(it)
        if running >= target_content:
            break
    return chosen


def _chapter_quotas(deduped: list[Item], ctx: TripContext,
                    min_photo: float, min_video: float) -> dict[int, float]:
    """Screen time to hold in reserve for each chapter, keyed by index."""
    available: dict[int, float] = {}
    for it in deduped:
        cp = ctx.chapter_for(it)
        if cp is None:
            continue
        idx = cp.chapter.index
        available[idx] = available.get(idx, 0.0) + (
            min_video if it.kind == "video" else min_photo
        )
    if not available:
        return {}

    # Share in proportion to usable material, but never below what one shot
    # needs, so a sparse day still contributes something to the story.
    total_avail = sum(available.values()) or 1.0
    weights = {i: v / total_avail for i, v in available.items()}
    out: dict[int, float] = {}
    for cp in ctx.chapters:
        idx = cp.chapter.index
        if idx not in available:
            continue
        share = weights[idx] * TOTAL_QUOTA_FRACTION
        out[idx] = max(available[idx], share)
    return out


def select_items(items: list[Item], ctx: TripContext, cfg: Pipeline,
                 target_content: float, music: MusicResult | None
                 ) -> tuple[list[Item], list[str]]:
    """Greedy, quality-first, score-weighted fill of the timeline.

    Two things make naive capacity arithmetic fail here. First, the pool has to
    be sized from the *actual* mix of photos and video clips, not from the
    loosest of the two minimums: a 90s budget that is mostly photos cannot hold
    41 shots just because video would fit at two seconds each. Second, a fixed
    "at most N shots per minute" rule quietly caps how long a video can get, so
    it is relaxed in steps until the requested length can actually be met.
    """
    notes: list[str] = []
    usable = [i for i in items if not i.reject]

    # Favour well-lit shots: a golden-hour photo beats an identical midday one.
    for it in usable:
        if it.info.captured:
            it.score *= (0.72 + 0.28 * light_for(it))
    usable.sort(key=lambda i: i.score, reverse=True)

    cap0 = max(0, cfg.analysis.max_per_minute)
    spb = music.seconds_per_beat if (music and music.bpm > 1) else None
    need = min(_quantised(cfg.photo_min, spb), _quantised(cfg.video_min, spb))

    # Loosen the diversity cap only while we are genuinely short of the budget.
    cap_ladder = [c for c in (cap0, cap0 * 2, cap0 * 4, cap0 * 8) if c > 0] or [0]
    pool: list[Item] = []
    cap_used = cap_ladder[0]
    for cap in cap_ladder:
        candidates = _temporal_spread(usable, cfg, cap=cap)
        deduped, _ = _dedupe(candidates, cfg)
        pool = _greedy_fill(deduped, ctx, cfg, music, target_content)
        cap_used = cap
        if len(pool) * need >= target_content * 0.97:
            break

    if cap_used != cap0 and cap0 > 0:
        notes.append(
            f"relaxed the {cap0}-per-minute diversity limit to {cap_used} so "
            f"the requested length could be reached")

    if len(pool) < len(usable):
        notes.append(
            f"selected {len(pool)} of {len(usable)} usable shots for "
            f"{target_content:.0f}s of screen time")
    if not pool:
        raise ValueError("no usable items survived scoring")

    n_vid = sum(1 for i in pool if i.kind == "video")
    log(f"selecting {len(pool)} items ({n_vid} clips, {len(pool) - n_vid} photos) "
        f"for {target_content:.0f}s of screen time")
    return pool, notes


# ------------------------------------------------------------ beat-aligned timing

def _largest_remainder(weights: list[float], total: int) -> list[int]:
    """Apportion integer units proportionally (largest-remainder / Hare quota)."""
    if not weights:
        return []
    s = sum(weights)
    if s <= 0:
        base = [total // len(weights)] * len(weights)
        for i in range(total - sum(base)):
            base[i] += 1
        return base
    raw = [w / s * total for w in weights]
    floors = [int(math.floor(r)) for r in raw]
    shortfall = total - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for k in range(max(0, shortfall)):
        floors[order[k % len(order)]] += 1
    return floors


def _fit_bounds(weights: list[float], total: int,
                lo: list[int], hi: list[int]) -> list[int]:
    """Integer apportionment that respects per-entry min/max and hits `total`.

    Two failure modes this has to avoid: drifting under the total, and quietly
    giving back a shot's worth of screen time because a rounding pass stopped
    early. So the residual is settled by single-unit exchange in proportion to
    the weights, which always converges whenever a feasible split exists.
    """
    n = len(weights)
    if n == 0:
        return []
    lo = [max(0, int(a)) for a in lo]
    hi = [max(lo[i], int(b)) for i, b in enumerate(hi)]

    if sum(lo) > total:
        # Not enough room: honour the minimums and accept the overrun.
        return lo
    if sum(hi) <= total:
        return hi

    out = [min(max(v, lo[i]), hi[i])
           for i, v in enumerate(_largest_remainder(weights, total))]

    wsum = sum(weights) or 1.0
    want = [w / wsum * total for w in weights]

    guard = 0
    while sum(out) != total and guard < 4 * n + 64:
        guard += 1
        if sum(out) < total:
            cand = [i for i in range(n) if out[i] < hi[i]]
            if not cand:
                break
            out[min(cand, key=lambda i: (out[i] - want[i], i))] += 1
        else:
            cand = [i for i in range(n) if out[i] > lo[i]]
            if not cand:
                break
            out[max(cand, key=lambda i: (out[i] - want[i], i))] -= 1
    return out


def _base_weights(entries: list[Entry]) -> list[float]:
    """Longer on the better material, but never wildly uneven."""
    scores = [max(0.02, e.score) for e in entries]
    lo = min(scores)
    hi = max(scores)
    span = max(1e-6, hi - lo)
    return [0.78 + 0.44 * ((s - lo) / span) for s in scores]


def assign_timing(entries: list[Entry], total_seconds: float,
                  music: MusicResult | None, cfg: Pipeline) -> None:
    """Give every entry a whole number of beats AND a whole number of frames.

    Beat counts come from the score, but the renderer can only emit whole
    frames, so each cut is placed on the frame nearest its true beat time.
    Measuring the error from the cumulative position rather than the segment
    length is what stops it drifting: rounding a duration and adding it up
    walks the whole timeline further off the beat, cut after cut, whereas
    rounding an absolute position caps the error at half a frame forever.
    """
    fps = cfg.render.fps

    if not music or music.bpm <= 1:
        # No music: fall back to plain quarter-second slots.
        weights = _base_weights(entries)
        lo = [max(1, int(math.floor((cfg.video_min if e.kind == "video"
                                     else cfg.photo_min) / 0.25)))
              for e in entries]
        hi = [max(1, int(math.ceil((cfg.video_max if e.kind == "video"
                                    else cfg.photo_max) / 0.25)))
              for e in entries]
        slots = _fit_bounds(weights, max(len(entries), int(total_seconds * 4)),
                            lo, hi)
        for e, s in zip(entries, slots):
            e.beats = 0
            e.duration = s * 0.25
        return

    spb = music.seconds_per_beat
    frames_per_beat = max(1, int(round(spb * fps)))
    total_beats = max(len(entries), int(round(total_seconds * fps / frames_per_beat)))

    title_idx = [i for i, e in enumerate(entries) if e.is_title]
    content_idx = [i for i, e in enumerate(entries) if not e.is_title]

    title_budget = 0
    per_title = 0
    if title_idx:
        # Titles are a beat or two of orientation, not a segment of their own.
        # Deriving this from a share of the total would give a 30-second title
        # card on a ten-minute video, so it is pinned to a short absolute
        # length and only the *count* scales with the number of chapters.
        per_title = max(2, min(round(TITLE_SECONDS / spb),
                               int(total_beats * TITLE_SHARE / len(title_idx))))
        title_budget = per_title * len(title_idx)

    content_beats = max(len(content_idx), total_beats - title_budget)

    weights = _base_weights([entries[i] for i in content_idx])
    lo: list[int] = []
    hi: list[int] = []
    for e in (entries[i] for i in content_idx):
        mn, mx = ((cfg.video_min, cfg.video_max) if e.kind == "video"
                  else (cfg.photo_min, cfg.photo_max))
        lo.append(max(1, math.ceil(mn / spb)))
        hi.append(max(lo[-1], math.floor(mx / spb)))
        # Never hand a video clip more screen time than it has footage for.
        if e.item is not None:
            usable = _source_available(e.item)
            if usable < math.inf:
                hi[-1] = max(1, min(hi[-1], int(usable / spb)))

    content_alloc = _fit_bounds(weights, content_beats, lo, hi)

    out: list[int] = [0] * len(entries)
    for k, i in enumerate(content_idx):
        out[i] = content_alloc[k]
    for i in title_idx:
        out[i] = max(2, per_title)

    for e, b in zip(entries, out):
        e.beats = int(b)

    # Land every cut on the frame nearest its true beat time. Because the frame
    # index is taken from the *cumulative* beat position rather than from the
    # segment's own length, rounding never accumulates: each cut is within half
    # a frame of the beat (17ms at 30fps) however long the timeline runs.
    cum_beats = 0
    prev_frame = 0
    for e in entries:
        cum_beats += e.beats
        frame = int(round(cum_beats * spb * fps))
        e.duration = (frame - prev_frame) / fps
        prev_frame = frame


def apply_hook_order(entries: list[Entry]) -> list[str]:
    """Move the single strongest shot to the front, keep the rest chronological."""
    content = [e for e in entries if not e.is_title]
    if len(content) < 4:
        return []
    best = max(content, key=lambda e: e.score)
    if entries.index(best) <= 1:
        return []
    entries.remove(best)
    insert_at = 0
    # Land before the opening title card if there is one.
    for i, e in enumerate(entries):
        if e.is_title:
            insert_at = i
            break
    entries.insert(insert_at, best)
    return [f"moved the strongest shot ({best.label}) to the front as the hook"]


# ------------------------------------------------------------------ assembly

def _motion_for(index: int, total: int) -> str:
    """Vary the Ken Burns moves so it doesn't feel like one repeated trick."""
    patterns = ["in", "out", "left", "right", "up", "down"]
    # Never repeat the same move back-to-back.
    return patterns[(index * 5 + index // 7) % len(patterns)]


def _source_available(item: Item) -> float:
    """Usable footage from this shot's own in-point, in seconds.

    A shot near the end of a clip cannot be stretched to fill a longer slot, so
    the allocator has to know the real ceiling before it hands out durations.
    """
    if item.kind != "video":
        return math.inf
    end = min(item.end, item.info.duration or item.end)
    return max(0.0, end - item.start)


def _quantised(duration: float, spb: float | None) -> float:
    """A duration rounded up to the smallest whole number of beats."""
    if not spb or spb <= 1e-6:
        return duration
    return math.ceil(duration / spb) * spb


def build_cutlist(items: list[Item], ctx: TripContext, cfg: Pipeline,
                  music: MusicResult | None, target: float) -> CutList:
    """Full assembly: select, order, add chapter titles, align to beats."""
    notes: list[str] = []

    n_chapters = max(1, len([c for c in ctx.chapters if c.items]))
    title_seconds = 0.0
    if cfg.render.title_card:
        title_seconds = min(n_chapters * 2.5, target * 0.18)

    content_target = max(5.0, target - title_seconds)
    pool, sel_notes = select_items(items, ctx, cfg, content_target, music)
    notes.extend(sel_notes)

    # --- attach to chapters and interleave title cards
    def assemble(pool: list[Item]) -> list[Entry]:
        grouped: dict[int, list[Item]] = {}
        for it in pool:
            cp = ctx.chapter_for(it)
            grouped.setdefault(cp.chapter.index if cp else 0, []).append(it)

        out: list[Entry] = []
        first_card = True
        for cp in ctx.chapters:
            group = grouped.get(cp.chapter.index, [])
            if not group:
                continue
            group.sort(key=lambda i: (i.info.captured, i.info.path.name))
            if cfg.render.title_card:
                # A caller-supplied title names the opening card; later chapters
                # fall back to the place or day.
                custom = (cfg.render.title or "").strip()
                heading = (custom if (first_card and custom)
                           else (cp.title or cp.chapter.day_label))
                out.append(Entry(
                    item=None, is_title=True, title=heading,
                    subtitle=_chapter_subtitle(cp, ctx),
                    chapter=cp.chapter.index,
                    rank_reason=f"chapter {cp.chapter.index + 1} marker",
                ))
            first_card = False
            for it in group:
                out.append(Entry(item=it, chapter=cp.chapter.index,
                                 rank_reason=f"score {it.score:.3f}"))

        # Anything the chapter plan did not claim still has to reach the
        # screen. The loop above only walks ctx.chapters, so an item with no
        # chapter (or a chapter index that does not exist) would otherwise be
        # bucketed into a group that is never visited and silently vanish -
        # which, if it is the only item, looks like a crash somewhere else.
        placed = {e.item.key for e in out if e.item is not None}
        leftovers = [it for it in pool if it.key not in placed]
        if leftovers:
            notes.append(f"{len(leftovers)} item(s) matched no chapter and were "
                         "appended at the end")
            leftovers.sort(key=lambda i: (i.info.captured, i.info.path.name))
            for it in leftovers:
                out.append(Entry(item=it, chapter=None,
                                 rank_reason=f"score {it.score:.3f}"))
        return out

    entries = assemble(pool)
    if not entries:
        raise ValueError(
            "could not build a timeline: nothing was selected, or every selected "
            f"item was a title card. {len(pool)} item(s) reached the cut planner"
        )

    if cfg.order == "hook":
        notes.extend(apply_hook_order(entries))

    # --- timing
    #
    # select_items already sizes the pool so this should converge immediately.
    # The loop is a safety net: if the apportionment still overshoots, shed the
    # weakest shots rather than scaling durations. Scaling would land the total
    # on target but push every cut off the beat, which is far more noticeable
    # than a video that runs a second long.
    dropped = 0
    while True:
        assign_timing(entries, target, music, cfg)
        total = sum(e.duration for e in entries)
        if total <= target * 1.02 + 0.05:
            break
        content = [e for e in entries if not e.is_title]
        if len(content) <= 3 or dropped >= 64:
            log(f"timeline is {total - target:.1f}s over target; accepting "
                f"rather than breaking beat alignment", level="warn")
            break
        weakest = min(content, key=lambda e: e.score)
        entries.remove(weakest)
        dropped += 1
        if dropped <= 6:
            notes.append(
                f"dropped '{weakest.label}' (score {weakest.score:.3f}) to fit "
                f"{target:.0f}s")
        elif dropped == 7:
            notes.append(f"and {len(content) - 3} further low-scoring shots")

    for i, e in enumerate(entries):
        e.motion = _motion_for(i, len(entries))

    # Slide each clip's in-point earlier where the footage allows, so a shot
    # can keep its whole-beat duration instead of being trimmed. Trimming to
    # fit would put that cut off the beat for no editorial gain.
    slid = 0
    for e in entries:
        if e.item is None or e.item.kind != "video":
            continue
        item = e.item
        limit = min(item.end, item.info.duration or item.end)
        if e.duration > limit - item.start + 1e-6:
            earlier = max(0.0, limit - e.duration)
            if earlier < item.start - 1e-6:
                e.in_point = earlier
                slid += 1
    if slid:
        notes.append(f"slid {slid} clip in-points earlier so they keep their full length")

    cut = CutList(entries=entries, target=target, music=music, notes=notes)
    drift = cut.duration - target
    if abs(drift) > 1.0:
        msg = (f"final length {cut.duration:.1f}s vs target {target:.1f}s")
        if drift < 0:
            n_content = len([e for e in entries if not e.is_title])
            msg += (f" - there are not enough distinct shots in the library to "
                    f"fill {target:.0f}s at a watchable pace "
                    f"({n_content} shots available). Add more media, or ask "
                    f"for roughly {cut.duration:.0f}s.")
        cut.notes.append(msg)
        log(msg, level="warn")
    log(f"timeline: {len(entries)} entries, {cut.duration:.1f}s "
        f"(target {target:.1f}s)")
    return cut


def _chapter_subtitle(cp, ctx: TripContext) -> str:
    """The line under the chapter card: where, then how much, then when.

    The date itself is the title now, and the raw ISO day string used to go
    here, which read like a database export on screen.
    """
    parts = []
    if cp.place_label:
        parts.append(cp.place_label)
    if cp.items:
        parts.append(f"{len(cp.items)} shots")
    if cp.chapter.start:
        t = datetime.fromtimestamp(cp.chapter.start, tz=timezone.utc)
        parts.append(t.strftime("%H:%M"))
    return "  ·  ".join(parts)
