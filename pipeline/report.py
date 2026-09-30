"""Stage 8 - the decision record.

Produces a human-readable account of what the pipeline chose and why. This is
the part that makes the work reviewable and presentable rather than a black
box.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .analysis import Item
from .context import TripContext
from .ingest import Library
from .select import CutList
from .util import human_duration, log


def _fmt_time(ts: float) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _bar(value: float, width: int = 18) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def build_report(lib: Library, items: list[Item], ctx: TripContext,
                 cut: CutList, cfg, result, out_path: Path) -> Path:
    md = out_path.with_suffix(".md")
    lines: list[str] = []
    a = lines.append

    summary = cut.summary()
    music = cut.music

    a("# Trip recap - edit report")
    a("")
    a(f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    a("")

    # ------------------------------------------------------------ headline
    a("## Result")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| Output | `{result.video.name}` |")
    a(f"| Length | {human_duration(result.duration)} |")
    a(f"| Resolution | {result.width}x{result.height} @ {result.fps}fps |")
    a(f"| Shots on screen | {summary['entries']} "
      f"({summary['photos']} photos, {summary['videos']} clips, "
      f"{summary['titles']} title cards) |")
    a(f"| Music | {(music.label if music else 'none')} "
      f"@ {summary['bpm']:.0f} BPM |")
    a(f"| Source library | {lib.summary()} |")
    a("")

    # ------------------------------------------------------------- library
    a("## 1. What came in")
    a("")
    a(f"- **{len(lib.photos)}** photos")
    a(f"- **{len(lib.videos)}** video clips")
    st = ctx.stats()
    if st["days"]:
        a(f"- Spanning **{st['days']} days**, "
          f"{human_duration(ctx.span_seconds)} of real time")
    if st["stops"]:
        a(f"- **{st['stops']} GPS-tagged locations**")
    missing = [i for i in lib.items if not i.captured]
    if missing:
        a(f"- {len(missing)} files had no capture timestamp; ordered by filename")
    if lib.rejected:
        a(f"- {len(lib.rejected)} files skipped as unusable")
    a("")

    if ctx.stops and any(s.name for s in ctx.stops):
        a("### Locations (from EXIF GPS)")
        a("")
        a("| # | Place | Items | Window |")
        a("|---|---|---|---|")
        counts: dict[int, int] = {}
        for it in items:
            for i, s in enumerate(ctx.stops):
                if s.has_gps() and s.first <= it.info.captured <= s.last:
                    counts[i] = counts.get(i, 0) + 1
                    break
        for i, stop in enumerate(ctx.stops):
            if not stop.has_gps():
                continue
            a(f"| {i + 1} | {stop.label} | {counts.get(i, 0)} | "
              f"{_fmt_time(stop.first)} - {_fmt_time(stop.last)} |")
        a("")

    # ------------------------------------------------------------- scoring
    a("## 2. How shots were judged")
    a("")
    an = cfg.analysis
    a("Every frame was scored on these signals (weights in brackets):")
    a("")
    a("| Signal | Weight | What it detects |")
    a("|---|---|---|")
    a(f"| Sharpness | {an.w_sharpness:.2f} | Variance of the Laplacian plus "
      f"Sobel energy - blur and missed focus |")
    a(f"| Exposure | {an.w_exposure:.2f} | Distance from mid-grey, "
      f"blown highlights, crushed shadows |")
    a(f"| Faces | {an.w_faces:.2f} | Face count, weighted by whether eyes "
      f"look open. Usually the strongest signal on trips |")
    a(f"| Colour | {an.w_color:.2f} | Hasler-Suesstrunk colourfulness - "
      f"flags dull grey overcast frames |")
    a(f"| Contrast | {an.w_contrast:.2f} | Luminance spread - flags flat, "
      f"low-contrast shots |")
    a(f"| Motion | {an.w_motion:.2f} | Mean inter-frame change. Rewards "
      f"movement, penalises handheld shake |")
    a("| Light | multiplier | Real sunrise/sunset per location. "
      "Golden hour beats harsh midday |")
    a("")
    a(f"Hard rejects: sharpness below `{an.min_sharpness}`, more than "
      f"`{an.max_clipped:.0%}` blown highlights, more than `{an.max_black:.0%}` "
      f"crushed blacks.")
    a("")

    rejected = [i for i in items if i.reject]
    if rejected:
        from collections import Counter
        why = Counter(r.reject.split(" (")[0] for r in rejected)
        a("### What got thrown out")
        a("")
        a("| Reason | Count |")
        a("|---|---|")
        for reason, n in why.most_common():
            a(f"| {reason} | {n} |")
        a("")

    # ---------------------------------------------------------- structure
    a("## 3. Structure")
    a("")
    if ctx.chapters:
        a("The trip was split into chapters where time jumped by more than "
          "5 hours or you moved more than ~2.5 km. Each chapter then received "
          "screen time in proportion to how much usable material it "
          "contained (with a floor, so a quiet day still appears).")
        a("")
        a("| Chapter | Shots available | Shots used | Screen time |")
        a("|---|---|---|---|")
        # Measured from the finished cut list rather than the plan, so this
        # says what the video actually contains.
        used: dict[int, list[float]] = {}
        for e in cut.entries:
            used.setdefault(e.chapter, []).append(e.duration)
        for cp in ctx.chapters:
            secs = used.get(cp.chapter.index)
            if not secs:
                continue
            a(f"| {cp.label} | {len(cp.items)} | {len(secs)} | "
              f"{human_duration(sum(secs))} |")
        a("")
    if cut.notes:
        a("Additional notes:")
        a("")
        for n in cut.notes:
            a(f"- {n}")
        a("")

    # ------------------------------------------------------------- pacing
    a("## 4. Pacing and music")
    a("")
    if music and music.bpm:
        spb = 60.0 / music.bpm
        a(f"Every shot length is an **exact whole number of beats** at "
          f"{music.bpm:.1f} BPM (one beat = {spb:.2f}s). That is why the cuts "
          f"land on the beat rather than near it, and why the total length "
          f"lands on the target without rounding drift.")
        a("")
        a(f"- Music: {'generated' if music.generated else 'supplied'} "
          f"(`{music.label}`)")
        a(f"- Beat grid: {len(music.beat_times)} beats, "
          f"{len(music.downbeat_times)} bars")
        a(f"- Still duration range: {cfg.photo_min:.1f}-{cfg.photo_max:.1f}s")
        a(f"- Video duration range: {cfg.video_min:.1f}-{cfg.video_max:.1f}s")
        a("")
        if music.generated:
            a("The score was synthesised for this edit, so it is original "
              "work with no licensing obligations. If you would rather use a "
              "track, drop it in `music/` or pass `--music-track`.")
            a("")
    else:
        a("No music was used, so shot lengths were set purely from quality "
          "scores.")
        a("")

    # ------------------------------------------------------------ timeline
    a("## 5. Full timeline")
    a("")
    a("| # | In | Length | Shot | Score | Motion |")
    a("|---|---|---|---|---|---|")
    t = 0.0
    for i, e in enumerate(cut.entries):
        if e.is_title:
            name = f"**{e.title}**"
            score = "-"
            motion = "-"
        else:
            name = f"`{e.item.label}`"
            score = f"{e.score:.3f}"
            motion = e.motion
        a(f"| {i + 1} | {human_duration(t)} | {e.duration:.2f}s | {name} | "
          f"{score} | {motion} |")
        t += e.duration
    a("")
    a(f"Total: **{human_duration(cut.duration)}**")
    a("")

    # ------------------------------------------------------------ top picks
    used = [e for e in cut.entries if e.item]
    if used:
        top = sorted(used, key=lambda e: e.score, reverse=True)[:12]
        a("## 6. Highest scoring shots used")
        a("")
        a("| Shot | Score | Sharpness | Exposure | Faces |")
        a("|---|---|---|---|---|")
        for e in top:
            m = e.item.metrics
            a(f"| `{e.item.label}` | {_bar(e.score)} `{e.score:.3f}` | "
              f"{m.sharpness:.0f} | {m.exposure:.2f} | {m.faces} |")
        a("")

    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # machine-readable EDL
    edl = out_path.with_suffix(".edl.json")
    edl.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "output": str(result.video),
        "duration": result.duration,
        "target": cut.target,
        "music": {
            "label": music.label if music else None,
            "bpm": music.bpm if music else None,
            "generated": music.generated if music else None,
        },
        "render": {
            "fit": cfg.render.fit,
            "width": cfg.render.width,
            "height": cfg.render.height,
            "fps": cfg.render.fps,
            "ken_burns": bool(cfg.render.ken_burns),
            "titles": bool(cfg.render.title_card),
        },
        "notes": cut.notes,
        "entries": [
            {
                "index": i,
                "start": round(t, 3),
                "duration": round(e.duration, 3),
                "beats": e.beats,
                "type": e.kind,
                "title": e.title or None,
                "subtitle": e.subtitle or None,
                "source": str(e.item.info.path) if e.item else None,
                # The in-point actually rendered, which is not always the
                # shot's own start: clips get slid earlier when the slot is
                # longer than the footage from the shot's start to its end.
                "source_start": round(e.src_in, 3) if e.item else None,
                "source_end": (round(e.src_in + e.duration, 3)
                               if e.item else None),
                "shot_start": e.item.start if e.item else None,
                "shot_end": e.item.end if e.item else None,
                "has_audio": (e.item.info.has_audio if e.item else False),
                "captured": e.item.info.captured if e.item else None,
                "gps": ([e.item.info.latitude, e.item.info.longitude]
                        if e.item and (e.item.info.latitude or e.item.info.longitude)
                        else None),
                "score": round(e.score, 4) if e.item else None,
                "motion": e.motion,
                "chapter": e.chapter,
            }
            for t, (i, e) in zip(
                _cumulative(cut.entries), enumerate(cut.entries)
            )
        ],
    }, indent=2), encoding="utf-8")

    log(f"report: {md.name}")
    log(f"edl:    {edl.name}")
    return md


def _cumulative(entries):
    t = 0.0
    for e in entries:
        yield t
        t += e.duration
