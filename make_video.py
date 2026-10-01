#!/usr/bin/env python3
"""make_video.py - turn a folder of unorganised photos and videos into a
finished trip recap.

    ./make_video.py                      # media/ -> out/, 90 seconds
    ./make_video.py --target 300         # five minutes
    ./make_video.py --target 600         # ten minutes
    ./make_video.py --dry-run            # decide everything, render nothing

Everything is local. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from pipeline import (
    Render, Analysis, Music, Pipeline, analyse_library, attach_items,
    build_context, build_cutlist, build_report, prepare_music, render, scan,
)
from pipeline.config import default_paths

# Scratch directories older than this (seconds) are assumed abandoned and
# removed at the start of a run. Long enough that a slow build is never hit.
STALE_WORK_H = 24 * 3600
from pipeline.util import (
    PipelineError, ToolError, ensure_dir, human_duration, log, require_tools,
)

ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="make_video",
        description="Build a trip recap from unorganised photos and videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--media", default="media", help="folder of source photos/videos")
    p.add_argument("--music", default="music", help="folder to look for music tracks in")
    p.add_argument("--out", default="out", help="where finished videos are written")
    p.add_argument("--output", default="", help="explicit output .mp4 path")
    p.add_argument("--name", default="", help="output filename stem")

    p.add_argument("--target", type=float, default=90,
                   help="target length in seconds")
    p.add_argument("--order", choices=["chronological", "hook"], default="chronological",
                   help="chronological keeps trip order; hook opens on the best shot")

    m = p.add_argument_group("music")
    m.add_argument("--mood", default="uplifting",
                   choices=["uplifting", "cinematic", "calm", "energetic", "warm"],
                   help="mood of the generated score")
    m.add_argument("--music-track", default="", help="use this audio file instead of generating")
    m.add_argument("--bpm", type=int, default=0, help="force a tempo (0 = auto)")
    m.add_argument("--no-duck", action="store_true",
                   help="do not lower music under native audio")

    v = p.add_argument_group("look")
    v.add_argument("--fit", choices=["blur", "crop", "pad"], default="blur",
                   help="how portrait shots fill a landscape frame")
    v.add_argument("--fps", type=int, default=30)
    v.add_argument("--title", default="", help="main title for the opening card")
    v.add_argument("--no-titles", action="store_true", help="skip chapter title cards")
    v.add_argument("--no-ken-burns", action="store_true",
                   help="static photos instead of slow push/pan")
    v.add_argument("--zoom", type=float, default=0.10, help="ken burns zoom amount")

    o = p.add_argument_group("output / behaviour")
    o.add_argument("--dry-run", action="store_true",
                   help="make every decision and write the report, but render nothing")
    o.add_argument("--keep-temp", action="store_true", help="keep intermediate files")
    o.add_argument("--no-geocode", action="store_true",
                   help="skip reverse geocoding of GPS coordinates")
    o.add_argument("--jobs", type=int, default=0, help="parallel workers (0 = auto)")
    o.add_argument("--quality", type=int, default=18,
                   help="x264 CRF; lower is better quality and bigger file")
    return p.parse_args(argv)


def cfg_from_args(a: argparse.Namespace) -> Pipeline:
    cfg = Pipeline()
    cfg.target_seconds = a.target
    cfg.order = a.order
    cfg.jobs = a.jobs
    cfg.keep_temp = a.keep_temp

    cfg.render = Render(
        fps=a.fps,
        fit=a.fit,
        crf=a.quality,
        ken_burns=not a.no_ken_burns,
        zoom_amount=a.zoom,
        title_card=not a.no_titles,
        title=a.title,
    )
    cfg.music = Music(
        mode="track" if a.music_track else "auto",
        track_path=a.music_track,
        mood=a.mood,
        bpm=a.bpm,
        duck=not a.no_duck,
    )
    cfg.analysis = Analysis()
    return cfg


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    started = time.time()
    paths = default_paths(ROOT)
    media_dir = Path(a.media)
    if not media_dir.is_absolute():
        media_dir = ROOT / media_dir
    music_dir = Path(a.music)
    if not music_dir.is_absolute():
        music_dir = ROOT / music_dir
    out_dir = Path(a.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    cfg = cfg_from_args(a)

    print()
    log("=" * 66)
    log("trip recap builder")
    log("=" * 66)

    # ---------------------------------------------------------- preflight
    require_tools()
    ensure_dir(out_dir)

    # Every run gets its own scratch directory. Sharing one means two builds
    # racing each other will silently swap stills and segments - a run with
    # --fit crop will hand a cropped frame to a run that asked for blur-fill,
    # and the finished video is wrong in a way nothing reports.
    work_root = ensure_dir(paths["work"])
    now = time.time()
    for stale in work_root.glob("run-*"):
        # Only genuinely abandoned ones. Deleting by age rather than by name
        # keeps this from reaching into a build that is still running.
        try:
            if stale.is_dir() and now - stale.stat().st_mtime > STALE_WORK_H:
                shutil.rmtree(stale, ignore_errors=True)
        except OSError:
            pass
    work = ensure_dir(work_root / f"run-{os.getpid()}-{int(now) % 100000}")

    stem = a.name or "trip_recap"
    if a.output:
        final = Path(a.output)
        if not final.is_absolute():
            final = ROOT / final
    else:
        final = out_dir / f"{stem}.mp4"
    ensure_dir(final.parent)

    # ------------------------------------------------------------ ingest
    log("")
    log("[1/7] reading your library")
    lib = scan(media_dir)
    log(f"      {lib.summary()}")

    # ----------------------------------------------------------- analyse
    log("")
    log("[2/7] scoring every shot")
    items = analyse_library(lib, cfg.analysis, paths["cache"])

    # -------------------------------------------------------- trip shape
    log("")
    log("[3/7] mapping the trip")
    ctx = build_context(lib, cfg, paths["cache"], geocode=not a.no_geocode)
    attach_items(ctx, items)

    # ------------------------------------------------------------ music
    log("")
    log("[4/7] preparing music")
    music = prepare_music(
        cfg, work,
        library_audio=sorted(music_dir.glob("*")) if music_dir.exists() else [],
        target_seconds=cfg.target_seconds,
        prefer=bool(a.music_track),
        fps=cfg.render.fps,
    )

    # --------------------------------------------------------- cut list
    log("")
    log("[5/7] choosing shots and building the cut")
    cut = build_cutlist(items, ctx, cfg, music, cfg.target_seconds)

    if a.dry_run:
        log("")
        log("dry run - stopping before render")
        fake = type("R", (), {
            "video": final, "duration": cut.duration,
            "width": cfg.render.width, "height": cfg.render.height,
            "fps": cfg.render.fps, "segments": len(cut.entries),
        })()
        md = build_report(lib, items, ctx, cut, cfg, fake, final)
        log(f"\n  would have written {final}")
        log(f"  timeline: {len(cut.entries)} entries, "
            f"{human_duration(cut.duration)}")
        log(f"  report:   {md}")
        if not (cfg.keep_temp or a.keep_temp):
            shutil.rmtree(work, ignore_errors=True)
        return 0

    # ----------------------------------------------------------- render
    log("")
    log("[6/7] rendering")
    result = render(cut, cfg, music.path, work, final,
                    keep_temp=cfg.keep_temp or a.keep_temp)

    # ----------------------------------------------------------- report
    log("")
    log("[7/7] writing the report")
    md = build_report(lib, items, ctx, cut, cfg, result, final)

    elapsed = time.time() - started
    log("")
    log("=" * 66)
    log(f"done in {elapsed:.0f}s")
    log(f"  video:  {result.video}  ({human_duration(result.duration)}, "
        f"{result.width}x{result.height})")
    log(f"  report: {md}")
    log("=" * 66)
    print()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as exc:
        log(str(exc), level="error")
        sys.exit(2)
    except ToolError as exc:
        log(str(exc), level="error")
        sys.exit(3)
    except KeyboardInterrupt:
        log("interrupted", level="warn")
        sys.exit(130)
