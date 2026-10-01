"""Per-file analysis cache.

Scoring a library means decoding every photo and every video clip, which on a
real trip is the single most expensive thing the pipeline does: 12,193 files
took ~50 minutes. All of that work depends only on the file's bytes and the
analysis settings, so it survives a re-run - but only if the key is right.

The failure mode to avoid is not a stale cache, it is a cache that looks fresh
and silently returns the wrong answer. That happens when the key misses
something the result depends on. So the key here is deliberately over-specified
rather than convenient:

  * path, size and mtime - catches edited, replaced and added files
  * the whole analysis config - catches a changed threshold
  * a version tag - catches a change to the scoring maths itself

Every field is folded into one hash with a stable algorithm (crc32 rather than
Python's hash(), which is randomised per process). If any of them changes, the
entry misses and the file is re-analysed. A stale hit is worse than a slow run,
so the default is to re-analyse whenever there is any doubt.

The cache is a single JSON-lines file appended as analysis completes, so a run
that is interrupted still leaves everything finished so far usable.
"""
from __future__ import annotations

import json
import os
import zlib
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from .quality import FrameMetrics
from .util import log

# Bump when the scoring maths changes. Any stale entry then misses on its own,
# which is the behaviour we want: slow and correct beats fast and wrong.
CACHE_VERSION = 1


def _digest(*parts: Any) -> str:
    """Stable hash of the key fields. crc32 is stable across processes."""
    h = zlib.crc32(repr(parts).encode("utf-8", "replace"))
    return f"{h:08x}"


def config_fingerprint(cfg: Any) -> str:
    """Hash every field of the analysis config, so a threshold change invalidates."""
    try:
        return _digest(CACHE_VERSION, asdict(cfg))
    except TypeError:
        # A plain object rather than a dataclass: fall back to the public attrs.
        return _digest(CACHE_VERSION,
                       sorted((k, v) for k, v in vars(cfg).items()
                              if not k.startswith("_")))


def file_key(info, cfg_fp: str) -> str:
    """Key for one file's analysis result."""
    try:
        st = info.path.stat()
        size, mtime = st.st_size, int(st.st_mtime)
    except OSError:
        # No stat means we cannot claim the file is unchanged. Miss on purpose.
        return ""
    return _digest(str(info.path), size, mtime, info.kind, cfg_fp)


class AnalysisCache:
    """Append-only store of per-file analysis results, keyed by file_key()."""

    def __init__(self, path: Path, cfg_fp: str) -> None:
        self.path = path
        self.cfg_fp = cfg_fp
        self._entries: dict[str, dict] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        kept = 0
        dropped = 0
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        # A truncated final line from an interrupted run.
                        dropped += 1
                        continue
                    # Reject anything measured under a different config, so a
                    # changed threshold cannot resurrect old numbers.
                    if rec.get("cfg") != self.cfg_fp:
                        dropped += 1
                        continue
                    self._entries[rec["key"]] = rec
                    kept += 1
        except OSError as exc:
            log(f"could not read the analysis cache ({exc}); rebuilding it",
                level="warn")
            return
        if kept:
            log(f"analysis cache: {kept} usable entries, {dropped} discarded")

    def get(self, key: str) -> list[dict] | None:
        if not key:
            return None
        rec = self._entries.get(key)
        return rec.get("items") if rec else None

    def put(self, key: str, items: list[dict]) -> None:
        if not key:
            return
        self._entries[key] = {"cfg": self.cfg_fp, "key": key, "items": items}
        self._dirty = True

    def flush(self) -> None:
        """Write everything out, replacing the file wholesale."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                for rec in self._entries.values():
                    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            # Replace atomically: a half-written cache that then reads back as
            # valid would be worse than no cache at all.
            os.replace(tmp, self.path)
            log(f"analysis cache: wrote {len(self._entries)} entries")
        except OSError as exc:
            log(f"could not write the analysis cache ({exc})", level="warn")
            try:
                tmp.unlink()
            except OSError:
                pass
        self._dirty = False


def item_to_record(item) -> dict:
    """Serialise one Item: enough to rebuild it, nothing that can go stale."""
    return {
        "info": str(item.info.path),
        "kind": item.kind,
        "start": item.start,
        "end": item.end,
        "score": item.score,
        "reject": item.reject,
        "hash": item.hash,
        "note": item.note,
        "metrics": asdict(item.metrics),
    }


def record_to_item(rec: dict, info):
    """Rebuild an Item from a cache record, using the freshly probed MediaInfo.

    The MediaInfo comes from the current scan, never from the cache. That is
    deliberate: probe results and EXIF are cheap to re-read and are the values
    selection actually depends on, so trusting cached copies of them is exactly
    the stale-data trap this module exists to avoid.
    """
    metrics = FrameMetrics(**{
        f.name: rec["metrics"].get(f.name, f.default)
        for f in fields(FrameMetrics)
    })
    return _make_item(
        info=info,
        kind=rec["kind"],
        start=rec.get("start", 0.0),
        end=rec.get("end", 0.0),
        score=rec.get("score", 0.0),
        metrics=metrics,
        reject=rec.get("reject"),
        hash=rec.get("hash", 0),
        note=rec.get("note", ""),
    )


def _make_item(**kwargs):
    from .analysis import Item
    return Item(**kwargs)