"""Geospatial + solar metadata intelligence.

Turns raw EXIF into the three things that actually shape a trip recap:
where you were, what time of day you were there, and what the light was like.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .util import log

EARTH_RADIUS_KM = 6371.0088

# Two shots within this distance are treated as the same place.
SAME_PLACE_M = 400.0
# Beyond this, we consider it a different stop.
NEW_PLACE_M = 2500.0
# Visits this close together, with less time between them than this, are the
# same place: a walk across town, not a journey. See merge_visits().
VISIT_MERGE_M = 3000.0
VISIT_MERGE_GAP_H = 8.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ----------------------------------------------------------------- sunlight

def _julian_day(ts: float) -> float:
    return ts / 86400.0 + 2440587.5


def sun_times(ts: float, lat: float, lon: float) -> tuple[float, float]:
    """(sunrise, sunset) as unix timestamps, UTC.

    NOAA solar position approximation. Accurate to ~1 minute, which is far
    more than we need for deciding whether a photo was shot at golden hour.
    Returns (0.0, 0.0) in polar day/night.
    """
    try:
        jd = _julian_day(ts)
        n = jd - 2451545.0 + 0.0008
        lw = -lon  # NOAA uses west-positive
        j_star = n - lw / 360.0
        m = (357.5291 + 0.98560028 * j_star) % 360.0
        c = (1.9148 * math.sin(math.radians(m))
             + 0.0200 * math.sin(math.radians(2 * m))
             + 0.0003 * math.sin(math.radians(3 * m)))
        lam = (m + c + 180.0 + 102.9372) % 360.0
        j_transit = (2451545.0 + j_star + 0.0053 * math.sin(math.radians(m))
                     - 0.0069 * math.sin(math.radians(2 * lam)))
        sin_dec = math.sin(math.radians(lam)) * math.sin(math.radians(23.4397))
        cos_dec = math.cos(math.asin(sin_dec))
        phi = math.radians(lat)
        # -0.833 deg accounts for refraction + solar disc.
        cos_omega = ((math.sin(math.radians(-0.833)) - math.sin(phi) * sin_dec)
                     / (math.cos(phi) * cos_dec))
        if cos_omega < -1.0 or cos_omega > 1.0:
            return 0.0, 0.0
        omega = math.degrees(math.acos(cos_omega))
        noon = (j_transit - 2451545.0) * 24.0 * 3600.0
        rise = noon - omega * 12.0 * 360.0 / math.pi
        set_ = noon + omega * 12.0 * 360.0 / math.pi
        return rise, set_
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0, 0.0


def light_score(ts: float, lat: float, lon: float) -> float:
    """How flattering the light was, 0..1.

    Golden hour at the edges of the day scores best, harsh midday is penalised,
    and deep night is penalised hard (almost always unusable).
    """
    rise, set_ = sun_times(ts, lat, lon) if lon or lat else (0.0, 0.0)
    if not rise or not set_ or set_ <= rise:
        # No usable solar data; fall back to a gentle time-of-day prior.
        hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour + \
            datetime.fromtimestamp(ts, tz=timezone.utc).minute / 60.0
        if hour < 5.5 or hour > 20.5:
            return 0.25
        return 0.6

    if ts < rise - 0.75 * 3600 or ts > set_ + 0.75 * 3600:
        return 0.15   # night / deep twilight

    after_rise = ts - rise
    before_set = set_ - ts
    golden_window = 75 * 60.0

    if after_rise <= golden_window:
        # Peaks ~25 min after sunrise, fades by the end of the window.
        return float(max(0.6, 1.0 - abs(after_rise - 1500.0) / (2 * golden_window)))
    if before_set <= golden_window:
        peak = 1500.0
        return float(max(0.6, 1.0 - abs(before_set - peak) / (2 * golden_window)))

    # Daylight proper. Penalise the harsh midday window.
    midday_distance = abs(ts - (rise + set_) / 2.0)
    span = max(1.0, (set_ - rise) / 2.0)
    return float(max(0.45, 1.0 - 0.45 * (midday_distance / span)))


# --------------------------------------------------------- reverse geocoding

class Geocoder:
    """Nominatim reverse lookup, cached to disk, fully optional.

    Any network failure degrades to coordinates-only labels; nothing here can
    break the pipeline.
    """

    def __init__(self, cache_dir: Path, *, enabled: bool = True, zoom: int = 10):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.zoom = zoom
        self._mem: dict[tuple[float, float], str] = {}
        self._unavailable = False

    def _key(self, lat: float, lon: float) -> str:
        # ~1.1 km cells: everyone standing in the same spot shares a lookup.
        return f"{round(lat, 2)}_{round(lon, 2)}"

    def _cache_path(self, lat: float, lon: float) -> Path:
        return self.cache_dir / f"{self._key(lat, lon)}.json"

    def lookup(self, lat: float, lon: float) -> str:
        if not self.enabled or self._unavailable or not (lat or lon):
            return ""
        k = (round(lat, 2), round(lon, 2))
        if k in self._mem:
            return self._mem[k]

        path = self._cache_path(lat, lon)
        if path.exists():
            try:
                name = json.loads(path.read_text()).get("name", "")
                self._mem[k] = name
                return name
            except (OSError, json.JSONDecodeError):
                pass

        url = ("https://nominatim.openstreetmap.org/reverse?format=jsonv2"
               f"&lat={lat:.4f}&lon={lon:.4f}&zoom={self.zoom}")
        name = ""
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "trip-recap-pipeline/1.0 (personal project)",
                "Accept-Language": "en",
            })
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            name = _format_place(data)
            path.write_text(json.dumps({"name": name, "lat": lat, "lon": lon}))
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                log(f"geocoding rate-limited ({exc.code}); using coordinates only",
                    level="warn")
                self._unavailable = True
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, ValueError) as exc:
            log(f"geocoding unavailable ({exc.__class__.__name__}); "
                f"using coordinates only", level="warn")
            self._unavailable = True

        if name:
            time.sleep(1.0)  # Nominatim asks for <= 1 req/sec
        self._mem[k] = name
        return name


_PLACE_TIERS = (
    ("city", "town", "municipality", "village", "hamlet"),
    ("suburb", "neighbourhood", "city_district", "quarter"),
)


def _format_place(data: dict) -> str:
    addr = data.get("address") or {}
    parts = [p for p in str(addr.get("name", "")).split(",") if p]
    for key in ("city", "town", "village", "municipality", "hamlet"):
        if addr.get(key):
            parts.append(str(addr[key]))
            break
    else:
        for key in ("country", "state"):
            if addr.get(key):
                parts.append(str(addr[key]))
                break
    seen: list[str] = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.append(p)
    return ", ".join(seen[:2])


# ------------------------------------------------------------- clustering

@dataclass
class Stop:
    """A place you were, possibly across several hours or days."""

    lat: float = 0.0
    lon: float = 0.0
    name: str = ""
    times: list[float] = field(default_factory=list)

    @property
    def first(self) -> float:
        return min(self.times) if self.times else 0.0

    @property
    def last(self) -> float:
        return max(self.times) if self.times else 0.0

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        if self.lat or self.lon:
            return f"{abs(self.lat):.2f}°{'N' if self.lat >= 0 else 'S'} " \
                   f"{abs(self.lon):.2f}°{'E' if self.lon >= 0 else 'W'}"
        return "Unlocated"

    def has_gps(self) -> bool:
        return bool(self.lat or self.lon)


def cluster_stops(entries: list[tuple[float, float, float]]) -> list[Stop]:
    """entries = [(lat, lon, timestamp)]. Time-ordered in, Stops out."""
    stops: list[Stop] = []
    for lat, lon, ts in sorted(entries, key=lambda e: e[2]):
        if not (lat or lon):
            # No GPS: keep in whatever stop we are currently in.
            if stops:
                stops[-1].times.append(ts)
            continue
        if stops and stops[-1].has_gps():
            d = haversine_km(stops[-1].lat, stops[-1].lon, lat, lon) * 1000.0
            if d <= SAME_PLACE_M:
                # Running mean keeps the centroid stable.
                n = len(stops[-1].times)
                stops[-1].lat = (stops[-1].lat * n + lat) / (n + 1)
                stops[-1].lon = (stops[-1].lon * n + lon) / (n + 1)
                stops[-1].times.append(ts)
                continue
        stops.append(Stop(lat=lat, lon=lon, times=[ts]))
    return merge_visits(stops)


def merge_visits(stops: list[Stop]) -> list[Stop]:
    """Fold short trips away from a stop back into that stop.

    Clustering only ever compares a point with the stop it came from most
    recently, so a day spent walking a city breaks into a stop every few
    hundred metres. That is right for "where were you standing" and wrong for
    everything the rest of the pipeline cares about: the report would claim a
    dozen locations when there were three, place names would be looked up
    repeatedly for the same spot, and sunset times would be recomputed for
    every fragment.

    A visit is therefore only a new stop if it is genuinely somewhere else, or
    far enough away in time to be a different day. Nearby visits separated by a
    short gap are the same place and get merged.
    """
    merged: list[Stop] = []
    for s in stops:
        if merged and merged[-1].has_gps() and s.has_gps():
            prev = merged[-1]
            gap_h = (s.first - prev.last) / 3600.0
            d_m = haversine_km(prev.lat, prev.lon, s.lat, s.lon) * 1000.0
            if d_m <= VISIT_MERGE_M and gap_h <= VISIT_MERGE_GAP_H:
                n = len(prev.times)
                m = len(s.times)
                prev.lat = (prev.lat * n + s.lat * m) / (n + m)
                prev.lon = (prev.lon * n + s.lon * m) / (n + m)
                prev.times.extend(s.times)
                prev.times.sort()
                continue
        merged.append(s)
    return merged


def name_stops(stops: list[Stop], geocoder: Geocoder) -> None:
    pending = [s for s in stops if s.has_gps()]
    if not pending:
        return
    log(f"resolving place names for {len(pending)} stops")
    for i, s in enumerate(pending):
        s.name = geocoder.lookup(s.lat, s.lon)
        if (i + 1) % 5 == 0:
            log(f"  places {i + 1}/{len(pending)}")
    # Fall back to a compact coordinate label when nothing resolved.
    for s in stops:
        if not s.name and s.has_gps():
            s.name = f"{s.lat:.2f}, {s.lon:.2f}"


# ---------------------------------------------------------------- chapters

@dataclass
class Chapter:
    """One stretch of the trip: a day (or a big location change)."""

    index: int
    start: float
    end: float
    stop: Stop | None = None
    day_label: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def build_chapters(entries: list[tuple[float, float, float]],
                   stops: list[Stop],
                   *,
                   gap_hours: float = 5.0) -> list[Chapter]:
    """Split the trip where there is a long time gap or a real move.

    entries: [(lat, lon, ts)] in capture order.
    """
    ordered = sorted(entries, key=lambda e: e[2])
    if not ordered:
        return []

    gap_seconds = gap_hours * 3600.0
    chapters: list[Chapter] = []
    current: list[tuple[float, float, float]] = [ordered[0]]

    for prev, cur in zip(ordered, ordered[1:]):
        dt = cur[2] - prev[2]
        moved = False
        if prev[0] or prev[1]:
            d = haversine_km(prev[0], prev[1], cur[0], cur[1]) * 1000.0
            moved = d >= NEW_PLACE_M
        if dt >= gap_seconds or moved:
            chapters.append(_close_chapter(len(chapters), current, stops))
            current = []
        current.append(cur)

    chapters.append(_close_chapter(len(chapters), current, stops))
    # No filtering here. Every chapter holds at least one entry by
    # construction, and a chapter built from a single instant is perfectly
    # real - a burst of photos sharing a timestamp, or a place visited once.
    # Dropping the zero-duration ones silently deletes those items from the
    # whole plan, and if every chapter is a single instant the list comes back
    # empty and the run dies further down with an unrelated error.
    return chapters


def _close_chapter(index: int,
                   entries: list[tuple[float, float, float]],
                   stops: list[Stop]) -> Chapter:
    start = entries[0][2]
    end = entries[-1][2]
    dominant: Stop | None = None
    if stops:
        best = max(
            (s for s in stops if s.first <= end + 1 and s.last >= start - 1),
            key=lambda s: sum(1 for e in entries if s.first <= e[2] <= s.last),
            default=None,
        )
        dominant = best
    day = datetime.fromtimestamp(start, tz=timezone.utc)
    return Chapter(
        index=index,
        start=start,
        end=end,
        stop=dominant,
        day_label=day.strftime("%b %-d") if _supports_dash_d() else day.strftime("%b %d"),
    )


def _supports_dash_d() -> bool:
    # %-d is glibc/BSD; guard for platforms that only accept %d.
    try:
        datetime(2020, 1, 5).strftime("%-d")
        return True
    except ValueError:
        return False
