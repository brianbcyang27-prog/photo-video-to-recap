from __future__ import annotations

import datetime as dt
import math

from pipeline import geo

UTC = dt.UTC
JST = dt.timezone(dt.timedelta(hours=9))


def _at(y, m, d, hour, minute=0, tz=UTC) -> float:
    return dt.datetime(y, m, d, hour, minute, tzinfo=tz).timestamp()


def _hm(ts: float, tz=UTC) -> int:
    """Minutes past midnight in ``tz`` - convenient for comparing to tables."""
    local = dt.datetime.fromtimestamp(ts, tz)
    return local.hour * 60 + local.minute


def test_haversine_km():
    d = geo.haversine_km(35.0116, 135.7681, 34.6851, 135.8048)
    assert math.isclose(d, 36.46, abs_tol=0.01)


# Sunrise and sunset are checked against published almanac times, not against
# the implementation. The bound is six minutes: the algorithm is an
# approximation, and asserting to the second would only enshrine its error.
def test_sun_times_matches_published_almanac():
    cases = [
        ("London midsummer", 51.5074, -0.1278, _at(2025, 6, 15, 12, 0), UTC,
         3 * 60 + 43, 20 * 60 + 21),
        ("London midwinter", 51.5074, -0.1278, _at(2025, 12, 21, 12, 0), UTC,
         8 * 60 + 4, 15 * 60 + 53),
        ("Kyoto", 35.0116, 135.7681, _at(2025, 6, 15, 12, 0), JST,
         4 * 60 + 45, 19 * 60 + 11),
        ("Equator at equinox", 0.0, 0.0, _at(2025, 3, 20, 12, 0), UTC,
         6 * 60 + 5, 18 * 60 + 11),
    ]
    for label, lat, lon, when, tz, want_rise, want_set in cases:
        rise, set_ = geo.sun_times(when, lat, lon)
        assert rise != 0.0, f"{label}: no solar data"
        assert abs(_hm(rise, tz) - want_rise) <= 6, f"{label}: sunrise"
        assert abs(_hm(set_, tz) - want_set) <= 6, f"{label}: sunset"


def _noon(ts: float, lat: float, lon: float) -> float:
    rise, set_ = geo.sun_times(ts, lat, lon)
    return (rise + set_) / 2.0


def test_sun_times_solar_noon_tracks_longitude():
    """Solar noon is one hour earlier for every 15 degrees east of Greenwich.

    This is the sign of the longitude correction, the easiest thing in the
    function to get backwards. A reversed sign keeps every returned time
    looking plausible while shifting sunrise by twice the longitude error, so
    it is pinned here rather than left to the almanac cases above.

    Noon is used rather than sunrise because noon is the one quantity that
    depends on longitude alone; sunrise also depends on day length, which
    varies with latitude and season.
    """
    when = _at(2025, 6, 15, 12, 0)
    greenwich = _noon(when, 51.5, 0.0)
    tokyo = _noon(when, 35.0, 139.7)
    new_york = _noon(when, 40.7, -74.0)
    # Tokyo is 9h18m east of Greenwich, New York 4h56m west.
    assert abs((greenwich - tokyo) / 3600.0 - 9.31) < 0.1
    assert abs((greenwich - new_york) / 3600.0 + 4.93) < 0.1
    assert tokyo < greenwich < new_york


def test_sun_times_returns_the_requested_day():
    """Sunrise must land within a day of the instant asked about.

    A mismatched epoch shows up here first: the arithmetic still returns a
    plausible time of day, just on the wrong date.
    """
    when = _at(2025, 6, 15, 12, 0)
    rise, set_ = geo.sun_times(when, 51.5074, -0.1278)
    assert abs(rise - when) < 24 * 3600
    assert abs(set_ - when) < 24 * 3600
    assert rise < set_


def test_sun_times_polar_returns_nothing():
    assert geo.sun_times(_at(2025, 6, 21, 12, 0), 69.6492, 18.9553) == (0.0, 0.0)
    assert geo.sun_times(_at(2025, 12, 21, 12, 0), 69.6492, 18.9553) == (0.0, 0.0)


def test_light_score_prefers_golden_hour_over_midday():
    """Overhead noon light is the harshest of the day, not the best."""
    kyoto = (35.0116, 135.7681)
    # Stated in local time: Kyoto sunrise is 04:45 on this date, and the
    # golden hour peaks about 25 minutes after it.
    golden = geo.light_score(_at(2025, 6, 15, 5, 10, JST), *kyoto)
    midday = geo.light_score(_at(2025, 6, 15, 12, 0, JST), *kyoto)
    night = geo.light_score(_at(2025, 6, 15, 1, 0, JST), *kyoto)
    for value in (golden, midday, night):
        assert 0.0 <= value <= 1.0
    assert night < midday
    assert golden > midday


def test_light_score_uses_local_day_not_utc_day():
    """A 06:00 shot in Kyoto is early morning, and must not read as night.

    Sun times are derived from the UTC date, and at 06:00 in Tokyo it is still
    the previous afternoon in UTC. Scoring against that window would mark every
    morning shot east of Greenwich as though it had been taken in the dark -
    which is most of Asia, and a large share of any trip worth filming.
    """
    tz = dt.timezone(dt.timedelta(hours=round(139.7 / 15)))
    morning = geo.light_score(_at(2025, 6, 15, 6, 0, tz), 35.0, 139.7)
    midday = geo.light_score(_at(2025, 6, 15, 12, 0, tz), 35.0, 139.7)
    night = geo.light_score(_at(2025, 6, 15, 1, 0, tz), 35.0, 139.7)
    assert night < morning
    assert morning < midday or morning >= 0.6   # not a night score either way


def test_light_score_survives_a_full_day_everywhere():
    """No location and hour may produce a value outside 0..1 or an exception."""
    for lat in (-60.0, -33.9, 0.0, 35.0, 51.5, 69.6):
        for lon in (-74.0, 0.0, 139.7, 151.2):
            for hour in range(0, 24):
                value = geo.light_score(_at(2025, 6, 15, hour), lat, lon)
                assert 0.0 <= value <= 1.0, (lat, lon, hour, value)


def test_light_score_without_gps_uses_time_of_day_prior():
    day = geo.light_score(_at(2025, 6, 15, 12, 0), 0.0, 0.0)
    night = geo.light_score(_at(2025, 6, 15, 3, 0), 0.0, 0.0)
    assert day > night
    assert 0.0 <= night <= 1.0


def test_cluster_stops_merges_nearby_and_separates_distant():
    # (lat, lon, timestamp), time-ordered
    entries = [
        (35.0116, 135.7681, 0.0),
        (35.0118, 135.7683, 10.0),     # ~30m away: same stop
        (34.6851, 135.8048, 20.0),     # ~36km away: new stop
    ]
    stops = geo.cluster_stops(entries)
    assert len(stops) == 2
    assert len(stops[0].times) == 2
    assert len(stops[1].times) == 1


def test_cluster_stops_keeps_gpsless_shots_with_current_stop():
    entries = [
        (35.0116, 135.7681, 0.0),
        (0.0, 0.0, 10.0),              # no GPS at all
        (35.0118, 135.7683, 20.0),
    ]
    stops = geo.cluster_stops(entries)
    assert len(stops) == 1
    assert len(stops[0].times) == 3
