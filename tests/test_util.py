from __future__ import annotations

import math
from datetime import UTC, datetime

from pipeline import util


def test_human_duration():
    assert util.human_duration(59.2) == "0:59"
    assert util.human_duration(60) == "1:00"
    assert util.human_duration(90) == "1:30"
    assert util.human_duration(3600) == "1:00:00"
    assert util.human_duration(3661) == "1:01:01"


def test_clamp():
    assert util.clamp(1, 0, 5) == 1
    assert util.clamp(-1, 0, 5) == 0
    assert util.clamp(10, 0, 5) == 5
    assert math.isclose(util.clamp(2.5, 0, 5), 2.5)


def test_parse_exif_datetime_handles_basic_formats():
    ts = util.parse_exif_datetime("2025-09-27 14:23:05")
    dt = datetime.fromtimestamp(ts, tz=UTC)
    assert dt.year == 2025
    assert dt.month == 9
    assert dt.day == 27
    assert dt.hour == 14
    assert dt.minute == 23
    assert dt.second == 5


def test_parse_utc_offset():
    assert util.parse_utc_offset("+02:00") == 2 * 3600
    assert util.parse_utc_offset("-07:00") == -7 * 3600
    assert util.parse_utc_offset("+0230") == 2 * 3600 + 30 * 60
    assert util.parse_utc_offset("Z") == 0
    assert util.parse_utc_offset(None) == 0
