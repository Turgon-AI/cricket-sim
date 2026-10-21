"""Match-day weather from Open-Meteo (ERA5) hourly data: request planning, UTC normalisation, playing windows.

Time handling
  Open-Meteo answers a request in ONE fixed offset (`utc_offset_seconds`), even across a DST change, and with
  timezone=auto that offset is the one in force on the *fetch* date. We therefore never use response local times:
  every hour is converted back to UTC (time - utc_offset_seconds), and local time is recomputed per hour with
  zoneinfo for the venue's IANA zone at that instant. The result does not depend on when the data was fetched.

Playing window (per match day)
  window_source = 'ball_timestamps': first to last ball of that day from Ari's per-ball UTC timestamps
                  (Cricinfo-derived, so the exact times stay in an internal table).
  window_source = 'default_estimate': a format default in local time (below), converted with zoneinfo at the date.
  Aggregates use only hours inside the window. Rain in the 12 hours before the window is a separate column.

Hour conventions
  Hours sit on the venue's UTC grid (minute :00, or :30/:15 for half-hour zones, see series_grid).
  Instantaneous variables (temperature, humidity, ...) at hour t are used when floor(start) <= t <= ceil(end).
  Precipitation at t is the total for (t - 1 h, t]; it is used when that hour overlaps the window, i.e.
  start < t < end + 1 h. Pre-match rain: the 12 accumulations ending in (start - 12 h, start].
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

WX_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "cloud_cover", "wind_speed_10m",
           "wind_direction_10m", "precipitation", "surface_pressure"]
WX_COLS = ["temperature_c", "relative_humidity_pct", "dew_point_c", "cloud_cover_pct", "wind_speed_kmh",
           "wind_direction_deg", "precipitation_mm", "surface_pressure_hpa"]
PRECIP = WX_COLS.index("precipitation_mm")
HOUR = timedelta(hours=1)

# Local-time defaults when no ball timestamps exist. They cover the usual start/end spread of each format
# (from Ari's timestamps 2019-2026: Tests start ~10:00 and end ~17:30; ODIs start 09:00-15:00 and end 15:00-23:00;
# T20s mostly 14:00-23:00; IPL 15:30 or 19:30 starts ending by ~23:40). Day/night status is not in Cricsheet.
DEFAULT_WINDOWS = {"Test": (time(10, 0), time(18, 0)), "ODI": (time(10, 0), time(22, 0)),
                   "T20": (time(14, 0), time(23, 0)), "IPL": (time(15, 30), time(23, 30))}
CLUSTER_GAP = timedelta(hours=6)
MIN_BALLS_PER_DAY = 20
MAX_DAY_HOURS = {"Test": 13, "ODI": 13, "T20": 9, "IPL": 9}


def match_class(match_type: str | None, event_name: str | None) -> str:
    if event_name and event_name.startswith("Indian Premier League"):
        return "IPL"
    return {"Test": "Test", "MDM": "Test", "ODI": "ODI", "ODM": "ODI"}.get(match_type or "", "T20")


# ---------------------------------------------------------------- request planning


def plan_windows(days_by_venue: dict[str, set], window_days: int, last_available: date) -> tuple[list[tuple], list[tuple]]:
    """Group the days each venue needs into requests of at most window_days. -> (windows, deferred days)."""
    windows, deferred = [], []
    for venue_id in sorted(days_by_venue):
        cur = None
        for d in sorted(days_by_venue[venue_id]):
            if d > last_available:
                deferred.append((venue_id, d))
                continue
            if cur and (d - cur[0]).days + 1 <= window_days:
                cur = (cur[0], d)
            else:
                if cur:
                    windows.append((venue_id, *cur))
                cur = (d, d)
        if cur:
            windows.append((venue_id, *cur))
    return windows, deferred


def weather_file(venue_id: str, start: date, end: date) -> str:
    return f"{venue_id}__{start.isoformat()}__{end.isoformat()}.json.zst"


# ---------------------------------------------------------------- responses


def parse_response(resp: dict) -> list[tuple[datetime, tuple]]:
    """Hourly rows as (naive UTC datetime, values). Refuses a series that is not a gap-free fixed-offset grid."""
    hourly = resp["hourly"]
    offset = timedelta(seconds=int(resp.get("utc_offset_seconds") or 0))
    times = [datetime.fromisoformat(t) - offset for t in hourly["time"]]
    for a, b in zip(times, times[1:]):
        if b - a != HOUR:
            raise ValueError(f"Open-Meteo series is not hourly at {a}->{b}; cannot normalise to UTC")
    cols = [hourly.get(v) or [None] * len(times) for v in WX_VARS]
    return [(t, tuple(c[i] for c in cols)) for i, t in enumerate(times)]


def to_local(utc_naive: datetime, tz: str) -> datetime:
    return utc_naive.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz)).replace(tzinfo=None)


def local_to_utc(local_naive: datetime, tz: str) -> datetime:
    return local_naive.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------- playing windows


def ball_clusters(timestamps: list[datetime]) -> list[tuple[datetime, datetime, int]]:
    """Split a match's ball times (UTC) into playing days: a gap of more than 6 hours starts a new day."""
    ts = sorted(timestamps)
    if not ts:
        return []
    out = [[ts[0], ts[0], 1]]
    for t in ts[1:]:
        if t - out[-1][1] > CLUSTER_GAP:
            out.append([t, t, 1])
        else:
            out[-1][1] = t
            out[-1][2] += 1
    return [tuple(c) for c in out]


def day_windows(dates: list[date], klass: str, tz: str | None, clusters: list[tuple]) -> list[dict]:
    """One window per match date: from ball timestamps when a plausible cluster starts on that local date,
    else the format default. Without a timezone no window can be placed (weather needs coordinates anyway)."""
    by_date = {}
    if tz:
        for start, end, n in clusters:
            hours = (end - start).total_seconds() / 3600
            if n < MIN_BALLS_PER_DAY or hours > MAX_DAY_HOURS[klass]:
                continue
            d = to_local(start, tz).date()
            if d in dates and d not in by_date:
                by_date[d] = (start, end + timedelta(minutes=5), n)  # the last ball takes a few minutes too
    out = []
    lo, hi = DEFAULT_WINDOWS[klass]
    for i, d in enumerate(sorted(dates), start=1):
        row = {"match_day_number": i, "match_date": d, "window_start_utc": None, "window_end_utc": None,
               "window_source": None, "n_balls_timed": None}
        if d in by_date:
            start, end, n = by_date[d]
            row.update(window_start_utc=start, window_end_utc=end, window_source="ball_timestamps", n_balls_timed=n)
        elif tz:
            start = local_to_utc(datetime.combine(d, lo), tz)
            end = local_to_utc(datetime.combine(d, hi), tz)
            row.update(window_start_utc=start, window_end_utc=end, window_source="default_estimate")
        out.append(row)
    return out


def _floor(t: datetime, grid: int = 0) -> datetime:
    """Latest point of the hourly grid (minute `grid` past each hour) at or before t."""
    f = t.replace(minute=grid, second=0, microsecond=0)
    return f if f <= t else f - HOUR


def _ceil(t: datetime, grid: int = 0) -> datetime:
    f = _floor(t, grid)
    return f if f == t else f + HOUR


def series_grid(series: dict) -> int:
    """Minute past the hour of a venue's UTC series. Open-Meteo labels hours in local time, so a +5:30 or +5:45
    zone gives UTC stamps at :30 or :15. One venue's series must sit on one grid."""
    minutes = {t.minute for t in series}
    if len(minutes) > 1:
        raise ValueError(f"weather series mixes hourly grids {sorted(minutes)}")
    return minutes.pop() if minutes else 0


def window_hours(start: datetime, end: datetime, grid: int = 0) -> tuple[list[datetime], list[datetime], list[datetime]]:
    """-> (instant hours, precipitation hours, pre-match precipitation hours), all UTC, on the series grid."""
    inst, t = [], _floor(start, grid)
    while t <= _ceil(end, grid):
        inst.append(t)
        t += HOUR
    precip, t = [], _floor(start, grid) + HOUR
    while t < end + HOUR:
        precip.append(t)
        t += HOUR
    last_pre = _floor(start, grid)
    pre = [last_pre - k * HOUR for k in range(11, -1, -1)]
    return inst, precip, pre


def circular_mean_deg(values: list[float]) -> float | None:
    if not values:
        return None
    s = sum(math.sin(math.radians(v)) for v in values)
    c = sum(math.cos(math.radians(v)) for v in values)
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return None
    return math.degrees(math.atan2(s, c)) % 360.0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(series: dict[datetime, tuple], start: datetime, end: datetime, grid: int | None = None) -> dict:
    """Window aggregates over available hours. series: UTC hour -> values tuple (WX_COLS order)."""
    inst, precip_h, pre_h = window_hours(start, end, series_grid(series) if grid is None else grid)
    rows = [series[t] for t in inst if t in series]
    col = {c: [r[i] for r in rows] for i, c in enumerate(WX_COLS)}
    rain = [series[t][PRECIP] for t in precip_h if t in series and series[t][PRECIP] is not None]
    pre = [series[t][PRECIP] for t in pre_h if t in series and series[t][PRECIP] is not None]
    depression = [a - b for a, b in zip(col["temperature_c"], col["dew_point_c"]) if a is not None and b is not None]
    return {
        "window_hours_expected": len(inst), "window_hours_present": len(rows),
        "window_temperature_c": _mean(col["temperature_c"]),
        "window_max_temperature_c": max((x for x in col["temperature_c"] if x is not None), default=None),
        "window_humidity_pct": _mean(col["relative_humidity_pct"]),
        "window_dew_point_c": _mean(col["dew_point_c"]),
        "window_dew_point_depression_c": _mean(depression),
        "window_cloud_cover_pct": _mean(col["cloud_cover_pct"]),
        "window_wind_speed_kmh": _mean(col["wind_speed_kmh"]),
        "window_wind_direction_deg": circular_mean_deg([x for x in col["wind_direction_deg"] if x is not None]),
        "window_station_pressure_hpa": _mean(col["surface_pressure_hpa"]),
        "window_precipitation_mm": sum(rain) if len(rain) == len(precip_h) else None,
        "window_precip_hours_present": len(rain),
        "window_max_hourly_precipitation_mm": max(rain) if rain else None,
        "window_hours_with_rain": sum(1 for x in rain if x >= 0.1) if rain else None,
        "pre_match_12h_precipitation_mm": sum(pre) if len(pre) == len(pre_h) else None,
        "pre_match_hours_present": len(pre),
    }
