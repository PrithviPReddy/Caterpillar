"""Site geometry (local metres, +x east, +y north), weather, and ground workers."""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

ZONES = {
    "PAD_A": {"name": "Pad A - bulk excavation", "center": (0.0, 0.0), "radius": 35.0,
              "slope_deg": 1.5, "downhill_deg": 90.0},
    "ZONE_B": {"name": "Zone B - utility trench", "center": (-160.0, 90.0), "radius": 40.0,
               "slope_deg": 2.0, "downhill_deg": 0.0},
    "ZONE_C": {"name": "Zone C - batter slope", "center": (110.0, -170.0), "radius": 35.0,
               "slope_deg": 14.0, "downhill_deg": 180.0},
    "ZONE_D": {"name": "Zone D - storm drain trench", "center": (240.0, -45.0), "radius": 30.0,
               "slope_deg": 1.0, "downhill_deg": 0.0},
    "DUMP": {"name": "Spoil dump", "center": (820.0, 360.0), "radius": 45.0,
             "slope_deg": 3.0, "downhill_deg": 45.0},
    "PARK": {"name": "Parking & fuel bay", "center": (-70.0, -80.0), "radius": 25.0,
             "slope_deg": 0.5, "downhill_deg": 0.0},
}

GEOFENCES = [
    {"id": "GF-PWR-01", "type": "overhead_powerline", "name": "11 kV overhead line",
     "polygon": [(150.0, -62.0), (340.0, -62.0), (340.0, -28.0), (150.0, -28.0)],
     "max_height_m": 7.0},
    {"id": "GF-GAS-01", "type": "buried_utility", "name": "Buried gas main (no dig)",
     "polygon": [(-235.0, 38.0), (-85.0, 38.0), (-85.0, 50.0), (-235.0, 50.0)]},
    {"id": "GF-SPD-01", "type": "speed_zone", "name": "Pad A pedestrian zone",
     "polygon": [(-45.0, -45.0), (75.0, -45.0), (75.0, 55.0), (-45.0, 55.0)],
     "speed_limit_kmh": 15.0},
]

# Haul road from the Pad A loading point to the spoil dump.
HAUL_ROUTE = [(8.0, 0.0), (40.0, 25.0), (120.0, 70.0), (260.0, 140.0), (420.0, 230.0), (560.0, 380.0),
              (700.0, 420.0), (820.0, 360.0)]
HAUL_SPEED_LIMIT_KMH = 30.0

SITE_INFO = {
    "name": "Riverside Logistics Park - Phase 2 earthworks",
    "timezone": "site local time",
}


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_polygon(p, poly):
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def zone_at(p):
    best, best_d = None, 1e9
    for zid, z in ZONES.items():
        d = dist(p, z["center"])
        if d <= z["radius"] and d < best_d:
            best, best_d = zid, d
    return best


def terrain_at(p):
    """Return (slope_deg, downhill_deg) at a position."""
    zid = zone_at(p)
    if zid:
        z = ZONES[zid]
        return z["slope_deg"], z["downhill_deg"]
    return 1.0, 0.0


def geofences_at(p, buffer_m=0.0):
    hits = []
    for gf in GEOFENCES:
        if point_in_polygon(p, gf["polygon"]):
            hits.append(gf)
        elif buffer_m > 0 and _dist_to_polygon(p, gf["polygon"]) <= buffer_m:
            hits.append(gf)
    return hits


def _dist_to_polygon(p, poly):
    best = 1e9
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        best = min(best, _dist_to_segment(p, a, b))
    return best


def _dist_to_segment(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class Route:
    def __init__(self, pts):
        self.pts = pts
        self.cum = [0.0]
        for i in range(1, len(pts)):
            self.cum.append(self.cum[-1] + dist(pts[i - 1], pts[i]))
        self.length = self.cum[-1]

    def point_at(self, s):
        s = max(0.0, min(self.length, s))
        for i in range(1, len(self.pts)):
            if s <= self.cum[i]:
                seg = self.cum[i] - self.cum[i - 1]
                f = 0.0 if seg == 0 else (s - self.cum[i - 1]) / seg
                a, b = self.pts[i - 1], self.pts[i]
                return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        return self.pts[-1]


def hhmm(day: datetime, s: str) -> datetime:
    parts = [int(p) for p in s.split(":")]
    sec = parts[2] if len(parts) > 2 else 0
    return day.replace(hour=parts[0], minute=parts[1], second=sec, microsecond=0)


def _hours(t: datetime) -> float:
    return t.hour + t.minute / 60 + t.second / 3600


@dataclass
class Weather:
    """Deterministic diurnal profile plus an optional storm cell."""
    day: datetime
    t_min: float = 19.0
    t_max: float = 36.0
    rh_min: float = 30.0
    rh_max: float = 75.0
    wind_kmh: float = 12.0
    storm: dict | None = None  # {lightning_start, closest_at, closest_km, start_km, rain_start, rain_mm_h, forecast_issued}

    def __post_init__(self):
        if self.storm:
            s = self.storm
            self._l_start = hhmm(self.day, s["lightning_start"])
            self._l_close = hhmm(self.day, s["closest_at"])
            self._rain = hhmm(self.day, s["rain_start"])
            self._fc = hhmm(self.day, s.get("forecast_issued", "00:00"))

    def _diurnal(self, t):
        h = _hours(t)
        if h < 5.5:
            f = 0.15 * (5.5 - h) / 5.5
        elif h <= 15.0:
            f = (1 - math.cos(math.pi * (h - 5.5) / 9.5)) / 2
        else:
            f = (1 + math.cos(math.pi * min(h - 15.0, 14.5) / 14.5)) / 2
        return f

    def _storm_cooling(self, t):
        if not self.storm or t < self._rain - timedelta(minutes=30):
            return 0.0
        mins = (t - (self._rain - timedelta(minutes=30))).total_seconds() / 60
        return min(7.0, mins * 7.0 / 45.0)

    def ambient(self, t):
        return self.t_min + (self.t_max - self.t_min) * self._diurnal(t) - self._storm_cooling(t)

    def humidity(self, t):
        h = self.rh_max - (self.rh_max - self.rh_min) * self._diurnal(t)
        if self.storm and t >= self._rain:
            h = min(95.0, h + 35.0)
        return h

    def lightning_km(self, t):
        if not self.storm or t < self._l_start:
            return None
        s = self.storm
        if t <= self._l_close:
            span = (self._l_close - self._l_start).total_seconds()
            f = (t - self._l_start).total_seconds() / max(span, 1)
            return s["start_km"] + f * (s["closest_km"] - s["start_km"])
        mins_after = (t - self._l_close).total_seconds() / 60
        return s["closest_km"] + mins_after * 0.4

    def rain_mm_h(self, t):
        if not self.storm or t < self._rain:
            return 0.0
        return self.storm.get("rain_mm_h", 8.0)

    def gust_kmh(self, t):
        g = self.wind_kmh * 1.6
        lk = self.lightning_km(t)
        if lk is not None and lk < 15:
            g += (15 - lk) * 4.5
        return g

    def forecast(self, t):
        """What a weather service would tell the site at time t."""
        out = {"rain_start": None, "rain_mm_h": 0.0, "lightning_risk": False}
        if self.storm and t >= self._fc and t < self._rain + timedelta(hours=2):
            out["rain_start"] = self._rain
            out["rain_mm_h"] = self.storm.get("rain_mm_h", 8.0)
            out["lightning_risk"] = True
        return out

    def snapshot(self, t):
        fc = self.forecast(t)
        lk = self.lightning_km(t)
        return {
            "ambient_temp_c": round(self.ambient(t), 1),
            "humidity_pct": round(self.humidity(t), 0),
            "wind_gust_kmh": round(self.gust_kmh(t), 0),
            "rain_mm_h": round(self.rain_mm_h(t), 1),
            "lightning_km": None if lk is None else round(lk, 1),
            "forecast_rain_start": fc["rain_start"].strftime("%H:%M") if fc["rain_start"] else None,
            "forecast_lightning_risk": fc["lightning_risk"],
        }


@dataclass
class Worker:
    """Ground crew member wearing a UWB proximity tag."""
    id: str
    name: str
    role: str
    waypoints: list  # [(datetime, x, y)]
    on_site_from: datetime | None = None
    on_site_until: datetime | None = None
    pos: tuple | None = field(default=None)

    def update(self, t):
        if (self.on_site_from and t < self.on_site_from) or (self.on_site_until and t > self.on_site_until):
            self.pos = None
            return
        wps = self.waypoints
        if t <= wps[0][0]:
            self.pos = (wps[0][1], wps[0][2])
            return
        for i in range(1, len(wps)):
            if t <= wps[i][0]:
                t0, x0, y0 = wps[i - 1]
                t1, x1, y1 = wps[i]
                f = (t - t0).total_seconds() / max((t1 - t0).total_seconds(), 1)
                self.pos = (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
                return
        self.pos = (wps[-1][1], wps[-1][2])
