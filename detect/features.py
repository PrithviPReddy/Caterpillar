"""Minute-level aggregation and ML feature construction.

Shared by live detection and by the training pipeline so features are identical.
"""

import math

WORK_STATES = ("DIG", "SWING_LOADED", "DUMP", "SWING_EMPTY")
AGG_FIELDS = ["engine_load_pct", "engine_rpm", "coolant_temp_c", "oil_pressure_kpa", "oil_temp_c",
              "hyd_oil_temp_c", "egt_c", "fuel_rate_lph", "ambient_temp_c", "battery_v",
              "air_filter_dp_kpa", "control_jerk", "water_in_fuel_pct", "cab_temp_c"]


class MinuteAggregator:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.cur = None
        self._reset()

    def _reset(self):
        self.n = 0
        self.n_on = 0
        self.n_work = 0
        self.n_idle = 0
        self.sums = {}
        self.cnt = {}
        self.end_stops = 0
        self.impacts = 0
        self.last = None
        self.max_af = 0.0

    def add(self, tel):
        minute = tel["ts"][:16]
        rec = None
        if self.cur is not None and minute != self.cur:
            rec = self._flush()
        if self.cur != minute:
            self.cur = minute
        self.n += 1
        self.last = tel
        if tel.get("engine_on"):
            self.n_on += 1
            st = tel.get("state")
            if st in WORK_STATES:
                self.n_work += 1
            elif st == "IDLE":
                self.n_idle += 1
            for f in AGG_FIELDS:
                v = tel.get(f)
                if v is not None:
                    self.sums[f] = self.sums.get(f, 0.0) + v
                    self.cnt[f] = self.cnt.get(f, 0) + 1
            af = tel.get("air_filter_dp_kpa")
            if af is not None and tel.get("engine_rpm", 0) > 1700:
                self.max_af = max(self.max_af, af)
        if tel.get("end_stop_hit"):
            self.end_stops += 1
        if tel.get("impact_g", 0) > 2.5:
            self.impacts += 1
        return rec

    def _flush(self):
        last = self.last or {}
        rec = {
            "minute": self.cur,
            "machine_id": self.machine_id,
            "n": self.n,
            "running_frac": self.n_on / max(self.n, 1),
            "work_frac": self.n_work / max(self.n, 1),
            "idle_frac": self.n_idle / max(self.n, 1),
            "end_stops": self.end_stops,
            "impacts": self.impacts,
            "fuel_level_pct": last.get("fuel_level_pct"),
            "engine_hours": last.get("engine_hours"),
            "operator_id": last.get("operator_id"),
            "task_id": last.get("task_id"),
            "task_done_m3": last.get("task_done_m3"),
            "af_peak_kpa": self.max_af or None,
        }
        for f in AGG_FIELDS:
            c = self.cnt.get(f, 0)
            rec[f] = self.sums[f] / c if c else None
        self._reset()
        return rec


def _ewm(prev, x, tau_min):
    if prev is None:
        return x
    a = 1.0 - math.exp(-1.0 / tau_min)
    return prev + a * (x - prev)


class HealthFeatureState:
    """Per-machine running state that turns minute records into model features."""

    def __init__(self):
        self.l3 = self.l8 = self.l20 = None
        self.a15 = None
        self.relief = 0.0
        self.run_min = 0

    def update(self, rec):
        if rec["running_frac"] < 0.95 or rec["engine_load_pct"] is None:
            if rec["running_frac"] < 0.05:
                self.run_min = 0
                self.l3 = self.l8 = self.l20 = None
            return None
        self.run_min += 1
        ld = rec["engine_load_pct"]
        self.l3 = _ewm(self.l3, ld, 3.0)
        self.l8 = _ewm(self.l8, ld, 8.0)
        self.l20 = _ewm(self.l20, ld, 20.0)
        self.relief = _ewm(self.relief, rec["end_stops"], 10.0)
        self.a15 = _ewm(self.a15, rec["ambient_temp_c"], 15.0)
        return {
            "load": ld,
            "load_ewm3": self.l3,
            "load_ewm8": self.l8,
            "load_ewm20": self.l20,
            "rpm": rec["engine_rpm"],
            "ambient": rec["ambient_temp_c"],
            "ambient_ewm15": self.a15,
            "oil_temp": rec["oil_temp_c"],
            "relief_ewm": self.relief,
            "run_min": min(self.run_min, 120),
        }


# target -> (feature list, direction of concern, human label, unit)
HEALTH_TARGETS = {
    "coolant_temp_c": (["load_ewm3", "load_ewm8", "ambient_ewm15", "rpm", "run_min"], +1, "Coolant temperature",
                       "°C"),
    "oil_pressure_kpa": (["rpm", "oil_temp"], -1, "Oil pressure", "kPa"),
    "hyd_oil_temp_c": (["load_ewm8", "load_ewm20", "ambient_ewm15", "relief_ewm", "run_min"], +1,
                       "Hydraulic oil temperature", "°C"),
    "fuel_rate_lph": (["load", "rpm"], +1, "Fuel rate", "L/h"),
    "egt_c": (["load", "ambient", "rpm"], +1, "Exhaust gas temperature", "°C"),
}


def scoreable(rec, feats):
    """Only score steady, warm operation (thermostat-regulated regime)."""
    return (feats is not None and feats["run_min"] >= 25 and rec.get("coolant_temp_c") is not None
            and rec["coolant_temp_c"] > 78.0)
