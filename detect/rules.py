"""L1 thresholds, L2 trends, sensor health, security and site-weather rules."""

from datetime import timedelta

import numpy as np

from detect.manager import Finding, fmt_hm
from sim.config import CHANNELS, OIL_PRESSURE_MIN_KPA, SAFETY, THRESHOLDS

THRESH_TYPES = {
    "coolant_temp_c": ("COOLANT_TEMP_HIGH", "machine_health", "Coolant temperature high", 1.5),
    "hyd_oil_temp_c": ("HYD_OIL_TEMP_HIGH", "machine_health", "Hydraulic oil temperature high", 1.5),
    "egt_c": ("EGT_HIGH", "machine_health", "Exhaust temperature high", 15.0),
    "fuel_level_pct": ("FUEL_LOW", "fuel_energy", "Fuel level low", 1.0),
    "def_level_pct": ("DEF_LOW", "fuel_energy", "DEF level low", 1.0),
    "water_in_fuel_pct": ("WATER_IN_FUEL", "machine_health", "Water in fuel separator", 3.0),
    "air_filter_dp_kpa": ("AIR_FILTER_RESTRICTED", "machine_health", "Air filter restricted", 0.2),
    "battery_v": ("BATTERY_VOLTAGE_LOW", "machine_health", "System voltage low", 0.3),
    "trans_pressure_kpa": ("TRANSMISSION_PRESSURE_LOW", "machine_health", "Transmission pressure low", 100.0),
}
EVALUATE_WHEN_OFF = ("fuel_level_pct", "def_level_pct", "water_in_fuel_pct")


def _unit(ch):
    return CHANNELS.get(ch, ("",))[0]


def _spn(ch):
    return CHANNELS.get(ch, (None, None, None))[2]


class ThresholdDetector:
    period_s = 5

    def run(self, p, v, now, site):
        tel = v.tel
        if tel is None:
            return []
        on = tel["engine_on"]
        out = []
        for ch, th in THRESHOLDS.items():
            if tel.get(ch) is None:
                continue
            if not on and ch not in EVALUATE_WHEN_OFF:
                continue
            if on and v.run_s < 30:
                continue
            if ch == "air_filter_dp_kpa" and tel.get("engine_rpm", 0) < 1700:
                continue
            etype, cat, label, hyst = THRESH_TYPES[ch]
            key = f"{v.id}:{etype}"
            h = hyst if p.manager.is_active(key) else 0.0
            val = v.median(ch, 10)
            sev = lim = None
            fmi = None
            if "crit_high" in th:
                if val >= th["crit_high"] - h:
                    sev, lim, fmi = "critical", th["crit_high"], 0
                elif val >= th["warn_high"] - h:
                    sev, lim, fmi = "warning", th["warn_high"], 16
            else:
                if val <= th["crit_low"] + h:
                    sev, lim, fmi = "critical", th["crit_low"], 1
                elif val <= th["warn_low"] + h:
                    sev, lim, fmi = "warning", th["warn_low"], 18
            if sev is None:
                continue
            unit = _unit(ch)
            ev = {ch: {"value": round(val, 2), "unit": unit, "threshold": lim, "spn": _spn(ch)}}
            if on:
                ev["engine_load_pct"] = {"value": tel.get("engine_load_pct"), "unit": "%"}
                ev["ambient_temp_c"] = {"value": tel.get("ambient_temp_c"), "unit": "°C"}
            out.append(Finding(
                key=key, machine_id=v.id, type=etype, category=cat, level="L1_threshold", severity=sev,
                title=label,
                summary=f"{label}: {val:.1f} {unit} vs {sev} limit {lim} {unit} (state {tel.get('state')}).",
                evidence=ev, clear_after_s=60, dtc={"spn": _spn(ch), "fmi": fmi},
            ))
        if on and v.run_s >= 30 and tel.get("oil_pressure_kpa") is not None:
            rpm = min(v.recent("engine_rpm", 10))
            for rpm_min, warn, crit in OIL_PRESSURE_MIN_KPA:
                if rpm >= rpm_min:
                    break
            val = v.median("oil_pressure_kpa", 10)
            sev = "critical" if val < crit else ("warning" if val < warn else None)
            if sev:
                lim = crit if sev == "critical" else warn
                out.append(Finding(
                    key=f"{v.id}:OIL_PRESSURE_LOW", machine_id=v.id, type="OIL_PRESSURE_LOW",
                    category="machine_health", level="L1_threshold", severity=sev, title="Engine oil pressure low",
                    summary=f"Oil pressure {val:.0f} kPa at {rpm:.0f} rpm, below the {lim:.0f} kPa limit.",
                    evidence={"oil_pressure_kpa": {"value": round(val, 1), "unit": "kPa", "threshold": lim,
                                                   "spn": 100},
                              "engine_rpm": {"value": rpm, "unit": "rpm"}},
                    clear_after_s=60, dtc={"spn": 100, "fmi": 1 if sev == "critical" else 18}))
        return out


TREND_SPECS = [
    # channel, direction, limit, floor (only trend above this), min rate/min, max minutes, type, label
    ("coolant_temp_c", +1, 105.0, 95.0, 0.10, 45.0, "COOLANT_TEMP_RISING", "Coolant temperature"),
    ("hyd_oil_temp_c", +1, 98.0, 88.0, 0.12, 30.0, "HYD_OIL_TEMP_RISING", "Hydraulic oil temperature"),
    ("water_in_fuel_pct", +1, 50.0, 8.0, 0.15, 60.0, "WATER_IN_FUEL_RISING", "Water separator level"),
]


class TrendDetector:
    period_s = 30

    def run(self, p, v, now, site):
        tel = v.tel
        if tel is None or not tel["engine_on"] or v.run_s < 600:
            return []
        recs = [r for r in list(v.minutes)[-12:] if r["running_frac"] >= 0.95]
        if len(recs) < 8:
            return []
        out = []
        for ch, d, limit, floor, min_rate, max_min, etype, label in TREND_SPECS:
            ys = [r.get(ch) for r in recs]
            if any(y is None for y in ys):
                continue
            x = np.arange(len(ys), dtype=float)
            slope = float(np.polyfit(x, np.array(ys), 1)[0])
            cur = v.median(ch, 20)
            if cur is None or d * slope < min_rate or d * cur < d * floor or d * (limit - cur) <= 0:
                continue
            mins = (limit - cur) / slope
            if mins > max_min:
                continue
            sev = "critical" if mins <= 10 else "warning"
            eta = now + timedelta(minutes=mins)
            unit = _unit(ch)
            out.append(Finding(
                key=f"{v.id}:{etype}", machine_id=v.id, type=etype, category="machine_health",
                level="L2_trend", severity=sev, title=f"{label} rising towards limit",
                summary=(f"{label} {cur:.1f} {unit}, rising {slope:.2f} {unit}/min. At this rate it reaches the "
                         f"{limit:g} {unit} limit in ~{mins:.0f} min (≈{fmt_hm(eta)})."),
                evidence={ch: {"value": round(cur, 2), "unit": unit, "threshold": limit,
                               "trend_per_min": round(slope, 3), "spn": _spn(ch)}},
                prediction={"kind": "time_to_threshold", "eta": eta.isoformat(timespec="seconds"),
                            "minutes_to_impact": round(mins, 1), "value": limit, "unit": unit,
                            "confidence": 0.7},
                clear_after_s=1200, update_token=int(mins // 5), update_min_s=300,
            ))
        return out


FLATLINE_CHANNELS = ["coolant_temp_c", "fuel_temp_c", "oil_temp_c", "hyd_oil_temp_c", "egt_c",
                     "oil_pressure_kpa", "battery_v", "trans_pressure_kpa"]


class SensorHealthDetector:
    period_s = 60

    def run(self, p, v, now, site):
        if len(v.buf) < 600:
            return []
        window = list(v.buf)[-600:]
        if not all(t["engine_on"] for t in window):
            return []
        load = np.array([t["engine_load_pct"] for t in window])
        if np.ptp(load) < 5:
            return []
        out = []
        amb = window[-1].get("ambient_temp_c")
        for ch in FLATLINE_CHANNELS:
            vals = [t.get(ch) for t in window]
            if any(x is None for x in vals):
                continue
            if np.ptp(np.array(vals)) > 1e-6:
                continue
            val = vals[-1]
            note = "Reading is frozen while engine load varies"
            if ch.endswith("temp_c") and amb is not None and val < amb - 10:
                note += f"; physically implausible ({val} °C with ambient {amb} °C and engine running)"
            out.append(Finding(
                key=f"{v.id}:SENSOR_FLATLINE:{ch}", machine_id=v.id, type="SENSOR_FLATLINE",
                category="sensor_health", level="L3_ml_pattern", severity="warning",
                title=f"Sensor fault suspected: {CHANNELS[ch][1]}",
                summary=f"{CHANNELS[ch][1]} has read exactly {val} {_unit(ch)} for 10+ min while the engine is "
                        f"working. {note}.",
                evidence={ch: {"value": val, "unit": _unit(ch), "spn": _spn(ch), "note": note},
                          "engine_load_range_pct": {"value": round(float(np.ptp(load)), 1), "unit": "%"}},
                clear_after_s=180, dtc={"spn": _spn(ch), "fmi": 2},
            ))
        return out


class SecurityDetector:
    period_s = 10

    def run(self, p, v, now, site):
        tel = v.tel
        out = []
        if tel is None:
            return out
        on = tel["engine_on"]
        if not on and v.fuel_off_ref is not None:
            fuel = tel["fuel_level_pct"]
            drop = v.fuel_off_ref - fuel
            recent = v.recent("fuel_level_pct", 180)
            still_falling = len(recent) > 30 and recent[0] - recent[-1] > 0.3
            if drop >= 3.0 and still_falling:
                litres = drop / 100 * v.tank_l
                mins = max((now - v.fuel_off_ref_t).total_seconds() / 60, 1)
                rate = (recent[0] - recent[-1]) / 100 * v.tank_l / (len(recent) / 60)
                out.append(Finding(
                    key=f"{v.id}:FUEL_LOSS_ENGINE_OFF", machine_id=v.id, type="FUEL_LOSS_ENGINE_OFF",
                    category="security", level="L5_contextual", severity="critical",
                    title="Fuel dropping while machine is parked",
                    summary=(f"Fuel fell from {v.fuel_off_ref:.1f}% to {fuel:.1f}% (~{litres:.0f} L) with the engine "
                             f"off since {fmt_hm(v.fuel_off_ref_t)}; currently ~{rate:.1f} L/min. No refuelling or "
                             f"engine activity explains this: possible theft or leak."),
                    evidence={"fuel_level_pct": {"value": round(fuel, 1), "unit": "%", "baseline": v.fuel_off_ref},
                              "litres_lost": {"value": round(litres), "unit": "L"},
                              "loss_rate_lpm": {"value": round(rate, 1), "unit": "L/min"},
                              "engine_on": {"value": False},
                              "minutes_since_shutdown": {"value": round(mins), "unit": "min"}},
                    clear_after_s=300, capture_incident=False, update_token=int(litres // 50), update_min_s=120,
                ))
        if v.just_started:
            plan = v.plan
            early = plan["shift_start"] - timedelta(minutes=20)
            late = plan["shift_end"] + timedelta(minutes=20)
            if now < early or now > late or tel.get("operator_id") != plan.get("operator_id"):
                out.append(Finding(
                    key=f"{v.id}:ENGINE_START_OUTSIDE_SHIFT", machine_id=v.id, type="ENGINE_START_OUTSIDE_SHIFT",
                    category="security", level="L5_contextual", severity="warning",
                    title="Engine started outside the planned shift",
                    summary=(f"Engine started at {fmt_hm(now)} by {tel.get('operator_id')}; planned shift "
                             f"{fmt_hm(plan['shift_start'])}-{fmt_hm(plan['shift_end'])} for {plan.get('operator_id')}."),
                    evidence={"operator_id": {"value": tel.get("operator_id")}}, one_shot=True))
        return out


class SiteWeatherDetector:
    """Site-wide conditions, reported against machine_id SITE."""
    period_s = 30

    def run_site(self, p, now, site):
        w = site["weather"]
        out = []
        lk = w.get("lightning_km")
        if lk is not None and lk < SAFETY["lightning_warn_km"]:
            sev = "critical" if lk < SAFETY["lightning_crit_km"] else "warning"
            out.append(Finding(
                key="SITE:LIGHTNING_NEARBY", machine_id="SITE", type="LIGHTNING_NEARBY", category="site",
                level="L5_contextual", severity=sev, title="Lightning near the site",
                summary=(f"Lightning detected {lk:.0f} km away (warn < {SAFETY['lightning_warn_km']:.0f} km, "
                         f"stop < {SAFETY['lightning_crit_km']:.0f} km). Tall booms and ground crew are most at risk."),
                evidence={"lightning_km": {"value": lk, "unit": "km", "threshold": SAFETY["lightning_crit_km"]},
                          "wind_gust_kmh": {"value": w.get("wind_gust_kmh"), "unit": "km/h"},
                          "forecast_rain_start": {"value": w.get("forecast_rain_start")}},
                clear_after_s=1800, update_token=int(lk // 3), update_min_s=300))
        g = w.get("wind_gust_kmh") or 0
        if g >= SAFETY["wind_gust_warn_kmh"]:
            out.append(Finding(
                key="SITE:HIGH_WIND", machine_id="SITE", type="HIGH_WIND", category="site", level="L1_threshold",
                severity="warning", title="High wind gusts",
                summary=f"Wind gusts {g:.0f} km/h exceed the {SAFETY['wind_gust_warn_kmh']:.0f} km/h lifting limit.",
                evidence={"wind_gust_kmh": {"value": g, "unit": "km/h", "threshold": SAFETY["wind_gust_warn_kmh"]}},
                clear_after_s=600))
        return out
