"""L3 ML health monitoring.

Per-signal regression models learned from normal operation predict what each
signal *should* read given load, speed, ambient and warm-up. Persistent positive
(or negative) residuals flag degradation long before fixed alarm limits, and the
pattern of residuals points at a probable cause.
"""

import math

from detect.features import HEALTH_TARGETS, scoreable
from detect.manager import Finding

Z_ALERT = 3.0
STREAK_MIN = 3
EWM_A = 0.3
# smallest deviation that matters physically, regardless of how precise the model is
MIN_ABS = {"coolant_temp_c": 2.0, "oil_pressure_kpa": 20.0, "hyd_oil_temp_c": 5.0, "fuel_rate_lph": 1.2,
           "egt_c": 35.0}

TYPE_FOR = {
    "coolant_temp_c": "COOLING_DEGRADATION_DETECTED",
    "oil_pressure_kpa": "OIL_PRESSURE_DEVIATION",
    "hyd_oil_temp_c": "HYD_TEMP_DEVIATION",
    "fuel_rate_lph": "FUEL_EFFICIENCY_DEGRADATION",
    "egt_c": "FUEL_EFFICIENCY_DEGRADATION",
}


def _diagnose(dz):
    """dz: target -> direction-adjusted z (positive = bad)."""
    c, h, f, e, o = (dz.get(k, 0.0) for k in ("coolant_temp_c", "hyd_oil_temp_c", "fuel_rate_lph", "egt_c",
                                                  "oil_pressure_kpa"))
    if c >= Z_ALERT and f < 2 and e < 2:
        extra = " Hydraulic oil also running warm - shared cooler package likely fouled." if h >= 2 else ""
        return ("Cooling system losing capacity (radiator/cooler core fouling, fan drive or coolant level). "
                "Engine load and ambient are normal for this temperature." + extra)
    if o >= Z_ALERT:
        return "Oil supply degrading at normal speed and oil temperature (pump wear, leak or fuel dilution)."
    if f >= Z_ALERT or e >= Z_ALERT:
        return "More fuel and hotter exhaust for the same work: combustion efficiency loss (injectors, turbo, intake)."
    if h >= Z_ALERT:
        return "Hydraulic oil hotter than expected for the duty cycle (cooler, relief valve or oil level)."
    return "Signal deviates from the learned normal behaviour."


class HealthMonitor:
    def on_minute(self, p, v, rec, now):
        feats = v.hfs.update(rec)
        store = p.models
        if rec["running_frac"] < 0.05:
            v.res_ewm.clear()
            v.res_streak.clear()
            v.last_expected.clear()
            v.health_score = None
        if not store.has_health or v.kind != "excavator" or not scoreable(rec, feats):
            rec["health_scored"] = False
            return []
        rec["health_scored"] = True
        dz = {}
        zvec = []
        for target, (flist, direction, label, unit) in HEALTH_TARGETS.items():
            exp, sigma = store.expected(target, feats)
            act = rec[target]
            z = (act - exp) / sigma
            prev = v.res_ewm.get(target, 0.0)  # EWMA of the residual, in signal units
            ew = prev + EWM_A * ((act - exp) - prev)
            v.res_ewm[target] = ew
            v.last_expected[target] = (round(exp, 2), round(sigma, 3))
            rec[f"exp_{target}"] = exp
            rec[f"sig_{target}"] = sigma
            dz[target] = direction * ew / sigma
            zvec.append(z)
            bad = direction * ew >= max(Z_ALERT * sigma, MIN_ABS[target])
            v.res_streak[target] = v.res_streak[target] + 1 if bad else 0
        a = store.anomaly_score(zvec)
        # worst deviation as a multiple of its alert floor (1.0 = at the alert line)
        r = max(HEALTH_TARGETS[t][1] * v.res_ewm[t] / max(Z_ALERT * v.last_expected[t][1], MIN_ABS[t])
                for t in HEALTH_TARGETS)
        score = 100.0 * math.exp(-0.35 * max(0.0, r - 0.6))
        if a is not None:
            score = min(score, 100.0 * (1.0 - 0.5 * a))
        v.health_score = round(max(5.0, score))
        rec["health_score"] = v.health_score
        out = []
        fired = set()
        for target, (flist, direction, label, unit) in HEALTH_TARGETS.items():
            if v.res_streak[target] < STREAK_MIN:
                continue
            for t2 in HEALTH_TARGETS:  # only list co-deviations that clear their own floor
                if t2 != target and v.res_streak[t2] == 0:
                    dz[t2] = min(dz[t2], Z_ALERT - 0.01)
            etype = TYPE_FOR[target]
            if etype in fired:
                continue
            fired.add(etype)
            exp, sigma = v.last_expected[target]
            act = rec[target]
            evidence = {}
            for t2, (_, d2, lab2, u2) in HEALTH_TARGETS.items():
                e2, s2 = v.last_expected[t2]
                evidence[t2] = {"value": round(rec[t2], 2), "unit": u2, "expected": e2,
                                "deviation_sigma": round(d2 * dz[t2], 2)}
            evidence["engine_load_pct"] = {"value": round(rec["engine_load_pct"], 1), "unit": "%",
                                           "note": "last-minute average; normal duty"}
            evidence["ambient_temp_c"] = {"value": round(rec["ambient_temp_c"], 1), "unit": "°C"}
            gap = act - exp
            out.append(Finding(
                key=f"{v.id}:{etype}", machine_id=v.id, type=etype, category="machine_health",
                level="L3_ml_pattern", severity="warning",
                title=f"Early warning: {label.lower()} abnormal for current conditions",
                summary=(f"{label} {act:.1f} {unit} vs {exp:.1f} {unit} expected for this load and ambient "
                         f"({gap:+.1f} {unit}, {dz[target]:.1f}σ, sustained {v.res_streak[target]} min). "
                         f"Fixed alarm limits not reached yet. {_diagnose(dz)}"),
                evidence=evidence, clear_after_s=2700, update_token=int(dz[target] // 10), update_min_s=1200))
        return out
