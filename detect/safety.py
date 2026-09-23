"""Real-time safety rules. Severity depends on what the machine is doing (context)."""

import math

from detect.features import WORK_STATES
from detect.manager import Finding
from sim.config import EXCAVATOR_SPEC, FUEL_PRICE_PER_L, SAFETY
from sim.site import geofences_at, point_in_polygon

MOVING = WORK_STATES + ("TRAVEL", "HAULING", "RETURNING")


class SafetyDetector:
    period_s = 1

    def run(self, p, v, now, site):
        tel = v.tel
        if tel is None or not tel["engine_on"]:
            return []
        out = []
        state = tel["state"]
        moving = state in MOVING
        out += self._seatbelt(v, tel, state, moving)
        out += self._unattended(v, tel)
        out += self._proximity(v, tel, state, moving, site)
        if v.kind == "excavator":
            out += self._stability(v, tel)
            out += self._geofences(v, tel, state)
        else:
            out += self._overspeed(v, tel)
        return out

    def _seatbelt(self, v, tel, state, moving):
        s = v.belt_off_s
        if s <= 0:
            return []
        if moving and s >= SAFETY["seatbelt_grace_s"]:
            sev = "critical"
        elif s >= 120:
            sev = "info"
        else:
            return []
        ctx = "while the machine is working/moving" if moving else "while idling"
        return [Finding(
            key=f"{v.id}:SEATBELT_UNFASTENED", machine_id=v.id, type="SEATBELT_UNFASTENED", category="safety",
            level="L5_contextual", severity=sev, title="Seatbelt unfastened",
            summary=f"Operator seated with seatbelt unfastened for {s} s {ctx} (state {state}).",
            evidence={"seatbelt_fastened": {"value": False}, "seat_occupied": {"value": True},
                      "machine_state": {"value": state}, "duration_s": {"value": s, "unit": "s"}},
            clear_after_s=45)]

    def _unattended(self, v, tel):
        s = v.unattended_s
        if s < SAFETY["unattended_after_s"]:
            return []
        litres = v.unattended_fuel_l
        return [Finding(
            key=f"{v.id}:UNATTENDED_MACHINE", machine_id=v.id, type="UNATTENDED_MACHINE", category="safety",
            level="L5_contextual", severity="warning", title="Machine left running unattended",
            summary=(f"Engine running with nobody in the seat for {s / 60:.0f} min "
                     f"({litres:.1f} L burned, ${litres * FUEL_PRICE_PER_L:.2f})."),
            evidence={"seat_occupied": {"value": False}, "engine_on": {"value": True},
                      "minutes_unattended": {"value": round(s / 60, 1), "unit": "min"},
                      "idle_fuel_l": {"value": round(litres, 2), "unit": "L"}},
            clear_after_s=5, update_token=int(s // 600), update_min_s=600)]

    def _proximity(self, v, tel, state, moving, site):
        out = []
        mx, my = tel["x"], tel["y"]
        for w in site["workers"]:
            d = math.hypot(w["x"] - mx, w["y"] - my)
            if d > 60:
                continue
            if v.kind == "excavator":
                radius = max(tel.get("reach_m", 5.0), EXCAVATOR_SPEC["tail_swing_m"]) + SAFETY["proximity_buffer_m"]
                if not moving:
                    continue
                if d < radius:
                    sev = "critical"
                elif d < radius + SAFETY["proximity_warn_extra_m"]:
                    sev = "warning"
                else:
                    continue
            else:
                spd = tel.get("travel_speed_kmh", 0.0)
                if spd < 3:
                    continue
                radius = 6.0
                if d < radius:
                    sev = "critical"
                elif d < 12.0:
                    sev = "warning"
                else:
                    continue
            etype = "PROXIMITY_DANGER_ZONE" if sev == "critical" else "PROXIMITY_WARNING"
            where = "inside" if sev == "critical" else "approaching"
            out.append(Finding(
                key=f"{v.id}:PROXIMITY:{w['id']}", machine_id=v.id, type=etype, category="safety",
                level="L5_contextual", severity=sev,
                title=f"Person {where} the danger zone",
                summary=(f"{w['role']} {w['name']} ({w['id']}) is {d:.1f} m from {v.id} while it is {state}; "
                         f"danger radius {radius:.1f} m."),
                evidence={"distance_m": {"value": round(d, 1), "unit": "m", "threshold": round(radius, 1)},
                          "worker_id": {"value": w["id"]}, "worker_role": {"value": w["role"]},
                          "machine_state": {"value": state},
                          "reach_m": {"value": tel.get("reach_m"), "unit": "m"},
                          "swing_speed_dps": {"value": tel.get("swing_speed_dps"), "unit": "°/s"},
                          "travel_speed_kmh": {"value": tel.get("travel_speed_kmh"), "unit": "km/h"}},
                clear_after_s=60, capture_incident=(sev == "critical")))
        return out

    def _stability(self, v, tel):
        out = []
        lm = max(v.recent("load_moment_pct", 5) or [0.0])
        if lm >= SAFETY["load_moment_warn_pct"]:
            sev = "critical" if lm >= SAFETY["load_moment_crit_pct"] else "warning"
            payload = max(v.recent("payload_t", 8) or [0.0])
            out.append(Finding(
                key=f"{v.id}:STABILITY_LIMIT", machine_id=v.id, type="STABILITY_LIMIT", category="safety",
                level="L5_contextual", severity=sev, title="Tipping margin low",
                summary=(f"Load moment {lm:.0f}% of the slope-compensated tipping limit (reach {tel['reach_m']:.1f} m, "
                         f"payload {payload:.1f} t, pitch {tel['pitch_deg']:.1f}°, roll {tel['roll_deg']:.1f}°, "
                         f"swing {tel['swing_angle_deg']:.0f}°)."),
                evidence={"load_moment_pct": {"value": round(lm, 1), "unit": "%",
                                              "threshold": SAFETY["load_moment_crit_pct"]},
                          "reach_m": {"value": tel["reach_m"], "unit": "m"},
                          "payload_t": {"value": round(payload, 2), "unit": "t"},
                          "pitch_deg": {"value": tel["pitch_deg"], "unit": "°"},
                          "roll_deg": {"value": tel["roll_deg"], "unit": "°"},
                          "swing_angle_deg": {"value": tel["swing_angle_deg"], "unit": "°"}},
                clear_after_s=600, capture_incident=(sev == "critical")))
        tilt = max(abs(v.median("pitch_deg", 10) or 0), abs(v.median("roll_deg", 10) or 0))
        if tilt >= SAFETY["tilt_warn_deg"]:
            out.append(Finding(
                key=f"{v.id}:SLOPE_TILT", machine_id=v.id, type="SLOPE_TILT", category="safety",
                level="L1_threshold", severity="warning", title="Machine on a steep slope",
                summary=f"Chassis tilt {tilt:.1f}° exceeds {SAFETY['tilt_warn_deg']:.0f}°.",
                evidence={"tilt_deg": {"value": round(tilt, 1), "unit": "°", "threshold": SAFETY["tilt_warn_deg"]}},
                clear_after_s=60))
        return out

    def _geofences(self, v, tel, state):
        out = []
        pos = (tel["x"], tel["y"])
        for gf in geofences_at(pos, buffer_m=3.0):
            if gf["type"] == "overhead_powerline":
                h = max(v.recent("max_height_m", 3) or [0.0])
                lim = gf["max_height_m"]
                if h > lim:
                    sev = "critical"
                elif h > lim - SAFETY["powerline_warn_margin_m"]:
                    sev = "warning"
                else:
                    continue
                out.append(Finding(
                    key=f"{v.id}:POWERLINE_CLEARANCE", machine_id=v.id, type="POWERLINE_CLEARANCE",
                    category="safety", level="L5_contextual", severity=sev,
                    title="Boom too close to overhead power line",
                    summary=(f"Highest point of the front linkage at {h:.1f} m inside {gf['name']} corridor "
                             f"(limit {lim:.1f} m). Arc-over is possible without contact."),
                    evidence={"max_height_m": {"value": round(h, 2), "unit": "m", "threshold": lim},
                              "geofence_id": {"value": gf["id"]}, "machine_state": {"value": state}},
                    clear_after_s=60, capture_incident=(sev == "critical")))
            elif gf["type"] == "buried_utility" and state == "DIG" and point_in_polygon(pos, gf["polygon"]):
                out.append(Finding(
                    key=f"{v.id}:NO_DIG_ZONE", machine_id=v.id, type="NO_DIG_ZONE", category="safety",
                    level="L5_contextual", severity="critical", title="Digging over a buried utility",
                    summary=f"Bucket engaged inside {gf['name']} geofence.",
                    evidence={"geofence_id": {"value": gf["id"]}}, clear_after_s=60, capture_incident=True))
        return out

    def _overspeed(self, v, tel):
        spd = tel.get("travel_speed_kmh", 0.0)
        pos = (tel["x"], tel["y"])
        limit, zone = 30.0, "haul road"
        for gf in geofences_at(pos):
            if gf["type"] == "speed_zone":
                limit, zone = gf["speed_limit_kmh"], gf["name"]
        if spd > limit + 2:
            v.overspeed_s += 1
            v.overspeed_max = max(v.overspeed_max, spd)
        else:
            v.overspeed_s = 0
            v.overspeed_max = 0.0
            return []
        if v.overspeed_s < 3:
            return []
        sev = "critical" if spd > limit + 10 else "warning"
        return [Finding(
            key=f"{v.id}:OVERSPEED", machine_id=v.id, type="OVERSPEED", category="safety", level="L5_contextual",
            severity=sev, title=f"Speeding in {zone}",
            summary=f"{v.id} at {spd:.0f} km/h (max {v.overspeed_max:.0f}) in {zone}, limit {limit:.0f} km/h.",
            evidence={"travel_speed_kmh": {"value": round(spd, 1), "unit": "km/h", "threshold": limit},
                      "max_speed_kmh": {"value": round(v.overspeed_max, 1), "unit": "km/h"},
                      "payload_t": {"value": tel.get("payload_t"), "unit": "t"}, "zone": {"value": zone}},
            clear_after_s=600, update_token=int(v.overspeed_max // 5), update_min_s=120)]
