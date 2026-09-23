"""L5 operator behaviour: warm-up, cool-down, harsh use, idling (with cause),
fatigue and heat stress, and the end-of-shift scorecard."""

import statistics
from datetime import timedelta


from detect.manager import Finding
from detect.playbook import TRAINING_MODULES, training_for
from sim.config import CO2_KG_PER_L_DIESEL, FUEL_PRICE_PER_L
from sim.operators import OPERATORS


def _op_base(p, op_id, key, fleet_key=None):
    ob = p.models.baselines.get("operators", {}).get(op_id or "", {})
    if key in ob:
        return ob[key]
    return p.models.baselines.get("fleet", {}).get(fleet_key or key)


class BehaviorDetector:
    period_s = 5

    def run(self, p, v, now, site):
        tel = v.tel
        if tel is None or v.kind != "excavator":
            return []
        out = []
        tel.get("operator_id") or v.plan.get("operator_id")
        # warm-up abuse after a cold start
        if not v.warmup_flagged and v.warmup_hot_s >= 20:
            v.warmup_flagged = True
            out.append(Finding(
                key=f"{v.id}:WARMUP_VIOLATION", machine_id=v.id, type="WARMUP_VIOLATION",
                category="operator_behavior", level="L5_contextual", severity="warning",
                title="Heavy work on a cold engine",
                summary=(f"Engine at >70% load for {v.warmup_hot_s} s within {v.run_s / 60:.0f} min of a cold start, "
                         f"coolant only {tel['coolant_temp_c']:.0f} °C and oil {tel['oil_temp_c']:.0f} °C. Cold, thick "
                         f"oil accelerates bearing and turbo wear."),
                evidence={"coolant_temp_c": {"value": tel["coolant_temp_c"], "unit": "°C", "threshold": 60},
                          "oil_temp_c": {"value": tel["oil_temp_c"], "unit": "°C"},
                          "high_load_seconds": {"value": v.warmup_hot_s, "unit": "s"},
                          "minutes_since_start": {"value": round(v.run_s / 60, 1), "unit": "min"}},
                one_shot=True, cooldown_s=3600))
        # hot shutdown
        if v.just_stopped:
            before = [t for t in list(v.buf)[-125:-1] if t["engine_on"]]
            if before:
                idle_s = 0
                for t in reversed(before):
                    if t["engine_load_pct"] < 20:
                        idle_s += 1
                    else:
                        break
                egt = max(t["egt_c"] for t in before[-60:])
                load = sum(t["engine_load_pct"] for t in before) / len(before)
                if idle_s < 60 and (egt > 380 or load > 35):
                    out.append(Finding(
                        key=f"{v.id}:HOT_SHUTDOWN", machine_id=v.id, type="HOT_SHUTDOWN",
                        category="operator_behavior", level="L5_contextual", severity="warning",
                        title="Engine shut down hot (no cool-down)",
                        summary=(f"Engine stopped after only {idle_s} s of idle; exhaust {egt:.0f} °C and average load "
                                 f"{load:.0f}% in the last 2 min. Turbo bearings can coke without a 3-5 min cool-down."),
                        evidence={"idle_before_shutdown_s": {"value": idle_s, "unit": "s", "threshold": 180},
                                  "egt_c": {"value": round(egt), "unit": "°C"},
                                  "avg_load_last_2min_pct": {"value": round(load, 1), "unit": "%"},
                                  "coolant_temp_c": {"value": before[-1]["coolant_temp_c"], "unit": "°C"}},
                        one_shot=True, cooldown_s=1800))
        # harsh operation (rolling hour)
        worked_min = v.stats["work_s"] / 60
        if worked_min >= 30:
            es = len(v.end_stop_times)
            imp = len(v.impact_times)
            base_es = _op_base(p, None, "end_stops_per_h")
            if es >= 25 or imp >= 10:
                jerk = v.recent_jerk(30)
                out.append(Finding(
                    key=f"{v.id}:HARSH_OPERATION", machine_id=v.id, type="HARSH_OPERATION",
                    category="operator_behavior", level="L5_contextual", severity="warning",
                    title="Harsh hydraulic operation",
                    summary=(f"{es} cylinder end-stop hits and {imp} bucket impacts in the last hour (fleet median "
                             f"{base_es:.0f}/h). Running cylinders into the stops over relief heats the oil and "
                             f"shock-loads pins, bushings and welds."),
                    evidence={"end_stops_per_h": {"value": es, "unit": "/h", "baseline": base_es, "threshold": 25},
                              "impacts_per_h": {"value": imp, "unit": "/h",
                                                "baseline": _op_base(p, None, "impacts_per_h")},
                              "control_jerk": {"value": round(jerk, 3) if jerk else None,
                                               "baseline": _op_base(p, None, "jerk")},
                              "hyd_oil_temp_c": {"value": tel["hyd_oil_temp_c"], "unit": "°C"}},
                    clear_after_s=3600, update_token=es // 15, update_min_s=1800))
        # heat stress
        if v.cab_hi is not None and v.hot_cab_s >= 1800:
            sev = "critical" if v.cab_hi >= 39 else "warning"
            out.append(Finding(
                key=f"{v.id}:HEAT_STRESS", machine_id=v.id, type="HEAT_STRESS", category="safety",
                level="L5_contextual", severity=sev, title="Heat stress risk in the cab",
                summary=(f"Cab heat index {v.cab_hi:.0f} °C ({tel['cab_temp_c']:.0f} °C, {tel['humidity_pct']:.0f}% RH) "
                         f"for {v.hot_cab_s / 60:.0f}+ min. Cab A/C may be underperforming."),
                evidence={"cab_heat_index_c": {"value": round(v.cab_hi, 1), "unit": "°C", "threshold": 32},
                          "cab_temp_c": {"value": tel["cab_temp_c"], "unit": "°C"},
                          "ambient_temp_c": {"value": tel["ambient_temp_c"], "unit": "°C"},
                          "minutes_hot": {"value": round(v.hot_cab_s / 60), "unit": "min"}},
                clear_after_s=600))
        return out

    # ------------------------------------------------------------ minute-level
    def on_minute(self, p, v, rec, now):
        if v.kind != "excavator":
            return []
        out = []
        tel = v.tel
        idle = v.idle_last(60)
        mins = {k: s / 60 for k, s in idle.items()}
        if mins.get("truck_wait", 0) >= 8:
            trucks = p.truck_status()
            stuck = [t for t in trucks if not t["cycling"] and t["where"] != "park" and t["stationary_min"] >= 5]
            cycling = [t for t in trucks if t["cycling"]]
            fuel_l = v.idle_fuel["truck_wait"]
            out.append(Finding(
                key=f"{v.id}:TRUCK_WAIT_IDLE", machine_id=v.id, type="TRUCK_WAIT_IDLE", category="productivity",
                level="L5_contextual", severity="warning", title="Excavator waiting on trucks",
                summary=(f"{v.id} idled {mins['truck_wait']:.0f} of the last 60 min waiting for trucks: only "
                         f"{len(cycling)} truck(s) cycling"
                         + ("; " + ", ".join(f"{t['id']} stopped {t['stationary_min']:.0f} min at the {t['where']}"
                                             for t in stuck) if stuck else "")
                         + ". Not an operator issue: fleet balance."),
                evidence={"truck_wait_min_last_h": {"value": round(mins["truck_wait"], 1), "unit": "min",
                                                    "threshold": 8},
                          "trucks_cycling": {"value": len(cycling)},
                          "trucks_stopped": {"value": ", ".join(t["id"] for t in stuck) or None},
                          "idle_fuel_today_l": {"value": round(fuel_l, 1), "unit": "L"}},
                clear_after_s=300, update_token=len(cycling), update_min_s=600))
        if mins.get("operator_idle", 0) >= 15:
            fuel_l = v.idle_fuel["operator_idle"]
            out.append(Finding(
                key=f"{v.id}:EXCESSIVE_IDLE", machine_id=v.id, type="EXCESSIVE_IDLE", category="operator_behavior",
                level="L5_contextual", severity="warning", title="Excessive idling",
                summary=(f"Engine idled {mins['operator_idle']:.0f} of the last 60 min with the operator seated and "
                         f"no truck delay. Today's avoidable idle: {fuel_l:.1f} L "
                         f"(${fuel_l * FUEL_PRICE_PER_L:.2f}, {fuel_l * CO2_KG_PER_L_DIESEL:.0f} kg CO2)."),
                evidence={"operator_idle_min_last_h": {"value": round(mins["operator_idle"], 1), "unit": "min",
                                                       "threshold": 15}},
                clear_after_s=300))
        if rec["minute"].endswith(("0", "2", "4", "6", "8")) and tel and tel["engine_on"] and tel["seat_occupied"]:
            f = self._fatigue(p, v, now)
            if f:
                out.append(f)
        if (not v.summary_done and tel is not None and not tel["engine_on"] and v.stats["engine_on_s"] > 3600
                and now >= v.plan["shift_end"] - timedelta(minutes=5)):
            v.summary_done = True
            out.append(self._summary(p, v, now))
        return out

    def _fatigue(self, p, v, now):
        if v.in_seat_since is None:
            return None
        hours = (now - v.in_seat_since).total_seconds() / 3600
        op = v.tel.get("operator_id")
        base_jerk = _op_base(p, op, "jerk_fresh", "jerk")
        jerk = v.recent_jerk(15)
        mp = sum(1 for t in v.micro_pause_times if (now - t).total_seconds() <= 1800) * 2
        cts = v.cycle_times(15, now)
        heat = min(1.0, max(0.0, ((v.cab_hi or 25) - 27.0) / 10.0))
        h = now.hour + now.minute / 60
        circ = 1.0 if 13.5 <= h <= 16.0 else (0.5 if h < 6.5 else 0.0)
        jr = jerk / base_jerk if (jerk and base_jerk) else 1.0
        symptoms = min(1.0, max(0.0, 0.5 * (jr - 1.0) / 0.3 + 0.5 * min(1.0, mp / 12.0)))
        risk = 0.20 * min(1.0, hours / 3.0) + 0.20 * heat + 0.10 * circ + 0.50 * symptoms
        cv = statistics.pstdev(cts) / statistics.mean(cts) if len(cts) >= 8 else None
        v.derived["fatigue"] = {"risk": round(risk, 2), "hours_in_seat": round(hours, 2),
                                "jerk_ratio": round(jr, 2), "micro_pauses_per_h": mp,
                                "cab_heat_index_c": round(v.cab_hi or 0, 1), "cycle_cv": round(cv, 3) if cv else None}
        if risk < 0.55 or hours < 0.5:
            return None
        sev = "critical" if risk >= 0.8 else "warning"
        return Finding(
            key=f"{v.id}:FATIGUE_RISK", machine_id=v.id, type="FATIGUE_RISK", category="safety",
            level="L5_contextual", severity=sev, title="Operator fatigue risk rising",
            summary=(f"Fatigue risk {risk:.2f}: {hours:.1f} h in the seat since the last break, cab heat index "
                     f"{v.cab_hi:.0f} °C, control jerk {jr:.2f}x this operator's fresh baseline and {mp} "
                     f"micro-pauses/h (hesitations mid-cycle)."),
            evidence={"fatigue_risk": {"value": round(risk, 2), "threshold": 0.55},
                      "hours_in_seat": {"value": round(hours, 2), "unit": "h"},
                      "control_jerk_ratio": {"value": round(jr, 2), "baseline": 1.0},
                      "micro_pauses_per_h": {"value": mp, "unit": "/h",
                                             "baseline": _op_base(p, op, "micro_pauses_per_h")},
                      "cab_heat_index_c": {"value": round(v.cab_hi, 1), "unit": "°C"},
                      "cycle_time_cv": {"value": round(cv, 3) if cv else None}},
            clear_after_s=1800, update_token=round(risk, 1), update_min_s=900)

    def _summary(self, p, v, now):
        s = v.stats
        on_h = s["engine_on_s"] / 3600
        work_h = s["work_s"] / 3600
        idle_s = sum(s[k] for k in s if k.startswith("idle_"))
        idle_pct = 100 * idle_s / max(s["engine_on_s"], 1)
        avoidable_idle_l = v.idle_fuel["operator_idle"] + v.idle_fuel["unattended"]
        m3 = sum(t["done"] for t in p.task_progress(v).values())
        fuel = s["fuel_used_l"]
        belt = 100 * (1 - s["belt_off_moving_s"] / max(s["seated_moving_s"], 1))
        es_h = s["end_stops"] / max(work_h, 0.1)
        ec = v.event_counts
        score = 100
        score -= min(20, 4 * ec["SEATBELT_UNFASTENED"] + (100 - belt) * 0.5)
        score -= min(20, 10 * ec["PROXIMITY_DANGER_ZONE"] + 8 * ec["POWERLINE_CLEARANCE"]
                     + 6 * ec["STABILITY_LIMIT"])
        score -= min(15, max(0.0, es_h - 12) * 0.5)
        score -= 5 * (ec["WARMUP_VIOLATION"] + ec["HOT_SHUTDOWN"])
        score -= min(10, max(0.0, idle_pct - 20) * 0.5 + 3 * ec["UNATTENDED_MACHINE"])
        score = max(0, round(score))
        rec_mods = {}
        for etype, n in ec.items():
            tm = training_for(etype)
            if tm and n:
                rec_mods[tm["id"]] = tm["title"]
        if es_h > 25:
            rec_mods["TM-03"] = TRAINING_MODULES["TM-03"]
        op = v.plan.get("operator_id")
        opname = OPERATORS[op].name if op in OPERATORS else op
        items = ", ".join(f"{k} {t}" for k, t in rec_mods.items()) or "none"
        return Finding(
            key=f"{v.id}:SHIFT_SUMMARY:{now:%Y%m%d}", machine_id=v.id, type="SHIFT_SUMMARY",
            category="operator_behavior", level="L5_contextual", severity="info",
            title=f"Shift scorecard for {opname}: {score}/100",
            summary=(f"{on_h:.1f} engine h, {m3:.0f} m³ moved, {fuel:.0f} L fuel ({fuel / max(m3, 1):.2f} L/m³), "
                     f"idle {idle_pct:.0f}% ({avoidable_idle_l:.1f} L avoidable), seatbelt compliance {belt:.0f}%, "
                     f"{es_h:.0f} end-stops/h. Recommended training: {items}."),
            evidence={"score": {"value": score, "unit": "/100"},
                      "engine_hours": {"value": round(on_h, 2), "unit": "h"},
                      "volume_m3": {"value": round(m3, 1), "unit": "m³"},
                      "fuel_l": {"value": round(fuel, 1), "unit": "L"},
                      "fuel_per_m3": {"value": round(fuel / max(m3, 1), 3), "unit": "L/m³"},
                      "idle_pct": {"value": round(idle_pct, 1), "unit": "%"},
                      "avoidable_idle_l": {"value": round(avoidable_idle_l, 1), "unit": "L"},
                      "seatbelt_compliance_pct": {"value": round(belt, 1), "unit": "%"},
                      "end_stops_per_h": {"value": round(es_h, 1), "unit": "/h",
                                          "baseline": _op_base(p, None, "end_stops_per_h")},
                      "safety_events": {"value": int(ec["PROXIMITY_DANGER_ZONE"] + ec["POWERLINE_CLEARANCE"]
                                                     + ec["STABILITY_LIMIT"] + ec["SEATBELT_UNFASTENED"])},
                      "training_modules": {"value": ", ".join(rec_mods) or None}},
            one_shot=True, cooldown_s=86400)
