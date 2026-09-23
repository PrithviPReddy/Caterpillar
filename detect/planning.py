"""L4 predictive planning: will today's plan actually work?

Combines the plan, the ML task-duration model, live progress, burn rates, service
intervals, the weather forecast and maintenance history.
"""

from datetime import timedelta

import numpy as np

from detect.manager import Finding, fmt_hm
from sim.config import EXCAVATOR_SPEC
from sim.operators import OPERATORS

RESERVE_PCT = 5.0
AF_LIMIT_KPA = 6.2


def _segments(plan, start, end):
    segs = []
    t = start
    for a, b in sorted(plan["breaks"]):
        if b <= t:
            continue
        if a >= end:
            break
        if a > t:
            segs.append((t, min(a, end)))
        t = max(t, b)
    if t < end:
        segs.append((t, end))
    return segs


def work_minutes_between(plan, a, b):
    if b <= a:
        return 0.0
    return sum((e - s).total_seconds() for s, e in _segments(plan, a, b)) / 60.0


def add_work_minutes(plan, t, minutes):
    """Clock time after `minutes` of work starting at t, skipping breaks (overtime runs past shift end)."""
    remaining = minutes * 60.0
    for s, e in _segments(plan, t, plan["shift_end"]):
        span = (e - s).total_seconds()
        if remaining <= span:
            return s + timedelta(seconds=remaining)
        remaining -= span
    return max(t, plan["shift_end"]) + timedelta(seconds=remaining)


class PlanningDetector:
    period_s = 60

    def run(self, p, v, now, site):
        if v.kind != "excavator" or v.tel is None:
            return []
        plan, tel = v.plan, v.tel
        if not (plan["shift_start"] - timedelta(minutes=30) <= now < plan["shift_end"]):
            return []
        if v.plan_sched is None:
            if not tel["engine_on"]:
                return []
            self._build_schedule(p, v, now, site)
        out = []
        live = self._live_plan(p, v, now)
        need_min = min(live["remaining_task_min"], work_minutes_between(plan, now, plan["shift_end"]))
        fuel = self._fuel(p, v, now, need_min)
        if fuel:
            out.append(fuel)
        d = self._def(v, now, need_min)
        if d:
            out.append(d)
        out += self._service(v, now, need_min)
        out += self._task_delay(p, v, now, live)
        out += self._weather(v, now, site, live)
        af = self._air_filter(v, now)
        if af:
            out.append(af)
        if not v.briefed:
            v.briefed = True
            out.append(self._briefing(p, v, now, live, out))
        return out

    # ------------------------------------------------------------ schedule
    def _build_schedule(self, p, v, now, site):
        plan = v.plan
        amb = site["weather"]["ambient_temp_c"]
        t = max(now, plan["shift_start"])
        sched = {}
        for tk in plan["tasks"]:
            mins = p.models.task_minutes(tk, plan["operator_id"], amb + 6.0, False)
            fin = add_work_minutes(plan, t, mins)
            sched[tk["id"]] = {"pred_min": mins, "planned_start": t, "planned_finish": fin}
            t = fin
        v.plan_sched = sched

    def _live_plan(self, p, v, now):
        plan, tel = v.plan, v.tel
        cur = tel.get("task_id")
        ids = [t["id"] for t in plan["tasks"]]
        rows = []
        t = now
        remaining_total = 0.0
        seen_current = cur is None
        for tk in plan["tasks"]:
            tid = tk["id"]
            sch = v.plan_sched[tid]
            done = v.task_done_last.get(tid, 0.0)
            vol = tk["volume_m3"]
            row = {"id": tid, "name": tk["name"], "type": tk["type"], "volume_m3": vol, "done_m3": round(done, 1),
                   "pred_min": round(sch["pred_min"]), "planned_start": sch["planned_start"],
                   "planned_finish": sch["planned_finish"], "weather_sensitive": tk.get("weather_sensitive", False)}
            if tid == cur:
                seen_current = True
                on_min = v.task_on_min[tid]
                prior_rate = vol / max(sch["pred_min"], 1.0)
                rate = prior_rate
                if on_min >= 10 and done > 0:
                    overall = done / on_min
                    hist = [(m, d) for m, t_id, d in v.task_done_hist if t_id == tid][-30:]
                    recent = (hist[-1][1] - hist[0][1]) / max(len(hist) - 1, 1) if len(hist) >= 10 else overall
                    w = min(1.0, on_min / 60.0)
                    rate = (1 - w) * prior_rate + w * (0.5 * overall + 0.5 * recent)
                rem = max(vol - done, 0.0) / max(rate, 1e-3)
                fin = add_work_minutes(plan, t, rem)
                row.update(status="in_progress", rate_m3_min=round(rate, 2), planned_rate_m3_min=round(prior_rate, 2),
                           remaining_min=round(rem), predicted_finish=fin)
                remaining_total += rem
                t = fin
            elif not seen_current or done >= vol - 0.5:
                row.update(status="done", predicted_finish=None, remaining_min=0)
            else:
                rem = sch["pred_min"] * max(vol - done, 0) / vol
                fin = add_work_minutes(plan, t, rem)
                row.update(status="planned", remaining_min=round(rem), predicted_finish=fin)
                remaining_total += rem
                t = fin
            row["actual_start"] = v.task_first_seen.get(tid)
            row["actual_end"] = v.task_last_seen.get(tid) if row.get("status") == "done" else None
            rows.append(row)
        live = {"tasks": rows, "plan_end": t, "remaining_task_min": remaining_total, "ids": ids}
        v.derived["plan"] = [{**r, "planned_start": fmt_hm(r["planned_start"]),
                              "planned_finish": fmt_hm(r["planned_finish"]),
                              "predicted_finish": fmt_hm(r.get("predicted_finish")),
                              "actual_start": fmt_hm(r.get("actual_start")),
                              "actual_end": fmt_hm(r.get("actual_end"))} for r in rows]
        v.derived["plan_end"] = fmt_hm(t)
        v.derived["task_eta"] = {r["id"]: r.get("predicted_finish") for r in rows}
        return live

    # ------------------------------------------------------------ fuel & fluids
    def _fuel(self, p, v, now, need_min):
        tel, plan = v.tel, v.plan
        pct = tel["fuel_level_pct"]
        fuel_l = pct / 100 * v.tank_l
        usable = max(0.0, fuel_l - RESERVE_PCT / 100 * v.tank_l)
        burn = max(v.burn_lph, 5.0)
        hours_left = usable / burn
        need_h = need_min / 60
        runout = add_work_minutes(plan, now, hours_left * 60)
        v.derived["fuel"] = {"hours_left": round(hours_left, 2), "burn_lph": round(burn, 1),
                             "runout": fmt_hm(runout), "need_h": round(need_h, 2), "fuel_l": round(fuel_l)}
        margin = 0.25 if p.manager.is_active(f"{v.id}:FUEL_SHORTFALL") else 0.0
        if hours_left >= need_h + margin:
            return None
        breaks = [(a, b) for a, b in plan["breaks"] if now <= a < runout]
        refuel_at = breaks[-1][0] if breaks else None
        litres_needed = (need_h - hours_left) * burn + RESERVE_PCT / 100 * v.tank_l
        sev = "critical" if hours_left < 1.0 else "warning"
        ovn = None
        if plan.get("fuel_at_last_shutdown_pct") is not None and v.fuel_first is not None:
            ovn = round(v.fuel_first - plan["fuel_at_last_shutdown_pct"], 1)
        related = [f"{v.id}:FUEL_LOSS_ENGINE_OFF"] if (ovn is not None and ovn < -3) else []
        rtxt = (f"Refuel at the {fmt_hm(refuel_at)} break." if refuel_at else "Refuel as soon as possible.")
        otxt = f" Tank was {-ovn:.0f} points lower at start-up than at last shutdown." if ovn and ovn < -3 else ""
        return Finding(
            key=f"{v.id}:FUEL_SHORTFALL", machine_id=v.id, type="FUEL_SHORTFALL", category="planning",
            level="L4_predictive", severity=sev, title="Not enough fuel to finish the shift",
            summary=(f"{fuel_l:.0f} L on board at {burn:.1f} L/h lasts {hours_left:.1f} work-hours (runs dry "
                     f"≈{fmt_hm(runout)}), but {need_h:.1f} h of planned work remain. Need ~{litres_needed:.0f} L more. "
                     f"{rtxt}{otxt}"),
            evidence={"fuel_level_pct": {"value": round(pct, 1), "unit": "%"},
                      "fuel_l": {"value": round(fuel_l), "unit": "L"},
                      "burn_rate_lph": {"value": round(burn, 1), "unit": "L/h", "note": "learned from recent operation"},
                      "work_hours_remaining": {"value": round(need_h, 2), "unit": "h"},
                      "litres_needed": {"value": round(litres_needed), "unit": "L"},
                      "overnight_change_pct": {"value": ovn, "unit": "pts"},
                      "suggested_refuel_at": {"value": fmt_hm(refuel_at)}},
            prediction={"kind": "fuel_runout", "eta": runout.isoformat(timespec="seconds"),
                        "minutes_to_impact": round(hours_left * 60), "confidence": 0.8,
                        "detail": {"refuel_at": fmt_hm(refuel_at)}},
            related=related, clear_after_s=600, update_token=int((runout - now).total_seconds() // 1800),
            update_min_s=1800)

    def _def(self, v, now, need_min):
        tel = v.tel
        pct = tel.get("def_level_pct")
        if pct is None:
            return None
        def_l = pct / 100 * EXCAVATOR_SPEC["def_tank_l"]
        burn = max(v.burn_lph, 5.0) * EXCAVATOR_SPEC["def_ratio"]
        hours_left = def_l / burn
        v.derived["def"] = {"hours_left": round(hours_left, 1)}
        if hours_left >= need_min / 60 + 1:
            return None
        eta = add_work_minutes(v.plan, now, hours_left * 60)
        return Finding(
            key=f"{v.id}:DEF_SHORTFALL", machine_id=v.id, type="DEF_SHORTFALL", category="planning",
            level="L4_predictive", severity="warning", title="DEF will run out this shift",
            summary=f"{def_l:.1f} L DEF lasts {hours_left:.1f} h; engine derates when empty (≈{fmt_hm(eta)}).",
            evidence={"def_level_pct": {"value": pct, "unit": "%"}},
            prediction={"kind": "def_runout", "eta": eta.isoformat(timespec="seconds")}, clear_after_s=180)

    def _service(self, v, now, need_min):
        tel = v.tel
        h = tel["engine_hours"]
        nxt = tel.get("next_service_h")
        if nxt is None:
            return []
        to_go = nxt - h
        v.derived["service"] = {"hours_to_service": round(to_go, 2), "next_service_h": nxt}
        if to_go <= 0:
            return [Finding(
                key=f"{v.id}:SERVICE_OVERDUE", machine_id=v.id, type="SERVICE_OVERDUE", category="planning",
                level="L4_predictive", severity="warning", title=f"{nxt:.0f} h service overdue",
                summary=f"Hour meter {h:.1f} h passed the {nxt:.0f} h service point by {abs(to_go):.1f} h.",
                evidence={"engine_hours": {"value": round(h, 1), "unit": "h", "threshold": nxt}},
                clear_after_s=180)]
        if to_go * 60 < need_min:
            eta = add_work_minutes(v.plan, now, to_go * 60)
            return [Finding(
                key=f"{v.id}:SERVICE_DUE", machine_id=v.id, type="SERVICE_DUE", category="planning",
                level="L4_predictive", severity="info", title=f"{nxt:.0f} h service falls due during this shift",
                summary=(f"Hour meter {h:.1f} h: the {nxt:.0f} h service is due in {to_go:.1f} engine-hours "
                         f"(≈{fmt_hm(eta)}), before today's work is finished."),
                evidence={"engine_hours": {"value": round(h, 1), "unit": "h", "threshold": nxt},
                          "hours_to_service": {"value": round(to_go, 2), "unit": "h"}},
                prediction={"kind": "service_due", "eta": eta.isoformat(timespec="seconds"),
                            "minutes_to_impact": round(to_go * 60)},
                clear_after_s=180)]
        return []

    # ------------------------------------------------------------ plan risk
    def _task_delay(self, p, v, now, live):
        out = []
        for r in live["tasks"]:
            if r.get("status") != "in_progress":
                continue
            delay = (r["predicted_finish"] - r["planned_finish"]).total_seconds() / 60
            v.derived["task_delay_min"] = round(delay)
            key = f"{v.id}:TASK_DELAY:{r['id']}"
            limit = max(20.0, 0.10 * r["pred_min"])  # model error grows with task length
            if p.manager.is_active(key):
                limit *= 0.5
            if delay < limit or v.task_on_min[r["id"]] < 30:
                continue
            idle = v.idle_last(60)
            trucks = p.truck_status()
            stuck = [t["id"] for t in trucks if not t["cycling"] and t["where"] != "park" and t["stationary_min"] >= 5]
            carry = [x["id"] for x in live["tasks"] if x.get("status") != "done"
                     and x.get("predicted_finish") and x["predicted_finish"] > v.plan["shift_end"]]
            why = []
            if r["type"] == "truck_loading":
                if idle.get("truck_wait", 0) > 300:
                    why.append(f"{idle['truck_wait'] / 60:.0f} min waiting for trucks in the last hour")
                if stuck:
                    why.append(f"{', '.join(stuck)} out of the haul cycle")
            if r["rate_m3_min"] < r["planned_rate_m3_min"] * 0.85 and not why:
                why.append("production rate below plan")
            out.append(Finding(
                key=key, machine_id=v.id, type="TASK_DELAY", category="productivity",
                level="L4_predictive", severity="warning", title=f"{r['id']} running late",
                summary=(f"{r['id']} ({r['name']}) now predicted to finish ≈{fmt_hm(r['predicted_finish'])}, "
                         f"{delay:.0f} min after plan ({fmt_hm(r['planned_finish'])}). "
                         f"{r['done_m3']:.0f}/{r['volume_m3']} m³ done at {r['rate_m3_min'] * 60:.0f} m³/h vs "
                         f"{r['planned_rate_m3_min'] * 60:.0f} planned"
                         + (f"; cause: {'; '.join(why)}" if why else "") + "."
                         + (f" At risk of carrying over: {', '.join(carry)}." if carry else "")),
                evidence={"delay_min": {"value": round(delay), "unit": "min", "threshold": round(limit)},
                          "rate_m3_h": {"value": round(r["rate_m3_min"] * 60), "unit": "m³/h",
                                        "baseline": round(r["planned_rate_m3_min"] * 60)},
                          "done_m3": {"value": r["done_m3"], "unit": "m³"},
                          "tasks_at_risk": {"value": ", ".join(carry) or None}},
                prediction={"kind": "task_finish", "eta": r["predicted_finish"].isoformat(timespec="seconds"),
                            "detail": {"planned_finish": fmt_hm(r["planned_finish"])}},
                clear_after_s=300, update_token=int(delay // 20), update_min_s=1200))
        return out

    def _weather(self, v, now, site, live):
        w = site["weather"]
        rs = w.get("forecast_rain_start")
        if not rs:
            return []
        h, m = map(int, rs.split(":"))
        rain = now.replace(hour=h, minute=m, second=0)
        if now >= rain:
            return []
        out = []
        for r in live["tasks"]:
            if r.get("status") not in ("in_progress", "planned") or not r["weather_sensitive"]:
                continue
            fin = r.get("predicted_finish")
            if fin is None or fin <= rain - timedelta(minutes=10):
                continue
            start = now if r["status"] == "in_progress" else fin - timedelta(minutes=r["remaining_min"])
            if start >= rain:
                continue
            short = (fin - rain).total_seconds() / 60
            margin = (f"{short:.0f} min after the rain starts" if short >= 0
                      else f"only {-short:.0f} min before the rain, no safety margin")
            out.append(Finding(
                key=f"{v.id}:WEATHER_WINDOW:{r['id']}", machine_id=v.id, type="WEATHER_WINDOW", category="planning",
                level="L4_predictive", severity="warning", title=f"Rain will interrupt {r['id']}",
                summary=(f"Rain forecast from {rs} ({w.get('rain_mm_h') or 'heavy'} mm/h"
                         f"{', lightning risk' if w.get('forecast_lightning_risk') else ''}). {r['id']} ({r['name']}) "
                         f"is predicted to finish ≈{fmt_hm(fin)}, {margin}, with "
                         f"{r['volume_m3'] - r['done_m3']:.0f} m³ left. Open trenches/batters in rain risk collapse."),
                evidence={"forecast_rain_start": {"value": rs}, "task_predicted_finish": {"value": fmt_hm(fin)},
                          "minutes_short": {"value": round(short), "unit": "min"},
                          "remaining_m3": {"value": round(r["volume_m3"] - r["done_m3"]), "unit": "m³"}},
                prediction={"kind": "rain_start", "eta": rain.isoformat(timespec="seconds"),
                            "minutes_to_impact": round((rain - now).total_seconds() / 60)},
                clear_after_s=300, update_token=int(short // 15), update_min_s=900))
            break
        return out

    def _air_filter(self, v, now):
        hist = v.plan.get("maintenance_history", {}).get("air_filter_dp_daily_peak_kpa")
        if not hist or v.af_peak_today is None or v.task_on_min.total() < 60:
            return None
        ys = np.array(list(hist) + [v.af_peak_today])
        slope = float(np.polyfit(np.arange(len(ys)), ys, 1)[0])
        today = v.af_peak_today
        if slope <= 0.01:
            return None
        days = (AF_LIMIT_KPA - today) / slope
        v.derived["air_filter"] = {"peak_kpa": round(today, 2), "slope_kpa_day": round(slope, 3),
                                   "days_to_limit": round(days, 1)}
        if days >= 7:
            return None
        eta = now + timedelta(days=days)
        return Finding(
            key=f"{v.id}:MAINTENANCE_FORECAST:air_filter", machine_id=v.id, type="MAINTENANCE_FORECAST",
            category="planning", level="L4_predictive", severity="warning" if days < 3 else "info",
            title="Air filter will need replacing within a week",
            summary=(f"Air filter restriction peaked at {today:.2f} kPa today and is climbing {slope:.2f} kPa/day "
                     f"over the last {len(hist)} days. It reaches the {AF_LIMIT_KPA} kPa change limit in "
                     f"~{days:.1f} operating days. Order the element and schedule the change."),
            evidence={"air_filter_dp_kpa": {"value": round(today, 2), "unit": "kPa", "threshold": AF_LIMIT_KPA,
                                            "trend_per_min": None, "spn": 107},
                      "trend_kpa_per_day": {"value": round(slope, 3), "unit": "kPa/day"}},
            prediction={"kind": "days_to_limit", "eta": eta.isoformat(timespec="seconds"), "value": round(days, 1),
                        "unit": "days", "confidence": 0.75},
            clear_after_s=7200)

    # ------------------------------------------------------------ briefing
    def _briefing(self, p, v, now, live, findings):
        plan = v.plan
        op = plan["operator_id"]
        name = OPERATORS[op].name if op in OPERATORS else op
        flagged = [f for f in findings if f.severity in ("warning", "critical")]
        lines = []
        for r in live["tasks"]:
            lines.append(f"{r['id']} {r['name']}: {r['volume_m3']} m³, predicted {r['pred_min'] / 60:.1f} h, "
                         f"finish ≈{fmt_hm(r.get('predicted_finish') or r['planned_finish'])}")
        end = live["plan_end"]
        over = (end - plan["shift_end"]).total_seconds() / 60
        plan_txt = (f"Plan ends ≈{fmt_hm(end)}" + (f", {over:.0f} min past shift end" if over > 5 else ", within shift"))
        issues = [f"{f.title}" for f in findings if f.type not in ("SHIFT_BRIEFING",)]
        return Finding(
            key=f"{v.id}:SHIFT_BRIEFING:{now:%Y%m%d}", machine_id=v.id, type="SHIFT_BRIEFING", category="planning",
            level="L4_predictive", severity="warning" if flagged else "info",
            title=f"Pre-shift briefing for {name} on {v.id}",
            summary=(f"{len(plan['tasks'])} task{'s' if len(plan['tasks']) != 1 else ''} today. {plan_txt}. "
                     + (f"Flagged: {'; '.join(issues)}." if issues else "No issues flagged.")),
            evidence={"tasks": {"value": " | ".join(lines)},
                      "fuel_hours_left": {"value": v.derived.get("fuel", {}).get("hours_left"), "unit": "h"},
                      "work_hours_planned": {"value": round(live["remaining_task_min"] / 60, 2), "unit": "h"},
                      "hours_to_service": {"value": v.derived.get("service", {}).get("hours_to_service"), "unit": "h"},
                      "flags": {"value": ", ".join(f.type for f in findings) or None}},
            related=[f.key for f in findings], one_shot=True, cooldown_s=86400)
