"""Detection pipeline: telemetry in, schema-v1 events out."""

import math
from collections import Counter

from detect.behavior import BehaviorDetector
from detect.health import HealthMonitor
from detect.incidents import IncidentRecorder
from detect.manager import EventManager, fmt_hm
from detect.models import ModelStore
from detect.planning import PlanningDetector
from detect.playbook import actions_for, training_for
from detect.rules import SecurityDetector, SensorHealthDetector, SiteWeatherDetector, ThresholdDetector, TrendDetector
from detect.safety import SafetyDetector
from detect.schema import Event, EventContext, Location, Prediction, ShiftContext, SignalEvidence, TaskContext
from detect.view import MachineView
from sim.config import EXCAVATOR_SPEC, FUEL_PRICE_PER_L, SAFETY
from sim.operators import OPERATORS
from sim.site import HAUL_ROUTE, ZONES

LOADING_POINT = HAUL_ROUTE[0]


class DetectionPipeline:
    def __init__(self, plans, models=None, write_incidents=True):
        self.models = models or ModelStore()
        self.views = {mid: MachineView(mid, plan, self.models.baselines) for mid, plan in plans.items()}
        self.manager = EventManager(self._emit)
        self.recorder = IncidentRecorder(write_files=write_incidents)
        self.events = []
        self.listeners = []
        self.tick = 0
        self.now = None
        self.site = {"weather": {}, "workers": []}
        self._seq = 0
        self.behavior = BehaviorDetector()
        self.tick_detectors = [ThresholdDetector(), TrendDetector(), SensorHealthDetector(), SecurityDetector(),
                               SafetyDetector(), self.behavior, PlanningDetector()]
        self.site_detectors = [SiteWeatherDetector()]
        self.minute_hooks = [HealthMonitor(), self.behavior]

    # ------------------------------------------------------------------ main
    def ingest(self, frames, site, now):
        self.now = now
        self.site = site
        self.tick += 1
        by_id = {f["machine_id"]: f for f in frames}
        ready = self._bay_ready(by_id)
        findings = []
        for tel in frames:
            v = self.views[tel["machine_id"]]
            rec = v.update(tel, now, ready)
            if tel.get("task_id") is not None and tel.get("task_done_m3") is not None:
                v.task_done_last[tel["task_id"]] = tel["task_done_m3"]
            if rec is not None:
                v.on_minute(rec)
                for hook in self.minute_hooks:
                    findings += hook.on_minute(self, v, rec, now)
        for det in self.tick_detectors:
            due = self.tick % det.period_s == 0
            for v in self.views.values():
                if due or v.just_started or v.just_stopped:
                    findings += det.run(self, v, now, site)
        for det in self.site_detectors:
            if self.tick % det.period_s == 0:
                findings += det.run_site(self, now, site)
        events = self.manager.process(now, findings)
        self.recorder.tick(now)
        self._derive(site)
        return events

    def _bay_ready(self, by_id):
        n = 0
        for t in by_id.values():
            if t["machine_type"] != "articulated_truck" or not t["engine_on"]:
                continue
            if t.get("trans_pressure_kpa", 0) < 800:
                continue
            if (math.hypot(t["x"] - LOADING_POINT[0], t["y"] - LOADING_POINT[1]) < 12
                    and t.get("travel_speed_kmh", 0) < 1.0):
                n += 1
        return n

    # ------------------------------------------------------------------ events
    def _emit(self, f, status, now):
        self._seq += 1
        v = self.views.get(f.machine_id)
        tel = v.tel if v else None
        op_id = (tel.get("operator_id") if tel else None) or (v.plan.get("operator_id") if v else None)
        ev = Event(
            event_id=f"EVT-{self._seq:06d}", seq=self._seq, incident_key=f.key, status=status,
            ts=now.isoformat(timespec="seconds"), machine_id=f.machine_id,
            machine_type=v.plan["machine_type"] if v else "site",
            operator_id=op_id if v else None,
            operator_name=OPERATORS[op_id].name if (v and op_id in OPERATORS) else None,
            category=f.category, type=f.type, intelligence_level=f.level, severity=f.severity,
            title=f.title, summary=f.summary,
            evidence={k: SignalEvidence(**val) for k, val in f.evidence.items()},
            prediction=Prediction(**f.prediction) if f.prediction else None,
            context=self.build_context(v, now),
            recommended_actions=actions_for(f.type), training_module=training_for(f.type),
            dtc=f.dtc if (f.dtc and f.dtc.get("spn")) else None, related_incident_keys=f.related, one_shot=f.one_shot,
        )
        d = ev.model_dump()
        if f.capture_incident and status in ("open", "escalated") and v is not None:
            d["incident_id"] = self.recorder.start(v, d, now, self.site)
        if v is not None and status in ("open", "escalated"):
            v.event_counts[f.type] += 1
        self.events.append(d)
        for fn in self.listeners:
            fn(d)
        return d

    def build_context(self, v, now):
        weather = dict(self.site.get("weather", {}))
        if v is None or v.tel is None:
            return EventContext(weather=weather)
        tel = v.tel
        zone = tel.get("zone")
        loc = Location(x=tel["x"], y=tel["y"], zone=zone, zone_name=ZONES[zone]["name"] if zone else None)
        task = None
        tid = tel.get("task_id")
        if tid and tid in v.tasks_by_id:
            tk = v.tasks_by_id[tid]
            eta = v.derived.get("task_eta", {}).get(tid)
            sched = (v.plan_sched or {}).get(tid, {})
            done = tel.get("task_done_m3") or 0.0
            task = TaskContext(id=tid, name=tk["name"], type=tk["type"], done_m3=done, volume_m3=tk["volume_m3"],
                               progress_pct=round(100 * done / tk["volume_m3"], 1),
                               predicted_finish=eta.isoformat(timespec="seconds") if eta else None,
                               planned_finish=sched["planned_finish"].isoformat(timespec="seconds")
                               if sched.get("planned_finish") else None)
        plan = v.plan
        nb = next((a for a, b in plan.get("breaks", []) if a > now), None)
        shift = ShiftContext(start=plan["shift_start"].isoformat(timespec="seconds"),
                             end=plan["shift_end"].isoformat(timespec="seconds"),
                             elapsed_h=round(max(0.0, (now - plan["shift_start"]).total_seconds() / 3600), 2),
                             next_break=fmt_hm(nb))
        return EventContext(machine_state=tel.get("state"), engine_on=tel.get("engine_on"), location=loc,
                            task=task, shift=shift, weather=weather)

    # ------------------------------------------------------------------ helpers
    def truck_status(self):
        out = []
        for v in self.views.values():
            if v.kind != "truck" or v.tel is None:
                continue
            t = v.tel
            d_load = math.hypot(t["x"] - LOADING_POINT[0], t["y"] - LOADING_POINT[1])
            zone = t.get("zone")
            where = ("loader" if d_load < 30 else "dump" if zone == "DUMP" else "park" if zone == "PARK"
                     else "haul road")
            stat_min = v.stationary_s / 60
            cycling = t["engine_on"] and where != "park" and (stat_min < 5 or where == "loader") \
                and t.get("trans_pressure_kpa", 0) >= 800
            out.append({"id": v.id, "state": t["state"], "where": where, "stationary_min": round(stat_min, 1),
                        "cycling": bool(cycling), "speed_kmh": t.get("travel_speed_kmh")})
        return out

    def task_progress(self, v):
        return {tid: {"done": d, "volume": v.tasks_by_id[tid]["volume_m3"]}
                for tid, d in v.task_done_last.items() if tid in v.tasks_by_id}

    def _derive(self, site):
        """Cheap per-tick UI metrics (nearest worker, danger radius)."""
        if self.tick % 2:
            return
        for v in self.views.values():
            tel = v.tel
            if tel is None:
                continue
            best = None
            for w in site.get("workers", []):
                d = math.hypot(w["x"] - tel["x"], w["y"] - tel["y"])
                if best is None or d < best[1]:
                    best = (w["id"], d)
            v.derived["nearest_worker"] = {"id": best[0], "distance_m": round(best[1], 1)} if best else None
            if v.kind == "excavator":
                v.derived["danger_radius_m"] = round(max(tel.get("reach_m", 5), EXCAVATOR_SPEC["tail_swing_m"])
                                                     + SAFETY["proximity_buffer_m"], 1)

    def snapshot(self, mid):
        v = self.views[mid]
        idle = Counter()
        for k, s in v.stats.items():
            if k.startswith("idle_"):
                idle[k[5:-2]] = round(s / 60, 1)
        active = [{"key": k, "type": a["finding"].type, "severity": a["finding"].severity,
                   "title": a["finding"].title, "summary": a["finding"].summary,
                   "category": a["finding"].category, "level": a["finding"].level,
                   "actions": actions_for(a["finding"].type), "training": training_for(a["finding"].type),
                   "prediction": a["finding"].prediction,
                   "since": a["opened"].isoformat(timespec="seconds"),
                   "event_id": a["event"]["event_id"]}
                  for k, a in self.manager.active.items() if a["finding"].machine_id == mid]
        return {
            "telemetry": v.tel,
            "health_score": v.health_score,
            "residuals": {k: round(x, 2) for k, x in v.res_ewm.items()},
            "expected": v.last_expected,
            "derived": v.derived,
            "idle_min_today": dict(idle),
            "idle_fuel_l_today": {k: round(x, 2) for k, x in v.idle_fuel.items()},
            "idle_cost_today": round(sum(v.idle_fuel.values()) * FUEL_PRICE_PER_L, 2),
            "stats": {"engine_on_h": round(v.stats["engine_on_s"] / 3600, 2),
                      "work_h": round(v.stats["work_s"] / 3600, 2),
                      "fuel_used_l": round(v.stats["fuel_used_l"], 1),
                      "end_stops": v.stats["end_stops"], "impacts": v.stats["impacts"],
                      "micro_pauses": v.stats["micro_pauses"],
                      "end_stops_last_h": len(v.end_stop_times),
                      "seatbelt_compliance_pct": round(100 * (1 - v.stats["belt_off_moving_s"]
                                                              / max(v.stats["seated_moving_s"], 1)), 1)},
            "event_counts": dict(v.event_counts),
            "active_alerts": sorted(active, key=lambda a: ({"critical": 0, "warning": 1, "info": 2}[a["severity"]],
                                                          a["category"] != "safety", a["since"]), reverse=False),
        }

    def minute_series(self, mid, n=240):
        return list(self.views[mid].minutes)[-n:]
