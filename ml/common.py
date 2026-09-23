"""Helpers to run simulated days for training data and evaluation."""

import statistics

from detect.features import HEALTH_TARGETS
from detect.pipeline import DetectionPipeline
from detect.view import MachineView
from sim.world import World


def history_day(scenario):
    """Run a day with telemetry-only views (no alerts). Returns health rows and behaviour stats."""
    world = World(scenario)
    views = {mid: MachineView(mid, plan, {}) for mid, plan in world.plans.items()}
    rows = []
    while not world.done:
        frames = world.step()
        now = world.t
        ready = sum(1 for f in frames if f["machine_type"] == "articulated_truck" and f["engine_on"]
                    and f.get("trans_pressure_kpa", 0) >= 800 and abs(f["x"] - 8) + abs(f["y"]) < 40)
        for tel in frames:
            v = views[tel["machine_id"]]
            if v.kind != "excavator":
                continue
            rec = v.update(tel, now, ready)
            if rec is None:
                continue
            v.on_minute(rec)
            feats = v.hfs.update(rec)
            if feats is None:
                continue
            row = {"day": scenario["name"], "machine_id": v.id, "operator_id": rec["operator_id"],
                   "minute": rec["minute"], **feats, "coolant_now": rec["coolant_temp_c"]}
            for t in HEALTH_TARGETS:
                row[t] = rec[t]
            rows.append(row)
    stats = []
    for v in views.values():
        if v.kind != "excavator" or v.stats["work_s"] < 3600:
            continue
        work_h = v.stats["work_s"] / 3600
        jerks = [j for _, j in v.jerk_minutes if j is not None]
        idle_s = sum(s for k, s in v.stats.items() if k.startswith("idle_"))
        stats.append({
            "day": scenario["name"], "machine_id": v.id, "operator_id": v.plan["operator_id"],
            "end_stops_per_h": v.stats["end_stops"] / work_h,
            "impacts_per_h": v.stats["impacts"] / work_h,
            "micro_pauses_per_h": v.stats["micro_pauses"] / work_h,
            "jerk": statistics.mean(jerks) if jerks else None,
            "jerk_fresh": (v.stats["jerk_fresh_sum"] / v.stats["jerk_fresh_n"]) if v.stats["jerk_fresh_n"] else None,
            "idle_frac": idle_s / max(v.stats["engine_on_s"], 1),
            "burn_lph": v.stats["fuel_used_l"] / max(v.stats["engine_on_s"] / 3600, 0.1),
        })
    return rows, stats


def detection_day(scenario, models):
    """Run a day through the full detection pipeline. Returns events and ground truth."""
    world = World(scenario)
    pipe = DetectionPipeline(world.plans, models=models, write_incidents=False)
    engine_s = 0
    while not world.done:
        frames = world.step()
        engine_s += sum(1 for f in frames if f["machine_type"] == "hydraulic_excavator" and f["engine_on"])
        pipe.ingest(frames, world.site_snapshot(), world.t)
    return pipe.events, world.truth, engine_s / 3600
