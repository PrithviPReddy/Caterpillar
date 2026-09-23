"""Shared physical/behavioural relationships used by the tick simulator and the
fast task-level simulator (so both produce consistent data)."""

import math

import numpy as np

from sim.config import EXCAVATOR_SPEC, SOIL_DENSITY_T_PER_M3, TRUCK_SPEC

# dig, swing_loaded, dump, swing_empty (s) and swing angle (deg) at skill 0.5, soil 3
TASK_CYCLE_BASE = {
    "truck_loading": (7.0, 5.5, 3.0, 4.5, 90.0),
    "trenching": (10.0, 4.5, 2.5, 4.0, 90.0),
    "slope_cut": (11.0, 7.0, 3.5, 6.0, 150.0),
    "side_cast": (7.5, 5.0, 2.5, 4.5, 100.0),
}
TASK_BUCKET_M3 = {"truck_loading": 1.9, "trenching": 1.0, "slope_cut": 1.5, "side_cast": 1.9}
# every k cycles the machine repositions (tracks along the trench / face) for ~s seconds
TASK_REPOSITION = {"truck_loading": (0, 0.0), "trenching": (5, 30.0), "slope_cut": (8, 25.0), "side_cast": (10, 25.0)}

# Planner rule-of-thumb productivity (bank m3/h), used as the naive baseline.
NOMINAL_RATE_M3H = {"truck_loading": 190.0, "trenching": 110.0, "slope_cut": 120.0, "side_cast": 170.0}


def cycle_plan(op, task_type, soil_hardness, fatigue, rng, rain=False, dump_angle=None):
    dig, sl, dump, se, swing = TASK_CYCLE_BASE[task_type]
    skill_f = 1.25 - 0.35 * op.skill
    soil_f = 1.0 + 0.12 * (soil_hardness - 3)
    fat_f = 1.0 + 0.25 * fatigue
    rain_f = 1.12 if rain else 1.0
    aggr_f = 1.0 - 0.18 * op.aggressiveness
    sigma = 0.08 + 0.12 * fatigue

    def n():
        return math.exp(rng.normal(0.0, sigma))

    fill = 0.78 + 0.22 * op.skill - 0.04 * (soil_hardness - 3) + rng.normal(0.0, 0.05)
    return {
        "DIG": dig * skill_f * soil_f * fat_f * rain_f * n(),
        "SWING_LOADED": sl * skill_f * aggr_f * fat_f * n(),
        "DUMP": dump * skill_f * n(),
        "SWING_EMPTY": se * skill_f * aggr_f * fat_f * n(),
        "swing_angle": dump_angle if dump_angle is not None else swing,
        "fill": float(np.clip(fill, 0.6, 1.1)),
        "end_stop": rng.random() < (0.02 + 0.30 * op.aggressiveness),
        "impact": rng.random() < (0.01 + 0.10 * op.aggressiveness),
    }


def bucket_payload_t(fill, bucket_m3=None):
    return (bucket_m3 or EXCAVATOR_SPEC["bucket_m3"]) * fill * SOIL_DENSITY_T_PER_M3


def fatigue_rate_per_h(op, heat_index_c, hour_of_day):
    heat = max(0.0, heat_index_c - 27.0) / 8.0
    circadian = 1.0 if 13.5 <= hour_of_day <= 16.0 else (0.6 if hour_of_day < 7.0 else 0.0)
    return (0.07 + 0.15 * heat + 0.05 * circadian) * op.fatigue_sensitivity


def heat_index_c(temp_c, rh):
    """NOAA Rothfusz heat index, computed in °F and returned in °C."""
    t = temp_c * 9 / 5 + 32
    if t < 80:
        hi = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    else:
        hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
              - 6.83783e-3 * t * t - 5.481717e-2 * rh * rh + 1.22874e-3 * t * t * rh
              + 8.5282e-4 * t * rh * rh - 1.99e-6 * t * t * rh * rh)
    return (hi - 32) * 5 / 9


def simulate_task_duration_min(task, op, env, rng):
    """Fast cycle-level simulation of a whole task. Returns working minutes.

    task: {type, volume_m3, soil_hardness, n_trucks, haul_m}
    env:  {ambient_c, rh, rain}
    Hidden from any model trained on the output: fatigue build-up, cycle noise,
    truck bunching, random stoppages.
    """
    ttype = task["type"]
    bucket = task.get("bucket_m3", TASK_BUCKET_M3[ttype])
    rep_k, rep_s = TASK_REPOSITION[ttype]
    cycles = 0
    t = 0.0
    done = 0.0
    fatigue = rng.uniform(0.0, 0.25)
    hi = heat_index_c(env["ambient_c"], env["rh"])
    frate = fatigue_rate_per_h(op, hi, 11.0)
    n_trucks = task.get("n_trucks", 0) if ttype == "truck_loading" else 0
    if n_trucks:
        haul = task.get("haul_m", 750.0)
        v_l = TRUCK_SPEC["loaded_speed_kmh"] / 3.6
        v_e = TRUCK_SPEC["empty_speed_kmh"] / 3.6
        ready = [i * 45.0 for i in range(n_trucks)]
    stop_rate_per_s = 0.35 / 3600.0
    while done < task["volume_m3"]:
        t_iter = t
        if n_trucks:
            i = int(np.argmin(ready))
            t = max(t, ready[i]) + 20.0  # spot the truck
            payload = 0.0
            while payload < TRUCK_SPEC["target_payload_t"] and done < task["volume_m3"]:
                c = cycle_plan(op, ttype, task["soil_hardness"], fatigue, rng, env["rain"])
                ct = c["DIG"] + c["SWING_LOADED"] + c["DUMP"] + c["SWING_EMPTY"]
                t += ct
                payload += bucket_payload_t(c["fill"], bucket)
                done += bucket * c["fill"]
                fatigue = min(1.0, fatigue + frate * ct / 3600)
            trip = haul / v_l + haul / v_e + 40.0
            ready[i] = t + trip * math.exp(rng.normal(0, 0.12))
        else:
            c = cycle_plan(op, ttype, task["soil_hardness"], fatigue, rng, env["rain"])
            ct = c["DIG"] + c["SWING_LOADED"] + c["DUMP"] + c["SWING_EMPTY"]
            t += ct
            done += bucket * c["fill"]
            fatigue = min(1.0, fatigue + frate * ct / 3600)
            cycles += 1
            if rep_k and cycles % rep_k == 0:
                t += rep_s * rng.uniform(0.8, 1.3)
        if rng.random() < 1.0 - math.exp(-stop_rate_per_s * (t - t_iter)):
            t += rng.uniform(4, 20) * 60  # unplanned stoppage
    return t / 60.0
