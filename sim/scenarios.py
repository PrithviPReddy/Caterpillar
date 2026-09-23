"""Scenario definitions.

`demo_day` is the scripted showcase day. The script only injects *causes* (a fault
starts, a person walks somewhere, an operator habit, a truck breaks). Every alert in
the demo is produced by the detection layer from telemetry alone.

`random_day` produces varied normal days (optionally with one hidden fault) for
training data and evaluation.
"""

import copy

import numpy as np

STD_BREAKS = [("10:00", "10:15"), ("12:00", "12:30"), ("14:30", "14:40")]


def _air_filter_history(today_peak, per_day, days=14, seed=3):
    rng = np.random.default_rng(seed)
    return [round(today_peak - per_day * d + rng.normal(0, 0.04), 3) for d in range(days, 0, -1)]


DEMO_DAY = {
    "name": "demo_day",
    "title": "Demo day - Riverside Phase 2 (scripted incidents)",
    "date": "2025-06-12",
    "start": "05:00",
    "end": "16:40",
    "seed": 11,
    "weather": {
        "t_min": 19.0, "t_max": 37.0, "rh_min": 38.0, "rh_max": 75.0, "wind_kmh": 12.0,
        "storm": {"lightning_start": "14:55", "start_km": 30.0, "closest_at": "15:40", "closest_km": 5.0,
                  "rain_start": "15:50", "rain_mm_h": 9.0, "forecast_issued": "13:30"},
    },
    "machines": {
        "EXC001": {
            "type": "excavator", "operator": "OP1002",
            "engine_hours": 1996.2, "next_service_h": 2000.0,
            "fuel_pct": 58.0, "fuel_at_last_shutdown_pct": 58.0, "def_pct": 62.0,
            "cab_ac": "weak", "fatigue0": 0.10, "air_filter_clog": 1.1,
            "start_pos": (0.0, 0.0), "start_heading": 0.0,
            "shift": {"start": "06:00", "end": "16:00", "breaks": STD_BREAKS},
            "break_policy": {0: "off", 1: "running", 2: "off"},
            "shift_end_cooldown": False,
            "tasks": [
                {"id": "T-101", "name": "Bulk excavation, load out to trucks", "zone": "PAD_A",
                 "pos": (0.0, 0.0), "heading_deg": 0.0, "type": "truck_loading", "volume_m3": 1300,
                 "soil_hardness": 3, "dig_reach": 6.5, "dump_reach": 6.0, "dump_height": 4.2, "n_trucks": 3},
                {"id": "T-102", "name": "Cut batter slope, side-cast downhill", "zone": "ZONE_C",
                 "pos": (110.0, -170.0), "heading_deg": 0.0, "dump_angle": 150.0, "type": "slope_cut",
                 "volume_m3": 200, "soil_hardness": 3, "dig_reach": 6.5, "dump_reach": 6.5, "dump_height": 2.5,
                 "weather_sensitive": True},
                {"id": "T-103", "name": "Storm drain trench under 11 kV line", "zone": "ZONE_D",
                 "pos": (240.0, -45.0), "heading_deg": 90.0, "dump_angle": 90.0, "type": "trenching",
                 "volume_m3": 320, "soil_hardness": 4, "dig_reach": 6.0, "dump_reach": 5.5, "dump_height": 3.0,
                 "weather_sensitive": True},
            ],
        },
        "EXC002": {
            "type": "excavator", "operator": "OP1001",
            "engine_hours": 3412.7, "next_service_h": 3500.0,
            "fuel_pct": 81.0, "fuel_at_last_shutdown_pct": 81.3, "def_pct": 55.0,
            "cab_ac": "good", "fatigue0": 0.05,
            "air_filter_clog": 3.45, "air_filter_growth_per_day": 0.2,
            "maintenance_history": {"air_filter_dp_daily_peak_kpa": _air_filter_history(4.62, 0.24)},
            "start_pos": (-160.0, 100.0), "start_heading": 90.0,
            "shift": {"start": "05:50", "end": "16:00", "breaks": STD_BREAKS},
            "break_policy": {0: "off", 1: "off", 2: "off"},
            "tasks": [
                {"id": "T-201", "name": "Utility trench, water main", "zone": "ZONE_B",
                 "pos": (-160.0, 100.0), "heading_deg": 90.0, "dump_angle": 90.0, "type": "trenching",
                 "volume_m3": 1100, "soil_hardness": 3, "dig_reach": 6.0, "dump_reach": 5.5,
                 "dump_height": 3.0, "weather_sensitive": True},
            ],
        },
        "TRK01": {"type": "truck", "operator": "OP2001", "serves": "EXC001", "fuel_pct": 72.0,
                  "engine_hours": 5120.4, "start_pos": (-62.0, -88.0),
                  "shift": {"start": "06:00", "end": "16:00"}},
        "TRK02": {"type": "truck", "operator": "OP2002", "serves": "EXC001", "fuel_pct": 66.0,
                  "engine_hours": 4388.9, "start_pos": (-70.0, -88.0), "speed_factor": 0.97,
                  "shift": {"start": "06:00", "end": "16:00"}},
        "TRK03": {"type": "truck", "operator": "OP2003", "serves": "EXC001", "fuel_pct": 79.0,
                  "engine_hours": 2950.1, "start_pos": (-78.0, -88.0), "speed_factor": 1.03,
                  "shift": {"start": "06:00", "end": "16:00"}},
    },
    "workers": [
        {"id": "W01", "name": "Priya N.", "role": "Spotter",
         "on_site_from": "06:00", "on_site_until": "16:00",
         "waypoints": [("06:00", -18.0, 14.0), ("11:40", -18.0, 14.0), ("11:50", 95.0, -135.0),
                       ("14:05", 95.0, -135.0), ("14:15", 215.0, -10.0)]},
        {"id": "W02", "name": "Chris D.", "role": "Surveyor",
         "on_site_from": "07:10", "on_site_until": "08:05",
         "waypoints": [("07:10", -60.0, 40.0), ("07:27", -14.0, 10.0), ("07:30:30", 3.0, 5.0),
                       ("07:32", -8.0, -10.0), ("07:36", -50.0, -45.0), ("08:00", -70.0, -80.0)]},
        {"id": "W03", "name": "Omar S.", "role": "Pipe layer",
         "on_site_from": "06:00", "on_site_until": "16:00",
         "waypoints": [("06:00", -150.0, 118.0), ("16:00", -150.0, 118.0)]},
    ],
    "script": [
        {"at": "05:08", "action": "fuel_theft", "machine": "EXC001", "until": "05:26", "to_pct": 24.0,
         "note": "Diesel siphoned overnight while machine parked"},
        {"at": "06:40", "action": "seatbelt_off", "machine": "EXC001", "until": "06:47",
         "note": "Operator unbuckles to reach phone, keeps digging"},
        {"at": "07:45", "action": "inject_fault", "machine": "EXC001", "fault": "radiator_clog", "ramp_min": 240,
         "note": "Radiator core packing with dust - cooling efficiency degrading"},
        {"at": "07:45", "action": "overspeed", "machine": "TRK01", "until": "08:05", "kmh": 27.0,
         "note": "Driver ignores 15 km/h pedestrian zone"},
        {"at": "08:30", "action": "sensor_stuck", "machine": "TRK03", "channel": "fuel_temp_c", "value": 0.0,
         "note": "Fuel temperature sender fails (reads 0)"},
        {"at": "09:20", "action": "truck_breakdown", "machine": "TRK02", "until": "10:40",
         "note": "Transmission pump failure"},
        {"at": "10:02", "action": "refuel", "machine": "EXC001", "to_pct": 92.0, "note": "Fuel bowser at break"},
        {"at": "12:30", "action": "maintenance", "machine": "EXC001", "until": "12:45",
         "note": "Mechanic blows out radiator core"},
        {"at": "12:32", "action": "clear_fault", "machine": "EXC001", "fault": "radiator_clog"},
        {"action": "override", "machine": "EXC001", "task": "T-102", "progress": [0.35, 0.50],
         "params": {"dig_reach": 7.6, "dump_reach": 7.4}, "note": "Reaching further down the batter"},
        {"action": "override", "machine": "EXC001", "task": "T-102", "progress": [0.50, 0.62],
         "params": {"dig_reach": 8.9, "dump_reach": 8.7}, "note": "Full reach side-casting downhill"},
        {"action": "override", "machine": "EXC001", "task": "T-103", "progress": [0.18, 0.26],
         "params": {"dump_height": 8.2}, "note": "Stacking spoil high under the power line"},
        {"at": "14:20", "action": "inject_fault", "machine": "EXC002", "fault": "water_ingress", "ramp_min": 1,
         "params": {"rate_pct_min": 0.7}, "note": "Contaminated fuel batch - water separator filling"},
    ],
}


def random_day(seed, fault=None, fault_machine="EXC001"):
    """A varied normal day. `fault` = (name, start 'HH:MM', ramp_min, params) or None."""
    rng = np.random.default_rng(seed)
    ops = rng.permutation(["OP1001", "OP1002", "OP1003"])[:2]
    t_max = float(rng.uniform(22, 38))
    day = f"2025-{int(rng.integers(4, 9)):02d}-{int(rng.integers(1, 28)):02d}"
    n_trucks = int(rng.integers(2, 4))
    soil = int(rng.integers(2, 5))
    sc = {
        "name": f"random_{seed}",
        "date": day,
        "start": "05:30",
        "end": "16:30",
        "seed": int(seed),
        "weather": {"t_min": t_max - float(rng.uniform(10, 16)), "t_max": t_max,
                    "rh_min": float(rng.uniform(20, 50)), "rh_max": float(rng.uniform(60, 90))},
        "machines": {
            "EXC001": {
                "type": "excavator", "operator": str(ops[0]),
                "engine_hours": float(rng.uniform(800, 6000)),
                "fuel_pct": float(rng.uniform(55, 95)), "def_pct": float(rng.uniform(30, 90)),
                "cab_ac": str(rng.choice(["good", "weak"])),
                "air_filter_clog": float(rng.uniform(0.9, 2.5)),
                "start_pos": (0.0, 0.0), "start_heading": 0.0,
                "shift": {"start": "06:00", "end": "16:00", "breaks": STD_BREAKS},
                "tasks": [
                    {"id": "R-1", "name": "Load out", "zone": "PAD_A", "pos": (0.0, 0.0), "heading_deg": 0.0,
                     "type": "truck_loading", "volume_m3": 3000, "soil_hardness": soil,
                     "dig_reach": float(rng.uniform(5.8, 7.2)), "dump_reach": 6.0, "dump_height": 4.2,
                     "n_trucks": n_trucks},
                ],
            },
            "EXC002": {
                "type": "excavator", "operator": str(ops[1]),
                "engine_hours": float(rng.uniform(800, 6000)),
                "fuel_pct": float(rng.uniform(55, 95)), "def_pct": float(rng.uniform(30, 90)),
                "cab_ac": str(rng.choice(["good", "weak"])),
                "air_filter_clog": float(rng.uniform(0.9, 2.5)),
                "start_pos": (-160.0, 100.0), "start_heading": 90.0,
                "shift": {"start": "06:00", "end": "16:00", "breaks": STD_BREAKS},
                "tasks": [
                    {"id": "R-2", "name": "Trench", "zone": "ZONE_B", "pos": (-160.0, 100.0), "heading_deg": 90.0,
                     "dump_angle": 90.0, "type": str(rng.choice(["trenching", "side_cast"])),
                     "volume_m3": 3000, "soil_hardness": int(rng.integers(2, 5)),
                     "dig_reach": 6.0, "dump_reach": 5.5, "dump_height": 3.0},
                ],
            },
        },
        "workers": [],
        "script": [],
    }
    for i in range(n_trucks):
        sc["machines"][f"TRK0{i + 1}"] = {
            "type": "truck", "operator": f"OP200{i + 1}", "serves": "EXC001",
            "fuel_pct": float(rng.uniform(50, 90)), "engine_hours": float(rng.uniform(1000, 6000)),
            "start_pos": (-62.0 - 8 * i, -88.0), "speed_factor": float(rng.uniform(0.93, 1.07)),
            "shift": {"start": "06:00", "end": "16:00"},
        }
    if fault:
        name, start, ramp, params = fault
        sc["script"].append({"at": start, "action": "inject_fault", "machine": fault_machine,
                             "fault": name, "ramp_min": ramp, "params": params or {}})
    return sc


SCENARIOS = {"demo_day": DEMO_DAY}


def get_scenario(name):
    if name in SCENARIOS:
        return copy.deepcopy(SCENARIOS[name])
    if name.startswith("random_"):
        return random_day(int(name.split("_")[1]))
    raise KeyError(name)
