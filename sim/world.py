"""World: owns the clock, weather, machines, workers and the scenario script.

`truth` records what the script actually did (hidden ground truth for evaluation).
Detectors never read it.
"""

from datetime import datetime, timedelta

import numpy as np

from sim.config import EXCAVATOR_SPEC
from sim.excavator import Excavator
from sim.operators import OPERATORS
from sim.site import GEOFENCES, HAUL_ROUTE, ZONES, Route, Weather, Worker, hhmm
from sim.truck import LoaderBay, Truck

WINDOW_ACTIONS = ("seatbelt_off", "override", "overspeed", "maintenance")


class World:
    def __init__(self, scenario):
        self.scenario = scenario
        self.rng = np.random.default_rng(scenario.get("seed", 0))
        self.day = datetime.fromisoformat(scenario["date"])
        self.t = self.at(scenario["start"])
        self.t_end = self.at(scenario["end"])
        self.weather = Weather(self.day, **scenario.get("weather", {}))
        self.route = Route(HAUL_ROUTE)
        self.machines = {}
        self.bays = {}
        self.truth = []
        self.plans = {}
        for mid, mc in scenario["machines"].items():
            if mc["type"] == "excavator":
                m = Excavator(mid, mc, OPERATORS[mc["operator"]], self)
                self.machines[mid] = m
                if any(tk["type"] == "truck_loading" for tk in mc["tasks"]):
                    self.bays[mid] = LoaderBay(m, self.route)
        for mid, mc in scenario["machines"].items():
            if mc["type"] == "truck":
                bay = self.bays[mc["serves"]]
                self.machines[mid] = Truck(mid, mc, OPERATORS[mc["operator"]], self, bay)
        self.workers = []
        for w in scenario.get("workers", []):
            self.workers.append(Worker(
                w["id"], w["name"], w["role"],
                [(self.at(ts), x, y) for ts, x, y in w["waypoints"]],
                self.at(w["on_site_from"]) if w.get("on_site_from") else None,
                self.at(w["on_site_until"]) if w.get("on_site_until") else None,
            ))
        full = scenario.get("script", [])
        self._windows = [a for a in full if a["action"] in WINDOW_ACTIONS]
        self.script = sorted([a for a in full if a["action"] not in WINDOW_ACTIONS], key=lambda a: a["at"])
        self._si = 0
        self._register_windows()
        for mid, mc in scenario["machines"].items():
            self.plans[mid] = self._plan_for(mid, mc)

    # ---------------------------------------------------------------- helpers
    def at(self, hh_mm):
        return hhmm(self.day, hh_mm)

    def _plan_for(self, mid, mc):
        """The planning-system view of a machine's day (observable to detectors)."""
        sh = mc["shift"]
        return {
            "machine_id": mid,
            "machine_type": self.machines[mid].type_name,
            "operator_id": mc["operator"],
            "shift_start": self.at(sh["start"]),
            "shift_end": self.at(sh["end"]),
            "breaks": [(self.at(a), self.at(b)) for a, b in sh.get("breaks", [])],
            "tasks": [
                {k: tk[k] for k in ("id", "name", "zone", "type", "volume_m3", "soil_hardness")
                 if k in tk} | {
                    "weather_sensitive": tk.get("weather_sensitive", False),
                    "n_trucks": tk.get("n_trucks", 0),
                    "haul_m": tk.get("haul_m", round(self.route.length)),
                }
                for tk in mc.get("tasks", [])
            ],
            "maintenance_history": mc.get("maintenance_history", {}),
            "fuel_at_last_shutdown_pct": mc.get("fuel_at_last_shutdown_pct"),
        }

    def _register_windows(self):
        """Script entries that describe time windows rather than instant actions."""
        for a in self._windows:
            m = self.machines.get(a.get("machine"))
            kind = a["action"]
            self.truth.append({"t": a.get("at"), "machine_id": a.get("machine"), "action": kind,
                               "until": a.get("until"), "task": a.get("task"), "progress": a.get("progress"),
                               "note": a.get("note")})
            if kind == "seatbelt_off":
                m.seatbelt_windows.append((self.at(a["at"]), self.at(a["until"]), False))
            elif kind == "override":
                if "task" in a:
                    m.overrides.append({"task": a["task"], "lo": a["progress"][0], "hi": a["progress"][1],
                                        "params": a["params"]})
                else:
                    m.overrides.append({"t0": self.at(a["at"]), "t1": self.at(a["until"]), "params": a["params"]})
            elif kind == "overspeed":
                m.overspeed_windows.append((self.at(a["at"]), self.at(a["until"]), a["kmh"]))
            elif kind == "maintenance":
                m.maint_windows.append((self.at(a["at"]), self.at(a["until"])))

    def _run_script(self):
        while self._si < len(self.script) and self.at(self.script[self._si]["at"]) <= self.t:
            a = self.script[self._si]
            self._si += 1
            self._apply(a)

    def _apply(self, a):
        m = self.machines.get(a.get("machine"))
        kind = a["action"]
        t = self.t
        truth = {"t": t.isoformat(timespec="seconds"), "machine_id": a.get("machine"), "action": kind}
        if kind == "fuel_theft":
            until = self.at(a["until"])
            litres = m.fuel_l - a["to_pct"] / 100.0 * EXCAVATOR_SPEC["fuel_tank_l"]
            m.theft = (until, max(0.0, litres) / max((until - t).total_seconds(), 1.0))
        elif kind == "inject_fault":
            m.faults[a["fault"]] = {"start": t, "ramp_s": a.get("ramp_min", 60) * 60.0,
                                    "severity": a.get("severity", 1.0), **a.get("params", {})}
            truth["fault"] = a["fault"]
        elif kind == "clear_fault":
            m.faults.pop(a["fault"], None)
            truth["fault"] = a["fault"]
        elif kind == "refuel":
            m.fuel_l = a["to_pct"] / 100.0 * EXCAVATOR_SPEC["fuel_tank_l"]
        elif kind == "drain_water":
            m.water = 1.0
            m.faults.pop("water_ingress", None)
        elif kind == "truck_breakdown":
            m.breakdown = {"start": t, "until": self.at(a["until"]), "collapse_s": a.get("collapse_s", 90.0)}
        elif kind == "sensor_stuck":
            m.stuck[a["channel"]] = a["value"]
            truth["channel"] = a["channel"]
        truth["note"] = a.get("note")
        self.truth.append(truth)

    # ------------------------------------------------------------------- step
    def step(self, dt=1.0):
        self._run_script()
        for w in self.workers:
            w.update(self.t)
        frames = [m.step(self.t, dt) for m in self.machines.values()]
        self.t += timedelta(seconds=dt)
        return frames

    @property
    def done(self):
        return self.t >= self.t_end

    def site_snapshot(self):
        return {
            "ts": self.t.isoformat(timespec="seconds"),
            "weather": self.weather.snapshot(self.t),
            "workers": [
                {"id": w.id, "name": w.name, "role": w.role,
                 "x": round(w.pos[0], 2), "y": round(w.pos[1], 2)}
                for w in self.workers if w.pos is not None
            ],
        }

    def static_site(self):
        return {"zones": ZONES, "geofences": GEOFENCES, "haul_route": HAUL_ROUTE}
