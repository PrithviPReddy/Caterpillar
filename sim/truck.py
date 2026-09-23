"""Articulated haul truck running a load-haul-dump loop against a loader bay."""

import math

import numpy as np

from sim.config import TRUCK_SPEC as SPEC
from sim.site import GEOFENCES, ZONES, dist, point_in_polygon, zone_at

SPEED_ZONES = [g for g in GEOFENCES if g["type"] == "speed_zone"]
QUEUE_S = 25.0


class LoaderBay:
    """Coordinates trucks at one excavator. Queue position is observable (GPS)."""

    def __init__(self, excavator, route):
        self.exc = excavator
        self.route = route
        self.current = None
        self.queue = []

    def active(self):
        e = self.exc
        task = e.task
        return (not e.shift_done and task is not None and task["type"] == "truck_loading"
                and e.shift_start <= e.world.t < e.shift_end)

    def ready_truck(self):
        c = self.current
        return c if c is not None and c.state == "LOADING" else None

    def arrive(self, truck):
        if truck not in self.queue and truck is not self.current:
            self.queue.append(truck)
        self.promote()

    def promote(self):
        if self.current is None and self.queue:
            self.current = self.queue.pop(0)
            self.current.begin_spot()

    def deliver(self, payload_t, final=False):
        tr = self.current
        if tr is None:
            return
        tr.payload += payload_t
        if tr.payload >= SPEC["target_payload_t"] or final:
            tr.depart()
            self.current = None
            self.promote()


class Truck:
    kind = "truck"
    type_name = "articulated_truck"

    def __init__(self, mid, cfg, operator, world, bay):
        self.id = mid
        self.cfg = cfg
        self.op = operator
        self.world = world
        self.rng = world.rng
        self.bay = bay
        self.route = bay.route
        self.x, self.y = cfg.get("start_pos", ZONES["PARK"]["center"])
        self.state = "OFF_PARKED"
        self.state_t = 0.0
        self.s = 0.0
        self.v = 0.0  # m/s
        self.payload = 0.0
        self.engine_on = False
        amb = world.weather.ambient(world.t)
        self.coolant = self.fuel_temp = amb
        self.rpm = 0.0
        self.load = 0.0
        self.fuel_l = cfg.get("fuel_pct", 70.0) / 100.0 * SPEC["fuel_tank_l"]
        self.engine_hours = cfg.get("engine_hours", 4200.0)
        self.trans_p = 0.0
        self.speed_factor = cfg.get("speed_factor", 1.0)
        self.shift_start = world.at(cfg["shift"]["start"])
        self.shift_end = world.at(cfg["shift"]["end"])
        self.overspeed_windows = []  # (t0, t1, kmh)
        self.breakdown = None  # {start, collapse_s, until}
        self.stuck = {}
        self.telemetry = {}
        self.target = None

    # ------------------------------------------------------------ transitions
    def _set(self, st):
        self.state = st
        self.state_t = 0.0

    def begin_spot(self):
        self._set("SPOTTING")

    def depart(self):
        self._set("HAULING")

    def _broken(self, t):
        b = self.breakdown
        return b is not None and b["start"] <= t < b["until"]

    # ------------------------------------------------------------------- step
    def step(self, t, dt):
        self._decide(t, dt)
        self._physics(t, dt)
        self.state_t += dt
        self.telemetry = self._telemetry(t)
        return self.telemetry

    def _move_to(self, target, speed_mps, dt):
        d = dist((self.x, self.y), target)
        self.v = min(speed_mps, d / dt)
        if d <= speed_mps * dt:
            self.x, self.y = target
            return True
        self.x += speed_mps * dt * (target[0] - self.x) / d
        self.y += speed_mps * dt * (target[1] - self.y) / d
        return False

    def _target_speed_kmh(self, t, loaded, direction=1):
        base = SPEC["loaded_speed_kmh"] if loaded else SPEC["empty_speed_kmh"]
        base *= self.speed_factor
        ahead = self.route.point_at(self.s + direction * (8.0 + self.v * self.v / 1.6))
        for z in SPEED_ZONES:
            if point_in_polygon((self.x, self.y), z["polygon"]) or point_in_polygon(ahead, z["polygon"]):
                base = min(base, z["speed_limit_kmh"] - 2.0)
                for t0, t1, kmh in self.overspeed_windows:
                    if t0 <= t < t1:
                        base = kmh
        return base

    def _drive_route(self, t, dt, direction):
        loaded = direction > 0
        tgt = self._target_speed_kmh(t, loaded, direction) / 3.6
        remaining = (self.route.length - self.s) if direction > 0 else (self.s - QUEUE_S)
        tgt = min(tgt, max(2.0, math.sqrt(2 * 0.8 * max(remaining, 0.0))))
        self.v += float(np.clip(tgt - self.v, -1.2 * dt, 0.6 * dt))
        self.s += direction * self.v * dt
        self.x, self.y = self.route.point_at(self.s)

    def _decide(self, t, dt):
        st = self.state
        if self.breakdown and self._broken(t) and self.trans_p < 700 and st not in ("OFF_PARKED", "DOWN"):
            self._resume = st
            if self.bay.current is self:  # broken truck is pushed aside, next truck spots
                self.bay.current = None
                self.bay.promote()
                self._resume = "HAULING" if self.payload > 5 else "RETURNING"
            elif self in self.bay.queue:
                self.bay.queue.remove(self)
                self._resume = "RETURNING"
            self._set("DOWN")
            self.v = 0.0
            return
        if st == "DOWN":
            if not self._broken(t):
                self._set(getattr(self, "_resume", "RETURNING"))
            return
        if st == "OFF_PARKED":
            if self.shift_start <= t < self.shift_end and self.bay.active():
                self.engine_on = True
                self._set("TO_LOADER")
            return
        if st == "TO_LOADER":
            q = self.route.point_at(QUEUE_S)
            if self._move_to(q, 3.5, dt):
                self.s = QUEUE_S
                self._set("QUEUED")
                self.bay.arrive(self)
            return
        if st == "QUEUED":
            self.v = 0.0
            if not self.bay.active() and self.bay.current is not self:
                if self in self.bay.queue:
                    self.bay.queue.remove(self)
                self._set("TO_PARK")
            return
        if st == "SPOTTING":
            self.s = max(0.0, QUEUE_S * (1 - self.state_t / 20.0))
            self.x, self.y = self.route.point_at(self.s)
            self.v = QUEUE_S / 20.0
            if self.state_t >= 20.0:
                self.s = 0.0
                self.v = 0.0
                self._set("LOADING")
            return
        if st == "LOADING":
            self.v = 0.0
            return
        if st == "HAULING":
            self._drive_route(t, dt, +1)
            if self.s >= self.route.length - 0.5:
                self.v = 0.0
                self._set("DUMPING")
            return
        if st == "DUMPING":
            self.payload = max(0.0, self.payload * (1 - dt / 25.0)) if self.state_t > 10 else self.payload
            if self.state_t >= 40.0:
                self.payload = 0.0
                self._set("RETURNING")
            return
        if st == "RETURNING":
            self._drive_route(t, dt, -1)
            if self.s <= QUEUE_S + 0.5:
                self.s = QUEUE_S
                self.v = 0.0
                if self.bay.active():
                    self._set("QUEUED")
                    self.bay.arrive(self)
                else:
                    self._set("TO_PARK")
            return
        if st == "TO_PARK":
            if self._move_to(self.cfg.get("start_pos", ZONES["PARK"]["center"]), 3.5, dt):
                self.v = 0.0
                self.engine_on = False
                self._set("OFF_PARKED")

    def _physics(self, t, dt):
        amb = self.world.weather.ambient(t)
        on = self.engine_on
        st = self.state
        moving = self.v > 0.3
        if not on:
            tgt_rpm, tgt_load = 0.0, 0.0
        elif moving:
            tgt_rpm = 1650.0 + 150.0 * min(1.0, self.v / 8.0)
            tgt_load = (72.0 if self.payload > 5 else 42.0) + 6.0 * self.op.aggressiveness
        elif st == "DUMPING":
            tgt_rpm, tgt_load = 1500.0, 35.0
        else:
            tgt_rpm, tgt_load = 800.0, 8.0
        self.rpm += (tgt_rpm - self.rpm) * min(1.0, dt / 1.5)
        self.load += (tgt_load - self.load) * min(1.0, dt / 1.0)
        lf = self.load / 100.0
        c_eq = 82.0 + 0.25 * max(0.0, amb - 20.0) + 9.0 * lf if on else amb
        self.coolant += (c_eq - self.coolant) * dt / (300.0 if on else 2400.0)
        f_eq = amb + 12.0 if on else amb
        self.fuel_temp += (f_eq - self.fuel_temp) * dt / (900.0 if on else 3000.0)
        self.fuel_rate = (SPEC["fuel_idle_lph"] + SPEC["fuel_span_lph"] * lf ** 1.2) if on else 0.0
        self.fuel_l = max(0.0, self.fuel_l - self.fuel_rate * dt / 3600.0)
        if on:
            self.engine_hours += dt / 3600.0
        target_tp = 1800.0 if on else 0.0
        b = self.breakdown
        if on and b and b["start"] <= t < b["until"]:
            f = min(1.0, (t - b["start"]).total_seconds() / b.get("collapse_s", 90.0))
            target_tp = 1800.0 - 1550.0 * f
        self.trans_p += (target_tp - self.trans_p) * min(1.0, dt / 3.0)

    def reported_state(self):
        if not self.engine_on:
            return "OFF"
        return {"TO_LOADER": "TRAVEL", "TO_PARK": "TRAVEL", "SPOTTING": "TRAVEL", "QUEUED": "IDLE",
                "DOWN": "IDLE", "LOADING": "LOADING", "HAULING": "HAULING", "DUMPING": "DUMPING",
                "RETURNING": "RETURNING"}.get(self.state, "IDLE")

    def _telemetry(self, t):
        rng = self.rng
        on = self.engine_on
        tel = {
            "ts": t.isoformat(timespec="seconds"),
            "machine_id": self.id,
            "machine_type": self.type_name,
            "operator_id": self.op.id if on else None,
            "state": self.reported_state(),
            "x": round(self.x, 2), "y": round(self.y, 2),
            "zone": zone_at((self.x, self.y)),
            "route_s": round(self.s, 1),
            "engine_on": on,
            "engine_rpm": round(self.rpm + (rng.normal(0, 8) if on else 0.0), 0),
            "engine_load_pct": round(self.load, 1),
            "fuel_rate_lph": round(self.fuel_rate, 2),
            "fuel_level_pct": round(100 * self.fuel_l / SPEC["fuel_tank_l"], 2),
            "coolant_temp_c": round(self.coolant + rng.normal(0, 0.15), 2),
            "fuel_temp_c": round(self.fuel_temp + rng.normal(0, 0.2), 2),
            "trans_pressure_kpa": round(self.trans_p + (rng.normal(0, 15) if on else 0.0), 0),
            "battery_v": round((27.8 if on else 25.2) + rng.normal(0, 0.05), 2),
            "travel_speed_kmh": round(self.v * 3.6, 1),
            "payload_t": round(self.payload, 1),
            "ambient_temp_c": round(self.world.weather.ambient(t), 1),
            "seat_occupied": on,
            "seatbelt_fastened": on,
            "engine_hours": round(self.engine_hours, 3),
        }
        for ch, val in self.stuck.items():
            tel[ch] = val
        return tel
