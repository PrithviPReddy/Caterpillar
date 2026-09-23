"""Hydraulic excavator: operator-driven state machine + first-order engine/hydraulic physics.

Hidden state (never exported in telemetry): operator traits, fatigue, fault progress,
cooling efficiency, filter clog factor. Detectors must infer these from the sensors.
"""

import math

import numpy as np

from sim.config import EXCAVATOR_SPEC as SPEC
from sim.physics import (TASK_BUCKET_M3, TASK_REPOSITION, bucket_payload_t, cycle_plan,
                         fatigue_rate_per_h, heat_index_c)
from sim.site import ZONES, dist, terrain_at, zone_at

WORK_PHASES = ("DIG", "SWING_LOADED", "DUMP", "SWING_EMPTY")
PHASE_LOAD = {"DIG": 85.0, "SWING_LOADED": 62.0, "DUMP": 40.0, "SWING_EMPTY": 45.0}
PHASE_HYD_MPA = {"DIG": 27.0, "SWING_LOADED": 21.0, "DUMP": 13.0, "SWING_EMPTY": 16.0}
PHASE_VIB = {"DIG": 0.35, "SWING_LOADED": 0.18, "DUMP": 0.15, "SWING_EMPTY": 0.15}


def _ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


class Excavator:
    kind = "excavator"
    type_name = "hydraulic_excavator"

    def __init__(self, mid, cfg, operator, world):
        self.id = mid
        self.cfg = cfg
        self.op = operator
        self.world = world
        self.rng = world.rng
        t = world.t
        self.x, self.y = cfg.get("start_pos", ZONES["PARK"]["center"])
        self.heading = cfg.get("start_heading", 0.0)
        amb = world.weather.ambient(t)
        self.coolant = self.oil_temp = self.hyd_temp = self.fuel_temp = amb
        self.egt = amb
        self.cab = amb
        self.rpm = 0.0
        self.load = 0.0
        self.load_s = 0.0
        self.fuel_l = cfg["fuel_pct"] / 100.0 * SPEC["fuel_tank_l"]
        self.def_l = cfg.get("def_pct", 70.0) / 100.0 * SPEC["def_tank_l"]
        self.engine_hours = cfg["engine_hours"]
        self.next_service_h = cfg.get("next_service_h", math.ceil(self.engine_hours / 500.0) * 500.0)
        self.water = cfg.get("water_in_fuel_pct", 4.0)
        self.air_clog = cfg.get("air_filter_clog", 1.0)
        self.air_clog_growth_per_h = cfg.get("air_filter_growth_per_day", 0.03) / 10.0
        self.cab_ac = cfg.get("cab_ac", "good")
        self.battery_rest = 25.3

        self.engine_on = False
        self.activity = "OFF"
        self.act_t = 0.0
        self.act_dur = 0.0
        self.phase = None
        self.phase_t = 0.0
        self.cyc = None
        self.seat = False
        self.belt = False
        self.fatigue = cfg.get("fatigue0", 0.05)
        self.swing = 0.0
        self.swing_speed = 0.0
        self.reach = 5.0
        self.tip_h = 1.0
        self.payload = 0.0
        self.cyc_payload = 0.0
        self.hesitate = 0.0
        self.end_stop_hold = 0.0
        self.end_stop_flag = False
        self.impact = 0.0
        self.relief_rate = 0.0  # EWMA of end-stop hits per minute
        self.travel_target = None
        self.bucket_count = 0
        self.cycles_since_move = 0
        self.theft = None  # (until, litres per second)
        self.faults = {}  # name -> {start, ramp_s, severity}
        self.stuck = {}  # channel -> value
        self.overrides = []  # (t0, t1, dict)
        self.seatbelt_windows = []  # (t0, t1, fastened)
        self.telemetry = {}

        day = world.day
        sh = cfg["shift"]
        self.shift_start = world.at(sh["start"])
        self.shift_end = world.at(sh["end"])
        self.breaks = [(world.at(a), world.at(b)) for a, b in sh.get("breaks", [])]
        self.maint_windows = []
        self.break_policy = dict(cfg.get("break_policy", {}))  # idx -> "running" | "off"
        self.force_cooldown = cfg.get("shift_end_cooldown")  # None => operator trait
        self.shift_done = False
        self.tasks = [dict(tk, done_m3=0.0, started_at=None, finished_at=None) for tk in cfg["tasks"]]
        self.ti = 0
        self.cold_start = True
        self._last_off_t = None
        _ = day

    # ------------------------------------------------------------------ helpers
    @property
    def task(self):
        return self.tasks[self.ti] if self.ti < len(self.tasks) else None

    def _set(self, activity, dur=0.0):
        self.activity = activity
        self.act_t = 0.0
        self.act_dur = dur

    def _break_at(self, t):
        for i, (a, b) in enumerate(self.breaks):
            if a <= t < b:
                return i
        return None

    def _in_window(self, windows, t):
        for w in windows:
            if w[0] <= t < w[1]:
                return w
        return None

    def _override(self, key, default):
        for ov in self.overrides:
            if key not in ov["params"]:
                continue
            if "task" in ov:
                tk = self.task
                if tk and tk["id"] == ov["task"]:
                    prog = tk["done_m3"] / tk["volume_m3"]
                    if ov["lo"] <= prog < ov["hi"]:
                        return ov["params"][key]
            elif ov["t0"] <= self.world.t < ov["t1"]:
                return ov["params"][key]
        return default

    def _bucket(self, task):
        return task.get("bucket_m3", TASK_BUCKET_M3[task["type"]])

    def fault_progress(self, name):
        f = self.faults.get(name)
        if not f:
            return 0.0
        el = (self.world.t - f["start"]).total_seconds()
        return max(0.0, min(1.0, el / max(f["ramp_s"], 1.0))) * f.get("severity", 1.0)

    def _work_pos(self, task):
        if "pos" in task:
            return tuple(task["pos"])
        return ZONES[task["zone"]]["center"]

    # ------------------------------------------------------------ transitions
    def _start_engine(self):
        self.engine_on = True
        self.seat = True
        self.belt = self.rng.random() < self.op.seatbelt_compliance
        self._set("STARTING", 6.0)

    def _stop_engine(self):
        self.engine_on = False
        self.seat = False
        self.belt = False
        self.phase = None
        self.payload = 0.0
        self._last_off_t = self.world.t
        self._set("OFF")

    def _warmup_duration(self):
        if not self.cold_start and self.coolant > 60:
            return self.rng.uniform(20, 40)
        if self.rng.random() < self.op.warmup_compliance:
            return self.rng.uniform(300, 420)
        return self.rng.uniform(40, 80)

    def _go_work(self):
        task = self.task
        if task is None:
            self.travel_target = ZONES["PARK"]["center"]
            self._set("TRAVEL")
            return
        pos = self._work_pos(task)
        if dist((self.x, self.y), pos) > 2.0:
            self.travel_target = pos
            self._set("TRAVEL")
        else:
            self.heading = task.get("heading_deg", 0.0)
            if task["started_at"] is None:
                task["started_at"] = self.world.t
            self.phase = None
            self._set("WORK")

    def _stop_reason(self, t):
        if t >= self.shift_end:
            return "shift_end"
        if self._in_window(self.maint_windows, t):
            return "maint"
        if self._break_at(t) is not None:
            return "break"
        return None

    def _handle_stop(self, reason, t):
        self.phase = None
        self.payload = 0.0
        if reason == "shift_end":
            comply = self.force_cooldown if self.force_cooldown is not None else (
                self.rng.random() < self.op.cooldown_compliance)
            self._set("COOLDOWN", self.rng.uniform(170, 240) if comply else self.rng.uniform(2, 6))
            self._after_cooldown = "shift_done"
        elif reason == "break":
            idx = self._break_at(t)
            pol = self.break_policy.get(idx)
            if pol is None:
                pol = "off" if self.rng.random() < self.op.engine_off_on_break else "running"
            if pol == "running":
                self.seat = False
                self.belt = False
                self._set("UNATTENDED")
            else:
                self._set("COOLDOWN", self.rng.uniform(60, 120) if self.rng.random() < self.op.cooldown_compliance
                          else self.rng.uniform(3, 8))
                self._after_cooldown = "off"
        else:
            self._set("COOLDOWN", self.rng.uniform(30, 60))
            self._after_cooldown = "off"

    def _next_task(self):
        task = self.task
        if task is not None:
            task["finished_at"] = self.world.t
            bay = self.world.bays.get(self.id)
            if bay and bay.current is not None:
                bay.current.depart()
                bay.current = None
                bay.promote()
        self.ti += 1
        self._go_work()

    # ------------------------------------------------------------------- step
    def step(self, t, dt):
        self.end_stop_flag = False
        self.impact = 0.0
        self._decide(t, dt)
        self._physics(t, dt)
        self.act_t += dt
        self.telemetry = self._telemetry(t)
        return self.telemetry

    def _decide(self, t, dt):
        a = self.activity
        if a == "OFF":
            if self.shift_done or self._in_window(self.maint_windows, t):
                return
            if self.shift_start <= t < self.shift_end and self._break_at(t) is None:
                self._start_engine()
            return
        if a == "STARTING":
            if self.act_t >= self.act_dur:
                self._set("WARMUP", self._warmup_duration())
                self.cold_start = False
            return
        if a == "WARMUP":
            if self.act_t >= self.act_dur:
                self._go_work()
            return
        if a == "COOLDOWN":
            if self.act_t >= self.act_dur:
                nxt = getattr(self, "_after_cooldown", "off")
                self._stop_engine()
                if nxt == "shift_done":
                    self.shift_done = True
            return
        if a == "UNATTENDED":
            if self._in_window(self.maint_windows, t):
                self._stop_engine()
            elif self._break_at(t) is None:
                self.seat = True
                self.belt = self.rng.random() < self.op.seatbelt_compliance
                self._go_work()
            return
        if a == "TRAVEL":
            reason = self._stop_reason(t)
            if reason:
                self._handle_stop(reason, t)
                return
            tx, ty = self.travel_target
            d = dist((self.x, self.y), (tx, ty))
            step = SPEC["travel_speed_mps"] * dt
            if d <= step:
                self.x, self.y = tx, ty
                if self.task is None:
                    self._handle_stop("shift_end", t)
                    self.shift_end = min(self.shift_end, t)
                else:
                    self._go_work()
            else:
                self.heading = math.degrees(math.atan2(tx - self.x, ty - self.y)) % 360
                self.x += step * (tx - self.x) / d
                self.y += step * (ty - self.y) / d
            return
        if a == "REPOSITION":
            if self.act_t >= self.act_dur:
                self._set("WORK")
            return
        if a in ("WORK", "WAIT"):
            self._work(t, dt)

    def _work(self, t, dt):
        task = self.task
        if self.phase is None:
            reason = self._stop_reason(t)
            if reason:
                self._handle_stop(reason, t)
                return
            if task is None or task["done_m3"] >= task["volume_m3"]:
                self._next_task()
                return
            if task["type"] == "truck_loading":
                bay = self.world.bays.get(self.id)
                if bay is None or bay.ready_truck() is None:
                    self.activity = "WAIT"
                    return
            self.activity = "WORK"
            rain = self.world.weather.rain_mm_h(t) > 0.5
            self.cyc = cycle_plan(self.op, task["type"], task.get("soil_hardness", 3), self.fatigue,
                                  self.rng, rain, task.get("dump_angle"))
            self.phase = "DIG"
            self.phase_t = 0.0
            if self.cyc["impact"]:
                self.impact = float(self.rng.uniform(2.8, 4.6))
        if self.hesitate > 0:
            self.hesitate -= dt
            return
        if self.rng.random() < max(0.0, self.fatigue - 0.35) * 0.012:
            self.hesitate = float(self.rng.uniform(2.0, 5.0))
            return
        self.phase_t += dt
        if self.phase_t < self.cyc[self.phase]:
            return
        self.phase_t -= self.cyc[self.phase]
        if self.phase == "DIG":
            if self.cyc["end_stop"]:
                self.end_stop_hold = 1.2
                self.end_stop_flag = True
                self.relief_rate += 1.0
            self.cyc_payload = bucket_payload_t(self.cyc["fill"], self._bucket(task))
            self.payload = self.cyc_payload
            self.phase = "SWING_LOADED"
        elif self.phase == "SWING_LOADED":
            self.phase = "DUMP"
        elif self.phase == "DUMP":
            vol = self._bucket(task) * self.cyc["fill"]
            task["done_m3"] += vol
            self.bucket_count += 1
            if task["type"] == "truck_loading":
                bay = self.world.bays.get(self.id)
                if bay:
                    bay.deliver(self.cyc_payload, final=task["done_m3"] >= task["volume_m3"])
            self.payload = 0.0
            self.phase = "SWING_EMPTY"
        else:
            self.phase = None
            k, secs = TASK_REPOSITION[task["type"]]
            self.cycles_since_move += 1
            if k and self.cycles_since_move >= k:
                self.cycles_since_move = 0
                self._set("REPOSITION", secs * self.rng.uniform(0.8, 1.3))

    # ---------------------------------------------------------------- physics
    def _kinematics(self, task):
        dig_reach = self._override("dig_reach", task.get("dig_reach", 6.5) if task else 5.5)
        dump_reach = self._override("dump_reach", task.get("dump_reach", 6.0) if task else 5.5)
        dump_h = self._override("dump_height", task.get("dump_height", 4.2) if task else 2.0)
        A = self.cyc["swing_angle"] if self.cyc else 90.0
        ph = self.phase
        f = min(1.0, self.phase_t / max(self.cyc[ph], 0.1)) if ph else 0.0
        if ph == "DIG":
            self.swing = self.rng.normal(0, 1.5)
            self.swing_speed = abs(self.rng.normal(0, 1.0))
            self.reach = dig_reach + 0.8 - 1.6 * f
            self.tip_h = 0.5 - 3.0 * math.sin(math.pi * f)
        elif ph == "SWING_LOADED":
            self.swing = A * (1 - math.cos(math.pi * f)) / 2
            self.swing_speed = A * math.pi / (2 * self.cyc[ph]) * math.sin(math.pi * f)
            self.reach = dig_reach + (dump_reach - dig_reach) * f
            self.tip_h = 0.5 + (dump_h - 0.5) * f
        elif ph == "DUMP":
            self.swing = A
            self.swing_speed = 0.0
            self.reach = dump_reach
            self.tip_h = dump_h
            self.payload = self.cyc_payload * (1.0 - max(0.0, f - 0.2) / 0.8)
        elif ph == "SWING_EMPTY":
            self.swing = A * (1 + math.cos(math.pi * f)) / 2
            self.swing_speed = A * math.pi / (2 * self.cyc[ph]) * math.sin(math.pi * f)
            self.reach = dump_reach + (dig_reach - dump_reach) * f
            self.tip_h = dump_h + (0.5 - dump_h) * f
        else:
            self.swing_speed = 0.0
            self.reach += (4.5 - self.reach) * 0.05
            self.tip_h += (1.2 - self.tip_h) * 0.05

    def _physics(self, t, dt):
        rng = self.rng
        w = self.world.weather
        amb = w.ambient(t)
        rh = w.humidity(t)
        task = self.task
        on = self.engine_on
        a = self.activity
        working = on and a == "WORK" and self.phase is not None and self.hesitate <= 0

        # --- targets
        if not on:
            tgt_rpm, tgt_load, hyd = 0.0, 0.0, 0.0
        elif a == "STARTING":
            tgt_rpm, tgt_load, hyd = SPEC["low_idle_rpm"], 20.0, 2.0
        elif a in ("TRAVEL", "REPOSITION"):
            tgt_rpm, tgt_load, hyd = SPEC["work_rpm"], 55.0, 22.0
        elif working:
            soil = task.get("soil_hardness", 3) if task else 3
            tgt_rpm = SPEC["work_rpm"] + 60.0 * self.op.aggressiveness
            tgt_load = PHASE_LOAD[self.phase] + (5.0 * (soil - 3) + 5.0 * self.op.aggressiveness
                                                  if self.phase == "DIG" else 0.0)
            hyd = PHASE_HYD_MPA[self.phase] + (2.0 * (soil - 3) if self.phase == "DIG" else 0.0)
        else:  # idle-type activities, micro-pauses
            tgt_rpm, tgt_load, hyd = SPEC["low_idle_rpm"], 9.0, 3.5
            if a == "WORK":  # hesitation mid-cycle: engine stays at work speed
                tgt_rpm, tgt_load, hyd = SPEC["work_rpm"], 18.0, 6.0
        if self.end_stop_hold > 0:
            self.end_stop_hold -= dt
            tgt_load, hyd = 100.0, 34.5
        self.rpm += (tgt_rpm - self.rpm) * min(1.0, dt / 1.5)
        self.load += (tgt_load - self.load) * min(1.0, dt / 0.7)
        self.load = float(np.clip(self.load + (rng.normal(0, 2.5) if on else 0.0), 0, 100))
        self.load_s += (self.load - self.load_s) * dt / 120.0
        self.hyd_p = max(0.0, hyd + (rng.normal(0, 0.6) if on else 0.0))
        self.relief_rate *= math.exp(-dt / 600.0)

        # --- kinematics
        if self.phase is not None and on:
            if self.hesitate <= 0:
                self._kinematics(task)
        else:
            self._kinematics(None)
        if a == "TRAVEL":
            self.swing += _ang_diff(0.0, self.swing) * 0.1

        # --- thermal / fluids
        lf = self.load_s / 100.0
        eff = 1.0 - 0.65 * self.fault_progress("radiator_clog")
        if on:
            t_eq = 82.0 + (0.25 * max(0.0, amb - 20.0) + 9.0 * lf) / eff
            tau = 300.0 if self.coolant < t_eq else 400.0
        else:
            t_eq, tau = amb, 2400.0
        self.coolant += (t_eq - self.coolant) * dt / tau
        oil_eq = self.coolant + 4.0 + 8.0 * lf if on else amb
        self.oil_temp += (oil_eq - self.oil_temp) * dt / (420.0 if on else 2400.0)
        hyd_eq = amb + 16.0 + 30.0 * lf + 1.2 * self.relief_rate if on else amb
        self.hyd_temp += (hyd_eq - self.hyd_temp) * dt / (900.0 if on else 3000.0)
        inj = self.fault_progress("injector_wear")
        egt_eq = amb + 150.0 + 380.0 * self.load / 100.0 + 60.0 * inj if on else amb
        self.egt += (egt_eq - self.egt) * min(1.0, dt / (15.0 if on else 300.0))
        fuel_eq = amb + 14.0 if on else amb
        self.fuel_temp += (fuel_eq - self.fuel_temp) * dt / (900.0 if on else 3000.0)
        if on:
            leak = 1.0 - 0.5 * self.fault_progress("oil_pump_wear")
            base = 180.0 + 0.2 * (self.rpm - 900.0) if self.rpm >= 900 else 180.0 * self.rpm / 900.0
            visc = 1.0 + 0.5 * max(0.0, 90.0 - self.oil_temp) / 70.0
            self.oil_p = max(0.0, base * visc * leak + rng.normal(0, 4.0))
            self.fuel_rate = (SPEC["fuel_idle_lph"] + SPEC["fuel_span_lph"] * (self.load / 100.0) ** 1.2) \
                * (1.0 + 0.14 * inj) * (1.0 + rng.normal(0, 0.02))
            if a == "STARTING":
                self.fuel_rate *= 0.6
        else:
            self.oil_p = 0.0
            self.fuel_rate = 0.0
        used = self.fuel_rate * dt / 3600.0
        self.fuel_l = max(0.0, self.fuel_l - used)
        self.def_l = max(0.0, self.def_l - used * SPEC["def_ratio"])
        if self.theft and not on and t < self.theft[0]:
            self.fuel_l = max(0.0, self.fuel_l - self.theft[1] * dt)
        if on:
            self.engine_hours += dt / 3600.0
            self.air_clog += self.air_clog_growth_per_h * dt / 3600.0
            self.water += (0.002 + self.faults.get("water_ingress", {}).get("rate_pct_min", 0.0)) * dt / 60.0
        self.air_dp = 0.4 + 1.2 * (self.rpm / SPEC["work_rpm"]) ** 2 * self.air_clog
        alt = self.fault_progress("alternator_wear")
        if on:
            self.batt = 27.9 - 2.4 * alt + rng.normal(0, 0.05)
        else:
            self.battery_rest -= 0.02 * dt / 3600.0
            self.batt = self.battery_rest + rng.normal(0, 0.03)
        if on:
            cab_eq = 23.0 + 0.1 * (amb - 25.0) if self.cab_ac == "good" else amb - 4.0
        else:
            cab_eq = amb
        self.cab += (cab_eq - self.cab) * dt / 600.0

        # --- fatigue (hidden)
        if self.seat and on:
            hi = heat_index_c(self.cab, rh)
            self.fatigue += fatigue_rate_per_h(self.op, hi, t.hour + t.minute / 60) * dt / 3600.0
        else:
            self.fatigue *= math.exp(-dt / 2400.0)
        self.fatigue = float(np.clip(self.fatigue, 0.0, 1.0))

        # --- seatbelt script windows
        wdw = self._in_window(self.seatbelt_windows, t)
        if wdw and self.seat:
            self.belt = wdw[2]
        elif self.seat and getattr(self, "_belt_forced", False) and not wdw:
            self.belt = True
        self._belt_forced = bool(wdw)

    # -------------------------------------------------------------- telemetry
    def reported_state(self):
        if not self.engine_on:
            return "OFF"
        if self.activity in ("TRAVEL", "REPOSITION"):
            return "TRAVEL"
        if self.activity == "WORK" and self.phase:
            return self.phase
        return "IDLE"

    def _telemetry(self, t):
        rng = self.rng
        on = self.engine_on
        slope, downhill = terrain_at((self.x, self.y))
        rel = math.radians(downhill - self.heading)
        state = self.reported_state()
        dig_pitch = 1.2 if state == "DIG" else 0.0
        pitch = slope * math.cos(rel) + dig_pitch + rng.normal(0, 0.25 if on else 0.02)
        roll = slope * math.sin(rel) + rng.normal(0, 0.25 if on else 0.02)
        loaded = state in ("SWING_LOADED", "DUMP") and self.payload > 0
        moment = ((self.payload if loaded else 0.0) + SPEC["bucket_linkage_t"]) * max(self.reach, 0.0)
        rel_up = math.radians(_ang_diff(self.heading + self.swing, downhill))
        cap = SPEC["rated_moment_tm"] * float(np.clip(1.0 - 1.6 * math.sin(math.radians(slope)) * math.cos(rel_up),
                                                      0.2, 1.15))
        vib = PHASE_VIB.get(state, 0.45 if state == "TRAVEL" else (0.06 if on else 0.0))
        vib = max(0.0, vib + (rng.normal(0, 0.03) if on else 0.0))
        working = state in WORK_PHASES
        activity = 0.0
        jerk = 0.0
        if working and self.hesitate <= 0:
            activity = float(np.clip(rng.normal(0.75, 0.08), 0.3, 1.0))
            jerk = float(np.clip((0.12 + 0.35 * self.op.aggressiveness + 0.35 * self.fatigue)
                                 * rng.uniform(0.8, 1.2), 0, 1))
        elif state == "TRAVEL":
            activity, jerk = 0.5, 0.15 + 0.2 * self.op.aggressiveness
        elif on and self.seat:
            activity = float(abs(rng.normal(0.02, 0.01)))
        max_h = max(self.tip_h + 1.0, 4.8) if on else 4.8
        amb = self.world.weather.ambient(t)
        task = self.task
        tel = {
            "ts": t.isoformat(timespec="seconds"),
            "machine_id": self.id,
            "machine_type": self.type_name,
            "operator_id": self.op.id if (self.seat or on) else None,
            "state": state,
            "x": round(self.x, 2), "y": round(self.y, 2),
            "heading_deg": round(self.heading, 1),
            "zone": zone_at((self.x, self.y)),
            "engine_on": on,
            "engine_rpm": round(self.rpm + (rng.normal(0, 8) if on else 0.0), 0),
            "engine_load_pct": round(self.load, 1),
            "fuel_rate_lph": round(self.fuel_rate, 2),
            "fuel_level_pct": round(100 * self.fuel_l / SPEC["fuel_tank_l"] + (rng.normal(0, 0.15) if on else 0), 2),
            "def_level_pct": round(100 * self.def_l / SPEC["def_tank_l"], 2),
            "coolant_temp_c": round(self.coolant + rng.normal(0, 0.15), 2),
            "oil_pressure_kpa": round(self.oil_p, 1),
            "oil_temp_c": round(self.oil_temp + rng.normal(0, 0.15), 2),
            "fuel_temp_c": round(self.fuel_temp + rng.normal(0, 0.2), 2),
            "egt_c": round(self.egt + (rng.normal(0, 3) if on else 0.0), 1),
            "air_filter_dp_kpa": round(self.air_dp + (rng.normal(0, 0.03) if on else 0.0), 3),
            "water_in_fuel_pct": round(self.water, 2),
            "battery_v": round(self.batt, 2),
            "hyd_oil_temp_c": round(self.hyd_temp + rng.normal(0, 0.15), 2),
            "hyd_pressure_mpa": round(self.hyd_p, 2),
            "pitch_deg": round(pitch, 2),
            "roll_deg": round(roll, 2),
            "swing_angle_deg": round(self.swing % 360, 1),
            "swing_speed_dps": round(self.swing_speed, 1),
            "reach_m": round(self.reach, 2),
            "boom_tip_height_m": round(self.tip_h, 2),
            "max_height_m": round(max_h, 2),
            "payload_t": round(self.payload if loaded else 0.0, 2),
            "load_moment_pct": round(100.0 * moment / cap, 1) if on else 0.0,
            "travel_speed_kmh": (round(3.6 * SPEC["travel_speed_mps"], 1) if self.activity == "TRAVEL"
                                 else (1.5 if state == "TRAVEL" else 0.0)),
            "vibration_g": round(vib, 3),
            "impact_g": round(max(self.impact, vib * 2.0), 2),
            "control_activity": round(activity, 3),
            "control_jerk": round(jerk, 3),
            "cab_temp_c": round(self.cab, 1),
            "ambient_temp_c": round(amb, 1),
            "humidity_pct": round(self.world.weather.humidity(t), 0),
            "seat_occupied": self.seat,
            "seatbelt_fastened": self.belt,
            "end_stop_hit": self.end_stop_flag,
            "engine_hours": round(self.engine_hours, 3),
            "next_service_h": self.next_service_h,
            "bucket_count": self.bucket_count,
            "task_id": task["id"] if task else None,
            "task_done_m3": round(task["done_m3"], 1) if task else None,
            "task_volume_m3": task["volume_m3"] if task else None,
        }
        for ch, val in self.stuck.items():
            tel[ch] = val
        return tel
