"""MachineView: everything the detection layer remembers about one machine.

Built only from telemetry, site sensors (worker tags, weather) and the plan.
"""

from collections import Counter, deque
from datetime import datetime, timedelta

from detect.features import WORK_STATES, HealthFeatureState, MinuteAggregator
from sim.config import EXCAVATOR_SPEC, TRUCK_SPEC
from sim.physics import heat_index_c

HOUR = timedelta(hours=1)


class MachineView:
    def __init__(self, mid, plan, baselines):
        self.id = mid
        self.plan = plan
        self.kind = "excavator" if plan["machine_type"] == "hydraulic_excavator" else "truck"
        self.tank_l = EXCAVATOR_SPEC["fuel_tank_l"] if self.kind == "excavator" else TRUCK_SPEC["fuel_tank_l"]
        self.tasks_by_id = {t["id"]: t for t in plan.get("tasks", [])}
        self.buf = deque(maxlen=900)
        self.minutes = deque(maxlen=900)
        self.agg = MinuteAggregator(mid)
        self.hfs = HealthFeatureState()
        self.tel = None
        self.prev = None
        self.just_started = False
        self.just_stopped = False
        self.engine_start_t = None
        self.first_start_t = None
        self.engine_off_t = None
        self.run_s = 0
        self.fuel_first = None
        self.fuel_off_ref = None
        self.fuel_off_ref_t = None
        self.in_seat_since = None
        self.seat_empty_s = 0
        self.belt_off_s = 0
        self.unattended_s = 0
        self.unattended_fuel_l = 0.0
        self.overspeed_s = 0
        self.overspeed_max = 0.0
        self.stationary_s = 0
        self.hot_cab_s = 0
        self.cab_hi = None
        self.warmup_hot_s = 0
        self.warmup_flagged = False
        self.end_stop_times = deque()
        self.impact_times = deque()
        self.micro_pause_times = deque()
        self.pause_run = 0
        self.cycle_starts = deque(maxlen=120)
        self._jerk_sum = 0.0
        self._jerk_n = 0
        self.jerk_minutes = deque(maxlen=120)  # (minute, mean jerk while working)
        self._idle_counter = Counter()
        self.idle_minutes = deque(maxlen=180)  # (minute, Counter(cause -> seconds))
        self.idle_fuel = Counter()
        self.burn_lph = baselines.get("fleet", {}).get("burn_lph", 20.0)
        self.task_on_min = Counter()
        self.task_done_last = {}
        self.task_first_seen = {}
        self.task_last_seen = {}
        self.task_done_hist = deque(maxlen=240)  # (minute, task_id, done_m3) while running
        self.plan_sched = None
        self.briefed = False
        self.summary_done = False
        self.res_ewm = {}
        self.res_streak = Counter()
        self.last_expected = {}
        self.health_score = None
        self.af_peak_today = None
        self.stats = Counter()
        self.event_counts = Counter()
        self.training = {}
        self.derived = {}

    # ------------------------------------------------------------ per tick
    def update(self, tel, now, bay_ready):
        prev = self.tel
        self.prev = prev
        self.tel = tel
        self.buf.append(tel)
        on = tel["engine_on"]
        was_on = prev["engine_on"] if prev else on
        self.just_started = on and not was_on
        self.just_stopped = was_on and not on
        fuel = tel.get("fuel_level_pct")
        if self.fuel_first is None:
            self.fuel_first = fuel
        if self.just_started:
            self.engine_start_t = now
            self.run_s = 0
            self.warmup_hot_s = 0
            self.warmup_flagged = False
            if self.first_start_t is None:
                self.first_start_t = now
        if on:
            self.run_s += 1
            self.stats["engine_on_s"] += 1
            self.stats["fuel_used_l"] += tel.get("fuel_rate_lph", 0.0) / 3600.0
        if self.just_stopped:
            self.engine_off_t = now
            self.fuel_off_ref, self.fuel_off_ref_t = fuel, now
        if not on:
            if self.fuel_off_ref is None:
                self.fuel_off_ref, self.fuel_off_ref_t = fuel, now
            elif fuel is not None and fuel > self.fuel_off_ref + 1.0:  # refuelled
                self.fuel_off_ref, self.fuel_off_ref_t = fuel, now

        seat = tel.get("seat_occupied", False)
        if seat and on:
            if self.in_seat_since is None:
                self.in_seat_since = now
            self.seat_empty_s = 0
        else:
            self.seat_empty_s += 1
            if self.seat_empty_s >= 300:
                self.in_seat_since = None
        state = tel.get("state")
        working = state in WORK_STATES
        moving = working or state in ("TRAVEL", "HAULING", "RETURNING")
        if on and seat and not tel.get("seatbelt_fastened", True):
            self.belt_off_s += 1
            if moving:
                self.stats["belt_off_moving_s"] += 1
        else:
            self.belt_off_s = 0
        if on and seat and moving:
            self.stats["seated_moving_s"] += 1
        if on and not seat:
            self.unattended_s += 1
            self.unattended_fuel_l += tel.get("fuel_rate_lph", 0.0) / 3600.0
        else:
            self.unattended_s = 0
            self.unattended_fuel_l = 0.0

        if self.kind == "excavator":
            self._update_excavator(tel, now, on, seat, state, working, bay_ready)
        else:
            spd = tel.get("travel_speed_kmh", 0.0)
            self.stationary_s = self.stationary_s + 1 if (on and spd < 0.5) else 0

        rec = self.agg.add(tel)
        return rec

    def _update_excavator(self, tel, now, on, seat, state, working, bay_ready):
        if on and self.run_s < 900 and tel["coolant_temp_c"] < 60 and tel["engine_load_pct"] > 70:
            self.warmup_hot_s += 1
        if tel.get("end_stop_hit"):
            self.end_stop_times.append(now)
            self.stats["end_stops"] += 1
        if tel.get("impact_g", 0) > 2.5:
            self.impact_times.append(now)
            self.stats["impacts"] += 1
        for dq in (self.end_stop_times, self.impact_times, self.micro_pause_times):
            while dq and now - dq[0] > HOUR:
                dq.popleft()
        if working and tel.get("control_activity", 1.0) < 0.05:
            self.pause_run += 1
        else:
            if self.pause_run >= 2:
                self.micro_pause_times.append(now)
                self.stats["micro_pauses"] += 1
            self.pause_run = 0
        prev_state = self.prev.get("state") if self.prev else None
        if state == "DIG" and prev_state != "DIG":
            self.cycle_starts.append(now)
        if working:
            self.stats["work_s"] += 1
            if tel.get("control_activity", 0) > 0.05:
                self._jerk_sum += tel.get("control_jerk", 0.0)
                self._jerk_n += 1
                if now < self.plan["shift_start"] + 2 * HOUR:
                    self.stats["jerk_fresh_sum"] += tel.get("control_jerk", 0.0)
                    self.stats["jerk_fresh_n"] += 1
        cab_hi = heat_index_c(tel["cab_temp_c"], tel["humidity_pct"])
        self.cab_hi = cab_hi
        if seat and on and cab_hi >= 32.0:
            self.hot_cab_s += 1
        elif not (seat and on):
            self.hot_cab_s = max(0, self.hot_cab_s - 5)
        if on and state == "IDLE":
            if not seat:
                cause = "unattended"
            elif self.run_s < 480:
                cause = "warmup"
            else:
                task = self.tasks_by_id.get(tel.get("task_id"))
                if task and task["type"] == "truck_loading" and bay_ready == 0:
                    cause = "truck_wait"
                else:
                    cause = "operator_idle"
            self._idle_counter[cause] += 1
            self.stats[f"idle_{cause}_s"] += 1
            self.idle_fuel[cause] += tel.get("fuel_rate_lph", 0.0) / 3600.0

    def on_minute(self, rec):
        self.minutes.append(rec)
        mean_jerk = self._jerk_sum / self._jerk_n if self._jerk_n >= 10 else None
        self.jerk_minutes.append((rec["minute"], mean_jerk))
        self._jerk_sum, self._jerk_n = 0.0, 0
        self.idle_minutes.append((rec["minute"], self._idle_counter))
        self._idle_counter = Counter()
        if rec["running_frac"] >= 0.5:
            if rec.get("fuel_rate_lph") is not None:
                a = 1.0 / 60.0
                self.burn_lph += a * (rec["fuel_rate_lph"] * rec["running_frac"] - self.burn_lph)
            mdt = datetime.fromisoformat(rec["minute"])
            in_break = any(x <= mdt < y for x, y in self.plan.get("breaks", []))
            tid = rec.get("task_id")
            if tid and not in_break and (rec["work_frac"] > 0 or self.task_on_min[tid] > 0):
                self.task_on_min[tid] += 1
                self.task_first_seen.setdefault(tid, mdt)
                self.task_last_seen[tid] = mdt
                self.task_done_hist.append((rec["minute"], rec["task_id"], rec["task_done_m3"]))
        if rec.get("af_peak_kpa"):
            self.af_peak_today = max(self.af_peak_today or 0.0, rec["af_peak_kpa"])

    # ------------------------------------------------------------ helpers
    def recent(self, ch, n):
        out = []
        for tel in list(self.buf)[-n:]:
            v = tel.get(ch)
            if v is not None:
                out.append(v)
        return out

    def median(self, ch, n=10):
        vals = sorted(self.recent(ch, n))
        return vals[len(vals) // 2] if vals else None

    def idle_last(self, minutes=60):
        c = Counter()
        for _, cnt in list(self.idle_minutes)[-minutes:]:
            c.update(cnt)
        return c

    def recent_jerk(self, minutes=15):
        vals = [j for _, j in list(self.jerk_minutes)[-minutes:] if j is not None]
        return sum(vals) / len(vals) if len(vals) >= 5 else None

    def cycle_times(self, minutes=15, now=None):
        cs = list(self.cycle_starts)
        out = []
        for a, b in zip(cs, cs[1:]):
            d = (b - a).total_seconds()
            if d < 90 and (now is None or (now - b).total_seconds() <= minutes * 60):
                out.append(d)
        return out
