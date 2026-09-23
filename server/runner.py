"""Background simulation runner: plays a scenario in (accelerated) real time."""

import os
import queue
import threading
import time
from collections import deque
from datetime import datetime

import requests

from detect.models import ModelStore
from detect.pipeline import DetectionPipeline
from sim.scenarios import SCENARIOS, get_scenario
from sim.world import World

SEV_RANK = {"info": 0, "warning": 1, "critical": 2}
SKIP_IGNORE = {"SHIFT_BRIEFING", "SHIFT_SUMMARY", "SERVICE_DUE", "MAINTENANCE_FORECAST"}


class SimulationRunner:
    def __init__(self, scenario="demo_day", speed=60.0):
        self.lock = threading.RLock()
        self.models = ModelStore()
        self.speed = speed
        self.playing = False
        self.mode = "play"  # play | skip | jump
        self.jump_target = None
        self.skip_from_seq = 0
        self.focus_event_id = None
        self.agent_messages = deque(maxlen=300)
        self.webhook_url = os.environ.get("AGENT_WEBHOOK_URL") or None
        self.webhook_min_severity = os.environ.get("AGENT_WEBHOOK_MIN_SEVERITY", "info")
        self.webhook_stats = {"sent": 0, "failed": 0, "last_error": None}
        self._outbox = queue.Queue(maxsize=5000)
        self.load(scenario)
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._webhook_worker, daemon=True).start()

    # ------------------------------------------------------------ control
    def load(self, name):
        with self.lock:
            self.scenario_name = name
            self.world = World(get_scenario(name))
            self.pipeline = DetectionPipeline(self.world.plans, models=self.models)
            self.pipeline.listeners.append(self._on_event)
            self.site = self.world.site_snapshot()
            self.frames = []
            self.playing = False
            self.mode = "play"
            self.focus_event_id = None
            self.agent_messages.clear()
            self.loaded_at = time.time()

    def control(self, action, value=None):
        with self.lock:
            if action == "play":
                self.playing, self.mode = True, "play"
            elif action == "pause":
                self.playing = False
            elif action == "speed":
                self.speed = float(max(1.0, min(float(value), 1800.0)))
            elif action == "reset":
                self.load(value or self.scenario_name)
            elif action == "jump":
                h, m = map(int, str(value).split(":")[:2])
                target = self.world.t.replace(hour=h, minute=m, second=0)
                if target > self.world.t:
                    self.jump_target, self.mode, self.playing = target, "jump", True
            elif action == "skip":
                self.skip_from_seq = self.pipeline._seq
                self.mode, self.playing = "skip", True
            else:
                raise ValueError(f"unknown action {action}")
            return self.status()

    def status(self):
        return {"scenario": self.scenario_name, "scenarios": list(SCENARIOS),
                "sim_time": self.world.t.isoformat(timespec="seconds"),
                "start": self.world.at(self.world.scenario["start"]).isoformat(timespec="seconds"),
                "end": self.world.t_end.isoformat(timespec="seconds"),
                "speed": self.speed, "playing": self.playing, "mode": self.mode, "done": self.world.done,
                "events_total": self.pipeline._seq, "focus_event_id": self.focus_event_id,
                "webhook": {"url": self.webhook_url, **self.webhook_stats},
                "models_loaded": {"health": self.models.has_health, "task": self.models.task_model is not None}}

    # ------------------------------------------------------------ loop
    def _tick(self):
        self.frames = self.world.step()
        self.site = self.world.site_snapshot()
        self.pipeline.ingest(self.frames, self.site, self.world.t)

    def _loop(self):
        last = time.monotonic()
        acc = 0.0
        while True:
            time.sleep(0.05)
            now = time.monotonic()
            dt, last = now - last, now
            with self.lock:
                if not self.playing or self.world.done:
                    acc = 0.0
                    if self.world.done:
                        self.playing = False
                    continue
                if self.mode == "jump":
                    for _ in range(1500):
                        if self.world.done or self.world.t >= self.jump_target:
                            self.playing, self.mode = False, "play"
                            break
                        self._tick()
                    continue
                if self.mode == "skip":
                    for _ in range(1500):
                        if self.world.done:
                            break
                        self._tick()
                        hit = self._skip_hit()
                        if hit:
                            self.focus_event_id = hit["event_id"]
                            self.playing, self.mode = False, "play"
                            break
                    continue
                acc += dt * self.speed
                n = min(int(acc), 3000)
                acc -= n
                for _ in range(n):
                    if self.world.done:
                        break
                    self._tick()

    def _skip_hit(self):
        for ev in reversed(self.pipeline.events):
            if ev["seq"] <= self.skip_from_seq:
                return None
            if (ev["status"] in ("open", "escalated") and SEV_RANK[ev["severity"]] >= 1
                    and ev["type"] not in SKIP_IGNORE):
                return ev
        return None

    # ------------------------------------------------------------ agent I/O
    def _on_event(self, ev):
        if self.webhook_url and SEV_RANK[ev["severity"]] >= SEV_RANK.get(self.webhook_min_severity, 0):
            try:
                self._outbox.put_nowait(ev)
            except queue.Full:
                self.webhook_stats["failed"] += 1

    def _webhook_worker(self):
        session = requests.Session()
        while True:
            ev = self._outbox.get()
            url = self.webhook_url
            if not url:
                continue
            try:
                r = session.post(url, json=ev, timeout=5)
                r.raise_for_status()
                self.webhook_stats["sent"] += 1
            except Exception as exc:  # keep the sim running whatever the agent does
                self.webhook_stats["failed"] += 1
                self.webhook_stats["last_error"] = str(exc)[:200]

    def add_agent_message(self, msg):
        with self.lock:
            rec = dict(msg, received_at=datetime.now().isoformat(timespec="seconds"),
                       sim_time=self.world.t.isoformat(timespec="seconds"))
            self.agent_messages.append(rec)
            return rec
