"""Turns per-tick detector findings into a clean event stream.

Detectors report a Finding every time they evaluate and the condition holds. The
manager opens an event on first sight, escalates on higher severity, emits
throttled updates when the prediction moves, and resolves after the condition has
been absent for `clear_after_s`.
"""

from dataclasses import dataclass, field
from datetime import timedelta

SEV_RANK = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class Finding:
    key: str
    machine_id: str
    type: str
    category: str
    level: str
    severity: str
    title: str
    summary: str
    evidence: dict = field(default_factory=dict)
    prediction: dict | None = None
    one_shot: bool = False
    cooldown_s: float = 1800.0
    clear_after_s: float = 30.0
    capture_incident: bool = False
    dtc: dict | None = None
    related: list = field(default_factory=list)
    update_token: object = None  # emit an `update` when this changes (throttled)
    update_min_s: float = 600.0


class EventManager:
    def __init__(self, emit_fn):
        self.emit_fn = emit_fn  # (finding, status, now) -> event dict
        self.active = {}  # key -> {"finding", "opened", "last_seen", "last_emit", "token", "event"}
        self.oneshot_last = {}

    def process(self, now, findings):
        out = []
        for f in findings:
            if f.one_shot:
                last = self.oneshot_last.get(f.key)
                if last is not None and (now - last).total_seconds() < f.cooldown_s:
                    continue
                self.oneshot_last[f.key] = now
                out.append(self.emit_fn(f, "open", now))
                continue
            a = self.active.get(f.key)
            if a is None:
                ev = self.emit_fn(f, "open", now)
                self.active[f.key] = {"finding": f, "opened": now, "last_seen": now, "last_emit": now,
                                      "token": f.update_token, "event": ev, "peak": f.severity}
                out.append(ev)
                continue
            a["last_seen"] = now
            prev = a["finding"]
            a["finding"] = f
            if SEV_RANK[f.severity] > SEV_RANK[a["peak"]]:
                a["peak"] = f.severity
                ev = self.emit_fn(f, "escalated", now)
                a["event"], a["last_emit"], a["token"] = ev, now, f.update_token
                out.append(ev)
            elif (f.update_token is not None and f.update_token != a["token"]
                  and (now - a["last_emit"]).total_seconds() >= f.update_min_s):
                ev = self.emit_fn(f, "update", now)
                a["event"], a["last_emit"], a["token"] = ev, now, f.update_token
                out.append(ev)
            _ = prev
        for key in list(self.active):
            a = self.active[key]
            if (now - a["last_seen"]).total_seconds() > a["finding"].clear_after_s:
                f = a["finding"]
                dur = (a["last_seen"] - a["opened"]).total_seconds() / 60
                rf = Finding(**{**f.__dict__, "severity": "info",
                                "title": f"Resolved: {f.title}",
                                "summary": f"Condition cleared after {dur:.0f} min. Last state: {f.summary}"})
                out.append(self.emit_fn(rf, "resolved", now))
                del self.active[key]
        return out

    def active_findings(self, machine_id=None):
        return [a for a in self.active.values()
                if machine_id is None or a["finding"].machine_id == machine_id]

    def is_active(self, key):
        return key in self.active


def fmt_hm(t):
    return t.strftime("%H:%M") if t else None


def minutes(td: timedelta):
    return td.total_seconds() / 60.0
