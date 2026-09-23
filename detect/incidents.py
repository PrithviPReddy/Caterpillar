"""Black-box recorder: 60 s before and 30 s after a safety-critical event."""

import json
from datetime import timedelta
from pathlib import Path

INCIDENT_DIR = Path(__file__).resolve().parent.parent / "data" / "incidents"

KEEP = ["ts", "state", "x", "y", "engine_rpm", "engine_load_pct", "travel_speed_kmh", "swing_angle_deg",
        "swing_speed_dps", "reach_m", "max_height_m", "payload_t", "load_moment_pct", "pitch_deg", "roll_deg",
        "impact_g", "seat_occupied", "seatbelt_fastened", "control_activity", "hyd_pressure_mpa"]


class IncidentRecorder:
    def __init__(self, out_dir=INCIDENT_DIR, write_files=True):
        self.dir = Path(out_dir)
        self.write_files = write_files
        self.pending = []
        self.index = {}  # incident_id -> summary

    def start(self, view, event, now, site):
        inc_id = f"INC-{now:%Y%m%d-%H%M%S}-{view.id}"
        if inc_id in self.index:
            return inc_id
        rec = {
            "incident_id": inc_id,
            "machine_id": view.id,
            "trigger_event_id": event["event_id"],
            "trigger_type": event["type"],
            "severity": event["severity"],
            "title": event["title"],
            "summary": event["summary"],
            "trigger_ts": now.isoformat(timespec="seconds"),
            "site": site,
            "pre": [{k: t.get(k) for k in KEEP} for t in list(view.buf)[-60:]],
            "post": [],
            "complete": False,
        }
        self.pending.append((rec, view, now + timedelta(seconds=30)))
        self.index[inc_id] = rec
        return inc_id

    def tick(self, now):
        still = []
        for rec, view, until in self.pending:
            if view.tel is not None:
                rec["post"].append({k: view.tel.get(k) for k in KEEP})
            if now >= until:
                rec["complete"] = True
                if self.write_files:
                    self.dir.mkdir(parents=True, exist_ok=True)
                    (self.dir / f"{rec['incident_id']}.json").write_text(json.dumps(rec, default=str))
            else:
                still.append((rec, view, until))
        self.pending = still

    def get(self, inc_id):
        return self.index.get(inc_id)

    def list(self):
        return [{k: r[k] for k in ("incident_id", "machine_id", "trigger_type", "severity", "title", "trigger_ts",
                                   "complete")} for r in self.index.values()]
