"""Stream the live simulator into the Cocoon backend: telemetry samples and detected alerts.

The bridge polls this project's API and posts to the Cocoon backend (langgraph-agent):
- `POST /v1/sessions/{session_id}/telemetry`: each mapped machine's readings. The backend's own
  rules (seatbelt, prolonged idle, idle unbelted) run on them.
- `POST /v1/sessions/{session_id}/detections`: every alert our detection layer raises for a mapped
  machine or the whole site (proximity, ML health, fuel, weather, fatigue ...). The backend turns
  each into an alert episode, an announcement for the voice worker, and "Why?" evidence.
  Only sent when the backend has that route (D1); otherwise telemetry only.

    python scripts/cocoon_bridge.py                                   # EXC001 -> EXC_DEMO_001 / OP_DEMO_1_1
    python scripts/cocoon_bridge.py --map TRK01=TRK_DEMO_001:OP_DEMO_4_1 --map EXC001=EXC_DEMO_001:OP_DEMO_1_1
    python scripts/cocoon_bridge.py --session EXC_DEMO_001=ses_...    # post into the phone's existing session

Contract rules it follows (API_CONTRACT.md):
- catalog IDs only; sim IDs are mapped here, never sent (also rewritten inside alert text);
- `simulated: true`, timezone-aware `observed_at` in UTC;
- observation time never goes backwards (a sim reset starts the next service day), so no sample is stale;
- stable `event_id` / `detection_id`, so a retry is a duplicate, not a second observation;
- only fields the backend's schema declares (the request models forbid unknown fields).

Samples are sent when the reported state changes (engine, seat, belt, operating state, speed band) plus a
heartbeat every `--heartbeat` simulated seconds, not at the full 1 Hz.
"""

import argparse
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import requests

COPILOT = os.environ.get("COPILOT_API", "http://localhost:8100")
SITE_TZ = timezone(timedelta(hours=5, minutes=30))  # Asia/Kolkata, demo/demo_site_v1.json
GENERATOR = "cater.sim.v1"
DETECTOR = "cater.detect.v1"

# sim machine -> (catalog machine, catalog operator); operators follow demo_site_v1.json assignments
DEFAULT_MAP = {"EXC001": ("EXC_DEMO_001", "OP_DEMO_1_1")}

TRAVEL_STATES = {"TRAVEL", "HAULING", "RETURNING"}
IDLE_STATES = {"IDLE"}
# the backend's own policy rules and shift briefing already cover these, so forwarding them would announce twice
BACKEND_OWNED = {"SEATBELT_UNFASTENED", "EXCESSIVE_IDLE", "SHIFT_BRIEFING"}
SIGNAL_KEYS = ("value", "unit", "threshold", "expected", "deviation_sigma", "trend_per_min", "baseline", "spn", "note")


def operating_state(tel):
    """Sim reported state -> contract OperatingState (off | idle | working | travel)."""
    s = tel.get("state")
    if not tel.get("engine_on") or s == "OFF":
        return "off"
    if s in TRAVEL_STATES:
        return "travel"
    if s in IDLE_STATES:
        return "idle"
    return "working"  # DIG / SWING_* / DUMP on excavators, LOADING / DUMPING on trucks


def speed_kph(tel):
    return round(min(max(float(tel.get("travel_speed_kmh") or 0.0), 0.0), 100.0), 1)


class ObservationClock:
    """Maps naive sim time (site-local) onto a service date in UTC, strictly increasing.

    The sim runs on its scenario date; the backend times rules from `observed_at` and ignores samples
    older than the newest one it applied. So sim time is re-dated onto `service_date`, and when the sim
    restarts (time goes backwards) the mapping moves forward whole days until it is ahead again.
    One clock per session: the backend's ordering is per session, and machines interleave.
    """

    def __init__(self, service_date: date, tz=SITE_TZ, after: datetime | None = None):
        self.service_date = service_date
        self.tz = tz
        self.base = None       # first sim date seen
        self.offset = timedelta(0)
        self.last = after.astimezone(tz) if after else None  # the session's newest accepted observation

    def _local(self, sim_ts: datetime) -> datetime:
        if self.base is None:
            self.base = datetime.combine(sim_ts.date(), dtime())
        return datetime.combine(self.service_date, dtime(), self.tz) + (sim_ts - self.base) + self.offset

    def __call__(self, sim_ts: datetime) -> datetime:
        local = self._local(sim_ts)
        while self.last is not None and local <= self.last:
            self.offset += timedelta(days=1)
            local += timedelta(days=1)
        self.last = local
        return local.astimezone(timezone.utc)

    def peek(self, sim_ts: datetime) -> datetime:
        """The same mapping without enforcing order (detections and forecast times are not samples)."""
        return self._local(sim_ts).astimezone(timezone.utc)


def utc_z(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


@dataclass
class MachineStream:
    sim_id: str
    machine_id: str
    operator_id: str
    heartbeat_s: int = 30
    send_seat: bool = True
    session_id: str | None = None
    clock: ObservationClock | None = None
    last_ts: datetime | None = None
    last_key: tuple | None = None
    last_post_ts: datetime | None = None
    idle_since: datetime | None = None
    sent: int = 0
    counts: dict = field(default_factory=lambda: {"opened": 0, "cleared": 0, "announced": 0, "detections": 0})

    def reset(self):
        self.last_ts = self.last_key = self.last_post_ts = self.idle_since = None

    def samples(self, rows, run_id):
        """New telemetry rows -> request bodies worth sending (state changes + heartbeat)."""
        out = []
        for tel in rows:
            ts = datetime.fromisoformat(tel["ts"])
            if self.last_ts is not None and ts <= self.last_ts:
                continue
            self.last_ts = ts
            state = operating_state(tel)
            if state == "idle":
                self.idle_since = self.idle_since or ts
                idle_s = int((ts - self.idle_since).total_seconds())
            else:
                self.idle_since, idle_s = None, 0
            spd = speed_kph(tel)
            seat = bool(tel.get("seat_occupied"))
            key = (bool(tel.get("engine_on")), bool(tel.get("seatbelt_fastened")), state, int(spd // 5), seat)
            due = self.last_post_ts is None or (ts - self.last_post_ts).total_seconds() >= self.heartbeat_s
            if key == self.last_key and not due:
                continue
            self.last_key, self.last_post_ts = key, ts
            observed = self.clock(ts)
            readings = {"engine_on": key[0], "seatbelt_fastened": key[1], "idle_seconds": min(idle_s, 86_400),
                        "operating_state": state, "speed_kph": spd}
            if self.send_seat:
                readings["seat_occupied"] = seat
            out.append({
                "event_id": f"cater-{run_id}-{self.machine_id}-{int(observed.timestamp())}",
                "observed_at": utc_z(observed),
                "simulated": True,
                "readings": readings,
                "provenance": {"origin": "synthetic_scenario", "generator": GENERATOR,
                               "record_ref": f"{self.sim_id}@{tel['ts']}"},
            })
        return out


# ---------------------------------------------------------------------------------- detections

def _num(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _rename(text, ids):
    for sim_id, machine in ids.items():
        text = re.sub(rf"\b{re.escape(sim_id)}\b", machine, text)
    return text


def detection_body(ev, stream, run_id, ids):
    """One of our schema-v1.0 events -> a DetectionRequest for the backend (catalog IDs, UTC, bounded sizes)."""
    evidence = {}
    for name, sig in list((ev.get("evidence") or {}).items())[:20]:
        sig = {k: _num(sig.get(k)) for k in SIGNAL_KEYS if sig.get(k) is not None}
        sig.setdefault("value", None)
        if isinstance(sig.get("unit"), str):
            sig["unit"] = sig["unit"][:20]
        if isinstance(sig.get("note"), str):
            sig["note"] = sig["note"][:200]
        if isinstance(sig["value"], str):
            sig["value"] = _rename(sig["value"], ids)
        evidence[name] = sig
    pred = None
    if ev.get("prediction"):
        p = ev["prediction"]
        pred = {"kind": str(p.get("kind"))[:40], "minutes_to_impact": _num(p.get("minutes_to_impact")),
                "value": _num(p.get("value")), "unit": (p.get("unit") or None) and str(p["unit"])[:20],
                "eta": utc_z(stream.clock.peek(datetime.fromisoformat(p["eta"]))) if p.get("eta") else None}
    tm = ev.get("training_module")
    key = re.sub(r"[^A-Za-z0-9._:\-]", "_", _rename(ev["incident_key"], ids))[:128]
    return {
        "detection_id": f"cater-{run_id}-{ev['event_id']}",
        "observed_at": utc_z(stream.clock.peek(datetime.fromisoformat(ev["ts"]))),
        "simulated": True,
        "episode_key": key,
        "status": ev["status"],
        "detection_type": ev["type"],
        "category": ev["category"],
        "intelligence_level": ev["intelligence_level"],
        "severity": ev["severity"],
        "title": _rename(ev["title"], ids)[:120],
        "summary": _rename(ev["summary"], ids)[:600],
        "recommended_actions": [a[:200] for a in (ev.get("recommended_actions") or [])[:5]],
        "evidence": evidence,
        "prediction": pred,
        "training_module": {"module_id": tm["id"], "title": tm["title"][:120]} if tm else None,
        "provenance": {"generator": DETECTOR, "source_event_id": ev["event_id"]},
        "one_shot": bool(ev.get("one_shot")),
    }


def read_env_file(path):
    env = {}
    if path and Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class Backend:
    def __init__(self, base_url, token):
        self.base = base_url.rstrip("/")
        self.http = requests.Session()
        self.http.headers["Authorization"] = f"Bearer {token}"

    def capabilities(self):
        """What this backend accepts: (detections route, seat_occupied reading). Unknown = the v1 baseline."""
        try:
            spec = self.http.get(f"{self.base}/openapi.json", timeout=10).json()
        except (requests.RequestException, ValueError):
            return False, False
        readings = spec.get("components", {}).get("schemas", {}).get("TelemetryReadings", {})
        return ("/v1/sessions/{session_id}/detections" in spec.get("paths", {}),
                "seat_occupied" in readings.get("properties", {}))

    def ensure_session(self, room, identity, operator_id, machine_id):
        r = self.http.post(f"{self.base}/v1/sessions", timeout=10, json={
            "client_session_key": f"lk:{room}:{identity}", "room_name": room,
            "participant_identity": identity, "operator_id": operator_id, "machine_id": machine_id})
        if r.status_code >= 400:
            raise SystemExit(f"session create failed for {machine_id}: {r.status_code} {r.text}")
        return r.json()["session_id"]

    def newest_observation(self, session_id):
        r = self.http.get(f"{self.base}/v1/sessions/{session_id}/state", timeout=10)
        r.raise_for_status()
        ts = (r.json().get("machine_state") or {}).get("observed_at")
        return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None

    def post(self, session_id, route, body, ident):
        """Returns the result, or None for a request the backend rejected (logged, not retried)."""
        for attempt in range(6):
            try:
                r = self.http.post(f"{self.base}/v1/sessions/{session_id}/{route}", json=body, timeout=15,
                                   headers={"X-Request-ID": f"{ident}.{attempt}"[:128]})
            except requests.RequestException as exc:
                wait = min(2 ** attempt, 10)
                print(f"  backend unreachable ({exc.__class__.__name__}); retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 200:
                return r.json()
            err = (r.json().get("error") if r.headers.get("content-type", "").startswith("application/json")
                   else None) or {}
            if r.status_code == 401:
                raise SystemExit("backend said 401: check COCOON_SERVICE_TOKEN")
            if err.get("retryable") or r.status_code >= 500:
                time.sleep(min(2 ** attempt, 10))
                continue
            print(f"  rejected {ident}: {r.status_code} {err.get('code')} {err.get('message')} "
                  f"{err.get('details') or ''}", flush=True)
            return None
        print(f"  gave up on {ident} after retries", flush=True)
        return None


def parse_map(specs):
    if not specs:
        return dict(DEFAULT_MAP)
    out = {}
    for spec in specs:
        try:
            sim_id, rest = spec.split("=", 1)
            machine, operator = rest.split(":", 1)
        except ValueError:
            raise SystemExit(f"--map expects SIM=MACHINE:OPERATOR, got {spec!r}")
        out[sim_id] = (machine, operator)
    return out


def _note(res):
    parts = [f"opened {res['alerts_opened']}" if res["alerts_opened"] else "",
             f"cleared {res['alerts_cleared']}" if res["alerts_cleared"] else "",
             f"announced {res['announcements_created']}" if res["announcements_created"] else "",
             f"drafts {res['drafts_created']}" if res.get("drafts_created") else "",
             "duplicate" if res["duplicate"] else "",
             f"ignored ({res.get('ignored_reason')})" if res.get("stale") or res.get("ignored_reason") else ""]
    return " ".join(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--copilot", default=COPILOT, help="this project's API (default %(default)s)")
    ap.add_argument("--backend", help="Cocoon backend URL (default COCOON_BACKEND_URL or http://127.0.0.1:8000)")
    ap.add_argument("--env-file", help="langgraph-agent/.env to read COCOON_SERVICE_TOKEN / COCOON_PORT from")
    ap.add_argument("--map", action="append", metavar="SIM=MACHINE:OPERATOR",
                    help="sim machine -> catalog machine and operator (repeatable; default EXC001=EXC_DEMO_001:OP_DEMO_1_1)")
    ap.add_argument("--session", action="append", default=[], metavar="MACHINE=SESSION_ID",
                    help="post into an existing session (e.g. the phone's) instead of creating one")
    ap.add_argument("--room-prefix", default="demo-", help="room for created sessions: <prefix><machine>")
    ap.add_argument("--service-date", help="site-local date the sim day is replayed onto (default: today, IST)")
    ap.add_argument("--heartbeat", type=int, default=30, help="simulated seconds between unchanged samples")
    ap.add_argument("--poll", type=float, default=0.5, help="wall-clock seconds between polls of the simulator")
    ap.add_argument("--no-detections", action="store_true", help="send telemetry only")
    args = ap.parse_args()

    env = {**read_env_file(args.env_file), **os.environ}
    token = env.get("COCOON_SERVICE_TOKEN")
    if not token:
        raise SystemExit("COCOON_SERVICE_TOKEN is not set (export it, or pass --env-file langgraph-agent/.env)")
    base = args.backend or env.get("COCOON_BACKEND_URL") or f"http://127.0.0.1:{env.get('COCOON_PORT') or '8000'}"
    backend = Backend(base, token)
    has_detections, has_seat = backend.capabilities()
    send_detections = has_detections and not args.no_detections

    service_date = (date.fromisoformat(args.service_date) if args.service_date
                    else datetime.now(SITE_TZ).date())
    fixed = dict(s.split("=", 1) for s in args.session)
    mapping = parse_map(args.map)
    ids = {sim_id: machine for sim_id, (machine, _) in mapping.items()}
    streams = {}
    for sim_id, (machine, operator) in mapping.items():
        st = MachineStream(sim_id, machine, operator, heartbeat_s=args.heartbeat, send_seat=has_seat)
        st.session_id = fixed.get(machine) or backend.ensure_session(
            f"{args.room_prefix}{machine}", operator, operator, machine)
        newest = backend.newest_observation(st.session_id)
        st.clock = ObservationClock(service_date, after=newest)
        streams[sim_id] = st
        print(f"{sim_id} -> {machine} / {operator}  session {st.session_id}"
              + (f" (continuing after {newest.isoformat()})" if newest else ""), flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    sim = requests.Session()
    last_sim_time, last_seq = None, 0
    print(f"bridging {args.copilot} -> {base} (service date {service_date}, run {run_id}; "
          f"detections {'on' if send_detections else 'off'}, seat_occupied {'on' if has_seat else 'off'})",
          flush=True)

    while True:
        try:
            status = sim.get(f"{args.copilot}/api/status", timeout=5).json()
            now = datetime.fromisoformat(status["sim_time"])
            if last_sim_time is not None and now < last_sim_time:
                print(f"simulator restarted at {status['sim_time']}; continuing on the next service day", flush=True)
                for st in streams.values():
                    st.reset()
                last_seq = 0
            last_sim_time = now
            # telemetry first, so each detection's evidence has the machine state of its moment
            for st in streams.values():
                rows = sim.get(f"{args.copilot}/api/telemetry/{st.sim_id}", params={"seconds": 900}, timeout=5).json()
                for body in st.samples(rows, run_id):
                    res = backend.post(st.session_id, "telemetry", body, body["event_id"])
                    if res is None:
                        continue
                    st.sent += 1
                    rd = body["readings"]
                    note = _note(res)
                    st.counts["opened"] += len(res["alerts_opened"])
                    st.counts["cleared"] += len(res["alerts_cleared"])
                    st.counts["announced"] += len(res["announcements_created"])
                    if note or st.sent % 20 == 1:
                        print(f"{body['provenance']['record_ref'][-8:]} {st.machine_id} "
                              f"{rd['operating_state']:7s} belt={'on ' if rd['seatbelt_fastened'] else 'OFF'} "
                              f"seat={'yes' if rd.get('seat_occupied', True) else 'no '} "
                              f"idle={rd['idle_seconds']:>4}s {rd['speed_kph']:>5} km/h  {note}", flush=True)
            if send_detections:
                events = sim.get(f"{args.copilot}/api/events", params={"since": last_seq, "limit": 5000},
                                 timeout=5).json()
                for ev in events:
                    last_seq = max(last_seq, ev["seq"])
                    if ev["type"] in BACKEND_OWNED or ev["status"] == "update":
                        continue
                    targets = (list(streams.values()) if ev["machine_id"] == "SITE"
                               else [streams[ev["machine_id"]]] if ev["machine_id"] in streams else [])
                    for st in targets:
                        if st.clock.base is None:  # no telemetry yet for this session
                            continue
                        body = detection_body(ev, st, run_id, ids)
                        res = backend.post(st.session_id, "detections", body, body["detection_id"])
                        if res is None:
                            continue
                        st.counts["detections"] += 1
                        st.counts["opened"] += len(res["alerts_opened"])
                        st.counts["cleared"] += len(res["alerts_cleared"])
                        st.counts["announced"] += len(res["announcements_created"])
                        if res["alerts_opened"] or res["alerts_cleared"]:
                            print(f"{ev['ts'][11:19]} {st.machine_id} {ev['severity']:8s} {ev['status']:9s} "
                                  f"{ev['type']:30s} {_note(res)}", flush=True)
        except requests.RequestException as exc:
            print(f"simulator API not reachable at {args.copilot} ({exc.__class__.__name__}); retrying", flush=True)
            time.sleep(2)
        except KeyboardInterrupt:
            break
        try:
            time.sleep(args.poll)
        except KeyboardInterrupt:
            break
    for st in streams.values():
        print(f"{st.machine_id}: {st.sent} samples, {st.counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
