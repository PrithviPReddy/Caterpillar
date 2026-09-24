"""Stream the live simulator into the Cocoon backend as SIMULATED telemetry (contract v1).

The bridge polls this project's API for each mapped machine's 1 Hz telemetry and posts
`POST /v1/sessions/{session_id}/telemetry` to the Cocoon backend (langgraph-agent). The backend's
own rules (seatbelt, prolonged idle, idle unbelted) then fire from the physics-driven stream,
and the voice worker speaks the announcements.

    python scripts/cocoon_bridge.py                                   # EXC001 -> EXC_DEMO_001 / OP_DEMO_1_1
    python scripts/cocoon_bridge.py --map TRK01=TRK_DEMO_001:OP_DEMO_4_1 --map EXC001=EXC_DEMO_001:OP_DEMO_1_1
    python scripts/cocoon_bridge.py --session EXC_DEMO_001=ses_...    # post into the phone's existing session

Contract rules it follows (API_CONTRACT.md):
- catalog IDs only; sim IDs are mapped here, never sent;
- `simulated: true`, timezone-aware `observed_at` in UTC;
- observation time never goes backwards (a sim reset starts the next service day), so no sample is stale;
- stable `event_id` per sample, so a retry is a duplicate, not a second observation;
- only the v1 readings fields (the request model forbids unknown fields).

Samples are sent when the reported state changes (engine, belt, operating state, speed band) plus a
heartbeat every `--heartbeat` simulated seconds, not at the full 1 Hz.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import requests

COPILOT = os.environ.get("COPILOT_API", "http://localhost:8100")
SITE_TZ = timezone(timedelta(hours=5, minutes=30))  # Asia/Kolkata, demo/demo_site_v1.json
GENERATOR = "cater.sim.v1"

# sim machine -> (catalog machine, catalog operator); operators follow demo_site_v1.json assignments
DEFAULT_MAP = {"EXC001": ("EXC_DEMO_001", "OP_DEMO_1_1")}

TRAVEL_STATES = {"TRAVEL", "HAULING", "RETURNING"}
IDLE_STATES = {"IDLE"}


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

    def __call__(self, sim_ts: datetime) -> datetime:
        if self.base is None:
            self.base = datetime.combine(sim_ts.date(), dtime())
        local = datetime.combine(self.service_date, dtime(), self.tz) + (sim_ts - self.base) + self.offset
        while self.last is not None and local <= self.last:
            self.offset += timedelta(days=1)
            local += timedelta(days=1)
        self.last = local
        return local.astimezone(timezone.utc)


@dataclass
class MachineStream:
    sim_id: str
    machine_id: str
    operator_id: str
    heartbeat_s: int = 30
    session_id: str | None = None
    clock: ObservationClock | None = None
    last_ts: datetime | None = None
    last_key: tuple | None = None
    last_post_ts: datetime | None = None
    idle_since: datetime | None = None
    sent: int = 0
    counts: dict = field(default_factory=lambda: {"opened": 0, "cleared": 0, "announced": 0})

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
            key = (bool(tel.get("engine_on")), bool(tel.get("seatbelt_fastened")), state, int(spd // 5))
            due = self.last_post_ts is None or (ts - self.last_post_ts).total_seconds() >= self.heartbeat_s
            if key == self.last_key and not due:
                continue
            self.last_key, self.last_post_ts = key, ts
            observed = self.clock(ts)
            out.append({
                "event_id": f"cater-{run_id}-{self.machine_id}-{int(observed.timestamp())}",
                "observed_at": observed.isoformat().replace("+00:00", "Z"),
                "simulated": True,
                "readings": {
                    "engine_on": key[0],
                    "seatbelt_fastened": key[1],
                    "idle_seconds": min(idle_s, 86_400),
                    "operating_state": state,
                    "speed_kph": spd,
                },
                "provenance": {"origin": "synthetic_scenario", "generator": GENERATOR,
                               "record_ref": f"{self.sim_id}@{tel['ts']}"},
            })
        return out


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

    def post(self, session_id, body):
        """Returns the TelemetryResult, or None for a sample the backend rejected (logged, not retried)."""
        for attempt in range(6):
            try:
                r = self.http.post(f"{self.base}/v1/sessions/{session_id}/telemetry", json=body, timeout=15,
                                   headers={"X-Request-ID": f"{body['event_id']}.{attempt}"[:128]})
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
            print(f"  rejected {body['event_id']}: {r.status_code} {err.get('code')} {err.get('message')}",
                  flush=True)
            return None
        print(f"  gave up on {body['event_id']} after retries", flush=True)
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
    args = ap.parse_args()

    env = {**read_env_file(args.env_file), **os.environ}
    token = env.get("COCOON_SERVICE_TOKEN")
    if not token:
        raise SystemExit("COCOON_SERVICE_TOKEN is not set (export it, or pass --env-file langgraph-agent/.env)")
    base = args.backend or env.get("COCOON_BACKEND_URL") or f"http://127.0.0.1:{env.get('COCOON_PORT') or '8000'}"
    backend = Backend(base, token)

    service_date = (date.fromisoformat(args.service_date) if args.service_date
                    else datetime.now(SITE_TZ).date())
    fixed = dict(s.split("=", 1) for s in args.session)
    streams = []
    for sim_id, (machine, operator) in parse_map(args.map).items():
        st = MachineStream(sim_id, machine, operator, heartbeat_s=args.heartbeat)
        st.session_id = fixed.get(machine) or backend.ensure_session(
            f"{args.room_prefix}{machine}", operator, operator, machine)
        newest = backend.newest_observation(st.session_id)
        st.clock = ObservationClock(service_date, after=newest)
        streams.append(st)
        print(f"{sim_id} -> {machine} / {operator}  session {st.session_id}"
              + (f" (continuing after {newest.isoformat()})" if newest else ""), flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    sim = requests.Session()
    last_sim_time = None
    print(f"bridging {args.copilot} -> {base} (service date {service_date}, run {run_id})", flush=True)

    while True:
        try:
            status = sim.get(f"{args.copilot}/api/status", timeout=5).json()
            now = datetime.fromisoformat(status["sim_time"])
            if last_sim_time is not None and now < last_sim_time:
                print(f"simulator restarted at {status['sim_time']}; continuing on the next service day", flush=True)
                for st in streams:
                    st.reset()
            last_sim_time = now
            for st in streams:
                rows = sim.get(f"{args.copilot}/api/telemetry/{st.sim_id}", params={"seconds": 900}, timeout=5).json()
                for body in st.samples(rows, run_id):
                    res = backend.post(st.session_id, body)
                    if res is None:
                        continue
                    st.sent += 1
                    rd = body["readings"]
                    note = [f"opened {res['alerts_opened']}" if res["alerts_opened"] else "",
                            f"cleared {res['alerts_cleared']}" if res["alerts_cleared"] else "",
                            f"announced {res['announcements_created']}" if res["announcements_created"] else "",
                            f"drafts {res['drafts_created']}" if res.get("drafts_created") else "",
                            "duplicate" if res["duplicate"] else "",
                            f"ignored ({res.get('ignored_reason')})" if res["stale"] else ""]
                    note = " ".join(n for n in note if n)
                    st.counts["opened"] += len(res["alerts_opened"])
                    st.counts["cleared"] += len(res["alerts_cleared"])
                    st.counts["announced"] += len(res["announcements_created"])
                    if note or st.sent % 20 == 1:
                        print(f"{body['provenance']['record_ref'][-8:]} {st.machine_id} "
                              f"{rd['operating_state']:7s} belt={'on ' if rd['seatbelt_fastened'] else 'OFF'} "
                              f"idle={rd['idle_seconds']:>4}s {rd['speed_kph']:>5} km/h  {note}", flush=True)
        except requests.RequestException as exc:
            print(f"simulator API not reachable at {args.copilot} ({exc.__class__.__name__}); retrying", flush=True)
            time.sleep(2)
        except KeyboardInterrupt:
            break
        try:
            time.sleep(args.poll)
        except KeyboardInterrupt:
            break
    for st in streams:
        print(f"{st.machine_id}: {st.sent} samples, {st.counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
