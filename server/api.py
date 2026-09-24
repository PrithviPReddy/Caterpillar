"""FastAPI server: live telemetry, the event stream for the LangGraph agent, and controls.

    uvicorn server.api:app --port 8100

Agent integration (pick one):
  * SSE:     GET  /api/stream?min_severity=warning      (text/event-stream, one Event per message)
  * Webhook: POST /api/agent/webhook {"url": ...}       (we POST every Event as JSON to your URL)
  * Reply:   POST /api/agent/messages  AgentMessage     (shown in the operator cab UI)
"""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from detect.features import HEALTH_TARGETS
from detect.playbook import PLAYBOOK, TRAINING_MODULES
from detect.schema import AgentMessage, Event
from server.runner import SEV_RANK, SimulationRunner
from sim.config import CHANNELS, EXCAVATOR_SPEC, TRUCK_SPEC
from sim.operators import OPERATORS
from sim.site import GEOFENCES, HAUL_ROUTE, SITE_INFO, ZONES

app = FastAPI(title="Smart Operator Assistant - telemetry & events", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
runner = SimulationRunner()

SERIES_FIELDS = ["minute", "running_frac", "work_frac", "idle_frac", "engine_load_pct", "engine_rpm",
                 "coolant_temp_c", "oil_pressure_kpa", "hyd_oil_temp_c", "egt_c", "fuel_rate_lph",
                 "ambient_temp_c", "fuel_level_pct", "water_in_fuel_pct", "health_score", "end_stops",
                 "task_done_m3", "cab_temp_c"] + [f"exp_{t}" for t in HEALTH_TARGETS] + [f"sig_{t}" for t in HEALTH_TARGETS]


class Control(BaseModel):
    action: str
    value: str | float | None = None


class Webhook(BaseModel):
    url: str | None
    min_severity: str = "info"


def _enc(obj):
    return jsonable_encoder(obj)


@app.get("/")
def root():
    return {"service": "Smart Operator Assistant", "docs": "/docs", "state": "/api/state",
            "stream": "/api/stream", "schema": "/api/schema"}


@app.get("/api/status")
def status():
    with runner.lock:
        return runner.status()


@app.post("/api/control")
def control(c: Control):
    try:
        return runner.control(c.action, c.value)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/state")
def state():
    with runner.lock:
        p = runner.pipeline
        return _enc({
            "status": runner.status(),
            "site": runner.site,
            "machines": {mid: p.snapshot(mid) for mid in p.views},
            "trucks": p.truck_status(),
            "recent_events": p.events[-60:],
            "agent_messages": list(runner.agent_messages)[-30:],
            "incidents": p.recorder.list(),
        })


@app.get("/api/site")
def site():
    with runner.lock:
        w = runner.world
        plans = {mid: {**pl, "tasks": pl["tasks"]} for mid, pl in w.plans.items()}
        return _enc({"info": SITE_INFO, "zones": ZONES, "geofences": GEOFENCES, "haul_route": HAUL_ROUTE,
                     "plans": plans, "operators": {k: v.__dict__ for k, v in OPERATORS.items()},
                     "channels": {k: {"unit": u, "description": d, "spn": s} for k, (u, d, s) in CHANNELS.items()},
                     "specs": {"excavator": EXCAVATOR_SPEC, "truck": TRUCK_SPEC},
                     "scenario": {"name": w.scenario["name"], "title": w.scenario.get("title"),
                                  "date": w.scenario["date"]}})


@app.get("/api/events")
def events(since: int = 0, limit: int = 500, min_severity: str = "info", machine_id: str | None = None):
    with runner.lock:
        out = [e for e in runner.pipeline.events if e["seq"] > since
               and SEV_RANK[e["severity"]] >= SEV_RANK.get(min_severity, 0)
               and (machine_id is None or e["machine_id"] == machine_id)]
        return _enc(out[-limit:])


@app.get("/api/stream")
async def stream(since: int | None = None, min_severity: str = "info",
                 statuses: str = "open,escalated,update,resolved"):
    """Server-sent events. Each message: `event: machine_event`, `data: <Event JSON>`."""
    wanted = set(statuses.split(","))
    floor = SEV_RANK.get(min_severity, 0)

    async def gen():
        seq = runner.pipeline._seq if since is None else since
        idle = 0
        yield "retry: 2000\n\n"
        while True:
            with runner.lock:
                new = [e for e in runner.pipeline.events if e["seq"] > seq]
                if runner.pipeline._seq < seq:  # scenario was reset
                    seq, new = 0, list(runner.pipeline.events)
            for e in new:
                seq = e["seq"]
                if e["status"] in wanted and SEV_RANK[e["severity"]] >= floor:
                    yield f"id: {e['seq']}\nevent: machine_event\ndata: {json.dumps(_enc(e))}\n\n"
            idle = idle + 1 if not new else 0
            if idle >= 50:
                idle = 0
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/timeseries/{machine_id}")
def timeseries(machine_id: str, minutes: int = Query(360, le=900)):
    with runner.lock:
        if machine_id not in runner.pipeline.views:
            raise HTTPException(404, "unknown machine")
        rows = runner.pipeline.minute_series(machine_id, minutes)
        return _enc([{k: r.get(k) for k in SERIES_FIELDS} for r in rows])


@app.get("/api/telemetry/{machine_id}")
def telemetry(machine_id: str, seconds: int = Query(120, le=900)):
    with runner.lock:
        v = runner.pipeline.views.get(machine_id)
        if v is None:
            raise HTTPException(404, "unknown machine")
        return _enc(list(v.buf)[-seconds:])


@app.get("/api/incidents")
def incidents():
    with runner.lock:
        return _enc(runner.pipeline.recorder.list())


@app.get("/api/incidents/{incident_id}")
def incident(incident_id: str):
    with runner.lock:
        rec = runner.pipeline.recorder.get(incident_id)
        if rec is None:
            raise HTTPException(404, "unknown incident")
        return _enc(rec)


@app.get("/api/metrics")
def metrics():
    p = Path(__file__).resolve().parent.parent / "models" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/api/schema")
def schema():
    return {"event": Event.model_json_schema(), "agent_message": AgentMessage.model_json_schema(),
            "event_types": {k: {"default_actions": v[0], "training_module": v[1]} for k, v in PLAYBOOK.items()},
            "training_modules": TRAINING_MODULES}


@app.post("/api/agent/messages")
def post_agent_message(msg: AgentMessage):
    return _enc(runner.add_agent_message(msg.model_dump()))


@app.get("/api/agent/messages")
def get_agent_messages(machine_id: str | None = None, limit: int = 50):
    with runner.lock:
        msgs = [m for m in runner.agent_messages if machine_id is None or m["machine_id"] == machine_id]
        return _enc(msgs[-limit:])


@app.post("/api/agent/webhook")
def set_webhook(w: Webhook):
    runner.webhook_url = w.url or None
    runner.webhook_min_severity = w.min_severity
    return {"url": runner.webhook_url, "min_severity": runner.webhook_min_severity}
