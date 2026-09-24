# Connecting the LangGraph agent

The detection layer turns machine telemetry into **events**. Your agent receives each event, decides what it means for the operator, and posts a short message back. The message appears in the operator cab next to the alert.

```
simulator ──telemetry──▶ detection (rules + ML) ──Event──▶ LangGraph agent ──AgentMessage──▶ cab UI
```

You don't need the simulator running to start building. `docs/sample_events.jsonl` holds every event from the full demo day (139 events, one JSON object per line).

## 1. Receive events

Pick one method.

**Server-sent events.** This is the simplest option and needs no public URL.

```python
import json, requests

with requests.get("http://localhost:8100/api/stream",
                  params={"min_severity": "warning", "statuses": "open,escalated"},
                  stream=True) as r:
    buf = []
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data:"):
            buf.append(line[5:].strip())
        elif line == "" and buf:
            event = json.loads("".join(buf)); buf = []
            handle(event)          # -> your graph
```

**Webhook.** Your agent exposes an HTTP endpoint and the server posts each event to it as JSON:

```bash
curl -X POST localhost:8100/api/agent/webhook -H 'content-type: application/json' \
     -d '{"url": "http://localhost:8123/events", "min_severity": "warning"}'
```

**Backfill.** To catch up on anything you missed, call `GET /api/events?since=<seq>`.

## 2. Reply to the operator

```bash
curl -X POST localhost:8100/api/agent/messages -H 'content-type: application/json' -d '{
  "event_id": "EVT-000041",
  "incident_key": "EXC001:COOLING_DEGRADATION_DETECTED",
  "machine_id": "EXC001",
  "severity": "warning",
  "headline": "Cooling system losing capacity",
  "message": "Coolant is running 2 °C hotter than it should for this load. Finish this truck, then idle at mid-throttle; I have asked the mechanic to blow out the radiator at the 10:00 break.",
  "actions": ["Ease off heavy digging", "Radiator check at 10:00 break"],
  "requires_ack": false,
  "speak": false
}'
```

Always include `incident_key` or `event_id`. The cab UI uses it to attach your message to the right alert. Until you reply, the UI shows the event's `recommended_actions` (the playbook fallback).

`scripts/mock_agent.py` is a working client: it streams events, composes a message and posts it back. To go live, replace its `compose()` function with a call into your graph.

## 3. The event schema (v1.0)

The full JSON Schema is at `GET /api/schema`. These fields matter most:

| Field | Meaning |
|---|---|
| `incident_key` | Stable id for one alert across its lifecycle, e.g. `EXC001:FUEL_SHORTFALL` |
| `status` | `open`, `escalated` (severity went up), `update` (prediction moved), `resolved`. Treat `open` and `escalated` as new; use `update` to refresh; use `resolved` to close. |
| `severity` | `info`, `warning`, `critical` |
| `category` | `safety`, `machine_health`, `fuel_energy`, `productivity`, `operator_behavior`, `security`, `planning`, `sensor_health`, `site`. A good first routing key for the graph. |
| `type` | Machine-readable type, e.g. `PROXIMITY_DANGER_ZONE`. The full list with default actions is in `/api/schema`. |
| `intelligence_level` | How the alert was produced: `L1_threshold`, `L2_trend`, `L3_ml_pattern`, `L4_predictive`, `L5_contextual` |
| `summary` | A deterministic sentence with the key numbers. You can quote it directly. |
| `evidence` | Map of signal to `{value, unit, threshold, expected, deviation_sigma, trend_per_min, baseline, spn, note}`. `expected` is what the ML model predicted for current conditions. |
| `prediction` | For forward-looking events: `{kind, eta, minutes_to_impact, value, detail}`, e.g. fuel run-out time and the suggested refuel break |
| `context` | Machine state, location/zone, current task (progress, ML forecast finish, planned finish), shift (next break), live weather |
| `recommended_actions` | Baseline playbook steps. Refine these, don't just repeat them. |
| `training_module` | `{id, title}` for the training hub, e.g. `TM-08 Swing radius and ground crew awareness` |
| `incident_id` | Present on safety-critical events. A 90 s black-box recording is at `GET /api/incidents/{id}`. |
| `related_incident_keys` | Links between alerts, e.g. `FUEL_SHORTFALL` links to `FUEL_LOSS_ENGINE_OFF` (the overnight theft that caused it) |
| `dtc` | J1939-style `{spn, fmi}` where one applies |

Site-wide events such as lightning and wind use `machine_id: "SITE"`.

## 4. Tools the agent can call

Every endpoint is read-only JSON on `http://localhost:8100`.

| Tool | Endpoint | Use it for |
|---|---|---|
| Live machine snapshot | `GET /api/state` → `machines[<id>]` | Latest telemetry, active alerts, fuel forecast, fatigue, plan |
| Recent telemetry | `GET /api/telemetry/{id}?seconds=120` | Raw 1 Hz signals |
| Trends | `GET /api/timeseries/{id}?minutes=240` | Minute averages plus ML expected values (`exp_*`) |
| Plan | `GET /api/site` → `plans` | Shift, breaks, task list |
| Incident replay | `GET /api/incidents/{id}` | What happened in the 60 s before and 30 s after |

## 5. Suggested graph

1. **Route** on `category` and `severity`. Critical safety events take a fast path: a short imperative message with `speak: true`.
2. **Enrich** by calling the tools above only when needed, e.g. check fuel breaks for `FUEL_SHORTFALL`, or check the last hour for `HARSH_OPERATION`.
3. **Correlate** using `related_incident_keys` and the other active alerts. Example: fuel shortfall plus overnight fuel loss means "you're short because fuel was stolen; security has been told; refuel at 10:00".
4. **Compose** 1 to 3 sentences plus up to 3 actions. Use the operator's name, the number that matters, and when to act.
5. **Coach** after the fact: for `operator_behavior` events and `SHIFT_SUMMARY`, write a debrief and point to the `training_module`.
6. **De-duplicate**: reply once per `incident_key` for `open`, again on `escalated`, and at most every 10 min for `update`.

## 6. Event types in the demo day

| Time | Machine | Type | What the agent should get across |
|---|---|---|---|
| 05:09 | EXC001 | FUEL_LOSS_ENGINE_OFF | Fuel is being siphoned while the machine is parked: alert security |
| 06:00 | EXC001 | SHIFT_BRIEFING, FUEL_SHORTFALL, SERVICE_DUE | Pre-shift briefing: refuel at 10:00, service falls due mid-shift |
| 06:02 | EXC001 | WARMUP_VIOLATION | Heavy digging on a cold engine |
| 06:40 | EXC001 | SEATBELT_UNFASTENED | Critical because the machine is working |
| 07:29 | EXC001 | PROXIMITY_DANGER_ZONE | Surveyor inside the swing radius: STOP (black box recorded) |
| 07:45 | TRK01 | OVERSPEED | 26 km/h in a 15 km/h pedestrian zone |
| 08:40 | TRK03 | SENSOR_FLATLINE | Fuel temperature sensor frozen at 0 °C |
| 09:21 | TRK02 | TRANSMISSION_PRESSURE_LOW | Truck breakdown |
| 09:22 | EXC001 | COOLING_DEGRADATION_DETECTED | **ML early warning**, about 2 h before the fixed alarm |
| 09:37 | EXC001 | TRUCK_WAIT_IDLE | Idle is caused by the fleet, not the operator |
| 10:51 → 11:30 | EXC001 | COOLANT_TEMP_RISING → COOLANT_TEMP_HIGH | Trend, then the threshold alarm, escalating to critical |
| 11:56, 12:47 | EXC001 | STABILITY_LIMIT | Tipping margin exceeded on the 14° slope |
| 12:01 | EXC001 | UNATTENDED_MACHINE | Engine left running over lunch |
| 13:30 | EXC001 | WEATHER_WINDOW | Rain arrives before the trench is finished |
| 13:54 | EXC001 | POWERLINE_CLEARANCE | Boom at 9 m under the 7 m line limit |
| 14:23 → 15:51 | EXC001 | FATIGUE_RISK | Heat plus time in the seat plus jerky controls |
| 14:30 | EXC002 | WATER_IN_FUEL_RISING | Drain the separator; suspect the fuel batch |
| 15:20 → 15:31 | SITE | LIGHTNING_NEARBY | Lower booms; ground crew take shelter |
| 16:00 | EXC001 | HOT_SHUTDOWN, SHIFT_SUMMARY | Coaching debrief and training modules |

## 7. Cocoon backend (langgraph-agent)

### Live bridge: telemetry and detected alerts

`scripts/cocoon_bridge.py` streams the simulator into the Cocoon backend:
- **Telemetry:** `POST /v1/sessions/{id}/telemetry` for EXC001, mapped to `EXC_DEMO_001 / OP_DEMO_1_1`. The backend's own rules (seatbelt, idle, idle-unbuckled) run on it. It includes `seat_occupied`, so a machine left running with nobody in the seat is not reported as an unbuckled operator.
- **Detected alerts:** `POST /v1/sessions/{id}/detections` for every alert our detection layer raises for EXC001 or the whole site (proximity, stability, power line, ML health, fuel, weather, fatigue …). The backend turns each into an alert episode, an announcement the voice worker speaks, and "Why?" evidence.
  - Not forwarded: seatbelt, excessive idle and the shift briefing, which the backend already does itself.
  - Also not forwarded: `update` events.
  - One-off notices (hot shutdown, scorecard) are sent with `one_shot: true`.

```bash
./run_demo.sh --cocoon      # starts the local backend copy (cocoon/) if nothing runs on :8000, then the bridge
scripts/run_cocoon_backend.sh                                    # just the backend (mock LLM mode, :8000)
python scripts/cocoon_bridge.py --session EXC_DEMO_001=ses_...   # post into the phone's session
```

- **Capabilities:** the bridge reads the backend's `/openapi.json`. Against a backend without the detections route (Developer B's version before D1), it sends telemetry only, without `seat_occupied`.
- **Time:** sim time is site-local (Asia/Kolkata), re-dated onto today, sent in UTC, and never moves backwards within a session.
- **Ports:** this project's API is on **:8100**; the backend is on :8000.

**The backend change (D1).** `cocoon/` is a copy of the team's Cocoon project with D1 applied:
- the detections route
- `seat_occupied`
- a "Why?" fix
- schema v8, contracts, docs and tests (320 backend tests pass; the voice worker's contract test passes)

`docs/cocoon_D1_detections.patch` is the same change as a patch that applies to the original project (`patch -p1` from its root), for Developer B to review. Details are in `cocoon/API_CONTRACT.md` ("Detections from the simulator's detection layer") and `cocoon/langgraph-agent/docs/HANDOFF.md` (D1).

### Task-time estimator (REQ-05a / REQ-05b)

`python -m ml.task_estimator` calibrates on `Cocoon_Dataset_v1/data/generated/task_history.csv` and writes two files:
- `models/task_estimator_v1.json`: the model
- `models/task_benchmark_v1.json`: its evaluation

`ml.task_estimator.predict(model, task)` uses only the standard library, so the backend can import it or copy it.

- **Model:** `minutes = quantity / rate[type] × skill × weather × ground`, ridge-fitted in log space. Every prediction lists each multiplier and the minutes it adds ("beginner +12.9 min, rain +5.0 min"), plus an 80% range.
- **Reduced configuration (the five provided rows):** those rows have no quantity or ground condition, so the planner's estimate stands in for quantity / rate. Nothing is filled in.
- **Machine age** is not used. Each dataset machine has one age and its own task types, so an age effect can't be learned. Every prediction says so.

| | MAE |
|---|---|
| Holdout, last 6 of 30 days (n = 235): planner estimate | 6.81 min |
| Holdout: estimator | 2.31 min |
| Five provided tasks: planner | 7.6 min |
| Five provided tasks: estimator, frozen before scoring | 5.26 min |

**Caveats (in the report JSON):**
- n = 5.
- The dataset's skill and rain effects are generator assumptions, so the five-task result is a consistency check, not independent validation.
- The dataset has no wind effect, so T005 (windy, +15 min) stays unexplained.
