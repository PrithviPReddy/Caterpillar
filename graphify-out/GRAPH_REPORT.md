# Graph Report - cater  (2026-09-23)

## Corpus Check
- Corpus is ~33,018 words - fits in a single context window. You may not need a graph.

## Summary
- 507 nodes · 1166 edges · 25 communities (20 shown, 5 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 85 edges (avg confidence: 0.91)
- Token cost: 87,223 input · 0 output

## Community Hubs (Navigation)
- Detection Pipeline Core
- Streamlit Dashboard UI
- Planning & Event Manager
- Sim Config & Physics
- Trucks, Loader & World
- FastAPI Endpoints
- ML Feature Construction
- Excavator State Machine
- ML Training & Baselines
- Demo Event Types
- Agent Integration Contract
- Scenarios & Runners
- Weather Model
- ML Models & Results
- Simulation Runner
- Incidents & Model Loading
- Project Overview & Data
- Dependencies & Serving
- Fault Evaluation Harness
- Agent Tools & Incidents
- Mock LangGraph Agent
- Training Hub & Playbook
- Package Init (OMP Pin)
- Demo Launch Script

## God Nodes (most connected - your core abstractions)
1. `Finding` - 39 edges
2. `DetectionPipeline` - 38 edges
3. `Excavator` - 27 edges
4. `World` - 26 edges
5. `Event Schema v1.0` - 18 edges
6. `MachineView` - 17 edges
7. `Truck` - 17 edges
8. `ModelStore` - 16 edges
9. `PlanningDetector` - 15 edges
10. `SimulationRunner` - 15 edges

## Surprising Connections (you probably didn't know these)
- `7-Minute Demo Script` --semantically_similar_to--> `Demo Day Event Types`  [INFERRED] [semantically similar]
  README.md → docs/INTEGRATION.md
- `Mock Agent (scripts/mock_agent.py)` --references--> `compose()`  [EXTRACTED]
  docs/INTEGRATION.md → scripts/mock_agent.py
- `Physics-based Construction Site Simulator` --references--> `numpy>=1.26`  [INFERRED]
  README.md → requirements.txt
- `Trends (GET /api/timeseries/{id})` --shares_data_with--> `L3 ML Pattern (Residual-based Health Models)`  [INFERRED]
  docs/INTEGRATION.md → README.md
- `Event Schema v1.0` --references--> `pydantic>=2.6`  [INFERRED]
  docs/INTEGRATION.md → requirements.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Telemetry-to-Operator Advice Pipeline** — readme_construction_site_simulator, readme_detection_layer, docs_integration_event_schema_v1_0, readme_langgraph_agent, docs_integration_agentmessage, readme_streamlit_cab_display [EXTRACTED 1.00]
- **Suggested Agent Graph Flow (Route, Enrich, Correlate, Compose, Coach, De-duplicate)** — docs_integration_route_step, docs_integration_enrich_step, docs_integration_correlate_step, docs_integration_compose_step, docs_integration_coach_step, docs_integration_deduplicate_step [EXTRACTED 1.00]
- **Agent Event Delivery Methods (SSE, Webhook, Backfill)** — docs_integration_sse_stream, docs_integration_webhook, docs_integration_backfill [EXTRACTED 1.00]

## Communities (25 total, 5 thin omitted)

### Community 0 - "Detection Pipeline Core"
Cohesion: 0.06
Nodes (35): BehaviorDetector, _op_base(), L5 operator behaviour: warm-up, cool-down, harsh use, idling (with cause),…, HealthMonitor, IncidentRecorder, EventManager, ModelStore, 0 (normal) .. 1 (very anomalous) from the isolation forest on z-scores. (+27 more)

### Community 1 - "Streamlit Dashboard UI"
Cohesion: 0.09
Nodes (50): app, agent_msg_for(), api_get(), api_post(), control(), excavator_tiles(), fetch_state(), live() (+42 more)

### Community 2 - "Planning & Event Manager"
Cohesion: 0.11
Nodes (17): Finding, fmt_hm(), minutes(), Turns per-tick detector findings into a clean event stream. Detectors report a…, add_work_minutes(), PlanningDetector, L4 predictive planning: will today's plan actually work? Combines the plan, the…, Clock time after `minutes` of work starting at t, skipping breaks (overtime… (+9 more)

### Community 3 - "Sim Config & Physics"
Cohesion: 0.10
Nodes (29): dataclasses, datetime, MachineView: everything the detection layer remembers about one machine. Built…, math, numpy, Static configuration: machine specs, sensor channels, alarm thresholds,…, Hydraulic excavator: operator-driven state machine + first-order…, Operator (+21 more)

### Community 4 - "Trucks, Loader & World"
Cohesion: 0.09
Nodes (8): Ground crew member wearing a UWB proximity tag., Worker, LoaderBay, Coordinates trucks at one excavator. Queue position is observable (GPS)., Truck, The planning-system view of a machine's day (observable to detectors)., Script entries that describe time windows rather than instant actions., World

### Community 5 - "FastAPI Endpoints"
Cohesion: 0.12
Nodes (29): asyncio, fastapi, fastapi_encoders, fastapi_middleware_cors, fastapi_responses, get, post, pydantic (+21 more)

### Community 6 - "ML Feature Construction"
Cohesion: 0.10
Nodes (11): _ewm(), HealthFeatureState, MinuteAggregator, Minute-level aggregation and ML feature construction. Shared by live detection…, Only score steady, warm operation (thermostat-regulated regime)., Per-machine running state that turns minute records into model features., scoreable(), _diagnose() (+3 more)

### Community 8 - "ML Training & Baselines"
Cohesion: 0.14
Nodes (17): concurrent_futures, task_feature_row(), joblib, history_day(), Helpers to run simulated days for training data and evaluation., Run a day with telemetry-only views (no alerts). Returns health rows and…, build_baselines(), generate_history() (+9 more)

### Community 9 - "Demo Event Types"
Cohesion: 0.15
Nodes (20): COOLANT_TEMP_HIGH, COOLANT_TEMP_RISING, COOLING_DEGRADATION_DETECTED, Correlate Step, Demo Day Event Types, FUEL_LOSS_ENGINE_OFF, FUEL_SHORTFALL, intelligence_level Field (+12 more)

### Community 10 - "Agent Integration Contract"
Cohesion: 0.21
Nodes (17): Agent Messages Endpoint (POST /api/agent/messages), AgentMessage, Event Backfill (GET /api/events?since), Compose Step, Connecting the LangGraph Agent, De-duplicate Step, Event Category (Routing Key), Event Schema v1.0 (+9 more)

### Community 11 - "Scenarios & Runners"
Cohesion: 0.18
Nodes (11): collections, copy, queue, requests, main(), Run a scenario end-to-end without the UI and print the event timeline. usage:…, Background simulation runner: plays a scenario in (accelerated) real time., get_scenario() (+3 more)

### Community 12 - "Weather Model"
Cohesion: 0.28
Nodes (3): Deterministic diurnal profile plus an optional storm cell., What a weather service would tell the site at time t., Weather

### Community 13 - "ML Models & Results"
Cohesion: 0.23
Nodes (12): Evidence Map (expected, deviation_sigma, trend), Daily Task Dashboard, Hidden Ground Truth, Isolation Forest Model, L3 ML Pattern (Residual-based Health Models), ML Training Pipeline (ml.train), Model Results (Held-out Simulated Days), Predictive Maintenance (+4 more)

### Community 15 - "Incidents & Model Loading"
Cohesion: 0.22
Nodes (7): Black-box recorder: 60 s before and 30 s after a safety-critical event., Loads trained artefacts from models/. Everything degrades gracefully if absent., json, pathlib, playwright_sync_api, Dev helper: screenshot dashboard pages with a local Chrome. python…, sys

### Community 16 - "Project Overview & Data"
Cohesion: 0.24
Nodes (10): DTC (J1939 SPN/FMI), sample_events.jsonl (139 Demo Events), Cause-Injection Scenarios (No Scripted Alerts), Physics-based Construction Site Simulator, Data.csv Sample Dataset, 7-Minute Demo Script, Detection Layer (Rules + ML), IIoT Sample Dataset (+2 more)

### Community 17 - "Dependencies & Serving"
Cohesion: 0.27
Nodes (10): FastAPI Live Replay Server, Streamlit Operator Cab Display, fastapi>=0.110, numpy>=1.26, plotly>=5.20, pydantic>=2.6, Python Dependencies (requirements.txt), requests>=2.31 (+2 more)

### Community 18 - "Fault Evaluation Harness"
Cohesion: 0.28
Nodes (7): detection_day(), Run a day through the full detection pipeline. Returns events and ground truth., _eval_fault(), _eval_normal(), evaluate(), random_day(), A varied normal day. `fault` = (name, start 'HH:MM', ramp_min, params) or None.

### Community 19 - "Agent Tools & Incidents"
Cohesion: 0.32
Nodes (8): Agent Tool Endpoints (Read-only API), Incident Replay (GET /api/incidents/{id}), Trends (GET /api/timeseries/{id}), Enrich Step, incident_id, PROXIMITY_DANGER_ZONE, 90 s Black-Box Recording, Safety Monitoring Features

### Community 20 - "Mock LangGraph Agent"
Cohesion: 0.38
Nodes (6): argparse, compose(), main(), Stand-in for the LangGraph agent, and a reference client for the real one. It…, The whole 'agent' - swap this for graph.invoke({"event": ev})., stream()

### Community 21 - "Training Hub & Playbook"
Cohesion: 0.40
Nodes (6): Coach Step, recommended_actions (Playbook Fallback), SHIFT_SUMMARY, training_module Field, Operator Training Hub (TM-01..TM-11), Detection Playbook (Default Actions + Training Modules)

## Knowledge Gaps
- **7 isolated node(s):** `run_demo.sh script`, `OMP_NUM_THREADS`, `Operator`, `Data.csv Sample Dataset`, `IIoT Sample Dataset` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 124 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `compose()` connect `Mock LangGraph Agent` to `Agent Integration Contract`?**
  _High betweenness centrality (0.275) - this node is a cross-community bridge._
- **Why does `Mock Agent (scripts/mock_agent.py)` connect `Agent Integration Contract` to `Mock LangGraph Agent`?**
  _High betweenness centrality (0.269) - this node is a cross-community bridge._
- **Why does `Connecting the LangGraph Agent` connect `Agent Integration Contract` to `Project Overview & Data`, `Demo Event Types`, `Agent Tools & Incidents`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `Finding` (e.g. with `BehaviorDetector` and `HealthMonitor`) actually correct?**
  _`Finding` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `DetectionPipeline` (e.g. with `BehaviorDetector` and `HealthMonitor`) actually correct?**
  _`DetectionPipeline` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `World` (e.g. with `SimulationRunner` and `Excavator`) actually correct?**
  _`World` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `run_demo.sh script`, `OMP_NUM_THREADS`, `Operator` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._