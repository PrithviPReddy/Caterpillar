# Simulation and detection: the full guide

This guide explains what the project does, from the first simulated second to the alert on screen. Read it top to bottom once. After that, use the file map at the end to find where anything lives.

---

## 1. The idea in one paragraph

We don't have a real Caterpillar machine streaming data, so we built a **virtual construction site** that behaves like one:
- two excavators, three haul trucks, ground crew and weather
- operators with personalities
- engines that heat up under load
- radiators that clog, trucks that break down, fuel that gets stolen overnight

Every simulated second, each machine produces about 45 sensor readings, the way a real telematics unit would. A **detection layer** watches only those readings, just as it would on a real machine, and turns them into **events** (alerts). It uses fixed rules plus ML models trained on simulated history. The events go to the dashboard, the LangGraph agent and the Cocoon backend.

The key design rule: **the scenario never creates alerts.** It only creates *causes*, such as "the radiator starts clogging at 07:45" or "a surveyor walks behind the excavator at 07:27". Whether and when an alert fires is decided by the detectors, from sensor data alone. That's why we can measure how good the detection is.

---

## 2. The whole pipeline

```
 ┌──────────────────────────── sim/ (the virtual site) ────────────────────────────┐
 │  scenario script ──causes──▶ World (clock, 1 tick = 1 s)                         │
 │                               ├─ Weather (heat curve, storm cell)                │
 │                               ├─ Workers (walk along waypoints)                  │
 │                               ├─ Excavators (state machine + physics)            │
 │                               └─ Trucks (load-haul-dump loop, loader queue)      │
 │  hidden: operator traits, fatigue, fault progress   ──▶ `truth` (never read by   │
 │                                                          detectors; used only    │
 │                                                          to grade them)          │
 └───────────────────────────────┬──────────────────────────────────────────────────┘
                                 │ telemetry: ~45 channels per machine per second
                                 ▼
 ┌──────────────────────────── detect/ (the brain) ─────────────────────────────────┐
 │  MachineView (per-machine memory, 1-minute summaries, stats)                     │
 │     ├─ rules.py      L1 thresholds, L2 trends, sensor health, security, weather  │
 │     ├─ safety.py     seatbelt, proximity, stability, tilt, power line, speed     │
 │     ├─ health.py     L3 ML: "what should this read?" vs "what does it read?"     │
 │     ├─ behavior.py   warm-up, hot shutdown, harsh use, idle, fatigue, heat       │
 │     └─ planning.py   L4: fuel/DEF shortfall, service due, task delay, weather    │
 │  EventManager: open → escalated → update → resolved (no spam, no flapping)       │
 │  Event schema v1.0 (+ context, recommended actions, training module)             │
 │  Incident recorder: 90 s "black box" for critical safety events                  │
 └───────────────────────────────┬──────────────────────────────────────────────────┘
                                 ▼
 ┌─ server/ (FastAPI :8100) ─┐   ┌─ consumers ──────────────────────────────────────┐
 │ runner: play/pause/speed/ │──▶│ Streamlit dashboard (app/, :8501)                 │
 │ jump/"next alert"         │   │ LangGraph agent (SSE /api/stream or webhook)      │
 │ REST + SSE + webhook      │   │ mock agent (scripts/mock_agent.py)                │
 └───────────────────────────┘   │ Cocoon backend (scripts/cocoon_bridge.py)         │
                                 └───────────────────────────────────────────────────┘
 ml/ trains the models offline from many simulated days, then evaluates them on held-out days.
```

---

## 3. Running it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ml.train          # once: simulate 30 days, train, evaluate (~3 min)
./run_demo.sh                         # API :8100 + mock agent + dashboard http://localhost:8501
./run_demo.sh --cocoon                # also stream into the Cocoon backend (needs COCOON_SERVICE_TOKEN)
.venv/bin/python scripts/run_headless.py   # whole day without the UI, prints the alert timeline (~15 s)
```

In the dashboard, press **Play** or **Next alert**. Next alert fast-forwards to the next warning or critical alert and pauses there.

---

## 4. Part 1: the virtual site (`sim/`)

### 4.1 The clock

`World.step()` advances time by **one second**, which is one "tick". In each tick it does four things:
1. Runs any script actions that are due (for example, starting a fault).
2. Moves the ground workers along their paths.
3. Steps every machine: the operator decides, the physics update, and the machine emits telemetry.
4. Advances the clock.

The demo day runs from 05:00 to 16:40, which is 42,000 ticks. The server replays it at 60× real time by default, and you can change the speed.

### 4.2 The site (`sim/site.py`)

A flat map in metres (+x east, +y north):

| Zone | What happens there | Slope |
|---|---|---|
| PAD_A | Bulk excavation, loading trucks | 1.5° |
| ZONE_B | Utility trench (water main) | 2° |
| ZONE_C | Batter slope cut | **14°**, which matters for stability |
| ZONE_D | Storm-drain trench under an 11 kV power line | 1° |
| DUMP | Spoil dump at the end of the haul road | 3° |
| PARK | Parking and fuel bay | 0.5° |

It also has three **geofences**:
- an 11 kV overhead line, where the boom must stay below 7 m
- a buried gas main (no digging)
- a pedestrian zone around Pad A with a 15 km/h speed limit

The **haul road** runs about 957 m from the loading point to the dump.

**Weather** follows a daily cycle: cool morning, hottest mid-afternoon, humidity moving the opposite way. It can include a **storm cell**. On the demo day, lightning starts 30 km away at 14:55, comes to 5 km at 15:40, and rain begins at 15:50. A forecast of the storm is "issued" at 13:30, so the planner can react before it arrives.

**Workers** walk between timed waypoints. Their positions are shared the way UWB tags would share them (UWB is a short-range radio used to locate people).

### 4.3 The operators (`sim/operators.py`), hidden personalities

Each operator has traits the detectors **never see**. The detectors only see what those traits do to the telemetry.

| Trait | Maria G. (OP1001) | Jake T. (OP1002) | Leo P. (OP1003) |
|---|---|---|---|
| Skill (faster cycles, fuller bucket) | 0.90 | 0.65 | 0.40 |
| Aggressiveness (end-stops, impacts, jerky controls) | 0.15 | **0.80** | 0.35 |
| Buckles the seatbelt | 99% | 85% | 90% |
| Warms the engine up properly | 95% | **10%** | 60% |
| Cools the engine down before shutdown | 95% | **10%** | 60% |
| Turns the engine off on breaks | 95% | 40% | 30% |
| Fatigue sensitivity | 0.8 | 1.2 | 1.1 |

The demo day puts **Jake on EXC001**, so the hero machine has a driver who is fast but rough, skips warm-ups and tires quickly. Maria runs EXC002. Three truck drivers (OP2001–2003) have milder traits.

**Fatigue** is also hidden. It builds whenever the operator is in the seat with the engine running:

```
rate per hour = (0.07 + 0.15 × heat + 0.05 × afternoon dip) × fatigue sensitivity
heat          = max(0, cab heat index − 27 °C) / 8
```

It uses the heat index of the **cab**, so EXC001's weak A/C on a 37 °C day tires Jake faster. It recovers whenever he is out of the seat or the engine is off, with a time constant of about 40 minutes. Fatigue makes dig cycles slower and less consistent, controls jerkier, and hesitations more frequent. The fatigue detector has to *infer* it from those side-effects plus heat index and time in the seat.

### 4.4 The excavator (`sim/excavator.py`)

**What the operator does (the state machine):**

```
OFF ─start─▶ STARTING (6 s) ─▶ WARMUP ─▶ TRAVEL to the task ─▶ WORK ⟲ ─▶ … ─▶ COOLDOWN ─▶ OFF
                                  │                                ├─ WAIT (no truck under the bucket)
                                  │                                ├─ REPOSITION (track along the trench)
                                  │                                └─ UNATTENDED (left running on a break)
WORK = the dig cycle, repeated:  DIG → SWING_LOADED → DUMP → SWING_EMPTY
```

- **Warm-up:** a compliant operator idles 5–7 minutes after a cold start. A non-compliant one gives it 40–80 seconds, then digs on a cold engine.
- **Dig cycle:** the phase times (`sim/physics.py::cycle_plan`) depend on:
  - task type
  - soil hardness
  - skill (`×(1.25 − 0.35·skill)`)
  - fatigue (`×(1 + 0.25·fatigue)`)
  - rain (`×1.12`)
  - random noise, which grows with fatigue
- **Bucket fill** depends on skill and soil.
- **Harsh events:** each cycle can hit a cylinder end-stop (probability 0.02 + 0.30 × aggressiveness) or a bucket impact (0.01 + 0.10 × aggressiveness).
- **Trucks:** when loading trucks, a bucket can only be dumped if a truck is parked under it. Otherwise the machine goes to WAIT, which produces idle that is **not the operator's fault**.
- **Breaks:** at breaks and at shift end, the operator either cools down properly, shuts down hot, or leaves the engine running and walks off (UNATTENDED).
- **Tasks:** each task has a volume in m³. The machine keeps cycling until the bucket loads add up to that volume, then travels to the next task.

**The physics (first-order models).** Every quantity moves toward an equilibrium value with a time constant, the way real temperatures lag behind the load:

| Signal | Equilibrium | What changes it |
|---|---|---|
| Engine rpm, load | set by activity/phase (DIG 85% load, idle ~9%) | aggressiveness, soil |
| **Coolant temp** | `82 + (0.25·max(0, ambient−20) + 9·load) / efficiency` | **radiator clog** lowers efficiency (up to −65%) |
| Oil temp | coolant + 4 + 8·load | |
| **Oil pressure** | `(180 + 0.2·(rpm−900)) × viscosity × (1 − 0.5·pump wear)` | **oil pump wear**; cold oil reads higher |
| Hydraulic oil temp | `ambient + 16 + 30·load + 1.2·relief rate` | end-stop hits dump heat through the relief valve |
| **Fuel rate** | `idle + span·load^1.2 × (1 + 0.14·injector wear)` | **injector wear** |
| **Exhaust temp (EGT)** | `ambient + 150 + 380·load + 60·injector wear` | **injector wear** |
| Fuel level | falls with fuel burn | **theft** drains it while the engine is off |
| Water in fuel | slow rise | **water ingress** (bad fuel batch) |
| Air-filter pressure drop | grows with rpm² × clog factor | filter slowly clogs day by day |
| Battery voltage | 27.9 V running | alternator wear |
| Cab temperature | 23 °C with good A/C | a weak A/C lets the cab follow the outside heat |

The machine also computes its geometry:
- swing angle, reach, boom-tip height
- pitch and roll from the terrain
- payload
- **load moment** (how close it is to tipping; it gets worse on a slope when reaching downhill)

**Faults** ramp up gradually. `fault_progress` goes from 0 to 1 over the fault's ramp time, so a clog develops over hours the way it would in real life.

### 4.5 The haul trucks (`sim/truck.py`)

The trucks are 730-class articulated trucks running a loop:

```
TO_LOADER → QUEUED → SPOTTING → LOADING → HAULING → DUMPING → RETURNING → (repeat) … TO_PARK
```

- A **LoaderBay** queues the trucks at the excavator. Only one truck is loaded at a time. It leaves when full, and the next one moves up.
- Trucks **slow down ahead of** the 15 km/h pedestrian zone. A scripted "overspeed" window makes a driver ignore it.
- A **breakdown** collapses transmission pressure. The truck goes DOWN and gives up its place in the bay, so the others keep working. Because the fleet is now smaller, the excavator starts waiting.

### 4.6 What a machine reports (telemetry)

Each second, each machine emits one dictionary, `Excavator._telemetry`. The groups of channels are:
- **Identity and state:** timestamp, machine ID, operator ID (only while someone is seated), reported state, position, heading, zone
- **Engine:** rpm, load %, fuel rate, fuel %, DEF %, coolant, oil pressure and temperature, fuel temperature, EGT, air-filter pressure drop, water in fuel, battery voltage, engine hours, next service hours
- **Hydraulics and motion:** hydraulic temperature and pressure, pitch and roll, swing angle and speed, reach, boom-tip height, payload, load moment %, travel speed
- **Operator and cab:** seat occupied, seatbelt fastened, control activity, control jerk, end-stop hit, impact g, vibration, cab temperature
- **Environment and task:** ambient temperature, humidity, task ID, volume done / total, bucket count

Units and J1939 SPN codes (the standard heavy-equipment signal numbers) are in `sim/config.py::CHANNELS`.

**Hidden state never appears here:** operator traits, fatigue and fault progress. A detector that needs to know about a clogging radiator has to work it out from coolant behaviour.

### 4.7 The scenario (`sim/scenarios.py`)

A scenario defines the date, the weather, each machine's starting state (fuel, engine hours, service due), shift and breaks, task list, the workers' paths and a **script of causes**.

**The demo-day script (causes only):**

| Time | Cause injected |
|---|---|
| 05:08–05:26 | Diesel siphoned from EXC001 while it's parked (58% → 24%) |
| 06:40–06:47 | Jake unbuckles to reach his phone and keeps digging |
| 07:27–07:32 | Surveyor Chris D. walks through EXC001's swing radius (a waypoint path, not an alert) |
| 07:45 | EXC001's radiator starts clogging, ramping over 4 hours |
| 07:45–08:05 | TRK01's driver ignores the pedestrian speed zone (27 km/h) |
| 08:30 | TRK03's fuel-temperature sender fails and reads 0 |
| 09:20–10:40 | TRK02's transmission pump fails |
| 10:02 | EXC001 is refuelled to 92% at the break |
| 12:00 | Jake leaves the engine running over lunch (break policy) |
| 12:30–12:45 | A mechanic blows out the radiator (the fault clears) |
| During T-102 | Jake reaches further and further down the 14° slope (35–62% of the task) |
| During T-103 | Jake stacks spoil 8.2 m high under the power line (18–26% of the task) |
| 13:30 | A storm forecast is issued |
| 14:20 | EXC002 gets a contaminated fuel batch (water ingress) |
| 14:55–15:50 | Lightning approaches, then rain |
| 16:00 | Jake shuts down hot, without a cool-down |

`random_day(seed, fault)` generates varied normal days, with different operators, weather, soil, truck counts and tasks, and can include one hidden fault. Those days are the training and test data for the ML.

---

## 5. Part 2: the detection layer (`detect/`)

### 5.1 Per-machine memory (`detect/view.py`)

`DetectionPipeline.ingest(frames, site, now)` is called every tick. For each machine, a **MachineView** keeps:
- the last 900 seconds of raw telemetry (the "recent telemetry" buffer)
- **one-minute summaries** (`detect/features.py::MinuteAggregator`): averages of the key signals, and minutes spent working, idle or running
- counters: seatbelt-off seconds, unattended seconds, idle seconds split by cause, end-stops, impacts, jerk
- task progress: first and last minute the task was seen, volume done
- the ML's current expectation for each health signal, and the health score

It also works out whether a **truck is ready**: a truck within 12 m of the loading point, stopped, with healthy transmission pressure. That's how idle gets its cause. Idle with no truck ready is "waiting on trucks"; idle with a truck ready is on the operator.

### 5.2 The detectors and how often they run

| Detector | Runs every | Produces |
|---|---|---|
| `safety.py` SafetyDetector | 1 s | SEATBELT_UNFASTENED, UNATTENDED_MACHINE, PROXIMITY_WARNING / PROXIMITY_DANGER_ZONE, STABILITY_LIMIT, SLOPE_TILT, POWERLINE_CLEARANCE, NO_DIG_ZONE, OVERSPEED |
| `rules.py` ThresholdDetector | 5 s | L1 limits: coolant, hydraulic oil, EGT, fuel/DEF low, water in fuel, air filter, battery, transmission pressure, OIL_PRESSURE_LOW (the limit depends on rpm) |
| `rules.py` TrendDetector | 30 s | L2: "rising X/min, reaches the limit at HH:MM" (e.g. COOLANT_TEMP_RISING) |
| `rules.py` SensorHealthDetector | 60 s | SENSOR_FLATLINE (a frozen or implausible sensor) |
| `rules.py` SecurityDetector | 10 s | FUEL_LOSS_ENGINE_OFF (theft), ENGINE_START_OUTSIDE_SHIFT |
| `rules.py` SiteWeatherDetector | 30 s | LIGHTNING_NEARBY, HIGH_WIND (site-wide, `machine_id: SITE`) |
| `behavior.py` BehaviorDetector | 5 s + every minute | WARMUP_VIOLATION, HOT_SHUTDOWN, HARSH_OPERATION, EXCESSIVE_IDLE, TRUCK_WAIT_IDLE, FATIGUE_RISK, HEAT_STRESS, SHIFT_SUMMARY |
| `planning.py` PlanningDetector | 60 s | SHIFT_BRIEFING, FUEL_SHORTFALL, DEF_SHORTFALL, SERVICE_DUE / SERVICE_OVERDUE, TASK_DELAY, WEATHER_WINDOW, MAINTENANCE_FORECAST |
| `health.py` HealthMonitor (ML) | every minute | COOLING_DEGRADATION_DETECTED, OIL_PRESSURE_DEVIATION, HYD_TEMP_DEVIATION, FUEL_EFFICIENCY_DEGRADATION |

Some examples of *context* in the rules:
- **Seatbelt:** unbuckled while digging or moving is **critical** after a 15 s grace period. Unbuckled while idling is only **info** after 2 minutes.
- **Proximity:** the danger radius is the excavator's **current reach** (or tail swing, if larger) plus 1 m. There's a 6 m warning ring outside it. It only counts while the machine moves; a parked machine is no danger. For trucks the zones are 6 m and 12 m, and only above 3 km/h.
- **Stability:** load moment is compensated for slope. It gives a warning at 85% of the rated moment and critical at 100%.
- **Power line:** the boom-tip height is checked against the 7 m limit, with a 1 m warning margin.
- **Oil pressure:** the low limit depends on rpm, because pressure is normally lower at idle.
- **Lightning:** warning within 16 km, critical within 10 km.

### 5.3 The five levels of intelligence

Every event says how it was produced (`intelligence_level`):

| Level | Meaning | Example on the demo day |
|---|---|---|
| **L1 threshold** | A fixed limit was crossed | Coolant above 100 °C at 11:16 |
| **L2 trend** | The rate of change points at a limit | 10:51 "coolant rising 0.2 °C/min, reaches 105 °C around 11:33" |
| **L3 ML pattern** | A signal doesn't match what the model expects for current conditions | 09:22: cooling degradation, about 2 h before L1 |
| **L4 predictive** | Plan + models + burn rates + forecast | 06:00 "fuel lasts 5.2 h but 9.1 h of work remain: refuel at 10:00" |
| **L5 contextual** | The same reading means different things in different situations | Seatbelt while digging = critical; idle caused by a truck breakdown isn't the operator's fault |

### 5.4 From findings to clean events (`detect/manager.py`)

Detectors report a **Finding** every time they check and the condition holds, which could be every second. The **EventManager** turns that stream into a clean lifecycle, keyed by `incident_key` (e.g. `EXC001:COOLANT_TEMP_HIGH`):

- **open**: the first time the condition is seen, one event is emitted.
- **escalated**: severity goes up (warning → critical), one event is emitted.
- **update**: the prediction moves meaningfully (e.g. a new ETA). Updates are throttled to at most one every 10 minutes by default.
- **resolved**: the condition has been absent for `clear_after_s`. Each alert type sets how long, so brief dips don't make an alert flap open and closed.
- **one-shot** events (briefing, scorecard, hot shutdown) fire once, with a cooldown.

So the agent gets one message per real situation, not 3,600 per hour.

### 5.5 The event (`detect/schema.py`, Event schema v1.0)

Each event carries:
- **Identity:** `event_id`, `seq`, `incident_key`, `status`, `ts`, `machine_id`, operator
- **Classification:** `category` (safety, machine_health, fuel_energy, …), `type`, `intelligence_level`, `severity`
- **Content:**
  - `title`
  - `summary`, a plain sentence with the key numbers
  - `evidence`: each signal with value, unit, threshold, the ML-expected value, sigma, trend and SPN
  - `prediction` (ETA, minutes to impact)
- **Context:** machine state, location and zone, current task (progress, ML-forecast finish, planned finish), shift (elapsed time, next break), live weather
- **Help:** `recommended_actions` from the playbook, `training_module` (TM-01…TM-11, below), `related_incident_keys` (e.g. a fuel shortfall links to the overnight theft), `dtc` (J1939 SPN/FMI, the standard fault-code format)
- **Black box:** `incident_id` for critical safety events

**Training modules** (`detect/playbook.py`):

| Code | Module |
|---|---|
| TM-01 | Warm-up |
| TM-02 | Seatbelt |
| TM-03 | Smooth controls |
| TM-04 | Idle reduction |
| TM-05 | Cool-down |
| TM-06 | Slopes and lift chart |
| TM-07 | Power lines |
| TM-08 | Swing radius and ground crew |
| TM-09 | Fatigue and heat |
| TM-10 | Fluid checks |
| TM-11 | Haul-road speed around pedestrians |

**Black box** (`detect/incidents.py`): when a critical safety event opens, the recorder saves the **60 s before and 30 s after** it (telemetry, positions, weather) to `data/incidents/`. It's logged as a near miss and can be replayed at `/api/incidents/{id}`.

---

## 6. Part 3: the ML (`ml/`, `detect/health.py`, `detect/models.py`)

### 6.1 Where the training data comes from

`python -m ml.train` simulates **30 varied normal days** in parallel (`random_day`, seeds 1000+). It keeps each excavator's one-minute summaries and computes the model features with **the same code the live detector uses** (`detect/features.py`). That way training and live features can't drift apart.

### 6.2 Health models: "what should this signal read right now?" (L3)

There's one regression model (HistGradientBoosting) per signal. Each predicts the **normal** value from the current conditions:

| Signal | Predicted from | Validation error (RMSE) |
|---|---|---|
| Coolant temp | load (3- and 8-min averages), 15-min ambient average, rpm, minutes running | 0.23 °C (r² 0.98) |
| Oil pressure | rpm, oil temp | 1.2 kPa |
| Hydraulic oil temp | load (8- and 20-min), ambient, relief-valve activity, minutes running | 1.5 °C |
| Fuel rate | load, rpm | 0.11 L/h |
| Exhaust temp | load, ambient, rpm | 12.7 °C |

20% of the days are held out for validation.

**Live alert logic** (every minute, `health.py`):
1. Scoring happens only in steady, warm operation: at least 25 minutes running and coolant above 78 °C. Cold starts would confuse the model.
2. **Residual** = actual − expected, smoothed with an exponential moving average (α = 0.3).
3. A signal counts as deviating when the smoothed residual is at least **max(3σ, a physical minimum)**. The minimums are 2 °C coolant, 20 kPa oil, 5 °C hydraulic, 1.2 L/h fuel and 35 °C exhaust, so tiny but statistically "significant" wobbles never alarm.
4. It must deviate for **3 minutes in a row** before an event opens.
5. The **pattern** across signals gives a probable cause. For example, coolant high with normal fuel and exhaust means the cooling system is losing capacity.
6. When the engine is off, the residuals reset, so a repaired machine starts clean.

**Health score (0–100):** `100 × exp(−0.35 × max(0, r − 0.6))`, where r is the worst deviation as a multiple of its alert line. An **IsolationForest** on the combined z-scores (each residual divided by σ) caps the score when several signals are slightly off at once. The score is floored at 5.

### 6.3 Task-time model (L4, for the simulator's own plan)

A fast cycle-level simulator (`simulate_task_duration_min`) generates **5,000 tasks** with different volume, soil, task type, trucks, haul distance, heat, rain and operator. A HistGradientBoosting model learns `log(1 + minutes)` from those inputs.
- **Result:** 7.4% average error, against 30.2% for the planner's rule of thumb.
- **Live use:** the planning detector combines it with actual progress to forecast each task's finish and to raise TASK_DELAY when the forecast slips past the plan.

### 6.4 Operator baselines

`models/baselines.json` holds each operator's normal rates and the fleet average:
- end-stops per hour and impacts per hour
- jerk (plus "fresh" jerk, measured early in the shift before fatigue)
- idle fraction
- burn rate

HARSH_OPERATION and the fatigue detector compare the operator to their **own** normal.

### 6.5 How we know the ML works (evaluation)

Because the simulator records the hidden truth, we can grade the detectors. We inject each fault on held-out days (8 runs per fault, random start time and ramp) and measure when each method first notices:

| Fault | ML (L3) caught | Median time to detect | Fixed thresholds (L1) caught | Median time |
|---|---|---|---|---|
| Radiator clogging | 100% | 128 min | 75% | 285 min |
| Oil pump wear | 100% | 34 min | 100% | 122 min |
| Injector wear | 100% | 105 min | **0%** (never fires) | — |

**False alarms:** 0 ML alarms across 8 normal days (152 engine-hours). All results are in `models/metrics.json`, and the dashboard's **Model report** page draws them.

### 6.6 The Cocoon task estimator (`ml/task_estimator.py`)

This is a separate estimator calibrated on the team dataset `Cocoon_Dataset_v1` (1,184 tasks) for the Cocoon backend:

```
minutes = quantity / rate[task type] × skill × weather × ground
```

It's a transparent model, and each prediction lists what adds time (e.g. "beginner +12.9 min").
- **Held-out last 6 days:** 2.3 min average error, against 6.8 for the planner's estimate.
- **The 5 real tasks from the problem statement:** 5.3 min, against 7.6. The model was frozen before scoring these rows.
- The caveats are in `models/task_benchmark_v1.json` and `docs/INTEGRATION.md` §7.

---

## 7. Part 4: serving it (`server/`)

**`server/runner.py`, the SimulationRunner.** A background thread steps the world and the detection pipeline in (accelerated) real time. Its controls:

| Control | What it does |
|---|---|
| `play` / `pause` | Start or stop the replay |
| `speed` | Replay speed, 60× by default |
| `reset` | Go back to the start of the day |
| `jump` | Fast-forward to a given time |
| `skip` | "Next alert": run fast until the next new warning or critical, then pause there |

Every new event is sent to SSE subscribers and, if configured, to a webhook.

**`server/api.py`, the FastAPI app on port 8100:**

| Endpoint | Purpose |
|---|---|
| `GET /api/status`, `POST /api/control` | Clock and controls |
| `GET /api/state` | Every machine's latest telemetry, active alerts, forecasts, plan, health score |
| `GET /api/site` | Zones, geofences, haul road, weather, workers, plans |
| `GET /api/events?since=` | Event history (backfill) |
| `GET /api/stream` | **Server-sent events**, the live event feed for the agent |
| `GET /api/timeseries/{id}` | One-minute averages plus ML expected values, for the charts |
| `GET /api/telemetry/{id}?seconds=` | Raw 1 Hz readings |
| `GET /api/incidents[/{id}]` | Black-box recordings |
| `GET /api/metrics`, `GET /api/schema` | Model report and the Event JSON Schema |
| `POST /api/agent/messages` | The agent posts its advice here, and it appears in the cab UI |
| `POST /api/agent/webhook` | Register a URL to receive events |

**`scripts/mock_agent.py`** stands in for the LangGraph agent. It reads the stream, writes a short message for each alert, and posts it back. To go live, replace its `compose()` with the real graph.

---

## 8. Part 5: the dashboard (`app/`)

Streamlit, dark theme, live-refreshing. It has eight pages:

1. **Operator cab:** machine tiles (state, fuel, coolant, health score, seatbelt, proximity), active alerts with the agent's advice, a clock and controls.
2. **Site map:** zones, geofences, the haul road, machines, workers and the danger rings.
3. **Shift plan:** the briefing, and a Gantt chart of planned vs forecast task times with live ETAs.
4. **Machine health:** each signal with its **ML normal band**. The actual reading leaves the band well before the fixed limit.
5. **Operators & coaching:** idle by cause, end-stop rate vs fleet, the shift scorecard and training modules.
6. **Event stream:** every event with filters, plus the black-box replays.
7. **Model report:** accuracy, lead time vs thresholds, false alarms and the task-time model.
8. **Agent integration:** schema, stream status and the agent's messages.

Status colours always come with an icon and a label, never colour alone.

---

## 9. Part 6: connecting to the Cocoon backend (`scripts/cocoon_bridge.py`)

The team's product (Android app, voice, LangGraph backend) has its own API contract. The bridge:
- reads EXC001's live telemetry from this project and **maps it to catalog IDs** (`EXC_DEMO_001` / `OP_DEMO_1_1`)
- converts site-local time to UTC and keeps it **always moving forward**, so the backend never ignores a sample as stale
- posts `POST /v1/sessions/{id}/telemetry` whenever engine, belt, operating state or speed changes, plus a heartbeat every 30 simulated seconds

The backend runs its own rules on these readings and the voice assistant speaks them. On the demo day that gives:
- a seatbelt warning at 06:40, with an incident draft
- idle alerts at 12:00

A "Why?" is answered from the bridged data. Limits and the proposed next step are in `docs/INTEGRATION.md` §7.

---

## 10. What you'll see on the demo day (detected, not scripted)

| Time | Alert | Level | How it was detected |
|---|---|---|---|
| 05:09 | Fuel theft | L1/security | Fuel level falling ~11 L/min with the engine off |
| 06:00 | Briefing, fuel shortfall, service due | L4 | Plan × task model × burn rate vs fuel in tank; engine hours vs the 2,000 h service |
| 06:02 | Warm-up violation | behaviour | Heavy load with cold coolant, minutes after a cold start |
| 06:40 | Seatbelt (critical) | L5 | Belt open + seated + digging, after 15 s grace |
| 07:29 | Person in swing radius (critical, black box) | L5 | Worker's position inside reach + 1 m while slewing |
| 07:45 | Truck overspeed | L1 | 26 km/h inside the 15 km/h geofence |
| 08:40 | Sensor frozen | sensor health | Fuel temp stuck at exactly 0 °C while the engine runs |
| 09:21 | Transmission pressure low | L1 | TRK02 pressure collapse |
| 09:22 | **Cooling degradation (ML)** | L3 | Coolant ~2 °C above what the model expects for this load and ambient |
| 09:37 | Waiting on trucks | L5 | Idle with no truck ready (TRK02 down), so not the operator's fault |
| 10:51 → 11:30 | Coolant rising → coolant high → critical | L2 → L1 | Trend projection, then the fixed limit |
| 11:56, 12:47 | Stability limit | L5 | Slope-compensated load moment ≥ 100% on the 14° batter |
| 12:01 | Machine unattended | L5 | Engine running, nobody in the seat for 90 s |
| 13:30 | Weather window | L4 | Forecast rain before the trench's forecast finish |
| 13:54 | Power-line clearance | L1/geo | Boom tip ~9 m inside the 7 m corridor |
| 14:23 | Fatigue risk | behaviour | Heat index + time in seat + jerk rising vs this operator's fresh baseline |
| 14:30 | Water in fuel rising (EXC002) | L2 | Separator level trend |
| 15:20 → 15:31 | Lightning warning → critical | site | Strike distance under 16 km, then under 10 km |
| 16:00 | Hot shutdown, shift scorecard | behaviour | Engine off at high temperature with no cool-down; the day's summary with training modules |

That's roughly 47 actionable alerts over the day, and each one traces back to a cause in §4.7 or to Jake's hidden habits.

---

## 11. Honesty notes (say these if asked)

- **Everything is simulated.** Machine specifications and limits are *representative* of 336- and 730-class machines, not Caterpillar data. The people, the site and the machine names are fictional.
- The ML is trained and tested **on simulated data**. The numbers show that the method works and beats fixed thresholds *in this simulator*. They don't claim real-world accuracy.
- **What carries over to real machines is the pipeline.** Detection only reads the telemetry dictionary and the plan. Map real Product Link / VisionLink channels to the same names (units and SPNs are in `sim/config.py`), retrain on real history, and nothing downstream changes.
- The Cocoon task estimator's 5-task result is a consistency check on 5 rows, not a validation.

---

## 12. File map: where to change things

| I want to… | Go to |
|---|---|
| Change the demo story (causes, times) | `sim/scenarios.py` → `DEMO_DAY["script"]` |
| Add a machine, change fuel, service, tasks | `sim/scenarios.py` → `DEMO_DAY["machines"]` |
| Change an operator's habits | `sim/operators.py` |
| Change how a machine heats, burns or moves | `sim/excavator.py::_physics`, `sim/physics.py` |
| Add a sensor channel | `Excavator._telemetry` + `sim/config.py::CHANNELS` |
| Change a limit (coolant, seatbelt grace, proximity) | `sim/config.py` → `THRESHOLDS`, `SAFETY` |
| Add a new alert | A method returning `Finding`s in `detect/*.py`, plus actions/training in `detect/playbook.py` |
| Change how alerts open and clear | `clear_after_s` / `update_token` on the Finding; `detect/manager.py` |
| Retrain after changing physics | `python -m ml.train` (then run `scripts/run_headless.py` to recheck the timeline) |
| Refit the Cocoon task estimator | `python -m ml.task_estimator` |
| Change the dashboard | `app/dashboard.py` (pages), `app/ui.py` (styles, charts) |
| Connect the real LangGraph agent | `docs/INTEGRATION.md` (SSE / webhook / reply endpoint) |
| Feed the Cocoon backend | `scripts/cocoon_bridge.py`, `docs/INTEGRATION.md` §7 |

## 13. Glossary

| Term | Meaning |
|---|---|
| Tick | One simulated second |
| Telemetry | The per-second sensor readings of one machine |
| Finding | A detector saying "this condition holds right now" (internal) |
| Event | A clean, lifecycle-managed alert sent to consumers (open / escalated / update / resolved) |
| incident_key | The stable ID of one alert across its lifecycle, e.g. `EXC001:FUEL_SHORTFALL` |
| Residual | Actual reading minus the ML-expected reading |
| σ (sigma) | The model's typical error for a signal; "3σ" means three times that |
| Load moment | How close the excavator is to tipping, as % of rated capacity |
| End-stop | Slamming a hydraulic cylinder to its limit, which is harsh and heats the oil |
| Hot shutdown | Switching off a hot engine without idling it down, which is hard on the turbo |
| SSE | Server-sent events: a one-way live feed over HTTP |
| SPN / FMI | J1939 codes for a signal and its failure mode, the standard in heavy equipment |
