# Operator Copilot: Smart Operator Assistant

A physics-based simulator of a construction site (two excavators, three haul trucks, ground crew, weather) streams about 45 sensor channels per machine at 1 Hz. A detection layer, made of rules plus ML models trained on simulated history, turns that stream into structured **events**. A LangGraph agent turns those events into advice for the operator. A Streamlit cab display shows all of it live.

Nothing in the demo is a scripted alert. The scenario only injects *causes*: a radiator slowly clogging, a surveyor walking behind the counterweight, a truck's transmission failing, an operator who skips the warm-up. Every alert is **detected from telemetry alone**, and the detectors never see the hidden state.

## Quick start

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ml.train        # ~3 min: generates 30 simulated days, trains, evaluates
./run_demo.sh                       # API :8100, mock agent, UI http://localhost:8501
```

`./run_demo.sh --no-agent` skips the mock agent. Use it when the real LangGraph agent is connected (see [docs/INTEGRATION.md](docs/INTEGRATION.md)).

To run a full day without the UI and print the alert timeline (about 15 s): `.venv/bin/python scripts/run_headless.py`.

## Five levels of alert intelligence

| Level | What it is | Demo example |
|---|---|---|
| **L1 Threshold** | A fixed limit | Coolant > 100 °C; transmission pressure < 800 kPa |
| **L2 Trend** | Rate of change projected to a limit | "Coolant rising 0.21 °C/min, hits 105 °C at ≈11:33" |
| **L3 ML pattern** | Regression models learn what each signal *should* read for the current load, rpm, ambient and warm-up; persistent residuals flag degradation and suggest a cause | Radiator fouling flagged at 09:22, about 2 h before the fixed alarm |
| **L4 Predictive** | Plan + task-time model + burn rates + forecast | "143 L lasts 5.2 h but 9.1 h of work remain: refuel at the 10:00 break" |
| **L5 Context** | The same reading means different things in different situations | Seatbelt unbuckled while digging is critical, while parked it's info; idle caused by a truck breakdown is not the operator's fault |

## Features, mapped to the problem statement

- **Daily task dashboard:** a pre-shift briefing, a Gantt of plan vs ML forecast, a live ETA per task, and task-delay alerts that name the cause.
- **Safety:**
  - seatbelt use in context
  - proximity (dynamic swing-radius danger zone plus UWB-tagged crew)
  - tip-over margin (slope-compensated load moment)
  - overhead power line clearance and no-dig geofences
  - pedestrian-zone speeding
  - lightning and wind
  - heat stress and fatigue risk
  - a 90 s **black-box recording** for every critical safety event, logged as a near miss
- **Unusual behaviour:**
  - excessive idling, with the cause attributed (truck wait / operator / unattended)
  - harsh operation (cylinder end-stops, bucket impacts)
  - warm-up and hot-shutdown abuse
  - fuel theft while parked
  - engine starts outside the shift
  - frozen or implausible sensors
- **Task time estimation:** `ml/task_estimator.py`, calibrated on Cocoon_Dataset_v1 (holdout MAE 2.3 min vs 6.8 for the planner; 5 provided tasks: 5.3 vs 7.6 min), with a factor breakdown per task. The simulator's own plan uses an ML model trained on 5,000 simulated tasks (volume, soil, task type, trucks, haul distance, weather, operator). It has 7% error, against 30% for the planner's rule of thumb.
- **Operator training hub link:** every behaviour maps to a training module (TM-01…TM-11). An end-of-shift scorecard recommends modules.
- **Predictive maintenance:**
  - ML early warnings for cooling, oil supply and combustion efficiency
  - an air-filter days-to-limit forecast from 14-day history
  - service-due-during-shift checks
  - water-in-fuel trend

## Demo script (about 7 minutes)

Keep the UI on **Operator cab** and use **Next alert** (it fast-forwards and pauses on each new warning or critical).

1. **05:09 Fuel theft.** "The machine is parked and off, yet fuel is dropping 11 L/min. That's a siphon."
2. **06:00 Pre-shift briefing.** Open **Shift plan**. The copilot knows the tank is short for today's plan (linked to the theft), tells the operator which break to refuel at, and flags that the 2,000 h service falls due mid-shift.
3. **06:02 Warm-up violation, 06:40 seatbelt.** Context matters: an unbuckled seatbelt is critical *because the machine is digging*.
4. **07:29 Surveyor inside the swing radius.** Critical, with a black-box recording captured. Show it in **Event stream**.
5. **09:22 ML early warning, cooling degradation.** Open **Machine health**: measured coolant leaves the ML normal band while the fixed alarm is still about 2 h away. Then open **Model report**: across held-out fault days, the ML catches all three fault types, catches injector wear that the thresholds never catch, and raises zero false alarms.
6. **09:37 Waiting on trucks.** Idle is attributed to TRK02's breakdown, not the operator. The fleet problem is surfaced to dispatch.
7. **11:56 Tipping margin, 13:54 power line.** Slope-compensated stability, and boom height inside the 11 kV corridor.
8. **13:30 Weather window, 14:23 fatigue, 15:31 lightning.**
9. **16:01 Shift scorecard.** Open **Operators & coaching** for idle cost, end-stop rate vs fleet, and recommended training modules.

## Architecture

```
sim/        world.py (clock, script)  excavator.py / truck.py (state machines + first-order physics)
            site.py (zones, geofences, weather, workers)  operators.py (hidden traits)  scenarios.py
detect/     pipeline.py → view.py (per-machine memory) → detectors:
            rules.py (L1/L2, sensor health, security, site)  safety.py  health.py (ML)  planning.py  behavior.py
            manager.py (open / escalate / update / resolve)  schema.py (Event v1.0)  incidents.py (black box)
ml/         train.py: simulated history → health models, isolation forest, task-time model, baselines, evaluation
server/     FastAPI: live replay (play / speed / jump / next alert), SSE + webhook out, agent messages in
app/        Streamlit cab display (8 views)
docs/       INTEGRATION.md for the LangGraph side, sample_events.jsonl (a full demo day of events)
```

**Why simulate instead of using the sample CSVs?** The provided datasets are sparse, randomly ordered readings (Data.csv) or independent noise with labels that look random (the IIoT set). They're useful for plausible value ranges, but you can't learn temporal behaviour from them. The simulator encodes causal physics (load drives heat, clogging reduces cooling, fatigue makes control inputs jerkier). The **hidden ground truth** it records is what lets us *measure* the ML (detection rate, lead time, false alarms), instead of just claiming it works.

**Bringing in real data later:** the detection layer only consumes the telemetry dict (`sim/excavator.py::_telemetry`) and the plan. Map a real machine's channels to the same names (units and J1939 SPNs are in `sim/config.py::CHANNELS`), retrain with `ml/train.py` on real history, and nothing downstream changes.

## Model results (held-out simulated days)

`models/metrics.json` is regenerated by `ml.train`. The latest run:

| | ML (L3) | Fixed thresholds (L1) |
|---|---|---|
| Radiator clogging | 100% caught, median 2.1 h after onset | 75% caught, median 4.8 h |
| Oil pump wear | 100%, 34 min | 100%, 2.0 h |
| Injector wear | 100%, 1.7 h | never fires |
| False alarms on normal days | 0 per 152 engine-hours | |

The task-duration model has a mean absolute error of 7.4%, against 30.2% for the rule of thumb.

## Extending it

- **New sensor channel:** compute it in `Excavator._physics` / `_telemetry`, then register units and SPN in `CHANNELS`.
- **New alert:** add a method returning `Finding`s to any detector (or a new detector in `pipeline.tick_detectors`), plus default actions and a training module in `detect/playbook.py`.
- **New scenario:** copy `DEMO_DAY` in `sim/scenarios.py`. Script causes (`inject_fault`, `override`, `seatbelt_off`, `truck_breakdown`, `fuel_theft`, …), not alerts.

People, site and machine names are fictional. Specifications and limits are representative values for 336- and 730-class machines, not manufacturer data.
