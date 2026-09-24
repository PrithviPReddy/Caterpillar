"""Task-duration estimator calibrated on Cocoon_Dataset_v1 `task_history.csv` (REQ-05a, REQ-05b).

A transparent multiplicative model, fitted by ridge regression in log space:

    minutes = quantity / rate[task_type] x skill x weather x ground

`rate` is the nominal productivity (work units per minute) of an intermediate operator on a dry,
sunny day, and each other factor is a multiplier relative to that reference. Every prediction lists
the multipliers and the minutes each one adds, so the agent can say what is adding time.

Two configurations:
- full: runtime tasks with task type, quantity (in the type's unit), ground, weather and skill.
- reduced: the five provided benchmark rows have no quantity or ground condition, and those must
  not be filled in (DATA_GAPS DG-14). The planner's estimate stands in for quantity / rate, so
  minutes = estimate x skill x weather. The estimate is an input to the task, not its outcome.

Machine age is accepted but not used: every machine in the dataset has one age and its own task
types, so an age effect cannot be separated from the task type. The prediction reports this.

    python -m ml.task_estimator                      # fit, evaluate, benchmark, write models/
    python -m ml.task_estimator --dataset /path/to/Cocoon_Dataset_v1

`predict()` needs only the JSON artifact and the standard library, so the backend can load it as is.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

VERSION = "task-estimator-v1"
REF = {"operator_skill": "intermediate", "weather": "sunny", "ground_condition": "dry"}
SKILLS = ("beginner", "intermediate", "expert")
WEATHERS = ("sunny", "cloudy", "rainy", "windy")
GROUNDS = ("dry", "wet")
Z80 = 1.2816
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = Path(os.environ.get("COCOON_DATASET", Path.home() / "Downloads" / "Cocoon_Dataset_v1"))


# ------------------------------------------------------------------ prediction (stdlib only)
def _norm(v):
    return str(v).strip().lower().replace(" ", "_").replace("-", "_") if v is not None else None


def _mult(coefs, name, value):
    """Multiplier for a categorical factor; the reference level (and unknown values) are 1.0."""
    return math.exp(coefs.get(name, {}).get(value, 0.0))


def predict(model, task):
    """task: task_type, weather, operator_skill, and either work_quantity + work_unit (+ ground_condition)
    for the full configuration or estimated_duration_min for the reduced one. Returns minutes, an 80 %
    range, and the factor breakdown. Raises ValueError for inputs the model cannot handle."""
    ttype = _norm(task.get("task_type"))
    skill, weather = _norm(task.get("operator_skill")), _norm(task.get("weather"))
    if skill not in SKILLS or weather not in WEATHERS:
        raise ValueError(f"operator_skill must be one of {SKILLS} and weather one of {WEATHERS}")
    qty = task.get("work_quantity")
    if qty is not None:
        cfg = model["full"]
        spec = cfg["task_types"].get(ttype)
        if spec is None:
            raise ValueError(f"no calibration for task type {ttype!r}")
        if _norm(task.get("work_unit")) != spec["unit"]:
            raise ValueError(f"{ttype} is calibrated in {spec['unit']}, got {task.get('work_unit')!r}")
        ground = _norm(task.get("ground_condition")) or ("wet" if weather == "rainy" else "dry")
        base = float(qty) / spec["rate_per_min"]
        levels = [("operator_skill", skill), ("weather", weather), ("ground_condition", ground)]
        config, missing = "full", []
    elif task.get("estimated_duration_min") is not None:
        cfg = model["reduced"]
        base = float(task["estimated_duration_min"])
        levels = [("operator_skill", skill), ("weather", weather)]
        config, missing = "reduced", ["work_quantity", "ground_condition"]
    else:
        raise ValueError("need work_quantity (+ work_unit) or estimated_duration_min")

    minutes, factors = base, []
    for name, value in levels:
        m = _mult(cfg["coef"], name, value)
        if abs(m - 1.0) >= 0.005:
            factors.append({"factor": name, "value": value, "multiplier": round(m, 3),
                            "adds_min": round(minutes * (m - 1.0), 1)})
        minutes *= m
    sigma = cfg["sigma_log"]
    return {
        "estimator_version": model["version"],
        "configuration": config,
        "minutes": round(minutes, 1),
        "range_80_min": [round(minutes * math.exp(-Z80 * sigma), 1), round(minutes * math.exp(Z80 * sigma), 1)],
        "reference_minutes": round(base, 1),
        "reference": "intermediate operator, sunny, dry ground" if config == "full" else "the planner's estimate",
        "factors": factors,
        "missing_inputs": missing,
        "not_used": {"machine_age_years": model["age_policy"]},
    }


# ------------------------------------------------------------------ fitting
def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ridge(X, y, alpha, free=(0,)):
    import numpy as np
    X, y = np.asarray(X, float), np.asarray(y, float)
    pen = np.full(X.shape[1], alpha)
    pen[list(free)] = 0.0
    return np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)


def _design(rows, types, levels, with_types):
    """Columns: one per task type (log rate, full config) or an intercept, then one-hot non-reference levels."""
    X = []
    for r in rows:
        head = [1.0 if r["task_type"] == t else 0.0 for t in types] if with_types else [1.0]
        X.append(head + [1.0 if r[name] == lvl else 0.0 for name, lvl in levels])
    return X


def fit(rows, types, alpha=1.0):
    """Returns the full and reduced configurations fitted on `rows`."""
    import numpy as np
    full_levels = ([("operator_skill", s) for s in SKILLS if s != REF["operator_skill"]]
                   + [("weather", w) for w in WEATHERS if w != REF["weather"]]
                   + [("ground_condition", "wet")])
    red_levels = full_levels[:-1]

    # full: log(minutes / quantity) = -log(rate[type]) + factors
    y = [math.log(float(r["actual_duration_min"]) / float(r["work_quantity"])) for r in rows]
    b = _ridge(_design(rows, types, full_levels, True), y, alpha, free=range(len(types)))
    resid = np.array(y) - np.array(_design(rows, types, full_levels, True)) @ b
    full = {"task_types": {t: {"rate_per_min": round(math.exp(-b[i]), 6),
                               "unit": next(r["work_unit"] for r in rows if r["task_type"] == t)}
                           for i, t in enumerate(types)},
            "coef": {}, "sigma_log": round(float(resid.std(ddof=len(b))), 4)}
    for (name, lvl), c in zip(full_levels, b[len(types):]):
        full["coef"].setdefault(name, {})[lvl] = round(float(c), 4)

    # reduced: log(minutes / estimate) = intercept + skill + weather
    y = [math.log(float(r["actual_duration_min"]) / float(r["estimated_duration_min"])) for r in rows]
    Xr = _design(rows, types, red_levels, False)
    b = _ridge(Xr, y, alpha)
    resid = np.array(y) - np.array(Xr) @ b
    reduced = {"coef": {}, "sigma_log": round(float(resid.std(ddof=len(b))), 4)}
    # fold the intercept into each skill level so the planner's estimate is only scaled by named factors
    for (name, lvl), c in zip(red_levels, b[1:]):
        reduced["coef"].setdefault(name, {})[lvl] = round(float(c), 4)
    for s in SKILLS:
        reduced["coef"]["operator_skill"][s] = round(reduced["coef"]["operator_skill"].get(s, 0.0) + float(b[0]), 4)
    return full, reduced


def _metrics(pairs):
    err = [abs(p - a) for p, a in pairs]
    return {"n": len(pairs), "mae_min": round(sum(err) / len(err), 2),
            "mape_pct": round(100 * sum(e / a for e, (_, a) in zip(err, pairs)) / len(err), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--holdout-days", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    gen = args.dataset / "data" / "generated"
    manifest_sha = hashlib.sha256((gen / "manifest.json").read_bytes()).hexdigest()
    rows = _read(gen / "task_history.csv")
    # Idle stoppages inside a task are not predictable from task inputs; they are left to the idle rules.
    clean = [r for r in rows if float(r["idle_duration_min"]) == 0]
    types = sorted({r["task_type"] for r in clean})
    days = sorted({r["started_at"][:10] for r in clean})
    cut = days[-args.holdout_days]
    train = [r for r in clean if r["started_at"][:10] < cut]
    test = [r for r in clean if r["started_at"][:10] >= cut]

    # ---- time-based holdout: fit on the first days, score the last ones
    full, reduced = fit(train, types)
    probe = {"version": VERSION, "full": full, "reduced": reduced, "age_policy": ""}
    holdout = {
        "planner_estimate": _metrics([(float(r["estimated_duration_min"]), float(r["actual_duration_min"]))
                                      for r in test]),
        "full": _metrics([(predict(probe, r)["minutes"], float(r["actual_duration_min"])) for r in test]),
        "reduced": _metrics([(predict(probe, {**r, "work_quantity": None})["minutes"],
                              float(r["actual_duration_min"])) for r in test]),
    }
    los = [math.log(float(r["actual_duration_min"]) / predict(probe, r)["minutes"]) for r in test]
    inside = sum(abs(v) <= Z80 * full["sigma_log"] for v in los) / len(los)

    # ---- final model on every clean row
    full, reduced = fit(clean, types)
    model = {
        "version": VERSION,
        "dataset": {"name": "Cocoon_Dataset_v1", "manifest_sha256": manifest_sha, "rows_used": len(clean),
                    "rows_excluded_idle": len(rows) - len(clean), "origin": "synthetic"},
        "full": full,
        "reduced": reduced,
        "age_policy": ("not used: each dataset machine has a single age and its own task types, "
                       "so an age effect cannot be separated from task type"),
        "notes": [
            "Durations in task_history are constructed by the dataset generator, not measured.",
            "ground_condition is wet exactly when weather is rainy in the dataset, so their split is not identifiable.",
        ],
    }

    # ---- REQ-05b: the five provided rows, scored once with the frozen model (never fitted on them)
    bench_rows = _read(args.dataset / "data" / "raw" / "problem_statement_task_samples.csv")
    bench = []
    for r in bench_rows:
        task = {"task_type": r["Task Type"], "weather": r["Weather"], "operator_skill": r["Operator Skill"],
                "machine_age_years": r["Machine Age (yrs)"], "estimated_duration_min": float(r["Estimated Time (min)"])}
        p = predict(model, task)
        actual = float(r["Actual Time (min)"])
        bench.append({"task_id": r["Task ID"], "task_type": r["Task Type"], "weather": r["Weather"],
                      "operator_skill": r["Operator Skill"], "planner_min": task["estimated_duration_min"],
                      "actual_min": actual, "model_min": p["minutes"], "range_80_min": p["range_80_min"],
                      "planner_err": round(abs(task["estimated_duration_min"] - actual), 1),
                      "model_err": round(abs(p["minutes"] - actual), 1),
                      "adding_time": [f"{f['value']} {f['adds_min']:+.1f} min" for f in p["factors"]]})
    report = {
        "estimator_version": VERSION,
        "holdout": {"split": f"time-based, last {args.holdout_days} of {len(days)} days (n={len(test)})",
                    **holdout, "range_80_coverage": round(inside, 3)},
        "benchmark_five_tasks": {
            "planner_mae_min": round(sum(b["planner_err"] for b in bench) / len(bench), 2),
            "model_mae_min": round(sum(b["model_err"] for b in bench) / len(bench), 2),
            "configuration": "reduced (no quantity or ground condition in these rows)",
            "rows": bench,
            "caveats": [
                "n = 5: too few rows to claim a real-world accuracy.",
                "The model was never fitted on these rows; it was frozen before scoring them.",
                "The dataset's skill and rain effects are generator assumptions that may have been informed by "
                "the same five rows, so this is a consistency check, not independent validation.",
                "The dataset has no wind effect; T005 (windy) is not explained by the model.",
            ],
        },
    }
    args.out.mkdir(exist_ok=True)
    (args.out / "task_estimator_v1.json").write_text(json.dumps(model, indent=2))
    (args.out / "task_benchmark_v1.json").write_text(json.dumps(report, indent=2))

    print(f"Cocoon_Dataset_v1 manifest {manifest_sha[:12]}…  rows {len(clean)} (+{len(rows) - len(clean)} idle-affected excluded)")
    print(f"\nHoldout ({report['holdout']['split']}):")
    for k in ("planner_estimate", "full", "reduced"):
        print(f"  {k:17s} MAE {holdout[k]['mae_min']:5.2f} min   MAPE {holdout[k]['mape_pct']:4.1f}%")
    print(f"  80% range covers {inside:.0%} of held-out tasks")
    print("\nMultipliers (full model, vs intermediate / sunny / dry):")
    for name, lv in full["coef"].items():
        print("  " + name + ": " + ", ".join(f"{k} x{math.exp(v):.2f}" for k, v in lv.items()))
    print("\nFive provided tasks (reduced configuration, frozen model):")
    print("  task  type               weather  skill         planner actual  model  |plan err| |model err|")
    for b in bench:
        print(f"  {b['task_id']}  {b['task_type']:18s} {b['weather']:8s} {b['operator_skill']:13s}"
              f"{b['planner_min']:6.0f} {b['actual_min']:6.0f} {b['model_min']:6.1f}   {b['planner_err']:6.1f}   {b['model_err']:7.1f}")
    bf = report["benchmark_five_tasks"]
    print(f"  MAE: planner {bf['planner_mae_min']} min -> model {bf['model_mae_min']} min")
    print(f"\nwrote {args.out / 'task_estimator_v1.json'} and {args.out / 'task_benchmark_v1.json'}")


if __name__ == "__main__":
    main()
