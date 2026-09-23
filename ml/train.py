"""Train all models from simulated history and evaluate them on held-out days.

    python -m ml.train                # full run (~2-4 min on a laptop)
    python -m ml.train --quick        # smaller, for iteration

Outputs in models/: health_models.joblib, health_iforest.joblib, task_time_model.joblib,
baselines.json, metrics.json, and data/history_minutes.csv for inspection.
"""

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest

from detect.features import HEALTH_TARGETS
from detect.models import MODEL_DIR, TASK_FEATURES, ModelStore, task_feature_row
from ml.common import detection_day, history_day
from sim.operators import OPERATORS
from sim.physics import NOMINAL_RATE_M3H, simulate_task_duration_min
from sim.scenarios import random_day

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FAULTS = {
    # fault: (L3 type, L2 type or None, L1 type, ramp range min, severity)
    "radiator_clog": ("COOLING_DEGRADATION_DETECTED", "COOLANT_TEMP_RISING", "COOLANT_TEMP_HIGH", (180, 300), 1.0),
    "oil_pump_wear": ("OIL_PRESSURE_DEVIATION", None, "OIL_PRESSURE_LOW", (180, 300), 1.0),
    "injector_wear": ("FUEL_EFFICIENCY_DEGRADATION", None, "EGT_HIGH", (180, 300), 1.0),
}


# ----------------------------------------------------------------------------- data
def _hist(seed):
    return history_day(random_day(seed))


def generate_history(n_days, workers):
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_hist, range(1000, 1000 + n_days)))
    rows = [r for res in results for r in res[0]]
    stats = [s for res in results for s in res[1]]
    print(f"[history] {n_days} days -> {len(rows)} minute rows in {time.time() - t0:.0f}s")
    return pd.DataFrame(rows), pd.DataFrame(stats)


# ----------------------------------------------------------------------------- health
def train_health(df):
    df = df[(df["run_min"] >= 10) & (df["coolant_now"] > 78.0)].dropna(subset=list(HEALTH_TARGETS))
    days = sorted(df["day"].unique())
    rng = np.random.default_rng(0)
    val_days = set(rng.choice(days, size=max(2, len(days) // 5), replace=False))
    tr, va = df[~df["day"].isin(val_days)], df[df["day"].isin(val_days)]
    models, metrics, zcols = {}, {}, []
    for target, (feats, direction, label, unit) in HEALTH_TARGETS.items():
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0)
        m.fit(tr[feats].values, tr[target].values)
        res = va[target].values - m.predict(va[feats].values)
        sigma = float(max(np.std(res), 1e-3))
        naive = va[target].values - tr[target].mean()
        models[target] = {"model": m, "features": feats, "sigma": sigma, "unit": unit, "label": label}
        metrics[target] = {"label": label, "unit": unit, "rmse": round(float(np.sqrt(np.mean(res ** 2))), 3),
                           "mae": round(float(np.mean(np.abs(res))), 3),
                           "naive_rmse": round(float(np.sqrt(np.mean(naive ** 2))), 3),
                           "r2": round(float(1 - np.mean(res ** 2) / np.var(va[target].values)), 3),
                           "sigma": round(sigma, 3), "n_train": int(len(tr)), "n_val": int(len(va))}
        zcols.append(res / sigma)
        print(f"[health] {target:18s} rmse={metrics[target]['rmse']:.3f} {unit} (naive "
              f"{metrics[target]['naive_rmse']:.2f}) r2={metrics[target]['r2']:.3f}")
    Z = np.column_stack(zcols)
    iso = IsolationForest(n_estimators=200, random_state=0).fit(Z)
    s = -iso.score_samples(Z)
    iforest = {"model": iso, "score_lo": float(np.percentile(s, 50)), "score_hi": float(np.percentile(s, 99.9))}
    return models, iforest, metrics


# ----------------------------------------------------------------------------- tasks
def _task_sample(seed):
    rng = np.random.default_rng(seed)
    ttype = str(rng.choice(["truck_loading", "trenching", "slope_cut", "side_cast"], p=[0.4, 0.3, 0.15, 0.15]))
    op_id = str(rng.choice(["OP1001", "OP1002", "OP1003"]))
    vol = float(rng.uniform(80, 1500) if ttype == "truck_loading" else rng.uniform(40, 500))
    task = {"type": ttype, "volume_m3": round(vol), "soil_hardness": int(rng.integers(1, 6)),
            "n_trucks": int(rng.integers(1, 5)) if ttype == "truck_loading" else 0,
            "haul_m": float(rng.uniform(400, 1600)) if ttype == "truck_loading" else 0.0}
    ambient = float(rng.uniform(5, 40))
    rain = bool(rng.random() < 0.15)
    env = {"ambient_c": ambient, "rh": float(rng.uniform(20, 90)), "rain": rain}
    y = simulate_task_duration_min(task, OPERATORS[op_id], env, rng)
    return task_feature_row(task, op_id, ambient, rain), y, 60.0 * task["volume_m3"] / NOMINAL_RATE_M3H[ttype]


def train_task_model(n, workers):
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_task_sample, range(n), chunksize=64))
    X = np.array([r[0] for r in res])
    y = np.array([r[1] for r in res])
    naive = np.array([r[2] for r in res])
    idx = np.random.default_rng(1).permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, random_state=0)
    m.fit(X[tr], np.log1p(y[tr]))
    pred = np.expm1(m.predict(X[te]))
    mae = float(np.mean(np.abs(pred - y[te])))
    mape = float(np.mean(np.abs(pred - y[te]) / y[te])) * 100
    nmae = float(np.mean(np.abs(naive[te] - y[te])))
    nmape = float(np.mean(np.abs(naive[te] - y[te]) / y[te])) * 100
    within = float(np.mean(np.abs(pred - y[te]) / y[te] <= 0.15)) * 100
    m.fit(X, np.log1p(y))
    print(f"[task] n={n} MAE {mae:.1f} min ({mape:.1f}%) vs planner rule-of-thumb {nmae:.1f} min ({nmape:.1f}%) "
          f"in {time.time() - t0:.0f}s")
    sample = [{"pred": round(float(p)), "actual": round(float(a)), "naive": round(float(b))}
              for p, a, b in list(zip(pred, y[te], naive[te]))[:300]]
    return m, {"n": n, "features": TASK_FEATURES, "mae_min": round(mae, 1), "mape_pct": round(mape, 1),
               "within_15pct": round(within, 1), "baseline_mae_min": round(nmae, 1),
               "baseline_mape_pct": round(nmape, 1), "holdout_sample": sample}


# ----------------------------------------------------------------------------- baselines
def build_baselines(stats):
    def med(s):
        s = [x for x in s if x is not None and not pd.isna(x)]
        return round(float(statistics.median(s)), 4) if s else None
    fleet = {k: med(stats[k]) for k in ("end_stops_per_h", "impacts_per_h", "micro_pauses_per_h", "jerk",
                                          "jerk_fresh", "idle_frac", "burn_lph")}
    ops = {}
    for op, g in stats.groupby("operator_id"):
        ops[op] = {k: med(g[k]) for k in ("end_stops_per_h", "impacts_per_h", "micro_pauses_per_h", "jerk",
                                          "jerk_fresh", "idle_frac", "burn_lph")} | {"days": int(len(g))}
    return {"fleet": fleet, "operators": ops}


# ----------------------------------------------------------------------------- evaluation
def _eval_fault(args):
    seed, fault, start_min, ramp = args
    start = f"{6 + start_min // 60:02d}:{start_min % 60:02d}"
    sc = random_day(seed, fault=(fault, start, ramp, {}))
    events, truth, eng_h = detection_day(sc, ModelStore())
    return seed, fault, start, ramp, events


def _eval_normal(seed):
    events, truth, eng_h = detection_day(random_day(seed), ModelStore())
    return seed, events, eng_h


def evaluate(n_fault_days, n_normal_days, workers):
    t0 = time.time()
    rng = np.random.default_rng(7)
    jobs = []
    for i, fault in enumerate(FAULTS):
        for k in range(n_fault_days):
            jobs.append((5000 + 100 * i + k, fault, int(rng.integers(60, 210)), int(rng.integers(*FAULTS[fault][3]))))
    with ProcessPoolExecutor(workers) as ex:
        fres = list(ex.map(_eval_fault, jobs))
        nres = list(ex.map(_eval_normal, range(9000, 9000 + n_normal_days)))
    per_fault = {}
    runs = []
    for seed, fault, start, ramp, events in fres:
        l3, l2, l1 = FAULTS[fault][:3]
        day = events[0]["ts"][:10] if events else None

        def first(etype):
            for e in events:
                if e["machine_id"] == "EXC001" and e["type"] == etype and e["status"] in ("open", "escalated"):
                    return e["ts"]
            return None
        t3, t2, t1 = first(l3), (first(l2) if l2 else None), first(l1)
        fstart = pd.Timestamp(f"{day}T{start}") if day else None

        def mins(ts):
            return None if ts is None else round((pd.Timestamp(ts) - fstart).total_seconds() / 60, 1)
        runs.append({"seed": seed, "fault": fault, "fault_start": start, "ramp_min": ramp,
                     "ml_detect_min": mins(t3), "trend_detect_min": mins(t2), "threshold_detect_min": mins(t1)})
    for fault in FAULTS:
        rs = [r for r in runs if r["fault"] == fault]
        ml = [r["ml_detect_min"] for r in rs if r["ml_detect_min"] is not None]
        th = [r["threshold_detect_min"] for r in rs if r["threshold_detect_min"] is not None]
        leads = [r["threshold_detect_min"] - r["ml_detect_min"] for r in rs
                 if r["ml_detect_min"] is not None and r["threshold_detect_min"] is not None]
        per_fault[fault] = {
            "runs": len(rs),
            "ml_detection_rate_pct": round(100 * len(ml) / len(rs), 1),
            "threshold_detection_rate_pct": round(100 * len(th) / len(rs), 1),
            "ml_median_detect_min": round(statistics.median(ml), 1) if ml else None,
            "threshold_median_detect_min": round(statistics.median(th), 1) if th else None,
            "median_lead_time_min": round(statistics.median(leads), 1) if leads else None,
        }
        print(f"[eval] {fault:14s} ML caught {per_fault[fault]['ml_detection_rate_pct']}% "
              f"(median {per_fault[fault]['ml_median_detect_min']} min after onset) | thresholds caught "
              f"{per_fault[fault]['threshold_detection_rate_pct']}% (median {per_fault[fault]['threshold_median_detect_min']} min)"
              f" | lead {per_fault[fault]['median_lead_time_min']} min")
    l3_types = {v[0] for v in FAULTS.values()}
    fa = sum(1 for _, events, _ in nres for e in events if e["type"] in l3_types and e["status"] == "open")
    eng_h = sum(h for _, _, h in nres)
    fa_rate = round(100 * fa / max(eng_h, 1), 2)
    print(f"[eval] false ML alarms on {n_normal_days} normal days: {fa} over {eng_h:.0f} engine-h "
          f"({fa_rate}/100 h) in {time.time() - t0:.0f}s")
    return {"faults": per_fault, "runs": runs,
            "false_alarms": {"normal_days": n_normal_days, "engine_hours": round(eng_h, 1), "ml_alarms": fa,
                             "per_100_engine_h": fa_rate}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    n_days = args.days or (10 if args.quick else 30)
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    df, stats = generate_history(n_days, args.workers)
    df.to_csv(DATA_DIR / "history_minutes.csv", index=False)
    stats.to_csv(DATA_DIR / "history_operator_days.csv", index=False)
    models, iforest, hmetrics = train_health(df)
    joblib.dump(models, MODEL_DIR / "health_models.joblib")
    joblib.dump(iforest, MODEL_DIR / "health_iforest.joblib")
    baselines = build_baselines(stats)
    (MODEL_DIR / "baselines.json").write_text(json.dumps(baselines, indent=2))
    print(f"[baselines] {json.dumps(baselines['operators'])}")
    tmodel, tmetrics = train_task_model(1500 if args.quick else 5000, args.workers)
    joblib.dump(tmodel, MODEL_DIR / "task_time_model.joblib")
    metrics = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "history_days": n_days,
               "health_models": hmetrics, "task_model": tmetrics, "baselines": baselines}
    if not args.skip_eval:
        metrics["evaluation"] = evaluate(3 if args.quick else 8, 3 if args.quick else 8, args.workers)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[done] models written to {MODEL_DIR}")


if __name__ == "__main__":
    main()
