"""Loads trained artefacts from models/. Everything degrades gracefully if absent."""

import json
from pathlib import Path

import numpy as np

from sim.physics import NOMINAL_RATE_M3H

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

DEFAULT_BASELINES = {
    "fleet": {"end_stops_per_h": 12.0, "impacts_per_h": 2.0, "jerk": 0.25, "idle_frac": 0.2,
              "burn_lph": 20.0, "micro_pauses_per_h": 2.0},
    "operators": {},
}

TASK_FEATURES = ["volume_m3", "soil_hardness", "ambient_c", "rain", "n_trucks", "haul_m",
                 "type_truck_loading", "type_trenching", "type_slope_cut", "type_side_cast",
                 "op_OP1001", "op_OP1002", "op_OP1003"]


def task_feature_row(task, operator_id, ambient_c, rain):
    row = {
        "volume_m3": float(task["volume_m3"]),
        "soil_hardness": float(task.get("soil_hardness", 3)),
        "ambient_c": float(ambient_c),
        "rain": float(bool(rain)),
        "n_trucks": float(task.get("n_trucks", 0) if task["type"] == "truck_loading" else 0),
        "haul_m": float(task.get("haul_m", 0) if task["type"] == "truck_loading" else 0),
    }
    for t in ("truck_loading", "trenching", "slope_cut", "side_cast"):
        row[f"type_{t}"] = float(task["type"] == t)
    for o in ("OP1001", "OP1002", "OP1003"):
        row[f"op_{o}"] = float(operator_id == o)
    return [row[k] for k in TASK_FEATURES]


class ModelStore:
    def __init__(self, model_dir=MODEL_DIR):
        self.dir = Path(model_dir)
        self.health = None
        self.iforest = None
        self.task_model = None
        self.baselines = DEFAULT_BASELINES
        self.metrics = {}
        self.load()

    def load(self):
        try:
            import joblib
            if (self.dir / "health_models.joblib").exists():
                self.health = joblib.load(self.dir / "health_models.joblib")
            if (self.dir / "health_iforest.joblib").exists():
                self.iforest = joblib.load(self.dir / "health_iforest.joblib")
            if (self.dir / "task_time_model.joblib").exists():
                self.task_model = joblib.load(self.dir / "task_time_model.joblib")
        except Exception as exc:  # pragma: no cover
            print(f"[models] could not load models: {exc}")
        if (self.dir / "baselines.json").exists():
            self.baselines = json.loads((self.dir / "baselines.json").read_text())
        if (self.dir / "metrics.json").exists():
            self.metrics = json.loads((self.dir / "metrics.json").read_text())

    @property
    def has_health(self):
        return self.health is not None

    def expected(self, target, feats):
        m = self.health[target]
        x = np.array([[feats[f] for f in m["features"]]])
        return float(m["model"].predict(x)[0]), float(m["sigma"])

    def anomaly_score(self, zvec):
        """0 (normal) .. 1 (very anomalous) from the isolation forest on z-scores."""
        if self.iforest is None:
            return None
        s = -float(self.iforest["model"].score_samples(np.array([zvec]))[0])
        lo, hi = self.iforest["score_lo"], self.iforest["score_hi"]
        return float(np.clip((s - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    def task_minutes(self, task, operator_id, ambient_c, rain):
        if self.task_model is not None:
            x = np.array([task_feature_row(task, operator_id, ambient_c, rain)])
            return float(np.expm1(self.task_model.predict(x)[0]))  # trained on log1p(minutes)
        return 60.0 * task["volume_m3"] / NOMINAL_RATE_M3H[task["type"]]
