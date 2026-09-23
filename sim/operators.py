"""Operator profiles. These traits are hidden from the detection layer; detectors
only see their effects in the telemetry (cycle times, jerk, end-stop hits...)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Operator:
    id: str
    name: str
    years: float
    skill: float  # 0..1, faster cycles and better bucket fill
    aggressiveness: float  # 0..1, end-stop hits, impacts, jerky inputs
    seatbelt_compliance: float  # prob. of buckling up on entering the seat
    warmup_compliance: float  # prob. of a proper warm-up after a cold start
    cooldown_compliance: float  # prob. of idling down before shutdown
    engine_off_on_break: float  # prob. of shutting the engine off for breaks
    fatigue_sensitivity: float  # multiplier on fatigue build-up
    role: str = "excavator"


OPERATORS = {
    "OP1001": Operator("OP1001", "Maria G.", 18, 0.90, 0.15, 0.99, 0.95, 0.95, 0.95, 0.8),
    "OP1002": Operator("OP1002", "Jake T.", 22, 0.65, 0.80, 0.85, 0.10, 0.10, 0.40, 1.2),
    "OP1003": Operator("OP1003", "Leo P.", 32, 0.40, 0.35, 0.90, 0.60, 0.60, 0.30, 1.1),
    "OP2001": Operator("OP2001", "Sam R.", 29, 0.80, 0.30, 0.97, 0.9, 0.9, 0.9, 0.9, "truck"),
    "OP2002": Operator("OP2002", "Ana K.", 45, 0.75, 0.25, 0.98, 0.9, 0.9, 0.9, 0.9, "truck"),
    "OP2003": Operator("OP2003", "Tom W.", 33, 0.70, 0.40, 0.95, 0.9, 0.9, 0.9, 1.0, "truck"),
}
