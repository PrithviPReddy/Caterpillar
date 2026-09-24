"""Event contract between the detection layer and the LangGraph agent (schema v1.0).

Every message on the stream is one `Event`. An alert has a lifecycle keyed by
`incident_key`: status `open` -> (`escalated` | `update`)* -> `resolved`.
One-shot observations (e.g. a hot shutdown) are emitted once with status `open`.
"""

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

Severity = Literal["info", "warning", "critical"]
Status = Literal["open", "escalated", "update", "resolved"]
Category = Literal["safety", "machine_health", "fuel_energy", "productivity", "operator_behavior",
                   "security", "planning", "sensor_health", "site"]
Level = Literal["L1_threshold", "L2_trend", "L3_ml_pattern", "L4_predictive", "L5_contextual"]


class SignalEvidence(BaseModel):
    """One piece of evidence. Only `value` is always present."""
    value: float | int | bool | str | None
    unit: str | None = None
    threshold: float | None = None
    expected: float | None = Field(None, description="ML-expected value for current conditions")
    deviation_sigma: float | None = Field(None, description="How unusual vs. the ML model (z-score)")
    trend_per_min: float | None = None
    baseline: float | None = Field(None, description="Operator/fleet baseline for comparison")
    spn: int | None = Field(None, description="J1939-style suspect parameter number")
    note: str | None = None


class Prediction(BaseModel):
    kind: str = Field(..., description="time_to_threshold | fuel_runout | def_runout | service_due | "
                                       "task_finish | days_to_limit | rain_start")
    eta: str | None = Field(None, description="Predicted time of impact (site local ISO time)")
    minutes_to_impact: float | None = None
    value: float | None = None
    unit: str | None = None
    confidence: float | None = None
    detail: dict = Field(default_factory=dict)


class Location(BaseModel):
    x: float
    y: float
    zone: str | None = None
    zone_name: str | None = None


class TaskContext(BaseModel):
    id: str
    name: str
    type: str
    done_m3: float
    volume_m3: float
    progress_pct: float
    predicted_finish: str | None = None
    planned_finish: str | None = None


class ShiftContext(BaseModel):
    start: str
    end: str
    elapsed_h: float
    next_break: str | None = None


class EventContext(BaseModel):
    machine_state: str | None = None
    engine_on: bool | None = None
    location: Location | None = None
    task: TaskContext | None = None
    shift: ShiftContext | None = None
    weather: dict = Field(default_factory=dict)


class Event(BaseModel):
    schema_version: str = SCHEMA_VERSION
    event_id: str
    seq: int
    incident_key: str = Field(..., description="Stable id for this alert across its lifecycle")
    status: Status
    ts: str = Field(..., description="Site local time, ISO 8601")
    machine_id: str = Field(..., description="Machine id, or SITE for site-wide events")
    machine_type: str
    operator_id: str | None = None
    operator_name: str | None = None
    category: Category
    type: str = Field(..., description="Machine-readable event type, e.g. COOLANT_TEMP_HIGH")
    intelligence_level: Level
    severity: Severity
    title: str
    summary: str = Field(..., description="Deterministic one-paragraph description with the key numbers")
    evidence: dict[str, SignalEvidence] = Field(default_factory=dict)
    prediction: Prediction | None = None
    context: EventContext = Field(default_factory=EventContext)
    recommended_actions: list[str] = Field(default_factory=list,
                                           description="Baseline playbook actions; the agent may refine")
    training_module: dict | None = None
    incident_id: str | None = Field(None, description="Black-box recording id (safety incidents)")
    dtc: dict | None = Field(None, description="Diagnostic trouble code {spn, fmi}")
    related_incident_keys: list[str] = Field(default_factory=list)
    one_shot: bool = Field(False, description="A point-in-time notice (briefing, hot shutdown, scorecard): it is "
                                              "emitted once as `open` and never resolved")


class AgentMessage(BaseModel):
    """What the LangGraph agent posts back for display in the operator cab."""
    event_id: str | None = None
    incident_key: str | None = None
    machine_id: str
    severity: Severity = "info"
    headline: str
    message: str
    actions: list[str] = Field(default_factory=list)
    requires_ack: bool = False
    speak: bool = Field(False, description="Hint for text-to-speech in the cab")
