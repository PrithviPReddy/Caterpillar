"""D1: detections from the trusted external detector, and the optional seat_occupied reading."""

from __future__ import annotations

from fastapi.testclient import TestClient

from cocoon_agent.api.app import create_app

from .conftest import AUTH, make_settings, new_session, turn
from .test_auth import bearer, issue

T0 = "2026-09-24T02:00:00Z"


def sample(c: TestClient, sid: str, event_id: str, at: str, **readings) -> dict:
    body = {"event_id": event_id, "observed_at": at, "simulated": True,
            "readings": {"engine_on": True, "seatbelt_fastened": True, "idle_seconds": 0,
                         "operating_state": "working", "speed_kph": 0.0, **readings}}
    r = c.post(f"/v1/sessions/{sid}/telemetry", headers=AUTH, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def detection(detection_id: str, status: str = "open", severity: str = "critical", *,
              key: str = "EXC_DEMO_001:PROXIMITY:W02", dtype: str = "PROXIMITY_DANGER_ZONE",
              category: str = "safety", at: str = "2026-09-24T02:00:30Z", **extra) -> dict:
    return {
        "detection_id": detection_id, "observed_at": at, "simulated": True, "episode_key": key, "status": status,
        "detection_type": dtype, "category": category, "intelligence_level": "L5_contextual", "severity": severity,
        "title": "Person inside the danger zone",
        "summary": "Surveyor Chris D. (W02) is 3.2 m from EXC_DEMO_001 while it is swinging; danger radius 7.5 m.",
        "recommended_actions": ["Stop all motion now", "Confirm the person is clear before resuming"],
        "evidence": {"distance_m": {"value": 3.2, "unit": "m", "threshold": 7.5}, "worker_role": {"value": "Surveyor"}},
        "training_module": {"module_id": "TM-08", "title": "Swing radius and ground crew awareness"},
        "provenance": {"generator": "cater.detect.v1", "source_event_id": "EVT-000041"},
        **extra,
    }


def post(c: TestClient, sid: str, body: dict, expect: int = 200) -> dict:
    r = c.post(f"/v1/sessions/{sid}/detections", headers=AUTH, json=body)
    assert r.status_code == expect, r.text
    return r.json()


def test_open_announces_once_drafts_critical_safety_and_resolve_clears(client):
    sid = new_session(client)
    sample(client, sid, "s1", T0)
    first = post(client, sid, detection("d1"))
    assert first["outcome"] == "alert_opened" and len(first["alerts_opened"]) == 1
    assert len(first["drafts_created"]) == 1 and len(first["announcements_created"]) == 1
    alert = first["active_alerts"][0]
    assert alert["source"] == "detector" and alert["alert_type"] == "proximity_danger_zone"
    assert alert["severity"] == "critical" and alert["category"] == "safety"
    assert alert["intelligence_level"] == "L5_contextual" and alert["source_status"] == "demo_assumption"
    assert alert["evidence"]["detection"]["signals"]["distance_m"]["threshold"] == 7.5
    assert alert["evidence"]["readings"]["operating_state"] == "working"  # machine state at that moment
    assert "simulated detection" in alert["explanation"]

    # a retry is a duplicate; the same key opened again does not announce twice
    assert post(client, sid, detection("d1"))["duplicate"] is True
    again = post(client, sid, detection("d2"))
    assert again["outcome"] == "recorded" and again["ignored_reason"] == "already_active"
    assert again["announcements_created"] == []

    page = client.get(f"/v1/sessions/{sid}/events?after=0", headers=AUTH).json()["events"]
    assert [(e["type"], e["priority"]) for e in page] == [("alert_started", "critical")]
    assert page[0]["speech"] == "Person inside the danger zone. Stop all motion now."

    done = post(client, sid, detection("d3", "resolved", "info"))
    assert done["outcome"] == "alert_cleared" and done["active_alerts"] == []
    page = client.get(f"/v1/sessions/{sid}/events?after=1", headers=AUTH).json()["events"]
    assert [(e["type"], e["speech"]) for e in page] == [("alert_cleared", "Cleared: Person inside the danger zone.")]
    assert post(client, sid, detection("d4", "resolved", "info"))["ignored_reason"] == "no_active_episode"


def test_escalation_opens_a_new_critical_episode_and_info_update_never_speak(client):
    sid = new_session(client)
    sample(client, sid, "s1", T0)
    key, dtype = "EXC_DEMO_001:COOLANT_TEMP_HIGH", "COOLANT_TEMP_HIGH"
    common = dict(key=key, dtype=dtype, category="machine_health")
    warn = post(client, sid, detection("c1", "open", "warning", **common))
    assert warn["outcome"] == "alert_opened" and warn["drafts_created"] == []  # not safety: no draft
    assert post(client, sid, detection("c2", "update", "warning", **common))["ignored_reason"] == "update_only"
    crit = post(client, sid, detection("c3", "escalated", "critical", **common))
    assert crit["outcome"] == "alert_escalated"
    assert crit["alerts_cleared"] == warn["alerts_opened"] and crit["alerts_opened"] != warn["alerts_opened"]
    assert [a["severity"] for a in crit["active_alerts"]] == ["critical"]
    assert post(client, sid, detection("c4", "escalated", "warning", **common))["ignored_reason"] == "not_higher"
    info = post(client, sid, detection("i1", "open", "info", key="EXC_DEMO_001:SHIFT_BRIEFING",
                                       dtype="SHIFT_BRIEFING", category="planning"))
    assert info["ignored_reason"] == "below_announce_threshold" and info["alert_id"] is None
    kinds = [(e["type"], e["priority"]) for e in
             client.get(f"/v1/sessions/{sid}/events?after=0", headers=AUTH).json()["events"]]
    assert kinds == [("alert_started", "high"), ("alert_started", "critical")]  # clears of non-safety are silent


def test_conflict_missing_machine_state_and_operator_token_are_refused(client_factory, data_dir, tmp_path):
    with client_factory() as c:
        sid = new_session(c)
        err = post(c, sid, detection("d1"), expect=422)["error"]
        assert err["code"] == "machine_state_unavailable"
        sample(c, sid, "s1", T0)
        post(c, sid, detection("d1"))
        assert post(c, sid, detection("d1", severity="warning"), expect=409)["error"]["code"] == "idempotency_conflict"
        token, _ = issue(make_settings(data_dir), tmp_path, "operator", "--operator-id", "OP_TEST_1")
        r = c.post(f"/v1/sessions/{sid}/detections", headers=bearer(token), json=detection("d9"))
        assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"


def test_why_explains_the_detector_alert_from_its_saved_evidence(client):
    sid = new_session(client)
    sample(client, sid, "s1", T0)
    post(client, sid, detection("m1", "open", "warning", key="EXC_DEMO_001:COOLING_DEGRADATION_DETECTED",
                                dtype="COOLING_DEGRADATION_DETECTED", category="machine_health",
                                intelligence_level="L3_ml_pattern", title="Cooling system losing capacity",
                                summary="Coolant 94.1 °C is 2.3 °C above the 91.8 °C the model expects for this load.",
                                recommended_actions=["Ease off heavy digging", "Blow out the radiator at the break"]))
    r = turn(client, sid, "why-1", "Why did you warn me?")
    assert r.status_code == 200, r.text
    speech = r.json()["speech"]
    assert "machine-health model" in speech and "2.3 °C above" in speech and "Ease off heavy digging" in speech


def test_empty_seat_is_not_an_unbelted_operator_but_still_idles(client):
    sid = new_session(client)
    res = sample(client, sid, "u1", T0, seatbelt_fastened=False, seat_occupied=False, operating_state="idle")
    assert res["alerts_opened"] == []  # nobody in the seat: no seatbelt warning
    res = sample(client, sid, "u2", "2026-09-24T02:06:00Z", seatbelt_fastened=False, seat_occupied=False,
                 operating_state="idle")
    assert [a["alert_type"] for a in res["active_alerts"]] == ["prolonged_idle"]  # idling unattended still counts
    res = sample(client, sid, "u3", "2026-09-24T02:07:00Z", seatbelt_fastened=False, seat_occupied=True,
                 operating_state="idle")
    assert "seatbelt_unfastened" in {a["alert_type"] for a in res["active_alerts"]}  # someone sat down unbelted


def test_policy_alerts_keep_their_shape_after_v8(client_factory, data_dir):
    with client_factory() as c:
        sid = new_session(c)
        res = sample(c, sid, "b1", T0, seatbelt_fastened=False)  # seat_occupied omitted: behaviour as before
        alert = res["active_alerts"][0]
        assert alert["alert_type"] == "seatbelt_unfastened" and alert["source"] == "policy_rule"
        assert alert["category"] is None and alert["evidence"]["detection"] is None
    with TestClient(create_app(make_settings(data_dir))) as c:  # restart reads the same episode
        assert c.get(f"/v1/sessions/{sid}/state", headers=AUTH).json()["active_alerts"][0]["source"] == "policy_rule"


def test_one_shot_notice_is_announced_and_explainable_but_never_active(client):
    sid = new_session(client)
    sample(client, sid, "s1", T0)
    res = post(client, sid, detection("h1", "open", "warning", key="EXC_DEMO_001:HOT_SHUTDOWN", dtype="HOT_SHUTDOWN",
                                      category="operator_behavior", title="Engine shut down hot (no cool-down)",
                                      summary="Engine switched off 40 s after heavy digging; coolant 97 °C.",
                                      recommended_actions=["Idle 3-5 min before shutting down"], one_shot=True))
    assert res["outcome"] == "alert_opened" and res["alerts_cleared"] == res["alerts_opened"]
    assert res["active_alerts"] == [] and len(res["announcements_created"]) == 1
    speech = turn(client, sid, "why-h", "Why did you warn me?").json()["speech"]
    assert "coolant 97 °C" in speech
