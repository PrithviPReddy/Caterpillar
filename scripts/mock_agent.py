"""Stand-in for the LangGraph agent, and a reference client for the real one.

It subscribes to the event stream, turns each actionable event into a short
operator-facing message, and posts it back so it shows up in the cab UI.
Replace `compose()` with a call into the LangGraph graph; everything else stays.

    python scripts/mock_agent.py                 # default: warnings and criticals
    python scripts/mock_agent.py --min-severity info
"""

import argparse
import json
import time

import requests

API = "http://localhost:8000"

TONE = {
    "safety": "Safety first",
    "machine_health": "Machine health",
    "fuel_energy": "Fuel",
    "planning": "Planning",
    "productivity": "Productivity",
    "operator_behavior": "Coaching",
    "security": "Security",
    "sensor_health": "Sensor check",
    "site": "Site conditions",
}


def compose(ev):
    """The whole 'agent' - swap this for graph.invoke({"event": ev})."""
    ctx = ev.get("context") or {}
    task = (ctx.get("task") or {}).get("name")
    pred = ev.get("prediction") or {}
    when = f" Expected impact around {pred['eta'][11:16]}." if pred.get("eta") else ""
    where = f" (task: {task})" if task else ""
    if ev["severity"] == "critical" and ev["category"] == "safety":
        headline = f"STOP - {ev['title']}"
    else:
        headline = f"{TONE.get(ev['category'], 'Notice')}: {ev['title']}"
    message = f"{ev['summary']}{when}{where}"
    return {
        "event_id": ev["event_id"],
        "incident_key": ev["incident_key"],
        "machine_id": ev["machine_id"] if ev["machine_id"] != "SITE" else "EXC001",
        "severity": ev["severity"],
        "headline": headline,
        "message": message,
        "actions": ev.get("recommended_actions", [])[:3],
        "requires_ack": ev["severity"] == "critical",
        "speak": ev["severity"] == "critical",
    }


def stream(min_severity):
    url = f"{API}/api/stream?min_severity={min_severity}&statuses=open,escalated"
    with requests.get(url, stream=True, timeout=(5, None)) as r:
        r.raise_for_status()
        data = []
        for line in r.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line.startswith("data:"):
                data.append(line[5:].strip())
            elif line == "" and data:
                yield json.loads("\n".join(data))
                data = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-severity", default="warning", choices=["info", "warning", "critical"])
    args = ap.parse_args()
    print(f"mock agent listening on {API}/api/stream (>= {args.min_severity})", flush=True)
    while True:
        try:
            for ev in stream(args.min_severity):
                msg = compose(ev)
                requests.post(f"{API}/api/agent/messages", json=msg, timeout=5)
                print(f"{ev['ts'][11:19]} {ev['machine_id']:6s} {ev['type']:28s} -> {msg['headline']}", flush=True)
        except requests.RequestException as exc:
            print(f"stream dropped ({exc}); retrying in 2 s")
            time.sleep(2)


if __name__ == "__main__":
    main()
