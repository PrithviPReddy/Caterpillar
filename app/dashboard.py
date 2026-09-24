"""Operator Copilot - demo UI.

    streamlit run app/dashboard.py        (needs the API: uvicorn server.api:app --port 8100)
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ui  # noqa: E402

API = os.environ.get("COPILOT_API", "http://localhost:8100")

st.set_page_config(page_title="Operator Copilot", page_icon="🚜", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

PAGES = ["Operator cab", "Site map", "Shift plan", "Machine health", "Operators & coaching", "Event stream",
         "Model report", "Agent integration"]
HEALTH_SIGNALS = [("coolant_temp_c", "Coolant temperature", "°C", 2.0, 100.0),
                  ("hyd_oil_temp_c", "Hydraulic oil temperature", "°C", 5.0, 90.0),
                  ("oil_pressure_kpa", "Engine oil pressure", "kPa", 20.0, None),
                  ("fuel_rate_lph", "Fuel rate", "L/h", 1.2, None)]


# ----------------------------------------------------------------------------- API helpers
def api_get(path, **params):
    try:
        r = requests.get(API + path, params=params, timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path, body):
    try:
        r = requests.post(API + path, json=body, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.toast(f"API error: {exc}")
        return None


@st.cache_data(ttl=20, show_spinner=False)
def site_static(_nonce):
    return api_get("/api/site")


@st.cache_data(ttl=60, show_spinner=False)
def model_metrics():
    return api_get("/api/metrics") or {}


def control(action, value=None):
    api_post("/api/control", {"action": action, "value": value})
    if action == "reset":
        st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
        st.session_state["last_seq"] = 0


# ----------------------------------------------------------------------------- sidebar
status = api_get("/api/status")
with st.sidebar:
    st.markdown('<div class="cp-brand"><span class="mark"></span>Operator Copilot</div>'
                '<div class="cp-sub">Smart Operator Assistant · telemetry → ML → agent</div>', unsafe_allow_html=True)
    st.write("")
    page = st.radio("View", PAGES, key="page", label_visibility="collapsed")
    if status is None:
        st.error(f"API not reachable at {API}. Start it with `uvicorn server.api:app --port 8100`.")
        st.stop()
    static = site_static(st.session_state.get("nonce", 0))
    machine_ids = list(static["plans"]) if static else []
    default_idx = machine_ids.index("EXC001") if "EXC001" in machine_ids else 0
    mid = st.selectbox("Machine", machine_ids, index=default_idx, key="mid")
    st.divider()
    st.caption("SCENARIO PLAYBACK")
    c1, c2 = st.columns(2)
    if status["playing"]:
        c1.button("Pause", on_click=control, args=("pause",), width="stretch")
    else:
        c1.button("Play", on_click=control, args=("play",), type="primary", width="stretch")
    c2.button("Next alert ⏭", on_click=control, args=("skip",), width="stretch",
              help="Fast-forward until the next warning/critical alert, then pause")
    speeds = [10, 30, 60, 120, 300, 600]
    cur = min(speeds, key=lambda s: abs(s - status["speed"]))
    st.select_slider("Speed (× real time)", speeds, value=cur, key="speed",
                     on_change=lambda: control("speed", st.session_state["speed"]))
    j1, j2 = st.columns([3, 2])
    jump_to = j1.text_input("Jump to (HH:MM)", value="", placeholder="09:15", label_visibility="collapsed")
    j2.button("Jump", on_click=lambda: control("jump", jump_to) if jump_to else None, width="stretch")
    st.button("Restart scenario", on_click=control, args=("reset", status["scenario"]), width="stretch")
    st.divider()
    mods = status.get("models_loaded", {})
    wh = status.get("webhook", {})
    st.markdown(
        f'<div class="cp-kv"><span class="k">Scenario</span><span class="v">{ui.esc(status["scenario"])}</span>'
        f'<span class="k">ML models</span><span class="v">{"loaded" if mods.get("health") else "missing - run ml.train"}</span>'
        f'<span class="k">Agent webhook</span><span class="v">{ui.esc(wh.get("url") or "not set")}</span>'
        f'<span class="k">Events</span><span class="v">{status["events_total"]}</span></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- shared renderers
def top_bar(state):
    s = state["status"]
    t = datetime.fromisoformat(s["sim_time"])
    w = state["site"]["weather"]
    mode = "▶ playing" if s["playing"] else ("■ finished" if s["done"] else "❚❚ paused")
    active = [a for m in state["machines"].values() for a in m["active_alerts"]]
    n_crit = sum(a["severity"] == "critical" for a in active)
    n_warn = sum(a["severity"] == "warning" for a in active)
    rain = w.get("forecast_rain_start")
    lk = w.get("lightning_km")
    chips = [
        f'<div class="cp-clock"><div class="t">{t:%H:%M:%S}</div><div class="d">{t:%a %d %b %Y} · {mode} · ×{s["speed"]:.0f}</div></div>',
        ui.chip("Critical alerts", f'{ui.badge("critical", str(n_crit)) if n_crit else ui.badge("ok", "0")}'),
        ui.chip("Warnings", f'{ui.badge("warning", str(n_warn)) if n_warn else ui.badge("ok", "0")}'),
        ui.chip("Ambient", f'{w.get("ambient_temp_c", "–")} °C · {w.get("humidity_pct", "–"):.0f}% RH'
                if w else "–"),
        ui.chip("Rain forecast", ui.esc(rain or "none")),
        ui.chip("Lightning", (ui.badge("critical" if lk < 10 else "warning", f"{lk:.0f} km") if lk is not None
                              else "none")),
        ui.chip("Gusts", f'{w.get("wind_gust_kmh", 0):.0f} km/h'),
    ]
    st.markdown('<div class="cp-topbar">' + "".join(chips) + "</div>", unsafe_allow_html=True)


def toasts(state):
    last = st.session_state.get("last_seq", 0)
    new = [e for e in state["recent_events"] if e["seq"] > last]
    if new:
        st.session_state["last_seq"] = new[-1]["seq"]
    if last == 0:
        return
    shown = 0
    for e in new:
        if e["status"] in ("open", "escalated") and e["severity"] in ("warning", "critical") and shown < 3:
            st.toast(f"**{e['machine_id']}** · {e['title']}", icon="🚨" if e["severity"] == "critical" else "⚠️")
            shown += 1


def agent_msg_for(state, alert):
    for m in reversed(state.get("agent_messages", [])):
        if m.get("incident_key") == alert["key"] or m.get("event_id") == alert["event_id"]:
            return m
    return None


def excavator_tiles(m):
    t = m["telemetry"]
    d = m["derived"]
    exp = m.get("expected", {})
    keys = {a["type"] for a in m["active_alerts"]}
    cool = t["coolant_temp_c"]
    ce = exp.get("coolant_temp_c", [None])[0]
    cs = "critical" if cool >= 105 else "warning" if (cool >= 100 or "COOLING_DEGRADATION_DETECTED" in keys) else "ok"
    hyd = t["hyd_oil_temp_c"]
    hs = "critical" if hyd >= 98 else "warning" if hyd >= 90 else "ok"
    fuel = d.get("fuel", {})
    fs = "critical" if t["fuel_level_pct"] < 8 else "warning" if (t["fuel_level_pct"] < 15 or "FUEL_SHORTFALL" in keys) else "ok"
    lm = t["load_moment_pct"]
    ls = "critical" if lm >= 100 else "warning" if lm >= 85 else "ok"
    nw = d.get("nearest_worker")
    r = d.get("danger_radius_m") or 0
    if nw and t["engine_on"]:
        ps = ("critical" if (nw["distance_m"] < r or "PROXIMITY_DANGER_ZONE" in keys) else
              "warning" if (nw["distance_m"] < r + 6 or "PROXIMITY_WARNING" in keys) else "ok")
        pv, psub = f"{nw['distance_m']:.0f}", f"{nw['id']} · danger radius {r:.1f} m"
    else:
        ps, pv, psub = "ok", "–", "nobody tagged nearby"
    belt = t["seatbelt_fastened"]
    bs = "ok" if (belt or not t["seat_occupied"]) else ("critical" if "SEATBELT_UNFASTENED" in keys else "warning")
    fat = d.get("fatigue") or {}
    fr = fat.get("risk")
    fts = "critical" if (fr or 0) >= 0.8 else "warning" if (fr or 0) >= 0.55 else "ok"
    hsc = m.get("health_score")
    hss = "ok" if hsc is None or hsc >= 80 else "warning" if hsc >= 60 else "critical"
    idle = sum(m["idle_min_today"].values())
    items = [
        ui.tile("Coolant", f"{cool:.1f}", "°C", cs, f"ML expects {ce:.1f} °C" if ce else "limits 100 / 105 °C"),
        ui.tile("Hydraulic oil", f"{hyd:.1f}", "°C", hs, "limits 90 / 98 °C"),
        ui.tile("Fuel", f"{t['fuel_level_pct']:.0f}", "%", fs,
                f"≈{fuel.get('hours_left', 0):.1f} work-h left · dry ≈{fuel.get('runout', '–')}" if fuel else ""),
        ui.tile("Tipping margin used", f"{lm:.0f}", "%", ls, f"reach {t['reach_m']:.1f} m · pitch {t['pitch_deg']:.0f}°"),
        ui.tile("Nearest person", pv, "m" if pv != "–" else "", ps, psub),
        ui.tile("Seatbelt", "Fastened" if belt else ("Open" if t["seat_occupied"] else "Seat empty"), "", bs,
                f"compliance {m['stats']['seatbelt_compliance_pct']:.0f}% today"),
        ui.tile("Fatigue risk", f"{fr:.2f}" if fr is not None else "–", "", fts,
                f"{fat.get('hours_in_seat', 0):.1f} h in seat · cab HI {fat.get('cab_heat_index_c', 0):.0f} °C"
                if fat else "builds with time in seat, heat, control jerk"),
        ui.tile("Machine health (ML)", f"{hsc:.0f}" if hsc is not None else "–", "/100" if hsc is not None else "", hss,
                "residual models + isolation forest"),
        ui.tile("Engine load", f"{t['engine_load_pct']:.0f}", "%", "ok", f"{t['engine_rpm']:.0f} rpm · {t['state']}"),
        ui.tile("Idle today", f"{idle:.0f}", "min", "warning" if idle > 60 else "ok",
                f"${m['idle_cost_today']:.2f} of fuel"),
    ]
    ui.tiles(items)


def truck_tiles(m):
    t = m["telemetry"]
    keys = {a["type"] for a in m["active_alerts"]}
    tp = t.get("trans_pressure_kpa", 0)
    ts = "critical" if (t["engine_on"] and tp < 800) else "warning" if (t["engine_on"] and tp < 1200) else "ok"
    items = [
        ui.tile("Speed", f"{t['travel_speed_kmh']:.0f}", "km/h", "warning" if "OVERSPEED" in keys else "ok",
                f"state {t['state']}"),
        ui.tile("Payload", f"{t['payload_t']:.1f}", "t", "ok", "target 25 t"),
        ui.tile("Fuel", f"{t['fuel_level_pct']:.0f}", "%", "warning" if t["fuel_level_pct"] < 15 else "ok", ""),
        ui.tile("Coolant", f"{t['coolant_temp_c']:.1f}", "°C", "ok", ""),
        ui.tile("Transmission pressure", f"{tp:.0f}", "kPa", ts, "limits 1200 / 800 kPa"),
        ui.tile("Fuel temp sensor", f"{t['fuel_temp_c']:.1f}", "°C",
                "warning" if any(k == "SENSOR_FLATLINE" for k in keys) else "ok", "sensor health monitored"),
    ]
    ui.tiles(items)


def machine_header(m, mid):
    t = m["telemetry"]
    plan = static["plans"].get(mid, {})
    ops = static["operators"]
    op = t.get("operator_id") or plan.get("operator_id")
    opname = ops.get(op, {}).get("name", op)
    zone = static["zones"].get(t.get("zone") or "", {}).get("name", "in transit")
    spec = static["specs"]["excavator" if t["machine_type"] == "hydraulic_excavator" else "truck"]["model_class"]
    task_html = ""
    rows = m["derived"].get("plan") or []
    cur = next((r for r in rows if r["id"] == t.get("task_id")), None)
    if cur:
        pct = 100 * cur["done_m3"] / cur["volume_m3"]
        eta = cur.get("predicted_finish") or "–"
        plan_f = cur.get("planned_finish") or "–"
        task_html = (f'<div style="margin-top:.55rem"><div class="cp-muted">Current task · {ui.esc(cur["id"])} '
                     f'{ui.esc(cur["name"])}</div><div class="cp-prog"><div style="width:{min(pct, 100):.0f}%"></div></div>'
                     f'<div class="cp-muted">{cur["done_m3"]:.0f} / {cur["volume_m3"]} m³ ({pct:.0f}%) · '
                     f'ML forecast finish <b style="color:{ui.TEXT}">{eta}</b> · plan {plan_f}</div></div>')
    st.markdown(
        f'<div class="cp-card"><div class="cp-mhead"><span class="id">{ui.esc(mid)}</span>'
        f'<span class="cp-state">{ui.esc(t["state"])}</span><span class="meta">{ui.esc(spec)}</span>'
        f'<span class="meta">Operator: <b style="color:{ui.TEXT}">{ui.esc(opname)}</b> ({ui.esc(op)})</span>'
        f'<span class="meta">📍 {ui.esc(zone)}</span></div>{task_html}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- pages
def page_cab(state):
    m = state["machines"][mid]
    left, right = st.columns([1.65, 1], gap="medium")
    with left:
        machine_header(m, mid)
        alerts = m["active_alerts"]
        if alerts:
            ui.alert_card(alerts[0], agent_msg_for(state, alerts[0]))
        else:
            st.markdown(f'<div class="cp-card">{ui.badge("ok", "All clear")} '
                        f'<span class="cp-muted">No active alerts for {mid}. The copilot is watching '
                        f'{len(static["channels"])} channels.</span></div>', unsafe_allow_html=True)
        if m["telemetry"]["machine_type"] == "hydraulic_excavator":
            excavator_tiles(m)
            rows = api_get(f"/api/timeseries/{mid}", minutes=120) or []
            if rows:
                ui.show(ui.health_chart(rows, "coolant_temp_c", "Coolant: measured vs ML expected", "°C", 2.0, 100.0,
                                        height=220), key="cab_cool")
        else:
            truck_tiles(m)
    with right:
        t = m["telemetry"]
        ui.show(ui.site_map(static, state, focus=(t["x"], t["y"]), span=55, height=330, show_labels=False,
                            label_ids={mid}), key="cab_map")
        others = alerts[1:] if alerts else []
        st.markdown(f'<div class="cp-card"><div class="cp-muted" style="margin-bottom:.3rem">OTHER ACTIVE ALERTS '
                    f'({len(others)})</div>{ui.event_rows(others[:6], "None.")}</div>', unsafe_allow_html=True)
        msgs = [x for x in state.get("agent_messages", []) if x["machine_id"] == mid][-4:]
        body = "".join(f'<div class="cp-row"><div class="t">{ui.esc(x["sim_time"][11:16])}</div><div class="b">'
                       f'<div class="h">{ui.badge(x["severity"])} {ui.esc(x["headline"])}</div>'
                       f'<div class="s">{ui.esc(x["message"][:240])}</div></div></div>' for x in reversed(msgs))
        st.markdown(f'<div class="cp-card"><div class="cp-muted" style="margin-bottom:.3rem">COPILOT MESSAGES</div>'
                    f'{body or ui.event_rows([], "No agent messages yet. POST to /api/agent/messages.")}</div>',
                    unsafe_allow_html=True)


def page_map(state):
    ui.show(ui.site_map(static, state, height=620), key="site_map")
    rows = []
    for mmid, m in state["machines"].items():
        t = m["telemetry"]
        al = m["active_alerts"]
        rows.append({"Machine": mmid, "Type": "Excavator" if t["machine_type"] == "hydraulic_excavator" else "Truck",
                     "Operator": t.get("operator_id") or "–", "State": t["state"],
                     "Zone": static["zones"].get(t.get("zone") or "", {}).get("name", "–"),
                     "Fuel %": round(t["fuel_level_pct"]),
                     "Health": "–" if m.get("health_score") is None else f'{m["health_score"]:.0f}',
                     "Alerts": len(al), "Top alert": f'{ui.STATUS[al[0]["severity"]][1]} {al[0]["title"]}' if al else "✓ none"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def page_plan(state):
    now = datetime.fromisoformat(state["status"]["sim_time"])
    rain = state["site"]["weather"].get("forecast_rain_start")
    for emid, m in state["machines"].items():
        if m["telemetry"]["machine_type"] != "hydraulic_excavator":
            continue
        d = m["derived"]
        plan = static["plans"][emid]
        st.markdown(f"#### {emid} · {static['operators'].get(plan['operator_id'], {}).get('name', '')}")
        brief = next((e for e in reversed(state["recent_events"]) if e["machine_id"] == emid
                      and e["type"] == "SHIFT_BRIEFING"), None)
        if brief is None:
            brief = next((e for e in (api_get("/api/events", machine_id=emid) or []) if e["type"] == "SHIFT_BRIEFING"), None)
        if not d.get("plan"):
            st.markdown('<div class="cp-empty">Plan is built when the engine starts for the shift.</div>',
                        unsafe_allow_html=True)
            continue
        fuel, svc, de = d.get("fuel", {}), d.get("service", {}), d.get("def", {})
        keys = {a["type"] for a in m["active_alerts"]}
        end = d.get("plan_end") or "–"
        shift_end = plan["shift_end"][11:16]
        over = end > shift_end
        ui.tiles([
            ui.tile("Fuel covers", f"{fuel.get('hours_left', 0):.1f}", "h", "warning" if "FUEL_SHORTFALL" in keys else "ok",
                    f"needed {fuel.get('need_h', 0):.1f} h · burn {fuel.get('burn_lph', 0):.1f} L/h"),
            ui.tile("DEF covers", f"{de.get('hours_left', 0):.0f}", "h", "warning" if "DEF_SHORTFALL" in keys else "ok", ""),
            ui.tile("Service overdue by" if svc.get("hours_to_service", 0) < 0 else "Service due in",
                    f"{abs(svc.get('hours_to_service', 0)):.1f}", "h",
                    "warning" if keys & {"SERVICE_DUE", "SERVICE_OVERDUE"} else "ok",
                    f"{svc.get('next_service_h', 0):.0f} h service"),
            ui.tile("Rain", rain or "none", "", "warning" if any(k.startswith("WEATHER") for k in keys) else "ok",
                    "weather-sensitive tasks checked"),
            ui.tile("Plan ends (ML)", end, "", "warning" if over else "ok", f"shift ends {shift_end}"),
            ui.tile("Delay now", f"{d.get('task_delay_min', 0):+d}", "min",
                    "warning" if any(k == "TASK_DELAY" for k in keys) else "ok", "vs plan at shift start"),
        ])
        ui.show(ui.plan_gantt(d["plan"], now, rain, datetime.fromisoformat(plan["shift_end"]),
                              height=90 + 55 * len(d["plan"])), key=f"gantt_{emid}")
        if brief:
            st.markdown(f'<div class="cp-card"><div class="cp-muted">PRE-SHIFT BRIEFING · {brief["ts"][11:16]}</div>'
                        f'<div style="color:{ui.TEXT};margin-top:.25rem">{ui.esc(brief["summary"])}</div>'
                        f'<div class="cp-muted" style="margin-top:.35rem">{ui.esc(brief["evidence"]["tasks"]["value"])}</div>'
                        f'</div>', unsafe_allow_html=True)


def page_health(state):
    if static["plans"][mid]["machine_type"] != "hydraulic_excavator":
        st.info("ML health models cover the excavators. Pick EXC001 or EXC002 in the sidebar.")
        return
    m = state["machines"][mid]
    rows = api_get(f"/api/timeseries/{mid}", minutes=480) or []
    ml_alerts = [a for a in m["active_alerts"] if a["level"] in ("L1_threshold", "L2_trend", "L3_ml_pattern")
                 and a["category"] == "machine_health"]
    hsc = m.get("health_score")
    res = m.get("residuals", {})
    exp = m.get("expected", {})
    tl = [ui.tile("Health score", f"{hsc:.0f}" if hsc is not None else "–", "/100",
                  "ok" if hsc is None or hsc >= 80 else "warning" if hsc >= 60 else "critical",
                  "isolation forest on residuals")]
    for key, label, unit, floor, _ in HEALTH_SIGNALS:
        r = res.get(key)
        sig = exp.get(key, [None, None])[1]
        lim = max(floor, 3 * (sig or 0))
        stt = ("critical" if r is not None and abs(r) >= 3 * lim else
               "warning" if r is not None and abs(r) >= lim else "ok")
        tl.append(ui.tile(label, f"{r:+.1f}" if r is not None else "–", unit, stt, "deviation from ML expected (smoothed)"))
    ui.tiles(tl)
    if ml_alerts:
        for a in ml_alerts[:2]:
            ui.alert_card(a, agent_msg_for(state, a))
    if not rows:
        st.markdown('<div class="cp-empty">No data yet.</div>', unsafe_allow_html=True)
        return
    df = pd.DataFrame(rows)
    scored = df[df["exp_coolant_temp_c"].notna()] if "exp_coolant_temp_c" in df else df.iloc[0:0]
    if scored.empty:
        st.markdown('<div class="cp-empty">The models score steady, warm running; waiting for 25 min of it.</div>',
                    unsafe_allow_html=True)
        return
    num = scored.select_dtypes("number").columns
    smooth = scored.copy()
    smooth[num] = scored[num].rolling(5, min_periods=2).mean()
    srows = smooth.to_dict("records")
    c1, c2 = st.columns(2)
    for i, (key, label, unit, floor, limit) in enumerate(HEALTH_SIGNALS):
        with (c1 if i % 2 == 0 else c2):
            ui.show(ui.health_chart(srows, key, f"{label}, 5-min avg", unit, floor, limit), key=f"h_{key}")
    st.caption("Gray band = what the regression model (trained on 30 simulated normal days) expects for the current "
               "load, rpm, ambient and warm-up, ± the smallest deviation that matters physically. The ML alert fires "
               "when the smoothed residual stays outside the band for 3+ minutes - typically hours before the fixed "
               "alarm limit.")


def page_operators(state):
    mets = model_metrics()
    fleet = (mets.get("baselines") or {}).get("fleet", {})
    ui.show(ui.idle_chart(state["machines"]), key="idle")
    evs = api_get("/api/events") or []
    for emid, m in state["machines"].items():
        if m["telemetry"]["machine_type"] != "hydraulic_excavator":
            continue
        plan = static["plans"][emid]
        op = plan["operator_id"]
        opinfo = static["operators"].get(op, {})
        s = m["stats"]
        summary = next((e for e in reversed(evs) if e["machine_id"] == emid and e["type"] == "SHIFT_SUMMARY"), None)
        mine = [e for e in evs if e["machine_id"] == emid and e["status"] in ("open", "escalated")]
        tms = {}
        for e in mine:
            if e.get("training_module"):
                tms.setdefault(e["training_module"]["id"], [e["training_module"]["title"], 0])[1] += 1
        es_h = s["end_stops"] / max(s["work_h"], 0.1)
        score = summary["evidence"]["score"]["value"] if summary else None
        st.markdown(f"#### {opinfo.get('name', op)} ({op}) on {emid}")
        ui.tiles([
            ui.tile("Shift score", f"{score}" if score is not None else "live", "/100" if score is not None else "",
                    "ok" if score is None or score >= 75 else "warning" if score >= 55 else "critical",
                    "published at end of shift"),
            ui.tile("End-stop hits", f"{es_h:.0f}", "/h", "warning" if es_h > 25 else "ok",
                    f"fleet median {fleet.get('end_stops_per_h', 0):.0f}/h"),
            ui.tile("Seatbelt compliance", f"{s['seatbelt_compliance_pct']:.0f}", "%",
                    "ok" if s["seatbelt_compliance_pct"] >= 99 else "warning", "while machine moving"),
            ui.tile("Idle cost today", f"${m['idle_cost_today']:.0f}", "", "ok",
                    f"{sum(m['idle_min_today'].values()):.0f} min idle"),
            ui.tile("Fuel used", f"{s['fuel_used_l']:.0f}", "L", "ok", f"{s['engine_on_h']:.1f} engine h"),
            ui.tile("Safety events", f"{sum(1 for e in mine if e['category'] == 'safety')}", "",
                    "warning" if any(e["category"] == "safety" for e in mine) else "ok", "today"),
        ])
        if tms:
            items = "".join(f'<div class="cp-row"><div class="t">{ui.esc(k)}</div><div class="b"><div class="h">'
                            f'{ui.esc(v[0])}</div><div class="s">triggered by {v[1]} event(s) today</div></div></div>'
                            for k, v in sorted(tms.items()))
            st.markdown(f'<div class="cp-card"><div class="cp-muted">RECOMMENDED TRAINING (feeds the training hub)'
                        f'</div>{items}</div>', unsafe_allow_html=True)
        if summary:
            st.markdown(f'<div class="cp-card"><div class="cp-muted">SHIFT SCORECARD</div>'
                        f'<div style="color:{ui.TEXT}">{ui.esc(summary["summary"])}</div></div>', unsafe_allow_html=True)


def page_events(state):
    evs = api_get("/api/events", limit=2000) or []
    f1, f2, f3, f4 = st.columns(4)
    sev = f1.multiselect("Severity", ["critical", "warning", "info"], default=["critical", "warning"])
    lvls = f2.multiselect("Intelligence level", list(ui.LEVELS), format_func=lambda x: ui.LEVELS[x])
    cats = f3.multiselect("Category", sorted({e["category"] for e in evs}))
    mids = f4.multiselect("Machine", sorted({e["machine_id"] for e in evs}))
    sel = [e for e in evs if e["severity"] in sev and (not lvls or e["intelligence_level"] in lvls)
           and (not cats or e["category"] in cats) and (not mids or e["machine_id"] in mids)]
    df = pd.DataFrame([{"Time": e["ts"][11:19], "Sev": f'{ui.STATUS[e["severity"]][1]} {e["severity"]}',
                        "Status": e["status"], "Machine": e["machine_id"], "Level": ui.LEVELS[e["intelligence_level"]],
                        "Type": e["type"], "Title": e["title"], "id": e["event_id"]} for e in reversed(sel)])
    st.dataframe(df, width="stretch", hide_index=True, height=320)
    if sel:
        pick = st.selectbox("Inspect event", [e["event_id"] for e in reversed(sel)],
                            format_func=lambda i: next(f"{e['ts'][11:16]} {e['machine_id']} {e['title']}"
                                                       for e in sel if e["event_id"] == i))
        ev = next(e for e in sel if e["event_id"] == pick)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            ui.alert_card({"key": ev["incident_key"], "event_id": ev["event_id"], "severity": ev["severity"],
                           "level": ev["intelligence_level"], "since": ev["ts"], "title": ev["title"],
                           "summary": ev["summary"], "actions": ev["recommended_actions"],
                           "training": ev.get("training_module")}, agent_msg_for(state, {"key": ev["incident_key"],
                                                                                          "event_id": ev["event_id"]}))
            evd = pd.DataFrame([{"Signal": k, **{kk: vv for kk, vv in v.items() if vv is not None}}
                                for k, v in ev["evidence"].items()]).astype(object).fillna("")
            st.dataframe(evd, width="stretch", hide_index=True)
        with c2:
            st.caption("Exact payload sent to the LangGraph agent (schema v1.0)")
            st.json(ev, expanded=1)
    incs = api_get("/api/incidents") or []
    if incs:
        st.markdown("#### Black-box recordings (60 s before, 30 s after)")
        pick = st.selectbox("Incident", [i["incident_id"] for i in incs],
                            format_func=lambda i: next(f"{x['trigger_ts'][11:19]} {x['machine_id']} {x['title']}"
                                                       for x in incs if x["incident_id"] == i))
        rec = api_get(f"/api/incidents/{pick}")
        if rec:
            ui.show(ui.incident_chart(rec), key="incident")


def page_models():
    mets = model_metrics()
    if not mets:
        st.warning("No metrics yet. Run `python -m ml.train`.")
        return
    ev = mets.get("evaluation", {})
    faults = ev.get("faults", {})
    fa = ev.get("false_alarms", {})
    tm = mets.get("task_model", {})
    leads = [v["median_lead_time_min"] for v in faults.values() if v.get("median_lead_time_min")]
    ml_rate = sum(v["ml_detection_rate_pct"] for v in faults.values()) / max(len(faults), 1)
    th_rate = sum(v["threshold_detection_rate_pct"] for v in faults.values()) / max(len(faults), 1)
    st.markdown("#### What the ML adds over fixed alarm limits")
    ui.tiles([
        ui.tile("Faults caught by ML", f"{ml_rate:.0f}", "%", "ok", f"vs {th_rate:.0f}% by fixed thresholds"),
        ui.tile("Median early warning", f"{min(leads):.0f}–{max(leads):.0f}" if leads else "–", "min", "ok",
                "before the threshold alarm"),
        ui.tile("False ML alarms", f"{fa.get('per_100_engine_h', 0):.1f}", "/100 h", "ok",
                f"{fa.get('engine_hours', 0):.0f} engine-h of normal days"),
        ui.tile("Task-time error (ML)", f"{tm.get('mape_pct', 0):.1f}", "%", "ok",
                f"rule-of-thumb {tm.get('baseline_mape_pct', 0):.1f}%"),
        ui.tile("History used", f"{mets.get('history_days', 0)}", "days", "ok", "simulated, 2 excavators/day"),
    ])
    if faults:
        ui.show(ui.lead_chart(faults), key="lead")
        st.caption("Each fault was injected into held-out simulated days at a random time and speed. The models never "
                   "see the fault, only the sensors. Injector wear never trips a fixed limit - only the ML sees it.")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("#### Task-duration model (held-out tasks)")
        if tm.get("holdout_sample"):
            ui.show(ui.task_scatter(tm["holdout_sample"][:200]), key="task_sc")
    with c2:
        st.markdown("#### Health regression models")
        hm = mets.get("health_models", {})
        st.dataframe(pd.DataFrame([{"Signal": v["label"], "RMSE": f'{v["rmse"]} {v["unit"]}',
                                    "Naive RMSE": f'{v["naive_rmse"]} {v["unit"]}', "R²": v["r2"]}
                                   for v in hm.values()]), width="stretch", hide_index=True)
        st.markdown("#### Operator baselines (from history)")
        ops = (mets.get("baselines") or {}).get("operators", {})
        st.dataframe(pd.DataFrame([{"Operator": k, "End-stops/h": round(v["end_stops_per_h"] or 0, 1),
                                    "Impacts/h": round(v["impacts_per_h"] or 0, 1), "Jerk": round(v["jerk"] or 0, 2),
                                    "Idle %": round(100 * (v["idle_frac"] or 0), 1), "L/h": round(v["burn_lph"] or 0, 1),
                                    "Days": v["days"]} for k, v in ops.items()]), width="stretch", hide_index=True)


def page_integration(state):
    st.markdown("#### Connecting the LangGraph agent")
    st.markdown(f"""
| Direction | Endpoint | Notes |
|---|---|---|
| Detection → agent (pull) | `GET {API}/api/stream?min_severity=warning` | Server-sent events, one schema-v1 `Event` per message |
| Detection → agent (push) | `POST {API}/api/agent/webhook` `{{"url": "http://agent/..."}}` | We POST every `Event` as JSON |
| Agent → cab | `POST {API}/api/agent/messages` | `AgentMessage`; shown in the operator cab next to the alert |
| Schema | `GET {API}/api/schema` | JSON Schema for `Event` + `AgentMessage`, event types, training modules |
| Backfill | `GET {API}/api/events?since=<seq>` | Everything after a sequence number |
""")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Webhook")
        url = st.text_input("Agent webhook URL", value=state["status"]["webhook"].get("url") or "",
                            placeholder="http://localhost:8123/events")
        ms = st.selectbox("Minimum severity", ["info", "warning", "critical"], index=1)
        if st.button("Save webhook"):
            api_post("/api/agent/webhook", {"url": url or None, "min_severity": ms})
        st.markdown("##### Send an agent message (manual test)")
        alerts = [a for m in state["machines"].values() for a in m["active_alerts"]]
        target = st.selectbox("Reply to", ["(none)"] + [f"{a['key']}" for a in alerts])
        head = st.text_input("Headline", "Stop - person in swing radius")
        msg = st.text_area("Message", "Chris is 5 m behind your counterweight. Hold all functions until he signals clear.")
        if st.button("Send to cab"):
            a = next((x for x in alerts if x["key"] == target), None)
            api_post("/api/agent/messages", {"machine_id": target.split(":")[0] if a else mid,
                                             "incident_key": a["key"] if a else None,
                                             "event_id": a["event_id"] if a else None,
                                             "severity": a["severity"] if a else "info", "headline": head,
                                             "message": msg, "actions": [], "speak": True})
    with c2:
        st.markdown("##### Example event")
        ex = next((e for e in reversed(state["recent_events"]) if e["severity"] != "info"), None)
        st.json(ex or {}, expanded=False)


# ----------------------------------------------------------------------------- live fragments
def fetch_state():
    return api_get("/api/state")


def live(render, every):
    @st.fragment(run_every=every)
    def _frag():
        state = fetch_state()
        if state is None:
            st.error("Lost connection to the API.")
            return
        top_bar(state)
        toasts(state)
        render(state)
    _frag()


if page == "Operator cab":
    live(page_cab, 1.0)
elif page == "Site map":
    live(page_map, 1.0)
elif page == "Shift plan":
    live(page_plan, 3.0)
elif page == "Machine health":
    live(page_health, 3.0)
elif page == "Operators & coaching":
    live(page_operators, 5.0)
elif page == "Event stream":
    live(page_events, 6.0)
elif page == "Model report":
    page_models()
else:
    state = fetch_state()
    if state:
        page_integration(state)
