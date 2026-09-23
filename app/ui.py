"""Visual building blocks for the dashboard: tokens, HTML tiles/cards, Plotly figures.

Colour roles follow the dataviz reference palette (dark steps, validated):
series slots are for identity only; status colours (good/warning/critical) always
ship with an icon + label, never colour alone.
"""

import html
import math
from datetime import datetime, timedelta

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

BG = "#121211"
SURFACE = "#1a1a19"
SURFACE_2 = "#232321"
BORDER = "#2e2e2b"
TEXT = "#f2f1ec"
TEXT_2 = "#c3c2b7"
MUTED = "#8a8984"
ACCENT = "#f5b400"
GRID = "#2a2a28"

STATUS = {  # role -> (colour, icon, label)
    "ok": ("#0ca30c", "✓", "OK"),
    "info": ("#3987e5", "●", "Info"),
    "warning": ("#fab219", "▲", "Warning"),
    "critical": ("#d03b3b", "■", "Critical"),
    "off": ("#5a5955", "○", "Off"),
}
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]  # blue, orange, aqua, yellow (dark steps)
EXPECTED = "#8a8984"
LEVELS = {"L1_threshold": "L1 · Threshold", "L2_trend": "L2 · Trend", "L3_ml_pattern": "L3 · ML pattern",
          "L4_predictive": "L4 · Predictive", "L5_contextual": "L5 · Context"}
SEV_RANK = {"info": 0, "warning": 1, "critical": 2}


def esc(x):
    return html.escape("" if x is None else str(x))


def rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def inject_css():
    st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.6rem; padding-bottom: 2rem; max-width: 1600px; }}
  header[data-testid="stHeader"] {{ background: transparent; height: 2rem; }}
  [data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
  h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
  .cp-brand {{ display:flex; align-items:center; gap:.55rem; font-weight:700; font-size:1.15rem; color:{TEXT}; }}
  .cp-brand .mark {{ width:.85rem; height:.85rem; background:{ACCENT}; border-radius:3px; display:inline-block; }}
  .cp-sub {{ color:{MUTED}; font-size:.78rem; margin-top:-.1rem; }}
  .cp-topbar {{ display:flex; flex-wrap:wrap; gap:.6rem; align-items:stretch; margin-bottom:.9rem; }}
  .cp-clock {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px; padding:.55rem .9rem; min-width:190px; }}
  .cp-clock .t {{ font-size:1.9rem; font-weight:700; color:{TEXT}; font-variant-numeric: tabular-nums; line-height:1.1; }}
  .cp-clock .d {{ color:{MUTED}; font-size:.75rem; }}
  .cp-chip {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px; padding:.5rem .8rem; display:flex;
             flex-direction:column; justify-content:center; min-width:120px; }}
  .cp-chip .k {{ color:{MUTED}; font-size:.7rem; text-transform:uppercase; letter-spacing:.06em; }}
  .cp-chip .v {{ color:{TEXT}; font-size:1.02rem; font-weight:600; font-variant-numeric: tabular-nums; }}
  .cp-grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(165px, 1fr)); gap:.6rem; margin:.4rem 0 .8rem; }}
  .cp-tile {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px; padding:.65rem .8rem .7rem;
             border-top:3px solid var(--st); min-height:104px; }}
  .cp-tile .lab {{ color:{TEXT_2}; font-size:.75rem; display:flex; justify-content:space-between; gap:.3rem; }}
  .cp-tile .val {{ color:{TEXT}; font-size:1.55rem; font-weight:700; font-variant-numeric: tabular-nums; margin-top:.15rem; }}
  .cp-tile .val small {{ font-size:.8rem; color:{TEXT_2}; font-weight:500; margin-left:.15rem; }}
  .cp-tile .sub {{ color:{MUTED}; font-size:.72rem; margin-top:.1rem; line-height:1.25; }}
  .cp-badge {{ display:inline-flex; align-items:center; gap:.3rem; font-size:.7rem; font-weight:700; padding:.1rem .45rem;
              border-radius:999px; border:1px solid var(--st); color:{TEXT}; background: var(--stbg); white-space:nowrap; }}
  .cp-badge .i {{ color: var(--st); }}
  .cp-level {{ display:inline-block; font-size:.68rem; color:{TEXT_2}; border:1px solid {BORDER}; border-radius:6px;
              padding:.05rem .4rem; margin-left:.35rem; white-space:nowrap; }}
  .cp-card {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:.9rem 1rem; margin-bottom:.7rem; }}
  .cp-alert {{ border-left:5px solid var(--st); background: linear-gradient(90deg, var(--stbg), {SURFACE} 55%); }}
  .cp-alert .ttl {{ font-size:1.25rem; font-weight:700; color:{TEXT}; margin:.35rem 0 .25rem; }}
  .cp-alert .sum {{ color:{TEXT_2}; font-size:.9rem; line-height:1.45; }}
  .cp-copilot {{ margin-top:.7rem; background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:10px; padding:.65rem .8rem; }}
  .cp-copilot .who {{ color:{ACCENT}; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }}
  .cp-copilot .head {{ color:{TEXT}; font-weight:600; margin:.2rem 0; }}
  .cp-copilot .msg {{ color:{TEXT_2}; font-size:.88rem; line-height:1.45; }}
  .cp-copilot ol {{ margin:.35rem 0 0 1.1rem; padding:0; color:{TEXT}; font-size:.88rem; }}
  .cp-copilot li {{ margin:.12rem 0; }}
  .cp-tm {{ display:inline-block; margin-top:.5rem; font-size:.75rem; color:{TEXT_2}; border:1px dashed {BORDER};
           border-radius:6px; padding:.15rem .45rem; }}
  .cp-row {{ display:flex; align-items:flex-start; gap:.5rem; padding:.45rem 0; border-bottom:1px solid {BORDER}; }}
  .cp-row:last-child {{ border-bottom:none; }}
  .cp-row .t {{ color:{MUTED}; font-size:.75rem; font-variant-numeric: tabular-nums; min-width:2.6rem; padding-top:.1rem; }}
  .cp-row .b {{ flex:1; }}
  .cp-row .h {{ color:{TEXT}; font-size:.86rem; font-weight:600; }}
  .cp-row .s {{ color:{MUTED}; font-size:.76rem; line-height:1.3; }}
  .cp-mhead {{ display:flex; flex-wrap:wrap; align-items:center; gap:.6rem 1rem; }}
  .cp-mhead .id {{ font-size:1.5rem; font-weight:800; color:{TEXT}; }}
  .cp-mhead .meta {{ color:{TEXT_2}; font-size:.85rem; }}
  .cp-state {{ font-size:.75rem; font-weight:700; padding:.15rem .55rem; border-radius:6px; background:{SURFACE_2};
              border:1px solid {BORDER}; color:{TEXT}; letter-spacing:.04em; }}
  .cp-prog {{ height:8px; background:{SURFACE_2}; border-radius:999px; overflow:hidden; margin:.45rem 0 .25rem; }}
  .cp-prog > div {{ height:100%; background:{SERIES[0]}; border-radius:999px; }}
  .cp-muted {{ color:{MUTED}; font-size:.8rem; }}
  .cp-kv {{ display:grid; grid-template-columns: auto 1fr; gap:.2rem .8rem; font-size:.84rem; }}
  .cp-kv .k {{ color:{MUTED}; }} .cp-kv .v {{ color:{TEXT}; }}
  .cp-empty {{ color:{MUTED}; font-size:.85rem; padding:.4rem 0; }}
  code {{ font-size: .82em; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- HTML pieces
def badge(sev, text=None):
    col, icon, label = STATUS.get(sev, STATUS["info"])
    return (f'<span class="cp-badge" style="--st:{col};--stbg:{rgba(col, .14)}"><span class="i">{icon}</span>'
            f'{esc(text or label)}</span>')


def level_chip(level):
    return f'<span class="cp-level">{esc(LEVELS.get(level, level))}</span>'


def tile(label, value, unit="", status="ok", sub=""):
    col, icon, lab = STATUS.get(status, STATUS["ok"])
    return (f'<div class="cp-tile" style="--st:{col}"><div class="lab"><span>{esc(label)}</span>'
            f'<span style="color:{col}" title="{lab}">{icon} {lab}</span></div>'
            f'<div class="val">{esc(value)}<small>{esc(unit)}</small></div><div class="sub">{esc(sub)}</div></div>')


def tiles(items):
    st.markdown('<div class="cp-grid">' + "".join(items) + "</div>", unsafe_allow_html=True)


def chip(k, v):
    return f'<div class="cp-chip"><div class="k">{esc(k)}</div><div class="v">{v}</div></div>'


def alert_card(alert, agent_msg=None):
    sev = alert["severity"]
    col = STATUS[sev][0]
    since = alert.get("since", "")[11:16]
    if agent_msg:
        acts = "".join(f"<li>{esc(a)}</li>" for a in agent_msg.get("actions", []))
        cop = (f'<div class="cp-copilot"><div class="who">Copilot (LangGraph agent)</div>'
               f'<div class="head">{esc(agent_msg.get("headline"))}</div><div class="msg">{esc(agent_msg.get("message"))}'
               f'</div>{"<ol>" + acts + "</ol>" if acts else ""}</div>')
    else:
        acts = "".join(f"<li>{esc(a)}</li>" for a in alert.get("actions", []))
        cop = (f'<div class="cp-copilot"><div class="who">Playbook (fallback until the agent replies)</div>'
               f'{"<ol>" + acts + "</ol>" if acts else "<div class=msg>No standard action.</div>"}</div>') if acts else ""
    tm = alert.get("training")
    tmh = f'<div class="cp-tm">Training: {esc(tm["id"])} · {esc(tm["title"])}</div>' if tm else ""
    st.markdown(
        f'<div class="cp-card cp-alert" style="--st:{col};--stbg:{rgba(col, .16)}">'
        f'<div>{badge(sev)}{level_chip(alert.get("level"))}<span class="cp-level">since {esc(since)}</span></div>'
        f'<div class="ttl">{esc(alert["title"])}</div><div class="sum">{esc(alert["summary"])}</div>{cop}{tmh}</div>',
        unsafe_allow_html=True)


def event_rows(events, empty="Nothing here yet."):
    if not events:
        return f'<div class="cp-empty">{esc(empty)}</div>'
    out = []
    for e in events:
        sev = e["severity"]
        tag = "" if e.get("status") in ("open", None) else f' <span class="cp-level">{esc(e["status"])}</span>'
        out.append(f'<div class="cp-row"><div class="t">{esc((e.get("ts") or e.get("since") or "")[11:16])}</div>'
                   f'<div class="b"><div class="h">{badge(sev, STATUS[sev][2])} {esc(e.get("machine_id", ""))} · '
                   f'{esc(e["title"])}{tag}</div><div class="s">{esc(e["summary"][:220])}</div></div></div>')
    return "".join(out)


# ----------------------------------------------------------------------------- figures
def style(fig, height=260, legend=True, hover="x unified"):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_2, size=12), hovermode=hover, showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    return fig


def show(fig, key):
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def health_chart(rows, target, label, unit, floor, limit=None, height=240):
    xs = [r["minute"] for r in rows]
    act = [r.get(target) for r in rows]
    exp = [r.get(f"exp_{target}") for r in rows]
    up = [None if e is None else e + floor for e in exp]
    lo = [None if e is None else e - floor for e in exp]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=up, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False,
                             connectgaps=False))
    fig.add_trace(go.Scatter(x=xs, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(EXPECTED, .18), name="Normal band (ML)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=exp, mode="lines", line=dict(color=EXPECTED, width=1.5, dash="dash"),
                             name="ML expected", hovertemplate="%{y:.1f} " + unit))
    fig.add_trace(go.Scatter(x=xs, y=act, mode="lines", line=dict(color=SERIES[0], width=2), name="Measured",
                             hovertemplate="<b>%{y:.1f} " + unit + "</b>"))
    if limit is not None:
        fig.add_hline(y=limit, line=dict(color=STATUS["critical"][0], width=1, dash="dot"),
                      annotation_text=f"Alarm limit {limit:g} {unit}", annotation_font_color=TEXT_2,
                      annotation_position="top left")
    fig.update_layout(title=dict(text=f"{label} ({unit})", font=dict(size=13, color=TEXT), x=0, y=0.98))
    return style(fig, height)


GF_STYLE = {"overhead_powerline": ("#d03b3b", "dash", .08), "buried_utility": ("#ec835a", "dot", .08),
            "speed_zone": ("#fab219", "dot", .025)}


def _circle(fig, cx, cy, r, color, dash="dash", fill=0.0, width=1.5):
    fig.add_shape(type="circle", x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r, layer="below",
                  line=dict(color=color, width=width, dash=dash), fillcolor=rgba(color, fill) if fill else None)


def site_map(static, state, focus=None, span=80, height=560, show_labels=True, label_ids=None):
    fig = go.Figure()
    for zid, z in static["zones"].items():
        cx, cy = z["center"]
        _circle(fig, cx, cy, z["radius"], "#6d6c67", dash="solid", fill=0.04, width=1)
        if show_labels:
            fig.add_annotation(x=cx, y=cy + z["radius"], text=esc(z["name"]), showarrow=False, yshift=9,
                               font=dict(size=10, color=MUTED))
    for gf in static["geofences"]:
        col, dash, alpha = GF_STYLE[gf["type"]]
        xs = [p[0] for p in gf["polygon"]] + [gf["polygon"][0][0]]
        ys = [p[1] for p in gf["polygon"]] + [gf["polygon"][0][1]]
        extra = f" (≤ {gf['max_height_m']:g} m)" if gf.get("max_height_m") else (
            f" ({gf['speed_limit_kmh']:g} km/h)" if gf.get("speed_limit_kmh") else "")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=rgba(col, alpha),
                                 line=dict(color=col, width=1.5, dash=dash), name=gf["name"] + extra,
                                 hovertemplate=f"<b>{esc(gf['name'])}</b>{esc(extra)}<extra></extra>"))
    rx = [p[0] for p in static["haul_route"]]
    ry = [p[1] for p in static["haul_route"]]
    fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines", line=dict(color="#4d4c48", width=4), name="Haul road",
                             hoverinfo="skip"))
    for sev_key in ("ok", "warning", "critical", "off"):
        xs, ys, texts, hov, syms = [], [], [], [], []
        for mid, m in state["machines"].items():
            t = m["telemetry"]
            if t is None:
                continue
            al = m.get("active_alerts", [])
            worst = max((SEV_RANK[a["severity"]] for a in al), default=-1)
            s = "off" if not t["engine_on"] else ("critical" if worst == 2 else "warning" if worst == 1 else "ok")
            if s != sev_key:
                continue
            is_exc = t["machine_type"] == "hydraulic_excavator"
            if is_exc and t["engine_on"]:
                r = m["derived"].get("danger_radius_m")
                if r:
                    _circle(fig, t["x"], t["y"], r, STATUS[s][0], dash="dot", fill=0.06)
            xs.append(t["x"])
            ys.append(t["y"])
            syms.append("square" if is_exc else "triangle-up")
            show_txt = (label_ids is None and t["engine_on"]) or (label_ids is not None and mid in label_ids)
            texts.append(f"{mid} {STATUS[s][1]}" if show_txt else "")
            top = al[0]["title"] if al else "No active alerts"
            hov.append(f"<b>{mid}</b> · {t['state']}<br>{esc(t.get('operator_id') or 'no operator')}<br>"
                       f"Fuel {t['fuel_level_pct']:.0f}%<br>{esc(top)}")
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers+text", text=texts, textposition="top center",
                textfont=dict(color=TEXT, size=11),
                marker=dict(symbol=syms, size=15, color=STATUS[sev_key][0], line=dict(color=BG, width=2)),
                name=f"Machine · {STATUS[sev_key][2]}", hovertext=hov, hovertemplate="%{hovertext}<extra></extra>"))
    wk = state["site"].get("workers", [])
    if wk:
        fig.add_trace(go.Scatter(
            x=[w["x"] for w in wk], y=[w["y"] for w in wk], mode="markers+text",
            text=[w["name"] if focus is not None else "" for w in wk], textposition="bottom center",
            textfont=dict(color=TEXT_2, size=10),
            marker=dict(size=10, color="#ffffff", line=dict(color=SERIES[0], width=3)), name="Ground crew (UWB tag)",
            hovertext=[f"<b>{esc(w['name'])}</b><br>{esc(w['role'])}" for w in wk],
            hovertemplate="%{hovertext}<extra></extra>"))
    fig = style(fig, height, hover="closest")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showticklabels=False, showgrid=False)
    fig.update_xaxes(showticklabels=False, showgrid=False)
    if focus is not None:
        fx, fy = focus
        fig.update_xaxes(range=[fx - span, fx + span])
        fig.update_yaxes(range=[fy - span * 0.75, fy + span * 0.75])
    fig.update_layout(legend=dict(orientation="h", y=-0.02, yanchor="top"), margin=dict(l=4, r=4, t=8, b=4))
    return fig


def _at(day, hhmm):
    if not hhmm:
        return None
    h, m = map(int, hhmm.split(":"))
    return day.replace(hour=h, minute=m, second=0)


def plan_gantt(rows, now, rain_hhmm=None, shift_end=None, height=230):
    day = now.replace(hour=0, minute=0, second=0)
    fig = go.Figure()
    labels = [f"{r['id']} · {r['name'][:24]}{'…' if len(r['name']) > 24 else ''}" for r in rows]
    prev_end = None
    series = {"Plan (at shift start)": ([], [], [], "#5a5955"), "Forecast": ([], [], [], SERIES[0]),
              "Done": ([], [], [], SERIES[2])}
    for lab, r in zip(labels, rows):
        ps, pf = _at(day, r["planned_start"]), _at(day, r["planned_finish"])
        series["Plan (at shift start)"][0].append(lab)
        series["Plan (at shift start)"][1].append(ps)
        series["Plan (at shift start)"][2].append((pf - ps).total_seconds() * 1000)
        start = _at(day, r.get("actual_start")) or prev_end or now
        if r.get("status") == "done":
            end = _at(day, r.get("actual_end")) or start
            key = "Done"
        else:
            end = _at(day, r.get("predicted_finish")) or start
            if r.get("status") == "planned":
                start = max(start, now)
            key = "Forecast"
        prev_end = end
        series[key][0].append(lab)
        series[key][1].append(start)
        series[key][2].append(max((end - start).total_seconds() * 1000, 60000))
    for i, (name, (ys, bases, xs, col)) in enumerate(series.items()):
        if not ys:
            continue
        fig.add_trace(go.Bar(y=ys, base=bases, x=xs, orientation="h", name=name, offsetgroup=str(i > 0),
                             marker=dict(color=col, line=dict(color=BG, width=2)),
                             width=0.36, hovertemplate="%{y}<br>%{base|%H:%M} → %{customdata}<extra>" + name + "</extra>",
                             customdata=[(b + timedelta(milliseconds=x)).strftime("%H:%M")
                                         for b, x in zip(bases, xs)]))
    for t, txt, col, dash in ((now, "now", ACCENT, "solid"), (_at(day, rain_hhmm), "rain forecast", "#3987e5", "dash"),
                              (shift_end, "shift end", MUTED, "dot")):
        if t is None:
            continue
        fig.add_shape(type="line", x0=t, x1=t, y0=0, y1=1, yref="paper", line=dict(color=col, width=1.5, dash=dash))
        fig.add_annotation(x=t, y=1, yref="paper", text=txt, showarrow=False, yshift=8,
                           font=dict(size=10, color=col))
    fig.update_layout(barmode="group", bargap=0.35)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(type="date", tickformat="%H:%M")
    return style(fig, height, hover="closest")


IDLE_CAUSES = [("truck_wait", "Waiting for trucks"), ("operator_idle", "Operator idle"),
               ("unattended", "Left running unattended"), ("warmup", "Warm-up")]


def idle_chart(machines, height=200):
    fig = go.Figure()
    ids = [mid for mid, m in machines.items() if m["telemetry"]["machine_type"] == "hydraulic_excavator"]
    for i, (key, label) in enumerate(IDLE_CAUSES):
        vals = [machines[mid]["idle_min_today"].get(key, 0.0) for mid in ids]
        fig.add_trace(go.Bar(y=ids, x=vals, orientation="h", name=label,
                             marker=dict(color=SERIES[i], line=dict(color=BG, width=2)),
                             hovertemplate="<b>%{x:.0f} min</b> " + label + "<extra>%{y}</extra>"))
    fig.update_layout(barmode="stack", legend=dict(traceorder="normal"))
    fig.update_xaxes(title=dict(text="idle minutes today", font=dict(size=11, color=MUTED)))
    return style(fig, height, hover="closest")


def lead_chart(per_fault, height=240):
    names = {"radiator_clog": "Radiator clogging", "oil_pump_wear": "Oil pump wear", "injector_wear": "Injector wear"}
    faults = list(per_fault)
    ys = [names.get(f, f) for f in faults]
    ml = [per_fault[f]["ml_median_detect_min"] for f in faults]
    th = [per_fault[f]["threshold_median_detect_min"] for f in faults]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=ys, x=ml, orientation="h", name="ML early warning (L3)",
                         marker=dict(color=SERIES[0], line=dict(color=BG, width=2)),
                         text=[f"{v:.0f} min" if v is not None else "" for v in ml], textposition="outside",
                         hovertemplate="<b>%{x:.0f} min</b> after fault onset<extra>ML</extra>"))
    fig.add_trace(go.Bar(y=ys, x=[v if v is not None else 0 for v in th], orientation="h",
                         name="Fixed alarm threshold (L1)",
                         marker=dict(color=SERIES[1], line=dict(color=BG, width=2)),
                         text=[f"{v:.0f} min" if v is not None else "never fired" for v in th],
                         textposition="outside",
                         hovertemplate="<b>%{x:.0f} min</b> after fault onset<extra>Threshold</extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title=dict(text="median minutes from fault onset to first alert (lower is better)",
                                font=dict(size=11, color=MUTED)))
    fig.update_yaxes(autorange="reversed")
    return style(fig, height, hover="closest")


def task_scatter(sample, height=300):
    act = [s["actual"] for s in sample]
    fig = go.Figure()
    m = max(act + [s["pred"] for s in sample] + [s["naive"] for s in sample]) if sample else 1
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", line=dict(color=MUTED, width=1, dash="dot"),
                             name="Perfect", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=act, y=[s["naive"] for s in sample], mode="markers", name="Planner rule-of-thumb",
                             marker=dict(size=8, color=SERIES[1], opacity=.7, line=dict(color=BG, width=1)),
                             hovertemplate="actual %{x:.0f} · estimate <b>%{y:.0f} min</b><extra>rule</extra>"))
    fig.add_trace(go.Scatter(x=act, y=[s["pred"] for s in sample], mode="markers", name="ML task-time model",
                             marker=dict(size=8, color=SERIES[0], opacity=.8, line=dict(color=BG, width=1)),
                             hovertemplate="actual %{x:.0f} · predicted <b>%{y:.0f} min</b><extra>ML</extra>"))
    fig.update_xaxes(title=dict(text="actual task duration (min)", font=dict(size=11, color=MUTED)))
    fig.update_yaxes(title=dict(text="estimate (min)", font=dict(size=11, color=MUTED)))
    return style(fig, height, hover="closest")


def incident_chart(rec, height=330):
    rows = rec["pre"] + rec["post"]
    t0 = datetime.fromisoformat(rec["trigger_ts"])
    xs = [(datetime.fromisoformat(r["ts"]) - t0).total_seconds() for r in rows]
    panels = [("swing_speed_dps", "Swing speed (°/s)"), ("load_moment_pct", "Load moment (% of limit)"),
              ("max_height_m", "Max linkage height (m)")]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=[p[1] for p in panels])
    for i, (k, lab) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=xs, y=[r.get(k) for r in rows], mode="lines", line=dict(color=SERIES[0], width=2),
                                 name=lab, hovertemplate="%{y:.1f}<extra>" + lab + "</extra>"), row=i, col=1)
        fig.add_vline(x=0, line=dict(color=STATUS["critical"][0], width=1, dash="dot"), row=i, col=1)
    fig.update_annotations(font=dict(size=11, color=TEXT_2))
    fig.update_xaxes(title=dict(text="seconds relative to trigger", font=dict(size=11, color=MUTED)), row=3, col=1)
    return style(fig, height, legend=False)


def ratio(a, b):
    return 0 if not b else a / b


def fmt_num(x, nd=0, dash="–"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return dash
    return f"{x:,.{nd}f}"
