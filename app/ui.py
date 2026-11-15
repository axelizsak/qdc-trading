"""Shared look and context of the cockpit: terminal styling, header, key figures, pipeline strip."""
from datetime import date
from html import escape
from pathlib import Path

import streamlit as st

from data import config as settings
from data.schemas import REVIEW_GATES
from graph import pipeline

STEPS = ["data", "technical", "macro_news", "portfolio", "risk"]
STEP_LABELS = {"data": "Data", "technical": "Technical", "macro_news": "Macro / News", "portfolio": "Portfolio",
               "risk": "Risk", "logbook": "Logbook", "review": "Review loop"}
FACTOR_LABELS = {
    "ecb_rates": "ECB rates", "fed_us_rates": "Fed / US rates", "inflation": "Inflation",
    "growth_recession": "Growth", "eurusd": "EUR / USD", "oil": "Oil", "energy_weather": "Energy & weather",
    "gold_silver": "Gold & silver", "copper_metals": "Copper & metals", "uranium": "Uranium",
    "geopolitics_defence": "Geopolitics", "china_em": "China & EM", "tech_ai_semis": "Tech, AI & semis",
    "crypto": "Crypto", "risk_appetite": "Risk appetite",
}

# Terminal palette. Data colours: validated dark steps; amber is for interface accents only.
T = {
    "ink": "#e6e3dc", "muted": "#8a877f", "line": "#232323", "panel": "#0b0b0b", "amber": "#ffa31a",
    "positive": "#3987e5", "negative": "#e66767", "neutral": "#2b2b2b", "rule": "#5a5852",
    "up": "#22c55e", "down": "#ef5b5b", "series": ["#3987e5", "#d95926", "#199e70"],
}
MONO = "'IBM Plex Mono', ui-monospace, monospace"

CSS = f"""
<style>
.block-container {{padding-top: 3.2rem; max-width: 1280px;}}
.qdc-bar {{display:flex; align-items:baseline; gap:14px; font-family:{MONO}; font-size:.78rem;
  letter-spacing:.1em; text-transform:uppercase; color:{T['muted']};}}
.qdc-bar .brand {{color:{T['amber']}; font-weight:600; letter-spacing:.16em;}}
.qdc-bar .sep {{color:{T['rule']};}}
.qdc-bar .hot {{color:{T['amber']};}}
.qdc-title {{font-family:{MONO}; font-size:1.55rem; font-weight:500; color:{T['ink']}; margin:.6rem 0 .1rem;}}
.qdc-sub {{color:{T['muted']}; font-size:.92rem; margin-bottom:1.1rem;}}
.qdc-label {{font-family:{MONO}; color:{T['amber']}; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase;
  border-bottom:1px solid {T['line']}; padding-bottom:6px; margin:1.9rem 0 .8rem;}}
.qdc-label span {{color:{T['muted']}; letter-spacing:.06em; text-transform:none; font-family:inherit; margin-left:10px;}}
.qdc-kpis {{display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:1px; background:{T['line']};
  border:1px solid {T['line']}; border-radius:4px; overflow:hidden;}}
.qdc-kpi {{background:{T['panel']}; padding:14px 16px 13px;}}
.qdc-kpi .l {{font-family:{MONO}; font-size:.66rem; letter-spacing:.14em; text-transform:uppercase; color:{T['muted']};}}
.qdc-kpi .v {{font-size:1.6rem; font-weight:500; color:{T['ink']}; margin-top:6px; line-height:1.15;}}
.qdc-kpi .v.hot {{color:{T['amber']};}}
.qdc-kpi .d {{font-family:{MONO}; font-size:.74rem; margin-top:5px; color:{T['muted']};}}
.qdc-kpi .d.up {{color:{T['up']};}} .qdc-kpi .d.down {{color:{T['down']};}}
.qdc-pipe {{display:flex; align-items:stretch; gap:6px; flex-wrap:wrap;}}
.qdc-step {{flex:1; min-width:118px; border:1px solid {T['line']}; border-radius:4px; padding:10px 12px; background:{T['panel']};}}
.qdc-step .n {{font-family:{MONO}; font-size:.74rem; letter-spacing:.1em; text-transform:uppercase; color:{T['ink']};}}
.qdc-step .m {{font-size:.72rem; color:{T['muted']}; margin-top:3px;}}
.qdc-step .s {{font-family:{MONO}; font-size:.72rem; margin-top:8px;}}
.qdc-arrow {{align-self:center; color:{T['rule']}; font-family:{MONO};}}
.ok {{color:{T['up']};}} .fail {{color:{T['down']};}} .idle {{color:{T['muted']};}} .wait {{color:{T['amber']};}}
.qdc-note {{font-family:{MONO}; font-size:.7rem; letter-spacing:.08em; color:{T['muted']}; text-transform:uppercase;}}
.qdc-note b {{color:{T['amber']}; font-weight:500;}}
</style>
"""


def default_day() -> date:
    """Today: any day can be run, days before the trading window are practice."""
    return date.today()


def as_of() -> date:
    return st.session_state.get("as_of", default_day())


def pipeline_logs() -> Path:
    """Where the pipeline reads and writes: logs/mock/ while any step is on a mock."""
    return pipeline.default_logs_dir()


def macro_logs() -> Path:
    """The Macro/News agent always writes its real outputs to logs/."""
    return settings.LOGS_DIR


def tokens() -> dict:
    return T


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def header() -> None:
    """Top bar on every page: brand, trading day and its status, mock mode, date picker."""
    html(CSS)
    st.session_state.setdefault("as_of", default_day())
    left, right = st.columns([5, 1], vertical_alignment="center")
    with right:
        st.date_input("Trading day", key="as_of", format="YYYY-MM-DD", label_visibility="collapsed",
                      help="Trading days: " + ", ".join(d.strftime("%d %b") for d in settings.TRADING_DAYS))
    day = as_of()
    status, state, pending = pipeline.day_status(day, pipeline_logs())
    orders = len(state.get("approved_orders") or state.get("proposed_orders", []))
    parts = [f"<span class='brand'>QDC ▸ Terminal</span>", f"{day:%a %d %b %Y}"]
    if day not in settings.TRADING_DAYS:
        parts.append("practice day" if day < settings.TRADING_DAYS[0] else "not a trading day")
    where = day_label(status, state, pending)
    parts.append(f"<span class='hot'>waiting · {where}</span>" if status == "waiting for approval" else where)
    if status != "not started":
        parts.append(f"{orders} order{'s' if orders != 1 else ''}")
    if any(settings.USE_MOCK.values()):
        parts.append("mock mode")
    with left:
        html("<div class='qdc-bar'>" + " <span class='sep'>│</span> ".join(parts) + "</div>")


def page(title: str, question: str) -> None:
    html(f"<div class='qdc-title'>{escape(title)}</div><div class='qdc-sub'>{escape(question)}</div>")


def label(text: str, aside: str = "") -> None:
    html(f"<div class='qdc-label'>{escape(text)}" + (f"<span>{escape(aside)}</span>" if aside else "") + "</div>")


def kpis(items: list[tuple]) -> None:
    """Key figures in one row. Each item: (label, value, delta text or None, 'up'|'down'|'hot'|None)."""
    cells = []
    for item in items:
        name, value, delta, tone = (list(item) + [None, None])[:4]
        value_cls = "v hot" if tone == "hot" else "v"
        delta_html = f"<div class='d {tone or ''}'>{escape(delta)}</div>" if delta else ""
        cells.append(f"<div class='qdc-kpi'><div class='l'>{escape(name)}</div>"
                     f"<div class='{value_cls}'>{escape(str(value))}</div>{delta_html}</div>")
    html("<div class='qdc-kpis'>" + "".join(cells) + "</div>")


def day_label(status: str, state: dict, pending: dict | None) -> str:
    """Where the day stands, in a few words: not started, orders review, risk review, ..., approved, no trade."""
    if status == "waiting for approval":
        return pipeline.GATE_NAMES[pending["gate"]]
    if status == "done":
        return "no trade today" if state.get("outcome") == "stopped" else "approved"
    return status


def passed_reviews(state: dict) -> set[str]:
    """The reviews already approved in the current round."""
    round_ = state.get("round", 1)
    return {r["gate"] for r in state.get("reviews", []) if r["round"] == round_ and r["decision"] == "approved"}


def incident_text(incident: dict) -> str:
    return (f"{STEP_LABELS.get(incident['step'], incident['step'])} failed and was replaced by its fallback: "
            f"{incident['error']}")


def demote(markdown: str, levels: int = 2) -> str:
    """Markdown with every heading `levels` smaller, so an agent's report fits inside a page section."""
    return "\n".join("#" * levels + line if line.startswith("#") else line for line in markdown.splitlines())


def step_cells(cells: list[tuple[str, str, str, str]]) -> None:
    """Boxes left to right, each (name, mode, state, css class)."""
    boxes = [f"<div class='qdc-step'><div class='n'>{escape(name)}</div><div class='m'>{escape(mode)}</div>"
             f"<div class='s {cls}'>{escape(text)}</div></div>" for name, mode, text, cls in cells]
    html("<div class='qdc-pipe'>" + "<div class='qdc-arrow'>›</div>".join(boxes) + "</div>")


def pipeline_strip(status: str, state: dict, pending: dict | None) -> None:
    """The day's agents left to right, each with its mode and its state, then where your reviews stand."""
    failed = {i["step"] for i in state.get("incidents", [])}
    round_, passed = state.get("round", 1), passed_reviews(state)
    cells = []
    for key in STEPS:
        mode = "mock" if settings.USE_MOCK[key] else "real"
        if key == "portfolio" and round_ > 1:
            mode += f" · round {round_}"
        if status == "not started":
            cells.append((STEP_LABELS[key], mode, "○ not run", "idle"))
        elif key in failed:
            cells.append((STEP_LABELS[key], mode, "✕ failed", "fail"))
        elif key == "risk" and "orders" not in passed:
            cells.append((STEP_LABELS[key], mode, "– skipped" if status == "done" else "○ after your review", "idle"))
        else:
            cells.append((STEP_LABELS[key], mode, "● ok", "ok"))
        if key == "data":  # the review loop runs right after the data: it has no mock, it is code on the logs
            cells.append((STEP_LABELS["review"], "learns from the logs", *(
                ("○ not run", "idle") if status == "not started" else
                ("✕ failed", "fail") if "review" in failed else ("● ok", "ok"))))
    you = {"not started": ("○ not run", "idle"),
           "waiting for approval": (f"◐ {day_label(status, state, pending)}", "wait"),
           "done": ("■ no trade today", "idle") if state.get("outcome") == "stopped" else ("● approved", "ok")}[status]
    cells.append(("You", "3 reviews", *you))
    step_cells(cells)


def review_steps(pending: dict, state: dict) -> None:
    """The three human reviews of the day, with the one waiting highlighted."""
    passed = passed_reviews(state)
    cells = []
    for i, gate in enumerate(REVIEW_GATES, 1):
        if gate in passed:
            cells.append((f"{i} · {gate}", "you", "● approved", "ok"))
        elif gate == pending["gate"]:
            cells.append((f"{i} · {gate}", "you", "◐ waiting for you", "wait"))
        else:
            cells.append((f"{i} · {gate}", "you", "○ next", "idle"))
    step_cells(cells)


def mock_note(is_mock: bool, what: str = "No real data yet") -> None:
    if is_mock:
        html(f"<div class='qdc-note'><b>mock data</b> · {escape(what)}</div>")
