"""Shared look and context of the cockpit: light styling, header, key figures, step chips."""
from datetime import date
from html import escape
from pathlib import Path

import streamlit as st

from data import config as settings
from data.schemas import REVIEW_GATES
from graph import pipeline

STEP_LABELS = {"data": "Data", "technical": "Technical", "macro_news": "Macro / News", "portfolio": "Portfolio",
               "risk": "Risk", "logbook": "Logbook", "learning": "Learning", "dividends": "Ex-dividend dates",
               "references": "Customer references"}
FACTOR_LABELS = settings.MACRO_FACTOR_LABELS

# Light palette. In the data: blue = up / buy, red = down / sell, grey = neutral; amber = waiting for you.
T = {
    "ink": "#1f2328", "muted": "#6b7280", "line": "#e5e7eb", "panel": "#ffffff", "soft": "#f4f5f7",
    "accent": "#2563eb", "wait": "#d97706", "positive": "#2563eb", "negative": "#dc2626", "neutral": "#d1d5db",
    "rule": "#9ca3af", "up": "#16a34a", "down": "#dc2626", "series": ["#2563eb", "#ea580c", "#16a34a"],
}
FONT = "Inter"

CSS = f"""
<style>
.block-container {{padding-top: 3rem; max-width: 1240px;}}
.qdc-bar {{display:flex; align-items:baseline; gap:12px; font-size:.82rem; color:{T['muted']};}}
.qdc-bar .brand {{color:{T['accent']}; font-weight:600;}}
.qdc-bar .sep {{color:{T['line']};}}
.qdc-bar .wait {{color:{T['wait']}; font-weight:500;}}
.qdc-title {{font-size:1.65rem; font-weight:600; color:{T['ink']}; margin:.5rem 0 .1rem;}}
.qdc-sub {{color:{T['muted']}; font-size:.95rem; margin-bottom:1.2rem;}}
.qdc-label {{color:{T['muted']}; font-size:.78rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
  margin:1.8rem 0 .7rem;}}
.qdc-label span {{font-weight:400; letter-spacing:0; text-transform:none; margin-left:8px;}}
.qdc-kpis {{display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px;}}
.qdc-kpi {{border:1px solid {T['line']}; border-radius:10px; padding:14px 16px; background:{T['panel']};}}
.qdc-kpi .l {{font-size:.78rem; color:{T['muted']};}}
.qdc-kpi .v {{font-size:1.5rem; font-weight:600; color:{T['ink']}; margin-top:4px; line-height:1.2;}}
.qdc-kpi .v.wait {{color:{T['wait']};}}
.qdc-kpi .d {{font-size:.8rem; margin-top:4px; color:{T['muted']};}}
.qdc-kpi .d.up {{color:{T['up']};}} .qdc-kpi .d.down {{color:{T['down']};}}
.qdc-steps {{display:flex; align-items:stretch; gap:8px; flex-wrap:wrap; margin-bottom:14px;}}
.qdc-step {{flex:1; min-width:110px; border:1px solid {T['line']}; border-radius:10px; padding:10px 12px;
  background:{T['soft']};}}
.qdc-step .n {{font-size:.86rem; font-weight:600; color:{T['ink']};}}
.qdc-step .s {{font-size:.8rem; margin-top:4px;}}
.qdc-step.now {{border-color:{T['wait']}; background:#fffbeb;}}
.ok {{color:{T['up']};}} .fail {{color:{T['down']};}} .idle {{color:{T['muted']};}} .wait {{color:{T['wait']};}}
.qdc-chips {{display:flex; gap:8px; flex-wrap:wrap;}}
.qdc-chip {{border-radius:999px; padding:5px 12px; font-size:.84rem; font-weight:500;}}
.qdc-chip.ok {{background:#f0fdf4; color:#15803d;}} .qdc-chip.warning {{background:#fffbeb; color:#b45309;}}
.qdc-chip.breach {{background:#fef2f2; color:#b91c1c;}} .qdc-chip.none {{background:{T['soft']}; color:{T['muted']};}}
.qdc-chip.code {{background:#f4f5f7; color:#4b5563; border:1px solid #9ca3af;}}
.qdc-chip.llm {{background:#eff6ff; color:#1d4ed8; border:1px solid #2563eb;}}
.qdc-chip.you {{background:#fffbeb; color:#b45309; border:1px solid #d97706;}}
.qdc-chip.back {{color:#b45309; background:none; padding-left:0;}} .qdc-chip.next {{color:#1d4ed8; background:none;}}
code, pre {{font-variant-ligatures: none;}}
.qdc-note {{font-size:.8rem; color:{T['muted']};}} .qdc-note b {{color:{T['wait']}; font-weight:600;}}
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
    """The Macro/News agent and the learning step always write their real outputs to logs/."""
    return settings.LOGS_DIR


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def header() -> None:
    """Top bar on every page: brand, trading day and where it stands, mock mode, date picker."""
    html(CSS)
    st.session_state.setdefault("as_of", default_day())
    left, right = st.columns([5, 1], vertical_alignment="center")
    with right:
        st.date_input("Trading day", key="as_of", format="YYYY-MM-DD", label_visibility="collapsed",
                      help="Trading days: " + ", ".join(d.strftime("%d %b") for d in settings.TRADING_DAYS))
    day = as_of()
    status, state, pending = pipeline.day_status(day, pipeline_logs())
    parts = ["<span class='brand'>QDC Trading</span>", f"{day:%a %d %b %Y}"]
    if day not in settings.TRADING_DAYS:
        parts.append("practice day" if day < settings.TRADING_DAYS[0] else "not a trading day")
    where = day_label(status, state, pending)
    parts.append(f"<span class='wait'>waiting for you · {where}</span>" if status == "waiting for approval" else where)
    if any(settings.USE_MOCK.values()):
        parts.append("mock mode")
    with left:
        html("<div class='qdc-bar'>" + " <span class='sep'>|</span> ".join(parts) + "</div>")


def page(title: str, question: str) -> None:
    html(f"<div class='qdc-title'>{escape(title)}</div><div class='qdc-sub'>{escape(question)}</div>")


def label(text: str, aside: str = "") -> None:
    html(f"<div class='qdc-label'>{escape(text)}" + (f"<span>{escape(aside)}</span>" if aside else "") + "</div>")


def kpis(items: list[tuple]) -> None:
    """Key figures in one row. Each item: (label, value, delta text or None, 'up'|'down'|'wait'|None)."""
    cells = []
    for item in items:
        name, value, delta, tone = (list(item) + [None, None])[:4]
        value_cls = "v wait" if tone == "wait" else "v"
        delta_html = f"<div class='d {tone or ''}'>{escape(delta)}</div>" if delta else ""
        cells.append(f"<div class='qdc-kpi'><div class='l'>{escape(name)}</div>"
                     f"<div class='{value_cls}'>{escape(str(value))}</div>{delta_html}</div>")
    html("<div class='qdc-kpis'>" + "".join(cells) + "</div>")


def chips(items: list[tuple[str, str]]) -> None:
    """Status pills, each (text, 'ok'|'warning'|'breach'|'none')."""
    html("<div class='qdc-chips'>" + "".join(f"<span class='qdc-chip {cls}'>{escape(text)}</span>"
                                             for text, cls in items) + "</div>")


def day_label(status: str, state: dict, pending: dict | None) -> str:
    """Where the day stands, in a few words: not started, orders review, risk review, ..., approved, no trade."""
    if status == "waiting for approval":
        return pipeline.GATE_NAMES[pending["gate"]]
    if status == "done":
        return "no trade today" if state.get("outcome") == "stopped" else "approved"
    return "agents running" if status == "running" else status


def incident_text(incident: dict) -> str:
    return (f"{STEP_LABELS.get(incident['step'], incident['step'])} failed and was replaced by its fallback: "
            f"{incident['error']}")


def demote(markdown: str, levels: int = 2) -> str:
    """Markdown with every heading `levels` smaller, so an agent's text fits inside a page section."""
    return "\n".join("#" * levels + line if line.startswith("#") else line for line in markdown.splitlines())


@st.cache_data(show_spinner=False)
def _pdf_of(text: str) -> bytes | None:
    """The PDF of a Markdown text, or None if it cannot be made (the .md download still works)."""
    from app.pdf import to_pdf  # only loaded when a document is offered

    try:
        return to_pdf(text)
    except Exception:  # a document the PDF maker cannot read must not break the page
        return None


@st.cache_data(show_spinner=False)
def _deck_of(text: str) -> bytes | None:
    """The PowerPoint deck of an executive summary, or None if it cannot be made."""
    from app.deck import to_pptx  # only loaded when a deck is offered

    try:
        return to_pptx(text)
    except Exception:  # a summary the deck maker cannot read must not break the page
        return None


def document_download(label: str, text: str, name: str, deck: bool = False, **kwargs) -> None:
    """A button that offers a Markdown document as .md or .pdf, the reader's choice (`name` has no extension).
    `deck=True` (the executive summary) adds a PowerPoint deck."""
    pdf = _pdf_of(text)
    with st.popover(label, icon=kwargs.pop("icon", ":material/description:"), width=kwargs.pop("width", "content")):
        st.download_button("Markdown (.md)", text, f"{name}.md", "text/markdown", width="stretch",
                           key=f"{name}_md", **kwargs)
        if pdf is None:
            st.caption("The PDF could not be made for this document.")
        else:
            st.download_button("PDF (.pdf)", pdf, f"{name}.pdf", "application/pdf", width="stretch",
                               key=f"{name}_pdf", **kwargs)
        slides = _deck_of(text) if deck else None
        if deck and slides is None:
            st.caption("The PowerPoint could not be made for this summary.")
        elif slides is not None:
            st.download_button("PowerPoint (.pptx)", slides, f"{name}.pptx",
                               "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                               width="stretch", key=f"{name}_pptx", **kwargs)


def steps(cells: list[tuple[str, str, str, bool]]) -> None:
    """Boxes left to right, each (name, state text, css class of the state, is it the one waiting)."""
    html("<div class='qdc-steps'>" + "".join(
        f"<div class='qdc-step{' now' if now else ''}'><div class='n'>{escape(name)}</div>"
        f"<div class='s {cls}'>{escape(text)}</div></div>" for name, text, cls, now in cells) + "</div>")


HUMAN_STEPS = ["1 · Run the agents", "2 · Orders", "3 · Risk", "4 · Sign off", "5 · Export"]


def human_steps(status: str, state: dict, pending: dict | None, outcome: str) -> None:
    """The day as you live it, left to right: run, your three reviews, the export. Done steps read done, the one
    waiting for you is highlighted. `outcome` is what the logs say (data.day_outcome), for a day read from its files."""
    stopped = status == "done" and state.get("outcome") == "stopped" or outcome == "no trade today"
    if status == "done" or outcome in ("approved", "no trade today"):
        if stopped:
            cells = [(HUMAN_STEPS[0], "done", "ok", False)] + [(name, "no trade today", "idle", False)
                                                                for name in HUMAN_STEPS[1:]]
        else:
            cells = [(name, "done", "ok", False) for name in HUMAN_STEPS[:4]]
            cells.append((HUMAN_STEPS[4], "ready", "wait", True))
    elif status == "waiting for approval":
        at = REVIEW_GATES.index(pending["gate"]) + 1
        cells = [(name, "done" if i < at else "waiting for you" if i == at else "next",
                  "ok" if i < at else "wait" if i == at else "idle", i == at) for i, name in enumerate(HUMAN_STEPS)]
    else:
        cells = [(HUMAN_STEPS[0], "to do", "wait", True)] + [(name, "next", "idle", False) for name in HUMAN_STEPS[1:]]
    steps(cells)


def mock_note(is_mock: bool, what: str = "No real data yet") -> None:
    if is_mock:
        html(f"<div class='qdc-note'><b>Mock data</b> · {escape(what)}</div>")
