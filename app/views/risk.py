"""Risk Manager: the portfolio once your orders are filled, measured against reference levels. It never advises."""
import pandas as pd
import streamlit as st

from app import data, ui
from data.schemas import RISK_REPORT
from graph import pipeline

STATUS = {"ok": "✓ ok", "warning": "▲ warning", "breach": "✕ breach"}
as_of, logs = ui.as_of(), ui.pipeline_logs()
_, state, _ = pipeline.day_status(as_of, logs)
record = data.day_record(as_of, logs)  # what the day wrote to the logs; the saved state while a mock is running
ui.page("Risk Manager", "The portfolio once your orders are filled, on 22 metrics against reference levels. A warning "
                        "is information; only a breach breaks a rule.")
report = record["risk_report"] if len(record["risk_report"]) else pd.DataFrame(state.get("risk_report", []),
                                                                             columns=RISK_REPORT)
page = record["risk_page"] or f"{state.get('risk_board', '')}\n{state.get('risk_reading', '')}"
if report.empty:
    st.info("The Risk Manager runs after the orders review: nothing yet for this day.", icon=":material/info:")
    st.page_link("views/today.py", label="Go to your review", icon=":material/arrow_forward:")
    st.stop()

counts = report["status"].value_counts()
values = report.set_index("metric")["value"]


def figure(metric: str, fmt: str) -> str:
    return f"{values[metric]:{fmt}}" if metric in values.index and pd.notna(values[metric]) else "—"


ui.kpis([("Breaches", int(counts.get("breach", 0)), None, "down" if counts.get("breach", 0) else None),
         ("Warnings", int(counts.get("warning", 0)), None, "wait" if counts.get("warning", 0) else None),
         ("Volatility", figure("volatility", ".1%"), "a year, estimated"),
         ("Chance of ending below today", figure("prob_below_start", ".0%"), "at the end of the window")])

areas = data.md_table(page, "| Area")
if len(areas):
    ui.label("By area")
    ui.chips([(f"{a.Area}: {a.Status}", a.Status.lower()) for a in areas.itertuples()])
    st.write("")
    st.dataframe(areas.rename(columns={"In one line": "Why"}), hide_index=True, width="stretch",
                 column_config={"Why": st.column_config.TextColumn("Why", width="large")})

paragraphs = data.section(page, "### Independent reading").split("\n\n")
if len(paragraphs) > 1:
    ui.label("Independent reading", "written by the LLM from the numbers; it changes none of them")
    st.markdown(paragraphs[1])
    scenarios = data.md_table(page, "| Scenario")
    if len(scenarios):
        st.dataframe(scenarios, hide_index=True, width="stretch")

ui.document_download("The full risk report", page, f"risk_{as_of}")
ui.label("Every metric")
st.dataframe(report.drop(columns="date").assign(status=report["status"].map(STATUS)), hide_index=True, width="stretch",
             column_config={"metric": st.column_config.TextColumn("Metric"),
                            "value": st.column_config.NumberColumn("Value", format="%.4g"),
                            "limit": st.column_config.NumberColumn("Reference level", format="%g"),
                            "status": st.column_config.TextColumn("Status"),
                            "comment": st.column_config.TextColumn("Comment", width="large")})
