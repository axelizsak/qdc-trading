"""Logbook: the audit trail of the competition, then the selected day told in order.

Built by agents/logbook.py from everything the agents and the human wrote. Each row of the ledger holds the
hash of the row before it, so a changed or deleted row shows here as a broken chain.
"""
import streamlit as st

from agents import logbook
from app import ui

as_of, logs = ui.as_of(), ui.pipeline_logs()
ui.page("Logbook", "What the AI proposed, what you changed, and what it was worth, day by day.")



def untitled(markdown: str) -> str:
    """A logbook page without its first line (the title: the cockpit has its own)."""
    return "\n".join(markdown.splitlines()[1:])


ledger = logbook.load(logs)
if ledger.empty:
    ui.kpis([("Days recorded", 0), ("Events", 0), ("Hash chain", "—")])
    st.write("")
    st.caption("Nothing recorded yet: the logbook fills when a day reaches the Portfolio Manager.")
    st.stop()

problems = logbook.verify(logs)
ui.kpis([("Days recorded", ledger["date"].nunique()), ("Events", len(ledger)),
         ("Hash chain", "Intact" if not problems else "BROKEN", None, "up" if not problems else "down")])
for problem in problems[:5]:
    st.error(problem, icon=":material/link_off:")

ui.label("Every day", "human against AI, and the value against the benchmark")
st.markdown(ui.demote(untitled(logbook.render_index(ledger, problems))))

ui.label(f"{as_of:%a %d %b}", "the day in order, from the signals to the valuation")
if (ledger["date"].astype(str) == as_of.isoformat()).any():
    with st.container(border=True):
        st.markdown(ui.demote(untitled(logbook.render_day(ledger, as_of.isoformat()))))
else:
    st.caption("Nothing recorded for this day. Pick another date at the top right.")
