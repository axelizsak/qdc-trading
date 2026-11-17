"""How it works: the day as a diagram. Click a box to see what the step does, what it gave today, and its code."""
import streamlit as st

from app import charts, flow, ui
from graph import pipeline

KIND = {"code": "Code", "llm": "Agent that asks the LLM", "you": "You"}
as_of, logs = ui.as_of(), ui.pipeline_logs()
status, state, pending = pipeline.day_status(as_of, logs)
ui.page("How it works", "One trading day, step by step. Click a box.")
ui.chips([("Code", "code"), ("Asks the LLM", "llm"), ("You", "you"), ("- - - your review can send the day back", "back"),
          ("- - - the next morning, the learning reads the day", "next")])

selected = st.session_state.get("flow_step", "data")
event = st.altair_chart(charts.flow_diagram(flow.nodes(), flow.edges(), selected), on_select="rerun", key="flow",
                        width="stretch")
for name in ("box", "label"):
    picked = (event.get("selection") or {}).get(name) or []
    if picked and isinstance(picked[0], dict) and picked[0].get("step") in flow.BY_KEY:
        if picked[0]["step"] != selected:
            st.session_state["flow_step"] = picked[0]["step"]
            st.rerun()

step = flow.BY_KEY[selected]
with st.container(border=True):
    st.subheader(step.name)
    st.caption(KIND[step.kind])
    st.markdown(step.what)
    st.markdown(f"**In** · {step.inputs} → **Out** · {step.outputs}")
    st.markdown(f"**Today** · {flow.today(step.key, status, state, pending, logs, as_of)}")
    st.markdown("**Files** · " + " · ".join(f"`{f}`" for f in step.files))
    if st.toggle("Show the code", key=f"code_{step.key}"):
        for fn in step.code:
            where, source = flow.code_of(fn)
            st.caption(where)
            st.code(source, language="python", height=min(420, 22 * (source.count("\n") + 2)))
