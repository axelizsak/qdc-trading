"""Technical agent: what the prices say about each ETF today."""
import numpy as np
import pandas as pd
import streamlit as st

from agents import technical
from app import charts, data, ui
from data.universe import load_universe
from graph import pipeline

TOP = 10
as_of, logs = ui.as_of(), ui.pipeline_logs()
_, state, _ = pipeline.day_status(as_of, logs)
signals, signals_mock = data.day_signals(as_of, logs, state)
ui.page("Technical agent", "What the prices say about each ETF: three rule answers weighted by what each earned since "
                           "2022 (technical_det), and the LLM's reading of the same answers (technical_llm).")
ui.mock_note(signals_mock, "this day has not run yet")

det = signals[signals["agent"] == technical.AGENT_DET].set_index("ticker")
llm = signals[signals["agent"] == "technical_llm"].set_index("ticker")
if det.empty:
    st.info("No technical score for this day.", icon=":material/info:")
    st.stop()
both = det.index.intersection(llm.index)
calls = both[(det.loc[both, "score"].abs() >= 0.1) & (llm.loc[both, "score"].abs() >= 0.1)]
agree = float((np.sign(det.loc[calls, "score"]) == np.sign(llm.loc[calls, "score"])).mean()) if len(calls) else None
ui.kpis([("ETFs scored", len(det)), ("Leaning up", int((det["score"] >= 0.1).sum())),
         ("Leaning down", int((det["score"] <= -0.1).sum())),
         ("The LLM agrees on the direction", "—" if agree is None else f"{agree:.0%}",
          "no LLM score today" if llm.empty else f"on {len(calls)} ETFs both call")])

themes = load_universe().set_index("ticker")["theme"]
table = pd.DataFrame({"ticker": det.index, "theme": det.index.map(themes), "rule": det["score"].values,
                      "llm": llm["score"].reindex(det.index).values,
                      "confidence": det["confidence"].values}).sort_values("rule", ascending=False)
weights = data.weights_on(logs, as_of).get("technical") or technical.starting_weights()

chart, side = st.columns([2, 1], gap="large")
with chart:
    ui.label("Strongest and weakest", "technical_det")
    shown = pd.concat([table.head(TOP), table.tail(TOP)]).drop_duplicates("ticker")
    st.altair_chart(charts.diverging(shown, "rule", "ticker", ["ticker", "theme", "rule", "llm"]), width="stretch")
with side:
    ui.label("Weight of each answer", "learned since 2022")
    answers = pd.DataFrame({"answer": [a.replace("_", " ") for a in weights], "weight": list(weights.values())})
    st.altair_chart(charts.hbars(answers, "weight", "answer", ".0%", ["answer", "weight"]), width="stretch")
    st.page_link("views/learning.py", label="How the weights moved", icon=":material/arrow_forward:")

ui.label("Every ETF", "rule score and the LLM's score, from -1 (avoid) to +1 (buy)")
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "ticker": st.column_config.TextColumn("ETF"), "theme": st.column_config.TextColumn("Theme"),
    "rule": st.column_config.NumberColumn("Rule score", format="%+.2f"),
    "llm": st.column_config.NumberColumn("The LLM's score", format="%+.2f"),
    "confidence": st.column_config.ProgressColumn("Confirmations", min_value=0, max_value=1, format="percent")})



def cell(text: object) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def report() -> str:
    """The day's technical scores as a Markdown document: every ETF, then the reasoning behind each score."""
    out = [f"# Technical agent, {as_of:%a %d %b %Y}", "",
           f"{len(det)} ETFs scored: {int((det['score'] >= 0.1).sum())} leaning up, {int((det['score'] <= -0.1).sum())} "
           f"leaning down. Scores run from -1 (avoid) to +1 (buy).", "", "## Weight of each answer", "",
           "| Answer | Weight |", "|---|---:|"]
    out += [f"| {a.replace('_', ' ')} | {w:.0%} |" for a, w in weights.items()]
    out += ["", "## Every ETF", "", "| ETF | Theme | Rule score | LLM's score | Confirmations |", "|---|---|---:|---:|---:|"]
    for r in table.itertuples():
        llm_score = "n/a" if pd.isna(r.llm) else f"{r.llm:+.2f}"
        out.append(f"| {r.ticker} | {cell(r.theme)} | {r.rule:+.2f} | {llm_score} | {r.confidence:.0%} |")
    out += ["", "## Why, ETF by ETF", ""]
    for t in table["ticker"]:
        out += [f"### {t}", "", f"**Rule** · {cell(det.loc[t, 'rationale'])}"]
        if t in llm.index:
            out += ["", f"**LLM** · {cell(llm.loc[t, 'rationale'])}"]
        out.append("")
    return "\n".join(out)


ui.document_download("The full technical report", report(), f"technical_{as_of}")
pick = st.selectbox("Why", table["ticker"].tolist(), format_func=lambda t: f"Why · {t}", label_visibility="collapsed")
st.markdown(f"**Rule** · {det.loc[pick, 'rationale']}")
if pick in llm.index:
    st.markdown(f"**LLM** · {llm.loc[pick, 'rationale']}")
