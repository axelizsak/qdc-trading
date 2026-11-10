"""ETF analysts of the Macro/News agent: one Sonnet "hedge-fund quant" per theme of the universe.

Each analyst searches the news for its ETFs and gives every ETF a direction in [-1, 1], a confidence,
a rationale, a counter-argument and its sources. Leveraged and inverse ETFs on an index reuse the view
on their base ETF (e.g. NASDAQ 3X <- NASDAQ); on a commodity, the analyst judges the commodity itself.
The code applies the leverage in both cases. Views are logged in logs/etf_views.csv and blended into
the macro_news score (agents/macro_news.py).
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from agents.macro_llm import UNTRUSTED, Usage, ask, extract_json
from data import config as settings
from data.schemas import ETF_VIEWS, LIST_SEP, validate_etf_views
from data.universe import load_universe

# Leveraged and inverse ETFs (leverage comes from data/macro_exposure.csv).
SAME_AS = {  # on an index already covered by another analyst: reuse that ETF's view
    "STOXX SHORT": "EURO STOXX 50",
    "EURO STOXX 2X": "EURO STOXX 50",
    "NASDAQ 3X": "NASDAQ",
    "S&P 500 SHORT": "S&P 500",
}
UNDERLYING = {  # on a commodity: the analyst judges the commodity itself
    "COPPER LVG": "the copper price",
    "PALLADIUM LVG": "the palladium price",
}

SYSTEM = """You are a quant researcher at a top hedge fund. You cover {theme} ETFs and know what drives them: holdings, sector and macro drivers, flows, positioning and upcoming catalysts. Today is {today}. Your desk trades a EUR portfolio of European-listed ETFs on 7, 10, 14, 17 and 21 December 2026, so your horizon is the next few trading days.

Search the news of the last 7 days that matters for each ETF you are given, then judge each one as you would for your portfolio manager: a direction in [-1, 1] (the sign is up or down, the size how strongly you expect it to move), a confidence in [0, 1], a rationale that cites dated facts with numbers, the strongest counter-argument, and the URLs you relied on. With thin or mixed news, stay near 0 with a low confidence. Never invent a number. {untrusted}

Answer with only a JSON object in a ```json block:
{{"etfs": [{{"ticker": "<ticker exactly as given>", "direction": <number>, "confidence": <number>, "rationale": "...", "counter_argument": "...", "sources": ["<url>"]}}]}}"""


def _etf_lines(etfs: pd.DataFrame) -> str:
    lines = []
    for r in etfs.itertuples():
        line = f"- {r.ticker} · {r.name}"
        if r.ticker in UNDERLYING:
            line += f" · judge {UNDERLYING[r.ticker]} itself: the code applies the leverage"
        lines.append(line)
    return "\n".join(lines)


def _urls(sources) -> list[str]:
    """Source URLs, given as strings or as {"url": ...} objects."""
    urls = [s.get("url") if isinstance(s, dict) else s for s in sources or []]
    return [u.strip() for u in urls if isinstance(u, str) and u.strip().startswith("http")]


def analyse_theme(client, usage: Usage, as_of: date, theme: str, etfs: pd.DataFrame, attempts: int = 2) -> list[dict]:
    """One analyst; asked again once if its answer has no usable view."""
    for _ in range(attempts):
        text = ask(client, usage, settings.LLM_REASONING,
                   SYSTEM.format(theme=theme, today=as_of.isoformat(), untrusted=UNTRUSTED),
                   f"Your ETFs:\n{_etf_lines(etfs)}",
                   tools=[{"type": settings.WEB_SEARCH_TOOL_REASONING, "name": "web_search",
                           "max_uses": settings.MACRO_ANALYST_SEARCHES}],
                   output_config={"effort": "medium"})
        try:
            rows = _parse(text, as_of, set(etfs["ticker"]))
        except ValueError:  # no JSON in the answer
            rows = []
        if rows:
            return rows
    raise RuntimeError("no usable view after two answers")


def _parse(text: str, as_of: date, covered: set[str]) -> list[dict]:
    rows, seen = [], set()
    for view in extract_json(text).get("etfs", []):
        ticker = view.get("ticker")
        urls = _urls(view.get("sources"))
        if ticker not in covered or ticker in seen or not urls:
            continue
        seen.add(ticker)
        rows.append({"date": as_of.isoformat(), "ticker": ticker,
                     "direction": float(np.clip(view.get("direction", 0), -1, 1)),
                     "confidence": float(np.clip(view.get("confidence", 0), 0, 1)),
                     "rationale": view.get("rationale", ""), "counter_argument": view.get("counter_argument", ""),
                     "sources": LIST_SEP.join(dict.fromkeys(urls))})
    return rows


def analyse_all(client, usage: Usage, as_of: date) -> tuple[pd.DataFrame, dict[str, str]]:
    """One analyst per theme, in parallel. Returns (ETF views, {analyst: error} for the ones that failed)."""
    universe = load_universe()
    rows, failures = [], {}
    covered = universe[~universe["ticker"].isin(SAME_AS)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {theme: pool.submit(analyse_theme, client, usage, as_of, theme, etfs)
                for theme, etfs in covered.groupby("theme", sort=False)}
        for theme, job in jobs.items():
            try:
                rows += job.result()
            except Exception as exc:
                failures[f"analyst {theme}"] = f"{type(exc).__name__}: {exc}"
    by_ticker = {r["ticker"]: r for r in rows}
    for ticker, base in SAME_AS.items():
        if base in by_ticker:
            rows.append({**by_ticker[base], "ticker": ticker,
                         "rationale": f"Same underlying as {base}: {by_ticker[base]['rationale']}"})
    return validate_etf_views(pd.DataFrame(rows, columns=ETF_VIEWS)), failures


def blend(signals: pd.DataFrame, views: pd.DataFrame, exposure: pd.DataFrame,
          share: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """macro_news score = (1 - share) x factor score + share x analyst score, per ETF.

    The analyst score is the view times the ETF's leverage (from the exposure table), clipped to [-1, 1].
    An ETF without an analyst view keeps its factor score. Returns (blended signals, comparison table).
    """
    share = settings.MACRO_ANALYST_SHARE if share is None else share
    leverage = exposure.set_index("ticker")["multiplier"].astype(float)
    by_ticker = views.set_index("ticker")
    out, table = signals.copy(), []
    for i, row in out.iterrows():
        factor_score = float(row["score"])
        analyst = None
        if row["ticker"] in by_ticker.index:
            view = by_ticker.loc[row["ticker"]]
            analyst = float(np.clip(leverage[row["ticker"]] * view["direction"], -1, 1))
            out.at[i, "score"] = round((1 - share) * factor_score + share * analyst, 4)
            out.at[i, "confidence"] = round((1 - share) * float(row["confidence"]) + share * view["confidence"], 4)
            out.at[i, "rationale"] = f"{row['rationale']} | analyst {analyst:+.2f}: {str(view['rationale'])[:160]}"
        table.append({"ticker": row["ticker"], "factors": factor_score, "analyst": analyst,
                      "final": float(out.at[i, "score"])})
    return out, pd.DataFrame(table)
