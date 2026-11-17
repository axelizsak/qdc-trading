"""The sheet of each ETF, as the customer app shows it: what it did, where its trend stands, what macro means for it.

Code computes every number, from the prices of the data agent, the trend rule of the technical agent and the day's
macro views. The model has one job here, `explain`: put a line of the sheet in plain words, using only the sheet.
"""
from datetime import date

import numpy as np
import pandas as pd

from agents import llm, macro_llm
from agents.technical import trend_answer
from data import config as settings
from data.schemas import ETF_SHEETS, validate_etf_sheets

YEAR = 252  # trading days
RETURNS = {"1 month": 21, "1 year": YEAR, "3 years": 3 * YEAR}  # name -> sessions back
# Trend name -> (days of the average, sessions the return starts back, sessions it ends back): the technical
# agent's rule, a trend is up when the price is above its average and the return is positive, down when neither.
TRENDS = {"short": (20, 21, 0), "medium": (50, 63, 0), "long": (200, YEAR, 21)}
TREND_WORDS = {"short": "1-month", "medium": "3-month", "long": "12-month"}
MACRO_LINES = 3  # macro factors shown on a sheet: the ones that matter most for the ETF
EFFECT = 0.10  # a view times the ETF's exposure beyond this is supportive or a headwind; below it, to watch

EXPLAIN_SYSTEM = """You explain one line of an ETF's sheet to a customer of an investing app, in plain words.
Use only the numbers and words of the sheet you are given: no other figure, no news, no forecast. Never say to buy, sell or hold. Two or three short sentences, no jargon, no markdown. Answer in the language of the question."""


def _returns(closes: pd.Series) -> list[tuple]:
    rows = [("returns", name, closes.iloc[-1] / closes.iloc[-1 - back] - 1, "", "")
            for name, back in RETURNS.items() if len(closes) > back]
    daily = closes.pct_change().dropna().tail(YEAR)
    if len(daily) > 60:
        rows.append(("returns", "volatility", daily.std() * np.sqrt(YEAR), "", "over one year, yearly"))
    return rows


def _trends(closes: pd.Series) -> list[tuple]:
    rows = []
    for name, (days, start, end) in TRENDS.items():
        if len(closes) <= max(days, start):
            continue
        price, average = closes.iloc[-1], closes.tail(days).mean()
        moved = closes.iloc[-1 - end] / closes.iloc[-1 - start] - 1
        state = {"up": "up", "down": "down"}.get(trend_answer(price, average, moved), "neutral")
        rows.append(("trend", name, price / average - 1, state,
                     f"Price above its {days}-day average: {'yes' if price > average else 'no'} ({price / average - 1:+.1%}); "
                     f"{TREND_WORDS[name]} return positive: {'yes' if moved > 0 else 'no'} ({moved:+.1%})"))
    return rows


def _macro(exposure: pd.Series, views: pd.DataFrame) -> list[tuple]:
    rows = []
    for view in views.itertuples():
        weight = float(exposure.get(view.factor, 0)) * float(exposure.get("multiplier", 1))
        if weight:
            effect = view.direction * weight
            state = "supportive" if effect >= EFFECT else "headwind" if effect <= -EFFECT else "watch"
            meaning = settings.MACRO_FACTOR_MEANING[view.factor]
            rows.append((abs(weight), ("macro", view.factor.replace("_", " "), effect, state,
                                       f"Today's view: {view.direction:+.1f} (+1 means {meaning}). {view.rationale}"[:420])))
    return [row for _, row in sorted(rows, key=lambda r: r[0], reverse=True)[:MACRO_LINES]]


def build(prices: pd.DataFrame, views: pd.DataFrame, exposure: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """The sheets of the day, one row per line of a sheet. Format: ETF sheets.

    prices: Prices format, three years back for the 3-year return. views: the macro views to show (Macro views
    format, one day). exposure: Macro exposure format."""
    closes = prices.pivot(index="date", columns="ticker", values="adj_close_eur").sort_index()
    closes = closes[closes.index.astype(str) <= as_of.isoformat()]
    exposures = exposure.set_index("ticker")
    rows = []
    for ticker in closes.columns:
        series = closes[ticker].dropna()
        if len(series) < 2:
            continue
        lines = _returns(series) + _trends(series)
        if ticker in exposures.index:
            lines += _macro(exposures.loc[ticker], views)
        rows += [(as_of, ticker, *line) for line in lines]
    return validate_etf_sheets(pd.DataFrame(rows, columns=ETF_SHEETS))


def explain(sheet: pd.DataFrame, question: str, client=None) -> str:
    """A line of one ETF's sheet in plain words (one call to the app's model), from the sheet's own rows only."""
    lines = "\n".join(f"- {r.block} | {r.name} | {r.value:+.3f} | {r.state} | {r.detail}" for r in sheet.itertuples())
    prompt = f"Sheet of {sheet['ticker'].iloc[0]} (block | line | value | state | detail):\n{lines}\n\nQuestion: {question[:300]}"
    return " ".join(macro_llm.ask(client or llm.client(), macro_llm.Usage(), settings.LLM_APP, EXPLAIN_SYSTEM, prompt,
                                  output_config={"effort": "low"}).split())
