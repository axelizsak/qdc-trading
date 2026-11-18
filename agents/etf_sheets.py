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

# The name of each macro line on screen: a sheet names a factor with spaces for underscores ("risk appetite").
MACRO_NAMES = {factor.replace("_", " "): label for factor, label in settings.MACRO_FACTOR_LABELS.items()}
TREND_MONTHS = {"short": 1, "medium": 3, "long": 12}
TREND_STATE_WORDS = {"up": "Rising", "neutral": "No clear trend", "down": "Falling"}
LEAN = 0.15  # a news analyst's view beyond this leans one way; within it, no clear edge

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


def label(line: dict) -> str:
    """How a line of a sheet is named on screen."""
    if line["block"] == "macro":
        return MACRO_NAMES.get(line["name"], line["name"].capitalize())
    if line["block"] == "trend":
        months = TREND_MONTHS.get(line["name"])
        return f"{months} month{'s' if months > 1 else ''}" if months else line["name"].capitalize()
    return line["name"].capitalize()


def _months(names: list[str]) -> str:
    numbers = sorted(TREND_MONTHS[name] for name in names)
    words = [str(n) for n in numbers]
    return (", ".join(words[:-1]) + " and " + words[-1] if len(words) > 1 else words[0]) + \
        (" month" if numbers == [1] else " months")


def _tone(good: int, bad: int) -> str:
    return "good" if good > bad else "bad" if bad > good else "wait"


def reading(lines: list[dict], news: dict | None = None) -> list[dict]:
    """What the agents see in an ETF today, in up to three plain lines (trend, macro, news), each with a tone: good,
    wait or bad. Written by code from the sheet's own lines and the news analyst's view: no model, no new number.

    lines: the sheet's lines, each {"block", "name", "state"}. news: the analyst's view, {"direction", "confidence",
    "rationale"}, or None."""
    out = []
    trends = {line["name"]: line["state"] for line in lines if line["block"] == "trend" and line["name"] in TREND_MONTHS}
    if trends:
        parts = [f"{TREND_STATE_WORDS[state]} over {_months([n for n, s in trends.items() if s == state])}"
                 for state in ("up", "neutral", "down") if state in trends.values()]
        text = parts[0] + (", " + ", ".join(p[0].lower() + p[1:] for p in parts[1:]) if len(parts) > 1 else "") + "."
        states = list(trends.values())
        out.append({"topic": "Trend", "tone": _tone(states.count("up"), states.count("down")), "text": text})
    macro = [line for line in lines if line["block"] == "macro"]
    if macro:
        helps = [label(line) for line in macro if line["state"] == "supportive"]
        hurts = [label(line) for line in macro if line["state"] == "headwind"]
        parts = ([f"Tailwind{'s' if len(helps) > 1 else ''}: {', '.join(helps)}."] if helps else []) + \
                ([f"Headwind{'s' if len(hurts) > 1 else ''}: {', '.join(hurts)}."] if hurts else [])
        text = " ".join(parts) if parts else "No strong macro force either way today."
        out.append({"topic": "Macro", "tone": _tone(len(helps), len(hurts)), "text": text})
    if news:
        direction = float(news["direction"])
        lean = "leans positive" if direction >= LEAN else "leans negative" if direction <= -LEAN else "sees no clear edge"
        out.append({"topic": "News", "tone": "good" if direction >= LEAN else "bad" if direction <= -LEAN else "wait",
                    "text": f"Our news analyst {lean} for the next few days ({float(news['confidence']):.0%} sure)."})
    return out


def explain(sheet: pd.DataFrame, question: str, client=None) -> str:
    """A line of one ETF's sheet in plain words (one call to the app's model), from the sheet's own rows only."""
    lines = "\n".join(f"- {r.block} | {r.name} | {r.value:+.3f} | {r.state} | {r.detail}" for r in sheet.itertuples())
    prompt = f"Sheet of {sheet['ticker'].iloc[0]} (block | line | value | state | detail):\n{lines}\n\nQuestion: {question[:300]}"
    return " ".join(macro_llm.ask(client or llm.client(), macro_llm.Usage(), settings.LLM_APP, EXPLAIN_SYSTEM, prompt,
                                  output_config={"effort": "low"}).split())
