"""Loaders for the cockpit. Each returns (data, is_mock): real logs first, mock data when there are none."""
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data import config as settings
from data import mock
from data.schemas import (
    PRICES, SIGNALS, validate_etf_views, validate_evidence, validate_macro_views, validate_order_log, validate_prices,
    validate_signals, validate_weights,
)
from graph import pipeline


def _csv(path: Path, validate) -> pd.DataFrame | None:
    return validate(pd.read_csv(path)) if path.exists() else None


def _on(df: pd.DataFrame, day: date) -> pd.DataFrame:
    return df[df["date"].astype(str) == day.isoformat()]


def day_signals(as_of: date, logs: Path, state: dict) -> tuple[pd.DataFrame, bool]:
    if state.get("signals"):
        return validate_signals(pd.DataFrame(state["signals"], columns=SIGNALS)), False
    logged = _csv(logs / "signals.csv", validate_signals)
    if logged is not None and len(_on(logged, as_of)):
        return _on(logged, as_of), False
    return mock.mock_signals(as_of.isoformat()), True


def day_prices(as_of: date, state: dict) -> tuple[pd.DataFrame, bool]:
    if state.get("prices"):
        return validate_prices(pd.DataFrame(state["prices"], columns=PRICES)), False
    return mock.mock_prices(end=as_of.isoformat()), True


def order_log(logs: Path) -> pd.DataFrame | None:
    return _csv(logs / "orders.csv", validate_order_log)


def macro_views(logs: Path) -> tuple[pd.DataFrame, bool]:
    views = _csv(logs / "macro_views.csv", validate_macro_views)
    return (views, False) if views is not None and len(views) else (mock.mock_macro_views(), True)


def evidence(logs: Path) -> tuple[pd.DataFrame, bool]:
    ev = _csv(logs / "evidence.csv", validate_evidence)
    return (ev, False) if ev is not None and len(ev) else (mock.mock_evidence(), True)


def etf_views(logs: Path) -> tuple[pd.DataFrame, bool]:
    views = _csv(logs / "etf_views.csv", validate_etf_views)
    return (views, False) if views is not None and len(views) else (mock.mock_etf_views(), True)


def weights(logs: Path) -> tuple[pd.DataFrame, bool]:
    table = _csv(logs / "weights.csv", validate_weights)
    return (table, False) if table is not None and len(table) else (mock.mock_weights(), True)


def macro_report(day: date, *folders: Path) -> str | None:
    for folder in folders:
        path = folder / "macro" / f"{day.isoformat()}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


def macro_cost(report: str | None) -> str | None:
    match = re.search(r"Cost of this run: (\$[\d.,]+)", report or "")
    return match.group(1) if match else None


def last_close(prices: pd.DataFrame, as_of: date) -> pd.Series:
    """Latest close on or before `as_of`, per ticker."""
    p = prices[prices["date"].astype(str) <= as_of.isoformat()].sort_values("date")
    return p.groupby("ticker")["close_eur"].last()


BENCHMARK_LABEL = "MSCI World"


def portfolio_history(log: pd.DataFrame | None, prices: pd.DataFrame) -> pd.DataFrame:
    """Daily value of the portfolio and of €START_CAPITAL in the benchmark, from the fill of the first order on.

    Columns: date, series ("Portfolio" | BENCHMARK_LABEL), value_eur. An order fills at the last close before
    its day, so both lines start at €START_CAPITAL on that close.
    """
    empty = pd.DataFrame(columns=["date", "series", "value_eur"])
    if log is None:
        return empty
    done = log[log["decision"] != "rejected"].copy()
    if done.empty:
        return empty
    done["date"] = done["date"].astype(str)
    done["signed"] = done["units"].astype(float).where(done["side"] == "BUY", -done["units"].astype(float))
    quotes = (prices.assign(date=prices["date"].astype(str))
              .pivot(index="date", columns="ticker", values="close_eur").sort_index())
    before_first = quotes.index[quotes.index < done["date"].min()]  # the chart starts the day before the first order
    if len(before_first) == 0 or settings.BENCHMARK not in quotes.columns:
        return empty
    done["fill"] = [quotes.loc[settings.fills_from(quotes.index, d), t].dropna().iloc[-1]
                    for d, t in zip(done["date"], done["ticker"])]
    close = quotes.ffill()
    close = close[close.index >= before_first[-1]]
    rows = []
    for day in close.index:
        so_far = done[done["date"] <= day]
        cash = settings.START_CAPITAL_EUR - float((so_far["signed"] * so_far["fill"]).sum())
        units = so_far.groupby("ticker")["signed"].sum()
        value = cash + float((units * close.loc[day, units.index]).sum())
        rows.append({"date": day, "series": "Portfolio", "value_eur": value})
    benchmark = close[settings.BENCHMARK]
    for day in close.index:
        rows.append({"date": day, "series": BENCHMARK_LABEL,
                     "value_eur": settings.START_CAPITAL_EUR * benchmark[day] / benchmark.iloc[0]})
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out


def positions(as_of: date, logs: Path, prices: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Holdings valued at the latest close, after the orders of `as_of`, and cash."""
    holdings, cash = pipeline.load_book(logs, prices, as_of, with_the_day=True)
    close = last_close(prices, as_of)
    out = holdings.copy()
    out["units"] = out["units"].astype(float)
    out["last_close_eur"] = out["ticker"].map(close)
    out["value_eur"] = out["units"] * out["last_close_eur"]
    total = out["value_eur"].sum() + cash
    out["weight"] = out["value_eur"] / total if total else np.nan
    return out.sort_values("value_eur", ascending=False), cash


def weights_on(logs: Path, day: date) -> dict[str, dict[str, float]]:
    """The weights the learning step set for `day` (or the latest before it), per level: technical, agents, portfolio."""
    table = _csv(logs / "weights.csv", validate_weights)
    if table is None:
        return {}
    table = table[table["date"].astype(str) <= day.isoformat()]
    out = {}
    for level, rows in table.groupby("level", sort=False):
        last = rows[rows["date"].astype(str) == rows["date"].astype(str).max()]
        out[level] = dict(zip(last["name"], last["weight"].astype(float)))
    return out


BACKTEST = settings.ROOT / "backtest" / "results"
CURVES = {"current": "The system", "trend_only": "Trend rule alone"}


def backtest_curves() -> pd.DataFrame | None:
    """The daily backtest's value of €START_CAPITAL in the system, in the trend rule alone and in the benchmark.
    Columns: date, series, value_eur. None before the backtest was run."""
    path = BACKTEST / "system_backtest_daily.csv"
    if not path.exists():
        return None
    daily = pd.read_csv(path, usecols=["variant", "date", "value", "bench"])
    daily = daily[daily["variant"].isin(CURVES)]
    curves = daily.assign(series=daily["variant"].map(CURVES))[["date", "series", "value"]]
    bench = daily[daily["variant"] == "current"].assign(series=BENCHMARK_LABEL)[["date", "series", "bench"]]
    out = pd.concat([curves, bench.rename(columns={"bench": "value"})], ignore_index=True)
    return out.rename(columns={"value": "value_eur"}).assign(date=lambda d: pd.to_datetime(d["date"]))


def backtest_summary() -> pd.DataFrame | None:
    """One row per variant of the daily backtest (backtest/results/system_backtest_daily_summary.csv)."""
    path = BACKTEST / "system_backtest_daily_summary.csv"
    return pd.read_csv(path) if path.exists() else None


def md_table(markdown: str, first_header: str) -> pd.DataFrame:
    """The Markdown table whose header starts with `first_header` (e.g. '| Area'), as a DataFrame of strings."""
    lines = markdown.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(first_header)), None)
    if start is None:
        return pd.DataFrame()
    cells = lambda line: [c.strip() for c in line.strip().strip("|").split("|")]  # noqa: E731
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append(cells(line))
    return pd.DataFrame(rows, columns=cells(lines[start]))


def section(markdown: str, heading: str) -> str:
    """The text under a Markdown heading (e.g. '### To consider'), up to the next heading."""
    lines = markdown.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == heading), None)
    if start is None:
        return ""
    out = []
    for line in lines[start + 1:]:
        if line.startswith("#"):
            break
        out.append(line)
    return "\n".join(out).strip()


def memo_summary(memo: str) -> str:
    """The Portfolio Manager's own summary: the first paragraph of its memo that is not a title or an id."""
    for paragraph in memo.split("\n\n"):
        text = paragraph.strip()
        if text and not text.startswith(("#", "Recommendation ", "|")):
            return text
    return ""
