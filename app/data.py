"""Loaders for the cockpit. Each returns (data, is_mock): real logs first, mock data when there are none."""
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data import config as settings
from data import mock
from data.schemas import (
    PRICES, SIGNALS, validate_agent_weights, validate_etf_views, validate_evidence, validate_macro_views, validate_order_log,
    validate_prices, validate_signals, validate_source_registry,
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


def source_registry(logs: Path) -> tuple[pd.DataFrame, bool]:
    reg = _csv(logs / "source_registry.csv", validate_source_registry)
    return (reg, False) if reg is not None and len(reg) else (mock.mock_source_registry(), True)


def agent_weights(logs: Path) -> tuple[pd.DataFrame, bool]:
    weights = _csv(logs / "agent_weights.csv", validate_agent_weights)
    return (weights, False) if weights is not None and len(weights) else (mock.mock_agent_weights(), True)


def lessons(logs: Path) -> str | None:
    path = logs / "lessons.md"
    return path.read_text() if path.exists() else None


def macro_report(day: date, *folders: Path) -> str | None:
    for folder in folders:
        path = folder / "macro" / f"{day.isoformat()}.md"
        if path.exists():
            return path.read_text()
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
    """Daily value of the portfolio and of €START_CAPITAL in the benchmark, from the first executed order on.

    Columns: date, series ("Portfolio" | BENCHMARK_LABEL), value_eur. Fills at the close of the order date.
    """
    if log is None:
        return pd.DataFrame(columns=["date", "series", "value_eur"])
    done = log[log["decision"] != "rejected"].copy()
    if done.empty:
        return pd.DataFrame(columns=["date", "series", "value_eur"])
    done["date"] = done["date"].astype(str)
    done["signed"] = done["units"].astype(float).where(done["side"] == "BUY", -done["units"].astype(float))
    close = (prices.assign(date=prices["date"].astype(str))
             .pivot(index="date", columns="ticker", values="close_eur").sort_index().ffill())
    close = close[close.index >= done["date"].min()]
    rows = []
    for day in close.index:
        so_far = done[done["date"] <= day]
        fills = [close.loc[:d, t].dropna().iloc[-1] for d, t in zip(so_far["date"], so_far["ticker"])]
        cash = settings.START_CAPITAL_EUR - float((so_far["signed"] * fills).sum())
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
    holdings, cash = pipeline.load_book(logs, prices, as_of + timedelta(days=1))
    close = last_close(prices, as_of)
    out = holdings.copy()
    out["units"] = out["units"].astype(float)
    out["last_close_eur"] = out["ticker"].map(close)
    out["value_eur"] = out["units"] * out["last_close_eur"]
    total = out["value_eur"].sum() + cash
    out["weight"] = out["value_eur"] / total if total else np.nan
    return out.sort_values("value_eur", ascending=False), cash
