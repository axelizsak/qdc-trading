"""What the ETF analysts read before they judge, calculated by code. Owner: Axel. No LLM.

- The tape: what each ETF's price already did (its return over the last 1, 5 and 20 sessions, its
  volatility), so an analyst can tell news the price already reflects from news it does not.
- The scorecard: an ETF's recent calls next to what its price did over the following sessions, so an
  analyst learns from its own results. The same grades give the hit rate printed in the daily report.

An analyst of day T knows the closes up to T-1, so everything here uses quotes before the day only.
A call of day T is graded like the Portfolio Manager grades a score: from the last close before T to the
close HORIZON_SESSIONS sessions later (fewer while those closes are not known yet).
"""
from datetime import date

import numpy as np
import pandas as pd

from data import config as settings

MIN_CALL = 0.1  # a direction smaller than this is "no call": it is neither right nor wrong
GRADES = ["date", "ticker", "direction", "sessions", "ret", "hit"]


def closes(prices: pd.DataFrame | None, as_of: date) -> pd.DataFrame:
    """Close per ISO date (rows) and ticker (columns), quotes before `as_of` only. Uses the close adjusted
    for dividends when the source gives one, so a dividend is not read as a fall."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    close = prices["close_eur"].astype(float)
    if "adj_close_eur" in prices:
        close = pd.to_numeric(prices["adj_close_eur"], errors="coerce").fillna(close)
    table = (pd.DataFrame({"date": prices["date"].astype(str).str[:10], "ticker": prices["ticker"], "close": close})
             .pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index())
    return table[table.index < as_of.isoformat()]


def _day(iso: str) -> str:
    return f"{date.fromisoformat(iso):%d %b}"


def tape_lines(table: pd.DataFrame) -> dict[str, str]:
    """One line per ETF: 'tape: last close 13 Nov; 1 session +0.4%, 5 sessions -3.2%, 20 sessions +4.1%;
    volatility 28% a year'. An ETF with fewer than two quotes has no line."""
    lines = {}
    for ticker in table.columns:
        quotes = table[ticker].dropna()
        if len(quotes) < 2:
            continue
        moves = [f"{n} session{'s' if n > 1 else ''} {quotes.iloc[-1] / quotes.iloc[-1 - n] - 1:+.1%}"
                 for n in (1, 5, 20) if len(quotes) > n]
        returns = quotes.pct_change().dropna().tail(20)
        vol = f"; volatility {returns.std() * np.sqrt(252):.0%} a year" if len(returns) >= 10 else ""
        lines[ticker] = f"tape: last close {_day(quotes.index[-1])}; {', '.join(moves)}{vol}"
    return lines


def grade_calls(views: pd.DataFrame | None, table: pd.DataFrame, leverage: pd.Series,
                sessions: int | None = None) -> pd.DataFrame:
    """Every past call that can be judged, one row each (GRADES): the return of the ETF from the last close
    before the call to `sessions` sessions later, and whether the call had its sign (None for "no call").
    `leverage` (per ticker, negative for an inverse ETF) turns a view on the underlying into one on the ETF."""
    sessions = settings.HORIZON_SESSIONS if sessions is None else sessions
    rows = []
    if views is None or views.empty or table.empty:
        return pd.DataFrame(columns=GRADES)
    for view in views.itertuples():
        day = str(view.date)[:10]
        if view.ticker not in table.columns:
            continue
        quotes = table[view.ticker].dropna()
        before = int((quotes.index < day).sum())  # quotes the analyst knew: the last one is the starting price
        known = min(sessions, len(quotes) - before)
        if before == 0 or known < 1:
            continue
        ret = float(quotes.iloc[before - 1 + known] / quotes.iloc[before - 1] - 1)
        expected = float(view.direction) * float(leverage.get(view.ticker, 1.0))
        hit = None if abs(float(view.direction)) < MIN_CALL or ret == 0 else bool(np.sign(ret) == np.sign(expected))
        rows.append({"date": day, "ticker": view.ticker, "direction": float(view.direction), "sessions": known,
                     "ret": ret, "hit": hit})
    return pd.DataFrame(rows, columns=GRADES)


def scorecard_lines(grades: pd.DataFrame, calls: int | None = None) -> dict[str, str]:
    """One line per ETF with its latest graded calls, newest first:
    'your calls: 13 Nov -0.20 -> +0.8% over 1 session (wrong); 12 Nov +0.30 -> +1.9% over 2 sessions (right)'."""
    calls = settings.MACRO_SCORECARD_CALLS if calls is None else calls
    lines = {}
    for ticker, mine in grades.groupby("ticker"):
        parts = []
        for g in mine.sort_values("date", ascending=False).head(calls).itertuples():
            verdict = "no call" if g.hit is None or pd.isna(g.hit) else ("right" if g.hit else "wrong")
            parts.append(f"{_day(g.date)} {g.direction:+.2f} -> {g.ret:+.1%} over {g.sessions} "
                         f"session{'s' if g.sessions > 1 else ''} ({verdict})")
        lines[ticker] = "your calls: " + "; ".join(parts)
    return lines


def record(grades: pd.DataFrame, tickers=None) -> tuple[int, int]:
    """(calls that were right, calls with a direction) among `tickers` (default: all)."""
    mine = grades if tickers is None else grades[grades["ticker"].isin(list(tickers))]
    judged = mine[mine["hit"].notna()]
    return int(judged["hit"].astype(bool).sum()), len(judged)


def record_line(grades: pd.DataFrame, tickers=None) -> str:
    right, total = record(grades, tickers)
    if total == 0:
        return "No call of yours on these ETFs can be graded yet."
    return f"Your record on these ETFs so far: {right} of {total} calls right ({right / total:.0%})."
