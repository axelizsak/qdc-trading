"""One rule for every weight of the system. Owner: Axel. No LLM.

    After each session, every expert's weight is multiplied by e^(LEARNING_SPEED x what following it earned in that
    session). Then the weights are scaled to sum to 1, each between WEIGHT_MIN and WEIGHT_MAX.

Three sets of experts follow it (graph/review_loop.py runs them every morning):
  technical  the technical agent's three answers (trend, mean_reversion, momentum), replayed from LEARNING_START
  agents     the four scores the Portfolio Manager combines, from their first logged score
  portfolio  the LLM's portfolio against the code's, from the first one logged

What following an expert earned in a session:
  - scores: Σ score x (ETF return - average ETF return) / Σ |score|, its calls against the average ETF;
  - a portfolio: its return.
A call of day D is judged on the session an order of D fills into: from the fill price (data/config.py, fills_from)
to the next close. A Saturday, a Sunday and a Monday call all fall on Monday's session; the latest one counts.
Only closes before the day of the review are used, so nothing is judged with a price not known yet.

An expert starts at its starting share (multiplier 1). One that gives no call on a day keeps its multiplier and takes
no share that day. Without evidence the weights are the starting ones.
"""
from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd

from agents import technical
from agents.portfolio_learning import return_table
from data import config as settings
from data.schemas import TECHNICAL_ANSWERS, validate_technical_answers

RESULTS = ["session", "call_date", "name", "result"]


def bound_weights(weights: dict[str, float], low: float | None = None, high: float | None = None) -> dict[str, float]:
    """Scale the weights to sum to 1 with each one between `low` and `high` (as close to the ratios as possible)."""
    low = settings.WEIGHT_MIN if low is None else low
    high = settings.WEIGHT_MAX if high is None else high
    names = list(weights)
    n = len(names)
    if n == 0:
        return {}
    low, high = min(low, 1 / n), max(high, 1 / n)
    raw = np.array([max(float(weights[a]), 0.0) for a in names])
    raw = np.ones(n) / n if raw.sum() == 0 else raw / raw.sum()
    lo_l, hi_l = 0.0, (high / raw[raw > 0].min()) * 2
    for _ in range(100):  # sum(clip(l x raw)) grows with l: find the l where it reaches 1
        mid = (lo_l + hi_l) / 2
        if np.clip(mid * raw, low, high).sum() < 1:
            lo_l = mid
        else:
            hi_l = mid
    out = np.clip(hi_l * raw, low, high)
    return dict(zip(names, (float(x) for x in out / out.sum())))


def starting_weights(agents: list[str]) -> dict[str, float]:
    """The four agents before any evidence: each family (technical, macro) counts the same, split equally between
    its agents in `agents`. A family without an agent gives its share to the other."""
    families = {name: [a for a in members if a in agents] for name, members in settings.AGENT_FAMILIES.items()}
    families = {name: members for name, members in families.items() if members}
    return {a: 1 / len(families) / len(members) for members in families.values() for a in members}


def shares(start: dict[str, float]) -> Callable[[list[str]], dict[str, float]]:
    """Starting weights from fixed shares: those of the experts present, scaled to sum to 1."""
    def of(present: list[str]) -> dict[str, float]:
        mine = {n: start.get(n, 0.0) for n in present}
        total = sum(mine.values())
        return {n: w / total for n, w in mine.items()} if total > 0 else {}
    return of


def _sessions(table: pd.DataFrame, days) -> dict[str, tuple[str, pd.Series]]:
    """For each call day, the session an order of that day fills into and the ETFs' returns over it."""
    out = {}
    for day in days:
        filled = np.flatnonzero(settings.fills_from(table.index, day))
        if len(filled) and filled[-1] + 1 < len(table):
            start = filled[-1]
            out[day] = (table.index[start + 1], table.iloc[start + 1] / table.iloc[start] - 1)
    return out


def _latest(rows: list[dict]) -> pd.DataFrame:
    """One result per session and expert: its latest call before the session."""
    out = pd.DataFrame(rows, columns=RESULTS)
    return out.sort_values("call_date").drop_duplicates(["session", "name"], keep="last").sort_values(
        ["session", "name"]).reset_index(drop=True)


def _score_result(score: pd.Series, ret: pd.Series) -> float:
    """Σ score x (return - average return) / Σ |score| over the ETFs with both; empty without a call."""
    frame = pd.DataFrame({"score": score.astype(float), "ret": ret.astype(float)}).dropna()
    size = frame["score"].abs().sum()
    return float((frame["score"] * (frame["ret"] - frame["ret"].mean())).sum() / size) if size > 0 else np.nan


def call_results(calls: pd.DataFrame, prices: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """What following each expert's scores earned, per session (columns RESULTS). `calls`: date, name, ticker, score.
    The result is empty when the expert made no call that day (every score 0); the expert is still present."""
    table = return_table(prices, as_of)
    calls = calls.assign(date=calls["date"].astype(str).str[:10])
    sessions = _sessions(table, sorted(calls["date"].unique()))
    rows = []
    for (day, name), mine in calls.groupby(["date", "name"]):
        if day in sessions:
            session, returns = sessions[day]
            rows.append({"session": session, "call_date": day, "name": name,
                         "result": _score_result(mine.set_index("ticker")["score"], returns)})
    return _latest(rows)


def answer_results(answers: pd.DataFrame) -> pd.DataFrame:
    """What following each of the technical agent's three answers earned, per session (columns RESULTS), from the
    Technical answers table: its scores (technical.signal_scores) against the session returns stored with them."""
    known = answers["session_return"].notna()
    scores = technical.signal_scores(answers)[known]  # the same sum as _score_result, for every session at once
    ret = answers.loc[known, "session_return"].astype(float)
    day = answers.loc[known, "date"].astype(str)
    excess = ret - ret.groupby(day).transform("mean")
    frames = []
    for name in settings.TECHNICAL_SIGNALS:
        size = scores[name].abs().groupby(day).sum()
        earned = (scores[name] * excess).groupby(day).sum()
        frames.append(pd.DataFrame({"session": size.index, "call_date": size.index, "name": name,
                                    "result": (earned / size.where(size > 0)).values}))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULTS)
    return out.sort_values(["session", "name"]).reset_index(drop=True)[RESULTS]


def technical_answers(prices: pd.DataFrame, features: pd.DataFrame, days: list[date], as_of: date) -> pd.DataFrame:
    """The technical agent's answers on the morning of each of `days`, from the data known then (technical.sheet_before),
    with each ETF's return over the session an order of that morning fills into. Technical answers format."""
    table = return_table(prices, as_of)
    sessions = _sessions(table, [d.isoformat() for d in days])
    prices = prices.assign(date=pd.to_datetime(prices["date"]))
    features = features.assign(date=pd.to_datetime(features["date"]))
    frames = []
    for i, day in enumerate(days):
        sheet = technical.sheet_before(prices, features, day)
        if sheet.empty:
            continue
        returns = sessions[day.isoformat()][1] if day.isoformat() in sessions else pd.Series(dtype=float)
        frames.append(technical.answers(sheet, day).assign(session_return=lambda a: a["ticker"].map(returns)))
        if len(days) > 50 and (i + 1) % 250 == 0:
            print(f"  technical answers: {i + 1} of {len(days)} sessions")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TECHNICAL_ANSWERS)
    return validate_technical_answers(out[TECHNICAL_ANSWERS])


def portfolio_results(portfolios: pd.DataFrame, prices: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """What each portfolio earned per session (columns RESULTS). `portfolios`: date, name, ticker, weight; cash earns 0."""
    table = return_table(prices, as_of)
    portfolios = portfolios.assign(date=portfolios["date"].astype(str).str[:10])
    sessions = _sessions(table, sorted(portfolios["date"].unique()))
    rows = []
    for (day, name), mine in portfolios.groupby(["date", "name"]):
        if day in sessions:
            session, returns = sessions[day]
            ret = returns.reindex(mine["ticker"]).fillna(0.0).to_numpy()
            rows.append({"session": session, "call_date": day, "name": name,
                         "result": float((mine["weight"].astype(float).to_numpy() * ret).sum())})
    return _latest(rows)


def _weights(start: dict[str, float], multiplier: dict[str, float]) -> dict[str, float]:
    return bound_weights({n: s * multiplier[n] for n, s in start.items()})


def daily_weights(results: pd.DataFrame, start: Callable[[list[str]], dict[str, float]], as_of: date,
                  known: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """The rule, session by session. Returns (one row per session and expert: date, name, weight it had for that
    session, result; plus a last set dated `as_of`, the weights for today's decisions over every expert seen so far
    and `known`), and the multipliers (weight / starting weight) to apply to whichever experts score today."""
    multiplier: dict[str, float] = {}
    rows = []
    for session, group in results.groupby("session", sort=True):
        earned = dict(zip(group["name"], group["result"]))
        begin = start(sorted(earned))
        present = [n for n in sorted(earned) if begin.get(n, 0) > 0]  # an expert with no starting share never counts
        begin = start(present)
        for n in present:
            multiplier.setdefault(n, 1.0)
        before = _weights(begin, multiplier)
        rows += [{"date": session, "name": n, "weight": before[n], "result": earned[n]} for n in present]
        if len(present) < 2:
            continue  # an expert alone is compared with nobody: its multiplier stays as it is
        for n in present:
            if pd.notna(earned[n]):
                multiplier[n] *= float(np.exp(settings.LEARNING_SPEED * earned[n]))
        after = _weights(begin, multiplier)
        multiplier.update({n: after[n] / begin[n] for n in present})  # a weight held at a bound stays there
    everyone = sorted(set(multiplier) | set(known or []))
    today = start(everyone)
    everyone = [n for n in everyone if today.get(n, 0) > 0]
    multiplier.update({n: multiplier.get(n, 1.0) for n in everyone})
    now = _weights(start(everyone), multiplier)
    rows += [{"date": as_of.isoformat(), "name": n, "weight": now[n], "result": np.nan} for n in everyone]
    return pd.DataFrame(rows, columns=["date", "name", "weight", "result"]), multiplier


def weights_now(start: dict[str, float], multiplier: dict[str, float]) -> dict[str, float]:
    """Today's weights of the experts in `start` (their starting weights among those present today)."""
    return _weights(start, {n: multiplier.get(n, 1.0) for n in start}) if start else {}


def weights_before(frame: pd.DataFrame, day: date) -> dict[str, float]:
    """The weights a decision of `day` uses: the set of the first date on or after it in a daily_weights frame."""
    dates = frame["date"].astype(str)
    later = dates[dates >= day.isoformat()]
    first = later.min() if len(later) else dates.max()
    rows = frame[dates == first]
    return dict(zip(rows["name"], rows["weight"].astype(float)))


def scoreboard(frame: pd.DataFrame, start: dict[str, float]) -> pd.DataFrame:
    """One row per expert of a daily_weights frame: start, weight now, sessions judged, mean result per session and
    the share of sessions where following it earned something."""
    last = frame["date"].astype(str).max()
    now = frame[frame["date"].astype(str) == last].set_index("name")["weight"]
    judged = frame.dropna(subset=["result"])
    rows = []
    for name in now.index:
        mine = judged[judged["name"] == name]["result"].astype(float)
        rows.append({"name": name, "start": start.get(name, np.nan), "weight": float(now[name]), "sessions": len(mine),
                     "mean_result": float(mine.mean()) if len(mine) else np.nan,
                     "right_share": float((mine > 0).mean()) if len(mine) else np.nan})
    return pd.DataFrame(rows, columns=["name", "start", "weight", "sessions", "mean_result", "right_share"])
