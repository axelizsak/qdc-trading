"""How the Portfolio Manager learns which score to trust. Owner: Person 3. No LLM.

Every day each agent gives each ETF a score. Once the following prices are known, code measures how well
each agent's scores ranked the ETFs (rank correlation of score and realised return, plus a hit rate)
and moves the agent weights toward the agents that were right, with limits:
  - evidence counts little at first (shrunk toward zero skill: LEARN_SHRINK),
  - every weight stays between AGENT_WEIGHT_MIN and AGENT_WEIGHT_MAX,
  - the prior weights (equal by default) are the starting point, so no evidence means no change.
A day is judged at the prices of the Belfius rule: the order of day D fills at the last close before D,
and the return is measured from there. Only quotes before `as_of` are used, so nothing leaks from the future.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from agents import portfolio_config as cfg
from data import config as settings
from data.schemas import AGENT_WEIGHTS, SIGNALS, validate_agent_weights, validate_signals


def return_table(prices: pd.DataFrame, as_of) -> pd.DataFrame:
    """Close per ISO date (rows) and ticker (columns), quotes before `as_of` only.
    Uses the close adjusted for dividends when the source gives one, so a dividend is not a fake loss."""
    close = prices["close_eur"].astype(float)
    if "adj_close_eur" in prices:
        close = pd.to_numeric(prices["adj_close_eur"], errors="coerce").fillna(close)
    table = (pd.DataFrame({"date": prices["date"].astype(str), "ticker": prices["ticker"], "close": close})
             .pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index())
    return table[table.index < str(as_of)]


def forward_return(table: pd.DataFrame, decision_date: str, horizon: int) -> pd.Series | None:
    """Return from the fill price of `decision_date` (last close before it) `horizon` quotes ahead.
    None when those quotes are not known yet."""
    before = np.flatnonzero(settings.fills_from(table.index, decision_date))
    if len(before) == 0 or before[-1] + horizon >= len(table):
        return None
    start = before[-1]
    return table.iloc[start + horizon] / table.iloc[start] - 1


def load_history(logs_dir: Path | str) -> pd.DataFrame:
    """Every real signal logged so far (Signals format), without the neutral rows of a failed agent."""
    path = Path(logs_dir) / "signals.csv"
    if not path.exists():
        return pd.DataFrame(columns=SIGNALS)
    signals = validate_signals(pd.read_csv(path))
    failed = signals["rationale"].astype(str).str.startswith(cfg.FAILED_MARKER)
    return signals[~failed].reset_index(drop=True)


def judge_scores(history: pd.DataFrame, prices: pd.DataFrame, as_of,
                 horizons: tuple[int, ...] = cfg.LEARN_HORIZONS) -> pd.DataFrame:
    """One row per (agent, date, horizon) that can be judged: ic (rank correlation of score and return),
    hit_rate (share of calls with the right sign, NA without a call) and n (ETFs)."""
    table = return_table(prices, as_of)
    rows = []
    for (agent, day), group in history.groupby(["agent", history["date"].astype(str)]):
        for horizon in horizons:
            realised = forward_return(table, day, horizon)
            if realised is None:
                continue
            frame = pd.DataFrame({"score": group.set_index("ticker")["score"].astype(float),
                                  "ret": realised}).dropna()
            if len(frame) < cfg.MIN_ETFS_FOR_IC:
                continue
            ic = frame["score"].rank().corr(frame["ret"].rank())
            calls = frame[(frame["score"].abs() >= cfg.HIT_MIN_SCORE) & (frame["ret"] != 0)]
            hit = float((np.sign(calls["score"]) == np.sign(calls["ret"])).mean()) if len(calls) else np.nan
            rows.append({"agent": agent, "date": day, "horizon": horizon,
                         "ic": 0.0 if pd.isna(ic) else float(ic), "hit_rate": hit, "n": len(frame)})
    return pd.DataFrame(rows, columns=["agent", "date", "horizon", "ic", "hit_rate", "n"])


def bound_weights(weights: dict[str, float], low: float = cfg.AGENT_WEIGHT_MIN,
                  high: float = cfg.AGENT_WEIGHT_MAX) -> dict[str, float]:
    """Scale the weights to sum to 1 with each one between `low` and `high` (as close to the ratios as possible)."""
    names = list(weights)
    n = len(names)
    if n == 0:
        return {}
    low, high = min(low, 1 / n), max(high, 1 / n)
    raw = np.array([max(float(weights[a]), 0.0) for a in names])
    if raw.sum() == 0:
        raw = np.ones(n)
    raw = raw / raw.sum()
    lo_l, hi_l = 0.0, (high / raw[raw > 0].min()) * 2
    for _ in range(100):  # sum(clip(l x raw)) grows with l: find the l where it reaches 1
        mid = (lo_l + hi_l) / 2
        if np.clip(mid * raw, low, high).sum() < 1:
            lo_l = mid
        else:
            hi_l = mid
    out = np.clip(hi_l * raw, low, high)
    out = out / out.sum()
    return dict(zip(names, (float(x) for x in out)))


def learn_weights(history: pd.DataFrame, prices: pd.DataFrame, prior: dict[str, float], as_of,
                  horizons: tuple[int, ...] = cfg.LEARN_HORIZONS) -> tuple[dict[str, float], pd.DataFrame]:
    """New agent weights from the evidence so far, and one stats row per agent:
    days judged, mean rank correlation, hit rate, prior and new weight."""
    judged = judge_scores(history, prices, as_of, horizons)
    prior = bound_weights(prior)
    rows, raw = [], {}
    for agent, p in prior.items():
        mine = judged[judged["agent"] == agent]
        days = int(mine["date"].nunique())
        skill = float(mine["ic"].mean()) if len(mine) else 0.0
        shrunk = skill * days / (days + cfg.LEARN_SHRINK)
        raw[agent] = p * float(np.exp(cfg.LEARN_ETA * shrunk))
        hit = mine["hit_rate"].mean() if len(mine) else np.nan
        rows.append({"agent": agent, "days": days, "ic": skill if days else np.nan, "hit_rate": hit, "prior": p})
    weights = bound_weights(raw)
    stats = pd.DataFrame(rows, columns=["agent", "days", "ic", "hit_rate", "prior"])
    stats["weight"] = stats["agent"].map(weights)
    return weights, stats


def update_agent_weights(logs_dir: Path | str, prices: pd.DataFrame, as_of, prior: dict[str, float]) -> dict[str, float]:
    """Learn the weights from <logs>/signals.csv and log them in <logs>/agent_weights.csv for `as_of`
    (the evidence of model re-assessment). Agent names must be in SIGNAL_AGENTS (data/config.py)."""
    logs = Path(logs_dir)
    weights, stats = learn_weights(load_history(logs), prices, prior, as_of)
    new = pd.DataFrame({"date": str(as_of), "agent": stats["agent"], "weight": stats["weight"],
                        "hit_rate": stats["hit_rate"]}, columns=AGENT_WEIGHTS)
    validate_agent_weights(new)
    path = logs / "agent_weights.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["date"].astype(str) != str(as_of)], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return weights


# --- How much to trust the LLM's portfolio -------------------------------------------------------------------
# Each day the PM logs three portfolios (agents/portfolio.py): the code's baseline, the LLM's own weights and
# the final blend. Once the following prices are known, the LLM's portfolio is compared with the baseline: if it
# did better the LLM counts more in the blend, if worse, less (bounded, and shrunk while the evidence is thin).
# TODO for the shared PR: promote PM_PORTFOLIOS to data/schemas.py with a validate_ function and a mock.
PM_PORTFOLIOS = ["date", "source", "ticker", "weight"]
SOURCES = {"baseline", "optimizer", "llm", "final"}


def load_records(logs_dir: Path | str) -> pd.DataFrame:
    path = Path(logs_dir) / "pm_portfolios.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=PM_PORTFOLIOS)


def record_portfolios(logs_dir: Path | str, as_of, portfolios: dict[str, pd.Series]) -> None:
    """Log the day's portfolios (source -> weights), replacing the rows already logged for `as_of`."""
    unknown = set(portfolios) - SOURCES
    if unknown:
        raise ValueError(f"pm_portfolios: unknown source {sorted(unknown)}")
    new = pd.DataFrame([{"date": str(as_of), "source": s, "ticker": t, "weight": float(w)}
                        for s, weights in portfolios.items() for t, w in weights.items()], columns=PM_PORTFOLIOS)
    path = Path(logs_dir) / "pm_portfolios.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["date"].astype(str) != str(as_of)], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def portfolio_returns(records: pd.DataFrame, prices: pd.DataFrame, as_of,
                      horizons: tuple[int, ...] = cfg.LEARN_HORIZONS) -> pd.DataFrame:
    """Return of every logged portfolio from its fill prices, `horizon` quotes ahead (cash earns nothing),
    plus the benchmark ETF as source 'benchmark'. Columns: date, horizon, source, ret."""
    table = return_table(prices, as_of)
    rows = []
    for day, group in records.groupby(records["date"].astype(str)):
        for horizon in horizons:
            realised = forward_return(table, day, horizon)
            if realised is None:
                continue
            for source, weights in group.groupby("source"):
                ret = float((weights["weight"].astype(float).to_numpy()
                             * realised.reindex(weights["ticker"]).fillna(0.0).to_numpy()).sum())
                rows.append({"date": day, "horizon": horizon, "source": source, "ret": ret})
            if cfg.MARKET_TICKER in realised.index and not pd.isna(realised[cfg.MARKET_TICKER]):
                rows.append({"date": day, "horizon": horizon, "source": "benchmark",
                             "ret": float(realised[cfg.MARKET_TICKER])})
    return pd.DataFrame(rows, columns=["date", "horizon", "source", "ret"])


def learn_trust(records: pd.DataFrame, prices: pd.DataFrame, as_of, prior: float = cfg.TRUST_PRIOR,
                anchor: str = cfg.ANCHOR) -> tuple[float, dict]:
    """(trust in the LLM's portfolio, track record). The LLM's portfolio is judged against the anchor code portfolio
    (the baseline when the anchor was not logged). The track record holds the days judged and the mean returns of the
    LLM's portfolio, the anchor, the other code portfolios and the benchmark over the same days."""
    returns = portfolio_returns(records, prices, as_of)
    wide = returns.pivot_table(index=["date", "horizon"], columns="source", values="ret")
    track = {"days": 0, "anchor": anchor, "llm": np.nan, "baseline": np.nan, "optimizer": np.nan,
             "benchmark": np.nan, "advantage": np.nan}
    if "llm" not in wide.columns:
        return prior, track
    anchor = anchor if anchor in wide.columns else "baseline"
    track["anchor"] = anchor
    if anchor not in wide.columns:
        return prior, track
    both = wide.dropna(subset=["llm", anchor])
    if both.empty:
        return prior, track
    days = int(both.index.get_level_values("date").nunique())
    advantage = float((both["llm"] - both[anchor]).mean())
    track.update(days=days, llm=float(both["llm"].mean()), advantage=advantage)
    for name in ("baseline", "optimizer", "benchmark"):
        track[name] = float(both[name].mean()) if name in both and both[name].notna().any() else np.nan
    shrunk = float(np.tanh(advantage / cfg.TRUST_SCALE)) * days / (days + cfg.TRUST_SHRINK)
    return float(np.clip(prior + cfg.TRUST_STEP * shrunk, cfg.TRUST_MIN, cfg.TRUST_MAX)), track
