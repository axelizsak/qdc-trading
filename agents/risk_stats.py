"""The calculations behind the Risk Manager's deeper metrics. Owner: Person 3. Pure functions, no LLM, no files.

  risk_contributions   each position's share of the portfolio variance (where the risk really sits)
  effective_bets       how many independent bets the positions amount to, from their correlations
  capture_ratios       how much of the benchmark's rising and falling days the portfolio keeps
  bootstrap_outcomes   the odds of ending the window in a given place, by resampling recent days in blocks
  factor_exposures     the portfolio's sensitivity to each macro factor, from data/macro_exposure.csv
  scenario_pnl         the portfolio's gain or loss in a scenario given as a strength per macro factor
"""
import numpy as np
import pandas as pd

from agents import risk_config as cfg


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Share of the portfolio variance carried by each position (they add up to 1). Zeros if there is no variance."""
    names = [t for t in weights.index if t in cov.index]
    if not names:
        return pd.Series(0.0, index=weights.index)
    w = weights[names].to_numpy(dtype=float)
    matrix = cov.loc[names, names].fillna(0.0).to_numpy()
    variance = float(w @ matrix @ w)
    if variance <= 0:
        return pd.Series(0.0, index=weights.index)
    return pd.Series(w * (matrix @ w) / variance, index=names).reindex(weights.index).fillna(0.0)


def effective_bets(weights: pd.Series, returns: pd.DataFrame) -> float:
    """Effective number of independent bets: the entropy of the variance shares of the principal components of the
    weighted returns. Two uncorrelated equal positions give 2, perfectly correlated ones give 1."""
    names = [t for t in weights.index if t in returns.columns]
    if len(names) <= 1:
        return float(len(names))
    weighted = returns[names].fillna(0.0) * weights[names]
    eigen = np.clip(np.linalg.eigvalsh(weighted.cov().to_numpy()), 0.0, None)
    if eigen.sum() <= 0:
        return float(len(names))
    share = eigen / eigen.sum()
    share = share[share > 1e-12]
    return float(np.exp(-(share * np.log(share)).sum()))


def capture_ratios(port: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """(up capture, down capture): the portfolio's mean return on the benchmark's rising (falling) days over the
    benchmark's mean return on those days. NaN when there are fewer than CAPTURE_MIN_DAYS such days."""
    up, down = bench > 0, bench < 0
    up_capture = port[up].mean() / bench[up].mean() if up.sum() >= cfg.CAPTURE_MIN_DAYS else np.nan
    down_capture = port[down].mean() / bench[down].mean() if down.sum() >= cfg.CAPTURE_MIN_DAYS else np.nan
    return float(up_capture), float(down_capture)


def shrink_drift(series: pd.Series, annual_prior: float = cfg.DRIFT_PRIOR_ANNUAL,
                 prior_days: int = cfg.DRIFT_PRIOR_DAYS) -> pd.Series:
    """The series with its average moved toward a long-run daily drift: n / (n + prior_days) of the way stays the recent
    average, the rest comes from the prior. The spread of the days is untouched."""
    n = len(series)
    target = annual_prior / 252
    return series - series.mean() + target + (series.mean() - target) * n / (n + prior_days)


def bootstrap_outcomes(port: pd.Series, bench: pd.Series | None, horizon: int, draws: int = cfg.BOOTSTRAP_DRAWS,
                       block: int = cfg.BOOTSTRAP_BLOCK, seed: int = cfg.BOOTSTRAP_SEED,
                       shrink: bool = True) -> dict[str, float]:
    """Odds for the return over the next `horizon` days, from resampling the recent daily returns in blocks.

    The portfolio and the benchmark are resampled on the same days, so their link is kept. Returns prob_below_start
    (the portfolio ends below where it is now), median and p5 (its median and 5 % worst return) and, with a benchmark,
    prob_below_bench (it ends behind the benchmark) and median_bench. The average return of both is moved toward a
    long-run drift first (`shrink`), so a recent rally is not projected forward. Empty when there is too little history."""
    n = len(port)
    if horizon < 1 or n < block:
        return {}
    if shrink:
        port = shrink_drift(port)
        bench = shrink_drift(bench) if bench is not None and len(bench) == n else bench
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n - block + 1, size=(draws, blocks))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)[:, :horizon]
    final = np.prod(1 + port.to_numpy(dtype=float)[index], axis=1) - 1
    out = {"prob_below_start": float((final < 0).mean()), "median": float(np.median(final)),
           "p5": float(np.quantile(final, 0.05))}
    if bench is not None and len(bench) == n:
        final_bench = np.prod(1 + bench.to_numpy(dtype=float)[index], axis=1) - 1
        out["prob_below_bench"] = float((final < final_bench).mean())
        out["median_bench"] = float(np.median(final_bench))
    return out


def factor_exposures(weights: pd.Series, table: pd.DataFrame, factors: list[str]) -> pd.Series:
    """Portfolio sensitivity per macro factor: sum over positions of weight x leverage x the ETF's exposure.
    A value of 0.6 means a factor move of +1 moves the portfolio by 0.6 times the shock size."""
    t = table.set_index("ticker") if "ticker" in table.columns else table
    names = [n for n in weights.index if n in t.index]
    if not names:
        return pd.Series(0.0, index=factors)
    scale = weights[names] * t.loc[names, "multiplier"].astype(float)
    return t.loc[names, factors].astype(float).mul(scale, axis=0).sum()


def scenario_pnl(weights: pd.Series, table: pd.DataFrame, shocks: dict[str, float],
                 size: float = cfg.FACTOR_SHOCK) -> tuple[float, pd.Series]:
    """(portfolio return, return contributed by each position) if each macro factor moves by its strength in [-1, 1]
    (1 = a full-strength move, which moves an ETF with exposure 1 by `size`)."""
    t = table.set_index("ticker") if "ticker" in table.columns else table
    names = [n for n in weights.index if n in t.index]
    factors = [f for f in shocks if f in t.columns]
    if not names or not factors:
        return 0.0, pd.Series(dtype=float)
    move = t.loc[names, factors].astype(float).mul(pd.Series(shocks)[factors], axis=1).sum(axis=1)
    contribution = size * weights[names] * t.loc[names, "multiplier"].astype(float) * move
    return float(contribution.sum()), contribution
