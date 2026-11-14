"""Grading the Risk Manager itself. Owner: Person 3. No LLM.

Every assessment is logged with the weights it was made on and the forecasts it contained. Once the following prices
are known, the forecasts are compared with what happened:
  volatility        the forecast annualised volatility against the volatility realised over the next GRADE_DAYS quotes
  5-day floor       how often the portfolio's 5-day return fell below the logged 5 % worst case (about 1 in 20 is right)
  odds              the logged chances of ending below the start and below the benchmark against what happened
                    (judged once the window of that day is over)
The volatility comparison also corrects the next forecast: scale = 1 + (mean realised / forecast - 1), shrunk toward 1
while the evidence is thin and kept between CAL_MIN and CAL_MAX. Every correction is logged in risk_calibration.csv,
so the re-assessment of the Risk Manager is on record. Only quotes before `as_of` are used, so nothing leaks from the future.

Files, all in <logs>:  risk_reports.csv (the report rows, RISK_REPORT format), risk_forecasts.csv (RISK_LOG: the weights and
forecasts of each assessment), risk_calibration.csv (the learned correction).
TODO for the shared PR: promote RISK_LOG and CALIBRATION to data/schemas.py with validate_ functions and mocks.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from agents import portfolio_learning
from agents import risk_config as cfg
from agents.portfolio_config import MARKET_TICKER
from data.schemas import RISK_REPORT

RISK_LOG = ["date", "kind", "name", "value"]
CALIBRATION = ["date", "vol_scale", "days_graded", "mean_ratio", "floor_breaches", "floor_days"]
FORECASTS = ["vol_raw", "p5_5d", "prob_below_start", "prob_below_bench", "horizon_days"]


def _replace(path: Path, new: pd.DataFrame, day: str) -> None:
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["date"].astype(str) != day], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def record_risk(logs_dir: Path | str, as_of, report: pd.DataFrame, weights: pd.Series, forecasts: dict[str, float]) -> None:
    """Log one assessment: its report rows, the weights it was made on and its forecasts. A rerun of the day replaces it."""
    logs, day = Path(logs_dir), str(as_of)
    _replace(logs / "risk_reports.csv", report[RISK_REPORT].assign(date=day), day)
    rows = [{"date": day, "kind": "weight", "name": t, "value": float(w)} for t, w in weights.items()]
    rows += [{"date": day, "kind": "forecast", "name": k, "value": float(v)} for k, v in forecasts.items()
             if k in FORECASTS and not pd.isna(v)]
    _replace(logs / "risk_forecasts.csv", pd.DataFrame(rows, columns=RISK_LOG), day)


def load_log(logs_dir: Path | str) -> pd.DataFrame:
    path = Path(logs_dir) / "risk_forecasts.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=RISK_LOG)


def _path(table: pd.DataFrame, weights: pd.Series, day: str, quotes: int) -> pd.Series | None:
    """Daily returns of fixed `weights` over the `quotes` quotes after the fill price of `day` (the last close before it).
    Fewer quotes are returned when fewer are known; None when there are none."""
    before = np.flatnonzero(table.index < day)
    if len(before) == 0 or before[-1] + 1 >= len(table):
        return None
    start = before[-1]
    stop = min(start + quotes, len(table) - 1)
    window = table.iloc[start:stop + 1].pct_change(fill_method=None).iloc[1:]
    names = [t for t in weights.index if t in window.columns]
    return (window[names].fillna(0.0) * weights[names]).sum(axis=1)


def grade_volatility(logs_dir: Path | str, prices: pd.DataFrame, as_of) -> pd.DataFrame:
    """One row per logged day that can be judged: vol_pred (the raw forecast), vol_real, ratio (real / forecast),
    p5_5d, ret_5d and breach (the 5-day return fell below the 5 % worst case)."""
    log = load_log(logs_dir)
    table = portfolio_learning.return_table(prices, as_of)
    rows = []
    for day, group in log.groupby(log["date"].astype(str)):
        weights = group[group["kind"] == "weight"].set_index("name")["value"].astype(float)
        forecast = group[group["kind"] == "forecast"].set_index("name")["value"].astype(float)
        path = _path(table, weights, day, cfg.GRADE_DAYS)
        if path is None or len(path) < cfg.MIN_GRADE_QUOTES or "vol_raw" not in forecast:
            continue
        real = float(path.std() * np.sqrt(252))
        row = {"date": day, "vol_pred": float(forecast["vol_raw"]), "vol_real": real,
               "ratio": real / forecast["vol_raw"] if forecast["vol_raw"] > 0 else np.nan,
               "p5_5d": float(forecast.get("p5_5d", np.nan)), "ret_5d": np.nan, "breach": np.nan}
        if len(path) >= 5:
            row["ret_5d"] = float(np.prod(1 + path.iloc[:5]) - 1)
            row["breach"] = float(row["ret_5d"] < row["p5_5d"]) if not pd.isna(row["p5_5d"]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=["date", "vol_pred", "vol_real", "ratio", "p5_5d", "ret_5d", "breach"])


def grade_outcomes(logs_dir: Path | str, prices: pd.DataFrame, as_of) -> pd.DataFrame:
    """The logged odds against what happened, for the days whose window is over (the quotes reach `horizon_days` ahead).
    Columns: date, pred_below_start, below_start (0 or 1), pred_below_bench, below_bench, horizon_days."""
    log = load_log(logs_dir)
    table = portfolio_learning.return_table(prices, as_of)
    rows = []
    for day, group in log.groupby(log["date"].astype(str)):
        weights = group[group["kind"] == "weight"].set_index("name")["value"].astype(float)
        forecast = group[group["kind"] == "forecast"].set_index("name")["value"].astype(float)
        if "horizon_days" not in forecast or "prob_below_start" not in forecast:
            continue
        horizon = int(forecast["horizon_days"])
        path = _path(table, weights, day, horizon)
        if path is None or len(path) < horizon:
            continue
        final = float(np.prod(1 + path) - 1)
        row = {"date": day, "pred_below_start": float(forecast["prob_below_start"]), "below_start": float(final < 0),
               "pred_below_bench": float(forecast.get("prob_below_bench", np.nan)), "below_bench": np.nan,
               "horizon_days": horizon}
        if MARKET_TICKER in table.columns and "prob_below_bench" in forecast:
            start = np.flatnonzero(table.index < day)[-1]
            bench = float(table[MARKET_TICKER].iloc[start + horizon] / table[MARKET_TICKER].iloc[start] - 1)
            row["below_bench"] = float(final < bench)
        rows.append(row)
    return pd.DataFrame(rows, columns=["date", "pred_below_start", "below_start", "pred_below_bench", "below_bench",
                                       "horizon_days"])


def update_calibration(logs_dir: Path | str, prices: pd.DataFrame, as_of) -> float:
    """Grade the past forecasts and log the volatility correction for `as_of`. Returns the scale (1.0 without evidence).

    scale = 1 + (mean realised / forecast - 1) x n / (n + CAL_SHRINK), clipped to CAL_MIN..CAL_MAX, with n the judged days;
    below GRADE_MIN_DAYS judged days no correction is made."""
    grades = grade_volatility(logs_dir, prices, as_of)
    ratios = grades["ratio"].dropna().clip(0.25, 4.0)
    n = len(ratios)
    scale = 1.0
    if n >= cfg.GRADE_MIN_DAYS:
        scale = float(np.clip(1 + (ratios.mean() - 1) * n / (n + cfg.CAL_SHRINK), cfg.CAL_MIN, cfg.CAL_MAX))
    breaches = grades["breach"].dropna()
    row = pd.DataFrame([{"date": str(as_of), "vol_scale": scale, "days_graded": n,
                         "mean_ratio": float(ratios.mean()) if n else np.nan,
                         "floor_breaches": int(breaches.sum()), "floor_days": len(breaches)}], columns=CALIBRATION)
    if Path(logs_dir).exists():
        _replace(Path(logs_dir) / "risk_calibration.csv", row, str(as_of))
    return scale


def grading_summary(logs_dir: Path | str, prices: pd.DataFrame, as_of) -> str:
    """One paragraph on how the Risk Manager's forecasts have done so far (empty when nothing can be judged yet)."""
    grades = grade_volatility(logs_dir, prices, as_of)
    outcomes = grade_outcomes(logs_dir, prices, as_of)
    parts = []
    ratios = grades["ratio"].dropna()
    if len(ratios):
        parts.append(f"Volatility forecasts judged on {len(ratios)} day(s): realised volatility was on average "
                     f"{ratios.mean():.2f} times the forecast.")
    breaches = grades["breach"].dropna()
    if len(breaches):
        parts.append(f"The 5-day worst case was breached on {int(breaches.sum())} of {len(breaches)} day(s) (about 5% is right).")
    if len(outcomes):
        parts.append(f"{len(outcomes)} window(s) judged: the portfolio ended below its start {outcomes['below_start'].mean():.0%} "
                     f"of the time against {outcomes['pred_below_start'].mean():.0%} forecast"
                     + (f", and behind the benchmark {outcomes['below_bench'].mean():.0%} against "
                        f"{outcomes['pred_below_bench'].mean():.0%} forecast." if outcomes["below_bench"].notna().any() else "."))
    return " ".join(parts)
