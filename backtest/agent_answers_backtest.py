"""Backtest of the technical agent alone, from its logged answers (logs/technical_answers.csv) and the learned
weights of its three answers (backtest/results/technical_weights_three_answers.csv), 2022 to today. Code only, no LLM.

Question: is the trend rule alone better than the three answers (trend, mean reversion, momentum) blended with the
learned weights?

Scores from the agent's own function (agents/technical.py, signal_scores), on every ETF of the day; the leveraged,
inverse and cash ETFs are left out only after:
    trend          = suggested_signal: buy +1, watch 0, neutral 0, avoid -1
    mean_reversion = mr_setup: buy +1, avoid -1, otherwise 0
    momentum       = rank of momentum_13612 among the ETFs of the day, scaled to -1 .. +1
    learned blend  = sum of the three with the weights known BEFORE the day

Two tests:
 1. Day by day (the learning rule's own measure): per session, sum(score x (ETF return - average ETF return))
    divided by sum(|score|), next session. t-stat over all sessions.
 2. Portfolio: every HOLD sessions, the TOP ETFs by score at equal weight (ties split evenly), held HOLD sessions,
    prices adjusted for dividends. The result depends on the day the 12-session calendar starts, so the test is run
    for every possible start (0 .. HOLD-1) and the average is reported. Leveraged, inverse and cash ETFs are left out.

Usage:
    python -m backtest.agent_answers_backtest                     # prices fetched by the data agent (network)
    python -m backtest.agent_answers_backtest prices.csv --hold 4 # or a price file: date, ticker, adj_close_eur

The Sharpe ratio here is the return a year over the volatility a year, without the cash rate: it is not comparable
with the system backtest's (backtest/system_backtest.py), which is measured above the overnight rate.

Result on 19 Nov 2026 (hold 12, average of 12 calendars, prices fetched by the data agent): Sharpe 1.01 trend only,
0.56 learned blend, 0.90 all ETFs at equal weight. The best single calendar gives 1.33 vs 0.77, the worst 0.74 vs
0.31 (backtest/README.md, section 6c).
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from agents import technical

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
# The learned weights of the three answers, 3 Jan 2022 to 18 Nov 2026: the technical rows of logs/weights.csv as they
# were in commit 7c8bef7, the last one before the trend rule alone. logs/weights.csv only keeps the trend rule since
# 19 Nov 2026, so the three-answer history is kept here.
THREE_ANSWER_WEIGHTS = ROOT / "backtest" / "results" / "technical_weights_three_answers.csv"
TOP = 5
LEFT_OUT = ["STOXX SHORT", "EURO STOXX 2X", "S&P 500 SHORT", "COPPER LVG", "PALLADIUM LVG", "NASDAQ 3X",
            "OVERNIGHT CASH"]
PRICES_FROM = date(2021, 6, 1)  # the fetched prices start here: the first answers are of 3 Jan 2022
STRATEGIES = ["Trend only", "Mean reversion only", "Momentum only", "Learned blend (the 3 answers)"]


def load_answers() -> pd.DataFrame:
    """The agent's answers with one score column per strategy."""
    answers = pd.read_csv(LOGS / "technical_answers.csv", keep_default_na=False, na_values=[""])
    return with_scores(answers, pd.read_csv(THREE_ANSWER_WEIGHTS))


def with_scores(answers: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """`answers` (Technical answers format) with the three scores and one column per strategy, without LEFT_OUT.
    `weights`: date, name, weight, the learned weights of the three answers."""
    answers = answers.join(technical.signal_scores(answers))  # ranked among every ETF of the day, as the agent does
    answers = answers[~answers["ticker"].isin(LEFT_OUT)].copy()

    # the weights known before day D are the last row strictly before D
    weights = weights.pivot(index="date", columns="name", values="weight").sort_index()
    days = sorted(answers["date"].unique())
    # after the last recorded day the weights stay as they were on 18 Nov 2026
    weights_before = weights.shift(1).reindex(days, method="ffill").fillna(1 / 3)

    blend = pd.Series(0.0, index=answers.index)
    for name in ["trend", "mean_reversion", "momentum"]:
        blend = blend + answers[name] * answers["date"].map(weights_before[name])

    answers["Trend only"] = answers["trend"]
    answers["Mean reversion only"] = answers["mean_reversion"]
    answers["Momentum only"] = answers["momentum"]
    answers["Learned blend (the 3 answers)"] = blend
    answers["date"] = pd.to_datetime(answers["date"])
    return answers


def day_by_day(answers: pd.DataFrame) -> pd.DataFrame:
    """Test 1: what following each score earned over the next session, compared with the average ETF."""
    excess = answers["session_return"] - answers.groupby("date")["session_return"].transform("mean")
    rows = []
    for name in STRATEGIES:
        earned = (answers[name] * excess).groupby(answers["date"]).sum()
        size = answers[name].abs().groupby(answers["date"]).sum()
        per_day = (earned / size).replace([np.inf, -np.inf], np.nan).dropna()
        t_stat = per_day.mean() / per_day.std() * np.sqrt(len(per_day))
        rows.append({"strategy": name, "per day (bp)": round(per_day.mean() * 1e4, 2), "t-stat": round(t_stat, 2)})
    return pd.DataFrame(rows).set_index("strategy")


def top_weights(scores: pd.Series) -> pd.Series:
    """The TOP best scores at equal weight; ETFs tied at the cut-off share the remaining places evenly."""
    cut = scores.sort_values(ascending=False).iloc[TOP - 1]
    above = scores[scores > cut].index
    tied = scores[scores == cut].index
    weights = pd.Series(0.0, index=scores.index)
    weights[above] = 1.0 / TOP
    weights[tied] = (TOP - len(above)) / len(tied) / TOP
    return weights


def portfolio(answers: pd.DataFrame, prices: pd.DataFrame, hold: int, start: int) -> pd.DataFrame:
    """Test 2 for one calendar: one row per holding period, one column per strategy (and all ETFs)."""
    calendar = prices["WORLD"].dropna().index
    filled = prices.reindex(calendar).ffill()  # only to read exit prices; the scores never use filled values
    days = [day for day in calendar if answers["date"].min() <= day <= answers["date"].max()]
    rows = []
    for day in days[start::hold]:
        position = calendar.get_loc(day)
        if position + 1 + hold >= len(calendar):
            break
        entry = calendar[position + 1]
        exit_day = calendar[position + 1 + hold]
        today = answers[answers["date"] == day].set_index("ticker")
        traded = [ticker for ticker in today.index if ticker in prices.columns and pd.notna(prices.at[entry, ticker])]
        today = today.loc[traded]
        if len(today) < 10:
            continue
        returns = filled.loc[exit_day, today.index] / filled.loc[entry, today.index] - 1
        row = {"date": day, "All ETFs, equal weight": returns.mean()}
        for name in STRATEGIES:
            row[name] = (top_weights(today[name]) * returns).sum()
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def sharpe(period_returns: pd.Series, hold: int) -> float:
    periods_a_year = 252 / hold
    growth = (1 + period_returns).prod() ** (periods_a_year / len(period_returns)) - 1
    volatility = period_returns.std() * np.sqrt(periods_a_year)
    return growth / volatility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prices", nargs="?", help="price file with date, ticker, adj_close_eur (default: fetched)")
    parser.add_argument("--hold", type=int, default=12)
    args = parser.parse_args()

    answers = load_answers()
    if args.prices:
        raw = pd.read_csv(args.prices, parse_dates=["date"])
    else:
        from data import data_agent
        raw = data_agent.fetch_prices(PRICES_FROM, date.today()).assign(date=lambda p: pd.to_datetime(p["date"]))
    prices = raw.pivot(index="date", columns="ticker", values="adj_close_eur").sort_index()

    print("1. Day by day, next session, 2022 to today")
    print(day_by_day(answers).to_string())

    columns = ["All ETFs, equal weight"] + STRATEGIES
    results = []
    for start in range(args.hold):
        periods = portfolio(answers, prices, args.hold, start)
        row = {"calendar starts on": periods.index[0].date()}
        for name in columns:
            row[name] = sharpe(periods[name], args.hold)
        results.append(row)
    by_calendar = pd.DataFrame(results).set_index("calendar starts on")
    print("\n2a. Portfolio, one row per calendar (the first one, 3 Jan 2022, gives the 1.33 vs 0.79 first quoted)")
    print(by_calendar[["All ETFs, equal weight", "Trend only", "Learned blend (the 3 answers)"]].round(2).to_string())
    summary = pd.DataFrame({"Sharpe, average": by_calendar.mean(), "lowest": by_calendar.min(),
                            "highest": by_calendar.max()})
    print(f"\n2b. Portfolio: top {TOP} by score, held {args.hold} sessions, every start of the calendar "
          f"({args.hold} calendars), Sharpe without the cash rate")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
