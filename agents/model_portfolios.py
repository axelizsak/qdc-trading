"""The reference portfolio of each strategy (Defensive, Balanced, Opportunity): fixed rules, no forecast.

A strategy gives a share to each sleeve (data/profiles.py). The core sleeves hold the same ETFs for everyone, the
satellites hold the strategy's own themes in equal parts. The code also measures what each portfolio has done in the
past, so that the numbers can be challenged (backtest/README.md, section 7), and the app can chart it.

Run:  python -m agents.model_portfolios            (the track record of each strategy since December 2021)
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data.profiles import SLEEVE_ETFS, STRATEGIES, TRADED
from data.schemas import STRATEGY_CHOICES, STRATEGY_PORTFOLIOS, validate_strategy_choices, validate_strategy_portfolios

YEAR = 252  # trading days
CASH = "OVERNIGHT CASH"  # what money earns without risk: the Sharpe ratio's yardstick
CHOICES = "strategy_choices.csv"


def choices(logs: Path) -> pd.DataFrame:
    """Every choice of the strategy the team trades, oldest first (Strategy choices format); empty before the first."""
    path = Path(logs) / CHOICES
    if not path.exists():
        return pd.DataFrame(columns=STRATEGY_CHOICES)
    return validate_strategy_choices(pd.read_csv(path, dtype=str, keep_default_na=False))


def traded(logs: Path, as_of: date) -> str:
    """The strategy the team trades on `as_of`: the latest choice made for that day or before, else TRADED."""
    made = choices(logs)
    made = made[made["date"] <= as_of.isoformat()]
    return str(made["strategy"].iloc[-1]) if len(made) else TRADED


def choose(logs: Path, day: date, strategy: str, reason: str) -> None:
    """Logs a new choice of the traded strategy, from `day` on. A choice for a day already chosen replaces it."""
    row = pd.DataFrame([[day.isoformat(), strategy, " ".join(str(reason).split())]], columns=STRATEGY_CHOICES)
    made = choices(logs)
    made = pd.concat([made[made["date"] != day.isoformat()], row], ignore_index=True).sort_values("date", kind="stable")
    Path(logs).mkdir(parents=True, exist_ok=True)
    validate_strategy_choices(made).to_csv(Path(logs) / CHOICES, index=False)


def weights(strategy: str) -> dict[str, tuple[str, float]]:
    """ticker -> (sleeve, share of the portfolio) for one strategy."""
    spec, out = STRATEGIES[strategy], {}
    for sleeve, share in spec["sleeves"].items():
        etfs = ({theme: 1 / len(spec["themes"]) for theme in spec["themes"]} if sleeve == "satellites"
                else SLEEVE_ETFS[sleeve])
        for ticker, part in etfs.items():
            if share * part > 0:
                out[ticker] = (sleeve, round(share * part, 6))
    return out


def build(as_of: date) -> pd.DataFrame:
    """The three reference portfolios of the day. Format: Strategy portfolios."""
    rows = [(as_of, strategy, ticker, weight, sleeve) for strategy in STRATEGIES
            for ticker, (sleeve, weight) in weights(strategy).items()]
    return validate_strategy_portfolios(pd.DataFrame(rows, columns=STRATEGY_PORTFOLIOS))


def curve(prices: pd.DataFrame, strategy: str) -> pd.Series:
    """The value of 1 invested in the strategy's reference portfolio at the start of the price history given, day by
    day, put back to its weights at the start of each month. From adjusted closes. An ETF that was not listed yet
    counts as cash at 0% until its first quote."""
    closes = prices.pivot(index="date", columns="ticker", values="adj_close_eur").sort_index().ffill()
    held = {ticker: weight for ticker, (_, weight) in weights(strategy).items()}
    returns = closes.reindex(columns=list(held)).pct_change(fill_method=None).iloc[1:].fillna(0)
    target = pd.Series(held)
    value, share, month, path = 1.0, target.copy(), None, []
    for day, row in returns.iterrows():
        if pd.Timestamp(day).month != month:  # first session of a month: back to the reference weights
            share, month = target.copy(), pd.Timestamp(day).month
        grown = share * (1 + row)
        value *= grown.sum()
        share = grown / grown.sum()
        path.append(value)
    return pd.Series(path, index=returns.index)


def track_record(prices: pd.DataFrame, strategy: str) -> dict:
    """What the strategy's reference portfolio did over the price history given: yearly return, yearly volatility,
    worst fall from a peak, Sharpe ratio over cash."""
    closes = prices.pivot(index="date", columns="ticker", values="adj_close_eur").sort_index().ffill()
    path = curve(prices, strategy)
    daily = path.pct_change().fillna(path.iloc[0] - 1)
    cash = closes.reindex(columns=[CASH])[CASH].pct_change(fill_method=None).reindex(daily.index).fillna(0)
    years = len(path) / YEAR
    return {"strategy": strategy, "years": round(years, 1), "return": path.iloc[-1] ** (1 / years) - 1,
            "volatility": daily.std() * np.sqrt(YEAR), "worst_fall": (path / path.cummax() - 1).min(),
            "sharpe": (daily - cash).mean() / (daily - cash).std() * np.sqrt(YEAR)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="What each strategy's reference portfolio did over a price history.")
    parser.add_argument("--since", type=date.fromisoformat, default=date(2021, 12, 1), help="YYYY-MM-DD")
    args = parser.parse_args(argv)
    from data import data_agent
    prices = data_agent.fetch_prices(args.since, date.today())
    print(pd.DataFrame([track_record(prices, strategy) for strategy in STRATEGIES]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
