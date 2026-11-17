"""The model portfolio of each investor profile: fixed rules, no forecast.

A profile gives a share to each sleeve (data/profiles.py). The core sleeves hold the same ETFs for everyone, the
satellites hold the profile's own themes in equal parts. The code also measures what each portfolio has done in
the past, so that the numbers can be challenged (backtest/README.md, section 7).
"""
from datetime import date

import numpy as np
import pandas as pd

from data.profiles import PROFILES, SLEEVE_ETFS
from data.schemas import PROFILE_PORTFOLIOS, validate_profile_portfolios

YEAR = 252  # trading days
CASH = "OVERNIGHT CASH"  # what money earns without risk: the Sharpe ratio's yardstick


def weights(profile: str) -> dict[str, tuple[str, float]]:
    """ticker -> (sleeve, share of the portfolio) for one profile."""
    spec, out = PROFILES[profile], {}
    for sleeve, share in spec["sleeves"].items():
        etfs = ({theme: 1 / len(spec["themes"]) for theme in spec["themes"]} if sleeve == "satellites"
                else SLEEVE_ETFS[sleeve])
        for ticker, part in etfs.items():
            if share * part > 0:
                out[ticker] = (sleeve, round(share * part, 6))
    return out


def build(as_of: date) -> pd.DataFrame:
    """The five model portfolios of the day. Format: Profile portfolios."""
    rows = [(as_of, profile, ticker, weight, sleeve) for profile in PROFILES
            for ticker, (sleeve, weight) in weights(profile).items()]
    return validate_profile_portfolios(pd.DataFrame(rows, columns=PROFILE_PORTFOLIOS))


def track_record(prices: pd.DataFrame, profile: str) -> dict:
    """What the profile's model portfolio did over the price history given, put back to its weights each month:
    yearly return, yearly volatility, worst fall from a peak, Sharpe ratio over cash. From adjusted closes.
    An ETF that was not listed yet counts as cash at 0% until its first quote."""
    closes = prices.pivot(index="date", columns="ticker", values="adj_close_eur").sort_index().ffill()
    held = {ticker: weight for ticker, (_, weight) in weights(profile).items()}
    returns = closes.reindex(columns=list(held)).pct_change(fill_method=None).iloc[1:].fillna(0)
    target = pd.Series(held)
    value, share, month, path = 1.0, target.copy(), None, []
    for day, row in returns.iterrows():
        if pd.Timestamp(day).month != month:  # first session of a month: back to the model's weights
            share, month = target.copy(), pd.Timestamp(day).month
        grown = share * (1 + row)
        value *= grown.sum()
        share = grown / grown.sum()
        path.append(value)
    curve = pd.Series(path, index=returns.index)
    daily = curve.pct_change().fillna(curve.iloc[0] - 1)
    cash = closes.reindex(columns=[CASH])[CASH].pct_change(fill_method=None).reindex(daily.index).fillna(0)
    years = len(curve) / YEAR
    return {"profile": profile, "years": round(years, 1), "return": curve.iloc[-1] ** (1 / years) - 1,
            "volatility": daily.std() * np.sqrt(YEAR), "worst_fall": (curve / curve.cummax() - 1).min(),
            "sharpe": (daily - cash).mean() / (daily - cash).std() * np.sqrt(YEAR)}
