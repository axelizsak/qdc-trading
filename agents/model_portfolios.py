"""The reference portfolio of each strategy (Defensive, Balanced, Opportunity): what the app shows a customer.

Each strategy starts from a fixed mix (`weights`): a share per sleeve (data/profiles.py), the same core ETFs for
everyone, the strategy's own themes in equal parts. The share of each sleeve never moves: it is the risk level, the
table customers and the jury see. Inside each sleeve, the reference follows our engine: when the team signs off a
day in the cockpit, the code's portfolio of the Portfolio Manager is built for each strategy, from the same agents'
scores and the same learned weights, inside the strategy's limits (no LLM: same input, same output), and each
sleeve is split half as in the fixed mix, half as in the engine's portfolio (`follow`). It becomes a new version of
the reference (`publish`) only when at least PUBLISH_TURNOVER of it would change, so customers are not bothered by
noise; the app then offers them the orders that follow it, which they decide on. The versions are logged
(logs/strategy_portfolios.csv); before the first one, the fixed mix is the reference.

The code also measures what each mix did in the past, so the numbers can be challenged (backtest/README.md, section
7), and the team chooses here the strategy it trades in the competition (`traded`, `choose`).

Run:  python -m agents.model_portfolios            (the track record of each strategy since December 2021)
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data.profiles import COMPLEX_FAMILIES, SLEEVE_ETFS, STRATEGIES, TRADED, sleeve_of
from data.schemas import STRATEGY_CHOICES, STRATEGY_PORTFOLIOS, validate_strategy_choices, validate_strategy_portfolios

YEAR = 252  # trading days
CASH = "OVERNIGHT CASH"  # what money earns without risk: the Sharpe ratio's yardstick; also where idle money sits
CHOICES = "strategy_choices.csv"
VERSIONS = "strategy_portfolios.csv"
PUBLISH_TURNOVER = 0.10  # a new version only when at least this share of the reference would change
MIN_LINE = 0.02  # a line below this share of the reference is shared out in its sleeve: no dust for customers
ENGINE_SHARE = 0.5  # inside a sleeve, this part of the split follows the engine, the rest stays as in the fixed mix


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


def versions(logs: Path) -> pd.DataFrame:
    """Every published version of every reference, oldest first (Strategy portfolios format); empty before the first."""
    path = Path(logs) / VERSIONS
    if not path.exists():
        return pd.DataFrame(columns=STRATEGY_PORTFOLIOS)
    found = pd.read_csv(path)
    return validate_strategy_portfolios(found.assign(date=found["date"].astype(str))).sort_values("date", kind="stable")


def current(logs: Path | None, strategy: str, as_of: date | None = None) -> tuple[dict[str, tuple[str, float]], str | None]:
    """The strategy's reference on `as_of` (default: the latest): ticker -> (sleeve, share), and the date of its
    version (None: the fixed starting mix, nothing published yet)."""
    found = versions(logs) if logs is not None else pd.DataFrame(columns=STRATEGY_PORTFOLIOS)
    found = found[found["strategy"] == strategy]
    if as_of is not None:
        found = found[found["date"] <= as_of.isoformat()]
    if found.empty:
        return weights(strategy), None
    day = found["date"].max()
    rows = found[found["date"] == day]
    return {r.ticker: (r.sleeve, float(r.target_weight)) for r in rows.itertuples()}, str(day)


def previous(logs: Path, strategy: str, day: str | None) -> dict[str, tuple[str, float]]:
    """The version before the one of `day`: the fixed starting mix when it is the first."""
    found = versions(logs)
    found = found[(found["strategy"] == strategy) & (found["date"] < day)] if day else found.iloc[0:0]
    if found.empty:
        return weights(strategy)
    rows = found[found["date"] == found["date"].max()]
    return {r.ticker: (r.sleeve, float(r.target_weight)) for r in rows.itertuples()}


def turnover(old: dict[str, float], new: dict[str, float]) -> float:
    """The share of a portfolio that changes from `old` to `new` (half the sum of the moves)."""
    return sum(abs(new.get(t, 0.0) - old.get(t, 0.0)) for t in set(old) | set(new)) / 2


def follow(target: pd.Series, strategy: str, families: dict[str, str]) -> dict[str, tuple[str, float]]:
    """The engine's portfolio of the day as the strategy's reference: every sleeve keeps its share of the fixed mix,
    and its ETFs are split ENGINE_SHARE as in the engine's portfolio, the rest as in the fixed mix (all of it when the
    engine holds nothing in that sleeve). Complex products stay within the strategy's limit, and lines under MIN_LINE
    are shared out among the others of their sleeve. ticker -> (sleeve, share)."""
    spec, fixed = STRATEGIES[strategy], weights(strategy)
    engine = {t: float(w) for t, w in target.items() if float(w) > 0 and t != CASH}
    out: dict[str, tuple[str, float]] = {}
    for sleeve, share in spec["sleeves"].items():
        if share <= 0:
            continue
        mine = {t: w for t, (s, w) in fixed.items() if s == sleeve}
        theirs = {t: w for t, w in engine.items() if sleeve_of(t, families.get(t, "")) == sleeve}
        split = {t: w / sum(mine.values()) * (1 - ENGINE_SHARE) for t, w in mine.items()}
        if theirs and sleeve != "cash":
            for t, w in theirs.items():
                split[t] = split.get(t, 0.0) + w / sum(theirs.values()) * ENGINE_SHARE
        else:
            split = {t: w / sum(mine.values()) for t, w in mine.items()}
        complex_ = [t for t in split if families.get(t) in COMPLEX_FAMILIES]
        room = spec["limits"]["complex_max"] / share
        if complex_ and sum(split[t] for t in complex_) > room:  # the excess goes to the sleeve's other ETFs
            others = [t for t in split if t not in complex_]
            scale = room / sum(split[t] for t in complex_)
            freed = sum(split[t] for t in complex_) * (1 - scale)
            split.update({t: split[t] * scale for t in complex_})
            rest = sum(split[t] for t in others)
            split.update({t: split[t] + freed * split[t] / rest for t in others} if rest else {})
        kept = {t: w for t, w in split.items() if w * share >= MIN_LINE} or {max(split, key=split.get): 1.0}
        out.update({t: (sleeve, round(share * w / sum(kept.values()), 6)) for t, w in kept.items()})
    big = max(out, key=lambda t: out[t][1])  # the rounding, on the largest line: the shares add up to 1 exactly
    out[big] = (out[big][0], round(out[big][1] + 1 - sum(w for _, w in out.values()), 6))
    return out


def publish(logs: Path, as_of: date, targets: dict[str, pd.Series], families: dict[str, str]) -> dict[str, float]:
    """Publishes each strategy's reference that follows the engine's new target (`follow`) as a new version when at
    least PUBLISH_TURNOVER of the current one would change. Returns strategy -> turnover for the versions published. Run again on the same day, it gives
    the same result (the day's own version is compared with the one before it)."""
    out, rows = {}, []
    for strategy, target in targets.items():
        new = follow(target, strategy, families)
        before, day = current(logs, strategy, as_of)
        if day == as_of.isoformat():
            before = previous(logs, strategy, day)
        moved = turnover({t: w for t, (_, w) in before.items()}, {t: w for t, (_, w) in new.items()})
        if moved >= PUBLISH_TURNOVER:
            out[strategy] = round(moved, 4)
            rows += [(as_of.isoformat(), strategy, t, w, sleeve) for t, (sleeve, w) in new.items()]
    if rows:
        found = versions(logs)
        found = found[~((found["date"] == as_of.isoformat()) & found["strategy"].isin(list(out)))]
        table = pd.concat([found, pd.DataFrame(rows, columns=STRATEGY_PORTFOLIOS)], ignore_index=True)
        Path(logs).mkdir(parents=True, exist_ok=True)
        validate_strategy_portfolios(table).to_csv(Path(logs) / VERSIONS, index=False)
    return out


def curve(prices: pd.DataFrame, strategy: str, held: dict[str, float] | None = None) -> pd.Series:
    """The value of 1 invested in a strategy's mix at the start of the price history given, day by day, put back to
    its weights at the start of each month. From adjusted closes. An ETF that was not listed yet counts as cash at 0%
    until its first quote. `held`: ticker -> share, default the strategy's fixed starting mix."""
    closes = prices.pivot(index="date", columns="ticker", values="adj_close_eur").sort_index().ffill()
    held = held or {ticker: weight for ticker, (_, weight) in weights(strategy).items()}
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
