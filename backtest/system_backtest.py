"""Backtest of the whole code-only system on past prices, with the competition's rules. Owner: Axel.

What is replayed, for every past 3-week window:
    technical agent (rule score) -> Portfolio Manager (code portfolio, no LLM) -> orders -> book
  - decisions on a Monday, Thursday, Monday, Thursday, Monday, as in the competition;
  - each one from the prices known that morning; an order fills at the last close before its day;
  - at most MAX_ORDERS_PER_DAY orders, the Portfolio Manager's hard limits, START_CAPITAL_EUR to start;
  - valued two days after the last decision, against the same money put in the benchmark on day one.

What is not replayed: everything that asks Claude (Macro/News, technical_llm, the Portfolio Manager's own
portfolio, the Risk Manager's reading). An LLM already knows what happened in the past and a web search
returns articles written later, so a backtest of them would look into the future. They are judged forward
only, by the review loop (graph/review_loop.py).

A variant changes one thing (a setting, or how the technical sheet becomes a score) and is compared with
the others on the same windows. To test an idea, add a line to VARIANTS (and a function to SCORES if the
idea is a new way to read the sheet), then run:

    python -m backtest.system_backtest                               # every variant, windows since 2022
    python -m backtest.system_backtest --variants current no_timing --since 2024-01-01

Results, in backtest/results/: system_backtest.csv (one row per variant and window),
system_backtest_summary.csv (one row per variant), system_backtest_signals.csv (what each answer of the
technical sheet was worth over the next 1 to 15 sessions) and system_backtest_fill_rule.csv (what the rule
"an order fills at the previous close" is worth to someone who orders after the open). All three are printed.

How to read the summary: windows start every week and last three, so they overlap; the t-value of the
mean excess return counts them as one third as many. Below about 2 in absolute value, the difference
from the benchmark is not distinguishable from luck.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from agents import portfolio, technical
from agents import portfolio_config as pm_cfg
from agents import strategy_config as tech_cfg
from data import config as settings
from data.schemas import HOLDINGS, SIGNALS

RESULTS = Path(__file__).resolve().parent / "results"
DECISION_OFFSETS = [0, 3, 7, 10, 14]  # Monday, Thursday, Monday, Thursday, Monday
VALUATION_OFFSET = 16  # the Wednesday after the last decision, like 23 December after 21 December
PRICE_DAYS = 420  # calendar days of prices a decision sees (momentum needs a year of quotes)
FEATURE_DAYS = 60  # calendar days of indicators a decision sees
MODULES = {"portfolio_config": pm_cfg, "strategy_config": tech_cfg, "config": settings}


# --- How the technical sheet becomes a score: one function per idea -------------------------------------

def rule_score(sheet: pd.DataFrame) -> pd.Series:
    """Today's rule: buy +1, watch +0.5, neutral 0, avoid -1 (SIGNAL_SCORES)."""
    return sheet["suggested_signal"].map(tech_cfg.SIGNAL_SCORES).astype(float)


def skip_overbought(sheet: pd.DataFrame) -> pd.Series:
    """The rule, but no entry on an ETF the mean-reversion measures call overbought."""
    return rule_score(sheet).where(~((sheet["mr_setup"] == "avoid") & (rule_score(sheet) > 0)), 0.0)


def buy_the_dip(sheet: pd.DataFrame) -> pd.Series:
    """The rule, plus +0.5 on an ETF that is oversold while its long-term trend is not down."""
    score = rule_score(sheet)
    return score.where(~((sheet["mr_setup"] == "buy") & (score < 0.5)), 0.5)


def dip_and_skip(sheet: pd.DataFrame) -> pd.Series:
    """Both: no entry on an overbought ETF, +0.5 on an oversold one."""
    score = skip_overbought(sheet)
    return score.where(~((sheet["mr_setup"] == "buy") & (score < 0.5)), 0.5)


def momentum_rank(sheet: pd.DataFrame) -> pd.Series:
    """Buy setups ranked by their 13612 momentum instead of all scored +1 (30% momentum = full score)."""
    strength = (sheet["momentum_13612"].astype(float) / 0.30).clip(0.2, 1.0).fillna(0.5)
    score = rule_score(sheet)
    return score.where(score <= 0, score * strength)


def confirmed_only(sheet: pd.DataFrame) -> pd.Series:
    """The rule, but a buy counts only with both confirmations (RSI and volume)."""
    score = rule_score(sheet)
    return score.where(~((score > 0) & (sheet["confirmations"] < 2)), 0.0)


SCORES = {f.__name__: f for f in (rule_score, skip_overbought, buy_the_dip, dip_and_skip, momentum_rank, confirmed_only)}


@dataclass(frozen=True)
class Variant:
    name: str
    what: str  # the one thing it changes, in words
    settings: dict = field(default_factory=dict)  # {"portfolio_config.TIMING_WEIGHT": 0.0}
    score: str = "rule_score"  # a key of SCORES


VARIANTS = [
    Variant("current", "the system as it is"),
    # The Portfolio Manager's settings
    Variant("anchor_baseline", "portfolio sized by score / volatility instead of the mean-variance optimiser",
            {"portfolio_config.ANCHOR": "baseline"}),
    Variant("no_timing", "orders never deferred to a later date", {"portfolio_config.TIMING_WEIGHT": 0.0}),
    Variant("less_invested", "60% invested when the market view is neutral, instead of 85%",
            {"portfolio_config.EXPOSURE_BASE": 0.60}),
    Variant("more_invested", "99% invested when the market view is neutral",
            {"portfolio_config.EXPOSURE_BASE": 0.99}),
    Variant("fewer_positions", "5 positions at most instead of 15 (optimiser) and 8 (baseline)",
            {"portfolio_config.OPTIMIZER_POSITIONS": 5, "portfolio_config.BASELINE_POSITIONS": 5}),
    Variant("cap_20", "one ETF at most 20% instead of 30%", {"portfolio_config.MAX_POSITION": 0.20}),
    Variant("risk_aversion_2", "the optimiser fears risk less (2 instead of 5)", {"portfolio_config.RISK_AVERSION": 2.0}),
    Variant("risk_aversion_1", "the optimiser fears risk much less (1 instead of 5)", {"portfolio_config.RISK_AVERSION": 1.0}),
    Variant("no_leverage", "no leveraged or inverse ETF at all", {"portfolio_config.MAX_LEVERAGED_SHARE": 0.0}),
    Variant("benchmark_allowed", "the code portfolios may buy the benchmark ETF itself",
            {"portfolio_config.BENCHMARK_ALLOWED": True}),
    # The competition's fill rule (to confirm with Belfius): the same system if an order fills at its own day's close
    Variant("fill_same_day", "an order fills at the close of its own day instead of the close before",
            {"config.FILL_AT_PREVIOUS_CLOSE": False}),
    # How the technical sheet becomes a score
    Variant("skip_overbought", "no entry on an ETF the mean-reversion measures call overbought", score="skip_overbought"),
    Variant("buy_the_dip", "+0.5 on an oversold ETF whose long-term trend is not down", score="buy_the_dip"),
    Variant("dip_and_skip", "both of the above", score="dip_and_skip"),
    Variant("momentum_rank", "buy setups ranked by their momentum instead of all scored +1", score="momentum_rank"),
    Variant("confirmed_only", "a buy counts only with both confirmations", score="confirmed_only"),
    # The score of each rule signal (found by the signal study: "watch" ETFs lag, "avoid" ETFs do not)
    Variant("watch_half", "a 'watch' signal scores +0.5, as before 16 November 2026",
            {"strategy_config.SIGNAL_SCORES": {"buy": 1.0, "watch": 0.5, "neutral": 0.0, "avoid": -1.0}}),
    Variant("watch_zero", "a 'watch' signal scores 0 (the current setting since 16 November 2026)",
            {"strategy_config.SIGNAL_SCORES": {"buy": 1.0, "watch": 0.0, "neutral": 0.0, "avoid": -1.0}}),
    Variant("watch_negative", "a 'watch' signal scores -0.5 instead of +0.5",
            {"strategy_config.SIGNAL_SCORES": {"buy": 1.0, "watch": -0.5, "neutral": 0.0, "avoid": -1.0}}),
    Variant("buy_only", "only 'buy' scores: 'watch' and 'avoid' score 0",
            {"strategy_config.SIGNAL_SCORES": {"buy": 1.0, "watch": 0.0, "neutral": 0.0, "avoid": 0.0}}),
    Variant("buy_only_confirmed", "only a 'buy' with both confirmations scores",
            {"strategy_config.SIGNAL_SCORES": {"buy": 1.0, "watch": 0.0, "neutral": 0.0, "avoid": 0.0}},
            score="confirmed_only"),
]


@contextmanager
def overrides(values: dict):
    """Set module settings ('portfolio_config.ANCHOR': 'baseline') for the time of a run, then put them back."""
    before = {}
    try:
        for key, value in values.items():
            module, name = key.split(".")
            before[key] = getattr(MODULES[module], name)
            setattr(MODULES[module], name, value)
        yield
    finally:
        for key, value in before.items():
            module, name = key.split(".")
            setattr(MODULES[module], name, value)


# --- The data a run needs, prepared once ------------------------------------------------------------------

@dataclass
class Market:
    prices: pd.DataFrame  # Prices format, `date` as timestamps, sorted
    closes: pd.DataFrame  # real close per day (rows) and ticker (columns)
    sheets: dict[date, pd.DataFrame]  # the technical sheet each decision day would have seen

    def before(self, day: date, days: int = PRICE_DAYS) -> pd.DataFrame:
        """The prices a decision of `day` sees: strictly before it."""
        stamp = pd.Timestamp(day)
        return self.prices[(self.prices["date"] < stamp) & (self.prices["date"] >= stamp - pd.Timedelta(days=days))]

    def fill(self, day: date) -> pd.Series:
        """The price an order of `day` fills at, per ticker: the last close before it, or its own close
        (FILL_AT_PREVIOUS_CLOSE in data/config.py)."""
        stamp = pd.Timestamp(day)
        known = self.closes.index < stamp if settings.FILL_AT_PREVIOUS_CLOSE else self.closes.index <= stamp
        return self.closes[known].ffill().iloc[-1].dropna()

    def close(self, day: date) -> pd.Series:
        """Last close on or before `day`, per ticker: what a holding is worth that evening."""
        return self.closes[self.closes.index <= pd.Timestamp(day)].ffill().iloc[-1].dropna()


def window_starts(closes: pd.DataFrame, since: date) -> list[date]:
    """Every Monday from `since` whose window ends inside the prices."""
    last = closes.index.max().date()
    day = since + timedelta(days=(7 - since.weekday()) % 7)
    out = []
    while day + timedelta(days=VALUATION_OFFSET) <= last:
        out.append(day)
        day += timedelta(days=7)
    return out


def build_sheet(prices: pd.DataFrame, features: pd.DataFrame, day: date) -> pd.DataFrame:
    """The technical sheet a decision of `day` would have seen, from data strictly before it."""
    stamp = pd.Timestamp(day)
    known = prices[(prices["date"] < stamp) & (prices["date"] >= stamp - pd.Timedelta(days=PRICE_DAYS))]
    recent = features[(features["date"] < stamp) & (features["date"] >= stamp - pd.Timedelta(days=FEATURE_DAYS))]
    if known.empty or recent.empty:
        return pd.DataFrame()
    return technical.build_sheet(known, recent, known["date"].max().date())


def prepare(prices: pd.DataFrame, features: pd.DataFrame, since: date) -> tuple[Market, list[date]]:
    prices = prices.assign(date=pd.to_datetime(prices["date"])).sort_values(["date", "ticker"]).reset_index(drop=True)
    features = features.assign(date=pd.to_datetime(features["date"]))
    closes = prices.pivot(index="date", columns="ticker", values="close_eur").sort_index()
    starts = window_starts(closes, since)
    days = sorted({start + timedelta(days=offset) for start in starts for offset in DECISION_OFFSETS})
    sheets = {day: build_sheet(prices, features, day) for day in days}
    return Market(prices, closes, sheets), starts


# --- One window -----------------------------------------------------------------------------------------------

def signals_for(sheet: pd.DataFrame, score: str, day: date) -> pd.DataFrame:
    """The sheet as technical_det signals (Signals format), scored by SCORES[score]."""
    return pd.DataFrame({"ticker": sheet["ticker"].values, "date": day.isoformat(), "agent": technical.AGENT_DET,
                         "score": SCORES[score](sheet).clip(-1, 1).values,
                         "confidence": (sheet["confirmations"] / 2).values, "rationale": "backtest"}, columns=SIGNALS)


def run_window(market: Market, start: date, variant: Variant) -> dict:
    """Trade one window like the competition and return what it was worth at the end."""
    days = [start + timedelta(days=offset) for offset in DECISION_OFFSETS]
    end = start + timedelta(days=VALUATION_OFFSET)
    units: dict[str, float] = {}
    cash = float(settings.START_CAPITAL_EUR)
    orders_made = broken_rules = failed_days = 0
    patched = {**variant.settings, "config.TRADING_DAYS": days, "portfolio_config.VALUATION_END": end}
    with overrides(patched):
        for day in days:
            sheet = market.sheets.get(day)
            if sheet is None or sheet.empty:
                failed_days += 1
                continue
            holdings = pd.DataFrame([{"ticker": t, "units": u} for t, u in units.items() if u > 1e-9], columns=HOLDINGS)
            try:
                _, orders, _ = portfolio.run(signals_for(sheet, variant.score, day), {technical.AGENT_DET: 1.0}, "",
                                             holdings, cash, market.before(day), day, use_llm=False)
            except Exception:
                failed_days += 1
                continue
            fill = market.fill(day)
            broken_rules += int(len(orders) > settings.MAX_ORDERS_PER_DAY)
            for order in orders.itertuples():
                signed = float(order.units) if order.side == "BUY" else -float(order.units)
                units[order.ticker] = units.get(order.ticker, 0.0) + signed
                cash -= signed * float(fill[order.ticker])
                broken_rules += int(units[order.ticker] < -1e-9)
            broken_rules += int(cash < -0.01)
            orders_made += len(orders)
    close = market.close(end)
    held = {t: u * float(close[t]) for t, u in units.items() if u > 1e-9}
    value = cash + sum(held.values())
    first_fill = market.fill(days[0])
    bench = float(close[settings.BENCHMARK] / first_fill[settings.BENCHMARK] - 1)
    ret = value / settings.START_CAPITAL_EUR - 1
    return {"variant": variant.name, "start": start.isoformat(), "end": end.isoformat(), "value": round(value, 2),
            "ret": ret, "bench_ret": bench, "excess": ret - bench, "orders": orders_made, "positions": len(held),
            "invested": sum(held.values()) / value if value else 0.0,
            "largest": max(held.values()) / value if held and value else 0.0,
            "failed_days": failed_days, "broken_rules": broken_rules}


# --- The quality of a score on its own: does it rank the ETFs? ------------------------------------------------

def score_quality(market: Market, score: str, sessions: int | None = None) -> dict:
    """How a score did over the next sessions, on every decision day: mean rank correlation with the return,
    share of buy calls that rose, and the mean return of the ETFs it calls a buy against all scored ETFs."""
    sessions = settings.HORIZON_SESSIONS if sessions is None else sessions
    closes = market.closes.ffill()
    ics, hits, picks, everything = [], [], [], []
    for day, sheet in market.sheets.items():
        before = int((closes.index < pd.Timestamp(day)).sum())
        if sheet.empty or before == 0 or before - 1 + sessions >= len(closes):
            continue
        forward = closes.iloc[before - 1 + sessions] / closes.iloc[before - 1] - 1
        frame = pd.DataFrame({"score": SCORES[score](sheet).values, "ret": forward.reindex(sheet["ticker"]).values}).dropna()
        if len(frame) < 5:
            continue
        if frame["score"].nunique() > 1:
            ics.append(frame["score"].rank().corr(frame["ret"].rank()))
        buys = frame[frame["score"] >= 0.5]
        if len(buys):
            hits.append(float((buys["ret"] > 0).mean()))
            picks.append(float(buys["ret"].mean()))
            everything.append(float(frame["ret"].mean()))
    return {"days": len(ics), "ic": float(np.nanmean(ics)) if ics else np.nan,
            "buy_hit_rate": float(np.mean(hits)) if hits else np.nan,
            "buy_ret": float(np.mean(picks)) if picks else np.nan, "all_ret": float(np.mean(everything)) if everything else np.nan}


STUDY_HORIZONS = [1, 3, 5, 10, 15]  # sessions: the next day, the next trading date, ..., the whole window
STUDY_GROUPS = {  # what the technical sheet says about an ETF -> the column and the value that says it
    "rule: buy": ("suggested_signal", "buy"), "rule: watch": ("suggested_signal", "watch"),
    "rule: neutral": ("suggested_signal", "neutral"), "rule: avoid": ("suggested_signal", "avoid"),
    "mean reversion: oversold (buy)": ("mr_setup", "buy"), "mean reversion: overbought (avoid)": ("mr_setup", "avoid"),
    "trend strength: strong": ("trend_strength", "strong"), "trend strength: absent": ("trend_strength", "absent"),
}


def signal_study(market: Market, horizons: list[int] | None = None) -> pd.DataFrame:
    """Where the technical sheet has predictive value, and over how many sessions: for each thing it says
    about an ETF (STUDY_GROUPS, and the top and bottom fifth by momentum), the mean return of those ETFs over
    the next sessions MINUS the mean return of all scored ETFs on the same day. One row per group and horizon:
    edge (that difference, averaged over all the calls), t (how many standard errors from zero, each day counted
    as one observation), calls and days."""
    horizons = STUDY_HORIZONS if horizons is None else horizons
    closes = market.closes.ffill()
    samples: dict[tuple[str, int], list[tuple[float, int]]] = {}
    for day, sheet in market.sheets.items():
        before = int((closes.index < pd.Timestamp(day)).sum())
        if sheet.empty or before == 0:
            continue
        momentum = sheet["momentum_13612"].astype(float)
        groups = {name: (sheet[column] == value).values for name, (column, value) in STUDY_GROUPS.items()}
        if momentum.notna().sum() >= 10:
            groups["momentum 13612: top fifth"] = (momentum >= momentum.quantile(0.8)).values
            groups["momentum 13612: bottom fifth"] = (momentum <= momentum.quantile(0.2)).values
        for horizon in horizons:
            if before - 1 + horizon >= len(closes):
                continue
            forward = (closes.iloc[before - 1 + horizon] / closes.iloc[before - 1] - 1).reindex(sheet["ticker"]).values
            known = ~np.isnan(forward)
            if known.sum() < 5:
                continue
            everyone = forward[known].mean()
            for name, mask in groups.items():
                chosen = mask & known
                if chosen.any():
                    samples.setdefault((name, horizon), []).append((forward[chosen].mean() - everyone, int(chosen.sum())))
    rows = []
    for (name, horizon), values in samples.items():
        edges, sizes = np.array([edge for edge, _ in values]), np.array([n for _, n in values], dtype=float)
        mean = float((edges * sizes).sum() / sizes.sum())  # every call counts the same, so the groups of one kind sum to zero
        overlap = max(horizon / 3, 1)  # decision days are about 3 sessions apart: longer horizons overlap
        error = np.sqrt(((sizes * (edges - mean)) ** 2).sum()) / sizes.sum() * np.sqrt(overlap) if len(edges) > 1 else np.nan
        rows.append({"group": name, "sessions": horizon, "edge": mean, "t": mean / error if error else np.nan,
                     "calls": int(sizes.sum()), "days": len(edges)})
    order = {name: i for i, name in enumerate([*STUDY_GROUPS, "momentum 13612: top fifth", "momentum 13612: bottom fifth"])}
    return pd.DataFrame(rows).sort_values(["group", "sessions"], key=lambda c: c.map(order) if c.name == "group" else c
                                          ).reset_index(drop=True)


def show_study(study: pd.DataFrame) -> str:
    """One line per group: its edge over the average ETF at each horizon, with a star when |t| is 2 or more."""
    horizons = sorted(study["sessions"].unique())
    lines = [f"{'return over the average ETF, after':<38}" + "".join(f"{f'{h} sessions':>14}" for h in horizons) + f"{'calls':>8}"]
    for group, rows in study.groupby("group", sort=False):
        by = rows.set_index("sessions")
        cells = "".join(f"{by.loc[h, 'edge']:>+12.2%}{'*' if abs(by.loc[h, 't']) >= 2 else ' '} " if h in by.index else f"{'':>14}"
                        for h in horizons)
        lines.append(f"{group:<38}{cells}{int(by['calls'].max()):>8}")
    return "\n".join(lines)


def fill_rule_study(market: Market, picks: int = 5) -> pd.DataFrame:
    """What the rule "an order of day T fills at the close of T-1" is worth to someone who orders after the
    market has opened. One row per past Monday and Thursday:
      average_etf      the return of the average ETF from the close of T-1 to the close of T
      morning_picks    the same for the `picks` ETFs with the largest opening gap (known minutes after the open)
      morning_gap      how far above the fill price those ETFs already were at the open
      morning_up       the share of those picks that ended the day above their fill price
      close_picks      the same return for the `picks` best ETFs of the day (an order sent at the close): the ceiling
    Under the other rule (a fill at the close of T itself) all of this is zero: the price paid is the price of the
    moment. The study says how much the answer to "which rule, and until when can we order?" matters."""
    prices = market.prices
    closes = prices.pivot(index="date", columns="ticker", values="close_eur").sort_index()
    opens = prices.pivot(index="date", columns="ticker", values="open_eur").sort_index()
    before = closes.shift(1)
    day_return, gap = closes / before - 1, opens / before - 1
    first = min(market.sheets) if market.sheets else closes.index.min().date()
    rows = []
    for day in closes.index[closes.index >= pd.Timestamp(first)]:
        if day.weekday() not in (0, 3):
            continue
        returns, gaps = day_return.loc[day].dropna(), gap.loc[day].dropna()
        both = returns.index.intersection(gaps.index)
        if len(both) < 4 * picks:
            continue
        returns, gaps = returns[both], gaps[both]
        chosen = gaps.nlargest(picks).index
        rows.append({"day": day.date().isoformat(), "average_etf": returns.mean(), "morning_picks": returns[chosen].mean(),
                     "morning_gap": gaps[chosen].mean(), "morning_up": float((returns[chosen] > 0).mean()),
                     "close_picks": returns.nlargest(picks).mean()})
    return pd.DataFrame(rows)


def show_fill_rule(study: pd.DataFrame) -> str:
    if study.empty:
        return "no day to study"
    mean = study.mean(numeric_only=True)
    days = len(settings.TRADING_DAYS)
    return (f"On {len(study)} past Mondays and Thursdays, from the close before to the close of the day:\n"
            f"  the average ETF                                   {mean['average_etf']:+.2%}\n"
            f"  the 5 ETFs with the largest opening gap           {mean['morning_picks']:+.2%}   "
            f"(already {mean['morning_gap']:+.2%} at the open; {mean['morning_up']:.0%} of them end the day up; "
            f"worst day {study['morning_picks'].min():+.1%})\n"
            f"  the 5 best ETFs of the day (order sent at the close) {mean['close_picks']:+.2%}\n"
            f"  {days} trading days of the morning picks, compounded: {(1 + mean['morning_picks']) ** days - 1:+.1%}")


# --- Every window of every variant, in parallel ------------------------------------------------------------------

_MARKET: Market | None = None


def _load(market: Market) -> None:
    global _MARKET
    _MARKET = market


def _task(job: tuple[date, Variant]) -> dict:
    return run_window(_MARKET, *job)


def run(market: Market, starts: list[date], variants: list[Variant], workers: int | None = None) -> pd.DataFrame:
    """One row per variant and window."""
    jobs = [(start, variant) for variant in variants for start in starts]
    if workers == 1:
        return pd.DataFrame([run_window(market, *job) for job in jobs])
    with ProcessPoolExecutor(max_workers=workers, initializer=_load, initargs=(market,)) as pool:
        return pd.DataFrame(list(pool.map(_task, jobs, chunksize=8)))


def summarise(results: pd.DataFrame, market: Market, variants: list[Variant]) -> pd.DataFrame:
    """One row per variant: what the windows returned against the benchmark and against the first variant
    (the system as it is) on the same windows, the risk taken, and how good its score is on its own."""
    returns = results.pivot(index="start", columns="variant", values="ret")
    reference = returns[variants[0].name]
    rows = []
    for variant in variants:
        mine = results[results["variant"] == variant.name].set_index("start")
        n = len(mine)
        independent = max(n / 3, 1)  # weekly windows of three weeks overlap

        def t_value(series: pd.Series) -> float:
            spread = series.std(ddof=1) if len(series) > 1 else np.nan
            return float(series.mean() / (spread / np.sqrt(independent))) if spread and spread > 0 else np.nan

        gap = (mine["ret"] - reference).dropna()
        with overrides(variant.settings):  # a variant may change what each rule signal scores
            quality = score_quality(market, variant.score)
        rows.append({"variant": variant.name, "what": variant.what, "windows": n,
                     "mean_ret": mine["ret"].mean(), "mean_bench": mine["bench_ret"].mean(),
                     "mean_excess": mine["excess"].mean(), "t_excess": t_value(mine["excess"]),
                     "beat_bench": (mine["excess"] > 0).mean(), "up_windows": (mine["ret"] > 0).mean(),
                     "worst_ret": mine["ret"].min(), "best_ret": mine["ret"].max(),
                     "risk": mine["ret"].std(), "bench_risk": mine["bench_ret"].std(),
                     "ret_per_risk": mine["ret"].mean() / mine["ret"].std() if mine["ret"].std() > 0 else np.nan,
                     "bench_ret_per_risk": mine["bench_ret"].mean() / mine["bench_ret"].std() if n > 1 else np.nan,
                     "beta": float(np.polyfit(mine["bench_ret"], mine["ret"], 1)[0]) if n > 2 else np.nan,
                     "vs_first": gap.mean(), "t_vs_first": t_value(gap),
                     "mean_invested": mine["invested"].mean(), "mean_orders": mine["orders"].mean(),
                     "broken_rules": int(mine["broken_rules"].sum()), "failed_days": int(mine["failed_days"].sum()),
                     "score_ic": quality["ic"], "buy_hit_rate": quality["buy_hit_rate"],
                     "buy_ret_3d": quality["buy_ret"], "all_ret_3d": quality["all_ret"]})
    return pd.DataFrame(rows)


def show(summary: pd.DataFrame) -> str:
    """The summary as text: risk is the spread of a window's return, ret/risk the return per unit of it, beta the
    share of the benchmark's move the portfolio takes, and the last columns compare with the first variant on
    the same windows (a t-value of 2 or more means the difference is unlikely to be luck)."""
    first = summary["variant"].iloc[0]
    lines = [f"{'variant':<19}{'return':>8}{'excess':>8}{'t':>6}{'worst':>8}{'risk':>7}{'ret/risk':>9}{'beta':>6}{'invested':>9}"
             f"{f'vs {first}':>12}{'t':>6}{'IC':>8}"]
    for r in summary.itertuples():
        lines.append(f"{r.variant:<19}{r.mean_ret:>+8.2%}{r.mean_excess:>+8.2%}{r.t_excess:>+6.1f}{r.worst_ret:>+8.1%}"
                     f"{r.risk:>7.2%}{r.ret_per_risk:>9.2f}{r.beta:>6.2f}{r.mean_invested:>9.0%}"
                     f"{r.vs_first:>+12.2%}{r.t_vs_first:>+6.1f}{r.score_ic:>+8.3f}")
    r = summary.iloc[0]
    lines.append(f"{'benchmark':<19}{r.mean_bench:>+8.2%}{'':>22}{r.bench_risk:>7.2%}{r.bench_ret_per_risk:>9.2f}{1:>6.2f}{1:>9.0%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backtest the code-only system on past 3-week windows.")
    parser.add_argument("--since", type=date.fromisoformat, default=date(2022, 1, 3), help="first window, YYYY-MM-DD")
    parser.add_argument("--variants", nargs="*", help="names from VARIANTS (default: all); the other variants "
                                                      "already in the results file are kept")
    parser.add_argument("--workers", type=int, help="processes (default: one per core)")
    args = parser.parse_args(argv)
    variants = [v for v in VARIANTS if not args.variants or v.name in args.variants]
    from data import data_agent
    prices = data_agent.fetch_prices(args.since - timedelta(days=PRICE_DAYS + 30), date.today())
    market, starts = prepare(prices, data_agent.compute_features(prices), args.since)
    print(f"{len(starts)} windows from {starts[0]} to {starts[-1]}, {len(market.sheets)} decision days, "
          f"{len(variants)} variant(s) to run")
    results = run(market, starts, variants, args.workers)
    known = RESULTS / "system_backtest.csv"
    if args.variants and known.exists():  # a partial run replaces its variants and keeps the others
        old = pd.read_csv(known)
        if set(old["start"]) == set(results["start"]):
            results = pd.concat([old[~old["variant"].isin(results["variant"])], results], ignore_index=True)
    variants = [v for v in VARIANTS if v.name in set(results["variant"])]
    order = {v.name: i for i, v in enumerate(variants)}
    results = results.sort_values(["variant", "start"], key=lambda c: c.map(order) if c.name == "variant" else c)
    summary = summarise(results, market, variants)
    study = signal_study(market)
    rule = fill_rule_study(market)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(known, index=False)
    summary.to_csv(RESULTS / "system_backtest_summary.csv", index=False)
    study.to_csv(RESULTS / "system_backtest_signals.csv", index=False)
    rule.to_csv(RESULTS / "system_backtest_fill_rule.csv", index=False)
    print(show(summary))
    print("\nWhat the technical sheet is worth (* = at least 2 standard errors from zero):\n" + show_study(study))
    print("\nWhat the fill rule is worth (an order of day T filled at the close of T-1):\n" + show_fill_rule(rule))
    print(f"\nWritten in {RESULTS}: system_backtest.csv, system_backtest_summary.csv, system_backtest_signals.csv, "
          "system_backtest_fill_rule.csv")


if __name__ == "__main__":
    main()
