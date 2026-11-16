"""Backtest of the whole code-only system on past prices, with the competition's rules. Owner: Axel.

Two ways to replay it:
  - windows (default): the competition itself, again and again. Each Monday starts a 3-week window from cash.
    It answers "what would our result in the competition have been".
  - daily (--daily): one book over the whole history, never reset, valued at every close. Decisions on every
    Monday and Thursday with the same rules. It gives the Sharpe ratio from daily returns, the measure the
    team optimises, and compares variants day by day.

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
    python -m backtest.system_backtest --daily                       # every variant, one book since 2022

Results, in backtest/results/: system_backtest.csv (one row per variant and window),
system_backtest_summary.csv (one row per variant), system_backtest_signals.csv (what each answer of the
technical sheet was worth over the next 1 to 15 sessions) and system_backtest_fill_rule.csv (what the rule
"an order fills at the previous close" is worth to someone who orders after the open). All three are printed.
The daily run writes system_backtest_daily.csv (one row per variant and close) and system_backtest_daily_summary.csv.

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
    Variant("before_16_nov", "the settings before 16 November 2026: 'watch' scores +0.5, idle budget stays in cash",
            {"portfolio_config.IDLE_TO_BENCHMARK": 0.0, "portfolio_config.BENCHMARK_CAP": 0.30,
             "strategy_config.SIGNAL_SCORES": {"buy": 1.0, "watch": 0.5, "neutral": 0.0, "avoid": -1.0}}),
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
    # Where the money without an idea goes (the leaderboard ranks on return)
    Variant("idle_in_cash", "idle budget stays in cash instead of following the benchmark",
            {"portfolio_config.IDLE_TO_BENCHMARK": 0.0}),
    Variant("benchmark_cap_30", "the benchmark top-up stops at 30% of the portfolio instead of 60%",
            {"portfolio_config.BENCHMARK_CAP": 0.30}),
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
    Variant("watch_negative", "a 'watch' signal scores -0.5 instead of 0",
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

    def fill_day(self, day: date) -> pd.Timestamp:
        """The date of the close an order of `day` fills at."""
        stamp = pd.Timestamp(day)
        return self.closes.index[self.closes.index < stamp if settings.FILL_AT_PREVIOUS_CLOSE else self.closes.index <= stamp][-1]


def decision_days(closes: pd.DataFrame, since: date) -> list[date]:
    """Every Monday and Thursday from `since` to the last close: the days the competition trades on."""
    last, day, out = closes.index.max().date(), since, []
    while day <= last:
        if day.weekday() in (0, 3):
            out.append(day)
        day += timedelta(days=1)
    return out


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
    windows = {start + timedelta(days=offset) for start in starts for offset in DECISION_OFFSETS}
    days = sorted(windows | set(decision_days(closes, since)))  # the windows' days and the daily book's
    sheets = {day: build_sheet(prices, features, day) for day in days}
    return Market(prices, closes, sheets), starts


# --- One window -----------------------------------------------------------------------------------------------

def signals_for(sheet: pd.DataFrame, score: str, day: date) -> pd.DataFrame:
    """The sheet as technical_det signals (Signals format), scored by SCORES[score]."""
    return pd.DataFrame({"ticker": sheet["ticker"].values, "date": day.isoformat(), "agent": technical.AGENT_DET,
                         "score": SCORES[score](sheet).clip(-1, 1).values,
                         "confidence": (sheet["confirmations"] / 2).values, "rationale": "backtest"}, columns=SIGNALS)


def trade(market: Market, day: date, units: dict[str, float], cash: float, score: str) -> tuple[float, int | None, int]:
    """One decision day on a book: the Portfolio Manager's orders, filled at the competition's price. Changes
    `units` in place. Returns (cash after, orders made or None when the day failed, trading rules broken)."""
    sheet = market.sheets.get(day)
    if sheet is None or sheet.empty:
        return cash, None, 0
    holdings = pd.DataFrame([{"ticker": t, "units": u} for t, u in units.items() if u > 1e-9], columns=HOLDINGS)
    try:
        _, orders, _ = portfolio.run(signals_for(sheet, score, day), {technical.AGENT_DET: 1.0}, "",
                                     holdings, cash, market.before(day), day, use_llm=False)
    except Exception:
        return cash, None, 0
    fill = market.fill(day)
    broken = int(len(orders) > settings.MAX_ORDERS_PER_DAY)
    for order in orders.itertuples():
        signed = float(order.units) if order.side == "BUY" else -float(order.units)
        units[order.ticker] = units.get(order.ticker, 0.0) + signed
        cash -= signed * float(fill[order.ticker])
        broken += int(units[order.ticker] < -1e-9)
    return cash, len(orders), broken + int(cash < -0.01)


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
            cash, made, broken = trade(market, day, units, cash, variant.score)
            if made is None:
                failed_days += 1
                continue
            orders_made += made
            broken_rules += broken
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


# --- The whole history as one book, valued at every close -----------------------------------------------------

RISK_FREE = "OVERNIGHT CASH"  # the euro overnight rate: what money earns without risk, the Sharpe ratio's yardstick
YEAR = 252  # trading days


def run_continuous(market: Market, variant: Variant, days: list[date] | None = None) -> pd.DataFrame:
    """The system as one book over the whole history: the competition's rules on every Monday and Thursday (at
    most MAX_ORDERS_PER_DAY orders, the competition's fill price, the hard limits), but no reset to cash every
    three weeks. One row per close from the fill of the first decision: variant, date, value (the book), bench
    (the same money put in the benchmark that day), riskfree (1 grown at the overnight rate), invested (share of
    the value in ETFs), and on the closes where a decision takes effect its orders, broken rules and failed days.
    A decision of day D is in force from the first close on or after D (its fill, the close before, is the same
    value either way)."""
    days = sorted(market.sheets) if days is None else days
    units: dict[str, float] = {}
    cash, decisions = float(settings.START_CAPITAL_EUR), []
    patched = {**variant.settings, "config.TRADING_DAYS": days, "portfolio_config.VALUATION_END": days[-1]}
    with overrides(patched):
        start = market.fill_day(days[0])
        for day in days:
            cash, made, broken = trade(market, day, units, cash, variant.score)
            decisions.append((pd.Timestamp(day), dict(units), cash, made or 0, broken, int(made is None)))
    closes = market.closes.ffill()
    closes = closes[closes.index >= start]
    bench = settings.START_CAPITAL_EUR * closes[settings.BENCHMARK] / closes[settings.BENCHMARK].iloc[0]
    riskfree = (closes[RISK_FREE] / closes[RISK_FREE].dropna().iloc[0]).fillna(1.0) if RISK_FREE in closes else 1.0
    rows, j = [], -1
    held, cash_now = {}, float(settings.START_CAPITAL_EUR)
    for t in closes.index:
        made = broken = failed = 0
        while j + 1 < len(decisions) and decisions[j + 1][0] <= t:
            j += 1
            _, held, cash_now, n, b, f = decisions[j]
            made, broken, failed = made + n, broken + b, failed + f
        invested = sum(u * float(closes.at[t, ticker]) for ticker, u in held.items() if u > 1e-9)
        value = cash_now + invested
        rows.append({"variant": variant.name, "date": t.date().isoformat(), "value": value, "bench": float(bench[t]),
                     "riskfree": float(riskfree[t]) if not isinstance(riskfree, float) else riskfree,
                     "invested": invested / value if value else 0.0, "orders": made, "broken_rules": broken,
                     "failed_days": failed})
    return pd.DataFrame(rows)


def _sharpe(excess: pd.Series) -> float:
    """Yearly Sharpe ratio from daily returns above the risk-free rate."""
    spread = excess.std(ddof=1)
    return float(excess.mean() / spread * np.sqrt(YEAR)) if spread and spread > 0 else np.nan


def _sharpe_gap_t(a: pd.Series, b: pd.Series, rf: pd.Series) -> float:
    """t-value of the difference of two Sharpe ratios measured on the same days (Jobson-Korkie, Memmel's
    correction): two variants that move together have Sharpe ratios that differ by less noise."""
    ea, eb = a - rf, b - rf
    sa, sb_, rho = ea.mean() / ea.std(), eb.mean() / eb.std(), ea.corr(eb)
    variance = (2 * (1 - rho) + 0.5 * (sa ** 2 + sb_ ** 2 - 2 * sa * sb_ * rho ** 2)) / len(ea)
    return float((sa - sb_) / np.sqrt(variance)) if variance > 0 else np.nan


def summarise_daily(daily: pd.DataFrame, variants: list[Variant]) -> pd.DataFrame:
    """One row per variant, from its daily returns: Sharpe ratio (yearly return above the overnight rate per unit
    of yearly volatility), in each half of the history too, return and volatility a year, the worst fall from a
    peak, beta to the benchmark, and the gap with the first variant on the same days (its Sharpe ratio minus the
    first's with the t-value of that gap; the return a year above the first's). A variant "holds" when its Sharpe
    ratio is above the first's in both halves."""
    first = variants[0].name
    frames = {v.name: daily[daily["variant"] == v.name].set_index("date") for v in variants}
    returns = {name: f["value"].pct_change().iloc[1:] for name, f in frames.items()}
    reference = frames[first]
    rf_all = reference["riskfree"].pct_change().iloc[1:]
    bench = reference["bench"].pct_change().iloc[1:]
    rows = []
    for v in variants:
        f, r = frames[v.name], returns[v.name]
        rf = rf_all.reindex(r.index).fillna(0.0)
        halves = [r.index[:len(r) // 2], r.index[len(r) // 2:]]
        both = r.index.intersection(returns[first].index)
        gap = r[both] - returns[first][both]
        b = bench.reindex(r.index)
        rows.append({"variant": v.name, "what": v.what, "years": len(r) / YEAR,
                     "sharpe": _sharpe(r - rf), "sharpe_h1": _sharpe((r - rf)[halves[0]]),
                     "sharpe_h2": _sharpe((r - rf)[halves[1]]),
                     "ret_year": float((f["value"].iloc[-1] / f["value"].iloc[0]) ** (YEAR / len(r)) - 1),
                     "vol_year": float(r.std() * np.sqrt(YEAR)),
                     "max_drawdown": float((f["value"] / f["value"].cummax() - 1).min()),
                     "beta": float(r.cov(b) / b.var()), "invested": float(f["invested"].mean()),
                     "orders": int(f["orders"].sum()), "broken_rules": int(f["broken_rules"].sum()),
                     "failed_days": int(f["failed_days"].sum()),
                     "ret_vs_first": float(gap.mean() * YEAR),
                     "t_sharpe_vs_first": np.nan if v.name == first else
                     _sharpe_gap_t(r[both], returns[first][both], rf_all.reindex(both).fillna(0.0))})
    out = pd.DataFrame(rows)
    out["sharpe_vs_first"] = out["sharpe"] - out["sharpe"].iloc[0]
    out["holds"] = (out["sharpe_h1"] > out["sharpe_h1"].iloc[0]) & (out["sharpe_h2"] > out["sharpe_h2"].iloc[0])
    rf = rf_all.reindex(bench.index).fillna(0.0)
    halves = [bench.index[:len(bench) // 2], bench.index[len(bench) // 2:]]
    out["bench_sharpe"], out["bench_sharpe_h1"], out["bench_sharpe_h2"] = (
        _sharpe(bench - rf), _sharpe((bench - rf)[halves[0]]), _sharpe((bench - rf)[halves[1]]))
    out["bench_ret_year"] = float((reference["bench"].iloc[-1] / reference["bench"].iloc[0]) ** (YEAR / len(bench)) - 1)
    out["bench_vol_year"] = float(bench.std() * np.sqrt(YEAR))
    out["bench_max_drawdown"] = float((reference["bench"] / reference["bench"].cummax() - 1).min())
    out["riskfree_year"] = float(reference["riskfree"].iloc[-1] ** (YEAR / len(bench)) - 1)
    return out


def show_daily(summary: pd.DataFrame) -> str:
    """The daily summary as text, best Sharpe ratio first. A t-value of 2 or more on the Sharpe gap means it is
    unlikely to be luck; "holds" means the variant is ahead in both halves of the history."""
    first = summary["variant"].iloc[0]
    lines = [f"{'variant':<19}{'Sharpe':>7}{'1st half':>9}{'2nd half':>9}{'return':>8}{'vol':>7}{'worst fall':>11}"
             f"{'beta':>6}{'invested':>9}{f'vs {first}':>12}{'t':>6}{'holds':>7}"]
    for r in summary.sort_values("sharpe", ascending=False).itertuples():
        holds = "" if r.variant == first else ("yes" if r.holds else "no")
        t = "" if r.variant == first else f"{r.t_sharpe_vs_first:+.1f}"
        lines.append(f"{r.variant:<19}{r.sharpe:>7.2f}{r.sharpe_h1:>9.2f}{r.sharpe_h2:>9.2f}{r.ret_year:>+8.1%}"
                     f"{r.vol_year:>7.1%}{r.max_drawdown:>+11.1%}{r.beta:>6.2f}{r.invested:>9.0%}"
                     f"{r.sharpe_vs_first:>+12.2f}{t:>6}{holds:>7}")
    r = summary.iloc[0]
    lines.append(f"{'benchmark':<19}{r.bench_sharpe:>7.2f}{r.bench_sharpe_h1:>9.2f}{r.bench_sharpe_h2:>9.2f}"
                 f"{r.bench_ret_year:>+8.1%}{r.bench_vol_year:>7.1%}{r.bench_max_drawdown:>+11.1%}{1:>6.2f}{1:>9.0%}")
    lines.append(f"{r.years:.1f} years of daily returns; Sharpe = yearly return above the overnight rate "
                 f"({r.riskfree_year:+.1%} a year) / yearly volatility.")
    return "\n".join(lines)


def _task_daily(variant: Variant) -> pd.DataFrame:
    return run_continuous(_MARKET, variant)


def run_daily(market: Market, variants: list[Variant], workers: int | None = None) -> pd.DataFrame:
    """One daily book per variant, in parallel (a book is sequential: each decision starts from the last one)."""
    if workers == 1:
        return pd.concat([run_continuous(market, v) for v in variants], ignore_index=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_load, initargs=(market,)) as pool:
        return pd.concat(list(pool.map(_task_daily, variants)), ignore_index=True)


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


def signal_study(market: Market, horizons: list[int] | None = None, days: list[date] | None = None) -> pd.DataFrame:
    """Where the technical sheet has predictive value, and over how many sessions: for each thing it says
    about an ETF (STUDY_GROUPS, and the top and bottom fifth by momentum), the mean return of those ETFs over
    the next sessions MINUS the mean return of all scored ETFs on the same day. One row per group and horizon:
    edge (that difference, averaged over all the calls), t (how many standard errors from zero, each day counted
    as one observation), calls and days."""
    horizons = STUDY_HORIZONS if horizons is None else horizons
    closes = market.closes.ffill()
    samples: dict[tuple[str, int], list[tuple[float, int]]] = {}
    for day, sheet in market.sheets.items():
        if days is not None and day not in days:
            continue
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


def study_by_half(market: Market, sessions: int | None = None) -> pd.DataFrame:
    """The signal study over the first and the second half of the decision days, at one horizon: one row per
    group with its edge and t-value in each half. An answer of the sheet is worth trusting when both agree."""
    sessions = settings.HORIZON_SESSIONS if sessions is None else sessions
    days = sorted(market.sheets)
    halves = [signal_study(market, [sessions], part).set_index("group") for part in (days[:len(days) // 2], days[len(days) // 2:])]
    out = pd.DataFrame({"edge_h1": halves[0]["edge"], "t_h1": halves[0]["t"], "edge_h2": halves[1]["edge"], "t_h2": halves[1]["t"]})
    out["same_sign"] = (out["edge_h1"] * out["edge_h2"]) > 0
    return out.reset_index()


def show_by_half(table: pd.DataFrame) -> str:
    lines = [f"{'':<38}{'first half':>16}{'second half':>16}{'agree':>8}"]
    for r in table.itertuples():
        lines.append(f"{r.group:<38}{r.edge_h1:>+9.2%} ({r.t_h1:+.1f}){r.edge_h2:>+9.2%} ({r.t_h2:+.1f}){'yes' if r.same_sign else 'no':>8}")
    return "\n".join(lines)


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
        halves = [gap.iloc[:len(gap) // 2], gap.iloc[len(gap) // 2:]]  # the windows are sorted by date
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
                     "vs_first_h1": halves[0].mean(), "vs_first_h2": halves[1].mean(),
                     "holds": bool(halves[0].mean() > 0 and halves[1].mean() > 0),
                     "ret_h1": mine["ret"].iloc[:n // 2].mean(), "ret_h2": mine["ret"].iloc[n // 2:].mean(),
                     "mean_invested": mine["invested"].mean(), "mean_orders": mine["orders"].mean(),
                     "broken_rules": int(mine["broken_rules"].sum()), "failed_days": int(mine["failed_days"].sum()),
                     "score_ic": quality["ic"], "buy_hit_rate": quality["buy_hit_rate"],
                     "buy_ret_3d": quality["buy_ret"], "all_ret_3d": quality["all_ret"]})
    return pd.DataFrame(rows)


def show(summary: pd.DataFrame) -> str:
    """The summary as text: risk is the spread of a window's return, ret/risk the return per unit of it, beta the
    share of the benchmark's move the portfolio takes. The last columns compare with the first variant on the same
    windows: over all of them (a t-value of 2 or more means the difference is unlikely to be luck), then over
    the first and the second half of the history. A variant "holds" only when it is ahead in both halves: one
    that is ahead in a single half has found a kind of market, not an improvement."""
    first = summary["variant"].iloc[0]
    lines = [f"{'variant':<19}{'return':>8}{'excess':>8}{'worst':>8}{'risk':>7}{'ret/risk':>9}{'beta':>6}{'invested':>9}"
             f"{f'vs {first}':>12}{'t':>6}{'1st half':>10}{'2nd half':>10}{'holds':>7}"]
    for r in summary.itertuples():
        holds = "" if r.variant == first else ("yes" if r.holds else "no")
        lines.append(f"{r.variant:<19}{r.mean_ret:>+8.2%}{r.mean_excess:>+8.2%}{r.worst_ret:>+8.1%}"
                     f"{r.risk:>7.2%}{r.ret_per_risk:>9.2f}{r.beta:>6.2f}{r.mean_invested:>9.0%}"
                     f"{r.vs_first:>+12.2%}{r.t_vs_first:>+6.1f}{r.vs_first_h1:>+10.2%}{r.vs_first_h2:>+10.2%}{holds:>7}")
    r = summary.iloc[0]
    lines.append(f"{'benchmark':<19}{r.mean_bench:>+8.2%}{'':>16}{r.bench_risk:>7.2%}{r.bench_ret_per_risk:>9.2f}{1:>6.2f}{1:>9.0%}")
    return "\n".join(lines)


def main_daily(market: Market, variants: list[Variant], args) -> None:
    print(f"One book from {min(market.sheets)} to {max(market.sheets)}, {len(market.sheets)} decision days, "
          f"{len(variants)} variant(s) to run")
    daily = run_daily(market, variants, args.workers)
    known = RESULTS / "system_backtest_daily.csv"
    if args.variants and known.exists():  # a partial run replaces its variants and keeps the others
        old = pd.read_csv(known)
        if old["date"].min() == daily["date"].min() and old["date"].max() == daily["date"].max():
            daily = pd.concat([old[~old["variant"].isin(daily["variant"])], daily], ignore_index=True)
    variants = [v for v in VARIANTS if v.name in set(daily["variant"])]
    summary = summarise_daily(daily, variants)
    RESULTS.mkdir(exist_ok=True)
    daily.round({"value": 2, "bench": 2, "riskfree": 6, "invested": 4}).to_csv(known, index=False)
    summary.to_csv(RESULTS / "system_backtest_daily_summary.csv", index=False)
    print(show_daily(summary))
    print(f"\nWritten in {RESULTS}: system_backtest_daily.csv, system_backtest_daily_summary.csv")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backtest the code-only system on past 3-week windows.")
    parser.add_argument("--since", type=date.fromisoformat, default=date(2022, 1, 3), help="first window, YYYY-MM-DD")
    parser.add_argument("--variants", nargs="*", help="names from VARIANTS (default: all); the other variants "
                                                      "already in the results file are kept")
    parser.add_argument("--workers", type=int, help="processes (default: one per core)")
    parser.add_argument("--daily", action="store_true", help="one book over the whole history, valued every day")
    args = parser.parse_args(argv)
    variants = [v for v in VARIANTS if not args.variants or v.name in args.variants]
    from data import data_agent
    prices = data_agent.fetch_prices(args.since - timedelta(days=PRICE_DAYS + 30), date.today())
    market, starts = prepare(prices, data_agent.compute_features(prices), args.since)
    if args.daily:
        main_daily(market, variants, args)
        return
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
    halves = study_by_half(market)
    halves.to_csv(RESULTS / "system_backtest_signals_by_half.csv", index=False)
    print(f"\nThe same at {settings.HORIZON_SESSIONS} sessions, in each half of the history (edge and t-value):\n" + show_by_half(halves))
    print("\nWhat the fill rule is worth (an order of day T filled at the close of T-1):\n" + show_fill_rule(rule))
    print(f"\nWritten in {RESULTS}: system_backtest.csv, system_backtest_summary.csv, system_backtest_signals.csv, "
          "system_backtest_fill_rule.csv")


if __name__ == "__main__":
    main()
