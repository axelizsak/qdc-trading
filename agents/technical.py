"""[2] Technical agent. Owner: Person 2. Two strategies (settings in agents/strategy_config.py).

Input:  prices (Prices format) + indicators from the data agent (Features format, full history)
Output:
  1. build_sheet: one row per ETF with the answers for the Portfolio Manager
     Strategy 1, momentum (momentum_strategy): for ETFs in a strong trend
     - direction: trend on the short, medium and long term (up / down / none)
     - momentum setup: buy / avoid / none
     - strength: Clenow trend quality, ADX label, momentum and volume confirmations
     - measures: momentum 13612 (to rank), vol 60 (to size), market trend (context)
     - suggested signal: buy / watch / neutral / avoid, with a one-line rationale
     Strategy 2, mean reversion (mean_reversion_strategy): for equity ETFs without a strong trend
     - regime: trend (momentum fits) / range (mean reversion fits) / mixed
     - mean reversion setup: buy / avoid / none / n/a
       oversold or overbought by Connors RSI 2, Bollinger z-score 20, Donchian 20-day channel
     - rebound done: price back above its SMA 5 (Connors exit)
  2. run: two scores per ETF in the Signals format, for the pipeline.
     - technical_det: the momentum suggested signal, by rule. The mean reversion answers are in the sheet
       and the rationale; how to combine both into one rule is decided after their backtests.
     - technical_llm: the LLM's own score from the same sheet (agents/technical_llm.py). It weighs what the
       rules cannot: a buy setup on a stretched ETF, a trend without confirmation, an inverse ETF.

Leveraged and inverse ETFs: direction and strength are read on the underlying index
(column `underlying` of data/etf_universe.csv), then reversed for an inverse ETF.
Momentum 13612 and volatility are read on the ETF itself.
"""
from datetime import date

import numpy as np
import pandas as pd

from agents import strategy_config as cfg
from agents import technical_llm
from data.features import signal_prices
from data.schemas import validate_signals
from data.universe import load_universe

AGENT_DET = "technical_det"
AGENT_LLM = "technical_llm"


# ---------- Question 1: direction ------------------------------------------------------------

def trend_answer(price: float, sma: float, ret: float) -> str | None:
    """'up' if the price is above its moving average and the return is positive,
    'down' if both are negative, 'none' if they disagree. None when a value is missing."""
    if pd.isna(price) or pd.isna(sma) or pd.isna(ret):
        return None
    above_average = price > sma
    positive_return = ret > 0
    if above_average and positive_return:
        return "up"
    if (not above_average) and (not positive_return):
        return "down"
    return "none"


def reverse_answer(answer: str | None) -> str | None:
    """Trend of an inverse ETF: the opposite of its index."""
    if answer == "up":
        return "down"
    if answer == "down":
        return "up"
    return answer


def momentum_setup(short: str | None, medium: str | None, long: str | None) -> str:
    """'buy' if long and medium term are up and the short term is not down,
    'avoid' if long and medium term are both down, 'none' otherwise (or when data is missing)."""
    if long == "up" and medium == "up" and short is not None and short != "down":
        return "buy"
    if long == "down" and medium == "down":
        return "avoid"
    return "none"


def trend_direction(setup: str, long: str | None) -> str | None:
    """Direction the confirmations are checked against: the setup's, or else the long-term trend."""
    if setup == "buy":
        return "up"
    if setup == "avoid":
        return "down"
    if long == "up" or long == "down":
        return long
    return None


# ---------- Question 2: strength -------------------------------------------------------------

def adx_label(adx: float) -> str | None:
    """'strong' (ADX > 25), 'moderate' (20 to 25) or 'absent' (< 20). None when missing (BEL 20)."""
    if pd.isna(adx):
        return None
    if adx > cfg.ADX_STRONG:
        return "strong"
    if adx >= cfg.ADX_MODERATE:
        return "moderate"
    return "absent"


def momentum_confirms(rsi: float, direction: str | None) -> bool | None:
    """True when RSI 14 is above 50 in an uptrend (below 50 in a downtrend)."""
    if pd.isna(rsi) or direction is None:
        return None
    if direction == "up":
        return bool(rsi > cfg.RSI_MIDLINE)
    return bool(rsi < cfg.RSI_MIDLINE)


def volume_confirms(obv_change: float, direction: str | None) -> bool | None:
    """True when the OBV rose over the last 20 quotes in an uptrend (fell in a downtrend)."""
    if pd.isna(obv_change) or direction is None:
        return None
    if direction == "up":
        return bool(obv_change > 0)
    return bool(obv_change < 0)


def trend_quality(prices: pd.Series) -> float:
    """Clenow trend quality: annualized slope of an exponential regression times its R squared.

    A fast and regular rise gives a high value, a slow or choppy one a value near 0, a fall a negative
    value. Needs QUALITY_DAYS prices, otherwise NA.
    """
    prices = prices.dropna()
    if len(prices) < cfg.QUALITY_DAYS:
        return np.nan
    window = prices.iloc[-cfg.QUALITY_DAYS:].to_numpy(dtype=float)
    days = np.arange(len(window))
    log_prices = np.log(window)

    slope, intercept = np.polyfit(days, log_prices, 1)
    fitted = intercept + slope * days
    residual_sum = 0.0
    total_sum = 0.0
    mean_log = log_prices.mean()
    for i in range(len(log_prices)):
        residual_sum = residual_sum + (log_prices[i] - fitted[i]) ** 2
        total_sum = total_sum + (log_prices[i] - mean_log) ** 2
    if total_sum == 0:
        return np.nan
    r_squared = 1 - residual_sum / total_sum

    annualized_slope = np.exp(slope * cfg.TRADING_DAYS_PER_YEAR) - 1
    return float(annualized_slope * r_squared)


# ---------- Mean reversion strategy ----------------------------------------------------------

def channel_position(price: float, low: float, high: float) -> float:
    """Position of the price in its Donchian channel: 0 = on the low, 1 = on the high. NA if flat or missing."""
    if pd.isna(price) or pd.isna(low) or pd.isna(high) or high == low:
        return np.nan
    return float((price - low) / (high - low))


def oversold_signals(rsi_2: float, zscore: float, position: float) -> dict:
    """The three classic 'price stretched down' measures, each True / False / None (missing)."""
    signals = {"connors_rsi2": None, "bollinger": None, "donchian": None}
    if not pd.isna(rsi_2):
        signals["connors_rsi2"] = bool(rsi_2 < cfg.RSI2_OVERSOLD)
    if not pd.isna(zscore):
        signals["bollinger"] = bool(zscore <= cfg.ZSCORE_LOW)
    if not pd.isna(position):
        signals["donchian"] = bool(position < cfg.DONCHIAN_LOW)
    return signals


def overbought_signals(rsi_2: float, zscore: float, position: float) -> dict:
    """The same three measures, 'price stretched up'."""
    signals = {"connors_rsi2": None, "bollinger": None, "donchian": None}
    if not pd.isna(rsi_2):
        signals["connors_rsi2"] = bool(rsi_2 > cfg.RSI2_OVERBOUGHT)
    if not pd.isna(zscore):
        signals["bollinger"] = bool(zscore >= cfg.ZSCORE_HIGH)
    if not pd.isna(position):
        signals["donchian"] = bool(position > cfg.DONCHIAN_HIGH)
    return signals


def names_true(signals: dict) -> list[str]:
    """Names of the signals that say yes, e.g. ['connors_rsi2', 'bollinger']."""
    names = []
    for name, answer in signals.items():
        if answer is True:
            names.append(name)
    return names


def mean_reversion_setup(applicable: bool, long: str | None, adx: float,
                         oversold: list[str], overbought: list[str]) -> str:
    """'buy': long-term trend not down, no strong trend, price stretched down (1 measure is enough).
    'avoid': price stretched up (2 measures out of 3). 'none' otherwise.
    'n/a' for ETFs without timing (gold, crypto, leveraged...)."""
    if not applicable:
        return "n/a"
    if len(overbought) >= cfg.MR_MIN_OVERBOUGHT:
        return "avoid"
    if long is None or long == "down":
        return "none"                       # no falling knife
    if not pd.isna(adx) and adx > cfg.MR_ADX_MAX:
        return "none"                       # strong trend: the momentum strategy applies
    if len(oversold) >= cfg.MR_MIN_OVERSOLD:
        return "buy"
    return "none"


def rebound_done(price: float, sma_5: float) -> bool | None:
    """Connors exit: the rebound is done once the price is back above its SMA 5."""
    if pd.isna(price) or pd.isna(sma_5):
        return None
    return bool(price > sma_5)


def market_regime(adx: float, efficiency: float) -> str | None:
    """Which strategy fits this ETF now: 'trend' (ADX > 25) -> momentum,
    'range' (ADX < 20 and efficiency < 0.2) -> mean reversion, 'mixed' otherwise."""
    if pd.isna(adx):
        return None
    if adx > cfg.ADX_STRONG:
        return "trend"
    if adx < cfg.RANGE_ADX_MAX and not pd.isna(efficiency) and efficiency < cfg.RANGE_EFFICIENCY_MAX:
        return "range"
    return "mixed"


def mean_reversion_strategy(last: pd.Series, price: float, long: str | None, applicable: bool) -> dict:
    """All the mean reversion answers for one ETF on one day."""
    rsi_2 = value(last, "rsi_2")
    zscore = value(last, "zscore_20")
    position = channel_position(price, value(last, "low_20"), value(last, "high_20"))
    oversold = names_true(oversold_signals(rsi_2, zscore, position))
    overbought = names_true(overbought_signals(rsi_2, zscore, position))
    adx = value(last, "adx_14")
    return {
        "regime": market_regime(adx, value(last, "efficiency_ratio_20")),
        "mr_setup": mean_reversion_setup(applicable, long, adx, oversold, overbought),
        "rsi_2": rsi_2,
        "zscore_20": zscore,
        "channel_position_20": position,
        "mr_oversold": ", ".join(oversold),
        "mr_overbought": ", ".join(overbought),
        "rebound_done": rebound_done(price, value(last, cfg.REBOUND_DONE_SMA)),
    }


# ---------- Measures for the Portfolio Manager -------------------------------------------------

def momentum_13612(prices: pd.Series) -> float:
    """Average of the 1, 3, 6 and 12-month returns (Faber's 13612). NA without 12 months of prices."""
    prices = prices.dropna()
    longest = max(cfg.MOMENTUM_13612_DAYS)
    if len(prices) <= longest:
        return np.nan
    today = float(prices.iloc[-1])
    returns = []
    for days in cfg.MOMENTUM_13612_DAYS:
        past = float(prices.iloc[-1 - days])
        returns.append(today / past - 1)
    return float(sum(returns) / len(returns))


def market_trend(price: float, sma_200: float) -> str | None:
    """'up' when WORLD is above its SMA 200, 'down' below. None when missing."""
    if pd.isna(price) or pd.isna(sma_200):
        return None
    if price > sma_200:
        return "up"
    return "down"


# ---------- Suggested signal -----------------------------------------------------------------

def suggest_signal(setup: str, market: str | None, strength: str | None) -> str:
    """buy / watch / neutral / avoid (rules in strategy_config.py)."""
    if setup == "avoid":
        return "avoid"
    if setup != "buy":
        return "neutral"
    if market == "down":
        return "watch"
    if strength == "absent":
        return "watch"
    return "buy"


def count_confirmations(momentum: bool | None, volume: bool | None) -> int:
    """Number of confirmations that say yes (a missing one counts as no)."""
    count = 0
    if momentum is True:
        count = count + 1
    if volume is True:
        count = count + 1
    return count


def write_rationale(row: dict) -> str:
    """One line a person can read, e.g.
    'ST none, MT up, LT up -> buy setup; strength absent (ADX 16), 1/2; market up -> watch'."""
    labels = {"up": "up", "down": "down", "none": "none", None: "n/a"}
    text = (f"ST {labels[row['trend_short']]}, MT {labels[row['trend_medium']]}, "
            f"LT {labels[row['trend_long']]} -> {row['momentum_setup']} setup")
    if pd.isna(row["adx_14"]):
        text = text + f"; strength n/a, {row['confirmations']}/2"
    else:
        text = text + f"; strength {row['trend_strength']} (ADX {row['adx_14']:.0f}), {row['confirmations']}/2"
    side = "market" if row["leveraged"] >= 0 else "market (reversed for an inverse ETF)"
    text = text + f"; {side} {labels[row['market_trend']]} -> {row['suggested_signal']}"
    if row.get("mr_setup") is not None and row["mr_setup"] != "n/a":
        text = text + f" | regime {labels.get(row['regime'], row['regime'])}; mean reversion {row['mr_setup']}"
        if row["mr_oversold"] != "":
            text = text + f" (oversold: {row['mr_oversold']})"
        if row["mr_overbought"] != "":
            text = text + f" (overbought: {row['mr_overbought']})"
    if row["read_on"] != row["ticker"]:
        text = text + f" (read on {row['read_on']})"
    return text


# ---------- Data access ----------------------------------------------------------------------

def indicator_table(features: pd.DataFrame) -> pd.DataFrame:
    """Features (long format) back to one row per (date, ticker), one column per indicator."""
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.date
    wide = features.pivot_table(index=["date", "ticker"], columns="feature", values="value")
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def price_history(prices: pd.DataFrame, ticker: str, as_of: date) -> pd.Series:
    """Prices adjusted for dividends of one ETF up to as_of (raw prices if the source gave none)."""
    rows = prices[(prices["ticker"] == ticker) & (prices["date"] <= as_of)].sort_values("date")
    if len(rows) == 0:
        return pd.Series(dtype=float)
    close = signal_prices(rows)[0]
    return pd.Series(close.to_numpy(), index=rows["date"].to_numpy())


def indicator_history(indicators: pd.DataFrame, ticker: str, as_of: date) -> pd.DataFrame:
    """Indicator rows of one ETF up to as_of, oldest first."""
    rows = indicators[(indicators["ticker"] == ticker) & (indicators["date"] <= as_of)]
    return rows.sort_values("date").reset_index(drop=True)


def value(row: pd.Series, column: str) -> float:
    """A value from an indicator row, NA if the column does not exist."""
    if column in row.index:
        return row[column]
    return np.nan


# ---------- One ETF, then all ETFs -----------------------------------------------------------

def sheet_for_one_etf(etf: pd.Series, prices: pd.DataFrame, indicators: pd.DataFrame,
                      market: str | None, as_of: date) -> dict | None:
    """The answers for one ETF, or None when there is no data for it."""
    ticker = etf["ticker"]
    read_on = ticker
    if isinstance(etf["underlying"], str) and etf["underlying"] != "":
        read_on = etf["underlying"]
    inverse = etf["leveraged"] < 0

    index_rows = indicator_history(indicators, read_on, as_of)
    index_prices = price_history(prices, read_on, as_of)
    own_rows = indicator_history(indicators, ticker, as_of)
    own_prices = price_history(prices, ticker, as_of)
    if len(index_rows) == 0 or len(index_prices) == 0:
        return None
    last = index_rows.iloc[-1]
    price = float(index_prices.iloc[-1])

    vol = np.nan
    if len(own_rows) > 0:
        vol = value(own_rows.iloc[-1], cfg.VOL_COLUMN)

    row = {
        "date": as_of,
        "ticker": ticker,
        "family": etf["family"],
        "read_on": read_on,
        "leveraged": int(etf["leveraged"]),
    }

    # Strategy 1: momentum (read on the index for leveraged and inverse ETFs)
    momentum = momentum_strategy(index_rows, index_prices, inverse)
    for key, val in momentum.items():
        row[key] = val
    row["momentum_13612"] = momentum_13612(own_prices)
    row["vol_60"] = vol
    # An inverse ETF rises when the market falls: for it the market filter is read the other way round
    row["market_trend"] = reverse_answer(market) if inverse else market
    row["suggested_signal"] = suggest_signal(momentum["momentum_setup"], row["market_trend"], momentum["trend_strength"])

    # Strategy 2: mean reversion (equities only, never on leveraged or inverse ETFs)
    applicable = bool(etf["use_timing"]) and int(etf["leveraged"]) == 1
    reversion = mean_reversion_strategy(last, price, momentum["trend_long"], applicable)
    for key, val in reversion.items():
        row[key] = val

    row["rationale"] = write_rationale(row)
    return row


def momentum_strategy(index_rows: pd.DataFrame, index_prices: pd.Series, inverse: bool) -> dict:
    """All the momentum answers for one ETF on one day (direction, setup, strength)."""
    last = index_rows.iloc[-1]
    price = float(index_prices.iloc[-1])

    # Question 1: direction, on the index (or the ETF itself)
    answers = {}
    for horizon, columns in cfg.HORIZONS.items():
        answers[horizon] = trend_answer(price, value(last, columns["sma"]), value(last, columns["ret"]))
    index_setup = momentum_setup(answers["short"], answers["medium"], answers["long"])
    index_direction = trend_direction(index_setup, answers["long"])

    # Question 2: strength, on the index
    adx = value(last, "adx_14")
    momentum_ok = momentum_confirms(value(last, "rsi_14"), index_direction)
    obv_change = np.nan
    obv = index_rows["obv"] if "obv" in index_rows.columns else pd.Series(dtype=float)
    if len(obv) > cfg.OBV_DAYS:
        obv_change = obv.iloc[-1] - obv.iloc[-1 - cfg.OBV_DAYS]
    volume_ok = volume_confirms(obv_change, index_direction)
    quality = trend_quality(index_prices)

    # An inverse ETF goes the other way
    if inverse:
        for horizon in answers:
            answers[horizon] = reverse_answer(answers[horizon])
        if not pd.isna(quality):
            quality = -quality

    return {
        "trend_short": answers["short"],
        "trend_medium": answers["medium"],
        "trend_long": answers["long"],
        "momentum_setup": momentum_setup(answers["short"], answers["medium"], answers["long"]),
        "trend_quality_90": quality,
        "adx_14": adx,
        "trend_strength": adx_label(adx),
        "momentum_confirms": momentum_ok,
        "volume_confirms": volume_ok,
        "confirmations": count_confirmations(momentum_ok, volume_ok),
    }


def build_sheet(prices: pd.DataFrame, features: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """One row per scored ETF at as_of: the answers for the Portfolio Manager."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    indicators = indicator_table(features)

    market = None
    world_rows = indicator_history(indicators, cfg.MARKET_TICKER, as_of)
    world_prices = price_history(prices, cfg.MARKET_TICKER, as_of)
    if len(world_rows) > 0 and len(world_prices) > 0:
        market = market_trend(float(world_prices.iloc[-1]), value(world_rows.iloc[-1], "sma_200"))

    rows = []
    for _, etf in load_universe().iterrows():
        if not bool(etf["buyable"]):
            continue
        if etf["family"] in cfg.SKIP_FAMILIES:
            continue
        row = sheet_for_one_etf(etf, prices, indicators, market, as_of)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def signals_from_sheet(sheet: pd.DataFrame) -> pd.DataFrame:
    """The suggested signal in the Signals format: score from SIGNAL_SCORES, confidence = confirmations / 2."""
    rows = []
    for _, row in sheet.iterrows():
        rows.append({
            "ticker": row["ticker"],
            "date": row["date"],
            "agent": AGENT_DET,
            "score": cfg.SIGNAL_SCORES[row["suggested_signal"]],
            "confidence": row["confirmations"] / 2,
            "rationale": row["rationale"],
        })
    return validate_signals(pd.DataFrame(rows, columns=["ticker", "date", "agent", "score", "confidence", "rationale"]))


def run(prices: pd.DataFrame, features: pd.DataFrame, as_of: date, lessons: str = "", client=None,
        use_llm: bool = True) -> pd.DataFrame:
    """Signals format: one technical_det row per scored ETF, and its technical_llm row when the LLM layer
    answered (it needs ANTHROPIC_API_KEY; without it, or with use_llm=False, only technical_det)."""
    sheet = build_sheet(prices, features, as_of)
    det = signals_from_sheet(sheet)
    if not use_llm or sheet.empty:
        return det
    llm = technical_llm.interpret(sheet, as_of, lessons, client)
    return validate_signals(pd.concat([det, llm], ignore_index=True)) if len(llm) else det
