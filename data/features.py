"""[1] Data agent, step 3: technical indicators of every ETF. Owner: Person 2. No LLM.

Input:  prices in the Prices format (date, ticker, open_eur, high_eur, low_eur, close_eur, adj_close_eur, volume)
Output: add_indicators     -> the prices table with one column per indicator added on the right (wide)
        compute_features   -> the same values in the shared Features format (long), for the pipeline

The 37 indicators were chosen one by one (see the project doc on the technical agent). They are
RAW values: moving averages, highs and lows and the OBV stay in euros or units. Reading them
(price / moving average - 1, distance to a high, trend of the OBV...) is the job of agents/technical.py.

All indicators use the ADJUSTED close (adj_close_eur): a dividend payment is not a market move, and the
raw close drops on each ex-dividend day of a distributing ETF. High and low are adjusted with the same
ratio (adj_close_eur / close_eur of the day). So the levels (sma_200, high_20...) are in adjusted euros:
compare them with adj_close_eur, never with close_eur. close_eur stays the price for valuation and orders.
If an ETF has no adjusted close on some day, the whole ETF falls back to the raw close (reported in
`used_raw_close`), so its series never mixes adjusted and raw prices.

Rules:
  - Each ETF is computed on its own rows, in date order. A window of N days = its last N quotes
    (each ETF keeps its own trading days, nothing is filled).
  - Not enough history for a window -> NA. Nothing is invented.
  - Daily return = close / previous available close - 1.
  - Indicators that use high and low (ATR, ADX) are NA for ETFs whose high/low are not reliable
    (HIGH_LOW_UNRELIABLE): for BEL 20, Yahoo often gives a close outside the high-low range.
  - Volume indicators are NA when the source gives no volume.

Quick run on cached prices:  python -m data.features data/cache/prices_ohlc_YYYY-MM-DD.csv
"""
import sys

import numpy as np
import pandas as pd

from data.schemas import FEATURES, PRICES, validate_features, validate_prices

TRADING_DAYS_PER_YEAR = 252

# High and low not reliable for these ETFs: indicators using them are left empty
HIGH_LOW_UNRELIABLE = ["BEL 20"]

# Order of the columns added on the right of the prices table, block by block
FEATURE_BLOCKS = {
    "moving_averages": ["sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
                        "ema_5", "ema_10", "ema_20", "sma_50_slope_10"],
    "momentum": ["ret_10d", "ret_1m", "ret_3m", "ret_6m", "ret_12m_ex_1m"],
    "oscillators": ["rsi_2", "rsi_5", "rsi_14", "zscore_20"],
    "trend_strength": ["efficiency_ratio_20", "adx_14"],
    "support_resistance": ["high_20", "low_20", "high_52w", "low_52w"],
    "volatility": ["vol_20", "vol_60", "vol_ratio_20_60", "downside_vol_60", "atr_14",
                   "bb_upper_20", "bb_lower_20"],
    "risk": ["drawdown_60", "sharpe_63", "worst_day_60"],
    "volume": ["volume_rel_20", "obv", "volume_trend_20_60"],
}

FEATURE_COLUMNS = []
for block_columns in FEATURE_BLOCKS.values():
    for column in block_columns:
        FEATURE_COLUMNS.append(column)


# ---------- Building blocks ---------------------------------------------------------------

def daily_returns(close: pd.Series) -> pd.Series:
    """Close / previous available close - 1. The first row is NA."""
    return close / close.shift(1) - 1


def wilder_average(values: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing (used by RSI, ATR and ADX): each day keeps (window - 1) / window of yesterday's
    average and adds 1 / window of today's value. NA until `window` values are available."""
    return values.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


# ---------- Moving averages (raw levels, in EUR) ---------------------------------------------

def compute_sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average: mean of the last `window` closes."""
    return close.rolling(window, min_periods=window).mean()


def compute_ema(close: pd.Series, window: int) -> pd.Series:
    """Exponential moving average (span = window): recent days weigh more. NA for the first window - 1 days."""
    ema = close.ewm(span=window, adjust=False).mean()
    ema[close.rolling(window, min_periods=1).count() < window] = np.nan
    return ema


def compute_slope(series: pd.Series, days: int) -> pd.Series:
    """Relative change of a series over `days` days: series / series `days` days ago - 1."""
    return series / series.shift(days) - 1


# ---------- Momentum (returns in %) -----------------------------------------------------------

def compute_return(close: pd.Series, days: int, skip: int = 0) -> pd.Series:
    """Return from `days` quotes ago to `skip` quotes ago. skip=0: up to today."""
    return close.shift(skip) / close.shift(days) - 1


# ---------- Oscillators ------------------------------------------------------------------------

def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index (Wilder), from 0 to 100. Above 70: overbought; below 30: oversold.

    100 when the price only went up, 0 when it only went down, 50 when it did not move at all.
    """
    change = close.diff()
    gains = change.clip(lower=0)
    losses = (-change).clip(lower=0)
    average_gain = wilder_average(gains, window)
    average_loss = wilder_average(losses, window)

    rsi = pd.Series(np.nan, index=close.index)
    for i in range(len(close)):
        gain = average_gain.iloc[i]
        loss = average_loss.iloc[i]
        if pd.isna(gain) or pd.isna(loss):
            continue
        if gain == 0 and loss == 0:
            rsi.iloc[i] = 50.0
        elif loss == 0:
            rsi.iloc[i] = 100.0
        else:
            rsi.iloc[i] = 100 - 100 / (1 + gain / loss)
    return rsi


def compute_zscore(close: pd.Series, window: int) -> pd.Series:
    """How many standard deviations the close is from its `window`-day mean (Bollinger logic)."""
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    zscore = (close - mean) / std
    zscore[std == 0] = np.nan          # a flat price has no z-score
    return zscore


# ---------- Trend strength ---------------------------------------------------------------------

def compute_efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Kaufman efficiency ratio, from 0 to 1: net move over `window` days / sum of all daily moves.
    1 = the price went straight in one direction; 0 = it went back and forth."""
    net_move = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    ratio = net_move / path
    ratio[path == 0] = np.nan          # a flat price has no direction
    return ratio


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Largest of: high - low, |high - previous close|, |low - previous close|."""
    previous_close = close.shift(1)
    ranges = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1)
    return ranges.max(axis=1, skipna=False)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average True Range (Wilder), in EUR: the usual size of a day's move, gaps included."""
    return wilder_average(true_range(high, low, close), window)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average Directional Index (Wilder), from 0 to 100: strength of the trend, whatever its direction.
    Below about 20: no clear trend; above about 25: a clear trend."""
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=close.index)
    minus_dm = pd.Series(0.0, index=close.index)
    for i in range(len(close)):
        up = up_move.iloc[i]
        down = down_move.iloc[i]
        if pd.isna(up) or pd.isna(down):
            plus_dm.iloc[i] = np.nan
            minus_dm.iloc[i] = np.nan
            continue
        if up > down and up > 0:
            plus_dm.iloc[i] = up
        if down > up and down > 0:
            minus_dm.iloc[i] = down

    atr = compute_atr(high, low, close, window)
    plus_di = 100 * wilder_average(plus_dm, window) / atr
    minus_di = 100 * wilder_average(minus_dm, window) / atr
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    dx[di_sum == 0] = 0.0
    return wilder_average(dx, window)


# ---------- Support and resistance (raw levels, in EUR, on closes) ---------------------------

def compute_rolling_high(close: pd.Series, window: int) -> pd.Series:
    """Highest close of the last `window` quotes, today included. close == high: a new high (breakout)."""
    return close.rolling(window, min_periods=window).max()


def compute_rolling_low(close: pd.Series, window: int) -> pd.Series:
    """Lowest close of the last `window` quotes, today included. close == low: a new low (support broken)."""
    return close.rolling(window, min_periods=window).min()


# ---------- Volatility and risk -------------------------------------------------------------------

def compute_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Standard deviation of daily returns over `window` days, annualised (x sqrt(252))."""
    return returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_downside_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Volatility of the down days only, annualised: sqrt(mean of min(return, 0)^2) x sqrt(252).
    Up days count as 0, so a trend that rises in big jumps is not penalised."""
    downside = returns.clip(upper=0) ** 2
    return np.sqrt(downside.rolling(window, min_periods=window).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_bollinger(close: pd.Series, window: int, n_std: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """Bollinger bands: `window`-day mean +/- n_std standard deviations. Returns (upper, lower)."""
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return mean + n_std * std, mean - n_std * std


def compute_drawdown(close: pd.Series, window: int) -> pd.Series:
    """Close / highest close of the last `window` quotes - 1. 0 at a high, -0.08 = 8 % below it."""
    return close / close.rolling(window, min_periods=window).max() - 1


def compute_sharpe(returns: pd.Series, window: int) -> pd.Series:
    """Mean daily return / standard deviation over `window` days, annualised (x sqrt(252)).
    No risk-free rate: it measures how steady the move was, not a performance."""
    mean = returns.rolling(window, min_periods=window).mean()
    std = returns.rolling(window, min_periods=window).std()
    sharpe = mean / std * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe[std == 0] = np.nan
    return sharpe


def compute_worst_day(returns: pd.Series, window: int) -> pd.Series:
    """Lowest daily return of the last `window` days (e.g. -0.06 = worst day was -6 %)."""
    return returns.rolling(window, min_periods=window).min()


# ---------- Volume ------------------------------------------------------------------------------

def compute_relative_volume(volume: pd.Series, window: int) -> pd.Series:
    """Today's volume / mean volume of the last `window` days (today included). 2 = twice the usual."""
    mean = volume.rolling(window, min_periods=window).mean()
    relative = volume / mean
    relative[mean == 0] = np.nan
    return relative


def compute_volume_trend(volume: pd.Series, short: int, long: int) -> pd.Series:
    """Mean volume over `short` days / mean volume over `long` days. Above 1: more trading than usual."""
    short_mean = volume.rolling(short, min_periods=short).mean()
    long_mean = volume.rolling(long, min_periods=long).mean()
    trend = short_mean / long_mean
    trend[long_mean == 0] = np.nan
    return trend


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: running total that adds the day's volume when the close rises and subtracts
    it when the close falls. Its level means nothing; its trend does (rising OBV = buyers in control).
    A day without volume data is NA and adds nothing; an ETF without any volume data is all NA."""
    obv = pd.Series(np.nan, index=close.index)
    total = 0.0
    started = False
    for i in range(len(close)):
        day_volume = volume.iloc[i]
        if pd.isna(day_volume):
            continue
        if not started:
            started = True             # first day with a volume: the running total starts at 0
        elif i > 0 and not pd.isna(close.iloc[i - 1]):
            if close.iloc[i] > close.iloc[i - 1]:
                total = total + day_volume
            elif close.iloc[i] < close.iloc[i - 1]:
                total = total - day_volume
        obv.iloc[i] = total
    return obv


# ---------- One ETF, then all ETFs ------------------------------------------------------------

def signal_prices(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, bool]:
    """Close, high and low adjusted for dividends, for the indicators of one ETF.

    Returns (close, high, low, adjusted). adjusted is False when at least one day has no adjusted close:
    then the raw prices are used for the whole ETF, never a mix of both.
    """
    raw_close = rows["close_eur"].astype(float)
    raw_high = rows["high_eur"].astype(float)
    raw_low = rows["low_eur"].astype(float)
    adj_close = pd.to_numeric(rows["adj_close_eur"], errors="coerce").astype(float)

    if adj_close.isna().any():
        return raw_close, raw_high, raw_low, False

    factor = adj_close / raw_close          # 1 for an accumulating ETF, below 1 before each dividend
    return adj_close, raw_high * factor, raw_low * factor, True


def indicators_for_one_etf(rows: pd.DataFrame, use_high_low: bool) -> pd.DataFrame:
    """The 37 indicators of one ETF. `rows`: its prices, sorted by date. Same index as `rows`."""
    close, high, low, _ = signal_prices(rows)
    volume = pd.to_numeric(rows["volume"], errors="coerce").astype(float)
    returns = daily_returns(close)

    out = pd.DataFrame(index=rows.index)

    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        out[f"sma_{window}"] = compute_sma(close, window)
    for window in [5, 10, 20]:
        out[f"ema_{window}"] = compute_ema(close, window)
    out["sma_50_slope_10"] = compute_slope(out["sma_50"], 10)

    # Momentum
    out["ret_10d"] = compute_return(close, 10)
    out["ret_1m"] = compute_return(close, 21)
    out["ret_3m"] = compute_return(close, 63)
    out["ret_6m"] = compute_return(close, 126)
    out["ret_12m_ex_1m"] = compute_return(close, TRADING_DAYS_PER_YEAR, skip=21)

    # Oscillators
    for window in [2, 5, 14]:
        out[f"rsi_{window}"] = compute_rsi(close, window)
    out["zscore_20"] = compute_zscore(close, 20)

    # Trend strength
    out["efficiency_ratio_20"] = compute_efficiency_ratio(close, 20)
    if use_high_low:
        out["adx_14"] = compute_adx(high, low, close, 14)
    else:
        out["adx_14"] = np.nan

    # Support and resistance
    out["high_20"] = compute_rolling_high(close, 20)
    out["low_20"] = compute_rolling_low(close, 20)
    out["high_52w"] = compute_rolling_high(close, TRADING_DAYS_PER_YEAR)
    out["low_52w"] = compute_rolling_low(close, TRADING_DAYS_PER_YEAR)

    # Volatility
    out["vol_20"] = compute_volatility(returns, 20)
    out["vol_60"] = compute_volatility(returns, 60)
    out["vol_ratio_20_60"] = out["vol_20"] / out["vol_60"]
    out["downside_vol_60"] = compute_downside_volatility(returns, 60)
    if use_high_low:
        out["atr_14"] = compute_atr(high, low, close, 14)
    else:
        out["atr_14"] = np.nan
    upper, lower = compute_bollinger(close, 20)
    out["bb_upper_20"] = upper
    out["bb_lower_20"] = lower

    # Risk
    out["drawdown_60"] = compute_drawdown(close, 60)
    out["sharpe_63"] = compute_sharpe(returns, 63)
    out["worst_day_60"] = compute_worst_day(returns, 60)

    # Volume
    out["volume_rel_20"] = compute_relative_volume(volume, 20)
    out["obv"] = compute_obv(close, volume)
    out["volume_trend_20_60"] = compute_volume_trend(volume, 20, 60)

    return out[FEATURE_COLUMNS]


def add_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """The prices table with the 37 indicator columns added on the right (wide format).

    One row per (date, ticker), as in the prices. Sorted by ticker, then date.
    """
    validate_prices(prices)
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    frames = []
    used_raw_close = []
    for ticker, rows in prices.groupby("ticker", sort=True):
        if not signal_prices(rows)[3]:
            used_raw_close.append(ticker)
        use_high_low = ticker not in HIGH_LOW_UNRELIABLE
        indicators = indicators_for_one_etf(rows, use_high_low)
        frames.append(pd.concat([rows[PRICES], indicators], axis=1))

    if len(used_raw_close) > 0:
        print(f"No adjusted close for {len(used_raw_close)} ETFs, indicators on the raw close: "
              f"{', '.join(used_raw_close)}")

    if len(frames) == 0:
        return pd.DataFrame(columns=PRICES + FEATURE_COLUMNS)
    table = pd.concat(frames, ignore_index=True)
    table[FEATURE_COLUMNS] = table[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return table


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """The same indicators in the shared Features format (date, ticker, feature, value).

    Values that cannot be computed (NA) are dropped, as the Features format requires.
    To go back to the wide table: features.pivot_table(index=['date', 'ticker'], columns='feature').
    """
    wide = add_indicators(prices)
    long = wide.melt(id_vars=["date", "ticker"], value_vars=FEATURE_COLUMNS,
                     var_name="feature", value_name="value")
    long = long.dropna(subset=["value"])
    long = long[FEATURES].sort_values(["ticker", "date", "feature"]).reset_index(drop=True)
    return validate_features(long)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m data.features data/cache/prices_ohlc_YYYY-MM-DD.csv")
        sys.exit(1)
    path = sys.argv[1]
    prices = pd.read_csv(path)
    if "adj_close_eur" not in prices.columns:
        prices["adj_close_eur"] = np.nan   # an old cache: indicators fall back to the raw close
    if "volume" not in prices.columns:
        prices["volume"] = np.nan          # an old cache without volume: volume indicators stay empty
    prices = prices[PRICES]
    table = add_indicators(prices)
    out = path.replace("prices_ohlc_", "indicators_")
    table.to_csv(out, index=False)
    last_day = table["date"].max()
    print(f"{table['ticker'].nunique()} ETFs, {len(table)} rows, {len(FEATURE_COLUMNS)} indicators. Written to {out}")
    print(f"\nLast day ({last_day}), a few columns:")
    print(table[table["date"] == last_day][["ticker", "adj_close_eur", "sma_200", "ret_3m", "rsi_14",
                                            "vol_20", "drawdown_60"]].round(3).to_string(index=False))
