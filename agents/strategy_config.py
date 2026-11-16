"""Settings of the technical agent's momentum strategy. Owner: Person 2.

Only numbers and names here, no calculation: to change the strategy, change this file.
All values are fixed before the backtest and are not tuned on its results.
Sources: Moskowitz, Ooi & Pedersen (2012), Faber (2007), Antonacci (dual momentum),
Keller (13612 filter), Clenow (trend quality = regression slope x R squared).
"""

# Question 1: which way is the ETF going, on each horizon?
# Trend is "up" when the price is above its moving average AND the return over the period is positive,
# "down" when both are negative, "none" when they disagree.
HORIZONS = {
    "short": {"sma": "sma_20", "ret": "ret_1m"},          # about 1 month
    "medium": {"sma": "sma_50", "ret": "ret_3m"},         # about 3 months
    "long": {"sma": "sma_200", "ret": "ret_12m_ex_1m"},   # about 1 year, last month skipped
}

# Question 2: how strong is the trend? (information for the Portfolio Manager, never blocks a setup)
ADX_STRONG = 25        # ADX 14 above 25: strong trend
ADX_MODERATE = 20      # between 20 and 25: moderate; below 20: absent
RSI_MIDLINE = 50       # momentum confirms when RSI 14 is on the side of 50 that matches the trend
OBV_DAYS = 20          # volume confirms when OBV moved in the trend's direction over the last 20 quotes
QUALITY_DAYS = 90      # Clenow trend quality: exponential regression over the last 90 quotes

# Measures for the Portfolio Manager
MOMENTUM_13612_DAYS = [21, 63, 126, 252]   # 1, 3, 6 and 12 months of quotes, averaged (Faber, Keller)
TRADING_DAYS_PER_YEAR = 252
VOL_COLUMN = "vol_60"
MARKET_TICKER = "WORLD"                    # market trend = WORLD above or below its SMA 200

# Mean reversion strategy (only for ETFs with use_timing = True: equities)
#   buy setup  : long-term trend not down (no falling knife) AND no strong trend (ADX <= 25)
#                AND the price is stretched down by at least MR_MIN_OVERSOLD of the three classic measures
#   avoid setup: the price is stretched up by at least MR_MIN_OVERBOUGHT of them (not the moment to enter;
#                two are required because being high is normal in an uptrend)
# Sources: Connors RSI 2 (RSI 2 < 10 above the SMA 200), Bollinger bands (close below the lower
# band = z-score 20 <= -2), Donchian channel (price at the bottom 10 % of its 20-day high-low range).
RSI2_OVERSOLD = 10
RSI2_OVERBOUGHT = 90
ZSCORE_LOW = -2.0
ZSCORE_HIGH = 2.0
DONCHIAN_LOW = 0.10       # position in the 20-day channel: 0 = on the low, 1 = on the high
DONCHIAN_HIGH = 0.90
MR_MIN_OVERSOLD = 1       # each measure is a published rule on its own
MR_MIN_OVERBOUGHT = 2
MR_ADX_MAX = 25           # above this the trend is strong: the momentum strategy applies instead
REBOUND_DONE_SMA = "sma_5"  # Connors exit: the rebound is done when the price is back above its SMA 5

# Market regime of each ETF (which strategy fits): strong trend -> momentum, range -> mean reversion
RANGE_ADX_MAX = 20
RANGE_EFFICIENCY_MAX = 0.2

# Suggested signal, and the number sent in the Signals table (technical_det)
#   buy     : buy setup, market above its SMA 200 and a trend that is not absent (ADX >= 20)
#   watch   : buy setup, but the market is below its SMA 200 or the trend is absent (shown, not scored)
#   For an inverse ETF the market filter is reversed: it is in its favour when the market is below its SMA 200.
#   neutral : no setup
#   avoid   : avoid setup (long and medium term both down)
# "watch" scores 0, not +0.5: in the system backtest (252 windows since 2022) the ETFs called "watch" lagged the
# average ETF over the next 3 sessions (-0.12%, t -2.0), and scoring them 0 gave a higher return per window
# (+0.51% against +0.43%) with less risk and a worst window of -4.0% instead of -6.7%. A lead, not a proof:
# rerun `python -m backtest.system_backtest --variants current watch_half` to compare.
SIGNAL_SCORES = {"buy": 1.0, "watch": 0.0, "neutral": 0.0, "avoid": -1.0}

# ETFs the agent does not score
SKIP_FAMILIES = ["cash"]   # OVERNIGHT CASH is the parking, not an investment
