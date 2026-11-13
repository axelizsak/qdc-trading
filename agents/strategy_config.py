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

# Suggested signal, and the number sent in the Signals table (technical_det)
#   buy     : buy setup, market above its SMA 200 and a trend that is not absent (ADX >= 20)
#   watch   : buy setup, but the market is below its SMA 200 or the trend is absent
#   neutral : no setup
#   avoid   : avoid setup (long and medium term both down)
SIGNAL_SCORES = {"buy": 1.0, "watch": 0.5, "neutral": 0.0, "avoid": -1.0}

# ETFs the agent does not score
SKIP_FAMILIES = ["cash"]   # OVERNIGHT CASH is the parking, not an investment
