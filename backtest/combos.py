"""Combinations of indicators to backtest. Owner: Person 2. EDIT THIS FILE to add your own.

Each combination is a list of conditions (indicator, operator, threshold), all required ("and").
Operators: ">", "<", ">=", "<=", "==".
Indicators: any column of data/features.py, plus the trend answers trend_short, trend_medium,
trend_long (+1 up, 0 none, -1 down), market_up, dist_sma20/50/200, range_position_20,
obv_trend_20, ret_12m, momentum_13612 (see backtest/momentum_dataset.py).

Rules to stay honest:
  1. write a combination and its reason BEFORE looking at the results;
  2. keep the list short (about 10);
  3. if you change a combination after seeing the 2025-2026 results, note it in SEEN_TEST:
     its test result is then no longer a real check.
"""

COMBOS = {
    # Faber: the classic long-term trend filter
    "long_trend_sma200": [
        ("dist_sma200", ">", 0),
    ],
    # Two horizons must agree: a long-term uptrend still going on over 3 months
    "long_and_medium_up": [
        ("trend_long", "==", 1),
        ("trend_medium", "==", 1),
    ],
    # Our current rule (agents/technical.py)
    "our_setup": [
        ("trend_long", "==", 1),
        ("trend_medium", "==", 1),
        ("trend_short", ">=", 0),
    ],
    # A long-term uptrend that moves in a straight line (Clenow's idea of trend quality)
    "long_trend_clean": [
        ("dist_sma200", ">", 0),
        ("efficiency_ratio_20", ">", 0.3),
    ],
    # A long-term uptrend with a strong trend (ADX convention)
    "long_trend_strong_adx": [
        ("trend_long", "==", 1),
        ("adx_14", ">", 25),
    ],
    # A long-term uptrend where buyers still dominate
    "long_trend_rsi_above_50": [
        ("trend_long", "==", 1),
        ("rsi_14", ">", 50),
    ],
    # A long-term uptrend confirmed by volume
    "long_trend_volume": [
        ("trend_long", "==", 1),
        ("obv_trend_20", "==", 1),
    ],
    # Buy the dip: long-term uptrend, but a short-term pullback (bridge to mean reversion)
    "pullback_in_uptrend": [
        ("trend_long", "==", 1),
        ("rsi_14", "<", 40),
    ],
    # A calm uptrend: long-term uptrend while volatility is falling
    "calm_uptrend": [
        ("trend_long", "==", 1),
        ("vol_ratio_20_60", "<", 1),
    ],
    # The opposite, expected to do WORSE than average: long and medium term both down
    "downtrend_to_avoid": [
        ("trend_long", "==", -1),
        ("trend_medium", "==", -1),
    ],
}

# Combinations whose idea came after seeing results on 2025-2026 (their test is not a clean check)
SEEN_TEST = [
    "long_trend_clean",   # the efficiency ratio was singled out by the indicator-by-indicator test of 13/11
]
