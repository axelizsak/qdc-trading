"""Reference levels of the Risk Manager. Owner: Person 3.

Only numbers and names here, no calculation. These are reference levels, not limits: the team gave the portfolio
no strict limits, so crossing one only turns a metric's status into `warning`. The Risk Manager reports and never
advises; the Portfolio Manager decides the allocation and the human decides what to do with a warning.
`breach` is kept for the hard rules of the competition (more than 5 orders, negative cash).

Each metric has one reference level and a direction: "max" warns above it, "min" warns below it.
Two families, so the team can read the portfolio against a fixed yardstick and against the benchmark:
  FIXED     a level in absolute terms (no position above 30 %, volatility above 20 %...)
  RELATIVE  a level against the benchmark ETF (volatility above 1.3 times the benchmark's...)
"""

RISK_DAYS = 60            # daily returns used for volatility, drawdown, beta, risk shares and tracking error
LIQUIDITY_DAYS = 20       # days of traded volume behind the liquidity measure
MIN_RETURNS = 20          # returns needed before a risk figure is calculated

FIXED = {
    "largest_position": (0.30, "max"),      # share of the portfolio in one ETF
    "largest_group": (0.50, "max"),         # share in one group of ETFs (the `bucket` of data/etf_universe.csv)
    "effective_positions": (3.0, "min"),    # 1 / sum of squared weights: how many equal positions it is worth
    "largest_risk_share": (0.40, "max"),    # the biggest position's share of the portfolio's variance
    "independent_bets": (2.0, "min"),       # effective number of independent bets, from the correlations
    "volatility": (0.20, "max"),            # annualised, the post-trade weights on the last RISK_DAYS days
    "drawdown": (-0.10, "min"),             # worst fall from a peak over those days (a negative number)
    "leveraged_share": (0.15, "max"),       # share in leveraged or inverse ETFs
    "gross_exposure": (1.00, "max"),        # sum of weight x |leverage|: above 1 means more market than capital
    "cash_reserve": (0.01, "min"),          # cash share after the orders (below 0 is a breach)
    "liquidity": (0.10, "max"),             # an order's size against the ETF's average daily volume
    "dividend_payers": (0.0, "max"),        # share in distributing ETFs that go ex-dividend before the window ends
    "prob_below_start": (0.55, "max"),      # chance the portfolio ends the window below its value now
    "worst_case_5pct": (-0.08, "min"),      # the 5 % worst return to the end of the window
    "factor_shock_loss": (-0.05, "min"),    # loss if one macro factor moves at full strength against the portfolio
}

RELATIVE = {
    "volatility_vs_benchmark": (1.3, "max"),   # portfolio volatility / benchmark volatility
    "drawdown_vs_benchmark": (1.3, "max"),     # portfolio worst fall / benchmark worst fall
    "beta": (1.3, "max"),                      # sensitivity to the benchmark's daily moves
    "tracking_error": (0.10, "max"),           # annualised standard deviation of the return gap to the benchmark
    "up_capture": (0.70, "min"),               # share of the benchmark's rising days the portfolio keeps
    "down_capture": (1.10, "max"),             # share of the benchmark's falling days the portfolio takes
    "prob_below_benchmark": (0.60, "max"),     # chance the portfolio ends the window behind the benchmark
}

# Metrics shown as percentages in the comments (the rest are plain numbers)
PERCENT = {"largest_position", "largest_group", "largest_risk_share", "volatility", "drawdown", "leveraged_share",
           "cash_reserve", "liquidity", "dividend_payers", "tracking_error", "prob_below_start", "prob_below_benchmark",
           "worst_case_5pct", "factor_shock_loss"}

# --- Outcome odds: where the portfolio ends the window ------------------------------------------------------
BOOTSTRAP_DRAWS = 2000    # resampled paths
BOOTSTRAP_BLOCK = 5       # days kept together in a resample, so short streaks survive
BOOTSTRAP_SEED = 0        # fixed: the same inputs always give the same odds
# The recent average return is a poor forecast: 60 days of a rally would promise another. The drift of the resampled
# days is moved toward a long-run drift, with the recent days counting as much as DRIFT_PRIOR_DAYS days of prior.
DRIFT_PRIOR_ANNUAL = 0.05
DRIFT_PRIOR_DAYS = 252

# --- Macro factors -------------------------------------------------------------------------------------------
FACTOR_SHOCK = 0.05       # a factor moving by 1 (full strength) moves an ETF with exposure 1 by this much

# --- Capture ratios ------------------------------------------------------------------------------------------
CAPTURE_MIN_DAYS = 5      # rising (or falling) benchmark days needed for a capture ratio

# --- Grading the Risk Manager itself -----------------------------------------------------------------------------
GRADE_DAYS = 10           # quotes after a day's fill used to judge its volatility forecast
MIN_GRADE_QUOTES = 5      # fewest quotes after the fill that still give a judgement
GRADE_MIN_DAYS = 3        # judged days needed before the volatility forecast is corrected
CAL_SHRINK = 10           # judged days at which the correction counts for half
CAL_MIN, CAL_MAX = 0.7, 1.6   # the volatility correction stays between these

# --- The board: metrics grouped for the one-screen summary -----------------------------------------------------------
CATEGORIES = {
    "Rules": ["orders", "cash_reserve"],
    "Concentration": ["largest_position", "largest_group", "effective_positions", "largest_risk_share", "independent_bets"],
    "Risk level": ["volatility", "drawdown", "gross_exposure", "leveraged_share", "liquidity"],
    "Against the benchmark": ["volatility_vs_benchmark", "drawdown_vs_benchmark", "beta", "tracking_error",
                              "up_capture", "down_capture"],
    "Outcome odds": ["prob_below_start", "prob_below_benchmark", "worst_case_5pct"],
    "Macro factors": ["factor_shock_loss", "macro_tilt"],
    "Dividends": ["dividend_payers"],
    "Data": ["benchmark_data"],
}
