"""Settings of the Portfolio Manager. Owner: Person 3.

Only numbers and names here, no calculation: to change the strategy, change this file.

The PM chooses as freely as possible. Belfius's rules (5 orders a day, cash never below zero, no shorting, ETFs of
the list) are checked by the pipeline, and two hard limits below apply to every portfolio. The rest is the code's
own portfolio, a suggestion: the LLM proposes its own weights, the final target blends the two, and how much the
LLM counts is learned from how its portfolios did against the code's.
The Risk Manager and the memo's flags tell the human about concentration, theme, volatility and leverage.
"""
from datetime import date

# --- Market and window ---------------------------------------------------------------------------------
MARKET_TICKER = "WORLD"       # the benchmark. Its score is the market view. It is not needed in the portfolio.
BENCHMARK_ALLOWED = False     # True lets the code portfolios buy the benchmark ETF itself when it scores well
IDLE_TO_BENCHMARK = 1.0       # share of the budget left idle that goes into the benchmark ETF (0 = it stays in cash)
BENCHMARK_CAP = 0.60          # largest share of the portfolio that top-up may put in the benchmark ETF
VALUATION_END = date(2026, 12, 23)   # last day of the competition window (assumption: valued at this close)

# --- The code's baseline portfolio (a suggestion the LLM may ignore) ---------------------------------------
# Hard limits on every portfolio the Portfolio Manager builds: the code's, the LLM's and the final blend.
# What a limit cuts stays in cash. They are the Risk Manager's reference levels; set one to 1.0 to switch it off.
MAX_POSITION = 0.30           # largest share of the portfolio in one ETF
MAX_LEVERAGED_SHARE = 0.15    # largest share in leveraged and inverse ETFs together

EXPOSURE_BASE = 0.85          # invested share when the market view is neutral
EXPOSURE_SLOPE = 0.30         # + this x the market ETF's final score (in [-1, 1])
EXPOSURE_MIN = 0.40
EXPOSURE_MAX = 0.99           # a cash cushion of 1 %: the fill price is only known to the cent
SAME_DAY_CUSHION = 0.05       # only when an order fills at the close of its own day: a buy is funded 5 % above the last close, a sell 5 % below
BASELINE_POSITIONS = 8        # ETFs in the baseline
BASELINE_CAP = 0.30           # baseline only: largest single weight
MIN_ENTRY_SCORE = 0.15        # baseline only: final score needed to enter
MIN_HOLD_SCORE = 0.0          # baseline only: a held ETF stays while its score is above this
MIN_TRADE_WEIGHT = 0.02       # no order for a change smaller than this share of the portfolio
MIN_LLM_WEIGHT = 0.005        # LLM weights below this are dropped
VOL_DAYS = 60                 # daily returns used for volatility
VOL_FLOOR = 0.05              # annualised; avoids oversizing a calm ETF in the baseline
VOL_DEFAULT = 0.20            # for an ETF with too little history

# --- Combining the scores ----------------------------------------------------------------------------------
# A score counts for CONFIDENCE_FLOOR even at confidence 0, and fully at confidence 1.
CONFIDENCE_FLOOR = 0.5
FAILED_MARKER = "agent failed"   # rationale prefix of the neutral fallback rows (graph/pipeline.py)

# --- How much the LLM's portfolio counts ----------------------------------------------------------------------------
# final target = trust x the LLM's weights + (1 - trust) x the code's ANCHOR portfolio. The trust starts here and
# then follows the system's learning rule (agents/learning.py): the portfolio that earned more gains weight.
TRUST_START = 0.6

# --- Choosing the 5 orders of the day ----------------------------------------------------------------------
# Impact of an order = how much it reduces the portfolio's ex-ante risk distance to its target (uses the covariance,
# so two correlated orders are near-substitutes). Timing only breaks near-ties: it moves the priority by at most
# +/- TIMING_WEIGHT. It is off on the last trading date, when waiting is no option.
TIMING_WEIGHT = 0.25
TIMING_DAYS = 20              # an ETF far above its average of this many days tends to give some back in the next days
TIMING_Z = 2.0                # a stretch of this many standard deviations gives the full timing score

# --- The code's second portfolio: a mean-variance optimiser --------------------------------------------------
# Scores become expected returns (MU_SCALE x score, a year) and the covariance is annualised and shrunk, so the
# optimiser trades return against the risk of the whole portfolio. ANCHOR is the code portfolio the LLM's is blended
# with and judged against ("optimizer" or "baseline"); both are logged every day.
ANCHOR = "optimizer"
MU_SCALE = 0.10
RISK_AVERSION = 5.0
OPTIMIZER_POSITIONS = 15
OPTIMIZER_CAP = 0.30          # relaxes when too few ETFs qualify, like the baseline's
COV_SHRINK = 0.2              # share of the covariance replaced by its diagonal

# --- Price checks -------------------------------------------------------------------------------------------
STALE_DAYS = 4                # an ETF whose last quote is older than the latest one by more than this is not traded
