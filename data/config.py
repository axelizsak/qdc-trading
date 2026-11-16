"""Competition rules and shared constants.

Source: Belfius session (27 Oct 2026) and the QDC technical structure doc.
"""
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# logs/ is graded evidence. To try a strategy without touching it, point the whole system at a copy:
#   cp -R logs /tmp/trial && QDC_LOGS_DIR=/tmp/trial streamlit run app/streamlit_app.py
LOGS_DIR = Path(os.environ["QDC_LOGS_DIR"]).resolve() if os.getenv("QDC_LOGS_DIR") else ROOT / "logs"
UNIVERSE_CSV = DATA_DIR / "etf_universe.csv"

load_dotenv(ROOT / ".env")

# --- Competition rules -------------------------------------------------------
START_CAPITAL_EUR = 10_000
BENCHMARK = "WORLD"  # iShares Core MSCI World, "Global equity (benchmark)" in the Belfius ETF list
MAX_ORDERS_PER_DAY = 5
# The price an order of day T fills at. True: the last close before T (what the team understood from Belfius, and
# one slide of their deck). False: the close of T itself (another slide). Every module reads this one setting:
# the book, the cash check, the Portfolio and Risk Managers, the logbook, the grading and the backtest.
# It is not a detail: backtest/README.md shows what the difference is worth. To confirm with Belfius.
FILL_AT_PREVIOUS_CLOSE = True


def fill_rule() -> str:
    """The rule in words, for the memos and the cockpit."""
    return "the last close before its day" if FILL_AT_PREVIOUS_CLOSE else "the close of its own day"


def fills_from(days, order_day: str):
    """Which quote days (ISO strings) an order of `order_day` can fill at: the fill price is the last close
    among them. Strictly before the order day under FILL_AT_PREVIOUS_CLOSE, up to it otherwise."""
    return days < order_day if FILL_AT_PREVIOUS_CLOSE else days <= order_day


# Decimals of an order's units. 4 = fractional units (what the team understood from Belfius, to confirm on the
# order form). Set it to 0 if the form only takes whole units: the Portfolio Manager then sizes in whole units.
ORDER_UNIT_DECIMALS = 4
MAX_REVISIONS = 2  # times a day can go back to the Portfolio Manager with human feedback
TRANSACTION_COST = 0.0
TRADING_DAYS = [
    date(2026, 12, 7),
    date(2026, 12, 10),
    date(2026, 12, 14),
    date(2026, 12, 17),
    date(2026, 12, 21),
]

# --- Agents ------------------------------------------------------------------
# Signal sources the Portfolio Manager combines, one weight each (Σ weights = 1).
# Agents that give every ETF a score. The Technical agent emits two (a deterministic score and an LLM-interpreted
# one), the Macro/News agent two (macro_market: the day's macro views through the exposure table; macro_etf: the
# ETF analysts' own call). The Portfolio Manager weights each one and learns which predicts better.
SIGNAL_AGENTS = ["technical_det", "technical_llm", "macro_market", "macro_etf"]
# Starting weights, before any evidence: each family counts the same, split equally between its agents that
# gave a score that day. An agent that gives no score takes no weight (graph/pipeline.py: load_agent_weights).
AGENT_FAMILIES = {"technical": ["technical_det", "technical_llm"], "macro": ["macro_market", "macro_etf"]}
# Every LLM score (macro_etf, technical_llm) is a view on the next HORIZON_SESSIONS trading sessions, and the
# review loop grades every score on them.
HORIZON_SESSIONS = 3
# The review loop also judges the technical rule score on this many past quote days, recomputed from prices.
REVIEW_BACKFILL_DAYS = 60

# One switch per pipeline step (graph/pipeline.py). True = use the mock from data/mock.py.
# Set your step to False once your real module works. While any step is on a mock,
# the pipeline writes to logs/mock/ (gitignored) instead of logs/.
USE_MOCK = {
    "data": False,
    "technical": False,
    "macro_news": False,
    "portfolio": False,
    "risk": False,
}

# Factors the Macro/News agent forms a view on (one macro_view per factor and day).
MACRO_FACTORS = [
    "ecb_rates",
    "fed_us_rates",
    "inflation",
    "growth_recession",
    "eurusd",
    "oil",
    "energy_weather",
    "gold_silver",
    "copper_metals",
    "uranium",
    "geopolitics_defence",
    "china_em",
    "tech_ai_semis",
    "crypto",
    "risk_appetite",
]

# --- LLM ---------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_FAST = os.getenv("CLAUDE_MODEL_FAST", "claude-haiku-5-5")  # summaries
LLM_REASONING = os.getenv("CLAUDE_MODEL_REASONING", "claude-sonnet-5-5")  # reasoning

# --- Macro / News agent ---------------------------------------------------------
# What +1 means for each factor. A macro view's direction in [-1, 1] follows this sign.
MACRO_FACTOR_MEANING = {
    "ecb_rates": "ECB more hawkish, euro rates higher",
    "fed_us_rates": "Fed more hawkish, US rates higher",
    "inflation": "inflation rising",
    "growth_recession": "growth stronger (-1 = recession)",
    "eurusd": "euro stronger against the dollar",
    "oil": "oil price higher",
    "energy_weather": "European energy and gas prices higher (cold weather, low storage, supply cuts)",
    "gold_silver": "gold and silver higher",
    "copper_metals": "copper and strategic metals higher",
    "uranium": "uranium higher",
    "geopolitics_defence": "geopolitical tension and defence spending rising",
    "china_em": "China and emerging markets outlook improving",
    "tech_ai_semis": "AI and semiconductor cycle strengthening",
    "crypto": "crypto prices higher",
    "risk_appetite": "risk-on (investors buying risk)",
}
MACRO_EXPOSURE_CSV = DATA_DIR / "macro_exposure.csv"

# Web tools. Haiku 5.5 takes the basic web search; dynamic filtering (_20260209) needs Sonnet/Opus.
WEB_SEARCH_TOOL_FAST = "web_search_20250305"
WEB_SEARCH_TOOL_REASONING = "web_search_20260209"
WEB_FETCH_TOOL_REASONING = "web_fetch_20260209"
MACRO_SEARCHES_PER_TOPIC = 3  # web searches per Haiku call (one call per topic)
MACRO_EXPLORE_SEARCHES = 6  # exploration budget (Sonnet)
MACRO_EXPLORE_FETCHES = 4
# ETF analysts: one Sonnet "hedge-fund quant" per theme of the universe, who searches and scores its ETFs.
MACRO_ANALYST_SEARCHES = 6  # web searches per theme
MACRO_SCORECARD_CALLS = 5  # past calls per ETF an analyst sees with what happened next

# Source registry: a source needs this many graded uses before its status can change.
SOURCE_MIN_USES = 3
SOURCE_PROMOTE_HIT_RATE = 0.6  # testing source at or above: proposed to Axel as regular
SOURCE_DROP_HIT_RATE = 0.4  # any source below: dropped

# USD per million tokens (prompts up to 100K tokens) and per web search, for the cost of a run.
# Check https://platform.claude.com/docs/en/about-claude/pricing before relying on them.
LLM_PRICES_USD = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
}
WEB_SEARCH_USD = 0.01
