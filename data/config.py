"""Competition rules and shared constants.

Source: Belfius session (27 Oct 2026) and the QDC technical structure doc.
"""
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
UNIVERSE_CSV = DATA_DIR / "etf_universe.csv"

load_dotenv(ROOT / ".env")

# --- Competition rules -------------------------------------------------------
START_CAPITAL_EUR = 10_000
BENCHMARK = "WORLD"  # iShares Core MSCI World, "Global equity (benchmark)" in the Belfius ETF list
MAX_ORDERS_PER_DAY = 5
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
# The Technical agent emits two: a deterministic score and an LLM-interpreted one.
SIGNAL_AGENTS = ["technical_det", "technical_llm", "macro_news"]

# One switch per pipeline step (graph/pipeline.py). True = use the mock from data/mock.py.
# Set your step to False once your real module works. While any step is on a mock,
# the pipeline writes to logs/mock/ (gitignored) instead of logs/.
USE_MOCK = {
    "data": True,
    "technical": True,
    "macro_news": True,
    "portfolio": True,
    "risk": True,
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
MACRO_ANALYST_SHARE = 0.5  # macro_news score = (1 - share) x factor views + share x ETF analysts

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
