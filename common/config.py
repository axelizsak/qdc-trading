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
MAX_ORDERS_PER_DAY = 5
TRANSACTION_COST = 0.0
TRADING_DAYS = [
    date(2026, 12, 7),
    date(2026, 12, 10),
    date(2026, 12, 14),
    date(2026, 12, 17),
    date(2026, 12, 21),
]

# --- Agents ------------------------------------------------------------------
# Agents whose scores the Portfolio Manager combines (Σ weights = 1).
SIGNAL_AGENTS = ["technical", "macro_news", "quant_ml"]

# --- LLM ---------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_FAST = os.getenv("CLAUDE_MODEL_FAST", "claude-haiku-5-5")  # summaries
LLM_REASONING = os.getenv("CLAUDE_MODEL_REASONING", "claude-sonnet-5-5")  # reasoning
