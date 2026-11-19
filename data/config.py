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
# It is not a detail: backtest/README.md shows what the difference is worth.
FILL_AT_PREVIOUS_CLOSE = True
# Dividends are not credited in the competition: a distributing ETF loses its dividend on the ex-date. True keeps
# every distributing ETF (data/etf_universe.csv, `distributing`) out of the portfolio the team trades. The customer
# app's reference portfolios keep them: a real customer is paid the dividend.
EXCLUDE_DISTRIBUTING = True


def fill_rule() -> str:
    """The rule in words, for the memos and the cockpit."""
    return "the last close before its day" if FILL_AT_PREVIOUS_CLOSE else "the close of its own day"


def fills_from(days, order_day: str):
    """Which quote days (ISO strings) an order of `order_day` can fill at: the fill price is the last close
    among them. Strictly before the order day under FILL_AT_PREVIOUS_CLOSE, up to it otherwise."""
    return days < order_day if FILL_AT_PREVIOUS_CLOSE else days <= order_day


# Decimals of an order's units. 4 = fractional units. Set it to 0 if the order form turns out to take whole units
# only: the Portfolio Manager then sizes in whole units.
ORDER_UNIT_DECIMALS = 4
MAX_REVISIONS = 2  # times a day can go back to the Portfolio Manager with human feedback
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
# gave a score that day. An agent that gives no score takes no weight (agents/learning.py: starting_weights).
AGENT_FAMILIES = {"technical": ["technical_det", "technical_llm"], "macro": ["macro_market", "macro_etf"]}
# The technical agent's three answers, each a score per ETF; technical_det is their weighted sum (agents/technical.py).
TECHNICAL_SIGNALS = ["trend", "mean_reversion", "momentum"]
# Every LLM score (macro_etf, technical_llm) is a view on the next HORIZON_SESSIONS trading sessions, and the
# lessons journal grades every score on them.
HORIZON_SESSIONS = 3

# --- Learning: one rule for every weight (agents/learning.py) ------------------------------------------------------
# After each session, every weight is multiplied by e^(LEARNING_SPEED x what following it earned in that session),
# then the weights are scaled to sum to 1, each between WEIGHT_MIN and WEIGHT_MAX. Three sets of weights follow it:
# the technical agent's three answers (replayed from LEARNING_START), the four agents (from their first score), and
# the LLM's portfolio against the code's (from the first one logged). Speed 2: a weight doubles against another once
# its calls have beaten the other's by about 35% in total.
LEARNING_SPEED = 2.0
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.80
LEARNING_START = date(2022, 1, 3)

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
# The challenge allows free tiers only. The model is Gemini on Google AI Studio (agents/llm.py). Its free limits,
# read on 17 November 2026: LLM_FAST and LLM_SEARCH 15 requests a minute and 500 a day each; LLM_REASONING 5 a
# minute and 20 a day, and it often does not answer. When a model does not answer, LLM_FAST answers in its place.
# Each model has its own quota, so two of them are used side by side: what takes time in a run is the 15 requests
# a minute, not the searches (about 1.5 seconds each).
# One free key per team member, each from their own Google account. The first is enough; with several, the
# requests take turns between the keys. Each key keeps its own pace and its own daily quota.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3")
LLM_FAST = os.getenv("LLM_MODEL_FAST", "gemini-3.5-flash-lite")  # the theme analysts
# The customer app's small jobs (reading a typed answer, explaining a line of a sheet). They must be quick. If
# customers ever use up this quota, the open model "gemma-4-26b-a4b-it" is free for 14,400 requests a day, but it
# takes 10 to 35 seconds to answer where this one takes 2.
LLM_APP = os.getenv("LLM_MODEL_APP", "gemini-3.1-flash-lite")
LLM_SEARCH = os.getenv("LLM_MODEL_SEARCH", "gemini-3.1-flash-lite")  # the topic searches and the exploration
LLM_REASONING = os.getenv("LLM_MODEL_REASONING", "gemini-4-hydrogen")  # the few judgement calls of a day
LLM_REQUESTS_PER_MINUTE = 14  # pace of the requests to one model with one key, just under its free limit
LLM_REST_MINUTES = 15  # a model that did not answer, or a key whose daily quota is used, is left alone this long
# News search, free and without a key (agents/web_search.py): the publishers' RSS feeds and Bing News' RSS search,
# with Google News' RSS search when those find little.
WEB_SEARCH_RESULTS = 10  # results the model gets per search
WEB_SEARCH_CHARS = 500  # characters kept of an article's summary
WEB_SEARCH_DAYS = 14  # only articles of the last days: a news agent must not read last year's news
# The main press: one of the two questions put to Bing News is limited to these sites, so that their articles come
# first. Every one of them is in WEB_SEARCH_SITES.
WEB_SEARCH_PRESS = ["reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com", "apnews.com", "bbc.com",
                    "marketwatch.com", "economist.com", "euronews.com"]
# Only these sites are kept. Left open, a news search returns many small sites that copy or invent news.
WEB_SEARCH_SITES = [
    # news agencies and financial press
    "reuters.com", "afp.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com", "apnews.com", "bbc.com", "economist.com",
    "marketwatch.com", "barrons.com", "finance.yahoo.com", "investing.com", "nasdaq.com", "morningstar.com",
    "theguardian.com", "euronews.com", "politico.eu", "lecho.be", "tijd.be", "handelsblatt.com", "asia.nikkei.com",
    "scmp.com", "koreaherald.com", "businesskorea.co.kr", "economictimes.indiatimes.com", "tradingeconomics.com",
    "marketscreener.com", "spglobal.com", "seekingalpha.com", "etf.com", "justetf.com",
    # central banks, statistics and official data
    "ecb.europa.eu", "federalreserve.gov", "atlantafed.org", "newyorkfed.org", "bis.org", "imf.org", "oecd.org",
    "ec.europa.eu", "bls.gov", "bea.gov", "treasury.gov", "cftc.gov", "sec.gov", "fda.gov",
    # energy, metals, shipping
    "eia.gov", "iea.org", "opec.org", "oilprice.com", "agsi.gie.eu", "gie.eu", "kitco.com", "gold.org", "mining.com",
    "lme.com", "world-nuclear-news.org", "sprott.com", "drewry.co.uk", "freightos.com", "hellenicshippingnews.com",
    # technology, health, defence, crypto
    "tsmc.com", "digitimes.com", "eetimes.com", "theregister.com", "fiercebiotech.com", "biopharmadive.com",
    "statnews.com", "defensenews.com", "breakingdefense.com", "coindesk.com", "theblock.co", "farside.co.uk",
    "defillama.com",
]
# The publishers' own feeds, read once per run: recent articles with a summary.
NEWS_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC top news
    "https://www.cnbc.com/id/20910258/device/rss/rss.html",  # CNBC economy
    "https://www.cnbc.com/id/19836768/device/rss/rss.html",  # CNBC energy
    "https://www.cnbc.com/id/15839069/device/rss/rss.html",  # CNBC investing
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",  # MarketWatch
    "https://www.investing.com/rss/news_14.rss",  # Investing.com economy
    "https://www.investing.com/rss/news_11.rss",  # Investing.com commodities
    "https://www.investing.com/rss/news_25.rss",  # Investing.com stock markets
    "https://www.investing.com/rss/news_301.rss",  # Investing.com crypto
    "https://www.ecb.europa.eu/rss/press.html",
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.federalreserve.gov/feeds/speeches.xml",
    "https://oilprice.com/rss/main",
    "https://www.mining.com/feed/",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://www.theblock.co/rss.xml",
    "https://www.world-nuclear-news.org/rss",
    "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.theguardian.com/uk/business/rss",
    "https://www.euronews.com/rss?level=vertical&name=business",
    "https://www.ft.com/rss/home",
    "https://www.economist.com/finance-and-economics/rss.xml",
]

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
# How each factor is named on screen (the cockpit and the customer app).
MACRO_FACTOR_LABELS = {
    "ecb_rates": "ECB rates", "fed_us_rates": "Fed / US rates", "inflation": "Inflation",
    "growth_recession": "Growth", "eurusd": "EUR / USD", "oil": "Oil", "energy_weather": "Energy & weather",
    "gold_silver": "Gold & silver", "copper_metals": "Copper & metals", "uranium": "Uranium",
    "geopolitics_defence": "Geopolitics", "china_em": "China & EM", "tech_ai_semis": "Tech, AI & semis",
    "crypto": "Crypto", "risk_appetite": "Risk appetite",
}
MACRO_EXPOSURE_CSV = DATA_DIR / "macro_exposure.csv"

# News searches of one run. They cost nothing, and a model asks for all of its searches in one request.
MACRO_SEARCHES_PER_TOPIC = 3  # searches per topic (one call per topic)
MACRO_EXPLORE_SEARCHES = 4  # exploration budget
# ETF analysts: one "hedge-fund quant" per theme of the universe, who searches and scores its ETFs.
MACRO_ANALYST_SEARCHES = 4  # searches per theme
MACRO_SCORECARD_CALLS = 5  # past calls per ETF an analyst sees with what happened next

# Source registry: a source needs this many graded uses before its status can change.
SOURCE_MIN_USES = 3
SOURCE_PROMOTE_HIT_RATE = 0.6  # testing source at or above: proposed to Axel as regular
SOURCE_DROP_HIT_RATE = 0.4  # any source below: dropped

