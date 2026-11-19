# Appendix: the code

**Repository:** https://github.com/axelizsak/qdc-trading (public). **The code as handed in with this report:** the git
tag `midterm-2026-11-22`, https://github.com/axelizsak/qdc-trading/tree/midterm-2026-11-22. **The customer app online:**
https://axelizsak.github.io/qdc-trading/ (no install, synthetic customers). Python 3.11, **514 automated tests**, none
of which calls the network or a model.

## What it is

A multi-agent system that reads the 50 ETFs of the Belfius list every morning. In the competition it trades our
€10,000: the agents propose, a human approves every order. The same agents inform the customer app, where Lena, the
coach, profiles the customer and explains every ETF; ReBel being execution-only, the customer places every order.

## Folders

| Folder | What it holds |
|---|---|
| `data/` | The 50 ETFs (`etf_universe.csv`), prices and indicators, the 3 strategies and 5 archetypes (`profiles.py`), the 15 lessons, every setting (`config.py`), every shared table (`schemas.py`) and its mock (`mock.py`) |
| `agents/` | Technical agent, Macro/News agent (15 macro views, 20 theme analysts), Portfolio Manager, Risk Manager, logbook, the learning rule (`learning.py`), the coach (`profiler.py`), the ETF sheets, the reference portfolios (`model_portfolios.py`), the one door to the LLM (`llm.py`) and to the news (`web_search.py`) |
| `graph/` | `morning.py` (what every customer shares, once a day), `pipeline.py` (a trading day, step by step, with its three human reviews), `review_loop.py` (the learning step) |
| `app/` | The cockpit (Streamlit): Today, Strategies, one page per agent, Learning, Logbook |
| `mobile/` | The customer app: `server.py`, `wallet.py` (what a customer may buy and holds), the phone screens (`static/`), the online demo (`demo.py`), and `mcp_server.py` (the app read-only, from Claude) |
| `backtest/` | The system replayed on the past (`system_backtest.py`, results in `backtest/results/`), and the figures of this report (`report_figures.py`) |
| `logs/` | Everything the agents said and the humans decided, committed as evidence (`logs/README.md` lists every file) |
| `tests/` | One test file per module |

## A trading day

1. **Morning** (`python -m graph.morning`). Prices of the 50 ETFs and 37 indicators; the Macro/News agent (15 macro factors, 20 theme analysts reading the news, every source cited); the sheet of each ETF and the points of the app's charts. Run once and shared by every customer. About 10 to 15 minutes, because the free tier of the LLM allows 14 requests a minute per model and key.
2. **Learning** (`graph/review_loop.py`). Every expert is judged on the last session it can be judged on and re-weighted by one rule: weight × e^(2 × what following it earned), then rescaled to sum to 1, each weight between 5% and 80%. It weighs the four agents and the LLM's portfolio against the code's. Replayed since 2022 on the technical agent's three answers, it favoured the trend answer, which the team now keeps alone. The lessons go to `logs/lessons.md`, which the agents read next.
3. **Agents.** The technical agent gives two scores per ETF (the trend rule, and the LLM's reading of the whole technical sheet); the Macro/News agent two more. The Portfolio Manager combines the four with their learned weights, blends the code's portfolio with the LLM's, keeps it within the strategy the team trades (limits on shares, themes and complex products) and proposes at most 5 orders. Each round is frozen in `logs/proposals/`, so every machine shows the same orders.
4. **Three human reviews in the cockpit.** The orders (approve, change, reject, add, or send back with feedback); the risk (22 measures and an independent reading by the LLM); the executive summary. Each review can send the day back or stop it.
5. **Logbook.** The orders with their fill price (the last close before the day), and an audit trail in which each row is hashed with the one before (`logs/logbook.csv`). When the day is signed off, the reference portfolios of the three strategies take the engine's new view if it moves them by 10% or more.

A day is run once: whoever opens it later reads it from `logs/`, and no agent is asked twice.

## The customer app

Lena asks six or seven questions, each chosen from the previous answers. The **archetype** (one of five) decides how
she helps; the **strategy** (Defensive, Balanced or Opportunity) comes from fixed rules on the suitability answers,
never from the model, and sets the reference portfolio shown and the products open. Each ETF has a sheet: its chart
and what the agents see today in three plain lines (trend, macro, news, with sources and the other side). The
customer places every order; the Portfolio tab compares what they hold with their reference, and when the reference
changes it offers the orders that would follow it, line by line. Learn holds 15 one-minute lessons and the
3-question check that opens leveraged, inverse and crypto ETFs at the Opportunity level.

## How to run it

| To | Command |
|---|---|
| Install | `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` |
| Keys (free) | copy `.env.example` to `.env`: a Gemini key from Google AI Studio, an EODHD key |
| Run the tests | `.venv/bin/python -m pytest` |
| The morning run | `.venv/bin/python -m graph.morning` |
| The cockpit (then Run the day) | `streamlit run app/streamlit_app.py` |
| A day without the cockpit | `.venv/bin/python -m graph.pipeline` |
| The customer app | `.venv/bin/python -m mobile.server`, then http://localhost:8600 |
| The backtest, one book since 2022 | `.venv/bin/python -m backtest.system_backtest` \--daily |
| The figures of this report | `.venv/bin/python -m backtest.report_figures` |
| A trial without touching the logs | `QDC_LOGS_DIR=/tmp/trial` before any command |

## Principles

- **Code calculates, the LLM judges.** No LLM produces a number that code could compute; it reads the numbers and argues.
- **A human approves every order.** The trading rules (5 orders a day, the strategy's limits) are checked in code on every proposal and every approval.
- **One learning rule for every weight**, applied the same way live and in the backtest.
- **Walk-forward only.** A decision uses only what was known that morning. The LLM agents are not backtested (an LLM knows the past): they are judged forward, from their first live day.
- **Free tools only** (the organisers' rule): Gemini on Google AI Studio's free tier, free data sources, free news feeds.
- **Evidence.** Everything in `logs/` is committed: what the agents suggested, what the humans decided, how the weights moved.
- **Sources are data, never instructions.** A source the model cites must be an article a search returned.

## Data used

| Data | Source | Used for |
|---|---|---|
| The 50 ETFs | The Belfius list, `data/etf_universe.csv` (theme, family, ISIN) | Everything |
| Daily prices since 2020 | Yahoo Finance, and EODHD for a few Xetra listings; adjusted closes in EUR | Indicators, learning, backtest, charts |
| 37 indicators | Computed by code from the prices (`data/features.py`) | Technical agent |
| Ex-dividend dates | Yahoo, the next one estimated from the last year (`data/dividends.py`) | Risk Manager |
| News | Publishers' RSS feeds and Bing News RSS, trusted sites only, last 14 days (`agents/web_search.py`) | Macro/News agent |
| Market data | Free public APIs without key: CFTC positioning, VIX and market stress, crypto funding rates, stablecoin supply, Polymarket odds, Open-Meteo forecasts (`agents/macro_sources/`) | Macro/News agent |
| Customers | Synthetic demo customers (`mobile/customers.py`); what they do goes to `logs/mock/`, never to the graded logs | Customer app |
