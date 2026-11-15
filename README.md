# QDC Trading — ReBel Agentic AI Challenge

Multi-agent system for the QTEM Data Challenge (Fall 2026, Belfius × QuantumBlack).
The agents analyse the 50-ETF universe and propose trades. A human approves every order before it is submitted.

## The competition

- **Capital:** €10,000 per team, no transaction costs, sufficient cash required.
- **Trading days:** Mon 7, Thu 10, Mon 14, Thu 17, Mon 21 December 2026.
- **Orders:** max 5 per trading day (side, units, ticker), submitted through the Belfius form.
- **Grading:** performance **and** methodology, critical thinking towards agent outputs, and re-assessment of the model during the trading window.

| Date | Deliverable |
|---|---|
| 22 Nov 2026, 23:59 | Mid-term report (analyses + code) |
| 7–23 Dec 2026 | Trading window |
| 16 Jan 2027, 23:59 | Final report + team organisation annex (the Belfius deck says 17 Jan) |

> **Fill price:** an order on day T fills at the last close before T, on the ETF's own exchange. The Belfius deck says both this and "the closing price of the trading day itself"; the team settled on the previous close, and every module values an order that way. After the first trading day, compare the fills on the Belfius platform with `logs/logbook/<date>.md`.
>
> **Units:** the Portfolio Manager sizes orders in fractional units (4 decimals). To confirm on the Belfius form before 7 December: if it only takes whole units, set `ORDER_UNIT_DECIMALS = 0` in `data/config.py`.

## How it works

Four scores per ETF, one portfolio, three human reviews, and a review loop that grades everything before the next day.

![Workflow of one trading day](docs/workflow.png)

**One day, step by step** (`graph/pipeline.py` runs it; the cockpit's "Run the day" button starts it)

| # | Step | What it does | Code | LLM? |
|---|---|---|---|---|
| 1 | Data | Real prices of the 50 ETFs, 37 indicators, and the ex-dividend dates of the distributing ETFs | `data/data_agent.py`, `data/features.py`, `data/dividends.py` | No |
| 2 | Review loop | Grades every past score against what the ETFs then did, re-weights the agents, writes the lessons | `graph/review_loop.py` | No |
| 3 | Technical | Two scores from prices: `technical_det` (rules: momentum and mean reversion) and `technical_llm` (Claude weighs the rules' answers when they disagree) | `agents/technical.py`, `agents/technical_llm.py` | Second score only |
| 4 | Macro / News | Two scores from the news: `macro_market` (15 macro views through the exposure table) and `macro_etf` (20 theme analysts, each reading the news, the price tape and its own past calls) | `agents/macro_news.py` and `agents/macro_*.py` | Yes |
| 5 | Portfolio Manager | Combines the four scores, builds the target portfolio, proposes at most 5 orders with a memo | `agents/portfolio.py` | Yes |
| 6 | Your review of the orders | Approve, change, reject or add orders | cockpit, Orders page | You |
| 7 | Risk Manager | Risk of the portfolio after those orders: a board, every metric, and an independent reading of the memo | `agents/risk.py` | Reading only |
| 8 | Your review of the risk, then of the summary | Accept, send the day back with feedback (at most twice), or stop: no trade today | cockpit, Orders page | You |
| 9 | Order log and logbook | The Belfius export, and an audit trail of everything said and decided, with a hash chain | `agents/logbook.py` | No |

**How the scores become a portfolio.** Every score is in [-1, 1]. The Portfolio Manager combines them with one weight per agent:

```
final_score(ETF) = Σ w_agent × score_agent(ETF)      with Σ w_agent = 1
```

Before any evidence, each family counts the same (technical 50%, macro 50%), split between its scores. An agent that gives no score that day takes no weight.

**How the system improves by itself.** Each day starts with the review loop:
- every past score is judged on the 3 trading sessions that followed it, and the technical rule score is also replayed on the last 60 days of prices;
- the agents that ranked the ETFs better gain weight, inside fixed bounds and slowly while the evidence is thin (`logs/agent_weights.csv`);
- the facts go into `logs/lessons.md` (which agent is right on which kind of call, the largest misses, the weakest macro sources), which the agents read before they decide;
- the ETF analysts see their own past calls with what happened, the Portfolio Manager learns how far to trust its own LLM, and the Risk Manager corrects its volatility forecast.

**Two rules.** Code calculates and the LLM judges: no LLM ever produces a number that code could compute. Nothing is traded without the three human reviews, and the trading rules (at most 5 orders, enough cash, no selling more than held) are checked in code on every proposal and every approval.

## Adjust the model

Every setting is one line in one file. Change it, run `python -m graph.review_loop` to see how the scores did, or run a day on a copy of the logs (see Setup).

| To change | Where |
|---|---|
| The technical rules: trend horizons, ADX and RSI levels, mean-reversion thresholds, the score of each signal | `agents/strategy_config.py` |
| What the LLM technical layer is asked to weigh | the prompt in `agents/technical_llm.py` |
| The macro factors, and how much each ETF moves with each factor | `MACRO_FACTORS` in `data/config.py`, `data/macro_exposure.csv` |
| What the macro searches look for, and the analysts' brief per theme | `CROSS_TOPICS` in `agents/macro_llm.py`, `THEME_BRIEFS` and the prompt in `agents/macro_analysts.py` |
| How the portfolio is built: invested share, number of positions, risk aversion, timing | `agents/portfolio_config.py` |
| The hard limits on every portfolio: one ETF at most 30%, leveraged and inverse ETFs at most 15% together | `MAX_POSITION`, `MAX_LEVERAGED_SHARE` in `agents/portfolio_config.py` |
| Fractional or whole units in the orders | `ORDER_UNIT_DECIMALS` in `data/config.py` (4, or 0 for whole units) |
| How fast the agent weights move, and their bounds | `LEARN_*` and `AGENT_WEIGHT_*` in `agents/portfolio_config.py` |
| The starting weight of each family of agents | `AGENT_FAMILIES` in `data/config.py` |
| The horizon every score is judged on, and the days of prices the technical score is replayed on | `HORIZON_SESSIONS`, `REVIEW_BACKFILL_DAYS` in `data/config.py` |
| The risk warning levels | `agents/risk_config.py` |
| The competition rules: capital, trading days, orders per day, benchmark | `data/config.py` |
| Which Claude model each agent uses | `CLAUDE_MODEL_FAST`, `CLAUDE_MODEL_REASONING` in `.env` |

## Repository layout

```
qdc-trading/
  data/        prices and indicators (data_agent.py, features.py), the 50 ETFs (etf_universe.csv),
               the rules and settings (config.py), every shared table (schemas.py), mock data (mock.py)
  agents/      technical.py + technical_llm.py + strategy_config.py     the two technical scores
               macro_news.py + macro_llm.py + macro_analysts.py
                 + macro_tape.py + macro_sources/                       the two macro scores
               portfolio*.py                                            the Portfolio Manager
               risk*.py                                                 the Risk Manager
               logbook.py                                               the audit trail
  graph/       pipeline.py (one day, step by step) and review_loop.py (what the system learns)
  app/         the Streamlit cockpit: Portfolio, Orders, Research, Logbook
  backtest/    backtests of the technical strategies
  ml_library/  shared models for those backtests
  logs/        everything the agents said and the humans decided (graded evidence, see logs/README.md)
  tests/       one file per module; no test calls the network or Claude
```

Python modules only: no notebooks in the repo.

## Shared data formats

These were fixed on day 1 and are defined in `data/schemas.py`. Build your module against mock data (`python -m data.mock`) so that nobody has to wait for anyone else.

| Table | Columns |
|---|---|
| Prices | `date, ticker, open_eur, high_eur, low_eur, close_eur, adj_close_eur, volume` (real closes; the adjusted close is for indicators) |
| Signals | `ticker, date, agent, score [-1,1], confidence [0,1], rationale` |
| Portfolio | `ticker, target_weight` |
| Orders | `date, side, ticker, units` (matches the Belfius form) |
| Holdings | `ticker, units` (cash is tracked separately) |
| Features | `date, ticker, feature, value` (long format: the 37 indicators of `data/features.py`) |
| Agent weights | `date, agent, weight, hit_rate` (weights sum to 1 per date) |
| Evidence | `date, factor, source, kind [number\|text\|exploration], value, summary, url` |
| Macro views | `date, factor, direction [-1,1], confidence, rationale, counter_argument, sources` |
| ETF views | `date, ticker, direction [-1,1], confidence, rationale, counter_argument, sources, catalyst_date, catalyst` (theme analysts; the catalyst is the next scheduled event for the ETF, when one was found) |
| Source registry | `source, kind, factors, status [active\|testing\|dropped], first_seen, n_uses, hit_rate` |
| Order log | `date, ticker, ai_side, ai_units, decision [approved\|modified\|rejected\|added], side, units, reason` |
| Incidents | `date, step, error` |
| Risk report | `date, metric, value, limit, status [ok\|warning\|breach], comment` (one row per metric) |
| Reviews | `date, round, gate [orders\|risk\|summary], decision [approved\|revised\|stopped], feedback` |
| PM portfolios | `date, source [baseline\|optimizer\|llm\|final], ticker, weight` (what the Portfolio Manager built each day) |
| Recommendations | `recommendation_id, created_at, date, ticker, side, units, price_eur, price_date, value_eur, weight_now, weight_target, impact, timing, why` (append-only) |
| Risk forecasts | `date, kind [weight\|forecast], name, value` (what each risk assessment was made on, and what it forecast) |
| Risk calibration | `date, vol_scale, days_graded, mean_ratio, floor_breaches, floor_days` |
| Ex-dividends | `ticker, ex_date, amount_eur, basis` (distributing ETFs; `basis` is "paid" or "expected: ..." when estimated from last year's schedule) |
| Macro exposure | `ticker, multiplier, <one column per macro factor>` (`data/macro_exposure.csv`, a proposal for the team to review) |

`factor` is one of `MACRO_FACTORS` in `data/config.py`. List columns (`sources`, `factors`) are `;`-joined strings.

`ticker` is always the Belfius short name from `data/etf_universe.csv` (e.g. `WORLD`, `NASDAQ 3X`), never the Yahoo symbol.

`agent` is one of `technical_det`, `technical_llm`, `macro_market` or `macro_etf` (see `SIGNAL_AGENTS` in `data/config.py`). Each agent family writes two rows per ETF, so the Portfolio Manager weights every score on its own and learns which one predicts better. Before any evidence, each family (technical, macro) counts the same, split between its agents that gave a score that day: an agent that gives no score takes no weight. Rows with agent `macro_news` in `logs/signals.csv` are the single blended score used until 14 November.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your ANTHROPIC_API_KEY and EODHD_API_KEY. Never commit .env
pytest
```

Run a day. Any day works: before 7 December it is practice, to collect the agents' signals every day and test the orders (practice orders never count in the competition, which starts from €10,000). The pipeline pauses at each of the three reviews (the cockpit's Orders page does the same):

```bash
python -m graph.pipeline                                      # today: agents run, orders proposed, pause
python -m graph.pipeline --answer a.json                      # answer the review waiting (format in graph/pipeline.py)
python -m graph.pipeline --approve-all                        # approve every review left, write the order log
```

Every step now runs its real module, so a day writes to `logs/`, which is graded evidence. To try a strategy or a setting without touching it, point the whole system at a copy of the logs:

```bash
cp -R logs /tmp/trial && rm -f /tmp/trial/checkpoints.sqlite*
QDC_LOGS_DIR=/tmp/trial streamlit run app/streamlit_app.py    # or: QDC_LOGS_DIR=/tmp/trial python -m graph.pipeline
```

A day costs about $0.15 of Claude on top of the Macro/News run: the LLM technical score, one Portfolio Manager call per round and one Risk Manager reading per round.

See what the system has learned, without running a day:

```bash
python -m graph.review_loop               # grades the past scores, writes logs/agent_weights.csv and logs/lessons.md
```

Run the Macro/News agent every business day. Claude is the only key it needs:

```bash
python -m agents.macro_news               # today: report in logs/macro/<date>.md, with the cost of the run
```

Open the cockpit (run the day, read the signals and macro views, approve orders, follow the portfolio):

```bash
streamlit run app/streamlit_app.py
```

Each step uses its mock while its switch in `USE_MOCK` (`data/config.py`) is on. Runs on mocks write to `logs/mock/`, which is not committed.

Data agent (prices of the 50 ETFs):

```bash
python -m data.symbols      # step 1, once: Yahoo symbol of each ETF -> data/etf_symbols.csv (about 3 minutes)
python -m data.data_agent   # step 2: daily OHLC prices since 2020 + quality checks -> data/cache/
```

NASDAQ 3X is not on Yahoo for Xetra: its prices come from EODHD (free plan, official Deutsche Boerse
data distributor). Create a free key on eodhd.com and add `EODHD_API_KEY=...` to your `.env`.

## Work split

| Person | Scope |
|---|---|
| Axel | Macro / News agent · LangGraph pipeline and review loop · Streamlit cockpit · repo management |
| Achille | Data agent · ETF mapping · Technical agent · backtests |
| Alexander | Portfolio Manager · Risk Manager · Logbook |

## Contributing

1. Never commit or push to `main`; a git hook refuses it. Only Axel merges pull requests.
2. Ship with `/ship` in Claude Code, or `scripts/ship.sh "what you changed"`: it creates a branch, runs the tests, commits, pushes and opens a PR.
3. Never edit files on github.com, and never commit `.env`, API keys or the Belfius/QuantumBlack documents.
4. A new data format goes into `data/schemas.py` with a mock and a test, in the same PR.
5. Commit everything in `logs/` (graded evidence). Full rules in `CLAUDE.md`.
