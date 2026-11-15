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
> **Units:** the Portfolio Manager sizes orders in fractional units (4 decimals). To confirm on the Belfius form before 7 December.

## How it works

Each agent gives every ETF a score in [-1, 1]. The Portfolio Manager combines them with one weight per agent:

```
final_score(ETF) = Σ w_agent × score_agent(ETF)      with Σ w_agent = 1
```

After each trading day, the review loop compares each agent's prediction with the realised return.
Agents that were right gain weight. Lessons go into a journal that the agents read before the next decision.
Every update is logged in `logs/`, and that log is our evidence of model re-assessment.

![Workflow of one trading day](docs/workflow.png)

Nothing is traded without three human reviews: the proposed orders, the risk after those orders, then the executive summary. At each one you approve, send the day back to the Portfolio Manager with your feedback (at most twice a day), or stop it: no trade today. The trading rules (at most 5 orders, enough cash, no selling more than held) are checked in code on every proposal and on every approval.

**Rule:** use code for calculations, and the LLM only where judgement is needed.

| # | Agent | Module | Input | Output | LLM? |
|---|---|---|---|---|---|
| 1 | Data | `data/data_agent.py` | ISIN list | ETF mapping (ISIN → Yahoo ticker), prices, returns, volatility, drawdowns, correlations | No |
| 2 | Technical | `agents/technical.py` | Prices + features | Two scores per ETF: deterministic (indicators + `ml_library` models) and LLM-interpreted | Interpretation layer |
| 3 | Macro / News | `agents/macro_news.py` | Web search (Claude API) | Macro summary + signal per theme (rates, geopolitics, commodities) | Yes |
| 4 | Portfolio Manager | `agents/portfolio.py` | All signals + agent weights + lessons journal + current portfolio + your feedback | Target weights + up to 5 proposed orders + investment memo | Yes |
| 5 | Risk Manager | `agents/risk.py` | The orders you approved + portfolio + the day's macro views | Risk of the portfolio after the orders (concentration, volatility, drawdown, exposure, leverage, liquidity, the benchmark, outcome odds), a one-screen board, and an independent reading of the memo | Numbers: no. Reading: yes |

The Logbook (`agents/logbook.py`, no LLM) joins every signal, recommendation, risk flag, human decision and computed fill into one append-only audit trail with a hash chain; the cockpit's Logbook page shows it.

There is no separate Quant/ML agent. The ML models are in a shared library (`ml_library/`) that the agents call, and every model is scored by the same walk-forward backtest (`backtest/`).

## Repository layout

```
qdc-trading/
  data/        Data agent, ETF mapping, competition rules (config.py), shared data formats (schemas.py), mock data
  agents/      technical, macro_news, portfolio, risk
  ml_library/  Shared ML models (linear regression, random forest, gradient boosting, ARIMA)
  backtest/    Common walk-forward backtest + metrics (same metrics for every model)
  graph/       LangGraph orchestration + review loop
  app/         Streamlit interface (human approval)
  logs/        orders, signals, agent weights, lessons journal
  tests/
  .env.example
```

## Shared data formats

These were fixed on day 1 and are defined in `data/schemas.py`. Build your module against mock data (`python -m data.mock`) so that nobody has to wait for anyone else.

| Table | Columns |
|---|---|
| Prices | `date, ticker, close_eur` |
| Signals | `ticker, date, agent, score [-1,1], confidence [0,1], rationale` |
| Portfolio | `ticker, target_weight` |
| Orders | `date, side, ticker, units` (matches the Belfius form) |
| Holdings | `ticker, units` (cash is tracked separately) |
| Features | `date, ticker, feature, value` (long format; starter set in `STARTER_FEATURES`, to validate) |
| Agent weights | `date, agent, weight, hit_rate` (weights sum to 1 per date) |
| Evidence | `date, factor, source, kind [number\|text\|exploration], value, summary, url` |
| Macro views | `date, factor, direction [-1,1], confidence, rationale, counter_argument, sources` |
| ETF views | `date, ticker, direction [-1,1], confidence, rationale, counter_argument, sources` (theme analysts) |
| Source registry | `source, kind, factors, status [active\|testing\|dropped], first_seen, n_uses, hit_rate` |
| Order log | `date, ticker, ai_side, ai_units, decision [approved\|modified\|rejected\|added], side, units, reason` |
| Incidents | `date, step, error` |
| Risk report | `date, metric, value, limit, status [ok\|warning\|breach], comment` (one row per metric) |
| Reviews | `date, round, gate [orders\|risk\|summary], decision [approved\|revised\|stopped], feedback` |
| PM portfolios | `date, source [baseline\|optimizer\|llm\|final], ticker, weight` (what the Portfolio Manager built each day) |
| Recommendations | `recommendation_id, created_at, date, ticker, side, units, price_eur, price_date, value_eur, weight_now, weight_target, impact, timing, why` (append-only) |
| Risk forecasts | `date, kind [weight\|forecast], name, value` (what each risk assessment was made on, and what it forecast) |
| Risk calibration | `date, vol_scale, days_graded, mean_ratio, floor_breaches, floor_days` |
| Macro exposure | `ticker, multiplier, <one column per macro factor>` (`data/macro_exposure.csv`, a proposal for the team to review) |

`factor` is one of `MACRO_FACTORS` in `data/config.py`. List columns (`sources`, `factors`) are `;`-joined strings.

`ticker` is always the Belfius short name from `data/etf_universe.csv` (e.g. `WORLD`, `NASDAQ 3X`), never the Yahoo symbol.

`agent` is one of `technical_det`, `technical_llm` or `macro_news` (see `SIGNAL_AGENTS` in `data/config.py`). The Technical agent writes two rows per ETF, so the Portfolio Manager gives the deterministic score and the LLM score separate weights, and the review loop can tell whether the LLM layer adds value.

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

A day costs about $0.10 of Claude on top of the Macro/News run: one Portfolio Manager call per round and one Risk Manager reading per round.

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
| Person 1 (Axel) | Macro / News agent · repo management · LangGraph integration · Streamlit interface |
| Person 2 | Data agent · ETF mapping · Technical agent (indicators + LLM layer) · shared ML library · common backtest |
| Person 3 | Portfolio Manager · Risk Manager · review loop |

The split between Person 2 and Person 3 is still to be confirmed.

## Contributing

1. Never commit or push to `main`; a git hook refuses it. Only Axel merges pull requests.
2. Ship with `/ship` in Claude Code, or `scripts/ship.sh "what you changed"`: it creates a branch, runs the tests, commits, pushes and opens a PR.
3. Never edit files on github.com, and never commit `.env`, API keys or the Belfius/QuantumBlack documents.
4. A new data format goes into `data/schemas.py` with a mock and a test, in the same PR.
5. Commit everything in `logs/` (graded evidence). Full rules in `CLAUDE.md`.
