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

> **To confirm with Belfius:** one slide says orders fill at the *previous day's* closing price, another says the *closing price of the trading day itself*.

## How it works

Each agent gives every ETF a score in [-1, 1]. The Portfolio Manager combines them with one weight per agent:

```
final_score(ETF) = Σ w_agent × score_agent(ETF)      with Σ w_agent = 1
```

After each trading day, the review loop compares each agent's prediction with the realised return.
Agents that were right gain weight. Lessons go into a journal that the agents read before the next decision.
Every update is logged in `logs/`, and that log is our evidence of model re-assessment.

```
            yfinance / web search
                     |
               [1] DATA AGENT
                     |
           +---------+---------+
           |                   |
     [2] TECHNICAL       [3] MACRO/NEWS
  indicators + ml_library   Claude web search
     + LLM layer
           |                   |
           +---------+---------+
                     |   signals
          [4] PORTFOLIO MANAGER
                     |   target weights
             [5] RISK MANAGER
                     |   orders
         HUMAN APPROVAL (Streamlit)
                     |
          Order log -> Belfius form
                     |
     REVIEW LOOP (after each trading day)
                     |
   updates agent weights + lessons journal
```

**Rule:** use code for calculations, and the LLM only where judgement is needed.

| # | Agent | Module | Input | Output | LLM? |
|---|---|---|---|---|---|
| 1 | Data | `data/data_agent.py` | ISIN list | ETF mapping (ISIN → Yahoo ticker), prices, returns, volatility, drawdowns, correlations | No |
| 2 | Technical | `agents/technical.py` | Prices + features | Two scores per ETF: deterministic (indicators + `ml_library` models) and LLM-interpreted | Interpretation layer |
| 3 | Macro / News | `agents/macro_news.py` | Web search (Claude API) | Macro summary + signal per theme (rates, geopolitics, commodities) | Yes |
| 4 | Portfolio Manager | `agents/portfolio.py` | All signals + agent weights + lessons journal | Target weights + investment memo | Yes |
| 5 | Risk Manager | `agents/risk.py` | Target weights + portfolio | Valid orders (concentration, cash, 5-order limit) | No |

There is no separate Quant/ML agent. The ML models are in a shared library (`ml_library/`) that the agents call, and every model is scored by the same walk-forward backtest (`backtest/`).

## Repository layout

```
qdc-trading/
  common/      Competition rules (config.py) and shared data formats (schemas.py)
  data/        Data agent, ETF universe, Yahoo mapping, mock data
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

These were fixed on day 1 and are defined in `common/schemas.py`. Build your module against mock data (`python -m data.mock`) so that nobody has to wait for anyone else.

| Table | Columns |
|---|---|
| Prices | `date, ticker, close_eur` |
| Signals | `ticker, date, agent, score [-1,1], confidence [0,1], rationale` |
| Portfolio | `ticker, target_weight` |
| Orders | `date, side, ticker, units` (matches the Belfius form) |

`ticker` is always the Belfius short name from `data/etf_universe.csv` (e.g. `WORLD`, `NASDAQ 3X`), never the Yahoo symbol.

`agent` is one of `technical_det`, `technical_llm` or `macro_news` (see `SIGNAL_AGENTS` in `common/config.py`). The Technical agent writes two rows per ETF, so the Portfolio Manager gives the deterministic score and the LLM score separate weights, and the review loop can tell whether the LLM layer adds value.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your ANTHROPIC_API_KEY. Never commit .env
pytest
```

Check which ETFs are available on Yahoo Finance:

```bash
python data/check_etfs_yfinance.py   # writes data/etf_yahoo_map.csv
```

## Work split

| Person | Scope |
|---|---|
| Person 1 (Axel) | Macro / News agent · repo management · LangGraph integration · Streamlit interface |
| Person 2 | Data agent · ETF mapping · Technical agent (indicators + LLM layer) · shared ML library · common backtest |
| Person 3 | Portfolio Manager · Risk Manager · review loop |

The split between Person 2 and Person 3 is still to be confirmed.

## Working together

- Use one branch per feature and open a pull request into `main`.
- Never commit `.env` or API keys.
- Commit everything in `logs/`, because it is graded evidence.
