# QDC Trading — ReBel Agentic AI Challenge

A multi-agent system that trades 50 ETFs for the QTEM Data Challenge (Belfius × QuantumBlack, fall 2026).
The agents propose, a human approves every order.

## The competition

- €10,000, no transaction costs, at most 5 orders a day, on Mon 7, Thu 10, Mon 14, Thu 17 and Mon 21 December 2026.
- Graded on performance **and** method: critical thinking towards the agents, and re-assessment of the model.
- Mid-term report 22 Nov 2026 · final report 16 Jan 2027.
- Our reading of two open rules, each one setting in `data/config.py`: an order fills at the **last close before its
  day** (`FILL_AT_PREVIOUS_CLOSE`), and in **fractional units** (`ORDER_UNIT_DECIMALS`).

## How it works

![Workflow of one trading day](docs/workflow.png)

| Step | What it does | Code |
|---|---|---|
| Data | Prices of the 50 ETFs, 37 indicators, ex-dividend dates | `data/` |
| Learning | Re-weights everything from what each expert earned (below), writes the lessons | `graph/review_loop.py`, `agents/learning.py` |
| Technical | Two scores: three rule answers weighted by what they earned since 2022, and the LLM's reading of them | `agents/technical*.py` |
| Macro / News | Two scores: 15 macro views, and 20 theme analysts reading the news | `agents/macro_*.py` |
| Portfolio Manager | Combines the four scores, blends the code's portfolio with the LLM's, picks at most 5 orders | `agents/portfolio*.py` |
| You | Review the orders | cockpit, Today page |
| Risk Manager | Measures the portfolio after your orders on 22 metrics; the LLM writes an independent reading | `agents/risk*.py` |
| You | Review the risk, then sign off the summary; each review can send the day back or stop it | cockpit, Today page |
| Logbook | Writes the orders and an audit trail of the day, hash-chained | `agents/logbook.py` |

**One rule for every weight.** After each session, a weight is multiplied by e^(2 × what following that expert earned),
then the weights are rescaled to sum to 1 and kept between 5% and 80%. It weighs the technical agent's three answers
(since 2022), the four agents (since their first score), and the LLM's portfolio against the code's.

**Free tiers only.** The challenge allows no paid API. The LLM is Gemini on Google AI Studio's free tier, behind one
file (`agents/llm.py`): `gemini-3.5-flash-lite` for the many calls, `gemini-4-hydrogen` for the few judgement calls
when it answers (20 requests a day), the small model otherwise. The model's web searches run through Linkup
(`agents/web_search.py`), paid from its free credit, and a source the model cites must be a page a search returned:
anything else is refused. The limits and the pace are settings in `data/config.py`.

**Two principles.** Code calculates, the LLM judges: no LLM produces a number code could compute. Nothing is traded
without the three human reviews, and the trading rules are checked in code on every proposal and every approval.

## Get started

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.11+
cp .env.example .env                                                  # GEMINI_API_KEY, LINKUP_API_KEY, EODHD_API_KEY; never commit it
.venv/bin/python -m pytest                                            # no test calls the network or the model
streamlit run app/streamlit_app.py                                    # the cockpit
```

**Every weekday morning:** open the cockpit and click **Run the day** (10 to 15 minutes at the free tier's pace, and about $0.60 of the search credit). The
scores and portfolios are logged before your first review, which is all the learning needs. Doing the three reviews
on practice days adds evidence of human judgement. Then commit `logs/` with `scripts/ship.sh "Add the days of …"`.

Without the cockpit: `python -m graph.pipeline` (runs to the first review), `--approve-all`, or `--answer file.json`.
To try something without touching the graded logs, point everything at a copy: `QDC_LOGS_DIR=/tmp/trial`.

## The cockpit

Dashboard · **Today** (your three reviews) · one page per agent · **Learning** (the weights and the backtest) ·
**How it works** (the diagram: click a box to see what it does and its code) · Logbook. A finished day is read from
its files in `logs/`, so any past day can be opened.

## The customer app

`python -m mobile.server`, then http://localhost:8600: a phone-sized app for the customer (light, ReBel style).

- **Bel**, the coach, places the customer in one of five investor profiles (`data/profiles.py`) the way Akinator
  guesses a character: the code keeps a probability for each profile, asks next the question that separates them
  best, and updates the probabilities with every answer. With the customer's consent, what Belfius already knows
  sets the start and sharpens the questions (`agents/profiler.py`).
- **The same answers** fill the suitability sheet (one of them a wealth range). The code flags answers that
  contradict each other: an adviser then looks at the plan first. The last question asks for the customer's own
  rules (no weapons, no crypto...), and the plan leaves those ETFs out. The LLM only reads typed answers.
- **One approval.** The customer sees the plan of their profile and approves it before anything is invested. The
  plans are still examples (`data/mock.py`), so the app writes to `logs/mock/`: one portfolio per profile, built by
  the engine, is the next step.

## Change the model

Every setting is one line. Test a change on the past first: add a line to `VARIANTS` in
`backtest/system_backtest.py` and run `python -m backtest.system_backtest --daily` (results and how to read them in
`backtest/README.md`).

| To change | Where |
|---|---|
| Competition rules, fill price, units, the learning speed and bounds, the agents' starting weights | `data/config.py` |
| The technical rules and the score of each answer | `agents/strategy_config.py` |
| How the portfolio is built, its hard limits, the LLM's starting share | `agents/portfolio_config.py` |
| The risk reference levels | `agents/risk_config.py` |
| The five investor profiles, the coach's questions and what each answer says about a profile | `data/profiles.py`, `agents/profiler.py` |
| The macro factors and each ETF's exposure to them | `data/config.py`, `data/macro_exposure.csv` |
| The models, the pace of the calls and the search budget | `data/config.py` |
| What the LLM is asked | the prompts in `agents/macro_analysts.py`, `agents/macro_llm.py`, `agents/technical_llm.py`, `agents/portfolio_llm.py`, `agents/profiler.py` |

## Repository

```
data/       prices, indicators, the 50 ETFs, the 5 investor profiles, settings, every shared table (schemas.py) and its mock (mock.py)
agents/     technical, macro/news, portfolio manager, risk manager, logbook, the learning rule, and the customer's coach
graph/      pipeline.py (one day, step by step) and review_loop.py (the learning step)
app/        the Streamlit cockpit
mobile/     the customer app: server.py and the phone screens (static/)
backtest/   the system replayed since 2022 (results in backtest/results/)
logs/       everything the agents said and the humans decided: graded evidence (logs/README.md)
tests/      one file per module
```

## Team

| | Scope |
|---|---|
| Axel | Macro / News, pipeline and learning, cockpit, customer app, backtest, repo |
| Achille | Data agent, technical agent |
| Alexander | Portfolio Manager, Risk Manager, Logbook |

Never push to `main`: ship with `scripts/ship.sh "what you changed"` (or `/ship` in Claude Code), and Axel merges.
Never commit `.env`, an API key or the Belfius / QuantumBlack documents. Full rules in `CLAUDE.md`.
