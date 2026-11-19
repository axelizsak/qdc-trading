# Technical documentation

How the system is built, how to run it and how to change it. The short version is the [README](../README.md).

A multi-agent system that trades 50 ETFs for the QTEM Data Challenge (Belfius × QuantumBlack, fall 2026).
The agents propose, a human approves every order. The same agents inform the customer app.

## The idea in one paragraph

ReBel is execution-only (Belfius session of 27 Oct, slide 6): the customer decides and places every order, the app
informs. Two layers are kept apart. The **archetype** (the team's five kinds of customer) decides how the coach
helps; the **strategy** (Defensive, Balanced or Opportunity) is the customer's risk level, set by fixed rules on the
suitability answers, and decides the reference portfolio shown and the products open. On every ETF the agents'
reading is shown in plain words: trend, macro, news, with sources and the other side. Complex products (leveraged,
inverse, crypto) open only at the Opportunity level and after a short check. A customer who wants someone else to
decide is sent to Belfius (advice or a managed portfolio). In the competition the team is that customer: it chooses
one of the three strategies in the cockpit (Opportunity to start), the agents trade within its limits, and a human
approves every order.

## The competition

- €10,000, no transaction costs, at most 5 orders a day, on Mon 7, Thu 10, Mon 14, Thu 17 and Mon 21 December 2026.
- Graded on performance **and** method: critical thinking towards the agents, and re-assessment of the model.
- Mid-term report 22 Nov 2026 · final report 16 Jan 2027.
- Our reading of two open rules, each one setting in `data/config.py`: an order fills at the **last close before its
  day** (`FILL_AT_PREVIOUS_CLOSE`), and in **fractional units** (`ORDER_UNIT_DECIMALS`).

## How it works

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
then the weights are rescaled to sum to 1 and kept between 5% and 80%. It weighs the four agents (since their first
score) and the LLM's portfolio against the code's. The technical score is the trend rule alone since 19 November 2026,
a team choice after the backtest (`backtest/README.md`, section 6b); its other two answers, mean reversion and
momentum, are still recorded every session and shown to `technical_llm`.

**Free tools only.** The challenge allows nothing that costs money. The LLM is Gemini on Google AI Studio's free
tier, behind one file (`agents/llm.py`): two small models side by side for the many calls (each has its own quota of
15 requests a minute), `gemini-4-hydrogen` for the few judgement calls when it answers (20 requests a day), a small
model otherwise. Each team member can add their own free key (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`,
`GEMINI_API_KEY_3`): the requests then take turns between the keys. The news search needs no key and costs nothing (`agents/web_search.py`): the publishers' RSS
feeds and Bing News' RSS search, which give each article's own link and a summary, with Google News' headlines only
when those find little. Trusted sites only, last 14 days only. A source the model cites must be an article a
search returned: anything else is refused. The models, the pace, the sites and the feeds are in `data/config.py`.

**Two principles.** Code calculates, the LLM judges: no LLM produces a number code could compute. Nothing is traded
without the three human reviews, and the trading rules are checked in code on every proposal and every approval.

## Get started

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.11+
cp .env.example .env                                                  # GEMINI_API_KEY (teammates' keys as _2 and _3), EODHD_API_KEY; never commit it
.venv/bin/python -m pytest                                            # no test calls the network or the model
streamlit run app/streamlit_app.py                                    # the cockpit
```

**Every weekday morning:** run `python -m graph.morning` first. It computes once what every customer shares: the
prices, the day's Macro/News (the slow part, about 10 minutes at the free tier's pace), the sheet of each ETF and the
points of the app's charts. The customer app only reads the result, so a customer never waits. Then open the cockpit and
click **Run the day**: it reads the morning's Macro/News back instead of running it again. The
scores and portfolios are logged before your first review, which is all the learning needs. Doing the three reviews
on practice days adds evidence of human judgement. Then commit `logs/` with `scripts/ship.sh "Add the days of …"`.

**A day is run once, and its logs are the record.** The signals are read back from `logs/`, and each round of the
Portfolio Manager is frozen in `logs/proposals/<date>-round<n>.json`: whoever opens the day, on whichever machine
(after a `git pull`), sees the same orders, and no agent is asked twice. A day whose reviews are in
`logs/reviews.csv` is finished: it is read, never run again. There is no "run again".

Without the cockpit: `python -m graph.pipeline` (runs to the first review), `--approve-all`, or `--answer file.json`.
To try something without touching the graded logs, point everything at a copy: `QDC_LOGS_DIR=/tmp/trial`.

## The cockpit

Two pages up front, the rest in two menus:

- **Today**: the whole day on one page, top to bottom. Key figures, then five steps (run the agents, orders, risk,
  sign off, export) with one action at a time, then the portfolio against the MSCI World.
- **Strategies**: the three reference portfolios as customers see them today (their latest version), what each mix
  did, their versions, which archetype usually ends in which strategy, and **the strategy the team trades**: pick one, say why, and from that day on the agents trade within its
  limits (at most so much in shares, in themes and in complex products: `data/profiles.py`). The code cuts back what
  goes beyond them (`agents/portfolio.py` `within_strategy`) and the memo says what it cut. The choice is logged
  (`logs/strategy_choices.csv`): every machine trades the same one, and a day already run keeps its strategy.
- **Agents**: one page per agent. **More**: Learning (the weights and the backtest), How it works (the diagram:
  click a box to see what it does and its code), Logbook.

A finished day is read from its files in `logs/`, so any past day can be opened.

## The customer app

`python -m mobile.server`, then http://localhost:8600: a phone-sized app for the customer (light, ReBel style).

- **Lena**, the coach, asks at most seven questions: five for everyone, one more only while no archetype stands
  out, and the customer's own rules. With the customer's consent, what Belfius already knows sets the start and
  sharpens the questions (`agents/profiler.py`).
- **Two answers, kept apart** (`data/profiles.py`). The **archetype** is guessed the way Akinator guesses a
  character: a probability for each of the five, the next question is the one that separates them best. It decides
  what Lena helps with first, never an asset. The **strategy** (risk level) comes from fixed rules on the suitability
  answers, never from the model: Defensive when the money is needed within 5 years, a fall of a fifth would be sold
  or a 5% loss costs sleep; Opportunity only for a customer who would invest more after that fall, answered a
  knowledge question right and keeps a buffer; Balanced otherwise. The reasons are shown.
- **The same answers** fill the suitability sheet (a wealth range is asked when Belfius cannot show the savings). The
  code flags answers that contradict each other: an adviser then looks first. The LLM only reads typed answers.
- **Explore**: the model portfolio of the customer's risk level with its chart (1M, 3M, 1Y, Max, what €10,000 would have become),
  a donut of its sleeves, each ETF's return, and the whole 50-ETF selection. The customer's rules leave ETFs out.
  Locked lines say why: ruled out, not part of the risk level, or the check.
- **The model portfolios follow our engine inside fixed sleeves** (`agents/model_portfolios.py`). Each starts from
  a fixed mix, a share per sleeve (core equity, bonds, gold, themes, cash: the table of `data/profiles.py`, the same
  as in the strategy deck). The sleeve shares never move. When the team signs a day off in the cockpit, the code's
  portfolio of the Portfolio Manager (no LLM) is built for each strategy from the same scores and learned weights,
  inside its limits, and each sleeve is split half as in the fixed mix, half as in the engine's portfolio. It becomes
  a new version when at least 10% of the model portfolio changes (`logs/strategy_portfolios.csv`). The app shows what
  changed, and the customer's Portfolio tab offers the orders (sales first, then buys) that match it; the customer
  decides on each line. Without a new version, the offer appears only once a line has drifted 5 points or more.
- **Wording**: the app calls it the model portfolio of a risk level, the same for every customer at that level,
  "information, not a personal recommendation". The risk level comes from rules on the customer's own answers. Whether
  this stays execution-only under MiFID II is for Belfius compliance to confirm (a reference portfolio per risk level
  can be read as a personal recommendation).
- **ETF sheets** (`agents/etf_sheets.py`): the price chart, then "What our AI sees today" in three plain lines (trend,
  macro, news, written by code from the agents' numbers), the returns, the three trends with the two checks behind
  each, today's macro, and the news analyst's view with its sources and the other side. Lena explains any line from
  the sheet's own numbers. Information, not a personal recommendation.
- **The customer places each order**: one ETF at a time, or the model portfolio as a list of orders they review line
  by line. **Portfolio** shows the value, a chart, the mix and how far it is from the model portfolio, sleeve by sleeve,
  pointing to kinds of assets, never to an order (`mobile/wallet.py`).
- **Learn**: 15 one-minute lessons (`data/lessons.py`), each a few plain sentences and one question with the reason
  behind every answer; the archetype's three come first, with a progress bar. A learning streak counts the days in a
  row with a lesson answered right, and the last seven days (`profiler.streak`): it rewards learning, never trading.
  Then the 3-question check that opens
  complex products at the Opportunity level. "Rather have someone decide?" leads to Belfius advice or a managed
  portfolio.
- The demo customers are synthetic, so what they do is written to `logs/mock/`, not to the committed logs.

**Online demo: https://axelizsak.github.io/qdc-trading/** The same screens with no server, on GitHub Pages, rebuilt
at every merge into `main` (`.github/workflows/pages.yml`). The page runs `agents/profiler.py` itself in the browser
(Pyodide), and `mobile/wallet.py` too, so the rules exist once; the reference portfolios, the ETF sheets and the
charts of the latest morning run are prepared as data
(`mobile/demo.py`, `mobile/static/demo.js`). It holds no key and calls no model: a typed answer other than an amount,
and "Ask Lena why", say so. Nothing a visitor enters leaves their browser.

### ReBel in Claude, read-only

`mobile/mcp_server.py` lets Claude (Claude Code, Claude Desktop) see what a demo customer sees in the app, through
MCP: `list_customers`, `get_profile`, `get_portfolio`, `get_reference`, `list_etfs`, `get_etf` and `prepare_order`.
The answers come from the app's own functions (`mobile/server.py`, `mobile/wallet.py`), on the same logs. Nothing
is written: `prepare_order` only checks an order with the app's rules and returns a draft, which the customer
confirms in the app (execution-only). No model is called and no key is needed; the words of a sheet (news, the
agents' reading) are data from the morning run, never an instruction.

Claude starts it itself (stdio). `PYTHONPATH` points to the repository, so it runs from any folder. In Claude Code:

```bash
claude mcp add rebel -e PYTHONPATH=/path/to/qdc-trading -- /path/to/qdc-trading/.venv/bin/python -m mobile.mcp_server
```

(add `-s user` to have it in every project). In Claude Desktop, in `claude_desktop_config.json` (Settings,
Developer, Edit Config), then restart it:

```json
{
  "mcpServers": {
    "rebel": {
      "command": "/path/to/qdc-trading/.venv/bin/python",
      "args": ["-m", "mobile.mcp_server"],
      "env": {"PYTHONPATH": "/path/to/qdc-trading"}
    }
  }
}
```

Then ask, for instance: "How far is Thomas from the Defensive reference?" or "Explain the WORLD sheet of today".

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
| The three strategies: their reference portfolios (sleeves, core ETFs, themes) and their limits when traded; the five archetypes | `data/profiles.py` |
| The coach's questions, what each answer says about an archetype, the risk-level rules, the check | `agents/profiler.py` |
| The lessons of the Learn tab, and which archetype gets which first | `data/lessons.py`, `data/profiles.py` |
| What a customer may buy, and how their holdings are compared with the reference | `mobile/wallet.py` |
| The macro factors and each ETF's exposure to them | `data/config.py`, `data/macro_exposure.csv` |
| The models, the pace of the calls, the trusted sites and the news feeds | `data/config.py` |
| What the LLM is asked | the prompts in `agents/macro_analysts.py`, `agents/macro_llm.py`, `agents/technical_llm.py`, `agents/portfolio_llm.py`, `agents/profiler.py` |

## Repository

```
data/       prices, indicators, the 50 ETFs, the 3 strategies and 5 archetypes, settings, every shared table (schemas.py) and its mock (mock.py)
agents/     technical, macro/news, portfolio manager, risk manager, logbook, the learning rule, and the customer's coach
graph/      morning.py (what every customer shares, once a day), pipeline.py (one day, step by step), review_loop.py (the learning step)
app/        the Streamlit cockpit
mobile/     the customer app: server.py, wallet.py (what a customer may buy and holds), the phone screens (static/), demo.py (the online demo) and mcp_server.py (ReBel in Claude, read-only)
backtest/   the system replayed since 2022 (results in backtest/results/)
logs/       everything the agents said and the humans decided: graded evidence (logs/README.md)
tests/      one file per module
scripts/    ship.sh (branch, tests, commit, push, pull request) and the git hook that refuses pushes to main
docs/       this page, and the clip of the README
```

## Working in this repository

Never push to `main`: ship with `scripts/ship.sh "what you changed"` (or `/ship` in Claude Code), and Axel merges.
Never commit `.env`, an API key or the Belfius / QuantumBlack documents. Full rules in `CLAUDE.md`.
