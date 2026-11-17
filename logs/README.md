# Logs

Everything here is committed, because it is graded evidence of the AI suggestions, the human decisions and the model re-assessment.

| File | Content |
|---|---|
| `orders.csv` | Every proposed order with the human decision, plus orders the human added (Order log format) |
| `reviews.csv` | Every human review of the day (orders, risk, summary) with the feedback sent back (Reviews format) |
| `summary/<date>.md` | The executive summary the human approved before the orders were submitted |
| `memo/<date>.md` | The Portfolio Manager's memo of the day (its last round) |
| `pm_portfolios.csv` | The portfolios the Portfolio Manager built each day: the code's baseline and optimiser, the LLM's own, and the final blend (PM portfolios format) |
| `recommendations.csv` | Every order the Portfolio Manager proposed, round by round, with its fill price and its reasons (Recommendations format, append-only) |
| `risk_reports.csv` | The Risk Manager's report of each day, one row per metric (Risk report format) |
| `risk/<date>.md` | The risk board and the independent reading the human saw at the risk review |
| `risk_forecasts.csv` | The weights each risk assessment was made on and what it forecast, so it can be graded later (Risk forecasts format) |
| `risk_calibration.csv` | The correction the Risk Manager learned for its volatility forecast (Risk calibration format) |
| `signals.csv` | Every agent signal (Signals format), one row per agent × ETF × day |
| `macro/<date>.md` | The Macro/News agent's report for the day: views, counter-arguments, sources, exploration findings, cost |
| `evidence.csv` | Every fact and number the Macro/News agent collected (Evidence format) |
| `macro_views.csv` | One view per macro factor and day, with the sources behind it (Macro views format) |
| `etf_views.csv` | One view per ETF and day from the theme analysts, with their sources (ETF views format) |
| `source_registry.csv` | Every source the Macro/News agent used: status (active, testing, dropped) and hit rate |
| `source_changes.csv` | When each Macro/News source was added or removed, to compare results before and after |
| `incidents.csv` | Every pipeline step that failed and was replaced by its fallback |
| `weights.csv` | Every weight of the system, every session: the technical agent's three answers (since 2022), the four agents, Claude's portfolio against the code's. Each row is the weight an expert had for a session and what following it earned in that session (Weights format): the evidence of model re-assessment |
| `technical_answers.csv` | What the technical rules answered for each ETF on the morning of every session since 2022, and what the ETF then did: what the technical weights are learned from (Technical answers format) |
| `agent_weights.csv` | The agent weights of the first learning rule, until 16 November 2026. No longer written: `weights.csv` replaced it |
| `lessons.md` | Lessons journal, written by the learning step (newest review first): the weights and the record behind them, which agent was right on which kind of call, the largest misses, the weakest sources. The agents read it before each decision |
| `logbook.csv` | The audit trail: one row per event (signals, recommendations, risk flags, human decisions, computed fills, valuations), append-only, each row hashed with the one before it (Logbook format, `agents/logbook.py`) |
| `logbook/<date>.md` | The day told in order, built from `logbook.csv` |
| `customer_log.csv` | What happened to each customer of the app: the profile the coach placed them in, with the reason, their suitability answers and the contradictions the code found, then their decision on the plan (Customer log format, append-only). Written to `mock/` while the plans of the app are examples |
| `logbook/INDEX.md` | One line per day: orders kept, changed, rejected and added by the human, risk warnings, portfolio value and return against the benchmark, and whether the hash chain is intact |

Not committed: `checkpoints.sqlite` (the saved graph state, used to resume a day between reviews) and `mock/` (runs while any step is on a mock).
