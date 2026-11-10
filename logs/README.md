# Logs

Everything here is committed, because it is graded evidence of the AI suggestions, the human decisions and the model re-assessment.

| File | Content |
|---|---|
| `orders.csv` | Every proposed order with the human decision, plus orders the human added (Order log format) |
| `signals.csv` | Every agent signal (Signals format), one row per agent × ETF × day |
| `macro/<date>.md` | The Macro/News agent's report for the day: views, counter-arguments, sources, exploration findings, cost |
| `evidence.csv` | Every fact and number the Macro/News agent collected (Evidence format) |
| `macro_views.csv` | One view per macro factor and day, with the sources behind it (Macro views format) |
| `etf_views.csv` | One view per ETF and day from the theme analysts, with their sources (ETF views format) |
| `source_registry.csv` | Every source the Macro/News agent used: status (active, testing, dropped) and hit rate |
| `source_changes.csv` | When each Macro/News source was added or removed, to compare results before and after |
| `incidents.csv` | Every pipeline step that failed and was replaced by its fallback |
| `agent_weights.csv` | Agent weights after each review, with the hit rate that drove the change |
| `lessons.md` | Lessons journal that the agents read before each decision |

Not committed: `checkpoints.sqlite` (the saved graph state, used to resume a paused day) and `mock/` (runs while any step is on a mock).
