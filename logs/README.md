# Logs

Everything here is committed, because it is graded evidence of the AI suggestions, the human decisions and the model re-assessment.

| File | Content |
|---|---|
| `orders.csv` | Every proposed order with the human decision, plus orders the human added (Order log format) |
| `signals.csv` | Every agent signal (Signals format), one row per agent × ETF × day |
| `macro/<date>.md` | The Macro/News agent's summary for the day |
| `incidents.csv` | Every pipeline step that failed and was replaced by its fallback |
| `agent_weights.csv` | Agent weights after each review, with the hit rate that drove the change |
| `lessons.md` | Lessons journal that the agents read before each decision |

Not committed: `checkpoints.sqlite` (the saved graph state, used to resume a paused day) and `mock/` (runs while any step is on a mock).
