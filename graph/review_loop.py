"""Review loop, run after each trading day. Owner: Person 3.

1. Compare each agent's prediction with the realised return.
2. Update the agent weights used by the Portfolio Manager, based on hit rate.
3. Write lessons to logs/lessons.md. Agents read it before the next decision.
4. Log every update in logs/agent_weights.csv. This is our evidence of model re-assessment.
"""
from datetime import date


def review(as_of: date) -> dict[str, float]:
    """Returns the new agent weights (sum to 1)."""
    raise NotImplementedError
