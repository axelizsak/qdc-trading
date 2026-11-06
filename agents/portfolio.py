"""[4] Portfolio Manager. Owner: Person 3. LLM: yes.

Input:  all signals + agent weights + lessons journal
Output: target weights (Portfolio format) + investment memo.

    final_score(ETF) = Σ w_agent × score_agent(ETF)      with Σ w_agent = 1
"""
import pandas as pd


def combine_scores(signals: pd.DataFrame, agent_weights: dict[str, float]) -> pd.Series:
    """final_score per ticker. `agent_weights` must sum to 1."""
    total = sum(agent_weights.values())
    if abs(total - 1) > 1e-9:
        raise ValueError(f"agent weights must sum to 1, got {total}")
    w = signals["agent"].map(agent_weights).fillna(0.0)
    return (w * signals["score"]).groupby(signals["ticker"]).sum().rename("final_score")


def run(signals: pd.DataFrame, agent_weights: dict[str, float], lessons: str = "") -> tuple[pd.DataFrame, str]:
    """Returns (target weights in the Portfolio format, investment memo)."""
    raise NotImplementedError
