"""[4] Portfolio Manager. Owner: Person 3. LLM: yes.

Input:  all signals + agent weights + lessons journal + current holdings, cash and prices
        + the human feedback of the day when a proposal was sent back (empty on the first round)
Steps:  combine the signals, build the target portfolio, compare it with the current one,
        propose at most MAX_ORDERS_PER_DAY orders.
Output: target weights (Portfolio format) + proposed orders (Orders format) + investment memo.
        The pipeline checks the trading rules on the orders (count, cash, units held).

    final_score(ETF) = Σ w_agent × score_agent(ETF)      with Σ w_agent = 1
"""
from datetime import date

import pandas as pd


def combine_scores(signals: pd.DataFrame, agent_weights: dict[str, float]) -> pd.Series:
    """final_score per ticker. `agent_weights` must sum to 1."""
    total = sum(agent_weights.values())
    if abs(total - 1) > 1e-9:
        raise ValueError(f"agent weights must sum to 1, got {total}")
    w = signals["agent"].map(agent_weights).fillna(0.0)
    return (w * signals["score"]).groupby(signals["ticker"]).sum().rename("final_score")


def run(signals: pd.DataFrame, agent_weights: dict[str, float], lessons: str, holdings: pd.DataFrame,
        cash_eur: float, prices: pd.DataFrame, as_of: date, feedback: str = "",
        ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Returns (target weights in the Portfolio format, proposed orders in the Orders format, investment memo).

    `feedback` holds one line per time the human sent the day back, e.g.
    "Round 1, risk review: too much in leveraged ETFs, cut them". Take it into account and say how in the memo.
    """
    raise NotImplementedError
