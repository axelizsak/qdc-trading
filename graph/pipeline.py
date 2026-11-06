"""LangGraph orchestration. Owner: Axel.

Data -> (Technical | Macro/News) -> Portfolio Manager -> Risk Manager
-> human approval (app/) -> order log (logs/orders.csv) -> Belfius form.

Fallback if LangGraph gets in the way: the same steps as a plain Python pipeline.
"""
from datetime import date


def run_trading_day(as_of: date) -> None:
    raise NotImplementedError
