"""Tests never call the network or the Claude API.

- The model's key is removed for the whole test session, before any fixture runs (also the module-scoped
  ones), so the agents that call the model by themselves (Portfolio Manager, Risk Manager, technical_llm) run
  on their code alone. A test that needs an answer from the model passes a fake client.
- Every test starts with every pipeline step on its mock. A test that needs a real module switches it on
  with monkeypatch.setitem(settings.USE_MOCK, step, False).
- Tests do not depend on the strategy settings the team tunes: idle money stays in cash unless a test says
  otherwise, so changing IDLE_TO_BENCHMARK in agents/portfolio_config.py never breaks a test about something else.
"""
import os

import pytest

from agents import portfolio_config
from data import config as settings

for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
    os.environ.pop(name, None)
    setattr(settings, name, "")


@pytest.fixture(autouse=True)
def all_mocks(monkeypatch):
    monkeypatch.setattr(settings, "USE_MOCK", {step: True for step in settings.USE_MOCK})
    monkeypatch.setattr(portfolio_config, "IDLE_TO_BENCHMARK", 0.0)
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        monkeypatch.setattr(settings, name, "")
        monkeypatch.delenv(name, raising=False)
