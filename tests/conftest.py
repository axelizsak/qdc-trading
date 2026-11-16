"""Tests never call the network or the Claude API.

- The Claude key is removed for the whole test session, before any fixture runs (also the module-scoped
  ones), so the agents that call Claude by themselves (Portfolio Manager, Risk Manager, technical_llm) run
  on their code alone. A test that needs an answer from Claude passes a fake client.
- Every test starts with every pipeline step on its mock. A test that needs a real module switches it on
  with monkeypatch.setitem(settings.USE_MOCK, step, False).
"""
import os

import pytest

from data import config as settings

os.environ.pop("ANTHROPIC_API_KEY", None)
settings.ANTHROPIC_API_KEY = ""


@pytest.fixture(autouse=True)
def all_mocks(monkeypatch):
    monkeypatch.setattr(settings, "USE_MOCK", {step: True for step in settings.USE_MOCK})
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
