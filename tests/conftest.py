"""Every test starts with every pipeline step on its mock and without a Claude key: tests never call the
network or the Claude API. A test that needs a real module switches it on with
monkeypatch.setitem(settings.USE_MOCK, step, False); the Portfolio Manager and the Risk Manager then run on
their code alone (no key: no LLM call)."""
import pytest

from data import config as settings


@pytest.fixture(autouse=True)
def all_mocks(monkeypatch):
    monkeypatch.setattr(settings, "USE_MOCK", {step: True for step in settings.USE_MOCK})
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
