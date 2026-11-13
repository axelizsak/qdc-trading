"""Every test starts with every pipeline step on its mock: tests never call the network or the Claude API.
A test that needs a real module switches it on with monkeypatch.setitem(settings.USE_MOCK, step, False)."""
import pytest

from data import config as settings


@pytest.fixture(autouse=True)
def all_mocks(monkeypatch):
    monkeypatch.setattr(settings, "USE_MOCK", {step: True for step in settings.USE_MOCK})
