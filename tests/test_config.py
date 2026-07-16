import dspy

from greycloak.config import build_lm
from greycloak.models import LMConfig


def _capture_lm(monkeypatch):
    captured = {}

    class FakeLM:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(dspy, "LM", FakeLM)
    return captured


def test_build_lm_forwards_extra(monkeypatch):
    captured = _capture_lm(monkeypatch)
    build_lm(LMConfig(model="openai/gpt-5.4-nano", temperature=1.0, max_tokens=16000,
                      extra={"reasoning_effort": "low"}))
    assert captured["model"] == "openai/gpt-5.4-nano"
    assert captured["temperature"] == 1.0
    assert captured["max_tokens"] == 16000
    assert captured["reasoning_effort"] == "low"


def test_build_lm_no_extra_by_default(monkeypatch):
    captured = _capture_lm(monkeypatch)
    build_lm(LMConfig(model="openai/gpt-4o-mini"))
    assert "reasoning_effort" not in captured
