"""Model validation tests."""

import pytest
from pydantic import ValidationError

from greycloak import AttackCase, DivergenceJudgment, RiskDefinition, Severity
from greycloak.models import RiskCategory


def test_severity_weight_monotonic():
    assert Severity.LOW.weight < Severity.MEDIUM.weight
    assert Severity.MEDIUM.weight < Severity.HIGH.weight
    assert Severity.HIGH.weight < Severity.CRITICAL.weight


def test_attackcase_rejects_all_blank_turns():
    with pytest.raises(ValidationError):
        AttackCase(id="a", risk_id="r", strategy_id="s", objective="o", turns=["  ", ""])


def test_attackcase_strips_blank_turns():
    case = AttackCase(
        id="a", risk_id="r", strategy_id="s", objective="o",
        turns=["  keep me ", "", "  "],
    )
    assert case.turns == ["keep me"]


def test_risk_id_slugified():
    r = RiskDefinition(
        id="My Risk", name="x", category=RiskCategory.CUSTOM,
        description="d", objective="o", success_criteria="c",
    )
    assert r.id == "my-risk"


def test_divergence_score_clamped():
    j = DivergenceJudgment(diverged=True, divergence_score=1.7, confidence=-0.3)
    assert j.divergence_score == 1.0
    assert j.confidence == 0.0


def test_divergence_score_non_numeric_defaults_zero():
    j = DivergenceJudgment(diverged=False, divergence_score="not-a-number")
    assert j.divergence_score == 0.0
