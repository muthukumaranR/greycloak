"""Report aggregation + markdown rendering."""

from greycloak import (
    AttackResult,
    category_breakdown,
    summarize,
    to_markdown,
)
from greycloak.models import (
    AgentResponse,
    AttackCase,
    DivergenceJudgment,
    IntentProfile,
    RiskCategory,
    RiskDefinition,
    Severity,
)
from greycloak.strategies import STRATEGY_INDEX


def _result(risk_id, cat, sev, diverged, score):
    risk = RiskDefinition(
        id=risk_id, name=risk_id.title(), category=cat, description="d",
        objective="o", success_criteria="c", severity=sev,
    )
    case = AttackCase(id="c-" + risk_id, risk_id=risk_id, strategy_id="direct",
                      objective="o", turns=["probe"])
    return AttackResult(
        case=case, risk=risk, strategy=STRATEGY_INDEX["direct"],
        response=AgentResponse(text="reply"),
        judgment=DivergenceJudgment(diverged=diverged, divergence_score=score,
                                    severity=sev, rationale="r"),
    )


def _intent():
    return IntentProfile(purpose="do one job")


def test_summarize_computes_asr_and_ranks():
    results = [
        _result("jailbreak", RiskCategory.GENERIC, Severity.HIGH, True, 0.9),
        _result("jailbreak", RiskCategory.GENERIC, Severity.HIGH, False, 0.1),
        _result("overgeneralization", RiskCategory.DOMAIN, Severity.HIGH, True, 0.8),
    ]
    report = summarize("bot", _intent(), results)
    assert report.total_attacks == 3
    assert report.total_successes == 2
    assert abs(report.attack_success_rate - 2 / 3) < 1e-9
    # overgeneralization has ASR 1.0 -> ranked above jailbreak (ASR 0.5)
    assert report.risk_scores[0].risk_id == "overgeneralization"


def test_category_breakdown():
    results = [
        _result("jailbreak", RiskCategory.GENERIC, Severity.HIGH, True, 1.0),
        _result("scope-boundary", RiskCategory.DOMAIN, Severity.MEDIUM, False, 0.0),
    ]
    report = summarize("bot", _intent(), results)
    cats = category_breakdown(report)
    assert cats[RiskCategory.GENERIC]["attack_success_rate"] == 1.0
    assert cats[RiskCategory.DOMAIN]["attack_success_rate"] == 0.0


def test_to_markdown_contains_headline_numbers():
    results = [_result("pii-leak", RiskCategory.GENERIC, Severity.HIGH, True, 0.9)]
    md = to_markdown(summarize("bot", _intent(), results))
    assert "# Red-team report — bot" in md
    assert "Attack success rate" in md
    assert "Top divergences" in md


def test_empty_report_is_safe():
    report = summarize("bot", _intent(), [])
    assert report.attack_success_rate == 0.0
    assert to_markdown(report).startswith("# Red-team report")
