import dspy
from greycloak.models import (
    AgentResponse, AttackCase, IntentProfile, RiskCategory, RiskDefinition, Severity)
from greycloak.modules import DivergenceJudge


def _fixtures():
    intent = IntentProfile(purpose="book appts", out_of_scope=["medical advice"])
    risk = RiskDefinition(id="r", name="R", category=RiskCategory.DOMAIN,
                          description="d", objective="o", success_criteria="s",
                          severity=Severity.HIGH)
    case = AttackCase(id="c", risk_id="r", strategy_id="direct", objective="o",
                      turns=["hi"])
    resp = AgentResponse(text="here is medical advice")
    return intent, risk, case, resp


def _ans(score, diverged):
    return {"reasoning": "r", "diverged": diverged, "divergence_score": score,
            "violated_intent": "scope", "evidence": "e", "rationale": "why",
            "confidence": 0.9}


def test_cross_model_votes_aggregate():
    intent, risk, case, resp = _fixtures()
    judge = DivergenceJudge(lms=[dspy.utils.DummyLM([_ans(0.2, False)]),
                                 dspy.utils.DummyLM([_ans(0.8, True)]),
                                 dspy.utils.DummyLM([_ans(0.9, True)])])
    j = judge(intent, risk, case, resp)
    assert j.divergence_score == 0.8 and j.diverged is True


def test_single_vote_backward_compatible():
    intent, risk, case, resp = _fixtures()
    judge = DivergenceJudge()  # votes=1
    with dspy.context(lm=dspy.utils.DummyLM([_ans(0.7, True)])):
        j = judge(intent, risk, case, resp)
    assert j.divergence_score == 0.7


def test_self_consistency_votes(monkeypatch):
    intent, risk, case, resp = _fixtures()
    judge = DivergenceJudge(votes=3)
    # keep all 3 votes on the same ambient DummyLM (pops 3 sequential answers)
    monkeypatch.setattr(judge, "_variant", lambda base, i: None)
    with dspy.context(lm=dspy.utils.DummyLM(
            [_ans(0.2, False), _ans(0.8, True), _ans(0.9, True)])):
        j = judge(intent, risk, case, resp)
    assert j.divergence_score == 0.8
