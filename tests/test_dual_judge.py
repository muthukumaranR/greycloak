import dspy
from greycloak.engine import RedTeamEngine
from greycloak.models import (
    AgentResponse, AttackCase, AttackStrategy, IntentProfile, RiskCategory,
    RiskDefinition, Severity)
from greycloak.modules import DivergenceJudge


def _ans(score, diverged):
    return {"reasoning": "r", "diverged": diverged, "divergence_score": score,
            "violated_intent": "s", "evidence": "e", "rationale": "w", "confidence": 0.9}


class _StubTarget:
    name = "t"
    def respond(self, turns):
        return AgentResponse(text="ok")


def _fixtures():
    intent = IntentProfile(purpose="p")
    risk = RiskDefinition(id="r", name="R", category=RiskCategory.DOMAIN, description="d",
                          objective="o", success_criteria="s", severity=Severity.HIGH)
    strat = AttackStrategy(id="direct", name="Direct", description="d", guidance="g")
    case = AttackCase(id="c", risk_id="r", strategy_id="direct", objective="o", turns=["hi"])
    return intent, risk, strat, case


def test_reported_judgment_is_report_judge():
    intent, risk, strat, case = _fixtures()
    opt_lm = dspy.utils.DummyLM([_ans(0.9, True)])
    rep_lm = dspy.utils.DummyLM([_ans(0.3, False)])
    engine = RedTeamEngine(
        attacker_lm=opt_lm, judge_lm=opt_lm, report_judge_lm=rep_lm,
        judge=DivergenceJudge(), report_judge=DivergenceJudge())
    result = engine.run_case(_StubTarget(), intent, risk, strat, case)
    assert result.judgment.divergence_score == 0.3        # J_rep -> reported
    assert result.opt_judgment.divergence_score == 0.9    # J_opt -> internal
    assert result.succeeded is False                       # ASR follows J_rep


def test_reused_when_report_lm_is_opt_lm():
    intent, risk, strat, case = _fixtures()
    lm = dspy.utils.DummyLM([_ans(0.8, True)])   # ONE answer: reused, not double-called
    engine = RedTeamEngine(attacker_lm=lm, judge_lm=lm, judge=DivergenceJudge())
    result = engine.run_case(_StubTarget(), intent, risk, strat, case)
    assert result.judgment.divergence_score == 0.8
    assert result.opt_judgment.divergence_score == 0.8
