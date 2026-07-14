import dspy
from greycloak.agent import FunctionAgent
from greycloak.baselines import DirectBaseline
from greycloak.modules import DivergenceJudge
from greycloak.optimize import measure_asr, split_pairs
from greycloak.models import (
    AttackStrategy, IntentProfile, RiskCategory, RiskDefinition, Severity)


def _pair(rid):
    risk = RiskDefinition(id=rid, name=rid, category=RiskCategory.DOMAIN, description="d",
                          objective="get advice", success_criteria="s", severity=Severity.HIGH)
    strat = AttackStrategy(id="direct", name="D", description="d", guidance="g")
    return risk, strat


def test_direct_baseline_no_lm():
    intent = IntentProfile(purpose="p")
    risk, strat = _pair("r")
    cases = DirectBaseline()(intent, risk, strat, n=1)
    assert cases and cases[0].turns and "get advice" in cases[0].turns[0].lower()


def test_measure_asr_counts_divergences():
    intent = IntentProfile(purpose="p")
    pairs = [_pair("r1"), _pair("r2")]
    target = FunctionAgent(lambda s: "sure, here is advice")
    judge = DivergenceJudge()
    lm = dspy.utils.DummyLM([
        {"reasoning": "r", "diverged": True, "divergence_score": 0.9,
         "violated_intent": "", "evidence": "", "rationale": "", "confidence": 0.9},
        {"reasoning": "r", "diverged": False, "divergence_score": 0.2,
         "violated_intent": "", "evidence": "", "rationale": "", "confidence": 0.9}])
    out = measure_asr(DirectBaseline(), pairs, intent, target, judge, judge_lm=lm, n=1)
    assert out["n"] == 2 and out["asr"] == 0.5


def test_split_pairs_deterministic():
    pairs = [(i, i) for i in range(10)]
    a1, b1 = split_pairs(pairs, 0.6, seed=3)
    a2, b2 = split_pairs(pairs, 0.6, seed=3)
    assert a1 == a2 and b1 == b2
    assert len(a1) == 6 and len(b1) == 4 and not set(a1) & set(b1)
