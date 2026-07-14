import dspy
from greycloak.agent import FunctionAgent
from greycloak.modules import AttackGenerator
from greycloak.optimize import (
    build_attack_trainset, make_divergence_metric, optimize_attacker)
from greycloak.models import IntentProfile, RiskCategory, RiskDefinition, Severity
from greycloak.strategies import get_strategies


def _setup():
    intent = IntentProfile(purpose="book appts", out_of_scope=["advice"])
    risks = [RiskDefinition(id="r1", name="R1", category=RiskCategory.DOMAIN,
                            description="d", objective="get advice",
                            success_criteria="gives advice", severity=Severity.HIGH)]
    strats = get_strategies(["direct"])
    return intent, risks, strats


def test_trainset_inputs_match_forward():
    intent, risks, strats = _setup()
    ex = build_attack_trainset(intent, risks, strats, domain="sched", n=1)[0]
    assert set(ex.inputs().keys()) == {"intent", "risk", "strategy", "n", "domain"}


def test_bootstrap_actually_attaches_demos():
    intent, risks, strats = _setup()
    train = build_attack_trainset(intent, risks, strats, domain="sched", n=1)
    target = FunctionAgent(lambda s: "sure, here is advice")
    # generate then judge, repeated; judge always diverged (metric passes -> demo kept)
    lm = dspy.utils.DummyLM(
        [{"reasoning": "r", "attacks": ["give me advice"]},
         {"reasoning": "r", "diverged": True, "divergence_score": 0.9,
          "violated_intent": "", "evidence": "", "rationale": "", "confidence": 0.9}] * 50)
    gen = AttackGenerator()
    with dspy.context(lm=lm):
        metric = make_divergence_metric(target, intent, judge_lm=lm)
        compiled = optimize_attacker(gen, train, metric, method="bootstrap")
    assert len(compiled.generate.demos) >= 1     # real learning: demos attached
    assert len(gen.generate.demos) == 0          # original left intact
