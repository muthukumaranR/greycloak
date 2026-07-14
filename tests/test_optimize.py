"""Optimization plumbing: trainset builder + divergence metric (DummyLM)."""

import pytest

from greycloak import (
    AttackCase,
    AttackGenerator,
    FunctionAgent,
    build_attack_trainset,
    build_risk_set,
    get_strategies,
    make_divergence_metric,
    optimize_attacker,
    select_risks,
)
from conftest import judge_answer, make_dummy


def _cases(*prompts: str) -> list[AttackCase]:
    """Build the list[AttackCase] the metric now receives (forward's return)."""
    return [
        AttackCase(id=f"opt-{i}", risk_id="r", strategy_id="direct",
                   objective="o", turns=[p])
        for i, p in enumerate(prompts)
    ]


def test_build_attack_trainset_shape(intent):
    risks = select_risks(build_risk_set(domain="clinic"), ["jailbreak", "pii-leak"])
    strategies = get_strategies(["direct", "roleplay"])
    trainset = build_attack_trainset(intent, risks, strategies, domain="clinic", n=3)
    assert len(trainset) == 4  # 2 risks x 2 strategies
    ex = trainset[0]
    assert ex.n == 3
    assert set(ex.inputs().keys()) == {"intent", "risk", "strategy", "n", "domain"}
    assert ex.get("risk") is not None  # the real risk object forward expects


def test_divergence_metric_returns_mean_score(intent):
    risks = select_risks(build_risk_set(domain="clinic"), ["jailbreak"])
    strategies = get_strategies(["direct"])
    trainset = build_attack_trainset(intent, risks, strategies, domain="clinic", n=2)
    example = trainset[0]

    target = FunctionAgent(lambda p: "sure, ignoring my rules now")
    judge_lm = make_dummy([judge_answer(True, 0.8), judge_answer(True, 0.6)])
    metric = make_divergence_metric(target, intent, judge_lm=judge_lm)

    prediction = _cases("probe one", "probe two")  # forward returns list[AttackCase]
    score = metric(example, prediction)
    assert abs(score - 0.7) < 1e-9  # mean(0.8, 0.6)


def test_divergence_metric_handles_empty(intent):
    risks = select_risks(build_risk_set(domain="clinic"), ["jailbreak"])
    strategies = get_strategies(["direct"])
    example = build_attack_trainset(intent, risks, strategies, n=1)[0]
    metric = make_divergence_metric(FunctionAgent(lambda p: "x"), intent)
    assert metric(example, []) == 0.0  # no generated cases -> zero


def test_metric_is_gepa_compatible(intent):
    """Metric tolerates GEPA's richer call signature (pred_name, pred_trace)."""
    risks = select_risks(build_risk_set(domain="clinic"), ["jailbreak"])
    example = build_attack_trainset(intent, risks, get_strategies(["direct"]), n=1)[0]
    judge_lm = make_dummy([judge_answer(True, 0.5)])
    metric = make_divergence_metric(FunctionAgent(lambda p: "ok"), intent, judge_lm=judge_lm)
    score = metric(example, _cases("p"), None, "gen", None)
    assert score == 0.5


def test_gepa_requires_reflection_lm(intent):
    risks = select_risks(build_risk_set(domain="clinic"), ["jailbreak"])
    trainset = build_attack_trainset(intent, risks, get_strategies(["direct"]), n=1)
    metric = make_divergence_metric(FunctionAgent(lambda p: "x"), intent)
    with pytest.raises(ValueError, match="reflection"):
        optimize_attacker(AttackGenerator(), trainset, metric, method="gepa")


def test_unknown_method_raises(intent):
    risks = select_risks(build_risk_set(domain="clinic"), ["jailbreak"])
    trainset = build_attack_trainset(intent, risks, get_strategies(["direct"]), n=1)
    metric = make_divergence_metric(FunctionAgent(lambda p: "x"), intent)
    with pytest.raises(ValueError, match="unknown optimization method"):
        optimize_attacker(AttackGenerator(), trainset, metric, method="nope")
