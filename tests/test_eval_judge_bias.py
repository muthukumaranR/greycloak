import dspy
from greycloak.eval_judge import evaluate_judge, load_judge_labels
from greycloak.modules import DivergenceJudge


def _const_lm(score, n):
    return dspy.utils.DummyLM([{"reasoning": "r", "diverged": score >= 0.5,
                                "divergence_score": score, "violated_intent": "",
                                "evidence": "", "rationale": "", "confidence": 0.9}] * n)


def test_bias_zero_when_judge_is_invariant():
    cases = load_judge_labels()
    # 3 calls/case (1 main + 2 probes); constant score -> zero delta
    ev = evaluate_judge(cases, DivergenceJudge(),
                        judge_lm=_const_lm(0.5, len(cases) * 3), probe_bias=True)
    assert ev.bias["verbosity"] == 0.0 and ev.bias["order"] == 0.0


def test_bias_empty_by_default():
    cases = load_judge_labels()
    ev = evaluate_judge(cases, DivergenceJudge(), judge_lm=_const_lm(0.5, len(cases)))
    assert ev.bias == {}
