import dspy
from greycloak.eval_judge import JudgeEval, evaluate_judge, load_judge_labels
from greycloak.modules import DivergenceJudge


def _perfect_lm(cases):
    answers = []
    for c in cases:
        div = c["label"] == "diverged"
        answers.append({"reasoning": "r", "diverged": div,
                        "divergence_score": float(c["score"]),
                        "violated_intent": "s", "evidence": "e", "rationale": "w",
                        "confidence": 0.9})
    return dspy.utils.DummyLM(answers)


def test_load_seed_labels():
    cases = load_judge_labels()
    assert len(cases) >= 6 and {"intent", "risk", "attack", "response", "label"} <= set(cases[0])


def test_perfect_judge_scores_1():
    cases = load_judge_labels()
    ev = evaluate_judge(cases, DivergenceJudge(), judge_lm=_perfect_lm(cases))
    assert isinstance(ev, JudgeEval)
    assert ev.accuracy == 1.0 and ev.cohen_kappa == 1.0
    assert ev.score_correlation > 0.9 and ev.disagreements == []


def test_disagreements_recorded():
    cases = load_judge_labels()
    lm = dspy.utils.DummyLM([{"reasoning": "r", "diverged": False,
                              "divergence_score": 0.0, "violated_intent": "",
                              "evidence": "", "rationale": "", "confidence": 0.5}] * len(cases))
    ev = evaluate_judge(cases, DivergenceJudge(), judge_lm=lm)
    assert ev.recall == 0.0 and len(ev.disagreements) > 0
