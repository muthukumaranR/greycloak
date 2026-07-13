import dspy
from typer.testing import CliRunner
from greycloak import cli, config
from greycloak.eval_judge import load_judge_labels


def test_eval_judge_cli(monkeypatch):
    cases = load_judge_labels()
    lm = dspy.utils.DummyLM([{"reasoning": "r", "diverged": c["label"] == "diverged",
                              "divergence_score": float(c["score"]), "violated_intent": "s",
                              "evidence": "e", "rationale": "w", "confidence": 0.9}
                             for c in cases])
    monkeypatch.setattr(config, "build_lm", lambda cfg: lm)
    result = CliRunner().invoke(cli.app, ["eval-judge", "--judge-model", "dummy"])
    assert result.exit_code == 0
    assert "kappa" in result.stdout.lower()
