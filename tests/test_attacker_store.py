import dspy
from greycloak.modules import AttackGenerator
from greycloak.store import load_attacker, save_attacker
from greycloak.models import (
    AttackStrategy, IntentProfile, RiskCategory, RiskDefinition, Severity)


def _fixtures():
    intent = IntentProfile(purpose="p")
    risk = RiskDefinition(id="r", name="R", category=RiskCategory.DOMAIN, description="d",
                          objective="o", success_criteria="s", severity=Severity.HIGH)
    strat = AttackStrategy(id="direct", name="D", description="d", guidance="g")
    return intent, risk, strat


def test_save_load_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("GREYCLOAK_RUNS_DIR", str(tmp_path))
    gen = AttackGenerator()
    path = save_attacker("atk1", gen)
    assert path.exists()
    loaded = load_attacker("atk1")
    assert isinstance(loaded, AttackGenerator)
    intent, risk, strat = _fixtures()
    with dspy.context(lm=dspy.utils.DummyLM([{"reasoning": "r", "attacks": ["x"]}])):
        cases = loaded(intent, risk, strat, n=1)
    assert cases and cases[0].turns == ["x"]
