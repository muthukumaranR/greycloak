import dspy
from greycloak.engine import build_run_context, run_campaign
from greycloak.models import AgentSpec, CampaignConfig, RedTeamReport
from greycloak.modules import AttackGenerator


def _cfg():
    return CampaignConfig(agent=AgentSpec(name="a", system_prompt="Book appts only.",
                                          domain="scheduling"),
                          risk_ids=["overgeneralization"], strategy_ids=["direct"],
                          attacks_per_pair=1)


def test_build_run_context_populated():
    ctx = build_run_context(_cfg())
    assert ctx.target is not None and ctx.opt_judge is not None
    assert ctx.risks and ctx.strategies
    assert ctx.attacker_lm is not None and ctx.report_judge_lm is not None


def test_run_campaign_accepts_injected_attacker(monkeypatch):
    used = {}
    class _Atk(AttackGenerator):
        def forward(self, intent, risk, strategy, n=2, domain=None):
            used["hit"] = True
            return super().forward(intent, risk, strategy, n=n, domain=domain)
    lm = dspy.utils.DummyLM([
        {"reasoning": "r", "purpose": "book", "in_scope": [], "out_of_scope": ["advice"],
         "prohibited_behaviors": [], "tone": ""},
        {"reasoning": "r", "attacks": ["do X"]},
        {"reasoning": "r", "diverged": False, "divergence_score": 0.1,
         "violated_intent": "", "evidence": "", "rationale": "", "confidence": 0.9},
    ])
    monkeypatch.setattr("greycloak.engine.build_lm", lambda cfg: lm)
    report = run_campaign(_cfg(), attacker=_Atk())
    assert isinstance(report, RedTeamReport) and used.get("hit")
