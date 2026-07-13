from greycloak.models import AgentSpec, CampaignConfig, LMConfig
from greycloak.provenance import build_provenance


def _cfg(model="openai/gpt-4o-mini"):
    return CampaignConfig(agent=AgentSpec(name="a", system_prompt="p"),
                          judge_lm=LMConfig(model=model), seed=7)


def test_provenance_fields():
    p = build_provenance(_cfg())
    assert p.judge_model == "openai/gpt-4o-mini" and p.seed == 7
    assert p.greycloak_version and p.dspy_version and p.config_hash


def test_config_hash_is_stable_and_sensitive():
    assert build_provenance(_cfg()).config_hash == build_provenance(_cfg()).config_hash
    assert build_provenance(_cfg()).config_hash != build_provenance(_cfg("openai/gpt-4o")).config_hash
