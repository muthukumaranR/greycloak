"""greycloak — a DSPy + Pydantic red-teaming framework for agents.

Provide an agent (a callable or just a system prompt + model), the risks you
care about, and greycloak generates adversarial probes, runs them, and judges
how far the agent *diverged from its declared intent* — across generic, domain-
specific, and custom risk tiers.

Quick start::

    from greycloak import AgentSpec, CampaignConfig, run_campaign

    spec = AgentSpec(
        name="triage-bot",
        system_prompt="You are a clinic scheduling assistant. Only help with "
                      "booking, rescheduling, and cancelling appointments. Never "
                      "give medical advice; refer clinical questions to a nurse.",
        domain="clinic appointment scheduling",
    )
    report = run_campaign(CampaignConfig(agent=spec))
    print(report.attack_success_rate)
"""

from __future__ import annotations

from .agent import (
    FunctionAgent,
    HostedLMAgent,
    MultiAgentSystem,
    TargetAgent,
    build_target,
)
from .config import build_lm, default_attacker_lm, default_judge_lm, load_env
from .engine import ProgressEvent, RedTeamEngine, run_campaign
from .models import (
    AgentResponse,
    AgentSpec,
    AttackCase,
    AttackResult,
    AttackStrategy,
    CampaignConfig,
    ConversationTurn,
    DivergenceJudgment,
    IntentProfile,
    LMConfig,
    RedTeamReport,
    RiskCategory,
    RiskDefinition,
    RiskScore,
    RunRecord,
    RunStatus,
    Severity,
    ToolCall,
    ToolPolicy,
    ToolSpec,
)
from .modules import AttackEscalator, AttackGenerator, DivergenceJudge, IntentExtractor
from .policy import ToolViolation, evaluate_tools, merge_policy_into_judgment
from .optimize import build_attack_trainset, make_divergence_metric, optimize_attacker
from .report import category_breakdown, summarize, to_markdown
from .risks import (
    GENERIC_RISKS,
    MULTI_AGENT_RISKS,
    build_risk_set,
    domain_risks,
    load_custom_risks,
    select_risks,
)
from .store import RunStore, default_store
from .strategies import DEFAULT_STRATEGIES, get_strategies

__version__ = "0.1.0"

__all__ = [
    # models
    "AgentSpec", "ToolSpec", "ToolPolicy", "IntentProfile", "RiskDefinition",
    "RiskCategory", "Severity", "AttackStrategy", "AttackCase", "AgentResponse",
    "ToolCall", "ConversationTurn", "DivergenceJudgment", "AttackResult",
    "RedTeamReport", "RiskScore", "LMConfig", "CampaignConfig", "RunStatus",
    "RunRecord",
    # policy
    "ToolViolation", "evaluate_tools", "merge_policy_into_judgment",
    # persistence
    "RunStore", "default_store",
    # agents
    "TargetAgent", "FunctionAgent", "HostedLMAgent", "MultiAgentSystem", "build_target",
    # dspy modules
    "IntentExtractor", "AttackGenerator", "AttackEscalator", "DivergenceJudge",
    # risks / strategies
    "GENERIC_RISKS", "MULTI_AGENT_RISKS", "DEFAULT_STRATEGIES", "build_risk_set",
    "domain_risks", "load_custom_risks", "select_risks", "get_strategies",
    # engine
    "RedTeamEngine", "run_campaign", "ProgressEvent",
    # optimization
    "build_attack_trainset", "make_divergence_metric", "optimize_attacker",
    # config
    "build_lm", "load_env", "default_attacker_lm", "default_judge_lm",
    # report
    "summarize", "to_markdown", "category_breakdown",
    "__version__",
]
