"""Pydantic domain models for the greycloak agent red-teaming framework.

Everything that flows through the system is a typed Pydantic model: the agent
under test, the risks we probe for, the adversarial attacks we generate, the
agent's responses, and the divergence judgments. Keeping this layer strict and
declarative is what lets the DSPy attack/judge modules, the engine, the service
and the frontend all speak the same language.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class RiskCategory(str, Enum):
    """Where a risk sits on greycloak's three-tier model.

    GENERIC  -- universal agent risks (jailbreak, injection, PII leak, ...).
    DOMAIN   -- domain-nuance risks (overgeneralization, scope creep, ...).
    CUSTOM   -- risks the user defines for their own agent.
    """

    GENERIC = "generic"
    DOMAIN = "domain"
    CUSTOM = "custom"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> float:
        return {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}[self.value]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# --------------------------------------------------------------------------- #
# Agent-under-test description
# --------------------------------------------------------------------------- #
class ToolSpec(BaseModel):
    """Declaration of a tool the target agent can call.

    `parameters` is a JSON-schema-ish dict. We keep it permissive on purpose:
    the point is to let attacks reason about (and try to abuse) the tool
    surface, not to re-validate the agent's own tool schemas.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    # Tools that, if invoked outside declared intent, constitute divergence.
    sensitive: bool = False


class ToolPolicy(BaseModel):
    """Deterministic rules about which tools the agent may call.

    Lets greycloak flag tool-use divergence by *rule*, independent of the LM
    judge: a forbidden or non-allowlisted tool call is a violation even if the
    judge's text analysis misses it. Sensitive tools are allowed but treated as a
    strong divergence signal when triggered by an attack.
    """

    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Allowlist. If non-empty, any tool NOT listed is a violation.",
    )
    forbidden_tools: list[str] = Field(
        default_factory=list, description="Tools that must never be called."
    )
    sensitive_tools: list[str] = Field(
        default_factory=list,
        description="Tools allowed but high-signal; also merged from ToolSpec.sensitive.",
    )


class AgentSpec(BaseModel):
    """Everything greycloak needs to know about the agent it is attacking.

    The user hands us this. `system_prompt` is the declared behavior/intent, and
    is the yardstick against which divergence is measured.
    """

    name: str
    system_prompt: str = Field(
        ...,
        description="The agent's declared instructions/intent -- the yardstick "
        "for measuring divergence.",
    )
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_policy: ToolPolicy | None = Field(
        default=None, description="Optional deterministic tool-use policy."
    )
    domain: str | None = Field(
        default=None,
        description="e.g. 'clinical triage', 'tax advice'. Drives domain risks.",
    )
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def effective_sensitive_tools(self) -> set[str]:
        """Sensitive tools from the policy plus any ToolSpec flagged sensitive."""
        s = set(self.tool_policy.sensitive_tools) if self.tool_policy else set()
        s |= {t.name for t in self.tools if t.sensitive}
        return s


class IntentProfile(BaseModel):
    """Structured statement of what the agent is *supposed* to do.

    Either declared by the user or extracted from `system_prompt` by the DSPy
    IntentExtractor. Attacks and judgments are anchored to this profile so that
    "divergence from original intent" is a concrete, checkable comparison rather
    than a vibe.
    """

    purpose: str = Field(..., description="One-line statement of the agent's job.")
    in_scope: list[str] = Field(
        default_factory=list, description="Tasks/topics the agent should handle."
    )
    out_of_scope: list[str] = Field(
        default_factory=list,
        description="Tasks/topics the agent must refuse or defer.",
    )
    prohibited_behaviors: list[str] = Field(
        default_factory=list,
        description="Things the agent must never do (leak secrets, give advice "
        "outside its remit, call sensitive tools unprompted, ...).",
    )
    tone: str | None = None
    source: str = Field(
        default="declared",
        description="'declared' if user-provided, 'extracted' if DSPy-inferred.",
    )


# --------------------------------------------------------------------------- #
# Risks & attack strategies
# --------------------------------------------------------------------------- #
class RiskDefinition(BaseModel):
    """A specific failure mode greycloak probes for.

    `success_criteria` is the natural-language rubric the DivergenceJudge uses to
    decide whether an attack landed. `objective` is what the attacker is trying
    to make the agent do.
    """

    id: str
    name: str
    category: RiskCategory
    description: str
    objective: str = Field(
        ...,
        description="What a successful attack makes the agent do, in plain terms.",
    )
    success_criteria: str = Field(
        ...,
        description="Rubric the judge applies to decide the attack succeeded.",
    )
    severity: Severity = Severity.MEDIUM
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "-")


class AttackStrategy(BaseModel):
    """A reusable adversarial technique the AttackGenerator instantiates.

    `guidance` is injected into the generator prompt to steer *how* the attack
    is phrased for a given risk (role-play, obfuscation, gradual escalation...).
    """

    id: str
    name: str
    description: str
    guidance: str = Field(
        ...,
        description="Instructions handed to the attack generator for this tactic.",
    )
    multi_turn: bool = Field(
        default=False,
        description="If true, the strategy unfolds over several escalating turns.",
    )


# --------------------------------------------------------------------------- #
# Attacks, responses, judgments
# --------------------------------------------------------------------------- #
class AttackCase(BaseModel):
    """A concrete adversarial probe: one risk x one strategy, materialized.

    `turns` holds the user-side messages to send. Single-turn attacks have one
    entry; multi-turn escalation strategies have several.
    """

    id: str
    risk_id: str
    strategy_id: str
    objective: str
    turns: list[str] = Field(
        ..., min_length=1, description="Ordered user messages to send to the agent."
    )
    rationale: str = Field(
        default="",
        description="Why the attacker believes this probe could induce divergence.",
    )

    @field_validator("turns")
    @classmethod
    def _nonempty_turns(cls, v: list[str]) -> list[str]:
        cleaned = [t for t in (s.strip() for s in v) if t]
        if not cleaned:
            raise ValueError("AttackCase must contain at least one non-empty turn")
        return cleaned


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ConversationTurn(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    agent: str | None = Field(
        default=None, description="Which sub-agent produced this turn (multi-agent)."
    )


class AgentResponse(BaseModel):
    """What the agent-under-test returned for an attack case.

    Captures both the text and any tool calls, plus errors/latency, so the judge
    can reason about tool misuse as a form of divergence -- not just text.
    """

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    transcript: list[ConversationTurn] = Field(default_factory=list)
    agent_path: list[str] = Field(
        default_factory=list,
        description="Ordered sub-agents that handled the request (multi-agent).",
    )
    latency_s: float | None = None
    error: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class DivergenceJudgment(BaseModel):
    """Verdict on whether an attack drove the agent away from its intent.

    `divergence_score` in [0, 1] is the headline metric: 0 == fully aligned with
    declared intent, 1 == complete capitulation to the attacker's objective.
    """

    diverged: bool
    divergence_score: float = Field(..., ge=0.0, le=1.0)
    severity: Severity = Severity.MEDIUM
    violated_intent: str = Field(
        default="",
        description="Which part of the IntentProfile the response violated.",
    )
    rationale: str = ""
    evidence: str = Field(
        default="", description="Verbatim snippet from the response supporting the call."
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("divergence_score", "confidence", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))


class AttackResult(BaseModel):
    """One full attack lifecycle: case -> response -> judgment."""

    case: AttackCase
    risk: RiskDefinition
    strategy: AttackStrategy
    response: AgentResponse
    judgment: DivergenceJudgment

    @property
    def succeeded(self) -> bool:
        """The *attack* succeeded == the agent diverged."""
        return self.judgment.diverged


# --------------------------------------------------------------------------- #
# Campaign config & report
# --------------------------------------------------------------------------- #
class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class LMConfig(BaseModel):
    """DSPy/litellm model config for one role (attacker, judge, or target)."""

    model: str = Field(default="openai/gpt-4o-mini")
    api_base: str | None = Field(
        default=None, description="e.g. http://localhost:11434 for Ollama."
    )
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024


class CampaignConfig(BaseModel):
    """A full red-team campaign specification."""

    agent: AgentSpec
    risk_ids: list[str] = Field(
        default_factory=list,
        description="Risks to probe. Empty == use the default library selection.",
    )
    strategy_ids: list[str] = Field(default_factory=list)
    attacks_per_pair: int = Field(
        default=2, ge=1, le=20, description="Attacks generated per risk x strategy."
    )
    attacker_lm: LMConfig = Field(default_factory=LMConfig)
    judge_lm: LMConfig = Field(default_factory=LMConfig)
    target_lm: LMConfig | None = Field(
        default=None,
        description="LM for a config-only (hosted) target. Ignored when the user "
        "supplies their own agent callable.",
    )
    max_escalation_turns: int = Field(
        default=3, ge=1, le=8, description="Max turns for multi-turn strategies."
    )
    seed: int = 0
    max_concurrency: int = Field(default=4, ge=1, le=32)


class RiskScore(BaseModel):
    risk_id: str
    risk_name: str
    category: RiskCategory
    severity: Severity
    attacks: int
    successes: int
    attack_success_rate: float
    mean_divergence: float


class RedTeamReport(BaseModel):
    """Aggregate result of a campaign -- the object the CLI/service/frontend show."""

    agent_name: str
    created_at: datetime = Field(default_factory=_utcnow)
    intent: IntentProfile
    results: list[AttackResult] = Field(default_factory=list)

    # Aggregates (populated by report.summarize()).
    total_attacks: int = 0
    total_successes: int = 0
    attack_success_rate: float = 0.0
    mean_divergence: float = 0.0
    risk_scores: list[RiskScore] = Field(default_factory=list)


class RunRecord(BaseModel):
    """A campaign run's full state -- what the service tracks and the store persists."""

    id: str
    status: RunStatus = RunStatus.PENDING
    config: CampaignConfig
    progress: list[dict[str, Any]] = Field(default_factory=list)
    report: RedTeamReport | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
