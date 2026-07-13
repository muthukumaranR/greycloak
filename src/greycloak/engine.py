"""The red-team engine: orchestrates generate -> attack -> judge.

Flow for a campaign:

1. Resolve the agent's :class:`IntentProfile` (declared, or DSPy-extracted).
2. For every (risk x strategy) pair, generate ``attacks_per_pair`` probes with the
   attacker LM.
3. Run each probe against the target. Multi-turn strategies escalate adaptively
   using the agent's own replies.
4. Judge each final response for divergence with the judge LM.
5. Aggregate into a :class:`RedTeamReport`.

Attacker, judge, and target LMs are scoped independently via ``dspy.context`` so
you can, e.g., attack/judge with a local unrestricted model and target a hosted
one.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import dspy
from loguru import logger

from .agent import MultiAgentSystem, TargetAgent, build_target
from .config import build_lm, load_env
from .models import (
    AgentResponse,
    AgentSpec,
    AttackCase,
    AttackResult,
    AttackStrategy,
    CampaignConfig,
    IntentProfile,
    RedTeamReport,
    RiskDefinition,
)
from .modules import AttackEscalator, AttackGenerator, DivergenceJudge, IntentExtractor
from .policy import evaluate_tools, merge_policy_into_judgment
from .report import summarize
from .risks import build_risk_set, select_risks
from .strategies import get_strategies

ProgressCallback = Callable[["ProgressEvent"], None]


@dataclass
class ProgressEvent:
    """Emitted as the campaign advances (for CLIs, the service, streaming UIs)."""

    phase: str  # "intent" | "generate" | "attack" | "done"
    message: str
    done: int = 0
    total: int = 0
    result: AttackResult | None = None
    extra: dict = field(default_factory=dict)


class RedTeamEngine:
    """Drives a campaign. Modules are injectable so they can be pre-compiled."""

    def __init__(
        self,
        attacker_lm,
        judge_lm,
        *,
        generator: AttackGenerator | None = None,
        judge: DivergenceJudge | None = None,
        escalator: AttackEscalator | None = None,
        intent_extractor: IntentExtractor | None = None,
        max_escalation_turns: int = 3,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.attacker_lm = attacker_lm
        self.judge_lm = judge_lm
        self.generator = generator or AttackGenerator()
        self.judge = judge or DivergenceJudge()
        self.escalator = escalator or AttackEscalator()
        self.intent_extractor = intent_extractor or IntentExtractor()
        self.max_escalation_turns = max_escalation_turns
        self._progress = progress or (lambda ev: None)

    # -- phases ------------------------------------------------------------- #
    def resolve_intent(
        self, system_prompt: str, domain: str | None, declared: IntentProfile | None
    ) -> IntentProfile:
        if declared is not None:
            return declared
        with dspy.context(lm=self.judge_lm):
            intent = self.intent_extractor(system_prompt, domain)
        logger.info("extracted intent: {}", intent.purpose)
        return intent

    def generate_cases(
        self,
        intent: IntentProfile,
        risk: RiskDefinition,
        strategy: AttackStrategy,
        n: int,
        domain: str | None,
    ) -> list[AttackCase]:
        with dspy.context(lm=self.attacker_lm):
            return self.generator(intent, risk, strategy, n=n, domain=domain)

    def run_case(
        self,
        target: TargetAgent,
        intent: IntentProfile,
        risk: RiskDefinition,
        strategy: AttackStrategy,
        case: AttackCase,
        spec: AgentSpec | None = None,
    ) -> AttackResult:
        response = target.respond(case.turns)
        if strategy.multi_turn and not response.error:
            case, response = self._escalate(target, intent, risk, strategy, case, response)
        with dspy.context(lm=self.judge_lm):
            judgment = self.judge(intent, risk, case, response)
        # Deterministic tool-use signals override an over-lenient text judgment.
        if spec is not None:
            violations = evaluate_tools(response, spec)
            judgment = merge_policy_into_judgment(judgment, violations, risk.severity)
        return AttackResult(
            case=case, risk=risk, strategy=strategy, response=response, judgment=judgment
        )

    def _escalate(
        self,
        target: TargetAgent,
        intent: IntentProfile,
        risk: RiskDefinition,
        strategy: AttackStrategy,
        case: AttackCase,
        response: AgentResponse,
    ) -> tuple[AttackCase, AgentResponse]:
        """Adaptively extend a multi-turn attack using the agent's replies."""
        turns = list(case.turns)
        for _ in range(self.max_escalation_turns - 1):
            convo = "\n".join(turns)
            with dspy.context(lm=self.attacker_lm):
                next_turn, stop = self.escalator(
                    intent, risk, strategy, convo, response.text
                )
            if stop or not next_turn.strip():
                break
            turns.append(next_turn.strip())
            response = target.respond(turns)
            if response.error:
                break
        case = case.model_copy(update={"turns": turns})
        return case, response

    # -- top level ---------------------------------------------------------- #
    def run(
        self,
        config: CampaignConfig,
        target: TargetAgent,
        risks: list[RiskDefinition],
        strategies: list[AttackStrategy],
        declared_intent: IntentProfile | None = None,
    ) -> RedTeamReport:
        self._progress(ProgressEvent("intent", "Resolving agent intent"))
        intent = self.resolve_intent(
            config.agent.system_prompt, config.agent.domain, declared_intent
        )

        # Phase 1: generate all cases up front (sequential -> deterministic).
        cases: list[tuple[RiskDefinition, AttackStrategy, AttackCase]] = []
        pairs = [(r, s) for r in risks for s in strategies]
        for i, (risk, strategy) in enumerate(pairs, 1):
            self._progress(
                ProgressEvent(
                    "generate",
                    f"Generating attacks: {risk.name} x {strategy.name}",
                    done=i,
                    total=len(pairs),
                )
            )
            generated = self.generate_cases(
                intent, risk, strategy, config.attacks_per_pair, config.agent.domain
            )
            cases.extend((risk, strategy, c) for c in generated)

        logger.info("generated {} attack case(s)", len(cases))

        # Phase 2: run + judge each case (optionally concurrent).
        results: list[AttackResult] = [None] * len(cases)  # type: ignore[list-item]

        def _work(idx: int) -> None:
            risk, strategy, case = cases[idx]
            result = self.run_case(target, intent, risk, strategy, case, spec=config.agent)
            results[idx] = result
            self._progress(
                ProgressEvent(
                    "attack",
                    f"{'DIVERGED' if result.succeeded else 'held'}: "
                    f"{risk.name} via {strategy.name}",
                    done=idx + 1,
                    total=len(cases),
                    result=result,
                )
            )

        if config.max_concurrency > 1 and len(cases) > 1:
            with ThreadPoolExecutor(max_workers=config.max_concurrency) as pool:
                list(pool.map(_work, range(len(cases))))
        else:
            for idx in range(len(cases)):
                _work(idx)

        results = [r for r in results if r is not None]
        report = summarize(config.agent.name, intent, results)
        self._progress(
            ProgressEvent(
                "done",
                f"Done: {report.total_successes}/{report.total_attacks} diverged",
                done=len(cases),
                total=len(cases),
            )
        )
        return report


def run_campaign(
    config: CampaignConfig,
    agent_fn: Callable | None = None,
    target: TargetAgent | None = None,
    custom_risks: list[RiskDefinition] | None = None,
    declared_intent: IntentProfile | None = None,
    progress: ProgressCallback | None = None,
) -> RedTeamReport:
    """One-call entry point used by the CLI and service.

    Builds LMs, the target, and the risk/strategy sets from ``config``, then runs
    the engine.

    * Pass ``agent_fn`` to attack your own agent callable.
    * Pass a prebuilt ``target`` (e.g. a :class:`MultiAgentSystem`) to attack an
      already-constructed system; multi-agent risks are then included by default.
    * Omit both to build a hosted LM target from ``config.agent`` + ``target_lm``.
    """

    load_env()
    attacker_lm = build_lm(config.attacker_lm)
    judge_lm = build_lm(config.judge_lm)
    if target is None:
        target = build_target(config.agent, agent_fn, config.target_lm)

    include_multi_agent = isinstance(target, MultiAgentSystem)
    all_risks = build_risk_set(
        domain=config.agent.domain,
        include_multi_agent=include_multi_agent,
        custom=custom_risks,
    )
    risks = select_risks(all_risks, config.risk_ids)
    strategies = get_strategies(config.strategy_ids)

    engine = RedTeamEngine(
        attacker_lm,
        judge_lm,
        max_escalation_turns=config.max_escalation_turns,
        progress=progress,
    )
    return engine.run(config, target, risks, strategies, declared_intent)
