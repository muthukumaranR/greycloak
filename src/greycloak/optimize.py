"""DSPy optimization of the attacker (and judge).

The whole reason greycloak's attack generation is a DSPy program -- not a pile of
prompt strings -- is so it can be *compiled*. Published results are striking:
Haize Labs took a DSPy Attack/Refine program from 10% to 44% attack-success-rate
with MIPRO and zero manual prompt engineering. This module wires greycloak's
:class:`AttackGenerator` up to that machinery.

The optimization signal is the judge's divergence score: an attack "batch" is
good to the extent that, when run against the target, it makes the agent diverge.
So the metric here actually *runs the target and judges it*, which is exactly the
objective we care about.
"""

from __future__ import annotations

from typing import Callable

import dspy
from loguru import logger

from .agent import TargetAgent
from .models import AttackCase, AttackStrategy, IntentProfile, RiskDefinition
from .modules import AttackGenerator, DivergenceJudge, _short_id
from .strategies import STRATEGY_INDEX


def build_attack_trainset(
    intent: IntentProfile,
    risks: list[RiskDefinition],
    strategies: list[AttackStrategy],
    domain: str | None = None,
    n: int = 3,
) -> list[dspy.Example]:
    """Build a DSPy trainset of (risk x strategy) inputs for optimizing attacks.

    Each :class:`dspy.Example` carries exactly the inputs the ``GenerateAttacks``
    signature consumes, so it can be fed straight to a DSPy optimizer.
    """
    from .modules import _render_intent

    examples: list[dspy.Example] = []
    for risk in risks:
        for strategy in strategies:
            examples.append(
                dspy.Example(
                    intent=_render_intent(intent),
                    domain=domain or "",
                    risk_name=risk.name,
                    risk_description=risk.description,
                    risk_objective=risk.objective,
                    strategy_name=strategy.name,
                    strategy_guidance=strategy.guidance,
                    n=n,
                    # carried for the metric; not signature inputs
                    _risk=risk,
                    _strategy=strategy,
                ).with_inputs(
                    "intent", "domain", "risk_name", "risk_description",
                    "risk_objective", "strategy_name", "strategy_guidance", "n",
                )
            )
    logger.info("built attack trainset with {} example(s)", len(examples))
    return examples


def make_divergence_metric(
    target: TargetAgent,
    intent: IntentProfile,
    judge: DivergenceJudge | None = None,
    judge_lm=None,
) -> Callable:
    """Return a DSPy metric: mean divergence the generated attacks induce.

    The metric runs each generated attack against ``target`` and judges it, then
    returns the mean divergence score in [0, 1]. Higher == stronger attacks. Use
    it directly with any DSPy optimizer (``metric=make_divergence_metric(...)``).
    """
    judge = judge or DivergenceJudge()

    def metric(example: dspy.Example, prediction, trace=None,
               pred_name=None, pred_trace=None) -> float:
        # Extra args (pred_name, pred_trace) keep this compatible with GEPA,
        # which calls metrics with a richer signature than MIPRO/Bootstrap.
        attacks = getattr(prediction, "attacks", None) or []
        risk: RiskDefinition = example.get("_risk")
        strategy: AttackStrategy = example.get("_strategy", None)
        if not attacks or risk is None:
            return 0.0
        scores: list[float] = []
        for prompt in attacks:
            if not str(prompt).strip():
                continue
            case = AttackCase(
                id=_short_id("opt"),
                risk_id=risk.id,
                strategy_id=getattr(strategy, "id", "direct"),
                objective=risk.objective,
                turns=[str(prompt)],
            )
            response = target.respond(case.turns)
            if judge_lm is not None:
                with dspy.context(lm=judge_lm):
                    judgment = judge(intent, risk, case, response)
            else:
                judgment = judge(intent, risk, case, response)
            scores.append(judgment.divergence_score)
        return sum(scores) / len(scores) if scores else 0.0

    return metric


def optimize_attacker(
    generator: AttackGenerator,
    trainset: list[dspy.Example],
    metric: Callable,
    method: str = "bootstrap",
    **kwargs,
) -> AttackGenerator:
    """Compile a stronger attacker against ``metric`` using a DSPy optimizer.

    ``method``:
      * ``"bootstrap"`` -- BootstrapFewShot (fast, robust, few LM calls).
      * ``"mipro"``     -- MIPROv2 (instruction + few-shot search; stronger, slower).
      * ``"gepa"``      -- GEPA reflective prompt evolution (strongest; requires a
                          ``reflection_lm=dspy.LM(...)`` kwarg). GEPA mutates the
                          attacker's instructions by reflecting on failures, which
                          is well suited to evolving jailbreak/attack prompts.

    Returns the optimized :class:`AttackGenerator`; the original is left intact.
    """
    if method == "mipro":
        optimizer = dspy.MIPROv2(metric=metric, auto=kwargs.pop("auto", "light"), **kwargs)
        compiled = optimizer.compile(generator, trainset=trainset, requires_permission_to_run=False)
    elif method == "bootstrap":
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=kwargs.pop("max_bootstrapped_demos", 4),
            max_labeled_demos=kwargs.pop("max_labeled_demos", 0),
            **kwargs,
        )
        compiled = optimizer.compile(generator, trainset=trainset)
    elif method == "gepa":
        reflection_lm = kwargs.pop("reflection_lm", None)
        if reflection_lm is None:
            raise ValueError(
                "GEPA requires a reflection LM: pass reflection_lm=dspy.LM(...)"
            )
        optimizer = dspy.GEPA(
            metric=metric,
            auto=kwargs.pop("auto", "light"),
            reflection_lm=reflection_lm,
            **kwargs,
        )
        compiled = optimizer.compile(generator, trainset=trainset)
    else:
        raise ValueError(f"unknown optimization method: {method!r}")
    logger.info("attacker optimized via {}", method)
    return compiled
