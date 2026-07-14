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

import contextlib
import random
import statistics
from typing import Callable

import dspy
from loguru import logger
from pydantic import BaseModel

from .agent import TargetAgent
from .baselines import DirectBaseline
from .engine import RedTeamEngine, build_run_context
from .models import AttackStrategy, CampaignConfig, IntentProfile, RiskDefinition
from .modules import AttackGenerator, DivergenceJudge
from .store import save_attacker
from .strategies import STRATEGY_INDEX


def build_attack_trainset(
    intent: IntentProfile,
    risks: list[RiskDefinition],
    strategies: list[AttackStrategy],
    domain: str | None = None,
    n: int = 3,
) -> list[dspy.Example]:
    """Build a DSPy trainset of (risk x strategy) inputs for optimizing attacks.

    Each :class:`dspy.Example` carries exactly the inputs
    :meth:`AttackGenerator.forward` consumes (the real ``intent``/``risk``/
    ``strategy`` objects), so the optimizer can call ``module(**example.inputs())``
    and actually generate attacks.
    """
    examples: list[dspy.Example] = []
    for risk in risks:
        for strategy in strategies:
            examples.append(dspy.Example(
                intent=intent, risk=risk, strategy=strategy, n=n, domain=domain or "",
            ).with_inputs("intent", "risk", "strategy", "n", "domain"))
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
        # `prediction` is AttackGenerator.forward's return: a list[AttackCase].
        cases = prediction if isinstance(prediction, list) else []
        risk: RiskDefinition = example.get("risk") or example.get("_risk")
        if not cases or risk is None:
            return 0.0
        scores: list[float] = []
        for case in cases:
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


def split_pairs(pairs, train_frac: float, seed: int):
    """Deterministically split a list of (risk, strategy) pairs into (train, eval)."""
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    k = max(1, round(len(pairs) * train_frac)) if len(pairs) > 1 else len(pairs)
    train = [pairs[i] for i in idx[:k]]
    eval_ = [pairs[i] for i in idx[k:]] or train  # tiny sets: eval on train (caller warns)
    return train, eval_


def build_examples_for_pairs(intent, pairs, domain=None, n: int = 3):
    """dspy.Examples (one per pair) for the optimizer, mirroring build_attack_trainset."""
    examples = []
    for risk, strategy in pairs:
        examples.append(dspy.Example(
            intent=intent, risk=risk, strategy=strategy, n=n, domain=domain or "",
        ).with_inputs("intent", "risk", "strategy", "n", "domain"))
    return examples


def measure_asr(attacker, pairs, intent, target, judge, domain=None,
                attacker_lm=None, judge_lm=None, n: int = 1) -> dict:
    """Run `attacker` over `pairs` against `target`, judge each with `judge`.

    Returns {asr, mean_divergence, n}. Pass the report judge (J_rep) as `judge`
    (and its LM as `judge_lm`) when measuring a number you intend to report.
    """
    total = 0
    successes = 0
    scores: list[float] = []
    for risk, strategy in pairs:
        actx = dspy.context(lm=attacker_lm) if attacker_lm is not None else contextlib.nullcontext()
        with actx:
            cases = attacker(intent, risk, strategy, n=n, domain=domain)
        for case in cases:
            response = target.respond(case.turns)
            jctx = dspy.context(lm=judge_lm) if judge_lm is not None else contextlib.nullcontext()
            with jctx:
                judgment = judge(intent, risk, case, response)
            total += 1
            successes += 1 if judgment.diverged else 0
            scores.append(judgment.divergence_score)
    return {"asr": successes / total if total else 0.0,
            "mean_divergence": sum(scores) / len(scores) if scores else 0.0,
            "n": total}


class OptimizationResult(BaseModel):
    method: str
    seeds: int
    train_frac: float
    arms: dict[str, dict]     # arm -> {asr_mean, asr_stdev, div_mean}
    opt_rep_gap: float        # compiled: J_opt ASR - J_rep ASR (reward-gaming signal)
    attacker_id: str | None = None


def run_optimization(config: CampaignConfig, method: str = "bootstrap",
                     train_frac: float = 0.6, seeds: int = 1,
                     save_id: str | None = None, progress=None) -> OptimizationResult:
    ctx = build_run_context(config)
    engine = RedTeamEngine(ctx.attacker_lm, ctx.judge_lm, judge=ctx.opt_judge,
                           report_judge_lm=ctx.report_judge_lm)
    intent = engine.resolve_intent(config.agent.system_prompt, config.agent.domain, None)
    domain = config.agent.domain
    pairs = [(r, s) for r in ctx.risks for s in ctx.strategies]
    rep_judge = DivergenceJudge()

    per_seed = {"direct": [], "baseline": [], "compiled": []}
    div_seed = {"direct": [], "baseline": [], "compiled": []}
    opt_rep_gaps = []
    compiled_final = None
    for s in range(seeds):
        train_pairs, eval_pairs = split_pairs(pairs, train_frac, seed=config.seed + s)
        train_ex = build_examples_for_pairs(intent, train_pairs, domain, config.attacks_per_pair)
        opt_metric = make_divergence_metric(ctx.target, intent, judge=ctx.opt_judge,
                                            judge_lm=ctx.judge_lm)
        base_gen = AttackGenerator()
        with dspy.context(lm=ctx.attacker_lm):
            compiled = optimize_attacker(base_gen, train_ex, opt_metric, method=method)
        compiled_final = compiled
        for name, atk in (("direct", DirectBaseline()), ("baseline", base_gen),
                          ("compiled", compiled)):
            r = measure_asr(atk, eval_pairs, intent, ctx.target, rep_judge, domain=domain,
                            attacker_lm=ctx.attacker_lm, judge_lm=ctx.report_judge_lm,
                            n=config.attacks_per_pair)
            per_seed[name].append(r["asr"]); div_seed[name].append(r["mean_divergence"])
        opt_r = measure_asr(compiled, eval_pairs, intent, ctx.target, ctx.opt_judge,
                            domain=domain, attacker_lm=ctx.attacker_lm, judge_lm=ctx.judge_lm,
                            n=config.attacks_per_pair)
        opt_rep_gaps.append(opt_r["asr"] - per_seed["compiled"][-1])

    def _agg(xs):
        return {"asr_mean": round(statistics.mean(xs), 4),
                "asr_stdev": round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0}
    arms = {name: {**_agg(per_seed[name]),
                   "div_mean": round(statistics.mean(div_seed[name]), 4)}
            for name in per_seed}
    attacker_id = None
    if save_id and compiled_final is not None:
        save_attacker(save_id, compiled_final); attacker_id = save_id
    return OptimizationResult(method=method, seeds=seeds, train_frac=train_frac, arms=arms,
                              opt_rep_gap=round(statistics.mean(opt_rep_gaps), 4),
                              attacker_id=attacker_id)
