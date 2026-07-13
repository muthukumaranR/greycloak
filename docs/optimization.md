# Optimizing the attacker (DSPy)

greycloak's attack generation is a DSPy program, so it can be **compiled** against a
metric instead of hand-tuned. The metric is the divergence the generated attacks
actually induce against the target — i.e. you optimize directly for "attacks that make
this agent diverge." Published results show this is worth doing: MIPRO-optimized DSPy
red-team programs reach roughly 4× the attack-success-rate of un-optimized ones.

## The pieces

```python
import dspy
from greycloak import (AgentSpec, HostedLMAgent, IntentProfile,
                       AttackGenerator, build_risk_set, select_risks, get_strategies,
                       build_attack_trainset, make_divergence_metric, optimize_attacker,
                       build_lm, LMConfig)

target = HostedLMAgent(AgentSpec(name="bot", system_prompt="...", domain="clinic"))
intent = IntentProfile(purpose="Schedule appointments only.", out_of_scope=["medical advice"])
risks = select_risks(build_risk_set(domain="clinic"), ["overgeneralization", "jailbreak"])
strategies = get_strategies(["direct", "roleplay"])

# 1. Trainset: one dspy.Example per (risk × strategy), carrying signature inputs.
trainset = build_attack_trainset(intent, risks, strategies, domain="clinic", n=3)

# 2. Metric: mean divergence the generated attacks induce against the target.
judge_lm = build_lm(LMConfig(model="openai/gpt-4o-mini", temperature=0.0))
metric = make_divergence_metric(target, intent, judge_lm=judge_lm)

# 3. Compile a stronger attacker.
attacker_lm = build_lm(LMConfig(model="openai/gpt-4o-mini", temperature=0.9))
with dspy.context(lm=attacker_lm):
    stronger = optimize_attacker(AttackGenerator(), trainset, metric, method="mipro")
```

The returned `stronger` is a compiled `AttackGenerator` you can hand to a
`RedTeamEngine(generator=stronger, ...)`.

## Methods

| `method` | optimizer | notes |
|---|---|---|
| `"bootstrap"` | `BootstrapFewShot` | Fast, few LM calls; good default. |
| `"mipro"` | `MIPROv2` | Joint instruction + few-shot search; stronger, slower. `auto="light"/"medium"/"heavy"`. |
| `"gepa"` | `GEPA` | Reflective prompt evolution; strongest. **Requires `reflection_lm=dspy.LM(...)`.** |

```python
reflection_lm = build_lm(LMConfig(model="openai/gpt-4o-mini"))
stronger = optimize_attacker(AttackGenerator(), trainset, metric,
                             method="gepa", reflection_lm=reflection_lm, auto="light")
```

## How the metric works

`make_divergence_metric(target, intent, judge=None, judge_lm=None)` returns a metric
that, for each generated attack, runs it against `target`, judges the response with
`DivergenceJudge` (under `judge_lm` if given), and returns the mean divergence score in
`[0, 1]`. Higher = stronger attacks. Its signature is compatible with MIPRO/Bootstrap
(`metric(example, prediction, trace)`) and GEPA (extra `pred_name`, `pred_trace`).

Because the metric *runs the target*, optimization makes real target/judge calls — keep
the trainset small while iterating, and use a cheap model for the judge.

## Optimizing the judge

The same idea applies to `DivergenceJudge`: build a labeled set of
(response, gold-diverged) pairs and compile it with any DSPy optimizer against an
accuracy metric. This is left to the user since it needs human-labeled ground truth.
