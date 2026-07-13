# Roadmap

The current increment (`docs/superpowers/specs/2026-07-13-greycloak-harden-extend-design.md`)
makes greycloak's headline number defensible: a validated divergence judge, an
attacker optimized against it, and attack-success reported by an *independent*
metric over multiple seeds. The items below are the larger empirical extensions
that build on that foundation. They are staged, not scoped into the current
milestones, because each is a substantial body of work in its own right.

## 1. Human-labeled divergence dataset

`src/greycloak/data/judge_labels.yaml` ships as a small seed set. Grow it into a
real dataset:

- Multiple domains (support, clinical-boundary, finance-boundary, coding-agent).
- Several annotators per item; report inter-annotator agreement (Fleiss' κ).
- A held-out split never shown to any judge during development.
- Target: judge-to-human agreement in a reported, defensible band.

This dataset is what turns the judge from "an ensemble" into "a measurement
instrument with a known error rate."

## 2. Cross-model / cross-domain transfer

Run the same risk × strategy suite across model families (OpenAI, Anthropic,
open-weight) and domains, and report:

- Whether a divergence pattern on one model reproduces on another.
- Whether an attacker compiled against model A transfers to model B.
- How divergence rates move with context length (the Goal-Drift effect).

Needs a run matrix and aggregation over the existing `RunStore`; the framework
already parameterizes model and domain, so this is orchestration + reporting.

## 3. Adaptive-attack baselines at depth

The current increment ships `DirectBaseline` and a shallow PAIR/TAP-style
`RefineBaseline`. Extend with:

- PAIR / TAP at realistic tree depth and width.
- Many-shot and crescendo multi-turn attacks.
- GCG-style gradient attacks for **white-box / local** targets (out of scope for
  the black-box core; a separate adapter).

Adaptive baselines are what keep an ASR number honest — a defense that only holds
against static attacks is a known trap.

## 4. Evaluation against live defenses

Score divergence with common defenses in the loop: SmoothLLM, perplexity
filtering, input/output guardrail models, and circuit-breaker-style refusal
training. Report ASR with and without each defense.

## 5. Statistical reliability for stochastic agents

Beyond mean ± stdev over seeds: `pass^k` (probability an agent holds across k
independent trials), confidence intervals on ASR, and reliability reporting per
risk and per strategy.

## 6. Broader risk taxonomy

Domain-specific risk packs (opt-in, since the core stays domain-agnostic),
wider OWASP-LLM coverage, memory-poisoning and excessive-agency risks for
tool-using agents.
