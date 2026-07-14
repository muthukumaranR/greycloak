# Landscape: agent red-teaming frameworks

greycloak is worth building for one reason, and it is narrow: no existing tool pairs a continuous intent-divergence score with an attacker compiled to push that score up, and then reports the result with a *different*, validated judge so the optimization can't grade its own work. Everything else it does, someone already does. This document is the evidence for that claim — a comparison across 23 sources, with 21 individual claims checked against primary sources rather than taken on trust.

Start with the obvious objection. "Agent-plus-tools as typed input, a divergence score, DSPy-compiled attacks, Pydantic throughout" sounds like four things nobody else has. It isn't. Taken one at a time, three of the four already have a strong occupant, and the fourth is a weekend of typing.

## Where greycloak actually leads

The one genuinely empty spot is the coupling. An intent-anchored 0–1 divergence score is used two ways at once: as the reward the attacker is optimized against, and — through a separate, validated judge — as the number you report. Optimize and measure with the same judge and you are measuring how well the attacker gamed the judge, not how far the agent drifted from what it was told to do. Keeping the two judges apart is the whole design. It is the part no one else has put together.

The rest is table stakes greycloak happens to cover in one place, not a lead:

- **Typed agent + tools + policy as input.** An implementation choice. SplxAI Agentic Radar already ingests typed agent specs.
- **DSPy / optimizer-compiled attacks.** Haize's `dspy-redteam` shipped a DSPy + MIPRO attacker against a continuous judge in 2024. The recipe is two years old.
- **Pydantic schemas.** A convenience, copyable in a weekend.
- **A three-tier risk taxonomy.** Custom-risk authoring already exists in Giskard, DeepTeam, and promptfoo.

## Tool by tool

| Tool | Agent input | Generic risks | Domain nuance (overgen.) | Custom risks | Intent divergence | DSPy | Pydantic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Giskard** | black-box wrap | ✅ OWASP LLM Top-10 | ⚠️ generic scan only | ✅ ScenarioGenerator | ❌ | ❌ | ❌ |
| **DeepTeam / DeepEval** | callback / model name | ✅ | ⚠️ | ✅ CustomVulnerability | ❌ | ❌ | ❌ |
| **promptfoo** | HTTP / Py provider | ✅ plugins | ⚠️ | ✅ custom plugin (gen+grader) | ⚠️ "intent" plugin (BYO prompts) | ❌ | ❌ (JS/YAML) |
| **Haize `dspy-redteam`** | target LLM | ⚠️ jailbreak only | ❌ | ❌ | ⚠️ danger score | ✅ MIPRO Attack/Refine | ❌ |
| **AgentDojo** | own task suites | ⚠️ injection only | ❌ | ⚠️ new tasks | ✅ utility+security | ❌ | ❌ |
| **AgentHarm** | benchmark tasks | ✅ 110 malicious tasks | ❌ | ❌ | ❌ (capability retention) | ❌ | ❌ |
| **Petri (Anthropic)** | target via auditor | ✅ open-ended | ⚠️ auditor-driven | ⚠️ seed instructions | ⚠️ judge-scored | ❌ | ❌ |
| **greycloak** | ✅ AgentSpec (prompt+tools) or callable | ✅ | ✅ overgeneralization tier | ✅ Pydantic/YAML | ✅ 0–1 score | ✅ signatures/modules | ✅ throughout |

Legend: ✅ first-class · ⚠️ partial or adjacent · ❌ absent.

## What the sources say

- **Giskard** is an Apache-2.0 library for testing and evaluating agentic systems. Scan wraps "an LLM, a black-box agent, or a multi-step pipeline" and auto-generates adversarial inputs across the **OWASP LLM Top-10**, with custom risks via `ScenarioGenerator` / `vulnerability_suite_generator_registry`. It does not take the agent's tools or prompt as structured inputs, and it isn't DSPy or Pydantic. — <https://github.com/Giskard-AI/giskard-oss>
- **Intent divergence is a real, published metric, not something we invented.** "Evaluating Goal Drift in LM Agents" gives an agent a goal in its system prompt, applies competing pressures, and quantifies the drift with two 0–1 metrics (GD_actions, GD_inaction) over a tool-using agent (`buy_stock`, `sell_stock`, …). — <https://arxiv.org/abs/2505.02709>
- **AgentDojo** covers prompt-injection robustness of tool-calling agents in an extensible environment, scoring utility and security together. It is not a domain-risk or custom-risk red-teamer. — <https://arxiv.org/abs/2406.13352>
- **AgentHarm** is a static benchmark of 110 (→440) malicious agent tasks across 11 harm categories. It measures capability retention after a jailbreak, not divergence from a declared intent. — <https://arxiv.org/abs/2410.09024>
- **Compiling an attacker with DSPy works.** Haize Labs compiled an Attack/Refine DSPy program with MIPRO and an LLM judge and reached **44% attack-success rate — four times the baseline** — on Vicuna, with no manual prompt engineering. — <https://blog.haizelabs.com/posts/dspy/> · <https://github.com/haizelabs/dspy-redteam>
- **DSPy optimizers have already been turned on jailbreak prompts.** "When Prompt Optimization Becomes Jailbreaking" runs three of them (MIPROv2, GEPA, SIMBA) against a continuous 0–1 danger score from an independent judge. — <https://arxiv.org/abs/2603.19247>
- **GEPA ships an adversarial mode of its own.** In §5.2, "GEPA for Adversarial Prompt Search," the authors invert the reward and evolve a universal, task-preserving distractor that cut GPT-5-Mini's pass@1 on AIME-2025 from 76% to 10%. That is capability degradation, not a harm or jailbreak red-team — but it is direct evidence that the same optimizer that sharpens a prompt can be pointed at breaking one. — <https://arxiv.org/pdf/2507.19457> (§5.2)

## Where this leaves greycloak

A new framework is warranted, on that one narrow basis. It is worth naming the neighbors plainly instead of talking up the gap. **DeepTeam** is the closest thing overall — custom risks, multi-turn attacks, and agent-drift work, and it is actively maintained. **Haize `dspy-redteam`** is closest on the attacker side, and it is essentially the attack half already built, though it points at a single LLM and scores a generic "danger" rather than deviation from a declared intent. **SplxAI Agentic Radar** is closest on typed-spec ingestion.

That lead has a clock on it. Several of these are converging on the same intersection — DeepTeam is adding drift work, and a cluster of 2026 preprints is circling the attacker-plus-continuous-judge recipe. Best case the window is a year; worst case it is already closing. What would settle it is one of them shipping a *reported* metric that is independent of the *optimized* one — and none has yet. Either way the conclusion is the same, which is why I'd put the effort there: build the coupling well and measure it honestly, because the feature checklist above won't hold the ground and the honest-measurement piece is the part that will.
