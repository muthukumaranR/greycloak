# Landscape: agent red-teaming frameworks

Cited comparison from a multi-source, adversarially-verified research pass
(21 confirmed claims, 23 sources). It answers: *can existing systems, alone or
combined, be a one-stop-shop for domain-specific agent red-teaming — or is a new
framework warranted?*

## Where greycloak leads — and where it doesn't

The defensible, currently-empty white space is narrow and specific:

> **No existing tool pairs a continuous intent-divergence score (0–1) with an
> attacker compiled to maximize exactly that score — and reports the outcome with
> an *independent* metric so the optimization cannot game its own reward.**

Everything else greycloak provides is individually already occupied and is best
treated as table stakes, not a differentiator:

- **Typed agent + tools + policy as input** — an implementation choice; typed
  agent-spec ingestion exists elsewhere (e.g. SplxAI Agentic Radar).
- **DSPy / optimizer-compiled attacks** — Haize `dspy-redteam` shipped a
  DSPy + MIPRO attacker against a continuous judge in 2024; the recipe is not new.
- **Pydantic-typed schemas** — an engineering convenience, copyable in a weekend.
- **Three-tier risk taxonomy** — custom-risk authoring also exists in Giskard,
  DeepTeam, and promptfoo.

The one genuinely distinguishing element is the **coupling**: an intent-anchored,
continuous divergence score used both as the optimizer's reward and (via a
*separate, validated* metric) as the reported outcome. That is what the framework
is built around.

## Tool-by-tool

| System | Agent input | Generic risks | Domain nuance (overgen.) | Custom risks | Intent divergence | DSPy | Pydantic |
|---|---|---|---|---|---|---|---|
| **Giskard** | black-box wrap | ✅ OWASP LLM Top-10 | ⚠️ generic scan only | ✅ ScenarioGenerator | ❌ | ❌ | ❌ |
| **DeepTeam / DeepEval** | callback / model name | ✅ | ⚠️ | ✅ CustomVulnerability | ❌ | ❌ | ❌ |
| **promptfoo** | HTTP / Py provider | ✅ plugins | ⚠️ | ✅ custom plugin (gen+grader) | ⚠️ "intent" plugin (BYO prompts) | ❌ | ❌ (JS/YAML) |
| **Haize `dspy-redteam`** | target LLM | ⚠️ jailbreak only | ❌ | ❌ | ⚠️ danger score | ✅ MIPRO Attack/Refine | ❌ |
| **AgentDojo** | own task suites | ⚠️ injection only | ❌ | ⚠️ new tasks | ✅ utility+security | ❌ | ❌ |
| **AgentHarm** | benchmark tasks | ✅ 110 malicious tasks | ❌ | ❌ | ❌ (capability retention) | ❌ | ❌ |
| **Petri (Anthropic)** | target via auditor | ✅ open-ended | ⚠️ auditor-driven | ⚠️ seed instructions | ⚠️ judge-scored | ❌ | ❌ |
| **greycloak** | ✅ AgentSpec (prompt+tools) or callable | ✅ | ✅ overgeneralization tier | ✅ Pydantic/YAML | ✅ 0–1 score | ✅ signatures/modules | ✅ throughout |

Legend: ✅ first-class · ⚠️ partial/adjacent · ❌ absent.

## Key verified findings (with sources)

- **Giskard** is an Apache-2.0 library for *testing and evaluating agentic
  systems*; Scan wraps "an LLM, a black-box agent, or a multi-step pipeline" and
  auto-generates adversarial inputs across the **OWASP LLM Top-10**; custom risks
  via `ScenarioGenerator` / `vulnerability_suite_generator_registry`. It does
  **not** take the agent's tools/prompt as structured inputs, and isn't
  DSPy/Pydantic. — <https://github.com/Giskard-AI/giskard-oss>
- **Intent-divergence is a real, published metric.** "Evaluating Goal Drift in LM
  Agents" gives agents a goal via system prompt, applies competing pressures, and
  quantifies drift with two 0–1 metrics (GD_actions, GD_inaction) over a
  tool-using agent (`buy_stock`, `sell_stock`, …). — <https://arxiv.org/abs/2505.02709>
- **AgentDojo** is scoped to prompt-injection robustness of tool-calling agents;
  extensible environment, jointly measures utility + security. Not a domain-risk
  or custom-risk red-teamer. — <https://arxiv.org/abs/2406.13352>
- **AgentHarm** is a static benchmark of 110 (→440) malicious agent tasks across
  11 harm categories; measures capability retention after jailbreak, not
  user-defined intent divergence. — <https://arxiv.org/abs/2410.09024>
- **DSPy is a proven red-teaming engine.** Haize Labs compiles a deep
  Attack/Refine DSPy program with **MIPRO + an LLM judge**, reaching **44% ASR
  (4× baseline)** on Vicuna with no manual prompt engineering.
  — <https://blog.haizelabs.com/posts/dspy/> · <https://github.com/haizelabs/dspy-redteam>
- **DSPy optimizers have been turned on jailbreak prompts.** "When Prompt
  Optimization Becomes Jailbreaking" applies three DSPy optimizers (MIPROv2, GEPA,
  SIMBA) to jailbreak prompts, optimizing a continuous 0–1 danger score from an
  independent judge. — <https://arxiv.org/abs/2603.19247>
- **GEPA has a built-in adversarial mode.** The GEPA paper (§5.2, "GEPA for
  Adversarial Prompt Search") inverts the reward to evolve a universal,
  task-preserving distractor that cut GPT-5-Mini pass@1 on AIME-2025 from 76% to
  10%. This is *capability degradation*, not a harm/jailbreak red-team — but it is
  direct evidence that the same optimizer that improves a prompt can be pointed at
  breaking one. — <https://arxiv.org/pdf/2507.19457> (§5.2)

## Conclusion

**A new framework is warranted, on a narrow and honest basis.** Most of what
greycloak does is individually available in the wild (typed ingestion, DSPy
attacks, custom risks, Pydantic types). The part that is not: coupling a
continuous, intent-anchored divergence score to an attacker optimized against it,
while reporting outcomes with an **independent** metric so the score cannot be
gamed. That coupling is what greycloak is built around; everything else is table
stakes it happens to also cover in one place.

Closest neighbors to stay honest about: **DeepTeam** (closest overall framework —
custom risks, multi-turn attacks, agent-drift vulnerabilities), **Haize
`dspy-redteam`** (closest on the attack axis — DSPy + a continuous judge, though
it targets a single LLM with a generic danger score rather than deviation from a
declared intent), and typed agent-spec ingestion tools such as **SplxAI Agentic
Radar**. The lead is also time-boxed — competitors are converging on this
intersection — which argues for building the coupling well rather than resting on
the feature list above.
