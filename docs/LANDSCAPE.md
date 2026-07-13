# Landscape: agent red-teaming frameworks

Cited comparison from a multi-source, adversarially-verified research pass
(21 confirmed claims, 23 sources). It answers: *can existing systems, alone or
combined, be a one-stop-shop for domain-specific agent red-teaming — or is a new
framework warranted?*

## The four capabilities greycloak targets

1. **Agent + prompts + tools as structured input** (not just a black-box endpoint)
2. **Intent-divergence measurement** (a 0–1 score of deviation from declared intent)
3. **DSPy-native, optimizable attack authoring**
4. **Pydantic-typed schemas** end to end
5. …plus a **three-tier risk taxonomy** (generic / domain-nuance / custom)

## Tool-by-tool

| System | Agent input | Generic risks | Domain nuance (overgen.) | Custom risks | Intent divergence | DSPy | Pydantic |
|---|---|---|---|---|---|---|---|
| **Giskard** | black-box wrap | ✅ OWASP LLM Top-10 | ⚠️ generic scan only | ✅ ScenarioGenerator | ❌ | ❌ | ❌ |
| **DeepTeam / DeepEval** | callback / model name | ✅ | ⚠️ | ✅ CustomVulnerability | ❌ | ❌ (own MIPROv2) | ❌ |
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
- **DSPy optimizers generalize to adversarial search.** A 2026 report applies
  three DSPy optimizers (MIPROv2, GEPA, SIMBA) to jailbreak prompts, optimizing a
  continuous 0–1 danger score from an independent judge. GEPA can be inverted for
  adversarial prompt search (dropped AIME-2025 pass@1 from 76%→10%).
  — <https://arxiv.org/abs/2603.19247> · <https://arxiv.org/pdf/2507.19457>

## Conclusion

**A new framework is warranted.** Every requirement is proven *individually* in
the wild, but no tool unifies them — least of all around **domain-specific intent
divergence** with a **DSPy-native, Pydantic-typed** core. A "combination" would
mean gluing Giskard's risk taxonomy, Haize's DSPy attacks, and Goal-Drift's
metric across three incompatible codebases. greycloak is the thin unifying layer:
it reuses the *ideas* (Attack/Refine + MIPRO, auditor/judge split, custom-vuln
schema, 0–1 divergence) inside one typed, optimizable pipeline.
