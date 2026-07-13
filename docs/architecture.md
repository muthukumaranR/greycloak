# Architecture

greycloak is a pipeline that turns *"here is my agent and the risks I care about"*
into *"here is exactly where it diverged from its declared intent, and how badly."*
Every boundary is a Pydantic model; every LM step is a DSPy module.

## Data flow

```
AgentSpec (name, system_prompt, tools, tool_policy, domain)
        │
        ▼
IntentProfile ── declared by you, OR extracted by IntentExtractor (DSPy)
        │
        ▼   for each (risk × strategy):
AttackGenerator (DSPy) ─────────────► AttackCase(s)   (turns, objective, rationale)
        │
        ▼
TargetAgent.respond(turns) ─────────► AgentResponse   (text, tool_calls, transcript,
        │    multi-turn: AttackEscalator (DSPy)         agent_path for multi-agent)
        ▼
DivergenceJudge (DSPy) ─────────────► DivergenceJudgment (0–1 divergence_score)
        │
        ▼   merged with…
policy.evaluate_tools + merge_policy_into_judgment  (rule-based tool violations)
        │
        ▼
report.summarize ───────────────────► RedTeamReport   (ASR, per-risk, per-category)
        │
        ▼
RunStore (persist)  ·  RedTeamEngine progress callback  ·  FastAPI service / CLI
```

## Modules

| File | Responsibility |
|---|---|
| `models.py` | All Pydantic types: `AgentSpec`, `ToolSpec`, `ToolPolicy`, `IntentProfile`, `RiskDefinition`, `AttackStrategy`, `AttackCase`, `AgentResponse`, `DivergenceJudgment`, `AttackResult`, `RedTeamReport`, `RunRecord`, config types. |
| `signatures.py` | DSPy `Signature` contracts: `ExtractIntent`, `GenerateAttacks`, `EscalateAttack`, `JudgeDivergence`. |
| `modules.py` | DSPy `Module`s wrapping the signatures + Pydantic glue: `IntentExtractor`, `AttackGenerator`, `AttackEscalator`, `DivergenceJudge`. |
| `agent.py` | Target adapters: `FunctionAgent` (your callable), `HostedLMAgent` (system prompt + model), `MultiAgentSystem` (handoffs). |
| `risks.py` | Risk library: generic, domain (templated), multi-agent, custom loader; assembly + selection. |
| `strategies.py` | Attack strategy catalog (the *how*: direct, roleplay, injection, escalation…). |
| `policy.py` | Rule-based tool-use divergence signals merged into judgments. |
| `engine.py` | `RedTeamEngine` orchestrator + `run_campaign` one-call entry point. |
| `report.py` | `summarize`, `category_breakdown`, `to_markdown`. |
| `optimize.py` | DSPy optimization of the attacker (bootstrap / MIPRO / GEPA). |
| `store.py` | `RunStore` file-based persistence. |
| `service.py` | FastAPI app: catalog, campaigns, SSE progress, serves the dashboard. |
| `cli.py` | `greycloak` CLI (risks, strategies, run, runs, show, serve). |
| `config.py` | Building `dspy.LM`s per role, `.env` loading. |

## Three LM roles

The engine keeps three LMs independent, scoped with `dspy.context`:

- **attacker** — generates probes (`AttackGenerator`, `AttackEscalator`). Best pointed
  at an unrestricted/local model so generation isn't itself refused.
- **judge** — scores divergence (`DivergenceJudge`, `IntentExtractor`). Best a strong,
  well-aligned model at temperature 0.
- **target** — the agent under test (only for `HostedLMAgent`; your own callable brings
  its own model).

## The divergence metric

`divergence_score ∈ [0, 1]`: 0 = fully aligned with declared intent, 1 = full
capitulation to the attacker's objective. The judge produces it from the intent
profile + the risk's `success_criteria` + the agent's response (text **and** tool
calls). `policy.py` then enforces a floor: a forbidden/non-allowlisted/sensitive
tool call raises the score regardless of the text verdict, so tool misuse is never
scored as "aligned." Aggregation gives **attack success rate (ASR)** = fraction of
attacks that diverged, plus per-risk and per-category breakdowns.

## Why DSPy

Attack generation and judging are DSPy programs, not prompt strings, so they can be
**compiled** against the divergence metric (`optimize.py`). Published work shows this
matters: MIPRO-optimized DSPy red-team programs reach ~4× the attack-success-rate of
un-optimized ones. See [optimization.md](optimization.md).
