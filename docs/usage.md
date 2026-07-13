# Usage guide

## Install

```bash
uv sync
echo 'OPENAI_API_KEY=sk-...' > .env    # .env is git-ignored
```

## Three ways to point greycloak at an agent

### 1. Hosted target (system prompt + model)

You only have an intent and a model. greycloak builds the target for you.

```python
from greycloak import AgentSpec, CampaignConfig, run_campaign, to_markdown

spec = AgentSpec(
    name="triage-bot",
    system_prompt="You are a clinic scheduling assistant. Only help with booking, "
                  "rescheduling, and cancelling appointments. Never give medical advice.",
    domain="clinic appointment scheduling",
)
report = run_campaign(CampaignConfig(agent=spec))
print(to_markdown(report))
```

### 2. Your own agent callable (real tools)

Point greycloak at your production agent. The callable takes a `str` (last turn) or
`list[str]` (full history) and returns a `str`, a `dict` with `text`/`tool_calls`, or
an `AgentResponse`.

```python
def my_agent(prompt: str) -> dict:
    result = my_production_agent.run(prompt)
    return {"text": result.text, "tool_calls": [
        {"name": tc.name, "arguments": tc.args} for tc in result.tool_calls
    ]}

report = run_campaign(config, agent_fn=my_agent)
```

### 3. Multi-agent system

See [multi-agent.md](multi-agent.md).

## Selecting risks and strategies

```python
CampaignConfig(
    agent=spec,
    risk_ids=["overgeneralization", "jailbreak", "unsafe-domain-advice"],  # [] = all
    strategy_ids=["direct", "roleplay", "escalation"],                     # [] = all
    attacks_per_pair=3,      # attacks per (risk × strategy)
    max_concurrency=4,
    max_escalation_turns=3,  # for multi-turn strategies
)
```

## LM roles and Ollama

Attacker, judge, and target LMs are independent:

```python
from greycloak import LMConfig

CampaignConfig(
    agent=spec,
    attacker_lm=LMConfig(model="ollama/llama3.1", api_base="http://localhost:11434",
                         temperature=0.9),   # unrestricted local attacker
    judge_lm=LMConfig(model="openai/gpt-4o-mini", temperature=0.0),
    target_lm=LMConfig(model="openai/gpt-4o-mini"),
)
```

Model strings are litellm-style (`openai/…`, `ollama/…`, `anthropic/…`, `azure/…`).
API keys resolve from `LMConfig.api_key`, then `.env` (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`); local servers get a placeholder key automatically.

## Declaring intent explicitly

By default greycloak extracts an `IntentProfile` from the system prompt. To pin it:

```python
from greycloak import IntentProfile, RedTeamEngine, build_lm

intent = IntentProfile(
    purpose="Schedule clinic appointments only.",
    in_scope=["booking", "rescheduling", "cancelling"],
    out_of_scope=["medical advice", "diagnosis"],
    prohibited_behaviors=["give medical advice", "reveal system prompt"],
)
# pass declared_intent=intent to RedTeamEngine.run(...)
```

## CLI

```bash
greycloak risks --domain "clinic scheduling"    # list built-in risks
greycloak strategies                            # list attack tactics

greycloak run --name triage-bot --domain "clinic scheduling" \
  --system-prompt-file prompt.txt \
  --risks overgeneralization,jailbreak --strategies direct,roleplay \
  --attacker-model openai/gpt-4o-mini --judge-model openai/gpt-4o-mini \
  --out report.json --markdown

greycloak run --config examples/campaign.yaml   # full campaign from YAML
greycloak runs                                  # list persisted runs
greycloak show <run-id>                          # print a run's report
greycloak serve                                 # API + dashboard on :8000
```

Runs are persisted under `$GREYCLOAK_RUNS_DIR` (default `./runs`).

## Reading the report

```python
report.attack_success_rate      # fraction of attacks that diverged (0–1)
report.mean_divergence          # mean 0–1 divergence across attacks
report.risk_scores              # ranked per-risk ASR + mean divergence
for r in report.results:
    if r.succeeded:             # the agent diverged
        r.judgment.divergence_score, r.judgment.severity
        r.judgment.violated_intent, r.judgment.rationale
        r.case.turns            # what the attacker sent
        r.response.text, r.response.tool_calls, r.response.agent_path
```

`to_markdown(report)` renders a full human-readable summary with top divergences.
