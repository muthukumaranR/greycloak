# HTTP API reference

Launch with `greycloak serve` (default `http://127.0.0.1:8000`). Runs execute in a
background thread and are persisted via `RunStore`.

## `GET /`
The web dashboard (single self-contained HTML page).

## `GET /api/risks?domain=<domain>`
Built-in risks, specialized to `domain`.

```json
{
  "generic":     [ { "id": "jailbreak", "name": "...", "category": "generic", ... } ],
  "domain":      [ { "id": "overgeneralization", ... } ],
  "multi_agent": [ { "id": "handoff-injection", ... } ]
}
```

## `GET /api/strategies`
List of attack strategies (`id`, `name`, `description`, `multi_turn`).

## `POST /api/campaigns`
Start a run. Body:

```json
{
  "config": {
    "agent": { "name": "triage-bot", "system_prompt": "...", "domain": "clinic scheduling" },
    "risk_ids": ["overgeneralization", "jailbreak"],
    "strategy_ids": ["direct", "roleplay"],
    "attacks_per_pair": 2,
    "attacker_lm": { "model": "openai/gpt-4o-mini", "temperature": 0.9 },
    "judge_lm":    { "model": "openai/gpt-4o-mini", "temperature": 0.0 },
    "target_lm":   { "model": "openai/gpt-4o-mini" },
    "max_concurrency": 4
  },
  "custom_risks": [ { "id": "...", "name": "...", "objective": "...", "success_criteria": "..." } ]
}
```

Response: `{ "id": "<run_id>", "status": "pending" }`.

The `config` body is a full `CampaignConfig`; `custom_risks` is optional. `target_lm`
is used only for hosted targets — the service always builds a hosted target from the
system prompt (bring-your-own-callable and multi-agent targets are Python-API only).

## `GET /api/campaigns`
List runs (in-memory + persisted), newest first:

```json
[ { "id": "...", "status": "completed", "agent": "triage-bot", "asr": 0.42,
    "created_at": "2026-07-12T...", "progress": 12 } ]
```

## `GET /api/campaigns/{id}`
Full run state:

```json
{
  "id": "...", "status": "running|completed|failed", "agent": "triage-bot",
  "progress": [ { "phase": "attack", "message": "...", "done": 3, "total": 12, "diverged": true } ],
  "error": null,
  "report": { "attack_success_rate": 0.42, "risk_scores": [...], "results": [...] }
}
```

`report` is `null` until the run completes. `404` if the run id is unknown.

## `GET /api/campaigns/{id}/events`  (Server-Sent Events)
Live progress stream. Each `data:` line is a progress item; the stream ends with a
single `result` event carrying the final report:

```
data: {"phase": "generate", "message": "...", "done": 1, "total": 12, "diverged": null}
data: {"phase": "attack", "message": "DIVERGED: ...", "done": 3, "total": 12, "diverged": true}
data: {"phase": "result", "status": "completed", "report": { ... }, "error": null}
```

If you connect after the run already finished, you receive the buffered progress plus
the final `result` immediately. Consume with a browser `EventSource` (the dashboard
does this, falling back to polling `GET /api/campaigns/{id}` on error).
