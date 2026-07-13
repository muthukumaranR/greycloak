# Worklog — greycloak

## Entries

### 2026-07-13T12:50:00Z
- [done] Recovered lost source — all 45 files (15 src modules, 12 tests, 8 docs, 4 examples, frontend, memory) had been wiped from disk (only .pyc bytecode remained; nothing committed to git). Reconstructed every file by replaying the session transcript's Write/Edit history and restored them in place.
- [done] Fixed incomplete refactor — service.py had `get_store()` defined but 4 call sites still referenced the removed module-global `_STORE` (left mid-refactor at end of prior session). Completed the refactor; full suite now 60/60 green.
- [new] Protect work — framework is still untracked/uncommitted; needs an initial git commit to prevent another loss.

### 2026-07-12T18:30:00Z
- [done] Tool policy + rule-based divergence — ToolPolicy (allowed/forbidden/sensitive); policy.py evaluates tool traces and merges a divergence floor into the judgment (forbidden 0.95 / not-allowlisted 0.85 / sensitive 0.60), never lowering an LM score. Wired into engine.run_case.
- [done] Multi-agent / handoff testing — MultiAgentSystem adapter (named sub-agents, optional router, HANDOFF: sentinel, max_hops), aggregates tool calls + agent_path; MULTI_AGENT_RISKS (handoff-injection, cross-agent-escalation, delegation-confusion) auto-enabled when target is multi-agent. Verified live: front->billing handoff, cross-agent-escalation flagged 0.70/critical.
- [done] Run persistence — RunStore (atomic JSON files under $GREYCLOAK_RUNS_DIR); RunRecord model; service persists on start/completion and lists memory+disk; CLI `runs` and `show`.
- [done] SSE progress streaming — /api/campaigns/{id}/events via StreamingResponse + per-run queue; frontend uses EventSource with polling fallback.
- [done] GEPA optimizer path — optimize_attacker method="gepa" (requires reflection_lm); metric made GEPA-signature-compatible.
- [done] Docs + README — docs/ set (architecture, usage, risks, multi-agent, optimization, api, index) + refreshed README covering all new features. Test suite now 60, all green.

### 2026-07-12T16:00:00Z
- [done] Research landscape — deep-research workflow (21 verified claims, 23 sources) confirmed no existing tool unifies agent+prompts+tools input, intent-divergence, DSPy-native attacks, and Pydantic schemas. Captured in docs/LANDSCAPE.md. A new framework is warranted.
- [done] Scaffold + core — uv project (src/greycloak), Pydantic models, agent adapters (FunctionAgent + HostedLMAgent with provider-agnostic tool-call capture), DSPy signatures/modules (IntentExtractor, AttackGenerator, AttackEscalator, DivergenceJudge).
- [done] Risk library + strategies — generic (5), domain overgeneralization tier (4, templated on domain), custom loader (YAML/dict); 7 attack strategies incl. multi-turn escalation.
- [done] Engine + report + CLI — orchestrator with separate attacker/judge/target LMs, multi-turn escalation, concurrency; ASR + per-risk/category metrics + markdown; typer CLI (risks/strategies/run/serve).
- [done] DSPy optimization — build_attack_trainset + make_divergence_metric + optimize_attacker (MIPRO/Bootstrap) so the attacker compiles against divergence.
- [done] FastAPI service + web dashboard — POST/GET campaign endpoints, background runs, catalog endpoints; single-page dashboard with live progress and divergence drill-down. Verified booting over real HTTP.
- [done] Tests — 39 pytest tests, fully offline via DSPy DummyLM (no API keys). All green.
- [done] Live verification — end-to-end against gpt-4o-mini: well-aligned agent held (0% ASR); deliberately mis-built agent caught at 100% ASR (divergence 1.0 unsafe advice, 0.8 overgeneralization) with intent-anchored judge rationales.
- [new] Future — persistence for runs (currently in-memory), SSE progress streaming, multi-agent/handoff testing, richer tool-trace divergence signals, GEPA optimizer path.
