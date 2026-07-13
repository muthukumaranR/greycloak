# Worklog — greycloak

## Entries

### 2026-07-14T00:45:00Z
- [done] Fix even-K judge aggregation inconsistency — `aggregate_judgments` now mirrors `_single`: after the majority-vote `diverged`, added `if not diverged and median >= 0.5: diverged = True` so an even split whose median score is >= 0.5 is never reported as not-diverged. Added 2 TDD tests to tests/test_judge_aggregate.py (7 total pass).
- [done] Wire run provenance end-to-end — `build_provenance(config)` was dead code (test-only). Now called in cli.py `run(...)` (set on the persisted `RunRecord`, passed to `to_markdown`), cli.py `show(...)` (passes `rec.provenance` to `to_markdown`), and service.py `start_campaign` (set on the `RunRecord` at creation so provenance persists even if the run fails). New tests/test_provenance_wiring.py verifies the CLI path persists a record carrying provenance with a config_hash (no real LM calls, env-var-scoped store). Full suite 81/81. Commit 9ce9eb4.

### 2026-07-14T00:15:00Z
- [done] Task 7: Run provenance (reproducibility) — new `provenance.py` with `build_provenance(config)` returning a `RunProvenance` (attacker/judge/report-judge/target models, seed, sha256 config_hash [16 hex chars], greycloak + dspy versions via importlib.metadata / `dspy.__version__`). Added `RunProvenance` model above `RunRecord` in models.py and `RunRecord.provenance: RunProvenance | None`. `to_markdown` gained an optional `provenance` param (inserted before `max_examples`; all callers use single positional arg so no break) and appends a "## Provenance" block in the existing line-list style. New tests/test_provenance.py (2 tests: fields + hash stable/sensitive). TDD red→green; full suite 78/78. Commit 48a6bcf.

### 2026-07-13T23:59:00Z
- [done] Task 6: greycloak eval-judge CLI — new `@app.command(name="eval-judge")` in `cli.py` validates the divergence judge against a human-labeled set and prints n/accuracy/precision/recall, cohen_kappa, score_correlation, mean_confidence (plus optional bias line and disagreement ids). Flags: `--judge-model`, `--votes`, `--labels`, `--probe-bias`. Imported `config` as a module (`from . import config`) so the CLI calls `config.build_lm(...)`/`config.load_env()` — makes `build_lm` monkeypatchable in tests. No circular import at load time. New tests/test_cli_eval_judge.py (1 test, DummyLM via monkeypatched build_lm). TDD red→green; full suite 76/76.

### 2026-07-13T23:45:00Z
- [done] Task 5: Judge bias probes — `evaluate_judge` gained `probe_bias: bool = False`; when set it populates `JudgeEval.bias` with `{verbosity, order}` = mean absolute score shift when the response text is padded with a verbose restatement / when `IntentProfile.in_scope` and `out_of_scope` are reversed. Added module-level `_score(...)` (single scored call under optional `dspy.context`) and `_bias_probes(...)` which REUSES the main-loop `pscores` as base scores so each case costs exactly 2 extra judge calls (total `len(cases)*3` with probes on, 1 without). Default leaves `bias` empty (`{}`). New tests/test_eval_judge_bias.py (2 tests). TDD red→green; full suite 75/75. Commit f32f5ee.

### 2026-07-13T23:05:00Z
- [done] Task 4: Judge validation harness + human-labeled seed set — new `eval_judge.py` scores any `DivergenceJudge` against a human-labeled set and returns a `JudgeEval` (accuracy/precision/recall, Cohen's kappa, Pearson score correlation, mean confidence, recorded disagreements). Kappa and Pearson are pure Python (no scipy). `load_judge_labels()` reads the new `data/judge_labels.yaml` seed (6 cases across medical-advice, jailbreak, and scope-boundary risks) via `Path(__file__).parent`. `_case_to_inputs` builds IntentProfile/RiskDefinition/AttackCase/AgentResponse from each labeled row; `evaluate_judge` runs the judge under an optional `dspy.context(lm=judge_lm)`. New tests/test_eval_judge.py (3 tests). TDD red→green; full suite 73/73. Commit 067b1ff.
- Note: plan's Step 4 snippet omitted `disagreements=` in the `JudgeEval(...)` return; added it so disagreements are actually surfaced (required by test + acceptance criteria).

### 2026-07-13T22:15:00Z
- [done] Task 3: Dual-judge engine (independent report judge) — every case now scored by J_opt (recorded as `AttackResult.opt_judgment`) and an independent J_rep (the reported `judgment`, hence ASR). Added `opt_judgment` field to `AttackResult`. `RedTeamEngine.__init__` gained keyword-only `report_judge_lm`/`report_judge` (defaulting to the opt judge_lm/a fresh DivergenceJudge). `run_case` reuses the opt verdict when `report_judge_lm is judge_lm` (guard keeps single-judge tests green and avoids double-scoring); policy floor merged into both verdicts. `run_campaign` builds the report judge from `config.report_judge_lm` and fires a loud Goodhart warning when it is unset/identical to the opt judge; opt judge built with configured votes/temperature/jury. New tests/test_dual_judge.py (2 tests). TDD red→green; full suite 70/70. Commit 10112e2.

### 2026-07-13T21:30:00Z
- [done] Task 2: Ensemble DivergenceJudge + config knobs — rebuilt `DivergenceJudge` as an ensemble: `votes>1` casts K self-consistency votes (cache-distinct via `LM.copy(rollout_id, temperature)`), `lms=[...]` casts one vote per cross-family LM, `votes==1`/no lms preserves original single-call behavior. Aggregates via `aggregate_judgments`. Extracted coercion into `_single(...)`; added `import contextlib`. CampaignConfig gained `judge_votes`, `judge_vote_temperature`, `judge_lms`, `report_judge_lm`. New tests/test_judge_ensemble.py (3 tests). TDD red→green; full suite 68/68. Commit a4ad734.

### 2026-07-13T20:08:00Z
- [done] Task 1: aggregate_judgments — added pure `aggregate_judgments(votes, risk_severity)` to modules.py (median score, majority-vote diverged, agreement-based confidence, fields from nearest-median vote; empty raises, K=1 identity). Added `import statistics`. New tests/test_judge_aggregate.py (5 tests). TDD red→green; full suite 65/65. Commit a928936.

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
