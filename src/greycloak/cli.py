"""greycloak command-line interface.

Examples::

    # List what ships
    greycloak risks
    greycloak strategies

    # Red-team a hosted target defined by a system prompt + model
    greycloak run --name triage-bot --domain "clinic scheduling" \\
        --system-prompt-file prompt.txt --target-model openai/gpt-4o-mini \\
        --attacker-model openai/gpt-4o-mini --judge-model openai/gpt-4o-mini \\
        --out report.json --markdown

    # Run the whole thing from a YAML campaign file
    greycloak run --config campaign.yaml

    # Launch the API + web dashboard
    greycloak serve
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
import yaml
from loguru import logger

import uuid

from . import config
from .engine import ProgressEvent, run_campaign
from .eval_judge import evaluate_judge, load_judge_labels
from .models import AgentSpec, CampaignConfig, LMConfig, RunRecord, RunStatus
from .modules import DivergenceJudge
from .optimize import run_optimization
from .provenance import build_provenance
from .report import to_markdown
from .risks import GENERIC_RISKS, domain_risks, load_custom_risks
from .store import default_store
from .strategies import DEFAULT_STRATEGIES

app = typer.Typer(help="DSPy + Pydantic red-teaming for LLM agents.", no_args_is_help=True)


def _progress(ev: ProgressEvent) -> None:
    if ev.total:
        typer.echo(f"[{ev.phase} {ev.done}/{ev.total}] {ev.message}", err=True)
    else:
        typer.echo(f"[{ev.phase}] {ev.message}", err=True)


@app.command()
def risks(domain: str = typer.Option("your domain", help="Domain for domain risks.")):
    """List the built-in generic and domain risks."""
    typer.echo("GENERIC RISKS")
    for r in GENERIC_RISKS:
        typer.echo(f"  {r.id:<24} [{r.severity.value}] {r.name}")
    typer.echo(f"\nDOMAIN RISKS (specialized to: {domain!r})")
    for r in domain_risks(domain):
        typer.echo(f"  {r.id:<24} [{r.severity.value}] {r.name}")


@app.command()
def strategies():
    """List the built-in attack strategies."""
    for s in DEFAULT_STRATEGIES:
        tag = " (multi-turn)" if s.multi_turn else ""
        typer.echo(f"  {s.id:<14} {s.name}{tag}\n      {s.description}")


@app.command()
def run(
    config: Path = typer.Option(None, help="YAML campaign config (overrides flags)."),
    name: str = typer.Option("agent-under-test", help="Agent name."),
    system_prompt: str = typer.Option(None, help="Agent system prompt text."),
    system_prompt_file: Path = typer.Option(None, help="File with the system prompt."),
    domain: str = typer.Option(None, help="Agent domain (drives domain risks)."),
    risk_ids: str = typer.Option("", "--risks", help="Comma-separated risk ids."),
    strategy_ids: str = typer.Option("", "--strategies", help="Comma-separated strat ids."),
    attacks_per_pair: int = typer.Option(2, help="Attacks per risk x strategy."),
    custom_risks_file: Path = typer.Option(None, help="YAML of custom risks."),
    attacker_model: str = typer.Option("openai/gpt-4o-mini"),
    judge_model: str = typer.Option("openai/gpt-4o-mini"),
    target_model: str = typer.Option("openai/gpt-4o-mini"),
    api_base: str = typer.Option(None, help="api_base for all roles (e.g. Ollama)."),
    max_concurrency: int = typer.Option(4),
    out: Path = typer.Option(None, help="Write report JSON here."),
    markdown: bool = typer.Option(False, help="Print a Markdown report to stdout."),
):
    """Run a red-team campaign against a hosted (config-only) target agent."""

    if config is not None:
        raw = yaml.safe_load(Path(config).read_text())
        campaign = CampaignConfig(**raw)
        custom = None
    else:
        prompt = system_prompt
        if system_prompt_file is not None:
            prompt = Path(system_prompt_file).read_text()
        if not prompt:
            typer.echo("Provide --system-prompt or --system-prompt-file", err=True)
            raise typer.Exit(2)
        campaign = CampaignConfig(
            agent=AgentSpec(name=name, system_prompt=prompt, domain=domain),
            risk_ids=[x for x in risk_ids.split(",") if x.strip()],
            strategy_ids=[x for x in strategy_ids.split(",") if x.strip()],
            attacks_per_pair=attacks_per_pair,
            attacker_lm=LMConfig(model=attacker_model, api_base=api_base),
            judge_lm=LMConfig(model=judge_model, api_base=api_base, temperature=0.0),
            target_lm=LMConfig(model=target_model, api_base=api_base),
            max_concurrency=max_concurrency,
        )
        custom = load_custom_risks(custom_risks_file) if custom_risks_file else None

    logger.info("starting campaign against {}", campaign.agent.name)
    report = run_campaign(campaign, custom_risks=custom, progress=_progress)

    # Persist to the run store so `greycloak runs` / `show` can find it later.
    run_id = uuid.uuid4().hex[:12]
    provenance = build_provenance(campaign)
    default_store().save(RunRecord(
        id=run_id, config=campaign, report=report,
        provenance=provenance, status=RunStatus.COMPLETED
    ))

    typer.echo(
        f"\nASR: {report.attack_success_rate:.0%} "
        f"({report.total_successes}/{report.total_attacks})  "
        f"mean divergence: {report.mean_divergence:.2f}  (run {run_id})"
    )
    if out is not None:
        Path(out).write_text(report.model_dump_json(indent=2))
        typer.echo(f"report written to {out}")
    if markdown:
        typer.echo("\n" + to_markdown(report, provenance=provenance))


@app.command()
def runs():
    """List persisted red-team runs (from $GREYCLOAK_RUNS_DIR, default ./runs)."""
    records = default_store().list_records()
    if not records:
        typer.echo("no runs found")
        return
    for r in records:
        asr = f"{r.report.attack_success_rate:.0%}" if r.report else "-"
        typer.echo(f"  {r.id}  {r.status.value:<10} {r.config.agent.name:<24} ASR={asr}")


@app.command()
def show(run_id: str, markdown: bool = typer.Option(True, help="Render markdown.")):
    """Show a persisted run's report."""
    rec = default_store().load(run_id)
    if rec is None:
        typer.echo(f"run {run_id} not found", err=True)
        raise typer.Exit(1)
    if rec.report is None:
        typer.echo(f"run {run_id} status={rec.status.value}, no report")
        return
    typer.echo(
        to_markdown(rec.report, provenance=rec.provenance)
        if markdown else rec.report.model_dump_json(indent=2)
    )


@app.command(name="eval-judge")
def eval_judge_cmd(
    judge_model: str = typer.Option("openai/gpt-4o-mini", "--judge-model"),
    votes: int = typer.Option(1, "--votes"),
    labels: Path = typer.Option(None, "--labels", help="YAML labels; default = packaged seed."),
    probe_bias: bool = typer.Option(False, "--probe-bias"),
):
    """Validate the divergence judge against a human-labeled set."""
    config.load_env()
    cases = load_judge_labels(labels)
    judge = DivergenceJudge(votes=votes)
    judge_lm = config.build_lm(LMConfig(model=judge_model, temperature=0.0))
    ev = evaluate_judge(cases, judge, judge_lm=judge_lm, probe_bias=probe_bias)
    typer.echo(
        f"n={ev.n}  accuracy={ev.accuracy:.2f}  precision={ev.precision:.2f}  "
        f"recall={ev.recall:.2f}\ncohen_kappa={ev.cohen_kappa}  "
        f"score_correlation={ev.score_correlation}  mean_confidence={ev.mean_confidence}")
    if ev.bias:
        typer.echo(f"bias: {ev.bias}")
    if ev.disagreements:
        typer.echo(f"{len(ev.disagreements)} disagreement(s): "
                   + ", ".join(d["id"] for d in ev.disagreements))


@app.command()
def optimize(
    config: Path = typer.Option(..., help="YAML campaign config."),
    method: str = typer.Option("bootstrap", help="bootstrap | mipro | gepa"),
    train_frac: float = typer.Option(0.6),
    seeds: int = typer.Option(1),
    save_id: str = typer.Option(None, "--out-attacker", help="Save compiled attacker under this id."),
):
    """Compile a stronger attacker and report the ASR lift (measured by the independent judge)."""
    raw = yaml.safe_load(Path(config).read_text())
    campaign = CampaignConfig(**raw)
    res = run_optimization(campaign, method=method, train_frac=train_frac,
                           seeds=seeds, save_id=save_id, progress=_progress)
    typer.echo(f"method={res.method} seeds={res.seeds} train_frac={res.train_frac}")
    for name in ("direct", "baseline", "compiled"):
        a = res.arms[name]
        typer.echo(f"  {name:<9} ASR={a['asr_mean']:.0%} +/- {a['asr_stdev']:.0%}  "
                   f"mean_div={a['div_mean']:.2f}")
    typer.echo(f"  J_opt-J_rep gap (reward-gaming): {res.opt_rep_gap:+.0%}")
    if not res.held_out:
        typer.echo("  WARNING: eval not held out (too few pairs) — ASR is on training pairs")
    if not res.report_judge_independent:
        typer.echo(f"  WARNING: report judge == optimization judge ({res.report_judge_model}); "
                   "ASR is Goodhart-subject — set report_judge_lm to a different model")
    if res.attacker_id:
        typer.echo(f"  saved compiled attacker: {res.attacker_id}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Launch the FastAPI service + web dashboard."""
    import uvicorn

    uvicorn.run("greycloak.service:app", host=host, port=port, reload=False)


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
