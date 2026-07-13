"""Metrics and reporting.

Turns a list of :class:`AttackResult` into an aggregated :class:`RedTeamReport`
with the headline numbers red-teamers actually look at: attack success rate
(ASR), mean divergence, and per-risk / per-category breakdowns. Also renders a
readable Markdown summary for the CLI and service.
"""

from __future__ import annotations

from collections import defaultdict

from .models import (
    AttackResult,
    IntentProfile,
    RedTeamReport,
    RiskCategory,
    RiskScore,
    Severity,
)

_SEV_ORDER = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}


def summarize(
    agent_name: str, intent: IntentProfile, results: list[AttackResult]
) -> RedTeamReport:
    """Aggregate raw results into a report with ASR + per-risk scores."""

    total = len(results)
    successes = sum(1 for r in results if r.succeeded)
    mean_div = (
        sum(r.judgment.divergence_score for r in results) / total if total else 0.0
    )

    by_risk: dict[str, list[AttackResult]] = defaultdict(list)
    for r in results:
        by_risk[r.risk.id].append(r)

    risk_scores: list[RiskScore] = []
    for risk_id, rs in by_risk.items():
        risk = rs[0].risk
        n = len(rs)
        succ = sum(1 for r in rs if r.succeeded)
        risk_scores.append(
            RiskScore(
                risk_id=risk_id,
                risk_name=risk.name,
                category=risk.category,
                severity=risk.severity,
                attacks=n,
                successes=succ,
                attack_success_rate=succ / n if n else 0.0,
                mean_divergence=sum(r.judgment.divergence_score for r in rs) / n
                if n
                else 0.0,
            )
        )

    # Rank: highest ASR first, breaking ties by intrinsic severity.
    risk_scores.sort(
        key=lambda s: (-s.attack_success_rate, _SEV_ORDER[s.severity], -s.mean_divergence)
    )

    return RedTeamReport(
        agent_name=agent_name,
        intent=intent,
        results=results,
        total_attacks=total,
        total_successes=successes,
        attack_success_rate=successes / total if total else 0.0,
        mean_divergence=mean_div,
        risk_scores=risk_scores,
    )


def category_breakdown(report: RedTeamReport) -> dict[RiskCategory, dict[str, float]]:
    """ASR and mean divergence grouped by risk category."""
    buckets: dict[RiskCategory, list[AttackResult]] = defaultdict(list)
    for r in report.results:
        buckets[r.risk.category].append(r)
    out: dict[RiskCategory, dict[str, float]] = {}
    for cat, rs in buckets.items():
        n = len(rs)
        succ = sum(1 for r in rs if r.succeeded)
        out[cat] = {
            "attacks": n,
            "successes": succ,
            "attack_success_rate": succ / n if n else 0.0,
            "mean_divergence": sum(r.judgment.divergence_score for r in rs) / n
            if n
            else 0.0,
        }
    return out


def to_markdown(report: RedTeamReport, max_examples: int = 5) -> str:
    """Render a human-readable Markdown report."""
    lines: list[str] = []
    lines.append(f"# Red-team report — {report.agent_name}")
    lines.append(f"_Generated {report.created_at.isoformat()}_\n")
    lines.append(f"**Intent:** {report.intent.purpose}\n")
    lines.append("## Summary")
    lines.append(f"- Attacks run: **{report.total_attacks}**")
    lines.append(
        f"- Attack success rate (divergences): "
        f"**{report.attack_success_rate:.0%}** "
        f"({report.total_successes}/{report.total_attacks})"
    )
    lines.append(f"- Mean divergence score: **{report.mean_divergence:.2f}**\n")

    cats = category_breakdown(report)
    if cats:
        lines.append("## By category")
        lines.append("| Category | Attacks | ASR | Mean divergence |")
        lines.append("|---|---|---|---|")
        for cat, stats in cats.items():
            lines.append(
                f"| {cat.value} | {stats['attacks']} | "
                f"{stats['attack_success_rate']:.0%} | {stats['mean_divergence']:.2f} |"
            )
        lines.append("")

    lines.append("## By risk (ranked)")
    lines.append("| Risk | Category | Severity | Attacks | ASR | Mean div |")
    lines.append("|---|---|---|---|---|---|")
    for s in report.risk_scores:
        lines.append(
            f"| {s.risk_name} | {s.category.value} | {s.severity.value} | "
            f"{s.attacks} | {s.attack_success_rate:.0%} | {s.mean_divergence:.2f} |"
        )
    lines.append("")

    successes = [r for r in report.results if r.succeeded]
    successes.sort(key=lambda r: -r.judgment.divergence_score)
    if successes:
        lines.append("## Top divergences")
        for r in successes[:max_examples]:
            lines.append(f"### {r.risk.name} — via {r.strategy.name}")
            lines.append(f"- **Divergence:** {r.judgment.divergence_score:.2f} "
                         f"({r.judgment.severity.value})")
            lines.append(f"- **Attacker:** {r.case.turns[-1][:300]}")
            lines.append(f"- **Agent:** {r.response.text[:300]}")
            if r.response.tool_calls:
                tc = ", ".join(t.name for t in r.response.tool_calls)
                lines.append(f"- **Tool calls:** {tc}")
            lines.append(f"- **Why:** {r.judgment.rationale}\n")
    return "\n".join(lines)
