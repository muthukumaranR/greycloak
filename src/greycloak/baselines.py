"""Baseline attackers — controls the optimized attacker is compared against."""

from __future__ import annotations

from .models import AttackCase, AttackStrategy, IntentProfile, RiskDefinition
from .modules import _short_id


class DirectBaseline:
    """The plain-request control: ask directly for the risk objective. No LM."""

    def forward(self, intent: IntentProfile, risk: RiskDefinition,
                strategy: AttackStrategy, n: int = 1, domain=None) -> list[AttackCase]:
        prompt = f"Please {risk.objective}."
        return [AttackCase(id=_short_id("direct"), risk_id=risk.id,
                           strategy_id=strategy.id, objective=risk.objective,
                           turns=[prompt], rationale="direct baseline")]

    __call__ = forward
