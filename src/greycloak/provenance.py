"""Capture reproducibility metadata for a run."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version

import dspy

from .models import CampaignConfig, RunProvenance


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def build_provenance(config: CampaignConfig) -> RunProvenance:
    config_hash = hashlib.sha256(
        config.model_dump_json(exclude_none=False).encode()).hexdigest()[:16]
    return RunProvenance(
        attacker_model=config.attacker_lm.model,
        judge_model=config.judge_lm.model,
        report_judge_model=(config.report_judge_lm or config.judge_lm).model,
        target_model=config.target_lm.model if config.target_lm else "",
        seed=config.seed,
        config_hash=config_hash,
        greycloak_version=_pkg("greycloak"),
        dspy_version=getattr(dspy, "__version__", "") or _pkg("dspy"),
    )
