"""LM configuration helpers.

greycloak keeps three LM roles independent:

* **attacker** -- generates adversarial probes. Point this at an unrestricted
  model (e.g. a local Ollama model) so red-team generation isn't itself refused.
* **judge**    -- scores divergence. Point this at a strong, well-aligned model.
* **target**   -- the agent under test (only used for hosted/config-only targets).

Each maps to a ``dspy.LM`` (litellm under the hood), so OpenAI, Ollama, Azure,
Anthropic, etc. are all reachable by model string.
"""

from __future__ import annotations

import os
from typing import Any

import dspy
from dotenv import load_dotenv
from loguru import logger

from .models import LMConfig

_ENV_LOADED = False


def load_env() -> None:
    """Load a local ``.env`` once (for API keys). Safe to call repeatedly."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True


def build_lm(cfg: LMConfig) -> dspy.LM:
    """Construct a ``dspy.LM`` from an :class:`LMConfig`.

    API keys resolve from the config first, then the environment (``.env``).
    Ollama is reached by using an ``ollama/<model>`` model string plus an
    ``api_base`` such as ``http://localhost:11434``.
    """

    load_env()
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.api_base:
        kwargs["api_base"] = cfg.api_base
    api_key = cfg.api_key or _default_key_for(cfg.model)
    if api_key:
        kwargs["api_key"] = api_key
    logger.debug("building LM model={} api_base={}", cfg.model, cfg.api_base)
    return dspy.LM(**kwargs)


def _default_key_for(model: str) -> str | None:
    """Best-effort default API key lookup from the environment by provider."""
    m = model.lower()
    if m.startswith("ollama") or "localhost" in m:
        return "ollama"  # litellm wants a non-empty key for local servers
    if m.startswith("anthropic") or "claude" in m:
        return os.getenv("ANTHROPIC_API_KEY")
    # default: OpenAI-compatible
    return os.getenv("OPENAI_API_KEY")


# Recommended defaults for the two "brain" roles.
def default_attacker_lm() -> LMConfig:
    return LMConfig(model="openai/gpt-4o-mini", temperature=0.9)


def default_judge_lm() -> LMConfig:
    return LMConfig(model="openai/gpt-4o-mini", temperature=0.0)
