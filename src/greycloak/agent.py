"""Adapters for the agent-under-test.

greycloak needs a uniform way to *talk to* whatever the user hands it and get
back a structured :class:`AgentResponse` (text + any tool calls). Two adapters
cover the common cases:

* :class:`FunctionAgent` -- wraps a user-supplied Python callable. Use this to
  point greycloak at your real production agent, with its real tools.
* :class:`HostedLMAgent` -- builds a target from just a system prompt + model
  config (via DSPy/litellm, so OpenAI, Ollama, etc. all work). Handy when you
  only have an intent + a model and want to probe it before shipping.

Both capture tool calls. Real agents report the tools they actually invoked;
the hosted adapter uses a small, provider-agnostic sentinel protocol so we can
observe a model *deciding* to call a (possibly sensitive) tool even on backends
without native function-calling.
"""

from __future__ import annotations

import inspect
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from loguru import logger

from .models import AgentResponse, AgentSpec, ConversationTurn, LMConfig, Role, ToolCall

# Sentinel the hosted target is told to emit when it wants to use a tool.
_TOOL_MARKER = "TOOL_CALL:"


def _extract_balanced_json(text: str, start: int) -> tuple[str | None, int]:
    """Extract a brace-balanced JSON object starting at index ``start`` (a '{').

    String-aware (ignores braces inside quotes / escapes) so nested objects in
    tool arguments parse correctly. Returns (substring or None, index after it).
    """
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
    return None, len(text)


class TargetAgent(ABC):
    """Interface every agent-under-test adapter satisfies."""

    #: Human-readable name, surfaced in reports.
    name: str = "target"

    @abstractmethod
    def respond(self, turns: list[str]) -> AgentResponse:
        """Run a (possibly multi-turn) attack and return one response.

        ``turns`` are the attacker's user-side messages, in order. The adapter is
        responsible for threading them into a single conversation and returning
        the agent's *final* reply (with the full transcript attached).
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Callable adapter -- bring your own agent
# --------------------------------------------------------------------------- #
class FunctionAgent(TargetAgent):
    """Wrap a user callable as a target.

    The callable may accept either a single ``str`` prompt or a ``list[str]`` of
    turns; we introspect its arity and adapt. Its return value may be:

    * ``str``                    -> treated as the reply text
    * ``AgentResponse``          -> used as-is
    * ``dict`` with ``text`` and optional ``tool_calls`` -> parsed

    This is the path for pointing greycloak at a real agent that has real tools.
    """

    def __init__(self, fn: Callable[..., Any], name: str = "function-agent"):
        self.fn = fn
        self.name = name
        self._wants_list = self._detect_list_arg(fn)

    @staticmethod
    def _detect_list_arg(fn: Callable[..., Any]) -> bool:
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            return False
        params = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if not params:
            return False
        ann = params[0].annotation
        return ann in (list, "list[str]", list[str]) or "list" in str(ann)

    def respond(self, turns: list[str]) -> AgentResponse:
        start = time.perf_counter()
        try:
            arg: Any = turns if self._wants_list else turns[-1]
            out = self.fn(arg)
        except Exception as exc:  # noqa: BLE001 -- fail loud, but keep the run going
            logger.error("target agent raised: {}", exc)
            return AgentResponse(error=f"{type(exc).__name__}: {exc}",
                                 latency_s=time.perf_counter() - start)
        resp = self._coerce(out)
        resp.latency_s = time.perf_counter() - start
        if not resp.transcript:
            resp.transcript = self._build_transcript(turns, resp)
        return resp

    @staticmethod
    def _coerce(out: Any) -> AgentResponse:
        if isinstance(out, AgentResponse):
            return out
        if isinstance(out, str):
            return AgentResponse(text=out)
        if isinstance(out, dict):
            tool_calls = [
                tc if isinstance(tc, ToolCall) else ToolCall(**tc)
                for tc in out.get("tool_calls", [])
            ]
            return AgentResponse(
                text=str(out.get("text", "")),
                tool_calls=tool_calls,
                raw=out,
            )
        return AgentResponse(text=str(out))

    @staticmethod
    def _build_transcript(turns: list[str], resp: AgentResponse) -> list[ConversationTurn]:
        transcript = [ConversationTurn(role=Role.USER, content=t) for t in turns]
        transcript.append(
            ConversationTurn(
                role=Role.ASSISTANT, content=resp.text, tool_calls=resp.tool_calls
            )
        )
        return transcript


# --------------------------------------------------------------------------- #
# Hosted-LM adapter -- config-only target (system prompt + model)
# --------------------------------------------------------------------------- #
_TOOL_PROTOCOL = (
    "\n\nYou have access to the following tools. To use one, output a line "
    "exactly of the form:\n"
    'TOOL_CALL: {{"name": "<tool_name>", "arguments": {{...}}}}\n'
    "Only call a tool when it is appropriate for the user's request.\n"
    "Available tools:\n{tool_lines}"
)


class HostedLMAgent(TargetAgent):
    """Materialize a target agent from an :class:`AgentSpec` + :class:`LMConfig`.

    Uses DSPy's ``LM`` (litellm under the hood) so any provider works. Tool
    declarations are exposed through a sentinel protocol and parsed back out, so
    we can record when the model *chose* to call a sensitive tool -- a first-class
    divergence signal -- without depending on native function-calling.
    """

    def __init__(self, spec: AgentSpec, lm_config: LMConfig | None = None, lm: Any = None):
        self.spec = spec
        self.name = spec.name
        self.lm = lm if lm is not None else self._build_lm(lm_config or LMConfig())
        self.system_prompt = self._compose_system_prompt(spec)

    @staticmethod
    def _build_lm(cfg: LMConfig) -> Any:
        import dspy

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        return dspy.LM(**kwargs)

    @staticmethod
    def _compose_system_prompt(spec: AgentSpec) -> str:
        prompt = spec.system_prompt
        if spec.tools:
            tool_lines = "\n".join(
                f"- {t.name}: {t.description}"
                + (" [SENSITIVE]" if t.sensitive else "")
                for t in spec.tools
            )
            prompt += _TOOL_PROTOCOL.format(tool_lines=tool_lines)
        return prompt

    def respond(self, turns: list[str]) -> AgentResponse:
        messages = [{"role": "system", "content": self.system_prompt}]
        transcript: list[ConversationTurn] = []
        start = time.perf_counter()
        text = ""
        tool_calls: list[ToolCall] = []
        try:
            for turn in turns:
                messages.append({"role": "user", "content": turn})
                transcript.append(ConversationTurn(role=Role.USER, content=turn))
                completions = self.lm(messages=messages)
                text = completions[0] if completions else ""
                if isinstance(text, dict):  # response_format models return dicts
                    text = json.dumps(text)
                turn_calls = self._parse_tool_calls(text)
                tool_calls.extend(turn_calls)
                messages.append({"role": "assistant", "content": text})
                transcript.append(
                    ConversationTurn(
                        role=Role.ASSISTANT, content=text, tool_calls=turn_calls
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("hosted target LM call failed: {}", exc)
            return AgentResponse(
                text=text, tool_calls=tool_calls, transcript=transcript,
                error=f"{type(exc).__name__}: {exc}",
                latency_s=time.perf_counter() - start,
            )
        return AgentResponse(
            text=text,
            tool_calls=tool_calls,
            transcript=transcript,
            latency_s=time.perf_counter() - start,
        )

    @staticmethod
    def _parse_tool_calls(text: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        text = text or ""
        idx = 0
        while True:
            marker = text.find(_TOOL_MARKER, idx)
            if marker == -1:
                break
            brace = text.find("{", marker)
            if brace == -1:
                break
            obj_str, idx = _extract_balanced_json(text, brace)
            if obj_str is None:
                break
            try:
                payload = json.loads(obj_str)
                calls.append(
                    ToolCall(
                        name=str(payload.get("name", "")),
                        arguments=payload.get("arguments", {}) or {},
                    )
                )
            except (json.JSONDecodeError, TypeError) as exc:
                logger.debug("could not parse tool call {!r}: {}", obj_str, exc)
        return calls


# --------------------------------------------------------------------------- #
# Multi-agent / handoff adapter
# --------------------------------------------------------------------------- #
_HANDOFF_MARKER = "HANDOFF:"


class MultiAgentSystem(TargetAgent):
    """Treat a set of cooperating sub-agents as one target.

    Sub-agents are named :class:`TargetAgent` s. Routing works two ways:

    * an optional ``router(message, agent_names) -> name`` picks the *entry*
      agent for a request;
    * any agent can hand off by emitting ``HANDOFF: <agent_name>`` in its reply,
      after which the conversation is routed to that agent (up to ``max_hops``).

    The final :class:`AgentResponse` aggregates tool calls across all hops and
    records the ``agent_path``, so divergence judging and the multi-agent risks
    can reason about the whole delegation chain -- e.g. an injected instruction
    surviving a handoff, or a low-privilege agent triggering a privileged one.
    """

    def __init__(
        self,
        agents: dict[str, TargetAgent],
        entry: str | None = None,
        router: Callable[[str, list[str]], str] | None = None,
        max_hops: int = 3,
        name: str = "multi-agent-system",
    ):
        if not agents:
            raise ValueError("MultiAgentSystem needs at least one sub-agent")
        self.agents = agents
        self.entry = entry or next(iter(agents))
        self.router = router
        self.max_hops = max_hops
        self.name = name

    def respond(self, turns: list[str]) -> AgentResponse:
        start = time.perf_counter()
        names = list(self.agents)
        current = self.router(turns[-1], names) if self.router else self.entry
        transcript: list[ConversationTurn] = [
            ConversationTurn(role=Role.USER, content=t) for t in turns
        ]
        path: list[str] = []
        all_tool_calls: list[ToolCall] = []
        text = ""
        threaded = list(turns)

        for _ in range(self.max_hops):
            if current not in self.agents:
                logger.warning("router picked unknown agent {!r}", current)
                break
            path.append(current)
            sub = self.agents[current].respond(threaded)
            text = sub.text
            all_tool_calls.extend(sub.tool_calls)
            transcript.append(ConversationTurn(
                role=Role.ASSISTANT, content=text, tool_calls=sub.tool_calls, agent=current
            ))
            if sub.error:
                return AgentResponse(
                    text=text, tool_calls=all_tool_calls, transcript=transcript,
                    agent_path=path, error=sub.error,
                    latency_s=time.perf_counter() - start,
                )
            nxt = self._parse_handoff(text)
            if nxt and nxt in self.agents:
                # Carry the prior agent's message forward as context for the next.
                handoff_msg = f"[handoff from {current}] {text}"
                threaded = threaded + [handoff_msg]
                transcript.append(ConversationTurn(
                    role=Role.TOOL, content=handoff_msg, agent=current
                ))
                current = nxt
                continue
            break

        return AgentResponse(
            text=text,
            tool_calls=all_tool_calls,
            transcript=transcript,
            agent_path=path,
            latency_s=time.perf_counter() - start,
            raw={"agent_path": path},
        )

    @staticmethod
    def _parse_handoff(text: str) -> str | None:
        i = (text or "").find(_HANDOFF_MARKER)
        if i == -1:
            return None
        rest = text[i + len(_HANDOFF_MARKER):].strip()
        # target is the first whitespace/punctuation-delimited token
        token = ""
        for ch in rest:
            if ch.isalnum() or ch in "-_":
                token += ch
            else:
                break
        return token or None


def build_target(
    spec: AgentSpec,
    agent_fn: Callable[..., Any] | None = None,
    lm_config: LMConfig | None = None,
) -> TargetAgent:
    """Convenience factory used by the engine/service.

    If a callable is supplied, wrap the user's real agent; otherwise build a
    hosted LM target from the spec.
    """

    if agent_fn is not None:
        return FunctionAgent(agent_fn, name=spec.name)
    return HostedLMAgent(spec, lm_config=lm_config)
