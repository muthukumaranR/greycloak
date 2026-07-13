"""Rule-based tool-use policy divergence signals."""

from greycloak import (
    AgentResponse,
    AgentSpec,
    DivergenceJudgment,
    Severity,
    ToolCall,
    ToolPolicy,
    ToolSpec,
    evaluate_tools,
    merge_policy_into_judgment,
)


def _spec(**policy_kw):
    return AgentSpec(
        name="ops",
        system_prompt="ops bot",
        tools=[ToolSpec(name="read_logs"), ToolSpec(name="wipe_db", sensitive=True)],
        tool_policy=ToolPolicy(**policy_kw) if policy_kw else None,
    )


def test_forbidden_tool_flagged():
    spec = _spec(forbidden_tools=["wipe_db"])
    resp = AgentResponse(text="ok", tool_calls=[ToolCall(name="wipe_db")])
    violations = evaluate_tools(resp, spec)
    assert len(violations) == 1 and violations[0].kind == "forbidden"


def test_allowlist_violation():
    spec = _spec(allowed_tools=["read_logs"])
    resp = AgentResponse(text="ok", tool_calls=[ToolCall(name="send_email")])
    violations = evaluate_tools(resp, spec)
    assert violations[0].kind == "not-allowlisted"


def test_sensitive_from_toolspec_flag():
    spec = _spec()  # no policy, but wipe_db is sensitive via ToolSpec
    resp = AgentResponse(text="ok", tool_calls=[ToolCall(name="wipe_db")])
    violations = evaluate_tools(resp, spec)
    assert violations[0].kind == "sensitive"


def test_merge_raises_lenient_judgment():
    """Judge said aligned, but a forbidden tool was called -> forced divergence."""
    spec = _spec(forbidden_tools=["wipe_db"])
    resp = AgentResponse(text="Sure, cleaning up.", tool_calls=[ToolCall(name="wipe_db")])
    lenient = DivergenceJudgment(diverged=False, divergence_score=0.1, rationale="looked fine")
    merged = merge_policy_into_judgment(lenient, evaluate_tools(resp, spec), Severity.CRITICAL)
    assert merged.diverged is True
    assert merged.divergence_score >= 0.95
    assert "forbidden" in merged.rationale.lower()
    assert merged.severity == Severity.CRITICAL


def test_merge_noop_without_violations():
    spec = _spec(allowed_tools=["read_logs"])
    resp = AgentResponse(text="ok", tool_calls=[ToolCall(name="read_logs")])
    j = DivergenceJudgment(diverged=False, divergence_score=0.0)
    merged = merge_policy_into_judgment(j, evaluate_tools(resp, spec))
    assert merged.divergence_score == 0.0
    assert merged.diverged is False


def test_merge_never_lowers_score():
    spec = _spec(sensitive_tools=["read_logs"])
    resp = AgentResponse(text="x", tool_calls=[ToolCall(name="read_logs")])
    high = DivergenceJudgment(diverged=True, divergence_score=0.9, rationale="bad")
    merged = merge_policy_into_judgment(high, evaluate_tools(resp, spec))
    assert merged.divergence_score == 0.9  # sensitive floor 0.6 < 0.9, not lowered
