"""Agent adapter tests: callable wrapping + hosted tool-call capture."""

from greycloak import AgentResponse, FunctionAgent, HostedLMAgent, ToolCall
from greycloak.models import AgentSpec, ToolSpec


def test_function_agent_str_return():
    agent = FunctionAgent(lambda p: f"echo: {p}")
    resp = agent.respond(["hello"])
    assert resp.text == "echo: hello"
    assert resp.latency_s is not None
    assert resp.transcript[-1].content == "echo: hello"


def test_function_agent_list_arg_detected():
    def multi(turns: list[str]) -> str:
        return f"saw {len(turns)} turns"

    agent = FunctionAgent(multi)
    assert agent.respond(["a", "b", "c"]).text == "saw 3 turns"


def test_function_agent_dict_with_tool_calls():
    def tool_agent(p):
        return {"text": "done", "tool_calls": [{"name": "delete_all", "arguments": {"x": 1}}]}

    resp = FunctionAgent(tool_agent).respond(["do it"])
    assert resp.text == "done"
    assert resp.tool_calls == [ToolCall(name="delete_all", arguments={"x": 1})]


def test_function_agent_captures_exceptions():
    def boom(p):
        raise RuntimeError("kaboom")

    resp = FunctionAgent(boom).respond(["x"])
    assert resp.error is not None and "kaboom" in resp.error


def test_function_agent_passthrough_response():
    def native(p):
        return AgentResponse(text="native", tool_calls=[ToolCall(name="t")])

    resp = FunctionAgent(native).respond(["x"])
    assert resp.text == "native"


class _FakeLM:
    """Minimal stand-in for dspy.LM: returns canned completion strings."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def __call__(self, messages=None, **kw):
        self.calls.append(messages)
        return [self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]]


def test_hosted_agent_parses_tool_calls():
    spec = AgentSpec(
        name="ops",
        system_prompt="You are an ops bot.",
        tools=[ToolSpec(name="shutdown", description="stop a server", sensitive=True)],
    )
    lm = _FakeLM(['Sure. TOOL_CALL: {"name": "shutdown", "arguments": {"server": "db1"}}'])
    agent = HostedLMAgent(spec, lm=lm)
    resp = agent.respond(["please shut down db1"])
    assert resp.tool_calls == [ToolCall(name="shutdown", arguments={"server": "db1"})]
    assert "SENSITIVE" in agent.system_prompt  # tool protocol injected


def test_hosted_agent_multi_turn_threads_conversation():
    spec = AgentSpec(name="bot", system_prompt="be helpful")
    lm = _FakeLM(["first reply", "second reply"])
    agent = HostedLMAgent(spec, lm=lm)
    resp = agent.respond(["turn one", "turn two"])
    assert resp.text == "second reply"
    # two user turns -> two LM calls, transcript has both user + assistant turns
    assert len(lm.calls) == 2
    roles = [t.role.value for t in resp.transcript]
    assert roles == ["user", "assistant", "user", "assistant"]
