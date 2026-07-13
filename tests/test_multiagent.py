"""Multi-agent system adapter: routing, handoffs, aggregation."""

from greycloak import FunctionAgent, MultiAgentSystem
from greycloak.models import AgentResponse, ToolCall


def test_single_agent_no_handoff():
    mas = MultiAgentSystem({"triage": FunctionAgent(lambda p: "handled")}, entry="triage")
    resp = mas.respond(["hello"])
    assert resp.text == "handled"
    assert resp.agent_path == ["triage"]


def test_handoff_routes_to_next_agent():
    agents = {
        "front": FunctionAgent(lambda p: "let me transfer you. HANDOFF: billing"),
        "billing": FunctionAgent(lambda p: "here is your invoice"),
    }
    mas = MultiAgentSystem(agents, entry="front")
    resp = mas.respond(["I have a billing question"])
    assert resp.agent_path == ["front", "billing"]
    assert resp.text == "here is your invoice"
    # billing sub-agent saw the handoff context
    assert any("handoff from front" in t.content for t in resp.transcript)


def test_handoff_respects_max_hops():
    # two agents that bounce to each other forever
    agents = {
        "a": FunctionAgent(lambda p: "HANDOFF: b"),
        "b": FunctionAgent(lambda p: "HANDOFF: a"),
    }
    mas = MultiAgentSystem(agents, entry="a", max_hops=3)
    resp = mas.respond(["go"])
    assert len(resp.agent_path) == 3  # capped


def test_router_picks_entry():
    def router(msg, names):
        return "billing" if "invoice" in msg else "front"

    agents = {
        "front": FunctionAgent(lambda p: "front says hi"),
        "billing": FunctionAgent(lambda p: "billing says hi"),
    }
    mas = MultiAgentSystem(agents, router=router)
    assert mas.respond(["where is my invoice"]).agent_path == ["billing"]
    assert mas.respond(["hello"]).agent_path == ["front"]


def test_tool_calls_aggregated_across_hops():
    agents = {
        "front": FunctionAgent(lambda p: {"text": "HANDOFF: ops", "tool_calls": [{"name": "lookup"}]}),
        "ops": FunctionAgent(lambda p: {"text": "done", "tool_calls": [{"name": "restart_server"}]}),
    }
    mas = MultiAgentSystem(agents, entry="front")
    resp = mas.respond(["fix it"])
    names = [tc.name for tc in resp.tool_calls]
    assert names == ["lookup", "restart_server"]


def test_error_in_subagent_stops_chain():
    def boom(p):
        raise RuntimeError("subagent down")

    agents = {"front": FunctionAgent(boom)}
    resp = MultiAgentSystem(agents, entry="front").respond(["x"])
    assert resp.error is not None
