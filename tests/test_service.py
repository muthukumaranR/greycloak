"""FastAPI service tests: catalog, campaign lifecycle, persistence, SSE."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from greycloak import service
from greycloak.models import IntentProfile, RedTeamReport
from greycloak.store import RunStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Isolate the run store to tmp and reset in-memory state per test."""
    monkeypatch.setattr(service, "_store_instance", RunStore(tmp_path / "runs"))
    service._RUNS.clear()
    service._QUEUES.clear()
    return TestClient(service.app)


def _fake_run_factory():
    def fake_run(config, custom_risks=None, target=None, declared_intent=None, progress=None):
        from greycloak.engine import ProgressEvent
        if progress:
            progress(ProgressEvent("generate", "generating", 1, 2))
            progress(ProgressEvent("done", "done", 2, 2))
        return RedTeamReport(
            agent_name=config.agent.name, intent=IntentProfile(purpose="test"),
            total_attacks=2, total_successes=1, attack_success_rate=0.5,
            mean_divergence=0.4,
        )
    return fake_run


def _wait_terminal(client, run_id, tries=100):
    for _ in range(tries):
        s = client.get(f"/api/campaigns/{run_id}").json()
        if s["status"] in ("completed", "failed"):
            return s
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "greycloak" in r.text.lower()


def test_list_risks_specializes_domain(client):
    data = client.get("/api/risks", params={"domain": "aviation maintenance"}).json()
    assert "jailbreak" in {x["id"] for x in data["generic"]}
    over = next(x for x in data["domain"] if x["id"] == "overgeneralization")
    assert "aviation maintenance" in over["description"]
    assert any(r["id"] == "handoff-injection" for r in data["multi_agent"])


def test_list_strategies(client):
    r = client.get("/api/strategies").json()
    assert any(s["id"] == "escalation" and s["multi_turn"] for s in r)


def test_campaign_lifecycle_and_persistence(client, monkeypatch, tmp_path):
    monkeypatch.setattr(service, "run_campaign", _fake_run_factory())
    res = client.post("/api/campaigns", json={"config": {"agent": {
        "name": "svc-bot", "system_prompt": "be safe"}}})
    run_id = res.json()["id"]

    s = _wait_terminal(client, run_id)
    assert s["status"] == "completed"
    assert s["report"]["attack_success_rate"] == 0.5
    assert len(s["progress"]) == 2

    # persisted to disk and listable
    assert any(r["id"] == run_id for r in client.get("/api/campaigns").json())
    reloaded = service.get_store().load(run_id)
    assert reloaded is not None and reloaded.report.attack_success_rate == 0.5


def test_sse_stream_ends_with_result(client, monkeypatch):
    monkeypatch.setattr(service, "run_campaign", _fake_run_factory())
    run_id = client.post("/api/campaigns", json={"config": {"agent": {
        "name": "svc-bot", "system_prompt": "be safe"}}}).json()["id"]
    _wait_terminal(client, run_id)

    body = client.get(f"/api/campaigns/{run_id}/events").text
    events = [json.loads(line[5:]) for line in body.splitlines()
              if line.startswith("data:")]
    assert events[-1]["phase"] == "result"
    assert events[-1]["report"]["attack_success_rate"] == 0.5


def test_get_unknown_campaign_404(client):
    assert client.get("/api/campaigns/nope").status_code == 404
    assert client.get("/api/campaigns/nope/events").status_code == 404
