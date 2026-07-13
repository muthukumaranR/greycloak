"""RunStore persistence unit tests."""

from greycloak import CampaignConfig, RunStore
from greycloak.models import AgentSpec, IntentProfile, RedTeamReport, RunRecord, RunStatus


def _record(rid, name="bot"):
    return RunRecord(
        id=rid,
        status=RunStatus.COMPLETED,
        config=CampaignConfig(agent=AgentSpec(name=name, system_prompt="p")),
        report=RedTeamReport(agent_name=name, intent=IntentProfile(purpose="x"),
                             attack_success_rate=0.3),
    )


def test_save_and_load_roundtrip(tmp_path):
    store = RunStore(tmp_path)
    store.save(_record("abc"))
    loaded = store.load("abc")
    assert loaded is not None
    assert loaded.id == "abc"
    assert loaded.report.attack_success_rate == 0.3
    assert loaded.config.agent.name == "bot"


def test_load_missing_returns_none(tmp_path):
    assert RunStore(tmp_path).load("missing") is None


def test_list_records_sorted_newest_first(tmp_path):
    store = RunStore(tmp_path)
    store.save(_record("one", "a"))
    store.save(_record("two", "b"))
    ids = [r.id for r in store.list_records()]
    assert set(ids) == {"one", "two"}


def test_delete(tmp_path):
    store = RunStore(tmp_path)
    store.save(_record("gone"))
    assert store.delete("gone") is True
    assert store.load("gone") is None
    assert store.delete("gone") is False


def test_corrupt_file_skipped(tmp_path):
    store = RunStore(tmp_path)
    store.save(_record("good"))
    (tmp_path / "bad.json").write_text("{not valid json")
    # listing must not crash on the corrupt file
    ids = [r.id for r in store.list_records()]
    assert "good" in ids and "bad" not in ids
