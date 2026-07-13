from typer.testing import CliRunner
from greycloak import cli
from greycloak.models import IntentProfile, RedTeamReport
from greycloak.store import default_store


def test_cli_run_persists_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv("GREYCLOAK_RUNS_DIR", str(tmp_path))
    fake = RedTeamReport(agent_name="a", intent=IntentProfile(purpose="p"))
    monkeypatch.setattr(cli, "run_campaign", lambda *a, **k: fake)
    result = CliRunner().invoke(
        cli.app, ["run", "--name", "a", "--system-prompt", "p", "--markdown"])
    assert result.exit_code == 0, result.output
    recs = default_store().list_records()
    assert recs, "no run persisted"
    assert recs[0].provenance is not None
    assert recs[0].provenance.config_hash
