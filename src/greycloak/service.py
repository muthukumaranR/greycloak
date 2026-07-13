"""FastAPI service exposing greycloak as an API + web dashboard.

Endpoints
---------
* ``GET  /``                          -> the web dashboard (static single page)
* ``GET  /api/risks?domain=...``      -> built-in generic + domain risks
* ``GET  /api/strategies``            -> built-in attack strategies
* ``POST /api/campaigns``             -> start a run (body = CampaignConfig); returns id
* ``GET  /api/campaigns``             -> list runs (in-memory + persisted)
* ``GET  /api/campaigns/{id}``        -> run status + progress + report
* ``GET  /api/campaigns/{id}/events`` -> Server-Sent Events stream of live progress

Runs execute in a background thread. Progress is available two ways: poll the
``GET`` endpoint, or subscribe to the SSE ``events`` stream. Every run is also
persisted to disk via :class:`RunStore`, so runs survive restarts.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from .engine import ProgressEvent, run_campaign
from .models import CampaignConfig, RiskDefinition, RunRecord, RunStatus
from .risks import GENERIC_RISKS, MULTI_AGENT_RISKS, domain_risks, load_custom_risks
from .store import default_store
from .strategies import DEFAULT_STRATEGIES

app = FastAPI(title="greycloak", description="Agent red-teaming service")

_FRONTEND = Path(__file__).parent / "frontend" / "index.html"
_TERMINAL = {RunStatus.COMPLETED, RunStatus.FAILED}
_DONE = object()  # sentinel put on a run's queue when it finishes

_RUNS: dict[str, RunRecord] = {}
_QUEUES: dict[str, Queue] = {}
_LOCK = threading.Lock()
_store_instance = None


def get_store():
    """Lazily create the run store (so import doesn't touch the filesystem)."""
    global _store_instance
    if _store_instance is None:
        _store_instance = default_store()
    return _store_instance


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if _FRONTEND.exists():
        return _FRONTEND.read_text()
    return "<h1>greycloak</h1><p>Frontend asset missing.</p>"


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
@app.get("/api/risks")
def list_risks(domain: str = "your domain") -> dict[str, Any]:
    return {
        "generic": [r.model_dump() for r in GENERIC_RISKS],
        "domain": [r.model_dump() for r in domain_risks(domain)],
        "multi_agent": [r.model_dump() for r in MULTI_AGENT_RISKS],
    }


@app.get("/api/strategies")
def list_strategies() -> list[dict[str, Any]]:
    return [s.model_dump() for s in DEFAULT_STRATEGIES]


# --------------------------------------------------------------------------- #
# Campaigns
# --------------------------------------------------------------------------- #
class StartCampaign(BaseModel):
    config: CampaignConfig
    custom_risks: list[dict[str, Any]] | None = None


def _get_record(run_id: str) -> RunRecord | None:
    """In-memory record if present, else load from the persistent store."""
    rec = _RUNS.get(run_id)
    return rec if rec is not None else get_store().load(run_id)


@app.post("/api/campaigns")
def start_campaign(body: StartCampaign) -> dict[str, str]:
    run_id = uuid.uuid4().hex[:12]
    record = RunRecord(id=run_id, config=body.config)
    with _LOCK:
        _RUNS[run_id] = record
        _QUEUES[run_id] = Queue()
    get_store().save(record)

    custom: list[RiskDefinition] | None = None
    if body.custom_risks:
        custom = load_custom_risks(body.custom_risks)

    threading.Thread(target=_run, args=(run_id, body.config, custom), daemon=True).start()
    logger.info("started campaign {} against {}", run_id, body.config.agent.name)
    return {"id": run_id, "status": record.status.value}


def _run(run_id: str, config: CampaignConfig, custom: list[RiskDefinition] | None) -> None:
    record = _RUNS[run_id]
    queue = _QUEUES[run_id]
    record.status = RunStatus.RUNNING

    def on_progress(ev: ProgressEvent) -> None:
        item = {
            "phase": ev.phase,
            "message": ev.message,
            "done": ev.done,
            "total": ev.total,
            "diverged": ev.result.succeeded if ev.result else None,
        }
        record.progress.append(item)
        queue.put(item)

    try:
        report = run_campaign(config, custom_risks=custom, progress=on_progress)
        record.report = report
        record.status = RunStatus.COMPLETED
    except Exception as exc:  # noqa: BLE001 -- surface failures to the UI
        logger.exception("campaign {} failed", run_id)
        record.error = f"{type(exc).__name__}: {exc}"
        record.status = RunStatus.FAILED
    finally:
        from .models import _utcnow

        record.updated_at = _utcnow()
        get_store().save(record)
        queue.put(_DONE)


@app.get("/api/campaigns")
def list_campaigns() -> list[dict[str, Any]]:
    seen: dict[str, RunRecord] = {}
    for rec in get_store().list_records():
        seen[rec.id] = rec
    with _LOCK:
        seen.update(_RUNS)  # in-memory wins (freshest)
    records = sorted(seen.values(), key=lambda r: r.created_at, reverse=True)
    return [
        {
            "id": r.id,
            "status": r.status.value,
            "agent": r.config.agent.name,
            "asr": r.report.attack_success_rate if r.report else None,
            "created_at": r.created_at.isoformat(),
            "progress": len(r.progress),
        }
        for r in records
    ]


@app.get("/api/campaigns/{run_id}")
def get_campaign(run_id: str) -> JSONResponse:
    record = _get_record(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    return JSONResponse({
        "id": record.id,
        "status": record.status.value,
        "agent": record.config.agent.name,
        "progress": record.progress,
        "error": record.error,
        "report": record.report.model_dump(mode="json") if record.report else None,
    })


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/api/campaigns/{run_id}/events")
def campaign_events(run_id: str) -> StreamingResponse:
    """Stream progress as Server-Sent Events, ending with the final report."""
    record = _get_record(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")

    def final_payload() -> dict:
        rec = _get_record(run_id)
        return {
            "phase": "result",
            "status": rec.status.value if rec else "unknown",
            "report": rec.report.model_dump(mode="json") if rec and rec.report else None,
            "error": rec.error if rec else None,
        }

    def stream() -> Iterator[str]:
        queue = _QUEUES.get(run_id)
        # Already finished (or restored from disk with no live queue): send final.
        if queue is None or record.status in _TERMINAL:
            for item in record.progress:
                yield _sse(item)
            yield _sse(final_payload())
            return
        while True:
            try:
                item = queue.get(timeout=1.0)
            except Empty:
                rec = _RUNS.get(run_id)
                if rec is not None and rec.status in _TERMINAL:
                    yield _sse(final_payload())
                    return
                yield ": keep-alive\n\n"
                continue
            if item is _DONE:
                yield _sse(final_payload())
                return
            yield _sse(item)

    return StreamingResponse(stream(), media_type="text/event-stream")
