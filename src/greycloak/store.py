"""File-based persistence for campaign runs.

Runs are written to ``<root>/<id>.json`` as serialized :class:`RunRecord` s, so
they survive process restarts and can be listed/reloaded by the CLI and service.
Deliberately simple (one JSON file per run) -- the right amount of durability for
a single-node red-teaming tool, with no database to operate.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from loguru import logger

from .models import RunRecord


class RunStore:
    """Persist and retrieve :class:`RunRecord` s under a directory."""

    def __init__(self, root: str | Path = "runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def save(self, record: RunRecord) -> None:
        """Atomically write a record (temp file + rename) to avoid torn reads."""
        path = self._path(record.id)
        data = record.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        logger.debug("persisted run {} ({})", record.id, record.status.value)

    def load(self, run_id: str) -> RunRecord | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        try:
            return RunRecord.model_validate_json(path.read_text())
        except Exception as exc:  # noqa: BLE001 -- corrupt file shouldn't crash listing
            logger.error("could not load run {}: {}", run_id, exc)
            return None

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def list_records(self) -> list[RunRecord]:
        records = [r for rid in self.list_ids() if (r := self.load(rid)) is not None]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    def delete(self, run_id: str) -> bool:
        path = self._path(run_id)
        if path.exists():
            path.unlink()
            return True
        return False


def default_store() -> RunStore:
    """Store rooted at ``$GREYCLOAK_RUNS_DIR`` (default ``./runs``)."""
    return RunStore(os.getenv("GREYCLOAK_RUNS_DIR", "runs"))
