"""Minimal MolRec Run-shaped package helpers (filesystem / hybrid).

Writes the small JSON side of a Run package so tools can discover a training
output without a Zarr reader::

    <record_root>/
      meta/meta.json
      status/status.json
      metrics/…   # via :class:`molix.io.metrics.MetricsWriter`

Provisional: layout follows molrec ``docs/spec/run.md``; I/O may later move
to molpy alongside the metrics JSONL binding.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECORD_SCHEMA_VERSION = 1
FORMAT_NAME = "molrec"

META_DIR = "meta"
META_FILENAME = "meta.json"
STATUS_DIR = "status"
STATUS_FILENAME = "status.json"

JSONObject = dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: JSONObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def meta_path(record_root: Path) -> Path:
    return Path(record_root) / META_DIR / META_FILENAME


def status_path(record_root: Path) -> Path:
    return Path(record_root) / STATUS_DIR / STATUS_FILENAME


def write_meta(
    record_root: Path | str,
    *,
    creator_name: str = "molnex",
    creator_version: str | None = None,
    extra: JSONObject | None = None,
) -> Path:
    """Write / overwrite ``meta/meta.json`` with required schema keys."""
    root = Path(record_root)
    payload: JSONObject = {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "format_name": FORMAT_NAME,
        "creator": {
            "name": creator_name,
            **({"version": creator_version} if creator_version else {}),
        },
        "created_at": _utc_now(),
    }
    if extra:
        payload.update(extra)
    path = meta_path(root)
    _atomic_write_json(path, payload)
    return path


def write_status(
    record_root: Path | str,
    *,
    state: str,
    stage: str | None = None,
    global_step: int | None = None,
    message: str | None = None,
    extra: JSONObject | None = None,
) -> Path:
    """Write / overwrite ``status/status.json`` (``state`` is required)."""
    if not state or not str(state).strip():
        raise ValueError("status.state must be a non-empty string")
    root = Path(record_root)
    now = _utc_now()
    path = status_path(root)
    existing: JSONObject = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (json.JSONDecodeError, OSError):
            existing = {}

    payload: JSONObject = {**existing}
    payload["state"] = state
    payload["updated_at"] = now
    if "started_at" not in payload and state == "running":
        payload["started_at"] = now
    if stage is not None:
        payload["stage"] = stage
    if global_step is not None:
        payload["global_step"] = global_step
    if message is not None:
        payload["message"] = message
    if state in {"succeeded", "failed", "cancelled"}:
        payload["finished_at"] = now
    if extra:
        payload.update(extra)
    _atomic_write_json(path, payload)
    return path


def ensure_run_package(
    record_root: Path | str,
    *,
    creator_name: str = "molnex",
    creator_version: str | None = None,
    state: str = "running",
    stage: str | None = "train",
) -> Path:
    """Create a minimal Run-shaped package (meta + status); metrics written separately."""
    root = Path(record_root)
    root.mkdir(parents=True, exist_ok=True)
    if not meta_path(root).exists():
        write_meta(root, creator_name=creator_name, creator_version=creator_version)
    write_status(root, state=state, stage=stage)
    return root


__all__ = [
    "FORMAT_NAME",
    "META_DIR",
    "META_FILENAME",
    "RECORD_SCHEMA_VERSION",
    "STATUS_DIR",
    "STATUS_FILENAME",
    "ensure_run_package",
    "meta_path",
    "status_path",
    "write_meta",
    "write_status",
]
