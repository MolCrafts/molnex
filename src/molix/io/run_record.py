"""Minimal MolRec Run-shaped package helpers (Zarr V3 group attributes).

The record root *is* a Zarr V3 hierarchy, written with plain JSON files so
tools can discover a training output without a Zarr library::

    <record_root>/
      zarr.json                # the root group
      meta/zarr.json           # group whose attributes are the meta document
      status/zarr.json         # group whose attributes are the status document
      metrics/…                # via :class:`molix.io.metrics.MetricsWriter`

Document sections are **group attributes** per molrec ``docs/spec/storage.md``
(a pure-JSON filesystem package is not part of the reference binding).

Provisional: layout follows molrec ``docs/spec/run.md``; I/O may later move
to molpy alongside the metrics JSONL binding.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Sole version key of a MolRec record (`meta["molrec_version"]`); identity is
# this key plus the record path. There is no brand key.
MOLREC_VERSION = 1

META_DIR = "meta"
STATUS_DIR = "status"
GROUP_FILENAME = "zarr.json"

JSONObject = dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: JSONObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_group(path: Path, attributes: JSONObject | None = None) -> None:
    payload: JSONObject = {"zarr_format": 3, "node_type": "group"}
    if attributes:
        payload["attributes"] = attributes
    _atomic_write_json(path, payload)


def _read_group_attrs(path: Path) -> JSONObject:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    attrs = loaded.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _ensure_root_group(record_root: Path) -> None:
    root_marker = record_root / GROUP_FILENAME
    if not root_marker.exists():
        _write_group(root_marker)


def meta_path(record_root: Path) -> Path:
    return Path(record_root) / META_DIR / GROUP_FILENAME


def status_path(record_root: Path) -> Path:
    return Path(record_root) / STATUS_DIR / GROUP_FILENAME


def read_meta(record_root: Path | str) -> JSONObject:
    """The ``meta`` document: attributes of the ``meta/`` group."""
    return _read_group_attrs(meta_path(Path(record_root)))


def read_status(record_root: Path | str) -> JSONObject:
    """The ``status`` document: attributes of the ``status/`` group."""
    return _read_group_attrs(status_path(Path(record_root)))


def write_meta(
    record_root: Path | str,
    *,
    creator_name: str = "molnex",
    creator_version: str | None = None,
    extra: JSONObject | None = None,
) -> Path:
    """Write / overwrite the ``meta/`` group attributes with the required keys."""
    root = Path(record_root)
    payload: JSONObject = {
        "molrec_version": MOLREC_VERSION,
        "creator": {
            "name": creator_name,
            **({"version": creator_version} if creator_version else {}),
        },
        "created_at": _utc_now(),
    }
    if extra:
        payload.update(extra)
    _ensure_root_group(root)
    path = meta_path(root)
    _write_group(path, payload)
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
    """Write / overwrite the ``status/`` group attributes (``state`` required)."""
    if not state or not str(state).strip():
        raise ValueError("status.state must be a non-empty string")
    root = Path(record_root)
    now = _utc_now()
    path = status_path(root)
    existing = _read_group_attrs(path)

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
    _ensure_root_group(root)
    _write_group(path, payload)
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
    "GROUP_FILENAME",
    "META_DIR",
    "MOLREC_VERSION",
    "STATUS_DIR",
    "ensure_run_package",
    "meta_path",
    "read_meta",
    "read_status",
    "status_path",
    "write_meta",
    "write_status",
]
