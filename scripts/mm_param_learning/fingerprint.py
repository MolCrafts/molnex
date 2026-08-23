"""Immutable run identity for Class-I look-sees. Checklist §0.

A launcher that cannot write this blob, or that finds a production
``decision: GO`` with a mismatched hash, must exit nonzero. Quasi-production
jobs write the blob and refuse to start when the hash cannot be formed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.mm_param_learning.constants import (
    COLLECTION_COORDINATE,
    TRAIN_COORDINATE,
    VAL_COORDINATE,
)

# Inclusive atom-count buckets. Same as scripts.mm_param_learning.align.
SIZE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("small", 1, 8),
    ("medium", 9, 16),
    ("large", 17, 32),
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Operating point for the quasi-production 8:2 one-epoch look-see.
# Powers of two only. Not a production multi-epoch pin.
QUASI_PROD: dict[str, Any] = {
    "kind": "quasi-production-epoch1",
    "collection": COLLECTION_COORDINATE,
    "train_coordinate": TRAIN_COORDINATE,
    "val_coordinate": VAL_COORDINATE,
    "omol_role": "val-neutral",
    "train_frac": 0.8,
    "seed": 2666,
    "hidden": 64,
    "n_blocks": 4,
    "n_torsion_terms": 8,
    "val_split": "omol25-small-molecule-8-2",
    "packed_max_atoms": 16384,
    "lr": 1e-4,
    "clip_max_norm": 1.0e3,
    "epochs": 1,
    "energy_scope": "full",
    "size_buckets": [[name, lo, hi] for name, lo, hi in SIZE_BUCKETS],
    "target_hardware": "GH200",
    "slurm_account": "naiss2026-1-53-gpu",
    "slurm_partition": "gpu",
}

# Three power-of-two capacities. One-day wall-clock campaign, not unbounded GO.
PROD_MODELS: dict[str, dict[str, int]] = {
    "small": {"hidden": 32, "n_blocks": 2, "n_torsion_terms": 4},
    "medium": {"hidden": 64, "n_blocks": 4, "n_torsion_terms": 8},
    "large": {"hidden": 128, "n_blocks": 8, "n_torsion_terms": 16},
}

PROD_DAY: dict[str, Any] = {
    **QUASI_PROD,
    "kind": "prod-day1",
    "omol_role": "4m",
    "val_role": "val-neutral",
    "val_split": "official-4m-train-val-neutral",
    "train_coordinate": "dataset:fairchem/omol25@1",
    "val_coordinate": "dataset:fairchem/omol25@1",
    "epochs": 1_000_000,
    "eval_every_epochs": 1,
    "wall_hours": 24,
    "slurm_account": "naiss2026-4-715-gpu",
}


def require_power_of_two(name: str, value: int) -> None:
    """Refuse a model/pack size that is not a power of two."""
    if int(value) < 1 or int(value) & (int(value) - 1):
        raise ValueError(f"{name} must be a power of two, got {value}")


def git_identity(root: Path = REPO_ROOT) -> dict[str, Any]:
    """HEAD + dirty-tree hash. Missing git is a hard error."""
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    dirty = bool(porcelain.strip())
    return {
        "code_revision": rev,
        "dirty": dirty,
        "dirty_hash": hashlib.sha256(porcelain.encode()).hexdigest()[:16],
    }


def fingerprint(config: dict[str, Any] | None = None, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    """One immutable blob: git identity + sorted config."""
    cfg = dict(QUASI_PROD if config is None else config)
    for key in ("hidden", "n_blocks", "n_torsion_terms", "packed_max_atoms"):
        if key in cfg:
            require_power_of_two(key, int(cfg[key]))
    ident = git_identity(root)
    payload = {**ident, "config": cfg}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    payload["fingerprint"] = digest
    return payload


def write_fingerprint(path: Path, blob: dict[str, Any]) -> Path:
    """Persist the fingerprint. Parent dirs are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    return path


def require_fingerprint_file(path: Path) -> dict[str, Any]:
    """Fail closed if the fingerprint file is missing or unreadable."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"fingerprint missing: {path} (launcher fail-closed)")
    blob = json.loads(path.read_text())
    if not blob.get("fingerprint"):
        raise ValueError(f"fingerprint file has no hash: {path}")
    return blob


def require_not_production_go(signoff: Path) -> None:
    """Refuse a multi-epoch launch if someone stamped GO without a review."""
    path = Path(signoff)
    if not path.is_file():
        return
    text = path.read_text()
    if "decision: GO" in text and "quasi-production" not in text:
        raise RuntimeError(f"{path} says decision: GO; this launcher is not a production gate")
