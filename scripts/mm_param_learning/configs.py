"""Class-I configuration pins.

Run identity is owned by **molexp** (``Workspace → Project → Experiment →
Run``); these dicts are only the science/launch pins a launcher reads at
startup. No git/config hashing lives here anymore.
"""

from __future__ import annotations

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

# Calibration gates run on the SAME official 4m/val-neutral coordinates as
# production. Only duration and scheduler/output identity differ, both
# enumerated here.
CALIBRATION: dict[str, Any] = {
    **PROD_DAY,
    "kind": "calibration-official-tiny",
    "epochs": 1,
    "eval_every_epochs": 1,
    "wall_hours": 1,
    "slurm_account": "naiss2026-1-53-gpu",
}


def require_power_of_two(name: str, value: int) -> None:
    """Refuse a model/pack size that is not a power of two."""
    if int(value) < 1 or int(value) & (int(value) - 1):
        raise ValueError(f"{name} must be a power of two, got {value}")
