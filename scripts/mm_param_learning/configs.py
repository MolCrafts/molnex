"""Class-I configuration pins.

Run identity is owned by **molexp** (``Workspace → Project → Experiment →
Run``); these dicts are only the science/launch pins a launcher reads at
startup. No git/config hashing lives here anymore.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts.mm_param_learning.constants import (
    COLLECTION_COORDINATE,
    TRAIN_COORDINATE,
    VAL_COORDINATE,
)

# Inclusive atom-count buckets. This is now the only statement of them:
# `align.py` held the other copy and was deleted (operator ruling, 2026-09-17).
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
    "batch_size": 256,
    "lr": 1e-4,
    "clip_max_norm": 1.0e3,
    "epochs": 1,
    "energy_scope": "full",
    "size_buckets": [[name, lo, hi] for name, lo, hi in SIZE_BUCKETS],
    "target_hardware": "GH200",
    "slurm_account": "naiss2026-1-53-gpu",
    "slurm_partition": "gpu",
}

# Three model tiers on one packed capacity. One-day wall-clock campaign, not
# unbounded GO. packed_max_atoms is the padded atom capacity per batch (ghost
# slots included) — the one static-shape knob. Every other cap follows the 4m
# density (MoleculePadder.from_counts: edges/bonds/angles/propers/impropers =
# density × atom budget × 1.10, never below the largest molecule) and the
# molecule count per batch floats (~2390 at 131072 atoms, 4m mean 54.85
# atoms/molecule). The retired scheme — a fixed molecule count padded to
# B × per-molecule max — needed 17.4 M proper rows at B=2048 and OOM'd
# (job 2439130); a random batch filled 10 % of that.
#
# The binding column is **edges**, not atoms: at the field's 10 A open-boundary
# list the 4m cutoff graph is nearly complete per molecule — 3769 edges per
# molecule on average, Sum(edges)/Sum(atoms) = 68.7 — so the edge cap is ~76x the
# atom cap and it, not the valence columns, sets HBM.
#
# GH200 96 GiB sweep, benchmarks/bench_class_i_hbm.py, random 10 % view, 12
# timed batches per cell, every cell 0 recompiles under error_on_recompile
# (runs/class-i-hbm/*.json, 2026-09-15):
#
#   atoms   mol/batch   small GiB/ms   medium GiB/ms   large GiB/ms   steps/epoch
#    65536       1185      6.4 /  77       7.1 /  84     10.1 / 106          3362
#   131072       2388     12.7 / 143      14.1 / 154     20.2 / 198          1669
#   262144       4806     25.4 / 264      28.1 / 285     40.3 / 369           829
#   524288       9598     47.1 / 513      47.5 / 555      OOM (77.5 peak)      415
#
# 131072 is the operating point: 89 / 90 / 93 % of each tier's peak molecule
# throughput at a quarter of 524288's memory and twice 262144's optimizer
# steps. The HBM ceiling is 2^19 for small and medium, 2^18 for large.
# **The three tiers are scaling-law points, not candidates to choose between.**
# Their x-axis is the parameter count, measured on Rondo (GH200 job 2621863,
# 2026-09-18):
#
#   tier     hidden/n_blocks/n_torsion   parameters
#   small          32 / 2 / 4                41,108
#   medium         64 / 4 / 8               204,188
#   large         128 / 8 / 16            1,191,212
#
# ~5.0x then ~5.8x, so ~29x end to end. These are re-measured rather than
# inherited: the architecture is Rondo now, the reference is ByteFF, and the
# dataset changed, so counts taken against the pre-Rondo model describe a model
# that no longer exists.
#
# **The HBM table below predates Rondo and is not Rondo's.** It was measured on
# 2026-09-15 against the model Rondo replaced, and `benchmarks/bench_class_i_hbm.py`
# — the benchmark that produced it — has since been deleted, so it is not
# reproducible as written. Rondo adds a dense per-molecule EEM solve
# (`RondoSpec.eem_max_atoms`, default 512) that the measured model did not have,
# and nothing currently ties that capacity to the data's largest molecule. Treat
# the numbers as the shape of the trade-off, not as this model's footprint, and
# re-measure before they gate anything. One real Rondo datapoint exists so far:
# `large` peaked at 1.55 GiB at a 16384-atom capacity (job 2621863).
PROD_MODELS: dict[str, dict[str, int]] = {
    "small": {"hidden": 32, "n_blocks": 2, "n_torsion_terms": 4, "packed_max_atoms": 131072},
    "medium": {"hidden": 64, "n_blocks": 4, "n_torsion_terms": 8, "packed_max_atoms": 131072},
    "large": {"hidden": 128, "n_blocks": 8, "n_torsion_terms": 16, "packed_max_atoms": 131072},
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
    # One 4m epoch is ~1.2e5 steps, so an epoch-boundary eval fires twice in a
    # 24 h run. State the stride in steps instead — the eval curve and the
    # best-checkpoint selection both need more than two points.
    "eval_every_n_steps": 5_000,
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
    # A calibration pass is far shorter than one production eval stride; keep
    # its epoch-boundary eval so the gate still reports a validation number.
    "eval_every_n_steps": None,
    "wall_hours": 1,
    "slurm_account": "naiss2026-1-53-gpu",
}

# Checklist §0: production may differ from calibration only in these keys
# (duration, scheduler identity, and output kind). Science pins must match.
_CALIBRATION_EXCEPTIONS = frozenset(
    {
        "kind",
        "epochs",
        "eval_every_epochs",
        "eval_every_n_steps",
        "wall_hours",
        "slurm_account",
    }
)


def production_calibration_drift() -> list[str]:
    """Return science-pin keys that differ between PROD_DAY and CALIBRATION."""
    keys = set(PROD_DAY) | set(CALIBRATION)
    return sorted(
        key
        for key in keys
        if key not in _CALIBRATION_EXCEPTIONS and PROD_DAY.get(key) != CALIBRATION.get(key)
    )


def require_power_of_two(name: str, value: int) -> None:
    """Refuse a model/pack size that is not a power of two."""
    if int(value) < 1 or int(value) & (int(value) - 1):
        raise ValueError(f"{name} must be a power of two, got {value}")


# Named AllScAIP OMol-only 4M protocol. FAIR commit
# 41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9, files omol_4m.yml /
# omol_direct.yaml / uma_v1_hof_lin_refs.yaml, date 2026-09-11.
# This pin does not grant production GO: .claude/notes/production-training-checklist.md
# still applies, and this profile needs a fresh molexp calibration. Historical
# Class-I checklist ticks do not transfer.
OMOL25_FAIRCHEM_4M: dict[str, Any] = {
    "name": "omol25-fairchem-4m",
    "kind": "omol25-fairchem-4m",
    "production_go": False,
    "energy_unit": "eV",
    "fair_commit": "41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9",
    "reference_sha256": "e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640",
    "lr": 8e-4,
    "weight_decay": 1e-3,
    "epochs": 80,
    "warmup_epochs": 5,
    "lr_start_factor": 0.2,
    "lr_min_factor": 0.01,
    "clip_max_norm": 100,
    "bf16": False,
    "tf32": True,
    "refs_dtype": "float64",
    "energy_rms": 1.433569,
    "force_rms": 1.433569,
    "energy_coefficient": 10,
    "force_coefficient": 10,
    "cutoff": 6.0,
    "max_neighbors": 30,
    "neighbor_pad": 50,
    "batch_max_atoms": 400,
    "shuffle_seed": 0,
    "eval_every_n_steps": 2000,
    "checkpoint_keep": 5,
    "world_size": None,
    "global_batch": None,
    "accumulate_grad_batches": None,
    "adamw_betas": None,
    "adamw_eps": None,
    "adamw_decay_groups": None,
    "monitoring_ids": None,
    "monitoring_seed": None,
    "dataset_revision": None,
    "shard_digests": None,
}

PROTOCOL_ADAPTATIONS: tuple[str, ...] = (
    "conservative molix.autograd.Derivative versus the upstream direct-force head",
    "zero-error structures remain in the force denominator",
    "Class-I kcal mol^-1 energy converted to eV at ForceAwareModel.energy via molix.units",
    "upstream nonfinite sanitization replaced by fail-on-invalid-real-labels",
    "Class-I pair list (field radius, F21) versus protocol 6 A / 30 neighbors / pad 50",
    "batch_max_atoms=400 is batch capacity, not a chemistry filter",
)

_PROTOCOL_UNRESOLVED: tuple[str, ...] = (
    "world_size",
    "global_batch",
    "accumulate_grad_batches",
    "adamw_betas",
    "adamw_eps",
    "adamw_decay_groups",
    "monitoring_ids",
    "monitoring_seed",
    "dataset_revision",
    "shard_digests",
)


def protocol_unresolved(profile: Mapping[str, Any]) -> list[str]:
    """Return protocol keys that are still ``None`` and must not be invented."""
    return [key for key in _PROTOCOL_UNRESOLVED if profile.get(key) is None]


def require_protocol_resolved(profile: Mapping[str, Any]) -> None:
    """Raise if *profile* still has unresolved comparison/production metadata."""
    missing = protocol_unresolved(profile)
    if missing:
        raise ValueError(
            "omol25-fairchem-4m has unresolved metadata "
            f"{missing}; do not invent world size, AdamW betas, or monitoring seeds"
        )
