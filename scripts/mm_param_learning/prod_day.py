"""One-day Class-I run for one named model size. Wall clock is the stop.

    PYTHONPATH=.:src:../molhub/src:../molpy/src \\
      python scripts/mm_param_learning/prod_day.py medium
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from scripts.mm_param_learning.class_i_fit import (
    ClassIFit,
    load_omol25_official,
    val_omol_mae,
)
from scripts.mm_param_learning.fingerprint import (
    PROD_DAY,
    PROD_MODELS,
    attach_fingerprint,
    fingerprint,
    write_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
_DEFAULT_MOLHUB_HOME = REPO_ROOT.parent / "workspaces" / "molhub-home"


def _require_nobackup_molhub_home() -> None:
    """Keep the 4m extract off the 30 GiB home quota."""
    if os.environ.get("MOLHUB_HOME"):
        return
    _DEFAULT_MOLHUB_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["MOLHUB_HOME"] = str(_DEFAULT_MOLHUB_HOME)


def config_for(size: str) -> dict:
    """QUASI_PROD pins plus one named capacity."""
    if size not in PROD_MODELS:
        raise ValueError(f"unknown model size {size!r}; known {sorted(PROD_MODELS)}")
    cfg = dict(PROD_DAY)
    cfg.update(PROD_MODELS[size])
    cfg["model_size"] = size
    return cfg


def run(size: str) -> None:
    """Fingerprint → official OMol25 4m/val-neutral → ClassIFit until wall clock."""
    _require_nobackup_molhub_home()
    cfg = config_for(size)
    blob = fingerprint(cfg)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_ROOT / f"prod-day1-{size}-{blob['fingerprint'][:12]}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_fingerprint(run_dir / "fingerprint.json", blob)
    print(f"size={size} fingerprint={blob['fingerprint']}", flush=True)
    print(f"run_dir={run_dir}", flush=True)

    train, val = load_omol25_official(
        train_role=str(cfg["omol_role"]),
        val_role=str(cfg["val_role"]),
        max_atoms=int(cfg["packed_max_atoms"]),
    )
    coverage = {
        "n_train_frames": len(train),
        "n_val_frames": len(val),
        "train_n_atoms_max": int(train.atom_counts.max()) if len(train) else 0,
        "val_n_atoms_max": int(val.atom_counts.max()) if len(val) else 0,
    }
    (run_dir / "split.json").write_text(
        json.dumps(attach_fingerprint(coverage, blob["fingerprint"]), indent=2) + "\n"
    )
    print(json.dumps({"coverage": coverage, "config": cfg}, sort_keys=True), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = ClassIFit(
        train,
        val,
        hidden=int(cfg["hidden"]),
        n_blocks=int(cfg["n_blocks"]),
        n_torsion_terms=int(cfg["n_torsion_terms"]),
        max_atoms=int(cfg["packed_max_atoms"]),
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        seed=int(cfg["seed"]),
        device=device,
        record_root=run_dir,
        eval_every_epochs=int(cfg["eval_every_epochs"]),
        energy_scope=str(cfg["energy_scope"]),
        eval_loss=val_omol_mae,
        clip_max_norm=float(cfg["clip_max_norm"]),
    ).run()
    (run_dir / "metrics.json").write_text(
        json.dumps(attach_fingerprint(metrics, blob["fingerprint"]), indent=2) + "\n"
    )
    print(metrics, flush=True)
    print(f"=== PROD-DAY1 {size} DONE {run_dir} ===", flush=True)


def smoke_official(
    *,
    size: str = "small",
    max_steps: int = 16,
    load_budget_s: float = 300.0,
) -> None:
    """Official 4m + named model, bounded steps. Fails if startup is too slow."""
    _require_nobackup_molhub_home()
    cfg = config_for(size)
    t0 = perf_counter()
    train, val = load_omol25_official(
        train_role=str(cfg["omol_role"]),
        val_role=str(cfg["val_role"]),
        max_atoms=int(cfg["packed_max_atoms"]),
    )
    load_s = perf_counter() - t0
    print(
        f"official_load_s={load_s:.1f} n_train={len(train)} n_val={len(val)}",
        flush=True,
    )
    if load_s > load_budget_s:
        raise RuntimeError(
            f"official load took {load_s:.1f}s > {load_budget_s}s; "
            "this must be caught in smoke, not on a 24h GPU job"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = ClassIFit(
        train,
        val,
        hidden=int(cfg["hidden"]),
        n_blocks=int(cfg["n_blocks"]),
        n_torsion_terms=int(cfg["n_torsion_terms"]),
        max_atoms=int(cfg["packed_max_atoms"]),
        epochs=1,
        lr=float(cfg["lr"]),
        seed=int(cfg["seed"]),
        device=device,
        energy_scope=str(cfg["energy_scope"]),
        eval_loss=val_omol_mae,
        clip_max_norm=float(cfg["clip_max_norm"]),
    ).run(max_steps=max_steps)
    print({"smoke": metrics, "official_load_s": load_s}, flush=True)
    print("=== OFFICIAL SMOKE DONE ===", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} small|medium|large|smoke")
    if sys.argv[1] == "smoke":
        smoke_official()
    else:
        run(sys.argv[1])
