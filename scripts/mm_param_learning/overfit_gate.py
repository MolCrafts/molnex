"""Checklist §3 per-head tiny-set overfit gate on official 4m coordinates.

Trains the Class-I model on a fixed tiny set (first ``N_TINY_TRAIN`` official
``4m`` molecules, ≤32 atoms) until each active head's own MSE drops below a
predeclared fraction of its step-0 value. Outputs ``overfit.json``; exit 0
iff every head passes.

    python scripts/mm_param_learning/overfit_gate.py small
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.mm_param_learning.class_i_fit import (
    ClassIFit,
    energy_force_terms,
    load_omol25_official_tiny,
)
from scripts.mm_param_learning.configs import CALIBRATION, PROD_MODELS
from scripts.mm_param_learning.prod_day import _require_nobackup_molhub_home

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"

# Predeclared learnability thresholds (checklist §3). A head passes when its
# final tiny-set MSE is at most this fraction of its step-0 MSE and strictly
# below it. A zero-initial head must stay zero.
ENERGY_REDUCTION_FRACTION = 0.1
FORCE_REDUCTION_FRACTION = 0.1
N_TINY_TRAIN = 8
N_TINY_VAL = 2
MAX_STEPS = 4000


def _cfg_for(size: str) -> dict:
    """PROD_DAY science pins + one capacity, tagged as an overfit gate."""
    if size not in PROD_MODELS:
        raise ValueError(f"unknown model size {size!r}; known {sorted(PROD_MODELS)}")
    cfg = dict(CALIBRATION)
    cfg.update(PROD_MODELS[size])
    cfg["model_size"] = size
    cfg["kind"] = f"overfit-gate-{size}"
    return cfg


def _head_ok(final: float, initial: float, fraction: float) -> bool:
    """A head overfits when its loss drops to ``fraction`` of step-0 (or stays 0)."""
    if initial <= 0.0:
        return final <= 0.0
    return final <= fraction * initial and final < initial


def _model_with_device_retry(
    fit: ClassIFit, *, attempts: int = 6, delay_s: float = 15.0
) -> torch.nn.Module:
    """Build the model, retrying a transient ``cudaErrorDevicesUnavailable``.

    GH200 nodes intermittently report a busy/unavailable device at the first
    ``.to("cuda")`` even though no compute has run yet. The condition clears on
    its own; retrying in-process avoids failing an otherwise healthy gate.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fit.model()
        except torch.AcceleratorError as exc:
            if "busy or unavailable" not in str(exc) or attempt == attempts:
                raise
            print(
                f"device busy at model move (attempt {attempt}/{attempts}); "
                f"retrying in {delay_s:.0f}s",
                flush=True,
            )
            time.sleep(delay_s)
    raise AssertionError("unreachable")  # pragma: no cover


def run(size: str, *, max_steps: int = MAX_STEPS) -> dict:
    """Bounded overfit on a fixed official-4m tiny set. Not a production run."""
    _require_nobackup_molhub_home()
    cfg = _cfg_for(size)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_ROOT / f"overfit-{size}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"size={size}", flush=True)
    print(f"run_dir={run_dir}", flush=True)

    train, val = load_omol25_official_tiny(n_train=N_TINY_TRAIN, n_val=N_TINY_VAL, max_atoms=32)
    fit = ClassIFit(
        train,
        val,
        hidden=int(cfg["hidden"]),
        n_blocks=int(cfg["n_blocks"]),
        n_torsion_terms=int(cfg["n_torsion_terms"]),
        max_atoms=int(cfg["packed_max_atoms"]),
        epochs=1,
        lr=float(cfg["lr"]),
        seed=int(cfg["seed"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
        energy_scope="full",
        clip_max_norm=float(cfg["clip_max_norm"]),
    )
    model = _model_with_device_retry(fit)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    loader = fit.datamodule().train_dataloader()

    initial: tuple[float, float] | None = None
    final_e = float("nan")
    final_f = float("nan")
    steps = 0
    while steps < max_steps:
        for batch in loader:
            steps += 1
            if steps > max_steps:
                break
            batch = batch.to(fit.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            e_term, f_term = energy_force_terms(out, batch)
            e_f = float(e_term.detach())
            f_f = float(f_term.detach())
            if initial is None:
                initial = (e_f, f_f)
            loss = e_term + f_term
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at overfit step {steps}")
            loss.backward()
            optimizer.step()
            final_e, final_f = e_f, f_f
            if steps % 200 == 0:
                print(
                    f"overfit step={steps} energy_mse={final_e:.6g} force_mse={final_f:.6g}",
                    flush=True,
                )

    assert initial is not None
    initial_e, initial_f = initial
    energy_ok = _head_ok(final_e, initial_e, ENERGY_REDUCTION_FRACTION)
    force_ok = _head_ok(final_f, initial_f, FORCE_REDUCTION_FRACTION)
    payload = {
        "size": size,
        "n_steps": steps,
        "n_tiny_train": N_TINY_TRAIN,
        "n_tiny_val": N_TINY_VAL,
        "energy_mse_initial": initial_e,
        "energy_mse_final": final_e,
        "force_mse_initial": initial_f,
        "force_mse_final": final_f,
        "energy_reduction_fraction": ENERGY_REDUCTION_FRACTION,
        "force_reduction_fraction": FORCE_REDUCTION_FRACTION,
        "energy_overfit_ok": float(energy_ok),
        "force_overfit_ok": float(force_ok),
        "ok": float(energy_ok and force_ok),
    }
    (run_dir / "overfit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"=== OVERFIT {size} DONE {run_dir} ===", flush=True)
    return payload


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} small|medium|large")
    raise SystemExit(0 if run(sys.argv[1])["ok"] else 1)
