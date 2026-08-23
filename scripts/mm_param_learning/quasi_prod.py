"""Quasi-production one-epoch look-see. Not a multi-epoch GO.

Writes a split manifest, exercises checkpoint resume, sweeps packed atom
budgets for HBM, then trains one epoch. Direct ClassIFit — run identity is
owned by molexp.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from scripts.mm_param_learning.align import SIZE_BUCKETS
from scripts.mm_param_learning.class_i_fit import (
    ClassIFit,
    energy_force_terms,
    load_omol25_official_tiny,
    load_omol25_split,
    split_coverage,
    train_objective,
    val_omol_mae,
    write_split_manifest,
)
from scripts.mm_param_learning.configs import CALIBRATION, QUASI_PROD

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"


def _n_atoms(record) -> int:
    atoms = record[0]["atoms"]
    if "number" in atoms:
        return int(len(atoms["number"]))
    return int(len(atoms["x"]))


def _bucket_counts(records) -> dict[str, int]:
    counts = {name: 0 for name, _, _ in SIZE_BUCKETS}
    for rec in records:
        n = _n_atoms(rec)
        for name, lo, hi in SIZE_BUCKETS:
            if lo <= n <= hi:
                counts[name] += 1
                break
    return counts


def run_step_smoke(
    train,
    val,
    *,
    device: str,
    record_root: Path,
    max_steps: int = 32,
) -> dict[str, float]:
    """Bounded train steps: energy/force terms, finite grads, val MAE."""
    fit = ClassIFit(
        train,
        val,
        hidden=64,
        n_blocks=4,
        max_atoms=int(QUASI_PROD["packed_max_atoms"]),
        epochs=1,
        device=device,
        record_root=record_root / "step-smoke",
        eval_every_epochs=1,
        energy_scope="full",
        eval_loss=val_omol_mae,
    )
    model = fit.model()
    model.train()
    n_finite = 0
    last_e = float("nan")
    last_f = float("nan")
    last_g = float("nan")
    n_steps = 0
    for step, batch in enumerate(fit.datamodule().train_dataloader(), start=1):
        if step > max_steps:
            break
        batch = batch.to(fit.device)
        model.zero_grad(set_to_none=True)
        out = model(batch)
        energy_term, force_term = energy_force_terms(out, batch)
        loss = energy_term + force_term
        finite = bool(
            torch.isfinite(out["energy"]).all()
            and torch.isfinite(out["forces"]).all()
            and torch.isfinite(loss)
        )
        if finite:
            loss.backward()
            grads_ok = all(
                param.grad is None or bool(torch.isfinite(param.grad).all())
                for param in model.parameters()
            )
            pieces = [
                param.grad.detach().reshape(-1)
                for param in model.parameters()
                if param.requires_grad and param.grad is not None
            ]
            last_g = float(torch.cat(pieces).norm().item()) if pieces else 0.0
        else:
            grads_ok = False
        n_finite += int(finite and grads_ok)
        last_e = float(energy_term.detach()) if finite else float("nan")
        last_f = float(force_term.detach()) if finite else float("nan")
        n_steps += 1
        print(
            f"smoke step={step} energy={last_e:.4g} force={last_f:.4g} "
            f"grad={last_g:.4g} finite={int(finite and grads_ok)}",
            flush=True,
        )
    e0 = fit.dressing.atomic_energies.detach()
    live = e0[e0.abs() > 0]
    return {
        "n_steps": float(n_steps),
        "n_finite": float(n_finite),
        "last_energy_term": last_e,
        "last_force_term": last_f,
        "last_grad_norm": last_g,
        "e0_n_live": float(live.numel()),
        "e0_abs_med": float(live.abs().median()) if live.numel() else 0.0,
    }


def run_resume_smoke(train, val, *, device: str, record_root: Path) -> dict[str, float]:
    """Train max_steps=1, resume last.pt, train max_steps=2."""
    ckpt = record_root / "resume-smoke"
    fit = ClassIFit(
        train,
        val,
        hidden=64,
        n_blocks=4,
        max_atoms=int(QUASI_PROD["packed_max_atoms"]),
        epochs=1,
        device=device,
        record_root=ckpt,
        eval_every_epochs=1,
        energy_scope="full",
        eval_loss=val_omol_mae,
    )
    fit.run(max_steps=1)
    last = ckpt / "checkpoints" / "last.pt"
    if not last.is_file():
        raise FileNotFoundError(f"resume smoke missing {last}")
    resumed = ClassIFit(
        train,
        val,
        hidden=64,
        n_blocks=4,
        max_atoms=int(QUASI_PROD["packed_max_atoms"]),
        epochs=2,
        device=device,
        record_root=ckpt,
        eval_every_epochs=1,
        energy_scope="full",
        eval_loss=val_omol_mae,
    )
    metrics = resumed.run(resume_from=last, max_steps=2)
    return {"resume_ok": 1.0, **metrics}


def pack_sweep(train, val, *, device: str) -> list[dict[str, float]]:
    """One forward+backward at successive packed atom budgets. Records HBM."""
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("pack_sweep needs CUDA")
    rows: list[dict[str, float]] = []
    for max_atoms in (4096, 8192, 16384, 32768, 65536, 131072):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fit = ClassIFit(
            train,
            val,
            hidden=64,
            n_blocks=4,
            max_atoms=max_atoms,
            epochs=1,
            device=device,
            energy_scope="full",
            eval_loss=val_omol_mae,
        )
        model = fit.model()
        try:
            batch = next(iter(fit.datamodule().train_dataloader())).to(device)
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = perf_counter()
            out = model(batch)
            loss = train_objective(out, batch)
            loss.backward()
            torch.cuda.synchronize()
            step_ms = 1e3 * (perf_counter() - t0)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            n_mols = int(batch["graphs", "num_atoms"].shape[0])
            n_atoms = int(batch["atoms", "Z"].shape[0])
            rows.append(
                {
                    "packed_max_atoms": float(max_atoms),
                    "n_mols": float(n_mols),
                    "n_atoms": float(n_atoms),
                    "peak_gib": float(peak),
                    "step_ms": float(step_ms),
                    "loss": float(loss.detach()),
                    "oom": 0.0,
                }
            )
            print(
                f"pack max_atoms={max_atoms} n_mols={n_mols} n_atoms={n_atoms} "
                f"peak_gib={peak:.3f} step_ms={step_ms:.1f} loss={float(loss.detach()):.4g}",
                flush=True,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            rows.append(
                {
                    "packed_max_atoms": float(max_atoms),
                    "n_mols": 0.0,
                    "n_atoms": 0.0,
                    "peak_gib": 0.0,
                    "step_ms": float("nan"),
                    "loss": float("nan"),
                    "oom": 1.0,
                }
            )
            print(f"pack max_atoms={max_atoms} OOM", flush=True)
            break
    return rows


def main() -> None:
    """Split → resume → pack sweep → one epoch."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_ROOT / f"quasi-prod-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir}", flush=True)

    cfg = QUASI_PROD
    train, val = load_omol25_split(
        role=str(cfg["omol_role"]),
        n_keep=0,
        max_atoms=32,
        train_frac=float(cfg["train_frac"]),
        seed=int(cfg["seed"]),
    )
    coverage = split_coverage(train, val)
    write_split_manifest(run_dir / "split.json", train, val)
    print(
        json.dumps({"coverage": coverage, "buckets": _bucket_counts(train)}, sort_keys=True),
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    smoke_n = min(64, len(train), len(val))
    resume = run_resume_smoke(
        train[:smoke_n],
        val[: max(smoke_n // 4, 2)],
        device=device,
        record_root=run_dir,
    )
    print(f"resume_smoke={resume}", flush=True)

    pack_rows: list[dict[str, float]] = []
    if device == "cuda":
        pack_rows = pack_sweep(train, val, device=device)
        (run_dir / "pack_sweep.json").write_text(json.dumps({"rows": pack_rows}, indent=2) + "\n")

    fit = ClassIFit(
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
        eval_every_epochs=1,
        energy_scope=str(cfg["energy_scope"]),
        eval_loss=val_omol_mae,
        clip_max_norm=float(cfg["clip_max_norm"]),
    )
    metrics = fit.run()
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(metrics, flush=True)
    print(f"=== QUASI-PROD DONE {run_dir} ===", flush=True)


def smoke() -> None:
    """Split → resume → 32-step term dump. Not a production run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_ROOT / f"smoke-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir}", flush=True)

    cfg = QUASI_PROD
    train, val = load_omol25_split(
        role=str(cfg["omol_role"]),
        n_keep=0,
        max_atoms=32,
        train_frac=float(cfg["train_frac"]),
        seed=int(cfg["seed"]),
    )
    coverage = split_coverage(train, val)
    write_split_manifest(run_dir / "split.json", train, val)
    print(
        json.dumps({"coverage": coverage, "buckets": _bucket_counts(train)}, sort_keys=True),
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resume = run_resume_smoke(train[:64], val[:16], device=device, record_root=run_dir)
    print(f"resume_smoke={resume}", flush=True)
    steps = run_step_smoke(train, val, device=device, record_root=run_dir, max_steps=32)
    print(f"step_smoke={steps}", flush=True)
    payload = {
        "coverage": coverage,
        "resume": resume,
        "steps": steps,
    }
    (run_dir / "smoke.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"=== SMOKE DONE {run_dir} ===", flush=True)


def calibrate() -> None:
    """Align + scale audit + timed pack sweep on official 4m/val-neutral. Not training."""
    from scripts.mm_param_learning.align import run_align
    from scripts.mm_param_learning.prod_day import _require_nobackup_molhub_home
    from scripts.mm_param_learning.scale_audit import run_inspect

    _require_nobackup_molhub_home()
    cfg = CALIBRATION
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_ROOT / f"calibrate-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir}", flush=True)

    train, val = load_omol25_official_tiny(n_train=48, n_val=12, max_atoms=32)
    coverage = {
        "n_train_frames": len(train),
        "n_val_frames": len(val),
        "train_n_atoms_min": int(train.atom_counts.min()) if len(train) else 0,
        "train_n_atoms_max": int(train.atom_counts.max()) if len(train) else 0,
        "val_n_atoms_min": int(val.atom_counts.min()) if len(val) else 0,
        "val_n_atoms_max": int(val.atom_counts.max()) if len(val) else 0,
    }
    (run_dir / "split.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(json.dumps({"coverage": coverage}, sort_keys=True), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    align = run_align(n_keep=48, device=device)
    print(
        json.dumps(
            {"align_ok": align.get("ok"), "buckets": align.get("buckets")},
            default=str,
        ),
        flush=True,
    )
    inspect = run_inspect(n_keep=16, device=device)
    print(json.dumps({"inspect": inspect}, default=str), flush=True)
    pack_rows: list[dict[str, float]] = []
    if device == "cuda":
        pack_rows = pack_sweep(train, val, device=device)
        (run_dir / "pack_sweep.json").write_text(json.dumps({"rows": pack_rows}, indent=2) + "\n")
    payload = {
        "coverage": coverage,
        "align": align,
        "inspect": inspect,
        "pack_sweep": pack_rows,
        "packed_max_atoms": int(cfg["packed_max_atoms"]),
    }
    (run_dir / "calibrate.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"=== CALIBRATE DONE {run_dir} ===", flush=True)


if __name__ == "__main__":
    main()
