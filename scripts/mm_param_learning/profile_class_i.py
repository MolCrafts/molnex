"""GPU profile of the Class-I ``LazyOMol25`` data + train path.

Complements ``gpu_bench.py`` (which times compute only, over in-memory
``OMol25Small`` records). This script exercises the lazy LMDB fast path and
the DataLoader worker knobs, then reports:

* DataLoader throughput (samples/s) at ``num_workers`` in {0, 2, 4},
* a cProfile of the per-sample path (``_sample_from_record`` →
  ``infer_bonds`` → valence enumeration),
* a ``torch.profiler`` CPU+CUDA table over a few real train steps.

Run on a GH200 GPU node via ``run_class_i_profile_gh200.sbatch``.
"""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, schedule

from molhub.dataset.omol25 import OMol25
from molpot.heads.energy import AtomicReferenceEnergy
from scripts.mm_param_learning.class_i_fit import (
    ClassIFit,
    LazyOMol25,
    train_objective,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"


def build_lazy_pair(
    role: str, n_train: int, n_val: int, max_atoms: int = 32
) -> tuple[LazyOMol25, LazyOMol25]:
    """One bounded LMDB view shared by train and val (LMDB is single-open)."""
    # The ≤max_atoms filter keeps ~6% of scanned rows; scan generously so
    # train+val can still be filled without a second pass.
    scan_cap = max(int(n_train + n_val) * 32, 128)
    src = OMol25(role=role, max_rows=scan_cap).load()
    counts = np.asarray(src.atom_counts(), dtype=np.int64)
    kept = [i for i in range(len(src)) if int(counts[i]) <= max_atoms]
    train_idx = kept[:n_train]
    val_idx = kept[n_train : n_train + n_val]
    dressing = AtomicReferenceEnergy(atomic_energies=torch.zeros(95))
    train = LazyOMol25(src, train_idx, [int(counts[i]) for i in train_idx], dressing=dressing)
    val = LazyOMol25(src, val_idx, [int(counts[i]) for i in val_idx], dressing=dressing)
    print(
        f"built lazy {role}: rows_scanned={len(src)} kept_train={len(train)} kept_val={len(val)}",
        flush=True,
    )
    return train, val


def _throughput(fit: ClassIFit, *, warmup: int = 4) -> dict[str, float]:
    """Wall-clock batches/s and samples/s over one full train epoch.

    ``warmup`` batches are pulled first so ``num_workers>0`` spawn/re-import
    cost does not dominate a short timed window; ``first_batch_s`` still
    records the cold startup latency for that worker count.
    """
    dm = fit.datamodule()
    loader = dm.train_dataloader()
    it = iter(loader)
    t0 = time.perf_counter()
    try:
        next(it)
    except StopIteration:
        return {
            "batches": 0,
            "wall_s": 0.0,
            "batches_per_s": 0.0,
            "samples_per_s": 0.0,
            "atoms_per_s": 0.0,
            "first_batch_s": time.perf_counter() - t0,
        }
    first_batch_s = time.perf_counter() - t0
    for _ in range(max(0, warmup - 1)):
        try:
            next(it)
        except StopIteration:
            break
    n_atoms = 0
    n_graphs = 0
    n_batches = 0
    t1 = time.perf_counter()
    for batch in it:
        n_graphs += int(batch["graphs", "num_atoms"].shape[0])
        n_atoms += int(batch["atoms", "Z"].shape[0])
        n_batches += 1
    dt = max(time.perf_counter() - t1, 1e-6)
    return {
        "batches": n_batches,
        "wall_s": dt,
        "batches_per_s": n_batches / dt,
        "samples_per_s": n_graphs / dt,
        "atoms_per_s": n_atoms / dt,
        "first_batch_s": first_batch_s,
    }


def cprofile_sample_path(train_ds: LazyOMol25, *, n: int = 300) -> None:
    """cProfile the lazy ``__getitem__`` path (in-process, num_workers=0)."""
    pr = cProfile.Profile()
    pr.enable()
    for i in range(n):
        train_ds[i]
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
    print("=== cProfile: LazyOMol25.__getitem__ (top by cumulative) ===")
    print(s.getvalue())


def torch_profile(fit: ClassIFit, *, active: int = 12) -> dict[str, float]:
    """torch.profiler over forward+backward+optimizer with a worker-backed loader."""
    dm = fit.datamodule()
    model = fit.model()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = dm.train_dataloader()
    it = iter(loader)
    wait, warmup = 2, 2
    schedule_ = schedule(wait=wait, warmup=warmup, active=active, repeat=1)
    n_done = 0
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule_,
        record_shapes=True,
    ) as prof:
        for _ in range(wait + warmup + active):
            try:
                batch = next(it)
            except StopIteration:
                print(f"loader exhausted after {n_done} profiled steps", flush=True)
                break
            batch = batch.to(fit.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = train_objective(out, batch)
            loss.backward()
            optimizer.step()
            prof.step()
            n_done += 1
    print(f"profiler_steps={n_done}")
    print("=== torch.profiler: top CPU operators ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=18))
    print("=== torch.profiler: top CUDA operators ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
    print("=== torch.profiler: self CUDA time by shape hint ===")
    total_cuda = sum(e.self_device_time_total for e in prof.key_averages())
    print(f"total_self_cuda_us={total_cuda:.0f}")
    opt_events = [
        e for e in prof.key_averages() if "adam" in e.key.lower() or "optimizer" in e.key.lower()
    ]
    optimizer_cuda_us = sum(e.self_device_time_total for e in opt_events)
    print(f"optimizer_self_cuda_us={optimizer_cuda_us:.0f}")
    return {
        "profiler_steps": float(n_done),
        "total_self_cuda_us": float(total_cuda),
        "optimizer_self_cuda_us": float(optimizer_cuda_us),
    }


def _full_step_timing(fit: ClassIFit, *, steps: int = 50) -> dict[str, float]:
    """CUDA-event wall-clock of zero_grad→forward→backward→optimizer.step."""
    dm = fit.datamodule()
    model = fit.model()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    it = iter(dm.train_dataloader())
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = batch.to(fit.device, non_blocking=True)
        torch.cuda.synchronize()
        start.record()
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = train_objective(out, batch)
        loss.backward()
        optimizer.step()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    if not times_ms:
        return {
            "full_step_n": 0.0,
            "full_step_median_ms": float("nan"),
            "full_step_min_ms": float("nan"),
            "full_step_max_ms": float("nan"),
        }
    ordered = sorted(times_ms)
    median = ordered[len(ordered) // 2]
    return {
        "full_step_n": float(len(ordered)),
        "full_step_median_ms": median,
        "full_step_min_ms": ordered[0],
        "full_step_max_ms": ordered[-1],
    }


def run_profile() -> None:
    device = "cuda"
    print(f"device={device} name={torch.cuda.get_device_name(0)}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_ROOT / f"profile-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir}", flush=True)

    n_train, n_val = 2048, 256
    t0 = time.perf_counter()
    train, val = build_lazy_pair("val-neutral", n_train, n_val)
    print(f"build_lazy_s={time.perf_counter() - t0:.1f}", flush=True)

    # Token packing: keep batches small enough that the loader yields many
    # batches (max_atoms=4096 would pack ~170 small molecules/batch, so 2048
    # samples → ~12 batches, too few to amortize worker spawn or profile).
    max_atoms = 512

    # 1) DataLoader throughput at several worker counts.
    throughput: dict[str, dict[str, float]] = {}
    for nw in (0, 2, 4):
        fit = ClassIFit(
            train,
            val,
            hidden=64,
            n_blocks=4,
            device=device,
            epochs=1,
            max_atoms=max_atoms,
            num_workers=nw,
            prefetch_factor=4 if nw else None,
        )
        stats = _throughput(fit)
        throughput[str(nw)] = stats
        print(
            f"throughput num_workers={nw}: "
            f"{stats['samples_per_s']:.0f} samples/s "
            f"({stats['atoms_per_s']:.0f} atoms/s, "
            f"{stats['batches_per_s']:.1f} batches/s over {stats['wall_s']:.2f}s, "
            f"first_batch={stats['first_batch_s'] * 1e3:.0f}ms, "
            f"n_batches={stats['batches']})",
            flush=True,
        )

    # 2) cProfile of the in-process sample path.
    cprofile_sample_path(train)

    # 3) torch.profiler over real train steps with workers (now incl. optimizer).
    fit = ClassIFit(
        train,
        val,
        hidden=64,
        n_blocks=4,
        device=device,
        epochs=1,
        max_atoms=max_atoms,
        num_workers=4,
        prefetch_factor=4,
    )
    profiler_stats = torch_profile(fit)

    # 4) CUDA-event wall-clock of the full step including optimizer.step().
    full_step = _full_step_timing(fit)
    print(f"full_step={full_step}", flush=True)

    payload = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0),
        "max_atoms": max_atoms,
        "throughput": throughput,
        "profiler": profiler_stats,
        "full_step": full_step,
    }
    (run_dir / "profile.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"=== DONE profile {run_dir} ===", flush=True)


if __name__ == "__main__":
    run_profile()
