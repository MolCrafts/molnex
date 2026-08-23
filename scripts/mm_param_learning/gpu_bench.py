"""Timed GPU Class-I step. Not a training launcher.

    PYTHONPATH=.:src:../molhub/src:../molpy/src \\
      python scripts/mm_param_learning/gpu_bench.py
"""

from __future__ import annotations

import time

import torch

from scripts.mm_param_learning.class_i_fit import ClassIFit, load_omol25_split


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(fn, *, warmup: int = 3, repeats: int = 12) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync()
    return 1e3 * (time.perf_counter() - t0) / repeats


def run_bench() -> None:
    """Time encode / energy / E+F / train step on the available accelerator."""
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_bench requires CUDA; login L40 is compute-prohibited")
    device = "cuda"
    print(f"device={device} name={torch.cuda.get_device_name(0)}")

    t_load = time.perf_counter()
    train, val = load_omol25_split(role="val-neutral", n_keep=32)
    print(f"load_ms={(time.perf_counter() - t_load) * 1e3:.1f}")

    fit = ClassIFit(train, val, hidden=64, n_blocks=4, device=device, epochs=1)
    dm = fit.datamodule()
    model = fit.model()
    batch = next(iter(dm.train_dataloader())).to(device, non_blocking=True)
    n_graphs = int(batch["graphs", "num_atoms"].shape[0])
    n_atoms = int(batch["atoms", "Z"].shape[0])
    print(
        f"n_train={len(train)} batch_graphs={n_graphs} atoms={n_atoms} "
        f"dtype={next(model.parameters()).dtype}"
    )

    imm = model.imm

    def encode() -> None:
        imm(batch)

    def energy() -> None:
        with torch.no_grad():
            model(batch)

    def energy_forces() -> None:
        model.zero_grad(set_to_none=True)
        out = model(batch)
        _ = out["energy"].sum() + out["forces"].sum()

    def step() -> None:
        model.zero_grad(set_to_none=True)
        out = model(batch)
        loss = out["energy"].sum()
        if "forces" in out:
            loss = loss + out["forces"].square().mean()
        loss.backward()

    encode_ms = _ms(encode)
    energy_ms = _ms(energy)
    ef_ms = _ms(energy_forces)
    step_ms = _ms(step)
    print(f"encode_ms={encode_ms:.2f}")
    print(f"energy_fwd_ms={energy_ms:.2f}")
    print(f"energy_forces_ms={ef_ms:.2f}")
    print(f"train_step_ms={step_ms:.2f}")
    print(f"atoms_per_s={1e3 * n_atoms / step_ms:.1f}")
    print(f"graphs_per_s={1e3 * n_graphs / step_ms:.1f}")
    print(f"peak_alloc_mib={torch.cuda.max_memory_allocated() / 1024**2:.1f}")


if __name__ == "__main__":
    run_bench()
