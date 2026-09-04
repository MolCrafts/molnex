"""PEO-box compiled vs eager: fp64 parity and same-job fp32 step times.

Hard-coded official PolarMACE goldens live in
``regressions/mace-polar-periodic-gto.py`` / ``mace-polar-official-water.py``
(this file does not import mace / e3nn / ASE). CUDA + the three
``MOLNEX_POLAR_*`` files required; skips when any is missing.

    MOLNEX_POLAR_BOX=... MOLNEX_POLAR_DUMP=... MOLNEX_POLAR_CUEQ_STATE=... \\
    PYTHONPATH=src python regressions/polar-peo-compile-compare.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch
from tensordict import TensorDict

from molix import config
from molix.compile import Compiler
from molix.md import NeighborList
from molpot.forcefield import ForceField
from molzoo.mace.checkpoint import CheckpointRemap, polar_keys
from molzoo.mace.polar import MacePolar

_BOX = Path(os.environ.get("MOLNEX_POLAR_BOX", "/tmp/mace-polar/peo_box.pt"))
_DUMP = Path(os.environ.get("MOLNEX_POLAR_DUMP", "/tmp/mace-polar/MACE-POLAR-1-M.model"))
_CUEQ = Path(os.environ.get("MOLNEX_POLAR_CUEQ_STATE", "/tmp/polar_cueq_state.pt"))
_OUT = Path(os.environ.get("MOLNEX_MEASURE_OUT", "polar-peo-compile-compare.json"))
_STEPS = int(os.environ.get("MOLNEX_MEASURE_STEPS", "20"))


def build(dtype: torch.dtype, device: torch.device):
    config.set_precision("fp64" if dtype == torch.float64 else "fp32")
    blob = torch.load(_BOX, map_location="cpu", weights_only=True)
    z = blob["Z"].to(device=device, dtype=torch.long)
    pos = blob["pos"].to(device=device, dtype=dtype)
    cell = blob["cell"].to(device=device, dtype=dtype)
    n = int(pos.shape[0])
    model = MacePolar.load(_DUMP, format="official", use_fallback=False, compute_forces=False)
    CheckpointRemap(polar_keys, on_unexpected="return").load(
        model, torch.load(_CUEQ, map_location="cpu", weights_only=True)
    )
    model = model.to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)
    work = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "Z": z,
                    "pos": pos,
                    "batch": torch.zeros(n, dtype=torch.long, device=device),
                },
                batch_size=[n],
            ),
            "graphs": TensorDict(
                {
                    "cell": cell.unsqueeze(0),
                    "total_charge": torch.zeros(1, dtype=dtype, device=device),
                    "total_spin": torch.ones(1, dtype=dtype, device=device),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )
    nl = NeighborList(
        cell=cell,
        cutoff=float(model._spec.r_max),
        positions=pos,
        skin=0.0,
        every=1,
        delay=0,
        check=False,
        capacity_factor=1.2,
        device=device,
    )
    nl.build(work)
    n_live = nl.num_edges
    full = work["edges"]
    work["edges"] = TensorDict({k: full[k][:n_live] for k in full.keys()}, batch_size=[n_live])
    ff = ForceField("mace-polar", units="metal")
    ff.def_pairstyle(model)
    return ff.to_potentials(), work, cell, n_live, nl


def efv(pots, work):
    with torch.no_grad():
        energy, forces, virial = pots.calc_energy_forces_virial(work)
    return energy.detach().reshape(()), forces.detach(), virial.detach()


def time_steps(pots, work, n_steps: int) -> tuple[list[float], float]:
    torch.cuda.reset_peak_memory_stats()
    efv(pots, work)
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        efv(pots, work)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times, torch.cuda.max_memory_allocated() / 2**30


def main() -> None:
    if not (_BOX.is_file() and _DUMP.is_file() and _CUEQ.is_file()):
        print("skip: PEO box, Polar dump or cueq overlay state not present")
        return
    if not torch.cuda.is_available():
        print("skip: needs CUDA for the 2.9e5-edge box")
        return

    device = torch.device("cuda")
    record: dict[str, object] = {
        "date": time.strftime("%Y-%m-%d"),
        "job": os.environ.get("SLURM_JOB_ID"),
        "cuda_name": torch.cuda.get_device_name(0),
    }

    pots, work, cell, n_live, _ = build(torch.float64, device)
    record["n_edges"] = n_live
    e0, f0, w0 = efv(pots, work)
    pots.bind_cell(cell)
    with torch.no_grad():
        pots.calc_energy(work)
    pots.compile_core(Compiler(dynamic=True))
    e1, f1, w1 = efv(pots, work)
    record["fp64_dE"] = float((e1 - e0).abs())
    record["fp64_max_dF"] = float((f1 - f0).abs().max())
    record["fp64_max_dW"] = float((w1 - w0).abs().max())
    assert record["fp64_dE"] < 1e-10, record
    assert record["fp64_max_dF"] < 1e-10, record
    assert record["fp64_max_dW"] < 1e-9, record
    _OUT.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print("polar-peo-compile-compare fp64:", json.dumps(record), flush=True)
    del pots, work
    torch.cuda.empty_cache()

    torch._dynamo.reset()
    pots, work, cell, _n_live, nl = build(torch.float32, device)
    pots.bind_cell(cell)
    with torch.no_grad():
        pots.calc_energy(work)

    eager_times, eager_peak = time_steps(pots, work, _STEPS)
    record["fp32_eager_step_mean_s"] = statistics.mean(eager_times)
    record["fp32_eager_step_min_s"] = min(eager_times)
    record["fp32_eager_peak_gib"] = eager_peak

    pots.compile_core(Compiler(dynamic=True))
    from torch._dynamo.utils import counters

    compiled_times, compiled_peak = time_steps(pots, work, _STEPS)
    graphs_after_warmup = int(counters["stats"]["unique_graphs"])
    record["fp32_compiled_step_mean_s"] = statistics.mean(compiled_times)
    record["fp32_compiled_step_min_s"] = min(compiled_times)
    record["fp32_compiled_peak_gib"] = compiled_peak
    record["fp32_speedup"] = float(
        record["fp32_eager_step_mean_s"] / record["fp32_compiled_step_mean_s"]
    )
    assert record["fp32_compiled_step_mean_s"] < record["fp32_eager_step_mean_s"], record

    with torch.no_grad():
        work["atoms", "pos"] = work["atoms", "pos"] + 0.01 * torch.randn_like(work["atoms", "pos"])
    nl.build(work)
    n_live2 = nl.num_edges
    full = work["edges"]
    work["edges"] = TensorDict({k: full[k][:n_live2] for k in full.keys()}, batch_size=[n_live2])
    efv(pots, work)
    record["n_edges_after_rebuild"] = n_live2
    record["recompiles_after_warmup"] = (
        int(counters["stats"]["unique_graphs"]) - graphs_after_warmup
    )
    assert record["recompiles_after_warmup"] <= 1, record

    _OUT.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print("polar-peo-compile-compare:", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
