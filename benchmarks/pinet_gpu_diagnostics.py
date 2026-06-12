from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from molix import config
from molzoo.pinet import PiNet, PiNetPotential


def sync() -> None:
    torch.cuda.synchronize()


def make_batch(
    *,
    n_graphs: int,
    atoms_per_graph: int,
    dtype: torch.dtype,
    device: torch.device,
) -> TensorDict:
    n_atoms = n_graphs * atoms_per_graph
    z_cycle = torch.tensor([1, 6, 7, 8], dtype=torch.long, device=device)
    z = z_cycle[torch.arange(n_atoms, device=device) % z_cycle.numel()]
    pos = torch.randn(n_atoms, 3, dtype=dtype, device=device)
    atom_batch = torch.arange(n_graphs, device=device).repeat_interleave(atoms_per_graph)

    edges = [
        (g * atoms_per_graph + i, g * atoms_per_graph + j)
        for g in range(n_graphs)
        for i in range(atoms_per_graph)
        for j in range(atoms_per_graph)
        if i != j
    ]
    edge_index = torch.tensor(edges, dtype=torch.long, device=device)

    return TensorDict(
        atoms=TensorDict(
            Z=z,
            pos=pos,
            batch=atom_batch,
            forces=torch.zeros(n_atoms, 3, dtype=dtype, device=device),
            batch_size=[n_atoms],
        ),
        edges=TensorDict(edge_index=edge_index, batch_size=[edge_index.shape[0]]),
        graphs=TensorDict(
            num_atoms=torch.full((n_graphs,), atoms_per_graph, dtype=torch.long, device=device),
            energy=torch.zeros(n_graphs, dtype=dtype, device=device),
            batch_size=[n_graphs],
        ),
        batch_size=[],
    )


def build_model(*, compute_forces: bool, dtype: torch.dtype, device: torch.device) -> PiNetPotential:
    encoder = PiNet(
        atom_types=[1, 6, 7, 8],
        r_max=4.5,
        cutoff_type="f1",
        basis_type="gaussian",
        n_basis=10,
        pp_nodes=[64, 64, 64, 64],
        pi_nodes=[64],
        ii_nodes=[64, 64, 64, 64],
        depth=5,
        activation="tanh",
        rank=3,
    )
    model = PiNetPotential(encoder=encoder, hidden_dim=64, compute_forces=compute_forces).to(
        device
    )
    warmup = make_batch(n_graphs=1, atoms_per_graph=4, dtype=dtype, device=device)
    model(warmup, compute_forces=False)
    return model.to(dtype=dtype)


def setup_case(
    *,
    graphs: int,
    atoms: int,
    compute_forces: bool,
) -> tuple[TensorDict, PiNetPotential]:
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    config.set_precision("fp32")
    device = torch.device("cuda")
    dtype = torch.float32
    batch = make_batch(n_graphs=graphs, atoms_per_graph=atoms, dtype=dtype, device=device)
    model = build_model(compute_forces=compute_forces, dtype=dtype, device=device)
    return batch, model


def clone_batch(batch: TensorDict) -> TensorDict:
    return batch.clone()


def energy_loss(out: dict[str, torch.Tensor], batch: TensorDict) -> torch.Tensor:
    return F.mse_loss(out["energy"], batch["graphs", "energy"])


def energy_force_loss(out: dict[str, torch.Tensor], batch: TensorDict) -> torch.Tensor:
    return energy_loss(out, batch) + 10.0 * F.mse_loss(out["forces"], batch["atoms", "forces"])


def bench(name: str, fn: Callable[[], Any], *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "iters": iters,
        "seconds": elapsed,
        "ms_per_iter": elapsed * 1000.0 / iters,
        "iter_per_second": iters / elapsed,
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def cudagraph_energy(model: PiNetPotential, batch: TensorDict, iters: int) -> dict[str, Any]:
    model.eval()
    static_batch = clone_batch(batch)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(5):
            model(static_batch, compute_forces=False)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = model(static_batch, compute_forces=False)

    timing = replay(graph, iters)
    return {
        "name": "cudagraph_energy_eval",
        **timing,
        "energy_mean": float(static_out["energy"].detach().mean().cpu()),
    }


def cudagraph_force(model: PiNetPotential, batch: TensorDict, iters: int) -> dict[str, Any]:
    model.eval()

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(5):
            model(clone_batch(batch), compute_forces=True)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    static_batch = clone_batch(batch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = model(static_batch, compute_forces=True)

    timing = replay(graph, iters)
    return {
        "name": "cudagraph_force_eval",
        **timing,
        "force_rms": float(static_out["forces"].detach().square().mean().sqrt().cpu()),
    }


def replay(graph: torch.cuda.CUDAGraph, iters: int) -> dict[str, float | int]:
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    sync()
    elapsed = time.perf_counter() - start
    return {
        "iters": iters,
        "seconds": elapsed,
        "ms_per_iter": elapsed * 1000.0 / iters,
        "iter_per_second": iters / elapsed,
    }


def env_info() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "pid": os.getpid(),
    }


def shape_info(*, graphs: int, atoms: int, batch: TensorDict) -> dict[str, int]:
    return {
        "graphs": graphs,
        "atoms_per_graph": atoms,
        "atoms_total": graphs * atoms,
        "edges_total": int(batch["edges", "edge_index"].shape[0]),
    }


def write_case_result(
    path: Path,
    *,
    graphs: int,
    atoms: int,
    batch: TensorDict,
    row: dict[str, Any],
) -> None:
    payload = {
        "env": env_info(),
        "shape": shape_info(graphs=graphs, atoms=atoms, batch=batch),
        "benchmarks": [row],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
