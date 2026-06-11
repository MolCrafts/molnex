"""PiNet EF training — the COMPLETE matrix: rank × precision × compiler.

Every prior PiNet number in this thread was rank=3, fp32. This sweeps the two
axes that were held fixed:
  rank      ∈ {1, 3, 5}              (PiNet tensor order — scalar / +vector / +higher)
  precision ∈ {fp32, fp64, bf16-mixed, fp16-mixed}
  compiler  ∈ {eager, inductor, cudagraphs, reduce-overhead}

Regime: EF *training* (PiNetPotential(compute_forces=True), train mode →
F=-dE/dpos with create_graph=True → double backward). Fixed shape
512 atoms / 4096 edges / 16 graphs; pp/pi/ii=[64,64], depth=3, n_basis=5.

Two tables: (1) wall ms/step per cell (FAIL = raised); (2) cudagraphs-vs-eager
gradient correctness (cosine + rel-L2) per rank×precision.

Run on GH200:  python benchmarks/bench_pinet_ef_full.py
"""

from __future__ import annotations

import copy
import time

import torch

from molix.profiler import MockBatch
from molix.profiler._utils import _fmt_table
from molzoo.pinet import PiNet, PiNetPotential

RANKS = [1, 3, 5]
PRECISIONS = ["fp32", "fp64", "bf16-mixed", "fp16-mixed"]
COMPILERS = ["eager", "inductor", "cudagraphs", "reduce-overhead"]
_KW = {
    "inductor": {"backend": "inductor"},
    "cudagraphs": {"backend": "cudagraphs"},
    "reduce-overhead": {"mode": "reduce-overhead"},
}


def _build(rank: int) -> PiNetPotential:
    enc = PiNet(
        atom_types=list(range(1, 8)),
        r_max=5.0,
        n_basis=5,
        pp_nodes=[64, 64],
        pi_nodes=[64, 64],
        ii_nodes=[64, 64],
        depth=3,
        rank=rank,
    )
    return PiNetPotential(encoder=enc, hidden_dim=64, compute_forces=True).cuda()


def _ef_loss(out):
    return out["energy"].float().pow(2).mean() + out["forces"].float().pow(2).mean()


class _Autocast(torch.nn.Module):
    def __init__(self, m, dtype):
        super().__init__()
        self.module = m
        self.dtype = dtype

    def forward(self, b):
        with torch.autocast("cuda", dtype=self.dtype):
            return self.module(b)


def _batch(dtype):
    b = MockBatch(n_atoms=512, n_edges=4096, n_graphs=16, atomic_numbers=7, device="cuda", seed=0)()
    b["atoms", "pos"] = b["atoms", "pos"].to(dtype)
    return b


def _setup(rank: int, precision: str):
    """Return (model_for_run, batch, base_dtype) for a precision; weights from a
    snapshot so compiled/eager variants in the same cell are comparable."""
    base = _build(rank)
    if precision == "fp32":
        return base, torch.float32
    if precision == "fp64":
        return base.double(), torch.float64
    dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    return _Autocast(base, dtype), torch.float32  # mixed: params fp32, autocast inside


def _inner(model):
    return model.module if isinstance(model, _Autocast) else model


def _run(model, batch):
    inner = _inner(model)
    inner.train()
    inner.zero_grad(set_to_none=True)
    b = batch.copy()
    b["atoms", "pos"] = batch["atoms", "pos"].detach().clone()
    loss = _ef_loss(model(b))
    loss.backward()
    return {
        n: p.grad.detach().double().clone()
        for n, p in inner.named_parameters()
        if p.grad is not None
    }


def _time(model, batch, n_warmup=12, n_steps=40) -> float:
    for _ in range(n_warmup):
        _run(model, batch)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _run(model, batch)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_steps


def _grad_diff(ga, gb):
    names = [n for n in ga if n in gb]
    a = torch.cat([ga[n].flatten() for n in names])
    b = torch.cat([gb[n].flatten() for n in names])
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((b - a).norm() / (a.norm() + 1e-30)).item()
    return cos, rel


def main() -> None:
    print(f"torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}")
    speed = []  # rows keyed by (rank, precision) with a col per compiler
    correct = []

    for rank in RANKS:
        for precision in PRECISIONS:
            dtype = torch.float64 if precision == "fp64" else torch.float32
            batch = _batch(dtype)
            row = {"rank": rank, "precision": precision}
            eager_grads = None

            # One weight snapshot per cell so eager and cudagraphs compare on
            # identical parameters (each variant loads this same state).
            ref, _ = _setup(rank, precision)
            _run(ref, batch)  # materialise lazy params before snapshot
            cell_state = copy.deepcopy(_inner(ref).state_dict())
            del ref
            torch.cuda.empty_cache()

            for comp in COMPILERS:
                tag = f"r{rank} {precision:11s} {comp}"
                try:
                    model, _ = _setup(rank, precision)
                    _run(model, batch)  # materialise lazy before load_state_dict
                    _inner(model).load_state_dict(cell_state)  # identical weights
                    if comp != "eager":
                        _inner(model).compile_energy(**_KW[comp])
                    ms = _time(model, batch)
                    row[comp] = f"{ms:.2f}"
                    print(f"  [ok]   {tag}  {ms:.2f} ms")
                    if comp == "eager":
                        eager_grads = _run(model, batch)
                    if comp == "cudagraphs" and eager_grads is not None:
                        for _ in range(3):
                            _run(model, batch)
                        cos, rel = _grad_diff(eager_grads, _run(model, batch))
                        correct.append(
                            {
                                "rank": rank,
                                "precision": precision,
                                "cosine": f"{cos:.6f}",
                                "rel_L2": f"{rel:.2e}",
                            }
                        )
                except Exception as e:  # noqa: BLE001
                    row[comp] = "FAIL"
                    print(f"  [FAIL] {tag}  {type(e).__name__}: {str(e).splitlines()[0][:50]}")
                finally:
                    torch.compiler.reset()
                    torch.cuda.empty_cache()
            speed.append(row)

    print("\n=== EF-train wall ms/step  (rank × precision × compiler) ===")
    print(_fmt_table(speed, ["rank", "precision", *COMPILERS], col_width=9))
    print("\n=== cudagraphs vs eager gradient correctness (per rank × precision) ===")
    print(_fmt_table(correct, ["rank", "precision", "cosine", "rel_L2"], col_width=9))


if __name__ == "__main__":
    main()
