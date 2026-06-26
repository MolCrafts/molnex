"""PiNet performance: CPU hotspots + GPU compiler×precision matrix.

--device cpu  : component + per-submodule forward breakdown (ModuleProfiler)
                plus a torch.profiler CPU op-level table — where PiNet's time
                goes on CPU.
--device cuda : a {compiler} × {precision} matrix. Each cell builds a PiNet
                variant and reports wall/step, atoms/s and launch-bound%.
                compilers: eager, inductor, cudagraphs, reduce-overhead
                (reduce-overhead = inductor + CUDA-graph trees — CUDA graphs
                are themselves treated as a compiler here).
                precisions: fp32, bf16-mixed, fp16-mixed, fp64.

Run:
    python benchmarks/bench_pinet.py --device cpu
    python benchmarks/bench_pinet.py --device cuda
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch

from molix.profiler import MockBatch, ModuleProfiler
from molzoo.pinet import PiNet

ATOM_TYPES = list(range(1, 8))  # cover MockBatch Z ∈ [1, 7]


def build_pinet() -> PiNet:
    """A representative PiNet encoder (depth 3, 64-wide MLPs, rank 3)."""
    return PiNet(
        atom_types=ATOM_TYPES,
        r_max=5.0,
        n_basis=5,
        pp_nodes=[64, 64],
        pi_nodes=[64, 64],
        ii_nodes=[64, 64],
        depth=3,
        rank=3,
    )


def _loss(out, batch=None):
    return out["atoms", "node_features"].float().sum()


def _factory(device: str, dtype: torch.dtype | None = None):
    f = MockBatch(n_atoms=512, n_edges=4096, n_graphs=16, atomic_numbers=7, device=device, seed=0)
    if dtype is None:
        return f

    def cast():
        b = f()
        b["atoms", "pos"] = b["atoms", "pos"].to(dtype)
        if ("edges", "edge_diff") in b.keys(include_nested=True):
            b["edges", "edge_diff"] = b["edges", "edge_diff"].to(dtype)
            b["edges", "edge_dist"] = b["edges", "edge_dist"].to(dtype)
        return b

    return cast


# ---------------------------------------------------------------------------
# CPU hotspots
# ---------------------------------------------------------------------------


def cpu_hotspots(n_steps: int) -> None:
    from torch.profiler import ProfilerActivity, profile

    print(f"\n{'#' * 78}\nPiNet CPU hotspots\n{'#' * 78}")
    enc = build_pinet()
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    prof = ModuleProfiler(enc, loss_fn=_loss, device="cpu", optimizer=opt)
    prof.run(_factory("cpu"), n_steps=n_steps, n_warmup=3, submodules=True).print_report()

    # CPU op-level attribution (ModuleProfiler's op table is CUDA-only).
    enc2 = build_pinet()
    batch = _factory("cpu")()
    for _ in range(3):
        _loss(enc2(batch)).backward()
        enc2.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU]) as p:
        for _ in range(n_steps):
            _loss(enc2(batch)).backward()
            enc2.zero_grad(set_to_none=True)
    print("\nTop CPU ops (self time, fwd+bwd):")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))


# ---------------------------------------------------------------------------
# GPU compiler × precision matrix
# ---------------------------------------------------------------------------

COMPILERS = {
    "eager": lambda m: m,
    "inductor": lambda m: torch.compile(m, backend="inductor"),
    "cudagraphs": lambda m: torch.compile(m, backend="cudagraphs"),
    "reduce-overhead": lambda m: torch.compile(m, mode="reduce-overhead"),
}
PRECISIONS = ["fp32", "bf16-mixed", "fp16-mixed", "fp64"]


class _AutocastWrap(torch.nn.Module):
    """Run the wrapped module's forward under autocast(dtype)."""

    def __init__(self, module: torch.nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, batch):
        with torch.autocast("cuda", dtype=self.dtype):
            return self.module(batch)


def _make_variant(precision: str):
    """Return (model, batch_factory) configured for *precision* on cuda."""
    enc = build_pinet().cuda()
    if precision == "fp32":
        return enc, _factory("cuda")
    if precision == "fp64":
        return enc.double(), _factory("cuda", torch.float64)
    dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    return _AutocastWrap(enc, dtype), _factory("cuda")


def gpu_matrix(n_steps: int) -> None:
    print(f"\n{'#' * 78}\nPiNet GPU compiler × precision matrix\n{'#' * 78}")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  n_steps={n_steps}")
    rows = []
    for precision in PRECISIONS:
        for cname, compile_fn in COMPILERS.items():
            label = f"{cname:16s} {precision:11s}"
            try:
                model, factory = _make_variant(precision)
                model = compile_fn(model)
                opt = torch.optim.Adam(
                    (model.module if isinstance(model, _AutocastWrap) else model).parameters(),
                    lr=1e-3,
                )
                res = ModuleProfiler(model, loss_fn=_loss, device="cuda", optimizer=opt).run(
                    factory, n_steps=n_steps, n_warmup=15, op_rows=0
                )
                rows.append(
                    {
                        "compiler": cname,
                        "precision": precision,
                        "wall_ms": f"{res.wall_ms_per_step:.3f}"
                        if res.wall_ms_per_step
                        else "n/a",
                        "atoms/s": f"{res.throughput_atoms_per_sec:,.0f}",
                        "lb%": f"{res.launch_bound_pct:.0f}" if res.launch_bound_pct else "n/a",
                    }
                )
                print(f"  [ok]   {label}  wall={rows[-1]['wall_ms']} ms  lb={rows[-1]['lb%']}%")
            except Exception as e:  # noqa: BLE001 — one bad cell shouldn't kill the matrix
                msg = str(e).splitlines()[0][:60]
                rows.append({"compiler": cname, "precision": precision, "wall_ms": "FAIL"})
                print(f"  [FAIL] {label}  {type(e).__name__}: {msg}")
            finally:
                torch.compiler.reset()
                torch.cuda.empty_cache()

    from molix.profiler._utils import _fmt_table

    print("\n=== matrix (wall ms/step, atoms/s, launch-bound%) ===")
    print(_fmt_table(rows, ["compiler", "precision", "wall_ms", "atoms/s", "lb%"], col_width=11))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    print(f"torch {torch.__version__} | device={args.device}")
    if args.device == "cpu":
        cpu_hotspots(args.steps)
    else:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        with nullcontext():
            gpu_matrix(args.steps)


if __name__ == "__main__":
    main()
