"""PiNet energy+force (EF) training profile — the RevMD17 regime.

The real EF train step is heavier than the bare encoder forward: the model
predicts a graph energy, derives forces ``F = -dE/dpos`` via
``torch.autograd.grad(create_graph=True)``, and the EF loss backprops through
the forces — a *double* backward. This script profiles that step
(``PiNetPotential(compute_forces=True)`` in train mode) so the numbers reflect
revMD17-style force training, not energy-only inference.

--device cpu  : component + submodule + CPU op table for the EF step.
--device cuda : matrix over precision × compiler × batch.
    Compiler note (from molzoo.pinet.PiNetPotential.compile_energy): force
    *training* must stay eager — double-backward through an inductor-compiled
    region raises, and CUDA graphs need static shapes. We profile eager across
    precisions and batch sizes (batch is the launch-bound remedy here), and
    attempt compile_energy / reduce-overhead to confirm/quantify the failure.

Run:
    python benchmarks/bench_pinet_ef.py --device cpu
    python benchmarks/bench_pinet_ef.py --device cuda
"""

from __future__ import annotations

import argparse

import torch

from molix.profiler import MockBatch, ModuleProfiler
from molzoo.pinet import PiNet, PiNetPotential

ATOM_TYPES = list(range(1, 8))


def build_ef_model() -> PiNetPotential:
    """PiNetPotential with force derivation enabled (EF training model)."""
    enc = PiNet(
        atom_types=ATOM_TYPES,
        r_max=5.0,
        n_basis=5,
        pp_nodes=[64, 64],
        pi_nodes=[64, 64],
        ii_nodes=[64, 64],
        depth=3,
        rank=3,
    )
    return PiNetPotential(encoder=enc, hidden_dim=64, compute_forces=True)


def ef_loss(out, batch=None):
    """Energy + force loss (predicted-vs-zero — only the graph shape matters)."""
    e = out["energy"].float()
    f = out["forces"].float()
    return e.pow(2).mean() + f.pow(2).mean()


def _factory(device: str, n_atoms=512, n_edges=4096, n_graphs=16, dtype=None):
    base = MockBatch(
        n_atoms=n_atoms,
        n_edges=n_edges,
        n_graphs=n_graphs,
        atomic_numbers=7,
        device=device,
        seed=0,
    )
    if dtype is None:
        return base

    def cast():
        b = base()
        b["atoms", "pos"] = b["atoms", "pos"].to(dtype)
        return b

    return cast


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


def cpu_hotspots(n_steps: int) -> None:
    from torch.profiler import ProfilerActivity, profile

    print(f"\n{'#' * 78}\nPiNet EF (energy+force) CPU hotspots\n{'#' * 78}")
    model = build_ef_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ModuleProfiler(model, loss_fn=ef_loss, device="cpu", optimizer=opt).run(
        _factory("cpu"), n_steps=n_steps, n_warmup=3, submodules=True
    ).print_report()

    model2 = build_ef_model()
    model2.train()
    batch = _factory("cpu")()
    for _ in range(3):
        ef_loss(model2(batch)).backward()
        model2.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU]) as p:
        for _ in range(n_steps):
            ef_loss(model2(batch)).backward()
            model2.zero_grad(set_to_none=True)
    print("\nTop CPU ops (self time, EF fwd + double-backward):")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))


# ---------------------------------------------------------------------------
# CUDA matrix
# ---------------------------------------------------------------------------

PRECISIONS = ["fp32", "bf16-mixed", "fp16-mixed", "fp64"]


class _AutocastWrap(torch.nn.Module):
    def __init__(self, module, dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, batch):
        # autocast must wrap the whole EF step so the inner force-grad sees it.
        with torch.autocast("cuda", dtype=self.dtype):
            return self.module(batch)


def _build(precision: str):
    m = build_ef_model().cuda()
    if precision == "fp32":
        return m, _factory("cuda")
    if precision == "fp64":
        return m.double(), _factory("cuda", dtype=torch.float64)
    dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    return _AutocastWrap(m, dtype), _factory("cuda")


def _run_cell(label, model, factory, n_steps, rows, compiler, precision, batch=None):
    try:
        params = (model.module if isinstance(model, _AutocastWrap) else model).parameters()
        opt = torch.optim.Adam(params, lr=1e-3)
        res = ModuleProfiler(model, loss_fn=ef_loss, device="cuda", optimizer=opt).run(
            factory, n_steps=n_steps, n_warmup=15, op_rows=0
        )
        row = {
            "compiler": compiler,
            "precision": precision,
            "batch": str(batch or "-"),
            "wall_ms": f"{res.wall_ms_per_step:.3f}" if res.wall_ms_per_step else "n/a",
            "atoms/s": f"{res.throughput_atoms_per_sec:,.0f}",
            "lb%": f"{res.launch_bound_pct:.0f}" if res.launch_bound_pct is not None else "n/a",
        }
        print(f"  [ok]   {label}  wall={row['wall_ms']} lb={row['lb%']}%")
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:70]
        row = {
            "compiler": compiler,
            "precision": precision,
            "batch": str(batch or "-"),
            "wall_ms": "FAIL",
            "atoms/s": type(e).__name__,
            "lb%": msg[:24],
        }
        print(f"  [FAIL] {label}  {type(e).__name__}: {msg}")
    finally:
        torch.compiler.reset()
        torch.cuda.empty_cache()
    rows.append(row)


def gpu_matrix(n_steps: int) -> None:
    print(f"\n{'#' * 78}\nPiNet EF GPU matrix\n{'#' * 78}")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  n_steps={n_steps}")
    rows = []

    print("\n# eager EF across precisions @ bs (512 atoms / 16 graphs):")
    for precision in PRECISIONS:
        model, factory = _build(precision)
        _run_cell(f"eager  {precision:11s}", model, factory, n_steps, rows, "eager", precision, 16)

    print("\n# batch-size remedy: eager fp32, 16 vs 64 graphs:")
    big = _factory("cuda", n_atoms=2048, n_edges=16384, n_graphs=64)
    m, _ = _build("fp32")
    _run_cell("eager  fp32 bs64", m, big, n_steps, rows, "eager", "fp32", 64)

    print("\n# compile attempts on force training (expected to FAIL — double backward):")
    m1 = build_ef_model().cuda()
    m1.compile_energy(backend="inductor")
    _run_cell(
        "compile_energy fp32", m1, _factory("cuda"), n_steps, rows, "compile_energy", "fp32", 16
    )
    m2 = build_ef_model().cuda()
    m2.compile_energy(backend="inductor", mode="reduce-overhead")
    _run_cell(
        "reduce-overhead fp32", m2, _factory("cuda"), n_steps, rows, "reduce-overhead", "fp32", 16
    )

    from molix.profiler._utils import _fmt_table

    print("\n=== EF matrix ===")
    print(
        _fmt_table(
            rows, ["compiler", "precision", "batch", "wall_ms", "atoms/s", "lb%"], col_width=10
        )
    )


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
        gpu_matrix(args.steps)


if __name__ == "__main__":
    main()
