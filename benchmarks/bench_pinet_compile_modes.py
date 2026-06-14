"""PiNet: where can compile_energy actually help, with the force part eager?

PiNetPotential compiles only the energy forward (compile_energy); ForceDerivation
is torch.compiler.disable'd, so the autograd.grad force computation always runs
eager. The open question this measures, head-on:

  * EF TRAINING   (create_graph=True  → double backward through the compiled
                   energy region): does compile_energy work? expected NO.
  * EF INFERENCE  (create_graph=False → single backward for the force): the
                   force part stays eager, energy compiled — expected YES.
  * ENERGY-ONLY   (no forces at all): the regime compile_energy was built for.

× backends {inductor, cudagraphs, reduce-overhead} — so "did we use CUDA
graphs?" is answered explicitly (cudagraphs backend, and reduce-overhead =
inductor+cudagraph), not just bundled into one mode.

Self-timed (no ModuleProfiler train() assumption) so train vs inference and
create_graph are controlled exactly. Run on GH200:
    python benchmarks/bench_pinet_compile_modes.py
"""

from __future__ import annotations

import time

import torch

from molix.profiler import MockBatch
from molzoo.pinet import PiNet, PiNetPotential

ATOM_TYPES = list(range(1, 8))
BACKENDS = {
    "eager": None,
    "inductor": {"backend": "inductor"},
    "cudagraphs": {"backend": "cudagraphs"},
    "reduce-overhead": {"mode": "reduce-overhead"},
}


def _model(compute_forces: bool) -> PiNetPotential:
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
    return PiNetPotential(encoder=enc, hidden_dim=64, compute_forces=compute_forces).cuda()


def _batch():
    return MockBatch(
        n_atoms=512, n_edges=4096, n_graphs=16, atomic_numbers=7, device="cuda", seed=0
    )()


def _ef_loss(out):
    return out["energy"].float().pow(2).mean() + out["forces"].float().pow(2).mean()


def _e_loss(out):
    return out["energy"].float().pow(2).mean()


def _time(model, batch, *, train: bool, with_forces: bool, n_warmup=15, n_steps=50) -> float:
    """Wall ms/step. train=True does loss.backward()+opt.step() (double backward
    when with_forces); train=False is forward-only (inference, create_graph=False)."""
    model.train(train)
    opt = torch.optim.Adam(model.parameters(), lr=0.0) if train else None
    loss_fn = _ef_loss if with_forces else _e_loss

    def step():
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        out = model(_freshen(batch))
        if train:
            loss_fn(out).backward()
            opt.step()

    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_steps


def _freshen(batch):
    # pos must be a fresh leaf each step (the model clones+requires_grad it, but
    # reusing across steps with retained graphs can leak); clone pos defensively.
    b = batch.copy()
    b["atoms", "pos"] = batch["atoms", "pos"].detach().clone()
    return b


def _cell(label, compute_forces, train, backend_kw, rows):
    try:
        model = _model(compute_forces)
        if backend_kw is not None:
            model.compile_energy(**backend_kw)
        ms = _time(model, _batch(), train=train, with_forces=compute_forces)
        rows.append({**label, "wall_ms": f"{ms:.3f}", "status": "ok"})
        print(f"  [ok]   {label}  wall={ms:.3f} ms")
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:54]
        rows.append({**label, "wall_ms": "FAIL", "status": msg})
        print(f"  [FAIL] {label}  {type(e).__name__}: {msg}")
    finally:
        torch.compiler.reset()
        torch.cuda.empty_cache()


def main() -> None:
    print(f"torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}")
    rows = []

    print("\n# EF TRAINING (compute_forces=True, create_graph=True → double backward):")
    for name, kw in BACKENDS.items():
        _cell({"regime": "EF-train", "backend": name}, True, True, kw, rows)

    print("\n# EF INFERENCE (compute_forces=True, eval → create_graph=False, force eager):")
    for name, kw in BACKENDS.items():
        _cell({"regime": "EF-infer", "backend": name}, True, False, kw, rows)

    print("\n# ENERGY-ONLY TRAINING (compute_forces=False — compile_energy's home turf):")
    for name, kw in BACKENDS.items():
        _cell({"regime": "E-train", "backend": name}, False, True, kw, rows)

    from molix.profiler._utils import _fmt_table

    print("\n=== compile-mode matrix (wall ms/step) ===")
    print(_fmt_table(rows, ["regime", "backend", "wall_ms", "status"], col_width=10))


if __name__ == "__main__":
    main()
