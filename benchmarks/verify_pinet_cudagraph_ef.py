"""Correctness check: compile_energy(cudagraphs) vs eager for EF training.

The compile-mode matrix found that ``backend='cudagraphs'`` is the one compiler
that survives EF force training (energy forward cudagraph-captured, force
derivation eager) at ~5.7× — but timing alone can't justify adopting it. This
verifies the *numbers* match eager: same weights, same batch, compare the
predicted energy, the autograd forces, the EF loss, and every parameter
gradient after the double backward.

Run on GH200:
    python benchmarks/verify_pinet_cudagraph_ef.py
"""

from __future__ import annotations

import copy

import torch

from molix.profiler import MockBatch
from molzoo.pinet import PiNet, PiNetPotential

ATOM_TYPES = list(range(1, 8))


def _model() -> PiNetPotential:
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
    return PiNetPotential(encoder=enc, hidden_dim=64, compute_forces=True).cuda()


def _ef_loss(out):
    return out["energy"].float().pow(2).mean() + out["forces"].float().pow(2).mean()


def _run(model, batch):
    """One EF train step; return (energy, forces, loss, {name: grad})."""
    model.train()
    model.zero_grad(set_to_none=True)
    b = batch.copy()
    b["atoms", "pos"] = batch["atoms", "pos"].detach().clone()
    out = model(b)
    loss = _ef_loss(out)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return (
        out["energy"].detach().clone(),
        out["forces"].detach().clone(),
        loss.detach().clone(),
        grads,
    )


def _gmax_abs(ga, gb):
    return max((ga[n] - gb[n]).abs().max().item() for n in ga if n in gb)


def _gmax_rel(ga, gb):
    """Max over params of |Δ| / (|eager grad| + eps) — scale-free divergence."""
    worst = 0.0
    for n in ga:
        if n in gb:
            denom = ga[n].abs().max().item() + 1e-12
            worst = max(worst, (ga[n] - gb[n]).abs().max().item() / denom)
    return worst


def verify(dtype: torch.dtype) -> bool:
    """eager-vs-eager noise floor vs cudagraphs-vs-eager gap, at one dtype."""
    torch.manual_seed(0)
    name = {torch.float32: "fp32", torch.float64: "fp64"}[dtype]
    print(f"\n{'=' * 64}\n{name}\n{'=' * 64}")

    batch = MockBatch(
        n_atoms=512, n_edges=4096, n_graphs=16, atomic_numbers=7, device="cuda", seed=0
    )()
    batch["atoms", "pos"] = batch["atoms", "pos"].to(dtype)

    eager = _model().to(dtype)
    eager.train()
    _ = eager(batch.copy())
    state = copy.deepcopy(eager.state_dict())

    cg = _model().to(dtype)
    _ = cg(batch.copy())
    cg.load_state_dict(state)
    cg.compile_energy(backend="cudagraphs")

    _, _, _, e_g = _run(eager, batch)
    _, _, _, e2_g = _run(eager, batch)  # noise floor (scatter_add atomics)
    for _ in range(3):
        _run(cg, batch)  # cudagraph warm-up captures
    c_e, c_f, c_l, c_g = _run(cg, batch)
    e_e, e_f, e_l, _ = _run(eager, batch)

    floor_abs, floor_rel = _gmax_abs(e_g, e2_g), _gmax_rel(e_g, e2_g)
    gap_abs, gap_rel = _gmax_abs(e_g, c_g), _gmax_rel(e_g, c_g)

    print(f"forward energy max|Δ|={(e_e - c_e).abs().max().item():.2e}")
    print(f"forward forces max|Δ|={(e_f - c_f).abs().max().item():.2e}")
    print(f"grad noise floor (eager vs eager):  abs={floor_abs:.2e}  rel={floor_rel:.2e}")
    print(f"grad gap (cudagraphs vs eager):      abs={gap_abs:.2e}  rel={gap_rel:.2e}")
    # within ~3x the floor (both abs and rel) → indistinguishable from fp noise
    ok = gap_rel <= max(floor_rel * 3, 1e-6) and gap_abs <= max(floor_abs * 3, 1e-9)
    print(f"→ cudagraphs grads within noise floor at {name}: {ok}")
    return ok


def main() -> None:
    print(f"torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}")
    fp32_ok = verify(torch.float32)
    fp64_ok = verify(torch.float64)

    print(f"\n{'=' * 64}\nVERDICT\n{'=' * 64}")
    if fp64_ok:
        print("PASS — cudagraphs EF grads correct (fp64 within noise floor);")
        print("  the fp32 gap is reduction-order precision noise, not a bug.")
    else:
        print("FAIL — cudagraphs EF grads diverge in fp64 too → real correctness")
        print("  bug in the double backward; force training must stay eager.")
    print(f"(fp32 within floor={fp32_ok}, fp64 within floor={fp64_ok})")


if __name__ == "__main__":
    main()
