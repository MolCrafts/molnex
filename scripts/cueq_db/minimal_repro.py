"""Minimal cuet-only reproduction of the force double-backward break + toggle bisect.

Failing combo (real model path): cuet SphericalHarmonics(vectors) -> a cuet
equivariant op. Force-like probe: g = dE/dx (create_graph); backprop g^2;
the cuet-op weight gradient must be > 0 to train forces. Baseline reproduces
weight-grad L1 == 0 (BROKEN). Each toggle is a cuet-only candidate fix.
"""

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
from cuequivariance_torch import SphericalHarmonics as CueSH

torch.set_default_dtype(torch.float64)
dev = "cuda" if torch.cuda.is_available() else "cpu"
SH_IRREPS = cue.Irreps("O3", "1x0e+1x1o+1x2e+1x3o")
OUT = cue.Irreps("O3", "8x0e")


def probe(sh_fn, lin, post=lambda y: y):
    v = torch.randn(16, 3, dtype=torch.float64, device=dev, requires_grad=True)
    E = lin(post(sh_fn(v))).sum()
    g = torch.autograd.grad(E, v, create_graph=True)[0]
    for p in lin.parameters():
        p.grad = None
    g.pow(2).mean().backward()
    return sum(float(p.grad.abs().sum()) for p in lin.parameters() if p.grad is not None)


def lin_irmul():
    return cuet.Linear(SH_IRREPS, OUT, layout=cue.ir_mul, dtype=torch.float64).to(dev)


cuet_sh = CueSH(ls=[0, 1, 2, 3], normalize=True).to(dev)

print("=== baseline: cuet sph -> cuet.Linear (expect BROKEN) ===")
print(f"  L1 = {probe(cuet_sh, lin_irmul()):.3e}")

print("=== toggle A: .contiguous() between ===")
print(f"  L1 = {probe(cuet_sh, lin_irmul(), post=lambda y: y.contiguous()):.3e}")

print("=== toggle B: identity add (1.0*y) plain-torch op between ===")
print(f"  L1 = {probe(cuet_sh, lin_irmul(), post=lambda y: y * 1.0):.3e}")

print("=== toggle C: clone+contiguous between ===")
print(f"  L1 = {probe(cuet_sh, lin_irmul(), post=lambda y: y.clone().contiguous()):.3e}")

print("=== toggle D: cuet sph method='naive' ===")
try:
    sh_naive = CueSH(ls=[0, 1, 2, 3], normalize=True, method="naive").to(dev)
    print(f"  L1 = {probe(sh_naive, lin_irmul()):.3e}")
except Exception as e:
    print(f"  ERR {type(e).__name__}: {str(e)[:70]}")

print("=== toggle E: detach-free reassembly (sum of zero) ===")
print(f"  L1 = {probe(cuet_sh, lin_irmul(), post=lambda y: y + 0.0 * y):.3e}")
