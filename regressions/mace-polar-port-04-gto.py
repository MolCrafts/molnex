"""Public Polar GTO kernel + Fukui + l=0 Hartree — dump literals.

Provenance: polar-1-M dump 2026-08-30. No mace/e3nn/graph_longrange at runtime.

    PYTHONPATH=src python regressions/mace-polar-port-04-gto.py
"""

from __future__ import annotations

import math

import torch

from molix import config
from molpot.potentials.elec.prefactors import eV_A
from molpot.potentials.gto import FukuiEquilibration, GTOCoulombKernel, GTOHartreeEnergy

config.set_precision("fp64")

SIGMA = 1.5
kernel = GTOCoulombKernel(sigma=SIGMA)
r = torch.tensor([1.0], dtype=torch.float64)
assert kernel(r).item() == math.erf(1.0 / 3.0)
assert kernel(r).item() != math.erf(1.0 / (math.sqrt(2.0) * SIGMA))

energy = GTOHartreeEnergy(sigma=SIGMA, l_max=0)
pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
q = torch.ones(2, 1, dtype=torch.float64)
batch = torch.zeros(2, dtype=torch.long)
got = energy(q, pos, batch)
assert got.shape == (1,)
assert abs(got.item() - eV_A * math.erf(1.0 / 3.0)) < 1e-10

eq = FukuiEquilibration()
density = torch.zeros(2, 2, 4, dtype=torch.float64)
fukui = torch.ones(2, 2, dtype=torch.float64)
out = eq(
    density,
    fukui,
    batch,
    total_charge=torch.zeros(1, dtype=torch.float64),
    total_spin=torch.full((1,), 2.0, dtype=torch.float64),
)
assert abs(out[:, 0, 0].sum().item() - 0.5) < 1e-12
assert abs(out[:, 1, 0].sum().item() + 0.5) < 1e-12

print("mace-polar-port-04-gto: ok")
