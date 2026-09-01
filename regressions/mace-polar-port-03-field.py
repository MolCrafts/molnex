"""Public Polar field primitives — SparseUvu, one-body update, field readout.

Provenance: polar-1-M dump 2026-08-30. No mace/e3nn/graph_longrange at runtime.

    PYTHONPATH=src python regressions/mace-polar-port-03-field.py
"""

from __future__ import annotations

import cuequivariance as cue
import torch

from molix import config
from molrep.interaction.polar import (
    AgnosticEmbeddedOneBodyVariableUpdate,
    OneBodyMLPFieldReadout,
    SparseUvuTensorProduct,
)

config.set_precision("fp64")

tp = SparseUvuTensorProduct("1x0e", "1x0e", "1x0e")
with torch.no_grad():
    tp.weight.fill_(1.0)
assert tp(torch.tensor([[2.0]]), torch.tensor([[3.0]])).item() == 6.0

n_0e = sum(mi.mul for mi in cue.Irreps("O3", "512x0e+512x1o") if mi.ir.l == 0)
assert n_0e == 512
assert [2 * n_0e, 64, 64, 64, n_0e] == [1024, 64, 64, 64, 512]
assert [2 * n_0e, 128, 128, 128, 1] == [1024, 128, 128, 128, 1]
assert cue.Irreps("O3", "32x0e+32x1o").dim == 128
assert (cue.Irreps("O3", "2x0e+2x1o") + cue.Irreps("O3", "2x0e")).dim == 10

torch.manual_seed(0)
n = 2
update = AgnosticEmbeddedOneBodyVariableUpdate(
    node_attrs_irreps="2x0e",
    node_feats_irreps="4x0e+4x1o",
    potential_irreps="2x0e+2x1o",
    charges_irreps="2x0e+2x1o",
)
assert update.nonlinearity.hs == [8, 64, 64, 64, 4]
assert update.readout.linear_1.irreps_out.dim == 160
out = update(
    torch.randn(n, 2, dtype=torch.float64),
    torch.randn(n, 16, dtype=torch.float64),
    torch.randn(n, 8, dtype=torch.float64),
    torch.randn(n, 8, dtype=torch.float64),
)
assert out.shape == (n, 10)

readout = OneBodyMLPFieldReadout(
    node_feats_irreps="4x0e+4x1o",
    potential_irreps="2x0e+2x1o",
    charges_irreps="2x0e+2x1o",
)
assert readout.mlp.hs == [8, 128, 128, 128, 1]
energy = readout(
    torch.randn(n, 16, dtype=torch.float64),
    torch.randn(n, 8, dtype=torch.float64),
    torch.randn(n, 8, dtype=torch.float64),
    torch.randn(n, 8, dtype=torch.float64),
)
assert energy.shape == (n,)

print("mace-polar-port-03-field: ok")
