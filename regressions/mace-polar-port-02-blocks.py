"""Public Polar molrep blocks — out_mul, spin source, layer mixer.

Provenance: polar-1-M dump 2026-08-30. No mace/e3nn/graph_longrange at runtime.

    PYTHONPATH=src python regressions/mace-polar-port-02-blocks.py
"""

from __future__ import annotations

import torch

from molix import config
from molrep.interaction.polar import EnvironmentDependentSpinSourceBlock, MultiLayerFeatureMixer
from molrep.readout.mace import NonLinearBiasReadout

config.set_precision("fp64")

torch.manual_seed(0)
n = 4
energy = NonLinearBiasReadout(irreps_in="8x0e", mlp_dim=4)
fukui = NonLinearBiasReadout(irreps_in="8x0e", mlp_dim=4, out_mul=2)
x = torch.randn(n, 8, dtype=torch.float64)
assert energy(x).shape == (n, 1)
assert fukui(x).shape == (n, 2)
assert {"linear_1", "linear_mid", "linear_2"} <= set(energy._modules)

src = EnvironmentDependentSpinSourceBlock(irreps_in="4x0e+4x1o", l_max=1)
h = torch.randn(n, 16, dtype=torch.float64)  # 4*1 + 4*3 = 16
out = src(h)
assert out.shape == (n, 2, 4)
assert src.linear(h).shape[-1] == 8
zeroed = EnvironmentDependentSpinSourceBlock(irreps_in="4x0e+4x1o", l_max=1, zero_charges=True)
assert zeroed(h)[..., 0].abs().max().item() == 0.0

import cuequivariance as cue

assert cue.Irreps("O3", "512x0e+512x1o").dim == 2048
mixer = MultiLayerFeatureMixer(irreps="4x0e+4x1o", num_layers=2)
stacked = torch.randn(2, n, 16, dtype=torch.float64)
mixed = mixer(stacked)
assert mixed.shape == (n, 16)

print("mace-polar-port-02-blocks: ok")
