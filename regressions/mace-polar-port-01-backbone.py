"""Public Polar-1-M backbone dump literals — no energy, no third-party MACE.

Provenance
----------
Official ``MACE-POLAR-1-M.model`` (ACEsuit/mace-foundations ``mace_polar_1``)
stub-unpickled 2026-08-30 via ``scripts/omol_port/_stub_unpickler.py``.
Runtime: no ``mace`` / ``e3nn`` / ``tests`` import.

    PYTHONPATH=src python regressions/mace-polar-port-01-backbone.py
"""

from __future__ import annotations

import torch

from molix import config
from molrep.interaction.residual import ResidualInteraction
from molrep.readout.mace import LinearReadout, NonLinearBiasReadout
from molzoo.mace import MACEMatpesSpec, MACEOMolSpec, MACEPolarSpec
from molzoo.mace.encoder import MACEEncoder

config.set_precision("fp64")

ATOMIC_NUMBERS = [1, 6, 8]
ATOMIC_ENERGIES = [-13.6, -1029.0, -2041.0]
POS = torch.tensor(
    [
        [0.00, 0.00, 0.00],
        [0.95, 0.00, 0.00],
        [-0.24, 0.93, 0.00],
        [0.00, 0.00, 1.40],
        [1.20, 1.10, 0.60],
    ],
    dtype=torch.float64,
)
Z = torch.tensor([8, 1, 1, 6, 1], dtype=torch.long)
pairs = [[i, j] for i in range(len(Z)) for j in range(len(Z)) if i != j]
EDGE_INDEX = torch.tensor(pairs, dtype=torch.long)
EDGE_DIFF = POS[EDGE_INDEX[:, 1]] - POS[EDGE_INDEX[:, 0]]
EDGE_DIST = EDGE_DIFF.norm(dim=-1).clamp(min=1e-6)

DUMP_E0_H = -15.99000358581543
DUMP_N_EL = 83
DUMP_CHANNELS = 512

spec = MACEPolarSpec(atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES)
assert spec.r_max == 6.0
assert spec.num_features == DUMP_CHANNELS
assert spec.max_hidden_l == 1
assert spec.num_interactions == 2
assert spec.l_max == 3
assert spec.edge_channels == 128
assert spec.correlation == 3
assert spec.num_bessel == 8
assert spec.mlp_dim == 16
assert spec.keep_last_layer_irreps is True
assert spec.last_readout_bias is True
assert spec.interaction == "residual"
assert spec.readout == "per_layer"
assert not isinstance(spec, MACEOMolSpec)
assert DUMP_N_EL * DUMP_CHANNELS == 42496
assert DUMP_E0_H == -15.99000358581543

tiny_polar = MACEPolarSpec(
    atomic_numbers=ATOMIC_NUMBERS,
    atomic_energies=ATOMIC_ENERGIES,
    r_max=5.0,
    num_bessel=4,
    num_polynomial_cutoff=5,
    l_max=3,
    num_features=16,
    max_hidden_l=1,
    num_interactions=2,
    correlation=2,
    mlp_dim=8,
    edge_channels=8,
    use_fallback=True,
)
encoder = MACEEncoder(tiny_polar).eval()
assert all(isinstance(block, ResidualInteraction) for block in encoder.interactions)
assert encoder.readouts is not None
assert isinstance(encoder.readouts[0], LinearReadout)
assert isinstance(encoder.readouts[1], NonLinearBiasReadout)

node_attrs = encoder.node_attrs(Z, EDGE_DIFF.dtype)
edge_feats, cutoff = encoder.radial_features(EDGE_DIST, Z, EDGE_INDEX)
feats = encoder.layer_features(
    node_feats=encoder.initial_node_features(node_attrs),
    node_attrs=node_attrs,
    edge_attrs=encoder.angular_features(EDGE_DIFF),
    edge_feats=edge_feats,
    edge_index=EDGE_INDEX,
    cutoff=cutoff,
)
assert feats[-1].shape[-1] == 64

tiny_matpes = MACEMatpesSpec(
    atomic_numbers=ATOMIC_NUMBERS,
    atomic_energies=ATOMIC_ENERGIES,
    r_max=5.0,
    num_bessel=4,
    num_polynomial_cutoff=5,
    l_max=1,
    num_features=16,
    max_hidden_l=1,
    num_interactions=2,
    correlation=2,
    mlp_dim=8,
    radial_mlp=[8],
    use_fallback=True,
)
matpes = MACEEncoder(tiny_matpes).eval()
m_attrs = matpes.node_attrs(Z, EDGE_DIFF.dtype)
m_edge, m_cut = matpes.radial_features(EDGE_DIST, Z, EDGE_INDEX)
m_feats = matpes.layer_features(
    node_feats=matpes.initial_node_features(m_attrs),
    node_attrs=m_attrs,
    edge_attrs=matpes.angular_features(EDGE_DIFF),
    edge_feats=m_edge,
    edge_index=EDGE_INDEX,
    cutoff=m_cut,
)
assert m_feats[-1].shape[-1] == 16

print("mace-polar-port-01-backbone: ok")
