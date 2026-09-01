"""Public MacePolar + polar_remap — dump literals, no official energy.

Provenance: polar-1-M dump 2026-08-30. No mace/e3nn/graph_longrange at runtime.

    PYTHONPATH=src python regressions/mace-polar-port-05-wire.py
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molix import config
from molzoo.mace.checkpoint import CheckpointRemap, polar_keys
from molzoo.mace.polar import MacePolar
from molzoo.mace.spec import MACEPolarSpec

config.set_precision("fp64")

spec = MACEPolarSpec(
    atomic_numbers=[1, 6, 8],
    atomic_energies=[-13.6, -1029.0, -2041.0],
    r_max=5.0,
    num_bessel=4,
    l_max=3,
    num_features=16,
    max_hidden_l=1,
    num_interactions=2,
    correlation=2,
    mlp_dim=8,
    edge_channels=8,
    use_fallback=True,
)
assert spec.num_recursion_steps == 2
assert spec.multipole_sigma == 1.5
assert spec.field_widths == [1.5, 3.0]
assert spec.field_norms == [20.0, 20.0, 0.5, 0.5]
assert spec.add_local_electron_energy is True

model = MacePolar(
    atomic_numbers=spec.atomic_numbers,
    atomic_energies=spec.atomic_energies,
    r_max=spec.r_max,
    num_bessel=spec.num_bessel,
    l_max=spec.l_max,
    num_features=spec.num_features,
    max_hidden_l=spec.max_hidden_l,
    num_interactions=spec.num_interactions,
    correlation=spec.correlation,
    mlp_dim=spec.mlp_dim,
    edge_channels=spec.edge_channels,
    use_fallback=True,
    compute_forces=False,
)
assert {"lr_source_maps", "field_dependent_charges_maps", "fukui_source_map"} <= set(model._modules)
assert len(model.field_dependent_charges_maps) == 2
assert model.field_feature_norms.numel() == 8

n = 3
pos = torch.tensor(
    [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    dtype=torch.float64,
)
Z = torch.tensor([8, 1, 1], dtype=torch.long)
pairs = [[i, j] for i in range(n) for j in range(n) if i != j]
edge_index = torch.tensor(pairs, dtype=torch.long)
atom_batch = torch.zeros(n, dtype=torch.long)
edge_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
batch = TensorDict(
    {
        "atoms": TensorDict(
            {"Z": Z, "pos": pos, "batch": atom_batch},
            batch_size=[n],
        ),
        "edges": TensorDict(
            {
                "edge_index": edge_index,
                "edge_diff": edge_diff,
                "edge_dist": edge_diff.norm(dim=-1).clamp(min=1e-6),
            },
            batch_size=[len(pairs)],
        ),
        "graphs": TensorDict(
            {
                "total_charge": torch.zeros(1, dtype=torch.float64),
                "total_spin": torch.ones(1, dtype=torch.float64),
            },
            batch_size=[1],
        ),
    },
    batch_size=[],
)
energy = model(batch)["graphs", "energy"]
assert energy.shape == (1,)
assert torch.isfinite(energy).all()

shifted = pos + 3.0
batch["atoms", "pos"] = shifted
batch["edges", "edge_diff"] = shifted[edge_index[:, 1]] - shifted[edge_index[:, 0]]
energy_t = model(batch)["graphs", "energy"]
assert torch.allclose(energy, energy_t, atol=1e-8)

polar_remap = CheckpointRemap(polar_keys, on_unexpected="return")
assert polar_remap._on_unexpected == "return"

print("mace-polar-port-05-wire: ok")
