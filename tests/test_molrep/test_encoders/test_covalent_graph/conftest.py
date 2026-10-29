"""A small covalent graph and a tiny covalent-graph encoder spec."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

#: Small sizes; the fields are the spec's.
TINY = dict(
    atom_embedding_dim=4,
    channel_dim=2,
    node_mlp=(8, 8, 2),
    edge_mlp=(8, 6, 2),
    width=8,
    n_layers=2,
    heads=2,
    head_dim=4,
    ffn=(8, 2),
)

DTYPE = torch.float64

#: Acetate CH3-COO(-): C0 methyl, C1 carboxyl, O2 (=), O3 (-), H4-6.
Z = [6, 6, 8, 8, 1, 1, 1]
BONDS = [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (0, 6)]
#: Equivalence ignores bond order and charge: the two O and the three H are one class each.
EQUI = [0, 1, 2, 2, 4, 4, 4]


def build(
    bonds=BONDS, z=Z, equi=EQUI, *, bond_type=(1, 2, 1, 1, 1, 1), charge=(0, 0, 0, -1, 0, 0, 0)
):
    n = len(z)
    index = torch.tensor(bonds, dtype=torch.long).T.contiguous()
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "Z": torch.tensor(z),
                    "formal_charge": torch.tensor(charge),
                    "ring_connectivity": torch.zeros(n, dtype=torch.long),
                    "min_ring_size": torch.zeros(n, dtype=torch.long),
                    "equi_index": torch.tensor(equi),
                },
                batch_size=[n],
            ),
            "bonds": TensorDict(
                {
                    "bond_index": index,
                    "bond_type": torch.tensor(bond_type),
                    "in_ring": torch.zeros(len(bonds), dtype=torch.long),
                },
                batch_size=[],
            ),
        },
        batch_size=[],
    )


@pytest.fixture
def acetate() -> TensorDict:
    return build()
