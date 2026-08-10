"""Shared fixtures for molrep.chem unit tests."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict


@pytest.fixture
def atom_dim() -> int:
    return 8


@pytest.fixture
def bond_dim() -> int:
    return 8


@pytest.fixture
def mini_batch() -> TensorDict:
    """Synthetic water-like topology with full valence namespaces.

    Atoms: O(0), H(1), H(2), C(3) for a second bond leg.
    Bonds: 0-1, 0-2, 0-3
    Angles: 1-0-2, 1-0-3
    Propers: 1-0-3-2 (one)
    Impropers: center=0, outer={1,2,3}
    """
    z = torch.tensor([8, 1, 1, 6], dtype=torch.long)
    n = z.shape[0]
    batch = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "Z": z,
                    "pos": torch.zeros(n, 3),
                    "batch": torch.zeros(n, dtype=torch.long),
                },
                batch_size=[n],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0, 0, 0], dtype=torch.long),
                    "atomj": torch.tensor([1, 2, 3], dtype=torch.long),
                    "bond_types": torch.tensor([1, 1, 1], dtype=torch.long),
                },
                batch_size=[3],
            ),
            "angles": TensorDict(
                {
                    "atomi": torch.tensor([1, 1], dtype=torch.long),
                    "atomj": torch.tensor([0, 0], dtype=torch.long),
                    "atomk": torch.tensor([2, 3], dtype=torch.long),
                },
                batch_size=[2],
            ),
            "propers": TensorDict(
                {
                    "atomi": torch.tensor([1], dtype=torch.long),
                    "atomj": torch.tensor([0], dtype=torch.long),
                    "atomk": torch.tensor([3], dtype=torch.long),
                    "atoml": torch.tensor([2], dtype=torch.long),
                },
                batch_size=[1],
            ),
            "impropers": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),  # center
                    "atomj": torch.tensor([1], dtype=torch.long),
                    "atomk": torch.tensor([2], dtype=torch.long),
                    "atoml": torch.tensor([3], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )
    return batch
