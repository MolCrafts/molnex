"""Shared fixtures for the Allegro tests (encoder here, products in test_molzoo)."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from tests.conftest import make_graph_batch

#: A small mainline-shaped encoder (two layers, l_max 2, parity on).
TINY = {
    "atomic_numbers": [1, 6, 8],
    "r_max": 4.0,
    "num_bessels": 4,
    "num_layers": 2,
    "l_max": 2,
    "num_scalar_features": 8,
    "num_tensor_features": 4,
    "scalar_embed_mlp_hidden_layers_width": 8,
    "allegro_mlp_hidden_layers_width": 8,
    "avg_num_neighbors": 3.0,
}

#: Build dtype of every model: the equivariant operators fix their working
#: precision at construction, so fp64 tolerances need an fp64 build.
DTYPE = torch.float64


@pytest.fixture
def cluster() -> TensorDict:
    """One 5-atom molecule (C, H, H, O, H) in Å, fully connected within 4 Å."""
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [-0.4, 0.9, 0.2], [0.2, -0.3, 1.3], [0.9, -0.6, 1.8]],
        dtype=torch.float64,
    )
    pairs = [[i, j] for i in range(5) for j in range(5) if i != j]
    return make_graph_batch(
        pos, torch.tensor([6, 1, 1, 8, 1]), torch.tensor(pairs), torch.zeros(5, dtype=torch.long)
    )
