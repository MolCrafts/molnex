"""Shared fixtures for the Allegro tests: a tiny fp64 cluster and model sizes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from tests.conftest import make_graph_batch

#: A small mainline-shaped architecture (two layers, l_max 2, parity on).
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
    "readout_mlp_hidden_layers_width": 8,
    "avg_num_neighbors": 3.0,
}

#: Directory with the official packages and upstream goldens
#: (``allegro_small_default.nequip.zip`` + ``goldens.pt``, … — see
#: ``src/molzoo/specs/allegro.md`` §7.3).
WEIGHTS_DIR = Path(os.environ.get("MOLNEX_ALLEGRO_WEIGHTS_DIR", "/nonexistent"))


@pytest.fixture(autouse=True)
def _build_at_fp64(fp64):
    """Every Allegro test builds at fp64 (``tests.conftest.fp64``)."""


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
