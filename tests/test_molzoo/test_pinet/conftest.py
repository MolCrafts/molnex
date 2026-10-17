"""Shared fixtures for the PiNet2 tests: a tiny fp64 cluster and model sizes."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix import config
from tests.conftest import make_graph_batch

#: A small but complete architecture; tests override ``rank`` / ``depth``.
TINY = {
    "atomic_numbers": [1, 6, 7, 8],
    "r_max": 4.0,
    "n_basis": 3,
    "pp_nodes": [8, 8],
    "pi_nodes": [8, 8],
    "ii_nodes": [8, 8],
    "depth": 2,
}

#: Fully connected 4-atom pair list, ``[:, 0]`` source.
PAIRS = [[i, j] for i in range(4) for j in range(4) if i != j]


@pytest.fixture(autouse=True)
def fp64():
    """Build every model at fp64 so invariance and finite-difference checks are tight."""
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


@pytest.fixture
def cluster() -> TensorDict:
    """One neutral 4-atom molecule (H, C, N, O) in Å, fully connected."""
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.3, 1.2, 0.2], [1.4, 1.1, -0.1]],
        dtype=torch.float64,
    )
    return make_graph_batch(
        pos,
        torch.tensor([1, 6, 7, 8]),
        torch.tensor(PAIRS),
        torch.zeros(4, dtype=torch.long),
        graphs={"total_charge": torch.zeros(1, dtype=torch.float64)},
    )
