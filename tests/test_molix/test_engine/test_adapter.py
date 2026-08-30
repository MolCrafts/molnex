"""Tests for ``molix.engine.adapter`` — registry + flat calling convention.

The adapter layer is the Python-side shim that maps the flat LAMMPS convention
``(Z, pos, edge_index) -> (energy, forces)`` onto a model's real input/output
shapes. Covered here: the self-registration registry, the molnex nested
``TensorDict`` construction (edge convention!), and the ``EngineForward``
output shapes for both built-in adapters. The C++ pair style itself is
exercised by an actual LAMMPS run (see ``interface/lammps/README.md``).

-- migrated verbatim from ``tests/test_molix/test_engine.py`` --
"""

from __future__ import annotations

import pytest
import torch

from molix.engine import (
    EngineAdapter,
    EngineForward,
    FlatTensorAdapter,
    MolnexTensorDictAdapter,
)
from tests.test_molix.test_engine.conftest import _DummyTDPotential, _FlatEFPotential


def test_registry_has_builtin_adapters():
    assert "molnex-tensordict" in EngineAdapter.names()
    assert "flat" in EngineAdapter.names()
    assert isinstance(EngineAdapter.from_name("flat"), FlatTensorAdapter)


def test_unknown_adapter_raises():
    with pytest.raises(ValueError, match="unknown adapter"):
        EngineAdapter.from_name("does-not-exist")


def test_tensordict_adapter_edge_convention(graph):
    """edge_diff = pos[target] - pos[source]; single-graph batch metadata."""
    Z, pos, edge_index = graph
    batch = MolnexTensorDictAdapter().build_inputs(_DummyTDPotential(), Z, pos, edge_index)
    expected = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    assert torch.allclose(batch["edges", "edge_diff"], expected)
    assert torch.allclose(batch["edges", "edge_dist"], expected.norm(dim=-1).clamp(min=1e-6))
    assert (batch["atoms", "batch"] == 0).all()
    assert batch["graphs", "num_atoms"].item() == 3


def test_lammps_forward_shapes_tensordict(graph):
    Z, pos, edge_index = graph
    e, f = EngineForward(_DummyTDPotential(), "molnex-tensordict")(Z, pos, edge_index)
    assert e.shape == ()  # scalar total energy
    assert f.shape == (3, 3)


def test_lammps_forward_shapes_flat(graph):
    Z, pos, edge_index = graph
    e, f = EngineForward(_FlatEFPotential(), "flat")(Z, pos, edge_index)
    assert e.shape == ()
    assert f.shape == (3, 3)
