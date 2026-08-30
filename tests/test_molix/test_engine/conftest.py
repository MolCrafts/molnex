"""Shared fixtures for the ``molix.engine`` unit tests.

Migrated verbatim from ``tests/test_molix/test_engine.py`` when that module was
split into this mirrored package (spec ``lammps-iface-01-meta``). Everything
here has more than one consumer across the sibling modules:

- :class:`_DummyTDPotential` — ``test_adapter.py`` (adapter + forward shapes)
  and ``test_static.py`` reuse the same "nested TensorDict in, energy/forces
  out" stub shape.
- :class:`_FlatEFPotential` — ``test_adapter.py`` (flat adapter) and
  ``test_export.py`` (every export-validation case).
- :func:`graph` — the 3-atom H/C/O cluster used by the adapter cases.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class _DummyTDPotential(nn.Module):
    """Mimics a molnex potential: nested-TensorDict in, ``{energy, forces}`` out."""

    def forward(self, batch):
        # Monomorphic forward, mirroring a real potential since b85d12f: a
        # potential that derives forces was built that way and always writes them.
        pos = batch["atoms", "pos"]
        bd = batch["edges", "edge_dist"]
        return {
            "energy": (bd**2).sum().reshape(1),
            "atomic_energy": torch.zeros(pos.shape[0]),
            "forces": torch.zeros_like(pos),
        }


class _FlatEFPotential(nn.Module):
    """Exportable flat-convention potential with analytic forces (no functorch)."""

    def forward(self, Z, pos, edge_index):
        src, tgt = edge_index[:, 0], edge_index[:, 1]
        d = pos[tgt] - pos[src]
        r = d.norm(dim=-1).clamp(min=1e-6)
        e = ((r - 1.0) ** 2).sum().reshape(1)
        g = (2 * (r - 1.0) / r).unsqueeze(-1) * d
        f = torch.zeros_like(pos)
        f.index_add_(0, src, g)
        f.index_add_(0, tgt, -g)
        return e, f


@pytest.fixture
def graph():
    Z = torch.tensor([1, 6, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.2, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
    return Z, pos, edge_index
