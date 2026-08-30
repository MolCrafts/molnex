"""Shared fixtures for the ``molix.engine`` unit tests.

Migrated verbatim from ``tests/test_molix/test_engine.py`` when that module was
split into this mirrored package (spec ``lammps-iface-01-meta``); the stub
potentials were extended for schema v3 (spec ``lammps-iface-05-pbc``), which
makes the per-atom energy an output every exported model must produce.
Everything here has more than one consumer across the sibling modules:

- :func:`pair_site_energies_and_forces` — the single stub "physics" shared by
  both potentials below and by ``test_static.py``'s recording potential.
- :class:`_DummyTDPotential` — ``test_adapter.py`` (adapter + forward shapes +
  ``local_mask``) and ``test_static.py`` reuse the same "nested TensorDict in,
  ``{energy, forces, atomic_energy}`` out" stub shape.
- :class:`_FlatEFPotential` — ``test_adapter.py`` (flat adapter) and
  ``test_export.py`` (every export-validation case).
- :func:`graph` — the 3-atom H/C/O cluster used by the adapter cases.

Two properties of the stubs are load-bearing for the v3 cases and are therefore
built in on purpose, not by luck:

1. **Decomposability.** The scalar ``energy`` is *literally*
   ``atomic_energy.sum()``, so the export-time consistency check
   (``|energy - Σ atomic_energy|`` within rel 1e-6) can never trip on a stub.
   A model that violates it is written locally in the test that needs it.
2. **Genuine cross-site coupling.** The pair term ``|r_j - r_i|²`` is split
   evenly over its two endpoints, so a site's energy depends on its
   neighbours' coordinates. Masking a site *out* of the energy therefore still
   leaves a non-zero force on it — the ghost-force property D2 makes the
   ``TestLocalMask`` cases assert. A stub with zero (or per-site independent)
   energies would make that assertion vacuous.

Units: the stubs are unitless; ``energy`` is in whatever unit ``pos`` implies.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

#: One spelling, one home: the canonical flat per-atom-energy key (ac-012).
from molix.schema import ATOMIC_ENERGY_FLAT_KEY as _ATOMIC_ENERGY_FLAT_KEY


def pair_site_energies_and_forces(
    pos: torch.Tensor, edge_index: torch.Tensor, edge_diff: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decomposable pair stub: ``E_e = |d_e|²`` split evenly over both endpoints.

    Args:
        pos: Positions ``(N, 3)``.
        edge_index: Source-target pairs ``(E, 2)``; ``[:, 0]`` source,
            ``[:, 1]`` target (CLAUDE.md edge convention).
        edge_diff: Edge vectors ``(E, 3)`` ``= pos[target] - pos[source]``.
            Taken as an argument rather than recomputed so the caller's
            padded/neutralised edges (``StaticForward``) are honoured.

    Returns:
        ``(atomic_energy (N,), forces (N, 3))``. ``atomic_energy.sum()`` is the
        total energy; ``forces`` are the closed-form ``-∂E_total/∂pos`` of that
        **unmasked** total (analytic, so no autograd graph is consumed and the
        caller's own differentiation of the site energies stays possible).
    """
    src, tgt = edge_index[:, 0], edge_index[:, 1]
    half = 0.5 * (edge_diff * edge_diff).sum(-1)
    zeros = torch.zeros(pos.shape[0], dtype=edge_diff.dtype, device=edge_diff.device)
    atomic = zeros.index_add(0, src, half).index_add(0, tgt, half)
    grad = 2.0 * edge_diff  # ∂E_total/∂pos[target] for one edge
    forces = torch.zeros_like(pos).index_add(0, src, grad).index_add(0, tgt, -grad)
    return atomic, forces


class _DummyTDPotential(nn.Module):
    """Mimics a molnex potential: nested-TensorDict in, flat output dict out.

    Returns ``{energy, forces, atomic_energy}`` — the plain-dict (flat) output
    shape, so the per-atom energy is read under the flat spelling. Consumes the
    ``edge_diff`` the adapter built (as PiNet does), which keeps the site
    energies differentiable with respect to the caller's ``pos``.
    """

    def forward(self, batch):
        # Monomorphic forward, mirroring a real potential since b85d12f: a
        # potential that derives forces was built that way and always writes them.
        pos = batch["atoms", "pos"]
        atomic, forces = pair_site_energies_and_forces(
            pos, batch["edges", "edge_index"], batch["edges", "edge_diff"]
        )
        return {
            "energy": atomic.sum().reshape(1),
            "forces": forces,
            _ATOMIC_ENERGY_FLAT_KEY: atomic,
        }


class _FlatEFPotential(nn.Module):
    """Exportable flat-convention potential with analytic forces (no functorch).

    ``forward(Z, pos, edge_index) -> (energy, forces, atomic_energy)`` — the
    schema-v3 triple :class:`~molix.engine.adapter.FlatTensorAdapter` requires
    of a third-party model.
    """

    def forward(self, Z, pos, edge_index):
        edge_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
        atomic, forces = pair_site_energies_and_forces(pos, edge_index, edge_diff)
        return atomic.sum().reshape(1), forces, atomic


@pytest.fixture
def graph():
    Z = torch.tensor([1, 6, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.2, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
    return Z, pos, edge_index
