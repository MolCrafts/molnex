"""Shared fixtures for the ``molix.deploy`` unit tests.

- :func:`pair_atom_energies_and_forces` — the stub pair term.
- :class:`_DummyTDPotential` — nested TensorDict in / out, used by
  :class:`molix.deploy.AtomEnergyAdapter`.
- :func:`graph` — the 3-atom H/C/O cluster.

Two properties of the stub are load-bearing:

1. **Decomposability.** The scalar ``energy`` is ``atom_energy.sum()``, so
   the export-time check (``|energy - Σ atom_energy|`` within rel 1e-6)
   does not trip on this stub. A model that violates it is written in the
   test that needs the refusal.
2. **Cross-atom coupling.** The pair term ``|r_j - r_i|²`` is split evenly
   over its two endpoints, so masking an atom out of the energy still leaves a
   force on it.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, FORCES_KEY


def pair_atom_energies_and_forces(
    pos: torch.Tensor, pair_index: torch.Tensor, pair_diff: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decomposable pair stub: ``E_e = |d_e|²`` split evenly over both endpoints.

    Args:
        pos: Positions ``(N, 3)``.
        pair_index: Source-target pairs ``(E, 2)``. ``[:, 0]`` source,
            ``[:, 1]`` target.
        pair_diff: Edge vectors ``(E, 3)`` ``= pos[target] - pos[source]``.

    Returns:
        ``(atom_energy (N,), forces (N, 3))``. ``atom_energy.sum()`` is the
        total energy. ``forces`` are ``-∂E_total/∂pos`` of that unmasked total,
        computed in closed form so the caller's own differentiation of the
        atom energies stays possible.
    """
    src, tgt = pair_index[:, 0], pair_index[:, 1]
    half = 0.5 * (pair_diff * pair_diff).sum(-1)
    zeros = torch.zeros(pos.shape[0], dtype=pair_diff.dtype, device=pair_diff.device)
    atomic = zeros.index_add(0, src, half).index_add(0, tgt, half)
    grad = 2.0 * pair_diff
    forces = torch.zeros_like(pos).index_add(0, src, grad).index_add(0, tgt, -grad)
    return atomic, forces


class _DummyTDPotential(nn.Module):
    """Nested TensorDict in and out, mimicking a molnex potential.

    Writes ``graphs.energy``, ``atoms.forces``, and ``atoms.energy``.
    """

    def forward(self, batch):
        pos = batch["atoms", "pos"]
        atomic, forces = pair_atom_energies_and_forces(
            pos, batch["pairs", "pair_index"], batch["pairs", "pair_diff"]
        )
        batch[ENERGY_KEY] = atomic.sum().reshape(1)
        batch[FORCES_KEY] = forces
        batch[ATOMIC_ENERGY_KEY] = atomic
        return batch


@pytest.fixture
def graph():
    Z = torch.tensor([1, 6, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.2, 0.0]])
    pair_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
    return Z, pos, pair_index
