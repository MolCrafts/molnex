"""Tests for ``molix.engine.static.StaticForward`` — fixed-shape flat forward.

``StaticForward`` fixes ``N`` atoms / ``E_max`` edges so the AOTI ``.pt2`` is
CUDA-graph capturable, and neutralises padded edges by pushing their
``edge_diff`` past the cutoff instead of dropping them (which would change the
shape). Both concerns are pure tensor plumbing around a wrapped model, so a
recording stub model is enough — no export, no CUDA.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.engine.static import StaticForward
from molix.schema import ENERGY_KEY, FORCES_KEY
from molix.units import DEAD_EDGE_CUTOFF_FACTOR


class _RecordingTDPotential(nn.Module):
    """Writes energy/forces into the batch and keeps the edge distances it saw.

    Mirrors a real molnex potential's contract (nested ``TensorDict`` in,
    ``ENERGY_KEY`` / ``FORCES_KEY`` written back) without any physics, so the
    test observes exactly what :class:`StaticForward` handed the model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen_edge_dist: torch.Tensor | None = None

    def forward(self, batch: TensorDict) -> TensorDict:
        pos = batch["atoms", "pos"]
        edge_dist = batch["edges", "edge_dist"]
        self.seen_edge_dist = edge_dist.detach().clone()
        batch[ENERGY_KEY] = edge_dist.sum().reshape(1)
        batch[FORCES_KEY] = torch.zeros_like(pos)
        return batch


class TestStaticForward:
    """Fixed ``(N, E_max)`` forward: ``(Z, pos, edge_index, mask) -> (energy, forces)``."""

    N_ATOMS = 3
    E_MAX = 4
    CUTOFF = 4.5

    @staticmethod
    def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two live edges of the 3-atom chain, padded to ``E_MAX`` with dead ones."""
        Z = torch.tensor([1, 6, 8])
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=torch.float64)
        # rows 2/3 are padding: they point at real atoms, only the mask says dead.
        edge_index = torch.tensor([[0, 1], [1, 2], [0, 0], [0, 0]])
        mask = torch.tensor([True, True, False, False])
        return Z, pos, edge_index, mask

    def test_forward_returns_scalar_energy_and_per_atom_forces(self):
        model = _RecordingTDPotential()
        Z, pos, edge_index, mask = self.inputs()

        energy, forces = StaticForward(model, self.N_ATOMS, self.E_MAX, self.CUTOFF)(
            Z, pos, edge_index, mask
        )

        assert energy.shape == ()
        assert forces.shape == (self.N_ATOMS, 3)

    def test_masked_edges_are_displaced_past_the_cutoff(self):
        """Dead edges get ``edge_diff`` of ``factor * cutoff`` per component, so
        every cutoff envelope evaluates to zero; live edges keep their geometry."""
        model = _RecordingTDPotential()
        Z, pos, edge_index, mask = self.inputs()

        StaticForward(model, self.N_ATOMS, self.E_MAX, self.CUTOFF)(Z, pos, edge_index, mask)

        seen = model.seen_edge_dist
        assert seen is not None
        assert seen.shape == (self.E_MAX,)
        # live edges: |pos[1] - pos[0]| = 1, |pos[2] - pos[1]| = 2
        assert seen[0].item() == pytest.approx(1.0, abs=1e-12)
        assert seen[1].item() == pytest.approx(2.0, abs=1e-12)
        # dead edges: the (pad, pad, pad) vector, i.e. sqrt(3) * factor * cutoff
        dead = 3.0**0.5 * DEAD_EDGE_CUTOFF_FACTOR * self.CUTOFF
        assert seen[2].item() == pytest.approx(dead, rel=1e-12)
        assert seen[3].item() == pytest.approx(dead, rel=1e-12)
        assert seen[2].item() > self.CUTOFF
