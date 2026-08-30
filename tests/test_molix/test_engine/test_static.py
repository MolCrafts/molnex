"""Tests for ``molix.engine.static.StaticForward`` — fixed-shape flat forward.

``StaticForward`` fixes ``N`` atoms / ``E_max`` edges so the AOTI ``.pt2`` is
CUDA-graph capturable, and neutralises padded edges by pushing their
``edge_diff`` past the cutoff instead of dropping them (which would change the
shape). Both concerns are pure tensor plumbing around a wrapped model, so a
recording stub model is enough — no export, no CUDA.

Schema v3 (spec ``lammps-iface-05-pbc``) gives this path the same convention as
the dynamic one, with the edge-validity mask renamed so the two masks can never
be mis-wired:

    (Z, pos, edge_index, edge_mask, local_mask) -> (energy, forces, atomic_energy)

``edge_mask (E_max,)`` marks live *edges*; ``local_mask (N,)`` marks the
*atoms* this rank owns, and the returned energy is
``(atomic_energy * local_mask).sum()``.

Units: the stub potential is unitless; energies/forces below are hard-coded
literals derived by hand from its documented pair term (see ``conftest.py``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.engine.static import StaticForward
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, FORCES_KEY
from molix.units import DEAD_EDGE_CUTOFF_FACTOR
from tests.test_molix.test_engine.conftest import pair_site_energies_and_forces


class _RecordingTDPotential(nn.Module):
    """Writes the v3 energy peers into the batch and keeps the distances it saw.

    Mirrors a real molnex potential's contract (nested ``TensorDict`` in,
    ``ENERGY_KEY`` / ``ATOMIC_ENERGY_KEY`` / ``FORCES_KEY`` written back) with
    the shared decomposable pair stub, so the test observes exactly what
    :class:`StaticForward` handed the model and the site energies still sum to
    the scalar energy.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen_edge_dist: torch.Tensor | None = None

    def forward(self, batch: TensorDict) -> TensorDict:
        pos = batch["atoms", "pos"]
        self.seen_edge_dist = batch["edges", "edge_dist"].detach().clone()
        atomic, forces = pair_site_energies_and_forces(
            pos, batch["edges", "edge_index"], batch["edges", "edge_diff"]
        )
        batch[ENERGY_KEY] = atomic.sum().reshape(1)
        batch[ATOMIC_ENERGY_KEY] = atomic
        batch[FORCES_KEY] = forces
        return batch


class TestStaticForward:
    """Fixed ``(N, E_max)`` forward: the five-input / three-output v3 convention."""

    N_ATOMS = 3
    E_MAX = 4
    CUTOFF = 4.5

    @staticmethod
    def inputs() -> tuple[torch.Tensor, ...]:
        """Two live edges of the 3-atom chain, padded to ``E_MAX`` with dead ones."""
        Z = torch.tensor([1, 6, 8])
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=torch.float64)
        # rows 2/3 are padding: they point at real atoms, only the mask says dead.
        edge_index = torch.tensor([[0, 1], [1, 2], [0, 0], [0, 0]])
        edge_mask = torch.tensor([True, True, False, False])
        local_mask = torch.ones(3, dtype=torch.bool)
        return Z, pos, edge_index, edge_mask, local_mask

    def test_forward_returns_scalar_energy_per_atom_energy_and_forces(self):
        model = _RecordingTDPotential()
        Z, pos, edge_index, edge_mask, local_mask = self.inputs()

        energy, forces, atomic = StaticForward(model, self.N_ATOMS, self.E_MAX, self.CUTOFF)(
            Z, pos, edge_index, edge_mask, local_mask
        )

        assert energy.shape == ()
        assert forces.shape == (self.N_ATOMS, 3)
        assert atomic.shape == (self.N_ATOMS,)

    def test_masked_edges_are_displaced_past_the_cutoff(self):
        """Dead edges get ``edge_diff`` of ``factor * cutoff`` per component, so
        every cutoff envelope evaluates to zero; live edges keep their geometry."""
        model = _RecordingTDPotential()
        Z, pos, edge_index, edge_mask, local_mask = self.inputs()

        StaticForward(model, self.N_ATOMS, self.E_MAX, self.CUTOFF)(
            Z, pos, edge_index, edge_mask, local_mask
        )

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


class TestEdgeDistanceClamp:
    """Target: ``StaticForward.forward`` small-``r`` regularisation.

    This path used to clamp ``edge_dist`` at 1e-8 while the adapter path clamped
    at 1e-6, although the module docstrings claim the two are equivalent (spec
    ``lammps-iface-05-pbc``, "顺带修复二"). Both must now consume one named
    constant in ``molix.schema``; the counterpart assertion for the adapter path
    lives in ``test_adapter.py::TestEdgeDistanceClamp``, so pinning each path to
    the same constant pins them to each other without either test module
    reaching into the other's production module. The constant's own value
    (1e-6) is pinned where it is declared, ``tests/test_molix/test_schema.py``.

    Note this is a **sub-threshold behaviour change**, not a pure refactor:
    distances in [1e-8, 1e-6) now clamp where they previously did not. Two
    atoms within 0.01 pm is physically degenerate, so the affected band is
    empty in practice but not empty in principle.
    """

    def test_coincident_atoms_clamp_to_the_shared_constant(self):
        from molix.schema import EDGE_DIST_CLAMP_MIN

        model = _RecordingTDPotential()
        Z = torch.tensor([1, 1])
        pos = torch.zeros(2, 3, dtype=torch.float64)  # coincident → |edge_diff| == 0
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_mask = torch.tensor([True, True])
        local_mask = torch.ones(2, dtype=torch.bool)

        StaticForward(model, 2, 2, 4.5)(Z, pos, edge_index, edge_mask, local_mask)

        seen = model.seen_edge_dist
        assert seen is not None
        assert seen.tolist() == [EDGE_DIST_CLAMP_MIN, EDGE_DIST_CLAMP_MIN]


class TestLocalMask:
    """Target: ``StaticForward.forward`` — the ``local_mask`` partition (D1/D2).

    Same geometry and hand-derived literals as
    ``test_adapter.py::TestLocalMask``, with every edge live so the padding
    machinery stays out of the way: 3 atoms at ``(0,0,0)``, ``(1,0,0)``,
    ``(0,2,0)``, bidirectional edges ``(0,1) (1,0) (1,2) (2,1)``, pair term
    ``|d_e|²`` split evenly over the endpoints ⇒

        atomic_energy = [1, 6, 5]  (total 12)

    The partial- and zero-mask cases only pass if the returned forces are
    ``-∂[(atomic_energy · local_mask).sum()]/∂pos``: a ghost site owns no site
    energy on this rank yet must still take the cross-term force (D2), and a
    rank owning nothing must produce none at all.
    """

    N_ATOMS = 3
    E_MAX = 4
    CUTOFF = 4.5

    @staticmethod
    def run(local_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Z = torch.tensor([1, 6, 8])
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
        edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
        edge_mask = torch.ones(4, dtype=torch.bool)
        cls = TestLocalMask
        forward = StaticForward(_RecordingTDPotential(), cls.N_ATOMS, cls.E_MAX, cls.CUTOFF)
        return forward(Z, pos, edge_index, edge_mask, local_mask)

    def test_all_ones_mask_sums_every_site(self):
        energy, _, atomic = self.run(torch.ones(3, dtype=torch.bool))
        assert energy.item() == pytest.approx(12.0, abs=1e-10)
        assert torch.allclose(
            atomic, torch.tensor([1.0, 6.0, 5.0], dtype=torch.float64), atol=1e-10, rtol=0.0
        )

    def test_partial_mask_sums_only_the_masked_subset(self):
        energy, _, _ = self.run(torch.tensor([True, True, False]))
        assert energy.item() == pytest.approx(7.0, abs=1e-10)  # 1 + 6

    def test_partial_mask_leaves_a_force_on_the_masked_out_site(self):
        """D2: E_m = 2·D01 + D12 ⇒ F = [(4,0,0), (-6,4,0), (2,-4,0)]."""
        _, forces, _ = self.run(torch.tensor([True, True, False]))
        assert forces[2].abs().max().item() > 1e-8  # ghost cross-term, not zero
        expected = torch.tensor(
            [[4.0, 0.0, 0.0], [-6.0, 4.0, 0.0], [2.0, -4.0, 0.0]], dtype=torch.float64
        )
        assert torch.allclose(forces, expected, atol=1e-8, rtol=0.0)

    def test_all_zero_mask_yields_zero_energy_and_zero_forces(self):
        energy, forces, _ = self.run(torch.zeros(3, dtype=torch.bool))
        assert energy.item() == pytest.approx(0.0, abs=1e-10)
        assert forces.abs().max().item() == pytest.approx(0.0, abs=1e-8)
