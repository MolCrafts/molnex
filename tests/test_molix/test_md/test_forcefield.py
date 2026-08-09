"""Tests for molix.md.forcefield — binding models to systems."""

import pytest
import torch
from tensordict import TensorDict

from molix.md import (
    CallableForceField,
    ForceField,
    ForceOutput,
    HarmonicForceField,
    LennardJonesCutForceField,
    LennardJonesForceField,
    NeighborList,
    PeriodicPotentialForceField,
    PotentialForceField,
)
from tests.test_molix.test_md.conftest import (
    make_cubic_lattice,
    make_pinet_template,
    make_tiny_potential,
)


class _RecordingNeighbors:
    """Neighbour-strategy stub recording *which* entry point a force field used.

    The discriminator for the seam's semantic shift: ``rebuild_neighbors``
    means "run the list's policy" (:meth:`update`), never "build now"
    (:meth:`rebuild`, which stays available as the explicit escape hatch).
    """

    edge_index = torch.zeros(1, 2, dtype=torch.long)
    shifts = torch.zeros(1, 3)
    num_edges = 0
    capacity = 1
    cutoff = 5.0
    skin = 0.0

    def __init__(self) -> None:
        self.rebuilds: list[torch.Tensor] = []
        self.updates: list[torch.Tensor] = []

    def rebuild(self, positions: torch.Tensor) -> None:
        self.rebuilds.append(positions.detach().clone())

    def update(self, positions: torch.Tensor) -> bool:
        self.updates.append(positions.detach().clone())
        return False

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> "_RecordingNeighbors":
        return self


class TestPotentialForceField:
    """The molpot-potential adapter over a fixed open-system template."""

    def test_force_seam_tracks_live_geometry(self):
        """PotentialForceField must recompute edge geometry from the live
        positions, not a frozen template ``edge_diff`` — regression for the
        constant-PES bug where swapping only ``pos`` left energy/force pinned
        to the initial geometry."""
        template = make_pinet_template()
        ref = make_tiny_potential()
        ref(template.clone())  # warmup lazy params
        ff = PotentialForceField(ref, template)
        pos0 = template["atoms", "pos"]
        out0 = ff(pos0)
        torch.manual_seed(1)
        pos1 = pos0 + 0.3 * torch.randn_like(pos0)  # non-rigid displacement
        out1 = ff(pos1)
        assert (out1.energy - out0.energy).abs().item() > 1e-6, "energy frozen at initial geometry"
        assert (out1.forces - out0.forces).abs().max().item() > 1e-6, (
            "forces frozen at initial geometry"
        )

    def test_to_dtype_casts_the_bound_system(self):
        """``.to(float64)`` must reach the working batch, not just parameters —
        the silent-no-op ``_apply`` gap was a review finding."""
        template = make_pinet_template()
        potential = make_tiny_potential()
        potential(template.clone())  # warmup lazy params
        ff = PotentialForceField(potential, template).to(torch.float64)
        assert ff._dtype == torch.float64
        assert ff._work["atoms", "pos"].dtype == torch.float64
        out = ff(template["atoms", "pos"].to(torch.float64))
        assert out.energy.dtype == torch.float64
        assert out.forces.dtype == torch.float64

    def test_accepts_positions_in_any_dtype(self):
        """The force field owns its precision: an fp64 trajectory may drive an
        fp32 model, and the output stays in the model's dtype (the integrator
        casts at the boundary)."""
        template = make_pinet_template()
        potential = make_tiny_potential()
        potential(template.clone())
        ff = PotentialForceField(potential, template)  # fp32 model
        out = ff(template["atoms", "pos"].to(torch.float64))
        assert out.forces.dtype == torch.float32


class TestCallableForceField:
    """The escape hatch for non-TensorDict force providers."""

    def test_wraps_a_plain_tuple_callable(self):
        ff = CallableForceField(lambda pos: ((pos * pos).sum(), -2.0 * pos))
        pos = torch.randn(5, 3)
        out = ff(pos)
        assert isinstance(out, ForceOutput)
        assert torch.equal(out.forces, -2.0 * pos)

    def test_applies_the_energy_scale(self):
        ff = CallableForceField(
            lambda pos: (torch.tensor(2.0), torch.ones_like(pos)), energy_scale=0.5
        )
        out = ff(torch.zeros(3, 3))
        assert float(out.energy) == 1.0
        assert torch.equal(out.forces, 0.5 * torch.ones(3, 3))

    def test_delegates_the_policy_to_the_neighbor_list(self):
        """The bound list's *policy* runs (``update``), not a forced ``rebuild``.

        The list owns the cadence (skin / every / delay / check); the force
        field only passes on the positions being evaluated and lets it decline.
        """
        recorder = _RecordingNeighbors()
        ff = CallableForceField(lambda pos: (pos.sum(), pos), neighbors=recorder)
        pos = torch.zeros(2, 3)
        ff.rebuild_neighbors(pos)
        assert len(recorder.updates) == 1
        assert torch.equal(recorder.updates[0], pos)
        assert recorder.rebuilds == []

    def test_without_neighbors_rebuild_is_a_noop(self):
        ff = CallableForceField(lambda pos: (pos.sum(), pos))
        ff.rebuild_neighbors(torch.zeros(2, 3))  # must not raise


class TestPotentialContract:
    """A potential that writes no forces must fail loudly, not integrate garbage."""

    def test_missing_forces_raises(self):
        template = make_pinet_template()

        class _EnergyOnly(torch.nn.Module):
            def forward(self, td):
                td["graphs", "energy"] = torch.zeros(1)
                return td

        ff = PotentialForceField(_EnergyOnly(), template)
        with pytest.raises(RuntimeError, match="wrote no"):
            ff(template["atoms", "pos"])


class _ShiftAwarePairPotential(torch.nn.Module):
    """Pair potential reading ``edges.shifts`` — envelope exactly zero beyond cutoff."""

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, td: TensorDict) -> TensorDict:
        pos = td["atoms", "pos"]
        ei = td["edges", "edge_index"]
        shifts = td["edges", "shifts"]
        leaf = pos.detach().requires_grad_(True)
        with torch.enable_grad():
            vec = leaf[ei[:, 1]] - leaf[ei[:, 0]] + shifts
            r = torch.linalg.norm(vec, dim=-1)
            pair = torch.where(r < self.cutoff, (r - self.cutoff) ** 2, torch.zeros_like(r))
            energy = pair.sum()
        (grad,) = torch.autograd.grad(energy, leaf)
        td["graphs", "energy"] = energy.detach().reshape(1)
        td["atoms", "forces"] = -grad
        return td


def _periodic_template(pos: torch.Tensor) -> TensorDict:
    n = pos.shape[0]
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.ones(n, dtype=torch.long),
                    "batch": torch.zeros(n, dtype=torch.long),
                },
                batch_size=[n],
            )
        },
        batch_size=[],
    )


class TestPeriodicPotentialForceField:
    """The joined periodic component: potential + rebuilding fixed-capacity list."""

    def test_dead_edge_padding_is_invisible(self):
        """Different capacity factors (more padding) must give identical physics."""
        pos, cell = make_cubic_lattice()
        outs = []
        for factor in (1.1, 3.0):
            nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, capacity_factor=factor)
            ff = PeriodicPotentialForceField(
                _ShiftAwarePairPotential(3.5), _periodic_template(pos), neighbors=nl
            )
            outs.append(ff(pos))
        assert torch.equal(outs[0].energy, outs[1].energy)
        assert torch.equal(outs[0].forces, outs[1].forces)

    def test_rebuild_is_visible_through_the_bound_buffers(self):
        """The working batch holds the list's buffers by reference, so an
        in-place rebuild changes the energy without any re-binding.

        Compression (not dilation) is the discriminating displacement: with an
        envelope that is exactly zero beyond the cutoff, a stale *superset*
        list gives the same energy — only pairs *entering* the cutoff that the
        stale list has never seen can differ.
        """
        pos, cell = make_cubic_lattice()
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, capacity_factor=8.0)
        ff = PeriodicPotentialForceField(
            _ShiftAwarePairPotential(3.5), _periodic_template(pos), neighbors=nl
        )
        compressed = pos * 0.8  # second-neighbour pairs enter the cutoff
        stale = ff(compressed)  # list still from the original positions
        ff.rebuild_neighbors(compressed)
        fresh = ff(compressed)
        assert nl.rebuild_count == 1
        assert not torch.equal(stale.energy, fresh.energy)

    def test_a_cast_keeps_the_rebuild_visible(self):
        """``.to(dtype)`` must leave the by-reference tie alive.

        ``TensorDict.apply`` in the parent ``_apply`` produces *new* leaf
        tensors and ``NeighborList.to`` rebinds ``shifts`` to a new tensor, so a
        cast severs the tie twice over unless the owner re-binds afterwards — a
        later rebuild would then update buffers the potential no longer sees,
        and the PES would freeze silently.

        Asserted through the public energy rather than through ``_work`` on
        purpose: this is the property a user has, and it must survive the
        ownership move of the binding knowledge onto ``NeighborList.build``.
        """
        pos, cell = make_cubic_lattice()
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, capacity_factor=8.0)
        ff = PeriodicPotentialForceField(
            _ShiftAwarePairPotential(3.5), _periodic_template(pos), neighbors=nl
        ).to(torch.float32)
        compressed = (pos * 0.8).to(torch.float32)
        stale = ff(compressed)
        ff.rebuild_neighbors(compressed)
        fresh = ff(compressed)
        assert nl.rebuild_count == 1
        assert not torch.equal(stale.energy, fresh.energy)


class TestHarmonicForceField:
    def test_energy_and_force_are_consistent(self):
        pos = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
        out = HarmonicForceField(k=2.0).to(torch.float64)(pos)
        (ref,) = torch.autograd.grad(out.energy, pos)
        assert torch.allclose(out.forces, -ref)


class TestLennardJonesCutForceField:
    """Periodic truncated-shifted LJ over the fixed-capacity neighbour list."""

    _EPS, _SIGMA = 0.7, 1.1

    def _dimer(self, d: float, *, box: float = 20.0, cutoff: float = 5.0, shift: bool = True):
        pos = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * box
        nl = NeighborList(cell=cell, cutoff=cutoff, positions=pos)
        ff = LennardJonesCutForceField(
            epsilon=self._EPS, sigma=self._SIGMA, neighbors=nl, shift=shift
        ).to(torch.float64)
        return ff, pos

    def test_force_matches_autograd(self):
        """Closed-form forces must equal -dE/dpos through the mask and the shifts."""
        pos, cell = make_cubic_lattice()
        torch.manual_seed(0)
        pos = pos + 0.3 * torch.randn_like(pos)
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos)
        ff = LennardJonesCutForceField(epsilon=0.7, sigma=2.5, neighbors=nl).to(torch.float64)
        leaf = pos.clone().requires_grad_(True)
        out = ff(leaf)
        (ref,) = torch.autograd.grad(out.energy, leaf)
        assert torch.allclose(out.forces, -ref, atol=1e-10), "lj/cut closed form != -dE/dx"

    def test_matches_all_pairs_in_the_open_limit(self):
        """A cutoff spanning the whole cluster + shift=False is the all-pairs LJ."""
        torch.manual_seed(1)
        pos = torch.randn(8, 3, dtype=torch.float64) * 1.5 + 15.0  # blob at box centre
        cell = torch.eye(3, dtype=torch.float64) * 30.0
        nl = NeighborList(cell=cell, cutoff=14.0, positions=pos)
        cut = LennardJonesCutForceField(epsilon=0.9, sigma=1.2, neighbors=nl, shift=False).to(
            torch.float64
        )
        ref = LennardJonesForceField(epsilon=0.9, sigma=1.2).to(torch.float64)
        out, expected = cut(pos), ref(pos)
        assert torch.allclose(out.energy, expected.energy, atol=1e-10)
        assert torch.allclose(out.forces, expected.forces, atol=1e-10)

    def test_shift_makes_energy_continuous_at_the_cutoff(self):
        """Shifted: E→0 continuously at r_cut; unshifted: E→E_lj(r_cut) (the step)."""
        r_cut = 3.0
        just_inside = r_cut - 1e-9
        shifted, pos = self._dimer(just_inside, cutoff=r_cut, shift=True)
        unshifted, _ = self._dimer(just_inside, cutoff=r_cut, shift=False)
        sr6 = (self._SIGMA / r_cut) ** 6
        e_at_cut = 4.0 * self._EPS * (sr6 * sr6 - sr6)
        assert abs(float(shifted(pos).energy)) < 1e-8
        assert abs(float(unshifted(pos).energy) - e_at_cut) < 1e-8

    def test_dead_edge_padding_is_invisible(self):
        """Different capacity factors (more padding) must give identical physics."""
        pos, cell = make_cubic_lattice()
        outs = []
        for factor in (1.1, 4.0):
            nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, capacity_factor=factor)
            ff = LennardJonesCutForceField(epsilon=0.8, sigma=2.5, neighbors=nl).to(torch.float64)
            outs.append(ff(pos))
        assert torch.equal(outs[0].energy, outs[1].energy)
        assert torch.equal(outs[0].forces, outs[1].forces)

    def test_minimum_image_across_the_boundary(self):
        """Atoms near opposite faces interact at the wrapped distance."""
        box, r_cut = 12.0, 3.0
        pos = torch.tensor([[0.6, 0.0, 0.0], [box - 0.6, 0.0, 0.0]], dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * box
        nl = NeighborList(cell=cell, cutoff=r_cut, positions=pos)
        ff = LennardJonesCutForceField(epsilon=self._EPS, sigma=self._SIGMA, neighbors=nl).to(
            torch.float64
        )
        open_ff, open_pos = self._dimer(1.2, cutoff=r_cut)
        assert torch.allclose(ff(pos).energy, open_ff(open_pos).energy, atol=1e-12)

    def test_rebuild_tracks_pair_departure(self):
        """A pair leaving the cutoff vanishes from the PES after a rebuild."""
        ff, pos = self._dimer(1.5, cutoff=3.0)
        assert float(ff(pos).energy) != 0.0
        apart = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float64)
        ff.rebuild_neighbors(apart)
        assert float(ff(apart).energy) == 0.0
        assert torch.equal(ff(apart).forces, torch.zeros_like(apart))

    def test_cutoff_defaults_to_the_lists(self):
        ff, _ = self._dimer(1.5, cutoff=3.0)
        assert float(ff.cutoff_sq) == pytest.approx(9.0)

    def test_rejects_cutoff_beyond_the_list_horizon(self):
        pos, cell = make_cubic_lattice()
        nl = NeighborList(cell=cell, cutoff=3.0, positions=pos)
        with pytest.raises(ValueError, match="horizon"):
            LennardJonesCutForceField(epsilon=1.0, sigma=1.0, neighbors=nl, cutoff=4.0)

    def test_to_reaches_the_neighbor_buffers(self):
        """``.to(dtype)`` must cast the list's shifts alongside the module buffers."""
        ff, pos = self._dimer(1.5)
        ff.to(torch.float32)
        assert ff.neighbors.shifts.dtype == torch.float32
        out = ff(pos.to(torch.float32))
        assert out.energy.dtype == torch.float32


class TestForceFieldNeighborSeam:
    """The capability flag and the policy semantics of the shared rebuild seam.

    ``rebuilds_neighbors`` answers *can this force field run a neighbour
    policy* (a read-only property mirroring ``Integrator.removed_dof``, so the
    integrator never duck-reads ``getattr(force, "neighbors", None)``);
    ``rebuild_neighbors(pos)`` runs that policy at the positions being
    evaluated and lets the list decline.
    """

    def _lj_cut(self, *, skin: float) -> tuple[LennardJonesCutForceField, torch.Tensor]:
        """lj/cut over a policy-configured list on the 64-atom, 12 Å lattice."""
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        nl = NeighborList(
            cell=cell, cutoff=3.5, positions=pos, skin=skin, every=1, delay=0, check=True
        )
        ff = LennardJonesCutForceField(epsilon=0.7, sigma=2.5, neighbors=nl).to(torch.float64)
        return ff, pos

    def test_a_listless_force_field_declares_no_policy(self):
        """Nothing to refresh ⇒ ``False``, so the integrator's switch stays off."""
        template = _periodic_template(make_cubic_lattice()[0])
        assert ForceField().rebuilds_neighbors is False
        assert HarmonicForceField(k=1.0).rebuilds_neighbors is False
        assert LennardJonesForceField(epsilon=1.0, sigma=1.0).rebuilds_neighbors is False
        assert (
            PotentialForceField(_ShiftAwarePairPotential(3.5), template).rebuilds_neighbors is False
        )
        assert CallableForceField(lambda pos: (pos.sum(), pos)).rebuilds_neighbors is False

    def test_a_list_backed_force_field_declares_a_policy(self):
        """A live list ⇒ ``True``; ``CallableForceField`` answers per instance."""
        pos, cell = make_cubic_lattice()
        periodic = PeriodicPotentialForceField(
            _ShiftAwarePairPotential(3.5),
            _periodic_template(pos),
            neighbors=NeighborList(cell=cell, cutoff=3.5, positions=pos),
        )
        lj_cut, _ = self._lj_cut(skin=0.0)
        bound = CallableForceField(lambda pos: (pos.sum(), pos), neighbors=_RecordingNeighbors())
        assert periodic.rebuilds_neighbors is True
        assert lj_cut.rebuilds_neighbors is True
        assert bound.rebuilds_neighbors is True

    def test_lj_cut_asks_the_policy_and_accepts_a_refusal(self):
        """``rebuild_neighbors`` is a *question* now: inside the half-skin the
        list declines, so a displacement smaller than ``skin/2`` must leave
        ``rebuild_count`` at zero while the policy clock still ticks."""
        ff, pos = self._lj_cut(skin=2.0)  # half-skin 1.0 A
        moved = pos.clone()
        moved[0, 0] += 0.1  # far inside the half-skin
        ff.rebuild_neighbors(moved)
        assert ff.neighbors.rebuild_count == 0, "the seam forced a build the policy declined"
        assert ff.neighbors.ago == 1, "the policy clock did not advance"

    def test_the_periodic_potential_asks_the_same_policy(self):
        """The TensorDict-side owner delegates to the same ``update`` gate."""
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        nl = NeighborList(
            cell=cell, cutoff=3.5, positions=pos, skin=2.0, every=1, delay=0, check=True
        )
        ff = PeriodicPotentialForceField(
            _ShiftAwarePairPotential(3.5), _periodic_template(pos), neighbors=nl
        )
        moved = pos.clone()
        moved[0, 0] += 0.1
        ff.rebuild_neighbors(moved)
        assert nl.rebuild_count == 0
        assert nl.ago == 1

    def test_the_seam_never_forces_a_build(self):
        """Recorded at the strategy: ``update`` yes, ``rebuild`` never."""
        recorder = _RecordingNeighbors()
        ff = CallableForceField(lambda pos: (pos.sum(), pos), neighbors=recorder)
        pos = torch.zeros(4, 3)
        ff.rebuild_neighbors(pos)
        ff.rebuild_neighbors(pos)
        assert len(recorder.updates) == 2
        assert recorder.rebuilds == []

    def test_a_forced_build_is_still_reachable_on_the_list(self):
        """The escape hatch the docstring promises: ``neighbors.rebuild(pos)``
        builds unconditionally — inside the half-skin, where the policy would
        decline — counts exactly once, and re-phases the policy clock."""
        ff, pos = self._lj_cut(skin=2.0)
        moved = pos.clone()
        moved[0, 0] += 0.1  # a displacement ``update`` would refuse to build on
        ff.neighbors.rebuild(moved)
        assert ff.neighbors.rebuild_count == 1
        assert ff.neighbors.ago == 0
