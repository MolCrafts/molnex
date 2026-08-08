"""Tests for molix.md.forcefield — binding models to systems."""

import pytest
import torch
from tensordict import TensorDict

from molix.md import (
    CallableForceField,
    ForceOutput,
    HarmonicForceField,
    PeriodicNeighborList,
    PeriodicPotentialForceField,
    PotentialForceField,
)
from tests.test_molix.test_md.conftest import make_pinet_template, make_tiny_potential


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

    def test_delegates_rebuild_to_the_neighbor_list(self):
        class _Recorder:
            edge_index = torch.zeros(1, 2, dtype=torch.long)
            shifts = torch.zeros(1, 3)
            num_edges = 0
            capacity = 1

            def __init__(self):
                self.rebuilds = 0

            def rebuild(self, positions):
                self.rebuilds += 1

            def to(self, device=None, dtype=None):
                return self

        recorder = _Recorder()
        ff = CallableForceField(lambda pos: (pos.sum(), pos), neighbors=recorder)
        ff.rebuild_neighbors(torch.zeros(2, 3))
        assert recorder.rebuilds == 1

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


def _cubic_lattice(n_side: int = 3, spacing: float = 3.0):
    grid = torch.arange(n_side, dtype=torch.float64) * spacing
    pos = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1).reshape(-1, 3)
    cell = torch.eye(3, dtype=torch.float64) * (n_side * spacing)
    return pos, cell


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
        pos, cell = _cubic_lattice()
        outs = []
        for factor in (1.1, 3.0):
            nl = PeriodicNeighborList(cell=cell, cutoff=3.5, positions=pos, capacity_factor=factor)
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
        pos, cell = _cubic_lattice()
        nl = PeriodicNeighborList(cell=cell, cutoff=3.5, positions=pos, capacity_factor=8.0)
        ff = PeriodicPotentialForceField(
            _ShiftAwarePairPotential(3.5), _periodic_template(pos), neighbors=nl
        )
        compressed = pos * 0.8  # second-neighbour pairs enter the cutoff
        stale = ff(compressed)  # list still from the original positions
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
