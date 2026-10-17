"""Tests for :mod:`molix.deploy.contract` — the atom-energy calling convention."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.deploy import AtomEnergyAdapter, example_inputs, verify
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, PAIR_DIST_CLAMP_MIN
from tests.test_molix.test_deploy.conftest import _DummyTDPotential


class _Returns(torch.nn.Module):
    """Potential stand-in that returns one fixed object."""

    def __init__(self, out: object) -> None:
        super().__init__()
        self._out = out

    def forward(self, batch):
        return self._out


class _EnergyOnlyTDPotential(_DummyTDPotential):
    def forward(self, batch):
        out = super().forward(batch)
        del out["atoms", "forces"]
        return out


class _NotDecomposable(_DummyTDPotential):
    """A graph energy with a term no atom owns (what a long-range term does)."""

    def forward(self, batch):
        out = super().forward(batch)
        out[ENERGY_KEY] = out[ENERGY_KEY] + 1.0
        return out


class _WrongForces(_DummyTDPotential):
    def forward(self, batch):
        out = super().forward(batch)
        out["atoms", "forces"] = out["atoms", "forces"] + 0.5
        return out


def _system() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Atoms at (0,0,0), (1,0,0), (0,2,0); edges (0,1) (1,0) (1,2) (2,1), no shifts.

    Pair term ``|d_e|²`` split evenly: atom energies ``[1, 6, 5]`` (total 12).
    """
    Z = torch.tensor([1, 6, 8])
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    pair_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
    return Z, pos, pair_index, torch.zeros(4, 3, dtype=torch.float64)


class TestAtomEnergyAdapter:
    def test_batch_edge_convention(self, graph):
        """pair_diff = pos[target] - pos[source] + shifts; one graph."""
        Z, pos, pair_index = graph
        shifts = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [0.0] * 3])
        batch = AtomEnergyAdapter.build_batch(Z, pos, pair_index, shifts)
        expected = pos[pair_index[:, 1]] - pos[pair_index[:, 0]] + shifts
        assert torch.allclose(batch["pairs", "pair_diff"], expected)
        assert torch.allclose(
            batch["pairs", "pair_dist"], expected.norm(dim=-1).clamp(min=PAIR_DIST_CLAMP_MIN)
        )
        assert (batch["atoms", "batch"] == 0).all()
        assert batch["graphs", "num_atoms"].item() == 3

    def test_coincident_atoms_clamp_to_the_shared_constant(self):
        Z = torch.tensor([1, 1])
        pos = torch.zeros(2, 3, dtype=torch.float64)
        batch = AtomEnergyAdapter.build_batch(
            Z, pos, torch.tensor([[0, 1], [1, 0]]), torch.zeros(2, 3, dtype=torch.float64)
        )
        assert batch["pairs", "pair_dist"].tolist() == [PAIR_DIST_CLAMP_MIN, PAIR_DIST_CLAMP_MIN]

    def test_all_ones_mask_sums_every_site(self):
        energy, _, atom_energy, _ = AtomEnergyAdapter(_DummyTDPotential())(
            *_system(), torch.ones(3, dtype=torch.bool)
        )
        assert energy.item() == pytest.approx(12.0, abs=1e-10)
        assert atom_energy.tolist() == pytest.approx([1.0, 6.0, 5.0], abs=1e-10)

    def test_partial_mask_keeps_a_force_on_the_masked_out_site(self):
        """Mask [T, T, F] sums atoms 0 and 1 (7); hand-derived partition forces."""
        energy, forces, _, _ = AtomEnergyAdapter(_DummyTDPotential())(
            *_system(), torch.tensor([True, True, False])
        )
        assert energy.item() == pytest.approx(7.0, abs=1e-10)
        expected = torch.tensor(
            [[4.0, 0.0, 0.0], [-6.0, 4.0, 0.0], [2.0, -4.0, 0.0]], dtype=torch.float64
        )
        assert torch.allclose(forces, expected, atol=1e-8, rtol=0.0)

    def test_virial_is_the_strain_derivative(self):
        """E = sum_e |d_e|^2, so dE/deps = sum_e d_e (x) 2 d_e."""
        _, _, _, virial = AtomEnergyAdapter(_DummyTDPotential())(
            *_system(), torch.ones(3, dtype=torch.bool)
        )
        expected = torch.zeros(3, 3, dtype=torch.float64)
        expected[:2, :2] = torch.tensor([[8.0, -8.0], [-8.0, 16.0]])
        assert torch.allclose(virial, expected, atol=1e-10, rtol=0.0)

    def test_folded_images_equal_staged_ghost_copies(self):
        """One periodic pair two ways: shifts on owners, or ghost rows under a mask."""
        adapter = AtomEnergyAdapter(_DummyTDPotential())
        Z = torch.tensor([1, 8])
        pos = torch.tensor([[0.2, 0.0, 0.0], [2.5, 0.3, 0.0]], dtype=torch.float64)
        lattice = torch.tensor([-3.0, 0.0, 0.0], dtype=torch.float64)
        folded = adapter(
            Z,
            pos,
            torch.tensor([[0, 1], [1, 0]]),
            torch.stack([lattice, -lattice]),
            torch.ones(2, dtype=torch.bool),
        )
        # Ghost rows 2 = atom 1 + lattice and 3 = atom 0 - lattice.
        staged_pos = torch.cat([pos, (pos[1] + lattice)[None], (pos[0] - lattice)[None]])
        staged = adapter(
            torch.tensor([1, 8, 8, 1]),
            staged_pos,
            torch.tensor([[0, 2], [2, 0], [1, 3], [3, 1]]),
            torch.zeros(4, 3, dtype=torch.float64),
            torch.tensor([True, True, False, False]),
        )
        assert torch.allclose(folded[0], staged[0], atol=1e-12, rtol=0.0)
        owner_forces = staged[1][:2] + staged[1][[3, 2]]
        assert torch.allclose(folded[1], owner_forces, atol=1e-12, rtol=0.0)
        assert torch.allclose(folded[3], staged[3], atol=1e-12, rtol=0.0)

    def test_energy_only_potential_gives_the_same_partition(self):
        mask = torch.ones(3, dtype=torch.bool)
        reference = AtomEnergyAdapter(_DummyTDPotential())(*_system(), mask)
        energy_only = AtomEnergyAdapter(_EnergyOnlyTDPotential())(*_system(), mask)
        for ours, ref in zip(energy_only, reference, strict=True):
            assert torch.allclose(ours, ref, atol=1e-12, rtol=0.0)
        assert AtomEnergyAdapter(_EnergyOnlyTDPotential()).model_outputs(*_system())[1] is None

    def test_plain_dict_output_is_refused(self):
        with pytest.raises(ValueError, match="TensorDict"):
            AtomEnergyAdapter(_Returns({"energy": torch.zeros(1)})).model_outputs(*_system())

    def test_missing_atom_energy_is_refused(self):
        out = TensorDict(
            atoms=TensorDict(node_energies=torch.zeros(3), batch_size=[3]),
            graphs=TensorDict(energy=torch.zeros(1), batch_size=[1]),
            batch_size=[],
        )
        with pytest.raises(ValueError, match="per-atom energy"):
            AtomEnergyAdapter(_Returns(out)).model_outputs(*_system())

    def test_atom_energy_is_read_under_the_nested_key(self):
        out = AtomEnergyAdapter(_DummyTDPotential()).model_outputs(*_system())
        batch = _DummyTDPotential()(AtomEnergyAdapter.build_batch(*_system()))
        assert torch.equal(out[2], batch[ATOMIC_ENERGY_KEY])

    def test_check_accepts_a_decomposable_potential(self):
        AtomEnergyAdapter(_DummyTDPotential()).check(*_system())

    def test_check_refuses_energy_no_site_owns(self):
        with pytest.raises(ValueError, match="sum of its atom energies"):
            AtomEnergyAdapter(_NotDecomposable()).check(*_system())

    def test_check_refuses_forces_that_are_not_the_gradient(self):
        with pytest.raises(ValueError, match="model forces disagree"):
            AtomEnergyAdapter(_WrongForces()).check(*_system())


class TestVerify:
    class _Triple(torch.nn.Module):
        def __init__(self, out):
            super().__init__()
            self.out = out

        def forward(self, Z, pos, pair_index, shifts, local_mask):
            return self.out

    def _run(self, out):
        return verify(self._Triple(out), *_system(), torch.ones(3, dtype=torch.bool))

    def test_wrong_return_count_is_refused(self):
        with pytest.raises(ValueError, match="four tensors"):
            self._run((torch.zeros(()), torch.zeros(3, 3), torch.zeros(3)))

    def test_wrong_shapes_are_refused(self):
        with pytest.raises(ValueError, match="forces must have shape"):
            self._run((torch.zeros(()), torch.zeros(3), torch.zeros(3), torch.zeros(3, 3)))
        with pytest.raises(ValueError, match="virial must have shape"):
            self._run((torch.zeros(()), torch.zeros(3, 3), torch.zeros(3), torch.zeros(6)))

    def test_energy_not_the_masked_site_sum_is_refused(self):
        with pytest.raises(ValueError, match="masked sum"):
            self._run((torch.tensor(1.0), torch.zeros(3, 3), torch.zeros(3), torch.zeros(3, 3)))

    def test_non_finite_outputs_are_refused(self):
        nan = torch.full((3, 3), float("nan"))
        with pytest.raises(ValueError, match="forces are not finite"):
            self._run((torch.tensor(0.0), nan, torch.zeros(3), torch.zeros(3, 3)))
        with pytest.raises(ValueError, match="virial is not finite"):
            self._run((torch.tensor(0.0), torch.zeros(3, 3), torch.zeros(3), nan))


class TestExampleInputs:
    def test_every_pair_is_inside_the_cutoff_and_every_species_present(self):
        Z, pos, pair_index, shifts, mask = example_inputs([1, 6, 8], 5.0, torch.float64, "cpu")
        n = Z.shape[0]
        assert n >= 8 and set(Z.tolist()) == {1, 6, 8} and bool(mask.all())
        assert pair_index.shape == (n * (n - 1), 2) and pair_index.is_contiguous()
        assert shifts.shape == (n * (n - 1), 3) and not shifts.any()

    def test_no_atom_is_paired_with_itself(self):
        # cdist's matmul path (> 25 rows) leaves the float64 diagonal slightly
        # above zero, so a distance test alone would pair an atom with itself.
        Z, _, pair_index, _, _ = example_inputs(list(range(1, 30)), 12.0, torch.float64, "cpu")
        n = Z.shape[0]
        assert bool((pair_index[:, 0] != pair_index[:, 1]).all())
        assert pair_index.shape == (n * (n - 1), 2)
