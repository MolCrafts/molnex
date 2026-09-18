"""Tests for ``molix.engine.adapter`` — registry + flat calling convention.

The adapter layer is the Python-side shim that maps the flat LAMMPS convention
onto a model's real input/output shapes. Schema v3 (spec
``lammps-iface-05-pbc``) changes that convention to

    (Z, pos, pair_index, local_mask) -> (energy, forces, atomic_energy)

because a domain-decomposed run must sum **only local** site energies
(``E⁽ᵖ⁾ = Σ_{i∈L} E_i``, D1): a scalar total is not partitionable, so the
per-atom energy becomes the authoritative output and the exported energy is
``(atomic_energy * local_mask).sum()``. Covered here: the self-registration
registry, the molnex nested ``TensorDict`` construction (edge convention and
the shared distance clamp), per-atom energy extraction for both built-in
adapters, and the ``local_mask`` semantics of ``EngineForward``. The C++ pair
style itself is exercised by an actual LAMMPS run (see
``interface/lammps/README.md``).

Units: the stub potential is unitless; energies/forces below are hard-coded
literals derived by hand from its documented pair term (see ``conftest.py``).

-- migrated verbatim from ``tests/test_molix/test_engine.py``, then updated to
   schema v3 --
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

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
    """pair_diff = pos[target] - pos[source]; single-graph batch metadata."""
    from molix.schema import PAIR_DIST_CLAMP_MIN

    Z, pos, pair_index = graph
    batch = MolnexTensorDictAdapter().build_inputs(_DummyTDPotential(), Z, pos, pair_index)
    expected = pos[pair_index[:, 1]] - pos[pair_index[:, 0]]
    assert torch.allclose(batch["pairs", "pair_diff"], expected)
    assert torch.allclose(
        batch["pairs", "pair_dist"], expected.norm(dim=-1).clamp(min=PAIR_DIST_CLAMP_MIN)
    )
    assert (batch["atoms", "batch"] == 0).all()
    assert batch["graphs", "num_atoms"].item() == 3


def test_lammps_forward_shapes_tensordict(graph):
    Z, pos, pair_index = graph
    local_mask = torch.ones(3, dtype=torch.bool)
    e, f, ea = EngineForward(_DummyTDPotential(), "molnex-tensordict")(
        Z, pos, pair_index, local_mask
    )
    assert e.shape == ()  # scalar partition energy
    assert f.shape == (3, 3)
    assert ea.shape == (3,)  # per-atom energy, the v3 third output


def test_lammps_forward_shapes_flat(graph):
    Z, pos, pair_index = graph
    local_mask = torch.ones(3, dtype=torch.bool)
    e, f, ea = EngineForward(_FlatEFPotential(), "flat")(Z, pos, pair_index, local_mask)
    assert e.shape == ()
    assert f.shape == (3, 3)
    assert ea.shape == (3,)


class TestEdgeDistanceClamp:
    """Target: ``MolnexTensorDictAdapter.build_inputs`` small-``r`` regularisation.

    The adapter path and the ``StaticForward`` path are documented as
    equivalent, yet used to clamp ``pair_dist`` at 1e-6 and 1e-8 respectively
    (spec ``lammps-iface-05-pbc``, "顺带修复二"). Both must now consume one named
    constant in ``molix.schema``. This class asserts *this* path's clamp equals
    that constant; ``test_static.py::TestEdgeDistanceClamp`` asserts the same of
    the other path against the same constant, which pins the two paths to each
    other without either test module reaching into the other's production
    module. The constant's own value (1e-6) is pinned where it is declared,
    ``tests/test_molix/test_schema.py``.
    """

    @staticmethod
    def degenerate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two coincident atoms — ``|pair_diff| == 0``, i.e. clamp territory."""
        Z = torch.tensor([1, 1])
        pos = torch.zeros(2, 3, dtype=torch.float64)
        pair_index = torch.tensor([[0, 1], [1, 0]])
        return Z, pos, pair_index

    def test_coincident_atoms_clamp_to_the_shared_constant(self):
        from molix.schema import PAIR_DIST_CLAMP_MIN

        Z, pos, pair_index = self.degenerate()
        batch = MolnexTensorDictAdapter().build_inputs(_DummyTDPotential(), Z, pos, pair_index)
        assert batch["pairs", "pair_dist"].tolist() == [
            PAIR_DIST_CLAMP_MIN,
            PAIR_DIST_CLAMP_MIN,
        ]


class TestMolnexTensorDictAdapter:
    """Target: ``MolnexTensorDictAdapter.read_outputs`` (schema-v3 triple).

    v3 recognises exactly two spellings of the per-atom energy: the nested
    ``ATOMIC_ENERGY_KEY`` for a ``TensorDict``-shaped model output, and the flat
    ``ATOMIC_ENERGY_FLAT_KEY`` for a plain dict. Anything else is refused rather
    than silently dropped — after 05 the per-atom energy is the *only* source of
    ``eng_vdwl``, so a silent drop is a silent zero energy.
    """

    ATOMIC = [1.0, 6.0, 5.0]

    @classmethod
    def flat_output(cls, *, per_atom_key: str | None) -> dict:
        """Plain-dict model output; ``per_atom_key`` ``None`` omits the per-atom energy."""
        out: dict[str, torch.Tensor] = {
            "energy": torch.tensor([12.0], dtype=torch.float64),
            "forces": torch.zeros(3, 3, dtype=torch.float64),
        }
        if per_atom_key is not None:
            out[per_atom_key] = torch.tensor(cls.ATOMIC, dtype=torch.float64)
        return out

    @classmethod
    def nested_output(cls) -> TensorDict:
        """Post-collate-shaped output: energy under ``graphs``, per-atom under ``atoms``."""
        return TensorDict(
            atoms=TensorDict(
                energy=torch.tensor(cls.ATOMIC, dtype=torch.float64),
                forces=torch.zeros(3, 3, dtype=torch.float64),
                batch_size=[3],
            ),
            graphs=TensorDict(energy=torch.tensor([12.0], dtype=torch.float64), batch_size=[1]),
            batch_size=[],
        )

    def test_flat_dict_per_atom_energy_is_read_under_the_flat_key(self):
        from molix.schema import ATOMIC_ENERGY_FLAT_KEY

        out = self.flat_output(per_atom_key=ATOMIC_ENERGY_FLAT_KEY)
        energy, forces, atomic = MolnexTensorDictAdapter().read_outputs(out)
        assert energy.item() == pytest.approx(12.0, abs=1e-10)
        assert forces.shape == (3, 3)
        assert torch.allclose(
            atomic, torch.tensor(self.ATOMIC, dtype=torch.float64), atol=1e-10, rtol=0.0
        )

    def test_nested_output_per_atom_energy_is_read_under_the_nested_key(self):
        from molix.schema import ATOMIC_ENERGY_KEY

        out = self.nested_output()
        _, _, atomic = MolnexTensorDictAdapter().read_outputs(out)
        assert torch.allclose(atomic, out[ATOMIC_ENERGY_KEY], atol=1e-10, rtol=0.0)

    def test_missing_per_atom_energy_is_refused(self):
        """Names both accepted spellings so the fix is obvious from the message."""
        with pytest.raises(ValueError) as excinfo:
            MolnexTensorDictAdapter().read_outputs(self.flat_output(per_atom_key=None))
        message = str(excinfo.value)
        assert "atomic_energy" in message
        assert "atoms" in message and "energy" in message

    def test_non_canonical_spelling_is_not_absorbed(self):
        """A per-atom tensor under a third spelling is refused, never silently dropped."""
        from molix.schema import ATOMIC_ENERGY_FLAT_KEY

        assert "node_energies" != ATOMIC_ENERGY_FLAT_KEY
        with pytest.raises(ValueError):
            MolnexTensorDictAdapter().read_outputs(self.flat_output(per_atom_key="node_energies"))

    def test_energy_only_flat_dict_returns_none_forces(self):
        """Forces are optional: the wrapper derives them, so an energy-only model is valid."""
        from molix.schema import ATOMIC_ENERGY_FLAT_KEY

        out = self.flat_output(per_atom_key=ATOMIC_ENERGY_FLAT_KEY)
        del out["forces"]
        energy, forces, atomic = MolnexTensorDictAdapter().read_outputs(out)
        assert forces is None
        assert energy.item() == pytest.approx(12.0, abs=1e-10)
        assert torch.allclose(
            atomic, torch.tensor(self.ATOMIC, dtype=torch.float64), atol=1e-10, rtol=0.0
        )

    def test_energy_only_nested_output_returns_none_forces(self):
        out = self.nested_output()
        del out["atoms", "forces"]
        _, forces, atomic = MolnexTensorDictAdapter().read_outputs(out)
        assert forces is None
        assert torch.allclose(
            atomic, torch.tensor(self.ATOMIC, dtype=torch.float64), atol=1e-10, rtol=0.0
        )


class TestFlatTensorAdapter:
    """Target: ``FlatTensorAdapter.read_outputs`` (schema-v3 triple).

    A third-party model that still returns only ``(energy, forces)`` cannot be
    partitioned in a domain-decomposed run, so v3 refuses it at export time
    instead of deploying a model that over-counts every ghost site (D1).
    """

    ATOMIC = [1.0, 6.0, 5.0]

    def test_three_tuple_returns_the_triple(self):
        atomic = torch.tensor(self.ATOMIC, dtype=torch.float64)
        energy = torch.tensor([12.0], dtype=torch.float64)
        forces = torch.zeros(3, 3, dtype=torch.float64)
        got_e, got_f, got_a = FlatTensorAdapter().read_outputs((energy, forces, atomic))
        assert got_e.item() == pytest.approx(12.0, abs=1e-10)
        assert got_f.shape == (3, 3)
        assert torch.allclose(got_a, atomic, atol=1e-10, rtol=0.0)

    def test_dict_output_returns_the_triple(self):
        from molix.schema import ATOMIC_ENERGY_FLAT_KEY

        atomic = torch.tensor(self.ATOMIC, dtype=torch.float64)
        out = {
            "energy": torch.tensor([12.0], dtype=torch.float64),
            "forces": torch.zeros(3, 3, dtype=torch.float64),
            ATOMIC_ENERGY_FLAT_KEY: atomic,
        }
        got_e, _, got_a = FlatTensorAdapter().read_outputs(out)
        assert got_e.item() == pytest.approx(12.0, abs=1e-10)
        assert torch.allclose(got_a, atomic, atol=1e-10, rtol=0.0)

    def test_two_tuple_is_refused(self):
        """v2's ``(energy, forces)`` return is fail-closed under v3."""
        out = (torch.tensor([12.0]), torch.zeros(3, 3))
        with pytest.raises(ValueError) as excinfo:
            FlatTensorAdapter().read_outputs(out)
        assert "atomic_energy" in str(excinfo.value)


class TestLocalMask:
    """Target: ``EngineForward.forward`` — the ``local_mask`` partition (D1/D2).

    Geometry: 3 atoms at ``(0,0,0)``, ``(1,0,0)``, ``(0,2,0)`` with the
    bidirectional edge set ``(0,1) (1,0) (1,2) (2,1)``. The stub's pair term is
    ``|d_e|²`` split evenly over the endpoints (``conftest.py``), so with
    ``D01 = 1`` and ``D12 = 5``:

        atomic_energy = [D01, D01 + D12, D12] = [1, 6, 5]   (total 12)

    Every expected number below is a hand-derived literal, never recomputed by
    the assertion side.

    Two of these cases only pass if the returned forces are
    ``-∂[(atomic_energy · local_mask).sum()]/∂pos`` — the masked partition
    energy — rather than the model's own total-energy forces. That is D2: a
    ghost site contributes no site energy to this rank yet must still receive
    the cross-term force, and, symmetrically, an all-zero mask (a rank owning
    nothing) must produce no force at all.
    """

    @staticmethod
    def system() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Z = torch.tensor([1, 6, 8])
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
        pair_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]])
        return Z, pos, pair_index

    @staticmethod
    def run(local_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Z, pos, pair_index = TestLocalMask.system()
        return EngineForward(_DummyTDPotential(), "molnex-tensordict")(
            Z, pos, pair_index, local_mask
        )

    def test_all_ones_mask_sums_every_site(self):
        energy, _, atomic = self.run(torch.ones(3, dtype=torch.bool))
        assert energy.item() == pytest.approx(12.0, abs=1e-10)
        assert torch.allclose(
            atomic, torch.tensor([1.0, 6.0, 5.0], dtype=torch.float64), atol=1e-10, rtol=0.0
        )

    def test_partial_mask_sums_only_the_masked_subset(self):
        """Atoms 0 and 1 are local; atom 2 is a ghost — its site energy is its owner's."""
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

    def test_flat_adapter_applies_the_same_mask(self):
        """The mask is the wrapper's business, not the adapter's."""
        Z, pos, pair_index = self.system()
        energy, _, _ = EngineForward(_FlatEFPotential(), "flat")(
            Z, pos, pair_index, torch.tensor([True, True, False])
        )
        assert energy.item() == pytest.approx(7.0, abs=1e-10)


class _EnergyOnlyTDPotential(_DummyTDPotential):
    """The stub with its force output dropped — an energy-only (``wrt=()``) shape."""

    def forward(self, batch):
        out = super().forward(batch)
        del out["forces"]
        return out


class TestEnergyOnlyModel:
    """Target: ``EngineForward.forward`` with a model that emits no forces.

    The deployed forces are always the wrapper's own ``-∂(masked sum)/∂pos``,
    so dropping the model's force output must change nothing about what the
    export produces — same energy, same forces, same per-atom energies as the
    forceful stub on the same system.
    """

    def test_derived_outputs_match_the_forceful_stub(self):
        Z, pos, pair_index = TestLocalMask.system()
        mask = torch.ones(3, dtype=torch.bool)
        e_ref, f_ref, a_ref = EngineForward(_DummyTDPotential(), "molnex-tensordict")(
            Z, pos, pair_index, mask
        )
        e, f, a = EngineForward(_EnergyOnlyTDPotential(), "molnex-tensordict")(
            Z, pos, pair_index, mask
        )
        assert e.item() == pytest.approx(e_ref.item(), abs=1e-12)
        assert torch.allclose(f, f_ref, atol=1e-12, rtol=0.0)
        assert torch.allclose(a, a_ref, atol=1e-12, rtol=0.0)

    def test_model_outputs_forces_slot_is_none(self):
        Z, pos, pair_index = TestLocalMask.system()
        _, forces, _ = EngineForward(_EnergyOnlyTDPotential(), "molnex-tensordict").model_outputs(
            Z, pos, pair_index
        )
        assert forces is None
