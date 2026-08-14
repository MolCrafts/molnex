import pint
import pytest
import torch

from molix.data.pipeline import Pipeline
from molix.data.task import DatasetTask, Runnable, SampleTask
from molix.data.tasks import AtomicDress, UnitConvert


class TestAtomicDress:
    def test_is_dataset_task(self):
        t = AtomicDress(elements=[1, 6])
        assert isinstance(t, DatasetTask)
        assert isinstance(t, Runnable)

    def test_fit_and_execute(self):
        samples = [
            {
                "Z": torch.tensor([1, 1]),
                "pos": torch.zeros(2, 3),
                "targets": {"U0": torch.tensor([2.0])},
            },
            {
                "Z": torch.tensor([6, 1]),
                "pos": torch.zeros(2, 3),
                "targets": {"U0": torch.tensor([7.0])},
            },
            {
                "Z": torch.tensor([6, 6]),
                "pos": torch.zeros(2, 3),
                "targets": {"U0": torch.tensor([12.0])},
            },
        ]

        task = AtomicDress(elements=[1, 6], target_key="U0", output_key="U0_dressed")
        task.fit(samples)

        dressed = torch.stack([task(s)["targets"]["U0_dressed"].reshape(-1)[0] for s in samples])
        assert torch.allclose(dressed, torch.zeros_like(dressed), atol=1e-5)

    def test_state_dict_roundtrip(self):
        samples = [
            {
                "Z": torch.tensor([1, 1]),
                "pos": torch.zeros(2, 3),
                "targets": {"U0": torch.tensor([2.0])},
            },
            {
                "Z": torch.tensor([6, 1]),
                "pos": torch.zeros(2, 3),
                "targets": {"U0": torch.tensor([7.0])},
            },
        ]

        t1 = AtomicDress(elements=[1, 6])
        t1.fit(samples)
        state = t1.state_dict()

        t2 = AtomicDress(elements=[1, 6])
        t2.load_state_dict(state)
        assert t1.atomic_energies == t2.atomic_energies


class TestUnitConvert:
    HARTREE_TO_EV = 27.211386245988  # CODATA 2018

    def test_scalar_energy_hartree_to_ev(self):
        """QM9 use-case: U0 Hartree → eV."""
        task = UnitConvert({"U0": ("hartree", "eV")})
        assert task.factors["U0"] == pytest.approx(self.HARTREE_TO_EV, rel=1e-6)

        sample = {
            "Z": torch.tensor([1, 1]),
            "pos": torch.zeros(2, 3),
            "targets": {"U0": torch.tensor([1.0])},
        }
        out = task(sample)
        torch.testing.assert_close(
            out["targets"]["U0"],
            torch.tensor([self.HARTREE_TO_EV], dtype=torch.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        # Original sample untouched.
        assert torch.allclose(sample["targets"]["U0"], torch.tensor([1.0]))

    def test_derived_unit_force(self):
        """Force: hartree/bohr → eV/Å — derived units flow through pint."""
        task = UnitConvert({"forces": ("hartree / bohr", "eV / angstrom")})
        # hartree/bohr ≈ 51.422 eV/Å
        assert task.factors["forces"] == pytest.approx(51.42208619, rel=1e-4)

    def test_multiple_fields(self):
        task = UnitConvert(
            {
                "U0": ("hartree", "eV"),
                "U": ("hartree", "eV"),
            }
        )
        sample = {
            "Z": torch.tensor([1]),
            "pos": torch.zeros(1, 3),
            "targets": {
                "U0": torch.tensor([1.0]),
                "U": torch.tensor([2.0]),
            },
        }
        out = task(sample)["targets"]
        torch.testing.assert_close(
            out["U0"],
            torch.tensor([self.HARTREE_TO_EV], dtype=torch.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            out["U"],
            torch.tensor([2 * self.HARTREE_TO_EV], dtype=torch.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_mixed_units_per_field(self):
        """Each field chooses its own src/dst independently — no preset bundle."""
        task = UnitConvert(
            {
                "energy": ("hartree", "eV"),
                "forces": ("hartree / bohr", "eV / angstrom"),
                "length": ("bohr", "angstrom"),
            }
        )
        assert task.factors["energy"] == pytest.approx(27.21139, rel=1e-4)
        assert task.factors["forces"] == pytest.approx(51.42209, rel=1e-4)
        assert task.factors["length"] == pytest.approx(0.52918, rel=1e-4)

    def test_incompatible_dimensions_raises(self):
        """Registry rejects hartree → Å (energy vs length)."""
        from molpy import UnitsError

        with pytest.raises((pint.errors.DimensionalityError, UnitsError)):
            UnitConvert({"x": ("hartree", "angstrom")})

    def test_unknown_unit_raises(self):
        from molpy import UnitsError

        with pytest.raises((pint.errors.UndefinedUnitError, UnitsError)):
            UnitConvert({"x": ("nonsense", "eV")})

    def test_missing_target_raises(self):
        task = UnitConvert({"U": ("hartree", "eV")})
        sample = {
            "Z": torch.tensor([1]),
            "pos": torch.zeros(1, 3),
            "targets": {"U0": torch.tensor([1.0])},
        }
        with pytest.raises(KeyError, match="'U'"):
            task(sample)

    def test_empty_conversions_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            UnitConvert({})

    def test_non_tuple_value_raises(self):
        with pytest.raises(ValueError, match="2-tuple"):
            UnitConvert({"x": "hartree"})  # not a tuple
        with pytest.raises(ValueError, match="2-tuple"):
            UnitConvert({"x": ("hartree",)})  # wrong arity

    def test_task_id_encodes_src_and_dst(self):
        t = UnitConvert({"U0": ("hartree", "eV")})
        assert "U0" in t.task_id
        assert "hartree" in t.task_id
        # User-supplied dst string is preserved (not registry-canonicalized).
        assert "eV" in t.task_id

    def test_task_id_order_invariant(self):
        t1 = UnitConvert({"U0": ("hartree", "eV"), "U": ("hartree", "eV")})
        t2 = UnitConvert({"U": ("hartree", "eV"), "U0": ("hartree", "eV")})
        assert t1.task_id == t2.task_id


class TestPipeline:
    def test_registration_methods(self):
        pipe = Pipeline("test")

        # Task subclass
        pipe.add(UnitConvert({"U0": ("hartree", "eV")}), name="units")

        # Bare callable
        pipe.add(lambda s: {**s, "extra": 1}, name="extra")

        # Pre-built Node
        from molix.data.pipeline import Node

        pipe.node(Node(name="tag", task=lambda s: {**s, "tag": "test"}))

        spec = pipe.build()
        assert len(spec.nodes) == 3
        assert spec.pipeline_id  # non-empty hash

    def test_isinstance_dispatch(self):
        pipe = Pipeline("test")
        pipe.add(AtomicDress(elements=[1, 6]))
        pipe.add(UnitConvert({"U0": ("hartree", "eV")}))
        spec = pipe.build()

        assert len(spec.prepare_nodes) == 2

        assert isinstance(spec.prepare_nodes[0].task, DatasetTask)
        assert isinstance(spec.prepare_nodes[1].task, SampleTask)
