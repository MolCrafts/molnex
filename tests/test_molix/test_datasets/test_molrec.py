"""Tests for ``MolRecSource`` — molrs ``MolRec`` (.zarr) distillation loader.

``MolRecSource`` adapts a single-teacher view of a molrs ``MolRec`` archive
into the flat-sample :class:`molix.data.source.DataSource` contract. One record
may carry observables from several teachers (``teacherA.energy``,
``teacherB.energy``, ...); a ``MolRecSource`` is pinned to exactly one teacher
at construction and exposes that teacher's observables as targets with the
``"<teacher_id>."`` prefix stripped.

Acceptance trace
----------------
* ac-002 → ``TestMolRecHappyPath`` (construction, isinstance, len, keys/shapes)
* ac-003 → ``TestMolRecHappyPath`` (dtypes, stripped target names)
* ac-004 → ``TestMolRecDynamicSchema`` (per-instance graph/atom schema)
* ac-005 → ``TestMolRecSourceID`` (teacher tag, per-teacher + subset difference)
* ac-006 → ``TestMolRecRuntimeErrors`` (unknown teacher, teacher isolation)
* ac-007 → ``TestMolRecQM9Roundtrip`` (15 float32 scalars lossless)
* ac-008 → ``TestMolRecForceRoundtrip`` (forces shape + lossless, energy lossless)

Note on dtypes: molrs zarr storage upcasts float32 -> float64 on read-back, so
these tests assert that ``MolRecSource`` *output* is ``torch.float32`` (the
source casts) and verify value losslessness with ``np.allclose`` /
``torch.allclose`` — never that the raw zarr is float32.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molix.data.collate import TargetSchema
from molix.data.source import DataSource

# Import under test: not yet implemented. The whole module is expected to fail
# at import time (RED) until ``src/molix/datasets/molrec.py`` exists.
from molix.datasets import MolRecSource

# ---------------------------------------------------------------------------
# Happy path — construction / shapes / dtypes / target names
# ---------------------------------------------------------------------------


class TestMolRecHappyPath:
    def test_construction_and_len(self, molrec_qm9_record):
        """ac-002: eager construction; ``len`` equals frame count."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        assert len(src) == molrec_qm9_record.n_frames

    def test_isinstance_data_source_protocol(self, molrec_qm9_record):
        """ac-002: conforms to the runtime-checkable DataSource Protocol."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        assert isinstance(src, DataSource)

    def test_not_subclass_of_source_module_classes(self, molrec_qm9_record):
        """ac-002: structural conformance only — no inheritance from data.source.

        ``MolRecSource`` satisfies the Protocol by shape, not by subclassing any
        concrete class defined in ``molix.data.source``.
        """
        import molix.data.source as source_mod

        source_module_classes = {obj for obj in vars(source_mod).values() if isinstance(obj, type)}
        for base in MolRecSource.__mro__[1:]:  # skip MolRecSource itself
            assert base not in source_module_classes

    def test_sample_top_level_keys(self, molrec_qm9_record):
        """ac-002: ``src[idx]`` is a flat dict with the documented keys."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        sample = src[0]
        assert set(sample.keys()) == {"Z", "pos", "box", "targets"}

    def test_sample_tensor_shapes(self, molrec_qm9_record):
        """ac-002: per-frame Z/pos/box shapes track the heterogeneous frame."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        # Frame 0 in the fixture is CH4 (5 atoms).
        n0 = len(molrec_qm9_record.expected_Z[0])
        sample = src[0]
        assert sample["Z"].shape == (n0,)
        assert sample["pos"].shape == (n0, 3)
        assert sample["box"].shape == (3, 3)

    def test_sample_tensor_dtypes(self, molrec_qm9_record):
        """ac-003: Z is long; pos/box/targets are float32 (source casts)."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        sample = src[0]
        assert sample["Z"].dtype == torch.long
        assert sample["pos"].dtype == torch.float32
        assert sample["box"].dtype == torch.float32
        for value in sample["targets"].values():
            assert value.dtype == torch.float32

    def test_heterogeneous_atom_counts(self, molrec_qm9_record):
        """ac-002: variable-length frames keep their own atom counts."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        counts = [src[i]["Z"].shape[0] for i in range(len(src))]
        assert counts == [len(z) for z in molrec_qm9_record.expected_Z]

    def test_target_keys_strip_teacher_prefix(self, molrec_qm9_record):
        """ac-003: target keys are observable names with ``teacherA.`` removed."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        target_keys = set(src[0]["targets"].keys())
        assert target_keys == set(molrec_qm9_record.scalars.keys())
        # The prefix must be gone (no leaked "teacherA." anywhere).
        assert not any(key.startswith("teacherA.") for key in target_keys)

    def test_scalar_target_has_shape_one(self, molrec_qm9_record):
        """ac-003: a scalar (graph-level) target is shaped ``(1,)`` per frame."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        targets = src[0]["targets"]
        for value in targets.values():
            assert value.shape == (1,)


# ---------------------------------------------------------------------------
# Dynamic per-instance schema — ac-004
# ---------------------------------------------------------------------------


class TestMolRecDynamicSchema:
    def test_no_class_level_target_schema(self):
        """ac-004: unlike QM9Source/RevMD17Source, no frozen class attribute."""
        assert "TARGET_SCHEMA" not in vars(MolRecSource)

    def test_target_schema_is_per_instance(self, molrec_qm9_record):
        """ac-004: ``target_schema`` is an instance-level :class:`TargetSchema`."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        assert isinstance(src.target_schema, TargetSchema)

    def test_scalar_only_record_schema(self, molrec_qm9_record):
        """ac-004 / ac-006: scalars -> graph_level; atom_level is empty."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        schema = src.target_schema
        assert set(schema.graph_level) == set(molrec_qm9_record.scalars.keys())
        assert schema.atom_level == frozenset()

    def test_force_record_schema_splits_scalar_and_vector(self, molrec_force_record):
        """ac-004: scalar energy -> graph_level; vector forces -> atom_level."""
        src = MolRecSource(molrec_force_record.path, "teacherA")
        schema = src.target_schema
        assert "energy" in schema.graph_level
        assert "forces" in schema.atom_level
        assert "forces" not in schema.graph_level
        assert "energy" not in schema.atom_level


# ---------------------------------------------------------------------------
# source_id — ac-005
# ---------------------------------------------------------------------------


class TestMolRecSourceID:
    def test_source_id_endswith_teacher_tag(self, molrec_force_record):
        """ac-005: source_id ends with ``:teacher=<teacher_id>``."""
        src = MolRecSource(molrec_force_record.path, "teacherA")
        assert src.source_id.endswith(":teacher=teacherA")

    def test_source_id_is_deterministic(self, molrec_force_record):
        """ac-005: two constructions of the same view share one id."""
        a = MolRecSource(molrec_force_record.path, "teacherA").source_id
        b = MolRecSource(molrec_force_record.path, "teacherA").source_id
        assert a == b

    def test_source_id_differs_per_teacher(self, molrec_force_record):
        """ac-005: teacherA and teacherB on one record get distinct ids."""
        a = MolRecSource(molrec_force_record.path, "teacherA").source_id
        b = MolRecSource(molrec_force_record.path, "teacherB").source_id
        assert a != b
        assert a.endswith(":teacher=teacherA")
        assert b.endswith(":teacher=teacherB")

    def test_source_id_subset_differs_from_full(self, molrec_qm9_record):
        """ac-005: a ``total=`` subset view differs from the full view."""
        full = MolRecSource(molrec_qm9_record.path, "teacherA").source_id
        subset = MolRecSource(molrec_qm9_record.path, "teacherA", total=2).source_id
        assert full != subset


# ---------------------------------------------------------------------------
# Runtime errors / teacher isolation — ac-006
# ---------------------------------------------------------------------------


class TestMolRecRuntimeErrors:
    def test_unknown_teacher_lists_available_prefixes(self, molrec_force_record):
        """ac-006: unknown teacher raises ValueError enumerating real prefixes."""
        with pytest.raises(ValueError) as exc:
            MolRecSource(molrec_force_record.path, "teacherZ")
        msg = str(exc.value)
        # Both available teacher prefixes must be named in the message.
        assert "teacherA" in msg
        assert "teacherB" in msg

    def test_teacher_a_and_b_targets_isolated(self, molrec_force_record):
        """ac-006: a teacherA view never sees teacherB's energy values."""
        src_a = MolRecSource(molrec_force_record.path, "teacherA")
        src_b = MolRecSource(molrec_force_record.path, "teacherB")
        e_a = float(src_a[0]["targets"]["energy"].item())
        e_b = float(src_b[0]["targets"]["energy"].item())
        assert e_a == pytest.approx(float(molrec_force_record.energy_a[0]), abs=1e-6)
        assert e_b == pytest.approx(float(molrec_force_record.energy_b[0]), abs=1e-6)
        assert e_a != e_b

    def test_teacher_a_view_has_no_teacher_b_keys(self, molrec_force_record):
        """ac-006: a teacherA view exposes only teacherA observables."""
        src_a = MolRecSource(molrec_force_record.path, "teacherA")
        keys = set(src_a[0]["targets"].keys())
        assert keys == {"energy", "forces"}


# ---------------------------------------------------------------------------
# Scientific roundtrip — QM9 scalars — ac-007
# ---------------------------------------------------------------------------


class TestMolRecQM9Roundtrip:
    def test_all_fifteen_scalars_lossless(self, molrec_qm9_record):
        """ac-007: all 15 float32 scalar targets survive write -> read exactly.

        molrs upcasts to float64 on disk; casting f32->f64->f32 is exact, so
        the source's float32 output must equal the original float32 values.
        Probability/energy exact column is 1e-10; we use 1e-6 numerical to also
        cover the storage round-trip safely.
        """
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        assert len(src) == molrec_qm9_record.n_frames
        for frame_idx in range(len(src)):
            targets = src[frame_idx]["targets"]
            assert set(targets.keys()) == set(molrec_qm9_record.scalars.keys())
            for name, original in molrec_qm9_record.scalars.items():
                got = targets[name]
                assert got.dtype == torch.float32
                expected = np.float32(original[frame_idx])
                assert np.allclose(got.numpy(), expected, atol=1e-6)

    def test_Z_lossless_and_long(self, molrec_qm9_record):
        """ac-007: atomic numbers map from symbols and are LongTensors."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        for frame_idx in range(len(src)):
            z = src[frame_idx]["Z"]
            assert z.dtype == torch.long
            assert z.tolist() == molrec_qm9_record.expected_Z[frame_idx]

    def test_pos_lossless_and_float32(self, molrec_qm9_record):
        """ac-007: positions are float32 and round-trip losslessly."""
        src = MolRecSource(molrec_qm9_record.path, "teacherA")
        for frame_idx in range(len(src)):
            pos = src[frame_idx]["pos"]
            assert pos.dtype == torch.float32
            expected = molrec_qm9_record.positions[frame_idx]
            assert pos.shape == expected.shape
            assert torch.allclose(pos, torch.from_numpy(expected), atol=1e-8)


# ---------------------------------------------------------------------------
# Scientific roundtrip — forces + energy — ac-008
# ---------------------------------------------------------------------------


class TestMolRecForceRoundtrip:
    def test_forces_shape_per_frame(self, molrec_force_record):
        """ac-008: a vector/atom target is shaped ``(n_atoms, 3)`` per frame."""
        src = MolRecSource(molrec_force_record.path, "teacherA")
        n_atoms = len(molrec_force_record.expected_Z)
        for frame_idx in range(len(src)):
            forces = src[frame_idx]["targets"]["forces"]
            assert forces.shape == (n_atoms, 3)
            assert forces.dtype == torch.float32

    def test_forces_lossless(self, molrec_force_record):
        """ac-008: force values round-trip losslessly (force numerical 1e-4)."""
        src = MolRecSource(molrec_force_record.path, "teacherA")
        for frame_idx in range(len(src)):
            forces = src[frame_idx]["targets"]["forces"]
            expected = molrec_force_record.forces_a[frame_idx]
            assert torch.allclose(forces, torch.from_numpy(expected), atol=1e-6)

    def test_energy_shape_and_lossless(self, molrec_force_record):
        """ac-008: scalar energy is ``(1,)`` float32 and lossless."""
        src = MolRecSource(molrec_force_record.path, "teacherA")
        for frame_idx in range(len(src)):
            energy = src[frame_idx]["targets"]["energy"]
            assert energy.shape == (1,)
            assert energy.dtype == torch.float32
            expected = np.float32(molrec_force_record.energy_a[frame_idx])
            assert np.allclose(energy.numpy(), expected, atol=1e-6)
