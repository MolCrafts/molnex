"""Unit tests for :mod:`molix.store.block` — READY-tier column block store.

Spec: ``.claude/specs/streaming-data-tiers-02-block-format.md`` (Design §§ 1-5 +
Testing strategy). Every fixture float is a multiple of ``0.5`` and therefore
exactly representable in ``float32``: **no FP tolerance appears anywhere in this
file**, digests are host-independent, and every equality is ``torch.equal`` /
``==`` / ``sha256``.

Layout mirror: ``src/molix/store/block.py`` → this file. One test class per
public type (:class:`SamplingUnit`, :class:`BlockManifest`,
:class:`BlockWriter`, :class:`BlockReader`); the layering guard is a
module-level test because it targets the module, not a type.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from molix.batch.collate import collate_molecules, collate_packed
from molix.batch.sampler import TaskMixtureBatchSampler
from molix.store.block import BlockManifest, BlockReader, BlockWriter, SamplingUnit
from molix.store.dataset import PackedView
from tests.test_molix.conftest import (
    BLOCK_PROVENANCE,
    frame_columns,
    halves,
    sampling_unit,
    unit_a,
    unit_b,
    unit_c,
    unit_samples,
    valence_columns,
    write_block,
)

# ---------------------------------------------------------------------------
# Fixture data — two-level ragged units (spec § Domain basis)
#
# unit 0 "mol-a"  N=3  n_frames=2   task "equilibrium"  (2 bonds, 1 angle, 0 propers)
# unit 1 "mol-b"  N=4  n_frames=3   task "relax"        (3 bonds, 2 angles, 1 proper)
# unit 2 "scan-c" N=4  n_frames=5   task "torsion"      (3 bonds, 2 angles, 1 proper)
#
# Units 0+1 alone reproduce the spec's hand-computed arithmetic fixture
# N = (3, 4), n_frames = (2, 3).
# ---------------------------------------------------------------------------

# The 24 columns frozen from unit "mol-a": 5 required metadata + Z + pos +
# 2 frame targets + 3 valence families (columns + derived ``_n``).
_EXPECTED_COLUMNS = frozenset(
    {
        "n_atoms",
        "n_frames",
        "task",
        "unit_key_len",
        "unit_key_bytes",
        "Z",
        "pos",
        "energy",
        "forces",
        "bond_n",
        "bond_atomi",
        "bond_atomj",
        "bond_type",
        "angle_n",
        "angle_atomi",
        "angle_atomj",
        "angle_atomk",
        "angle_type",
        "proper_n",
        "proper_atomi",
        "proper_atomj",
        "proper_atomk",
        "proper_atoml",
        "proper_type",
    }
)

# Commit record: unit_index (i8) + n_columns (i4) + n_columns x committed_len
# (i8) + crc32 (u4) = 16 + 8*C bytes (spec § Domain basis).
_RECORD_SIZE = 16 + 8 * len(_EXPECTED_COLUMNS)

_POINTER_NAMES = (
    "mol_atom_ptr",
    "mol_frame_ptr",
    "mol_frame_atom_ptr",
    "frame_mol",
    "unit_key_ptr",
    "bond_ptr",
    "angle_ptr",
    "proper_ptr",
)


def _column_sha256(root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.glob("*.bin"))
    }


def _overwrite_bytes(path: Path, offset: int, payload: bytes) -> None:
    """Write *payload* at *offset* through a second fd (sealed columns are 0o444)."""
    mode = stat.S_IMODE(path.stat().st_mode)
    os.chmod(path, 0o644)
    try:
        with open(path, "r+b") as handle:
            handle.seek(offset)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.chmod(path, mode)


def _is_file_backed(column: torch.Tensor, path: Path) -> bool:
    """Zero-copy witness for a mapped column (spec § 3 + its numpy contingency).

    Primary: ``untyped_storage().filename`` equals the column path. torch only
    records that for ``shared=True`` (MAP_SHARED) mappings, which cannot open a
    sealed ``0o444`` column read-write, so the spec's documented fallback
    (``torch.from_numpy(np.memmap(...))``) is accepted two ways: a ``np.memmap``
    in the array's base chain that shares memory with the column, or — when the
    mapping is reachable through neither handle — page-cache coherence, i.e. a
    byte written through a second fd is observed through the tensor, which is
    false for any copy.
    """
    filename = column.untyped_storage().filename
    if filename is not None:
        return Path(filename) == path

    array = column.numpy()
    base = array
    while base is not None and not isinstance(base, np.memmap):
        base = getattr(base, "base", None)
    if isinstance(base, np.memmap):
        return Path(base.filename) == path and bool(np.shares_memory(array, base))

    flat = column.reshape(-1)
    original = struct.pack("<f", float(flat[0]))
    probe = struct.pack("<f", 12.5)
    assert original != probe
    _overwrite_bytes(path, 0, probe)
    try:
        return float(flat[0]) == 12.5
    finally:
        _overwrite_bytes(path, 0, original)


def _assert_td_equal(actual: TensorDict, expected: TensorDict, path: str = "") -> None:
    """Leaf-for-leaf equality: key sets, ``batch_size``, dtypes, values."""
    assert set(actual.keys()) == set(expected.keys()), f"key sets differ at {path or '/'}"
    assert tuple(actual.batch_size) == tuple(expected.batch_size), f"batch_size differs at {path}"
    for key in sorted(actual.keys()):
        left, right = actual[key], expected[key]
        if isinstance(left, TensorDict) or isinstance(right, TensorDict):
            _assert_td_equal(left, right, f"{path}/{key}")
            continue
        assert left.dtype == right.dtype, f"dtype differs at {path}/{key}"
        assert torch.equal(left, right), f"values differ at {path}/{key}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def units() -> list[SamplingUnit]:
    return [unit_a(), unit_b(), unit_c()]


@pytest.fixture
def sealed_block(tmp_path: Path, units: list[SamplingUnit]) -> Path:
    return write_block(tmp_path / "block", units)


@pytest.fixture
def reader(sealed_block: Path) -> BlockReader:
    return BlockReader(sealed_block)


@pytest.fixture
def two_unit_block(tmp_path: Path) -> Path:
    """Spec § Testing-strategy arithmetic fixture: N = (3, 4), n_frames = (2, 3)."""
    return write_block(tmp_path / "two", [unit_a(), unit_b()])


# ---------------------------------------------------------------------------
# SamplingUnit
# ---------------------------------------------------------------------------


class TestSamplingUnit:
    def test_derives_n_atoms_and_n_frames(self):
        unit = unit_b()
        assert unit.n_atoms == 4
        assert unit.n_frames == 3

    def test_is_frozen(self):
        unit = unit_a()
        with pytest.raises(AttributeError):
            unit.key = "other"  # type: ignore[misc]

    def test_rejects_unit_without_pos(self):
        with pytest.raises(ValueError, match="'pos'"):
            SamplingUnit(
                key="mol-a",
                task="equilibrium",
                topology={"Z": torch.tensor([1, 1], dtype=torch.int16)},
                frames={"energy": halves(2, 0)},
            )

    def test_rejects_pos_rows_not_multiple_of_n_atoms(self):
        with pytest.raises(ValueError, match="'pos'"):
            SamplingUnit(
                key="mol-a",
                task="equilibrium",
                topology={"Z": torch.tensor([8, 1, 1], dtype=torch.int16)},
                frames={"pos": halves(7 * 3, 0).reshape(7, 3)},
            )

    def test_rejects_frame_space_row_count_mismatch(self):
        with pytest.raises(ValueError, match="'energy'"):
            SamplingUnit(
                key="mol-a",
                task="equilibrium",
                topology={"Z": torch.tensor([8, 1, 1], dtype=torch.int16)},
                frames={
                    "pos": halves(6 * 3, 0).reshape(6, 3),
                    "energy": halves(3, 0),
                },
            )

    def test_rejects_frame_atom_space_row_count_mismatch(self):
        with pytest.raises(ValueError, match="'forces'"):
            SamplingUnit(
                key="mol-a",
                task="equilibrium",
                topology={"Z": torch.tensor([8, 1, 1], dtype=torch.int16)},
                frames={
                    "pos": halves(6 * 3, 0).reshape(6, 3),
                    "forces": halves(5 * 3, 0).reshape(5, 3),
                },
            )

    def test_rejects_topology_atom_index_outside_range(self):
        topology = {
            "Z": torch.tensor([8, 1, 1], dtype=torch.int16),
            "bond_atomi": torch.tensor([0, 0], dtype=torch.int32),
            "bond_atomj": torch.tensor([1, 3], dtype=torch.int32),  # 3 >= N = 3
        }
        with pytest.raises(ValueError, match="bond_atomj"):
            SamplingUnit(
                key="mol-a",
                task="equilibrium",
                topology=topology,
                frames={"pos": halves(6 * 3, 0).reshape(6, 3)},
            )

    def test_rejects_unroutable_column_name(self):
        with pytest.raises(ValueError, match="mystery"):
            SamplingUnit(
                key="mol-a",
                task="equilibrium",
                topology={
                    "Z": torch.tensor([8, 1, 1], dtype=torch.int16),
                    "mystery": torch.tensor([1, 2, 3], dtype=torch.int32),
                },
                frames={"pos": halves(6 * 3, 0).reshape(6, 3)},
            )


# ---------------------------------------------------------------------------
# BlockManifest
# ---------------------------------------------------------------------------


class TestBlockManifest:
    def test_schema_version_is_one(self, sealed_block: Path):
        assert BlockManifest.read(sealed_block).schema_version == 1

    def test_unsealed_before_seal(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(unit_a())
        assert BlockManifest.read(tmp_path / "b").sealed is False

    def test_sealed_after_seal(self, sealed_block: Path):
        assert BlockManifest.read(sealed_block).sealed is True

    def test_column_declares_dtype_inner_shape_space_and_route(self, sealed_block: Path):
        spec = BlockManifest.read(sealed_block).column("pos")
        assert spec["dtype"] == "<f4"
        assert list(spec["inner_shape"]) == [3]
        assert spec["space"] == "frame_atom"
        assert spec["route"] == "atoms.pos"

    def test_column_routes_follow_the_naming_table(self, sealed_block: Path):
        manifest = BlockManifest.read(sealed_block)
        assert manifest.column("Z")["route"] == "atoms.Z"
        assert manifest.column("energy")["route"] == "graphs.targets.energy"
        assert manifest.column("forces")["route"] == "atoms.targets.forces"
        assert manifest.column("bond_atomi")["route"] == "bonds.bond_index"
        assert manifest.column("bond_type")["route"] == "bonds.bond_types"
        assert manifest.column("angle_atomk")["route"] == "angles.atomk"
        assert manifest.column("proper_type")["route"] == "propers.type"

    def test_column_spaces_follow_the_taxonomy(self, sealed_block: Path):
        manifest = BlockManifest.read(sealed_block)
        assert manifest.column("n_atoms")["space"] == "molecule"
        assert manifest.column("Z")["space"] == "molecule_atom"
        assert manifest.column("bond_atomi")["space"] == "molecule_term:bond"
        assert manifest.column("energy")["space"] == "frame"
        assert manifest.column("pos")["space"] == "frame_atom"
        assert manifest.column("unit_key_bytes")["space"] == "bytes"

    def test_write_atomic_roundtrips_the_json_contract(self, sealed_block: Path, tmp_path: Path):
        destination = tmp_path / "copy"
        destination.mkdir()
        BlockManifest.read(sealed_block).write_atomic(destination)
        assert json.loads((destination / "manifest.json").read_text()) == json.loads(
            (sealed_block / "manifest.json").read_text()
        )
        assert [p.name for p in destination.iterdir()] == ["manifest.json"]

    def test_block_digest_is_sha256_of_canonical_json(self, sealed_block: Path):
        raw = json.loads((sealed_block / "manifest.json").read_text())
        canonical = json.dumps(
            {
                key: raw[key]
                for key in ("columns", "rows", "split", "task_names", "units", "digests")
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert raw["block_digest"] == hashlib.sha256(canonical).hexdigest()

    def test_block_digest_covers_the_task_dictionary(self, sealed_block: Path):
        """2026-08-14 revision: ``task_names`` is part of the digested core.

        A truncated task dictionary breaks the ``task`` code -> label mapping
        without touching one payload byte, so no column sha256 can witness it.
        The witness is the digested core: the recomputed ``block_digest`` moves
        away from the value a consumer recorded externally (the molhub READY
        ledger copies it verbatim), while :meth:`BlockReader.verify` — which
        hashes column bytes only — stays clean.
        """
        recorded = BlockManifest.read(sealed_block).block_digest
        path = sealed_block / "manifest.json"
        document = json.loads(path.read_text())
        assert document["task_names"] == ["equilibrium", "relax", "torsion"]
        document["task_names"] = ["equilibrium"]
        path.write_text(json.dumps(document))

        assert BlockManifest.read(sealed_block).block_digest != recorded
        assert BlockReader(sealed_block).verify() is None
        assert json.loads(path.read_text())["block_digest"] == recorded

    def test_block_digest_is_recomputed_not_trusted_as_stored_text(self, sealed_block: Path):
        """A forged ``block_digest`` field is overwritten by the derivation."""
        path = sealed_block / "manifest.json"
        document = json.loads(path.read_text())
        honest = document["block_digest"]
        document["block_digest"] = "0" * 64
        path.write_text(json.dumps(document))
        assert BlockManifest.read(sealed_block).block_digest == honest

    def test_block_digest_ignores_provenance_and_location(
        self, tmp_path: Path, units: list[SamplingUnit]
    ):
        first = write_block(tmp_path / "one", units)
        writer = BlockWriter(
            tmp_path / "two",
            split="train",
            provenance={**BLOCK_PROVENANCE, "created_at": "2027-01-01T00:00:00+00:00"},
        )
        for unit in units:
            writer.append_unit(unit)
        writer.seal()
        digest_one = json.loads((first / "manifest.json").read_text())["block_digest"]
        digest_two = json.loads((tmp_path / "two" / "manifest.json").read_text())["block_digest"]
        assert digest_one == digest_two

    def test_digests_match_column_file_sha256(self, sealed_block: Path):
        digests = json.loads((sealed_block / "manifest.json").read_text())["digests"]
        assert set(digests) == _EXPECTED_COLUMNS
        for name, digest in digests.items():
            expected = hashlib.sha256((sealed_block / f"{name}.bin").read_bytes()).hexdigest()
            assert digest == expected

    def test_read_rejects_schema_version_mismatch(self, sealed_block: Path):
        path = sealed_block / "manifest.json"
        raw = json.loads(path.read_text())
        raw["schema_version"] = 2
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            BlockManifest.read(sealed_block)

    # -- units / rows accessors --------------------------------------------
    # Spec ``streaming-data-tiers-04-scheduler-integration-01-tables`` § 1: two
    # typed reads of the digest-covered document (``_DIGEST_FIELDS`` already
    # covers ``units`` / ``rows``), authoritative only on a sealed manifest.

    def test_units_equals_the_appended_unit_count(self, two_unit_block: Path):
        """ac-001: ``units`` is the committed unit count ``M`` of the sealed block."""
        manifest = BlockManifest.read(two_unit_block)
        assert isinstance(manifest.units, int)
        assert manifest.units == 2

    def test_rows_equals_the_stored_document_entry_for_entry(self, two_unit_block: Path):
        """ac-001: ``rows`` is a ``Mapping[str, int]`` view of the stored field."""
        manifest = BlockManifest.read(two_unit_block)
        stored = json.loads((two_unit_block / "manifest.json").read_text())["rows"]
        assert isinstance(manifest.rows, Mapping)
        assert dict(manifest.rows) == stored
        assert all(isinstance(value, int) for value in manifest.rows.values())

    def test_rows_match_the_hand_computed_two_level_arithmetic(self, two_unit_block: Path):
        """ac-001: N = (3, 4), n_frames = (2, 3) — the fixture at ``:729-752``.

        molecule = M = 2; molecule_atom = ΣN_m = 3 + 4 = 7;
        frame_atom = Σ n_m·N_m = 2·3 + 3·4 = 18.
        """
        rows = BlockManifest.read(two_unit_block).rows
        assert rows["molecule"] == 2
        assert rows["molecule_atom"] == 7
        assert rows["frame_atom"] == 18

    def test_rows_carries_one_key_per_declared_valence_family(self, two_unit_block: Path):
        """ac-001: ``molecule_term:<family>`` exists exactly for declared families.

        Counts are the per-unit term tables: bonds 2 + 3, angles 1 + 2,
        propers 0 + 1; ``improper`` / ``internal`` are never declared here.
        """
        rows = BlockManifest.read(two_unit_block).rows
        assert {name for name in rows if name.startswith("molecule_term:")} == {
            "molecule_term:bond",
            "molecule_term:angle",
            "molecule_term:proper",
        }
        assert rows["molecule_term:bond"] == 5
        assert rows["molecule_term:angle"] == 3
        assert rows["molecule_term:proper"] == 1

    def test_rows_omits_frame_for_a_block_declaring_no_frame_space_column(self, tmp_path: Path):
        """ac-001: a ``pos``-only block has no ``"frame"`` key; ``F`` stays a reader fact."""
        unit = SamplingUnit(
            key="bare",
            task="equilibrium",
            topology={"Z": torch.tensor([8, 1, 1], dtype=torch.int16)},
            frames={"pos": halves(6 * 3, 0).reshape(6, 3)},
        )
        rows = BlockManifest.read(write_block(tmp_path / "frameless", [unit])).rows
        assert "frame" not in rows
        assert rows["frame_atom"] == 6
        assert not [name for name in rows if name.startswith("molecule_term:")]

    def test_units_and_rows_are_stale_before_seal(self, tmp_path: Path):
        """ac-003: the manifest is rewritten on freeze / task registration / seal only.

        Two units of the *same* task trigger exactly one manifest write (the
        schema freeze, before the first unit's bytes), so the document still
        reports the pre-append counts while the writer holds two.
        """
        root = tmp_path / "unsealed"
        writer = BlockWriter(root, split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(sampling_unit("mol-a", "equilibrium", [8, 1, 1], 2, 0))
        writer.append_unit(sampling_unit("mol-a2", "equilibrium", [8, 1, 1], 2, 10))
        manifest = BlockManifest.read(root)
        assert manifest.sealed is False
        assert writer.unit_count == 2
        assert manifest.units < writer.unit_count
        assert dict(manifest.rows) == {}

    def test_units_and_rows_perform_no_digest_work(self, two_unit_block: Path, monkeypatch):
        """ac-002: ``verify()`` stays the only code path that hashes column bytes."""

        def boom(*args: object, **kwargs: object) -> str:
            raise RuntimeError("a manifest accessor must never hash column bytes")

        monkeypatch.setattr("molix.store.block._sha256_file", boom)
        stored = json.loads((two_unit_block / "manifest.json").read_text())["rows"]
        manifest = BlockManifest.read(two_unit_block)
        assert manifest.units == 2
        assert dict(manifest.rows) == stored


# ---------------------------------------------------------------------------
# BlockWriter
# ---------------------------------------------------------------------------


class TestBlockWriter:
    def test_append_unit_returns_sequential_indices(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="train", provenance=BLOCK_PROVENANCE)
        assert writer.append_unit(unit_a()) == 0
        assert writer.append_unit(unit_b()) == 1

    def test_unit_count_tracks_appends(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="train", provenance=BLOCK_PROVENANCE)
        assert writer.unit_count == 0
        writer.append_unit(unit_a())
        assert writer.unit_count == 1

    def test_schema_is_frozen_from_the_firstsampling_unit(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(unit_a())
        columns = json.loads((tmp_path / "b" / "manifest.json").read_text())["columns"]
        assert set(columns) == _EXPECTED_COLUMNS

    def test_directory_holds_one_bin_per_column_plus_manifest_and_log(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(unit_a())
        names = {path.name for path in (tmp_path / "b").iterdir()}
        assert names == {"manifest.json", "commit.log"} | {f"{c}.bin" for c in _EXPECTED_COLUMNS}

    def test_commit_log_grows_one_fixed_size_record_persampling_unit(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(unit_a())
        assert (tmp_path / "b" / "commit.log").stat().st_size == _RECORD_SIZE
        writer.append_unit(unit_b())
        assert (tmp_path / "b" / "commit.log").stat().st_size == 2 * _RECORD_SIZE

    def test_committed_bytes_equals_the_column_payload_size(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(unit_a())
        writer.append_unit(unit_b())
        on_disk = sum(path.stat().st_size for path in (tmp_path / "b").glob("*.bin"))
        assert writer.committed_bytes == on_disk

    def test_append_after_seal_raises(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(unit_a())
        writer.seal()
        with pytest.raises(ValueError):
            writer.append_unit(unit_b())

    def test_second_unit_missing_a_declared_column_raises(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(unit_a())
        bare = SamplingUnit(
            key="mol-d",
            task="equilibrium",
            topology={"Z": torch.tensor([8, 1, 1], dtype=torch.int16)},
            frames=frame_columns(2, 3, 0),
        )
        with pytest.raises(ValueError, match="bond_atomi"):
            writer.append_unit(bare)

    def test_second_unit_dtype_mismatch_raises(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(unit_a())
        widened = SamplingUnit(
            key="mol-e",
            task="equilibrium",
            topology={
                **valence_columns(3),
                "Z": torch.tensor([8, 1, 1], dtype=torch.int64),  # frozen as <i2
            },
            frames=frame_columns(2, 3, 0),
        )
        with pytest.raises(ValueError, match="'Z'"):
            writer.append_unit(widened)

    def test_append_unit_takes_no_split_parameter(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="valid", provenance=BLOCK_PROVENANCE)
        with pytest.raises(TypeError):
            writer.append_unit(unit_a(), split="train")  # type: ignore[call-arg]

    def test_split_is_recorded_once_and_is_not_a_column(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="valid", provenance=BLOCK_PROVENANCE)
        writer.append_unit(unit_a())
        writer.append_unit(unit_b())
        raw = json.loads((tmp_path / "b" / "manifest.json").read_text())
        assert raw["split"] == "valid"
        assert not [name for name in raw["columns"] if "split" in name]

    def test_resume_truncates_a_torn_column_tail(self, tmp_path: Path, units: list[SamplingUnit]):
        reference = _column_sha256(write_block(tmp_path / "clean", units))

        root = tmp_path / "torn"
        writer = BlockWriter(root, split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(units[0])
        writer.append_unit(units[1])
        del writer  # simulate the crash: no seal, handles dropped
        with open(root / "pos.bin", "ab") as handle:  # uncommitted tail
            handle.write(b"\x7f" * 24)

        resumed = BlockWriter.resume(root)
        resumed.append_unit(units[2])
        resumed.seal()
        assert _column_sha256(root) == reference

    def test_resume_discards_a_truncated_commit_record(
        self, tmp_path: Path, units: list[SamplingUnit]
    ):
        reference = _column_sha256(write_block(tmp_path / "clean", units))

        root = tmp_path / "torn"
        writer = BlockWriter(root, split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(units[0])
        writer.append_unit(units[1])
        del writer
        with open(root / "pos.bin", "ab") as handle:
            handle.write(b"\x7f" * 24)
        with open(root / "commit.log", "ab") as handle:  # half-written record
            handle.write(b"\x01" * (_RECORD_SIZE // 2))

        resumed = BlockWriter.resume(root)
        resumed.append_unit(units[2])
        resumed.seal()
        assert _column_sha256(root) == reference

    def test_resume_continues_at_the_next_unit_index(
        self, tmp_path: Path, units: list[SamplingUnit]
    ):
        root = tmp_path / "torn"
        writer = BlockWriter(root, split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(units[0])
        writer.append_unit(units[1])
        del writer
        with open(root / "pos.bin", "ab") as handle:
            handle.write(b"\x7f" * 24)

        resumed = BlockWriter.resume(root)
        assert resumed.unit_count == 2
        assert resumed.append_unit(units[2]) == 2

    def test_seal_makes_column_files_read_only(self, sealed_block: Path):
        for path in sealed_block.glob("*.bin"):
            assert stat.S_IMODE(path.stat().st_mode) == 0o444


# ---------------------------------------------------------------------------
# BlockReader
# ---------------------------------------------------------------------------


class TestBlockReader:
    def test_refuses_an_unsealed_block(self, tmp_path: Path):
        writer = BlockWriter(tmp_path / "b", split="train", provenance=BLOCK_PROVENANCE)
        writer.append_unit(unit_a())
        with pytest.raises(ValueError):
            BlockReader(tmp_path / "b")

    def test_frame_columns_read_back_bit_for_bit(
        self, reader: BlockReader, units: list[SamplingUnit]
    ):
        for name in ("pos", "energy", "forces"):
            expected = torch.cat([unit.frames[name] for unit in units], dim=0)
            assert torch.equal(reader.column(name), expected)

    def test_topology_columns_read_back_bit_for_bit(
        self, reader: BlockReader, units: list[SamplingUnit]
    ):
        names = ("Z", "bond_atomi", "bond_atomj", "bond_type", "angle_atomk", "proper_atoml")
        for name in names:
            expected = torch.cat([unit.topology[name] for unit in units], dim=0)
            assert torch.equal(reader.column(name), expected)

    def test_per_unit_term_counts_are_stored(self, reader: BlockReader):
        assert reader.column("bond_n").tolist() == [2, 3, 3]
        assert reader.column("angle_n").tolist() == [1, 2, 2]
        assert reader.column("proper_n").tolist() == [0, 1, 1]

    def test_unit_count_and_frame_count(self, reader: BlockReader):
        assert reader.unit_count == 3
        assert reader.frame_count == 10

    def test_len_equals_frame_count(self, reader: BlockReader):
        assert len(reader) == 10

    def test_max_atoms(self, reader: BlockReader):
        assert reader.max_atoms == 4

    def test_unit_frames_returns_the_appended_frames(self, reader: BlockReader):
        assert reader.unit_frames(0).tolist() == [0, 1]
        assert reader.unit_frames(1).tolist() == [2, 3, 4]
        assert reader.unit_frames(2).tolist() == [5, 6, 7, 8, 9]

    def test_atom_counts_are_per_frame(self, reader: BlockReader):
        counts = reader.atom_counts
        assert counts.dtype == torch.long
        assert counts.tolist() == [3, 3, 4, 4, 4, 4, 4, 4, 4, 4]

    def test_task_codes_decode_via_task_names(self, reader: BlockReader):
        names = reader.task_names
        assert set(names) == {"equilibrium", "relax", "torsion"}
        assert reader.task_codes.dtype == torch.int32
        decoded = [names[code] for code in reader.task_codes.tolist()]
        assert decoded == ["equilibrium"] * 2 + ["relax"] * 3 + ["torsion"] * 5

    def test_group_codes_equal_the_owning_unit_row(self, reader: BlockReader):
        assert reader.group_codes.dtype == torch.int32
        assert reader.group_codes.tolist() == [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]

    def test_unit_key_and_unit_index_roundtrip(self, reader: BlockReader):
        assert reader.unit_key(0) == "mol-a"
        assert reader.unit_key(2) == "scan-c"
        assert reader.unit_index("mol-b") == 1

    def test_edge_counts_raises_naming_the_deliberate_absence(self, reader: BlockReader):
        with pytest.raises(ValueError, match="edge") as excinfo:
            _ = reader.edge_counts
        assert "NeighborList" in str(excinfo.value)

    def test_verify_passes_on_a_sealed_block(self, reader: BlockReader):
        assert reader.verify() is None

    def test_verify_raises_naming_the_corrupted_column(
        self, sealed_block: Path, reader: BlockReader
    ):
        path = sealed_block / "pos.bin"
        flipped = bytes([path.read_bytes()[8] ^ 0xFF])
        _overwrite_bytes(path, 8, flipped)
        with pytest.raises(ValueError, match="'pos'"):
            reader.verify()

    def test_column_is_a_zero_copy_file_mapping(self, sealed_block: Path, reader: BlockReader):
        column = reader.column("pos")
        assert _is_file_backed(column, sealed_block / "pos.bin")

    # -- two-level row arithmetic (spec § Domain basis, hand-computed) ------

    def test_derived_pointers_match_hand_computed_literals(self, two_unit_block: Path):
        block = BlockReader(two_unit_block)
        assert block.mol_atom_ptr.tolist() == [0, 3, 7]
        assert block.mol_frame_ptr.tolist() == [0, 2, 5]
        assert block.mol_frame_atom_ptr.tolist() == [0, 6, 18]

    def test_frame_mol_matches_hand_computed_literal(self, two_unit_block: Path):
        assert BlockReader(two_unit_block).frame_mol.tolist() == [0, 0, 1, 1, 1]

    def test_atom_counts_match_hand_computed_literal(self, two_unit_block: Path):
        assert BlockReader(two_unit_block).atom_counts.tolist() == [3, 3, 4, 4, 4]

    def test_frame_three_reads_frame_atom_rows_ten_to_fourteen(self, two_unit_block: Path):
        # unit 1, k = 3 - 2 = 1 -> mol_frame_atom_ptr[1] + k*N = 6 + 4 = 10 .. 14
        block = BlockReader(two_unit_block)
        payload = block.frame_payload([3])
        assert torch.equal(payload["atoms"]["pos"], block.column("pos")[10:14])

    def test_frame_three_reads_topology_rows_three_to_seven(self, two_unit_block: Path):
        # unit 1 -> mol_atom_ptr[1] + [0, N) = 3 .. 7
        block = BlockReader(two_unit_block)
        payload = block.frame_payload([3])
        assert torch.equal(payload["atoms"]["Z"], block.column("Z")[3:7])

    def test_no_pointer_vector_is_stored_on_disk(self, two_unit_block: Path):
        columns = json.loads((two_unit_block / "manifest.json").read_text())["columns"]
        for name in _POINTER_NAMES:
            assert not (two_unit_block / f"{name}.bin").exists()
            assert name not in columns

    # -- edge cases --------------------------------------------------------

    def test_single_unit_block_reads_back(self, tmp_path: Path):
        root = write_block(tmp_path / "single", [unit_a()])
        block = BlockReader(root)
        assert block.unit_count == 1
        assert block.frame_count == 2
        assert torch.equal(block.column("pos"), unit_a().frames["pos"])

    def test_single_frame_unit_reads_back(self, tmp_path: Path):
        unit = sampling_unit("mol-single", "equilibrium", [8, 1, 1], 1, 0)
        root = write_block(tmp_path / "one-frame", [unit])
        block = BlockReader(root)
        assert block.frame_count == 1
        assert block.unit_frames(0).tolist() == [0]
        assert block.atom_counts.tolist() == [3]

    def test_unit_without_valence_columns_reads_back(self, tmp_path: Path):
        unit = SamplingUnit(
            key="bare",
            task="equilibrium",
            topology={"Z": torch.tensor([8, 1, 1], dtype=torch.int16)},
            frames={"pos": halves(6 * 3, 0).reshape(6, 3)},
        )
        root = write_block(tmp_path / "bare", [unit])
        block = BlockReader(root)
        columns = json.loads((root / "manifest.json").read_text())["columns"]
        assert not [name for name in columns if name.startswith(("bond_", "angle_", "proper_"))]
        payload = block.frame_payload([0, 1])
        assert "bonds" not in payload
        assert "angles" not in payload

    # -- equivalence oracle ------------------------------------------------

    def test_frame_payload_collate_equals_collate_molecules(
        self, reader: BlockReader, units: list[SamplingUnit]
    ):
        frames = [7, 0, 3]
        samples = unit_samples(units)
        expected = collate_molecules([samples[f] for f in frames])
        actual = collate_packed(PackedView(reader.frame_payload(frames)), range(len(frames)))
        _assert_td_equal(actual, expected)

    def test_frame_payload_collate_batch_sizes_are_hand_computed(self, reader: BlockReader):
        # frames (7, 0, 3) -> units (2, 0, 1): atoms 4+3+4, angles 2+1+2, propers 1+0+1
        batch = collate_packed(PackedView(reader.frame_payload([7, 0, 3])), range(3))
        assert tuple(batch.batch_size) == ()
        assert tuple(batch["atoms"].batch_size) == (11,)
        assert tuple(batch["edges"].batch_size) == (0,)  # blocks carry no edges
        assert tuple(batch["graphs"].batch_size) == (3,)
        assert tuple(batch["bonds"].batch_size) == ()
        assert tuple(batch["angles"].batch_size) == (5,)
        assert tuple(batch["propers"].batch_size) == (2,)

    # -- sampler metadata --------------------------------------------------

    def test_sampler_builds_a_deterministic_plan(self, reader: BlockReader):
        def build() -> list[list[int]]:
            sampler = TaskMixtureBatchSampler(
                reader,
                indices=list(range(reader.frame_count)),
                task_labels=[reader.task_names[c] for c in reader.task_codes.tolist()],
                group_ids=[int(g) for g in reader.group_codes.tolist()],
                mixture={"equilibrium": 0.25, "relax": 0.25, "torsion": 0.5},
                max_atoms=32,  # atoms only: edge_counts is unavailable by design
                seed=7,
            )
            return [list(batch) for batch in sampler]

        plan = build()
        assert plan
        assert plan == build()
        whole_units = ([0, 1], [2, 3, 4], [5, 6, 7, 8, 9])
        for batch in plan:
            assert tuple(batch) in [tuple(unit) for unit in whole_units]

    def test_sampler_plan_maps_only_count_columns(self, sealed_block: Path, monkeypatch):
        mapped: list[str] = []
        original = BlockReader._map_column

        def spy(self, name, *args, **kwargs):
            mapped.append(Path(str(name)).stem)
            return original(self, name, *args, **kwargs)

        monkeypatch.setattr(BlockReader, "_map_column", spy)

        block = BlockReader(sealed_block)
        TaskMixtureBatchSampler(
            block,
            indices=list(range(block.frame_count)),
            task_labels=[block.task_names[c] for c in block.task_codes.tolist()],
            group_ids=[int(g) for g in block.group_codes.tolist()],
            mixture={"equilibrium": 0.25, "relax": 0.25, "torsion": 0.5},
            max_atoms=32,
            seed=7,
        )
        allowed = {"n_atoms", "n_frames", "task", "unit_key_len", "unit_key_bytes"}
        allowed |= {f"{family}_n" for family in ("bond", "angle", "proper", "improper", "internal")}
        assert mapped
        assert set(mapped) <= allowed


# ---------------------------------------------------------------------------
# Layering guard (module-level: it targets the module, not a type)
# ---------------------------------------------------------------------------


def test_block_module_imports_no_tensordict_or_collate():
    """``block.py`` is IO-only: stdlib + torch (+ numpy fallback), nothing else.

    Mirrors the guard that keeps ``cache.py`` free of batch construction
    (``dynamic-batching-packed-collate-02-collate`` ac-005). Asserted over the
    parsed import statements, not raw text, so docstring prose may still name
    ``collate_packed``.
    """
    import molix.store.block as module

    source = Path(module.__file__).read_text()
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    forbidden = [name for name in imported if "tensordict" in name or "collate" in name]
    assert forbidden == []
    assert [name for name in imported if name.startswith("molix.store.")] == []
