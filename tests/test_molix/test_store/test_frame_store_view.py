"""Unit tests for :mod:`molix.store.frame_store_view` — multi-block dataset view.

Spec: ``.claude/specs/streaming-data-tiers-04-scheduler-integration-02-view.md``

Layout mirror: ``src/molix/store/frame_store_view.py`` → this file. One test class per
public type (:class:`FrameStoreDataset`); the layering / export / no-fabricated-zeros
guards are module-level tests because they target the module, not a type.

Every fixture float is a multiple of ``0.5`` and therefore exactly representable
in ``float32``: **no FP tolerance appears anywhere in this file**, and every
equality is ``torch.equal`` / ``==``.

Hand-computed reference fixture (spec § Domain basis), three sealed
``split="train"`` blocks::

    block  units (N, n_frames, task)                       F_b  M_b
    A      mol-a (3, 2, equilibrium), mol-b (4, 3, relax)    5    2
    B      scan-c (4, 5, torsion)                            5    1
    C      mol-d (3, 1, relax), scan-e (4, 4, torsion)       5    2

    frame_ptr   = (0, 5, 10, 15)
    unit_ptr    = (0, 2, 3, 5)
    len(view)   = 15
    atom_counts = (3,3,4,4,4, 4,4,4,4,4, 3,4,4,4,4)
    group_codes = (0,0,1,1,1, 2,2,2,2,2, 3,4,4,4,4)
    task_names  = ("equilibrium", "relax", "torsion")   # merged, first appearance
    task_codes  = (0,0,1,1,1, 2,2,2,2,2, 1,2,2,2,2)     # C remaps (0,1) -> (1,2)
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from tensordict import TensorDict

import molix.store.frame_store_view
from molix.batch.sampler import TaskMixtureBatchSampler, TokenBudgetBatchSampler
from molix.store.frame_store import FrameStoreReader, FrameStoreWriter
from molix.store.frame_store_view import (
    FrameStoreDataset,
    FrameStoreNotReadyError,
    FrameStoreSchemaMismatchError,
    FrameStoreSplitMismatchError,
)
from tests.conftest import frame

# ---------------------------------------------------------------------------
# Hand-computed goldens (spec § Domain basis)
# ---------------------------------------------------------------------------

_LEN = 15
_ATOM_COUNTS: tuple[int, ...] = (3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4)
_GROUP_CODES: tuple[int, ...] = (0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4)
_TASK_CODES: tuple[int, ...] = (0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2)
_TASK_NAMES: tuple[str, ...] = ("equilibrium", "relax", "torsion")
_TOTAL_UNITS = 5

_LOCATE_FRAMES: tuple[int, ...] = (0, 4, 5, 9, 10, 14)
_LOCATE_BLOCKS: tuple[int, ...] = (0, 0, 1, 1, 2, 2)
_LOCATE_LOCALS: tuple[int, ...] = (0, 4, 0, 4, 0, 4)

# Cross-block request of § Testing strategy: blocks C / A / B, atom counts 4, 3, 4.
_CROSS_FRAMES: tuple[int, ...] = (12, 1, 7)
_CROSS_ATOM_PTR: tuple[int, ...] = (0, 4, 7, 11)

# Sampler-integration epoch.
_MIXTURE: dict[str, float] = {"equilibrium": 0.0, "relax": 0.5, "torsion": 0.5}
_SAMPLER_MAX_ATOMS = 8
_SAMPLER_SEED = 42
_SAMPLER_FRAMES_PER_TASK: dict[str, int] = {"relax": 4, "torsion": 9}

_MODULE_PATH = Path(molix.store.frame_store_view.__file__)

# ---------------------------------------------------------------------------
# Fixture builders (in-module; consolidation with test_block.py is deferred by
# the spec's "Out of scope" § until a third store test needs the builder)
# ---------------------------------------------------------------------------

_PROVENANCE: dict[str, Any] = {
    "source": {"name": "molnex-unit-test", "coordinate": "n/a"},
    "generator": {"tool": "test_frame_store_view.py", "version": "1", "commit": "0"},
    "created_at": "2026-08-14T00:00:00+00:00",
    "units": {
        "pos": "angstrom",
        "energy": "kcal/mol",
        "forces": "kcal/mol/angstrom",
    },
}

#: ``(key, task, n_atoms, n_frames, halves offset)`` per block, in append order.
_BLOCK_UNITS: dict[str, tuple[tuple[str, str, int, int, int], ...]] = {
    "A": (("mol-a", "equilibrium", 3, 2, 0), ("mol-b", "relax", 4, 3, 40)),
    "B": (("scan-c", "torsion", 4, 5, 100),),
    "C": (("mol-d", "relax", 3, 1, 200), ("scan-e", "torsion", 4, 4, 300)),
}


def _halves(count: int, start: int) -> torch.Tensor:
    """``count`` distinct multiples of 0.5 — exactly representable in float32."""
    return (torch.arange(count, dtype=torch.float32) + float(start)) * 0.5


def _i32(values: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int32)


def _valence(n_atoms: int) -> dict[str, torch.Tensor]:
    """Covalent topology of a linear chain over *n_atoms* atoms (int32 columns)."""
    if n_atoms == 3:
        bonds = ([0, 0], [1, 2], [1, 1])
        angles = ([1], [0], [2], [1])
        propers: tuple[list[int], ...] = ([], [], [], [], [])
    else:
        bonds = ([0, 1, 2], [1, 2, 3], [1, 1, 1])
        angles = ([0, 1], [1, 2], [2, 3], [2, 2])
        propers = ([0], [1], [2], [3], [3])
    return {
        "bond_atomi": _i32(bonds[0]),
        "bond_atomj": _i32(bonds[1]),
        "bond_type": _i32(bonds[2]),
        "angle_atomi": _i32(angles[0]),
        "angle_atomj": _i32(angles[1]),
        "angle_atomk": _i32(angles[2]),
        "angle_type": _i32(angles[3]),
        "proper_atomi": _i32(propers[0]),
        "proper_atomj": _i32(propers[1]),
        "proper_atomk": _i32(propers[2]),
        "proper_atoml": _i32(propers[3]),
        "proper_type": _i32(propers[4]),
    }


def _unit(key: str, task: str, n_atoms: int, n_frames: int, offset: int) -> tuple:
    """One molecule's frames — the fixture's ingestion unit, ``(frames, key, task)``."""
    atomic_numbers = [8, 1, 1] if n_atoms == 3 else [6, 6, 8, 1]
    valence = _valence(n_atoms)
    frames = []
    for k in range(n_frames):
        base = offset + k * n_atoms * 3
        frames.append(
            frame(
                Z=atomic_numbers,
                pos=_halves(n_atoms * 3, base).reshape(n_atoms, 3).numpy(),
                forces=(-_halves(n_atoms * 3, base + 7)).reshape(n_atoms, 3).numpy(),
                energy=float(_halves(1, offset + 3 + k)[0]),
                bonds=(valence["bond_atomi"].numpy(), valence["bond_atomj"].numpy()),
                bond_types=valence["bond_type"].numpy(),
                terms={
                    "angles": {
                        "atomi": valence["angle_atomi"].numpy(),
                        "atomj": valence["angle_atomj"].numpy(),
                        "atomk": valence["angle_atomk"].numpy(),
                        "type_id": valence["angle_type"].numpy(),
                    },
                    "propers": {
                        "atomi": valence["proper_atomi"].numpy(),
                        "atomj": valence["proper_atomj"].numpy(),
                        "atomk": valence["proper_atomk"].numpy(),
                        "atoml": valence["proper_atoml"].numpy(),
                        "type_id": valence["proper_type"].numpy(),
                    },
                },
            )
        )
    return frames, key, task


def _bare_unit(key: str, task: str, n_atoms: int, n_frames: int, offset: int) -> tuple:
    """A unit carrying only geometry — a different column set."""
    frames = [
        frame(
            Z=[8] + [1] * (n_atoms - 1),
            pos=_halves(n_atoms * 3, offset + k * n_atoms * 3).reshape(n_atoms, 3).numpy(),
        )
        for k in range(n_frames)
    ]
    return frames, key, task


def _internal_unit(key: str, task: str, n_atoms: int, n_frames: int, offset: int) -> tuple:
    """A unit declaring an ``internals`` family the downstream view does not route."""
    frames = [
        frame(
            Z=[8] + [1] * (n_atoms - 1),
            pos=_halves(n_atoms * 3, offset + k * n_atoms * 3).reshape(n_atoms, 3).numpy(),
            terms={
                "internals": {
                    "atomi": [0],
                    "atomj": [1],
                    "atomk": [2],
                    "atoml": [0],
                    "type_id": [1],
                }
            },
        )
        for k in range(n_frames)
    ]
    return frames, key, task


def _write_block(
    root: Path,
    units: Sequence[tuple],
    *,
    split: str = "train",
    seal: bool = True,
) -> Path:
    """Write *units* into *root* one unit at a time; seal unless told otherwise."""
    writer = FrameStoreWriter(root, split=split, provenance=_PROVENANCE)
    for frames, key, task in units:
        writer.append(frames, key=key, task=task)
    if seal:
        writer.seal()
    return root


def _write_fixture_block(tmp_path: Path, name: str) -> Path:
    """Write one sealed block of the A/B/C fixture."""
    return _write_block(
        tmp_path / f"block-{name}",
        [_unit(*spec) for spec in _BLOCK_UNITS[name]],
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


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


def _assert_payload_equal(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    path: str = "",
) -> None:
    """Leaf-for-leaf equality of two packed-cache-v1 payload mappings."""
    assert set(actual) == set(expected), f"payload key sets differ at {path or '/'}"
    for key in sorted(actual):
        left, right = actual[key], expected[key]
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            _assert_payload_equal(left, right, f"{path}/{key}")
            continue
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            assert left.dtype == right.dtype, f"dtype differs at {path}/{key}"
            assert torch.equal(left, right), f"values differ at {path}/{key}"
            continue
        assert left == right, f"values differ at {path}/{key}"


def _names_number(message: str, roots: Sequence[Path], value: int) -> bool:
    """True when *message* names *value* outside of any root path it quotes.

    The block paths are stripped first, so a digit inside ``tmp_path`` cannot
    masquerade as the block row (or the offending code) the error must name.
    """
    stripped = message
    for root in roots:
        stripped = stripped.replace(str(root), " ")
    return re.search(rf"(?<!\d){value}(?!\d)", stripped) is not None


def _sampler(view: FrameStoreDataset, *, seed: int = _SAMPLER_SEED) -> TaskMixtureBatchSampler:
    """Block-native code tables, atoms-only budget."""
    return TaskMixtureBatchSampler(
        view,
        indices=list(range(len(view))),
        task_labels=view.task_codes,
        group_ids=view.group_codes,
        task_names=view.task_names,
        mixture=_MIXTURE,
        max_atoms=_SAMPLER_MAX_ATOMS,
        max_edges=None,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots(tmp_path: Path) -> list[Path]:
    """The three sealed ``split="train"`` blocks A, B, C, in view order."""
    return [_write_fixture_block(tmp_path, name) for name in ("A", "B", "C")]


@pytest.fixture
def view(roots: list[Path]) -> FrameStoreDataset:
    return FrameStoreDataset(roots)


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    """A ``prepare``-shaped directory holding the three sealed roots A, B, C.

    The blocks are written **C, B, A** on purpose: creation order is the reverse
    of the sorted order, so a discovery that handed back raw directory-listing
    order (which on ext4/xfs commonly *is* creation order) fails the block-id pin
    instead of passing by accident.
    """
    directory = tmp_path / "prepared"
    directory.mkdir()
    for name in ("C", "B", "A"):
        _write_fixture_block(directory, name)
    return directory


# ---------------------------------------------------------------------------
# FrameStoreDataset
# ---------------------------------------------------------------------------


class TestFrameStoreDataset:
    # -- global index map -----------------------------------------

    def test_len_is_the_total_frame_count_across_blocks(self, view: FrameStoreDataset):
        assert len(view) == _LEN

    def test_locate_returns_block_ordinals_and_local_rows(self, view: FrameStoreDataset):
        blocks, locals_ = view.locate(list(_LOCATE_FRAMES))
        assert tuple(blocks.shape) == (len(_LOCATE_FRAMES),)
        assert tuple(locals_.shape) == (len(_LOCATE_FRAMES),)
        assert blocks.dtype is torch.long
        assert locals_.dtype is torch.long
        assert blocks.tolist() == list(_LOCATE_BLOCKS)
        assert locals_.tolist() == list(_LOCATE_LOCALS)

    def test_locate_accepts_a_tensor_of_frames(self, view: FrameStoreDataset):
        blocks, locals_ = view.locate(torch.tensor(_LOCATE_FRAMES, dtype=torch.long))
        assert blocks.tolist() == list(_LOCATE_BLOCKS)
        assert locals_.tolist() == list(_LOCATE_LOCALS)

    def test_locate_of_one_frame_is_the_scalar_lookup(self, view: FrameStoreDataset):
        blocks, locals_ = view.locate([12])
        assert tuple(blocks.shape) == (1,)
        assert tuple(locals_.shape) == (1,)
        assert (int(blocks[0]), int(locals_[0])) == (2, 2)

    def test_locate_rejects_a_frame_beyond_the_view(self, view: FrameStoreDataset):
        with pytest.raises((IndexError, ValueError), match="15"):
            view.locate([_LEN])

    def test_locate_rejects_a_negative_frame(self, view: FrameStoreDataset):
        with pytest.raises((IndexError, ValueError), match="-1"):
            view.locate([-1])

    # -- surface -----------------------------------------------------------

    def test_blocks_are_the_given_roots_in_order(self, view: FrameStoreDataset, roots: list[Path]):
        assert list(view.blocks) == [Path(root) for root in roots]

    def test_block_ids_are_the_directory_basenames(self, view: FrameStoreDataset):
        assert view.block_ids == ("block-A", "block-B", "block-C")

    def test_max_open_defaults_to_four_and_is_read_only(self, view: FrameStoreDataset):
        assert view.max_open == 4
        with pytest.raises(AttributeError):
            view.max_open = 2  # type: ignore[misc]

    def test_split_is_the_single_split_of_every_block(self, view: FrameStoreDataset):
        assert view.split == "train"

    def test_max_atoms_is_the_largest_unit_atom_count(self, view: FrameStoreDataset):
        assert view.max_atoms == 4

    def test_typed_guards_are_value_error_subclasses(self):
        # 03's scheduler branches on the type; every existing `except ValueError`
        # handler must keep working.
        assert issubclass(FrameStoreNotReadyError, ValueError)
        assert issubclass(FrameStoreSplitMismatchError, ValueError)
        assert issubclass(FrameStoreSchemaMismatchError, ValueError)

    # -- concatenated sampler tables ----------------------

    def test_atom_counts_equal_the_hand_computed_vector(self, view: FrameStoreDataset):
        counts = view.atom_counts
        assert counts.dtype is torch.long
        assert tuple(counts.shape) == (_LEN,)
        assert torch.equal(counts, torch.tensor(_ATOM_COUNTS, dtype=torch.long))

    def test_group_codes_equal_the_hand_computed_vector(self, view: FrameStoreDataset):
        codes = view.group_codes
        assert codes.dtype is torch.long
        assert tuple(codes.shape) == (_LEN,)
        assert torch.equal(codes, torch.tensor(_GROUP_CODES, dtype=torch.long))

    def test_group_codes_partition_the_units_of_the_view(self, view: FrameStoreDataset):
        # Injective across blocks: block b's codes are its unit rows [0, M_b),
        # offset by unit_ptr[b], so the images partition [0, total_units).
        assert set(view.group_codes.tolist()) == set(range(_TOTAL_UNITS))

    def test_task_codes_equal_the_hand_computed_vector(self, view: FrameStoreDataset):
        codes = view.task_codes
        assert codes.dtype is torch.long
        assert tuple(codes.shape) == (_LEN,)
        assert torch.equal(codes, torch.tensor(_TASK_CODES, dtype=torch.long))

    def test_task_names_merge_in_first_appearance_order(self, view: FrameStoreDataset):
        assert view.task_names == _TASK_NAMES

    def test_block_c_local_task_codes_are_remapped_to_global(
        self, view: FrameStoreDataset, roots: list[Path]
    ):
        # A non-identity merge: an identity bug cannot pass this.
        assert FrameStoreReader(roots[2]).task_codes.tolist() == [0, 1, 1, 1, 1]
        assert view.task_codes[10:].tolist() == [1, 2, 2, 2, 2]

    # -- readiness guard, I8 --------------------------------------

    def test_missing_manifest_stalls_with_block_not_ready(self, tmp_path: Path, roots: list[Path]):
        gap = tmp_path / "block-gap"
        gap.mkdir()
        planned = [roots[0], gap, roots[1]]
        with pytest.raises(FrameStoreNotReadyError) as excinfo:
            FrameStoreDataset(planned)
        message = str(excinfo.value)
        assert str(gap) in message
        assert "manifest" in message
        assert _names_number(message, planned, 1), message

    def test_unsealed_block_stalls_with_block_not_ready(self, tmp_path: Path, roots: list[Path]):
        unsealed = _write_block(
            tmp_path / "block-unsealed",
            [_unit("mol-u", "relax", 3, 2, 400)],
            seal=False,
        )
        planned = [roots[0], roots[1], unsealed]
        with pytest.raises(FrameStoreNotReadyError) as excinfo:
            FrameStoreDataset(planned)
        message = str(excinfo.value)
        assert str(unsealed) in message
        assert "unsealed" in message.lower()
        assert _names_number(message, planned, 2), message

    def test_cold_block_deleted_after_construction_stalls(self, roots: list[Path]):
        # POSIX mmap survives unlink, so the guard is pinned on a block that is
        # *not* resident: with max_open=1 block 0 has been evicted, and reopening
        # it must stall instead of silently shortening the epoch.
        cold = FrameStoreDataset(roots, max_open=1)
        assert len(cold) == _LEN
        assert 0 not in cold.open_blocks, "block 0 must be evicted for this to be a cold read"
        shutil.rmtree(roots[0])
        with pytest.raises(FrameStoreNotReadyError) as excinfo:
            cold._columns([0])
        assert str(roots[0]) in str(excinfo.value)

    # -- split purity, I7 -----------------------------------------

    def test_mixed_splits_raise_split_mismatch_naming_both_roots(
        self, tmp_path: Path, roots: list[Path]
    ):
        valid = _write_block(
            tmp_path / "block-valid",
            [_unit("mol-v", "relax", 3, 2, 500)],
            split="valid",
        )
        with pytest.raises(FrameStoreSplitMismatchError) as excinfo:
            FrameStoreDataset([roots[0], valid])
        message = str(excinfo.value)
        assert str(roots[0]) in message
        assert str(valid) in message
        assert "train" in message
        assert "valid" in message

    def test_per_split_views_construct_independently(self, tmp_path: Path, roots: list[Path]):
        valid = _write_block(
            tmp_path / "block-valid",
            [_unit("mol-v", "relax", 3, 2, 500)],
            split="valid",
        )
        assert FrameStoreDataset(roots).split == "train"
        assert FrameStoreDataset([valid]).split == "valid"

    # -- concatenability guards ---------------------------

    def test_differing_routed_column_sets_raise_schema_mismatch(
        self, tmp_path: Path, roots: list[Path]
    ):
        bare = _write_block(
            tmp_path / "block-bare",
            [_bare_unit("mol-w", "relax", 3, 2, 600)],
        )
        with pytest.raises(FrameStoreSchemaMismatchError) as excinfo:
            FrameStoreDataset([roots[0], bare])
        message = str(excinfo.value)
        assert str(bare) in message
        # the symmetric difference, named
        assert "energy" in message
        assert "forces" in message

    def test_task_code_beyond_its_dictionary_raises_schema_mismatch(self, roots: list[Path]):
        # Truncating task_names leaves every column sha256 intact — no payload
        # byte moves — so the view's own shape check is what must catch the code
        # table and the name table disagreeing. (Since the 2026-08-14 digest-core
        # revision block_digest also moves, but only a consumer holding an
        # externally recorded digest can see that; a view opening the block on
        # its own cannot.)
        manifest_path = roots[0] / "manifest.json"
        document = json.loads(manifest_path.read_text())
        document["task_names"] = ["equilibrium"]
        manifest_path.write_text(json.dumps(document))
        with pytest.raises(FrameStoreSchemaMismatchError) as excinfo:
            FrameStoreDataset([roots[0]])
        message = str(excinfo.value)
        assert str(roots[0]) in message
        assert _names_number(message, [roots[0]], 1), message

    def test_internal_valence_family_block_is_refused(self, tmp_path: Path):
        # Single-root view, so the column-set clause cannot fire: the refusal is
        # the family's missing downstream route, not a cross-block difference.
        internal = _write_block(
            tmp_path / "block-internal",
            [_internal_unit("mol-i", "relax", 3, 2, 700)],
        )
        with pytest.raises(FrameStoreSchemaMismatchError, match="internal"):
            FrameStoreDataset([internal])

    def test_empty_roots_raise_value_error(self):
        with pytest.raises(ValueError):
            FrameStoreDataset([])

    def test_max_open_below_one_raises_value_error(self, roots: list[Path]):
        with pytest.raises(ValueError):
            FrameStoreDataset(roots, max_open=0)

    # -- edge counts are refused, never fabricated ----------------

    def test_edge_counts_refuses_and_names_the_downstream_builder(self, view: FrameStoreDataset):
        with pytest.raises(ValueError, match="NeighborList"):
            _ = view.edge_counts

    def test_token_budget_sampler_propagates_the_edge_counts_refusal(self, view: FrameStoreDataset):
        # The block store's ValueError must reach the caller verbatim, not be
        # reworded into the AttributeError-derived "use FrameStoreDataset" message.
        with pytest.raises(ValueError, match="NeighborList"):
            TokenBudgetBatchSampler(view, max_edges=32)

    def test_token_budget_sampler_with_an_atom_budget_only_succeeds(self, view: FrameStoreDataset):
        sampler = TokenBudgetBatchSampler(view, max_atoms=_SAMPLER_MAX_ATOMS, seed=_SAMPLER_SEED)
        plan = [list(batch) for batch in sampler]
        assert sorted(index for batch in plan for index in batch) == list(range(_LEN))

    # -- cross-block payload --------------------------------------

    def test_cross_block_payload_atom_ptr_is_the_caller_order_cumsum(self, view: FrameStoreDataset):
        payload = view._columns(list(_CROSS_FRAMES))
        assert payload["n_samples"] == len(_CROSS_FRAMES)
        assert payload["atom_ptr"].tolist() == list(_CROSS_ATOM_PTR)

    def test_cross_block_batch_sizes_are_hand_computed(self, view: FrameStoreDataset):
        # frames (12, 1, 7) -> units scan-e / mol-a / scan-c: atoms 4+3+4,
        # angles 2+1+2, propers 1+0+1; blocks carry no edges by design.
        batch = view.batch(list(_CROSS_FRAMES))
        assert tuple(batch.batch_size) == ()
        assert tuple(batch["atoms"].batch_size) == (11,)
        assert tuple(batch["graphs"].batch_size) == (3,)
        assert tuple(batch["edges"].batch_size) == (0,)
        assert tuple(batch["angles"].batch_size) == (5,)
        assert tuple(batch["propers"].batch_size) == (2,)

    def test_reversed_frame_list_yields_the_reversed_batch(self, view: FrameStoreDataset):
        frames = list(_CROSS_FRAMES)
        forward = view.batch(frames)
        backward = view.batch(frames[::-1])
        # equal atom counts would hide a lost order; the per-frame energies cannot
        assert torch.equal(
            backward["graphs"]["energy"], torch.flip(forward["graphs"]["energy"], (0,))
        )

    def test_single_block_request_returns_the_owning_reader_payload(
        self, view: FrameStoreDataset, roots: list[Path]
    ):
        _assert_payload_equal(
            view._columns([5, 6, 7]),
            FrameStoreReader(roots[1])._columns([0, 1, 2]),
        )

    def test_frame_payload_rejects_an_empty_frame_list(self, view: FrameStoreDataset):
        with pytest.raises(ValueError):
            view._columns([])

    def test_frame_payload_rejects_an_out_of_range_frame(self, view: FrameStoreDataset):
        with pytest.raises(ValueError, match="15"):
            view._columns([0, _LEN])

    # -- __getitem__ is the one-frame payload unpacked ------------

    def test_getitem_carries_the_targets_bonds_and_valence_namespaces(
        self, view: FrameStoreDataset
    ):
        """One frame comes back as a one-frame batch with every namespace it owns."""
        one = view[12]

        assert set(one.keys()) == {"atoms", "edges", "graphs", "bonds", "angles", "propers"}
        assert set(one["atoms"].keys()) == {"Z", "pos", "batch", "forces"}
        assert "energy" in one["graphs"].keys()
        assert tuple(one["bonds", "bond_index"].shape) == (2, 3)
        assert int(one["graphs", "num_atoms"].item()) == int(one["atoms"].batch_size[0])

    def test_getitem_rejects_an_index_outside_the_view(self, view: FrameStoreDataset):
        with pytest.raises((IndexError, ValueError)):
            view[_LEN]
        with pytest.raises((IndexError, ValueError)):
            view[-_LEN - 1]

    # -- bounded-K reader pool ------------------------------------

    def test_open_blocks_never_exceed_max_open_during_a_full_sweep(self, roots: list[Path]):
        bounded = FrameStoreDataset(roots, max_open=1)
        for row in range(_LEN):
            bounded._columns([row])
            assert len(bounded.open_blocks) <= 1

    def test_payload_is_identical_for_max_open_one_and_eight(self, roots: list[Path]):
        frames = list(_CROSS_FRAMES)
        _assert_payload_equal(
            FrameStoreDataset(roots, max_open=1)._columns(frames),
            FrameStoreDataset(roots, max_open=8)._columns(frames),
        )

    def test_reader_returns_the_block_reader_of_a_block_row(self, view: FrameStoreDataset):
        reader = view.reader(1)
        assert isinstance(reader, FrameStoreReader)
        assert reader.frame_count == 5

    # -- sampler integration, unit level --------------------------

    def test_sampler_epoch_batches_are_single_task(self, view: FrameStoreDataset):
        codes = view.task_codes
        for batch in _sampler(view):
            assert len(set(codes[torch.tensor(batch, dtype=torch.long)].tolist())) == 1

    def test_sampler_epoch_keeps_every_group_whole(self, view: FrameStoreDataset):
        groups = view.group_codes.tolist()
        frames_of_group: dict[int, set[int]] = {}
        for row, group in enumerate(groups):
            frames_of_group.setdefault(group, set()).add(row)
        planned: list[int] = []
        for batch in _sampler(view):
            planned.extend(batch)
            for group in {groups[row] for row in batch}:
                assert frames_of_group[group] <= set(batch)
        assert len(planned) == len(set(planned))

    def test_sampler_epoch_is_deterministic_across_constructions(self, view: FrameStoreDataset):
        first = [list(batch) for batch in _sampler(view)]
        second = [list(batch) for batch in _sampler(view)]
        assert first
        assert first == second

    def test_sampler_epoch_frame_totals_match_the_hand_computed_shares(
        self, view: FrameStoreDataset
    ):
        # relax = mol-b (3 frames) + mol-d (1) = 4; torsion = scan-c (5) +
        # scan-e (4) = 9; equilibrium is excluded by its 0.0 probability.
        assert _sampler(view).frames_per_task() == _SAMPLER_FRAMES_PER_TASK

    # -- directory discovery: load_from_disk -------------------------------
    #
    # The consuming half of molhub's `prepare` seam: `prepare` writes a directory
    # of sealed `molix.block/1` roots, this alternate constructor opens it. Its
    # only added domain content is *which* roots and *in what order* — every
    # refusal below block construction stays `__init__`'s.

    def test_load_from_disk_discovers_sealed_roots_in_sorted_child_order(self, prepared: Path):
        # Discovery order is sorted by child name, not by directory-listing order
        # (the fixture writes C, B, A): one prepared directory has exactly one
        # frame index, whatever the filesystem chose to report.
        discovered = FrameStoreDataset.load_from_disk(prepared)
        assert discovered.block_ids == ("block-A", "block-B", "block-C")
        assert len(discovered) == _LEN
        assert discovered.split == "train"

    def test_load_from_disk_equals_explicit_construction_over_the_same_roots(self, prepared: Path):
        # Same length, same tables, same first-frame bytes as the explicit
        # constructor over the same roots in sorted order.
        explicit = FrameStoreDataset(sorted(prepared.iterdir()))
        discovered = FrameStoreDataset.load_from_disk(prepared)
        assert len(discovered) == len(explicit) == _LEN
        assert torch.equal(discovered.atom_counts, torch.tensor(_ATOM_COUNTS, dtype=torch.long))
        assert torch.equal(discovered.task_codes, torch.tensor(_TASK_CODES, dtype=torch.long))
        assert discovered.task_names == _TASK_NAMES
        _assert_payload_equal(discovered._columns([0]), explicit._columns([0]))

    def test_load_from_disk_of_a_single_sealed_root_is_a_one_block_view(self, roots: list[Path]):
        # The directory *is* the root (it carries manifest.json itself), so the
        # view holds block A alone: 5 frames, atom counts (3, 3, 4, 4, 4).
        single = FrameStoreDataset.load_from_disk(roots[0])
        assert single.blocks == (roots[0],)
        assert single.block_ids == ("block-A",)
        assert len(single) == 5
        assert torch.equal(single.atom_counts, torch.tensor(_ATOM_COUNTS[:5], dtype=torch.long))

    def test_load_from_disk_ignores_children_that_are_not_sealed_roots(self, prepared: Path):
        # Only manifest.json-bearing children count: a stray file and a stray
        # directory without a manifest are skipped, not refused and not indexed.
        (prepared / "README.md").write_text("written by prepare, not a block\n")
        (prepared / "scratch").mkdir()
        discovered = FrameStoreDataset.load_from_disk(prepared)
        assert discovered.block_ids == ("block-A", "block-B", "block-C")
        assert len(discovered) == _LEN

    def test_load_from_disk_without_a_discoverable_root_raises_value_error(self, tmp_path: Path):
        # Loud, not an empty view: an empty index would hide a mis-typed path or a
        # `prepare` run that never wrote anything.
        barren = tmp_path / "barren"
        barren.mkdir()
        (barren / "notes.txt").write_text("no blocks here\n")
        with pytest.raises(ValueError) as excinfo:
            FrameStoreDataset.load_from_disk(barren)
        message = str(excinfo.value)
        assert str(barren) in message
        assert "manifest.json" in message

    def test_load_from_disk_passes_max_open_through_to_the_reader_pool(self, prepared: Path):
        bounded = FrameStoreDataset.load_from_disk(prepared, max_open=1)
        assert bounded.max_open == 1
        for row in range(_LEN):
            bounded._columns([row])
            assert len(bounded.open_blocks) <= 1

    def test_load_from_disk_still_refuses_discovered_roots_of_mixed_splits(self, tmp_path: Path):
        # I7 is a construction refusal and discovery does not soften it.
        mixed = tmp_path / "mixed"
        mixed.mkdir()
        _write_fixture_block(mixed, "A")
        _write_block(
            mixed / "block-valid",
            [_unit("mol-v", "relax", 3, 2, 500)],
            split="valid",
        )
        with pytest.raises(FrameStoreSplitMismatchError) as excinfo:
            FrameStoreDataset.load_from_disk(mixed)
        message = str(excinfo.value)
        assert "train" in message
        assert "valid" in message

    def test_load_from_disk_stalls_on_a_discovered_but_unsealed_root(self, tmp_path: Path):
        # An unsealed block *has* a manifest.json, so it is discovered and then
        # refused by I8 — it is never quietly skipped, which would silently
        # shorten the epoch by exactly the frames still being written.
        partial = tmp_path / "partial"
        partial.mkdir()
        _write_fixture_block(partial, "A")
        unsealed = _write_block(
            partial / "block-unsealed",
            [_unit("mol-u", "relax", 3, 2, 400)],
            seal=False,
        )
        with pytest.raises(FrameStoreNotReadyError) as excinfo:
            FrameStoreDataset.load_from_disk(partial)
        message = str(excinfo.value)
        assert str(unsealed) in message
        assert "unsealed" in message.lower()


# ---------------------------------------------------------------------------
# Module-level guards (they target the module, not a type)
# ---------------------------------------------------------------------------


def _imported_modules(path: Path) -> list[str]:
    """Every module named by an import statement of *path*, at any depth."""
    imported: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                imported.extend(f".{alias.name}" for alias in node.names)
            else:
                imported.append(node.module)
    return imported


def test_frame_store_view_symbols_are_exported_from_molix_store():
    """The four public symbols are importable, listed and documented."""
    import molix.store as store

    assert store.FrameStoreDataset is FrameStoreDataset
    assert store.FrameStoreNotReadyError is FrameStoreNotReadyError
    assert store.FrameStoreSchemaMismatchError is FrameStoreSchemaMismatchError
    assert store.FrameStoreSplitMismatchError is FrameStoreSplitMismatchError
    exported = set(store.__all__)
    assert {
        "FrameStoreDataset",
        "FrameStoreNotReadyError",
        "FrameStoreSchemaMismatchError",
        "FrameStoreSplitMismatchError",
    } <= exported
