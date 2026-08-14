"""Public-API regression for the multi-block dataset view over sealed blocks.

Spec: ``streaming-data-tiers-04-scheduler-integration-02-view`` (acceptance ac-012).

One minimal scenario, public API only: three sealed ``split="train"`` blocks are
written unit-by-unit with :class:`~molix.store.BlockWriter` into a
:class:`tempfile.TemporaryDirectory` and opened as **one**
:class:`~molix.store.BlockDataset`, whose global frame index is
``(block, local frame)`` in the given block order. The script drives the surface a
scheduler consumes — the concatenated sampler tables, the vectorized ``locate``,
the order-preserving cross-block ``frame_payload``, the bounded reader pool and
the three typed refusals — so it fails the moment the public API, the index
algebra, or the caller composition
``collate_packed(PackedView(view.frame_payload(frames)), ...)`` drifts.

Fixture (deterministic — no RNG anywhere, and every stored float is a multiple of
``0.5``, hence exact in ``float32``, so **no FP tolerance applies anywhere in this
script**; ``pos`` Å, ``energy`` kcal mol⁻¹, ``forces`` kcal mol⁻¹ Å⁻¹)::

    block  unit    task         N  n_frames  halves offset  global frames
    ---------------------------------------------------------------------
    A      mol-a   equilibrium  3     2            0        0, 1
           mol-b   relax        4     3           40        2, 3, 4
    B      scan-c  torsion      4     5          100        5, 6, 7, 8, 9
    C      mol-d   relax        3     1          200        10
           scan-e  torsion      4     4          300        11, 12, 13, 14

One unit is one group (the block layout *is* the grouping), so a group code is
the owning unit row — offset by the block's ``unit_ptr`` to stay globally unique.

Hand-derived from that table alone — every golden below is closed-form integer or
halves arithmetic, re-derived in :func:`_check_fixture_conventions` so a typo in a
literal cannot redefine the golden it is compared against:

1. **Index algebra.** ``F = (5, 5, 5)`` and ``M = (2, 1, 2)`` give the two
   exclusive cumsums ``frame_ptr = (0, 5, 10, 15)`` and ``unit_ptr = (0, 2, 3, 5)``,
   so ``len(view) = 15`` and ``locate([0,4,5,9,10,14])`` is blocks ``(0,0,1,1,2,2)``
   with local rows ``(0,4,0,4,0,4)``. Neither pointer is a public attribute — both
   are re-derived here from ``locate`` + ``group_codes`` (and cross-checked against
   the per-block :class:`~molix.store.BlockReader` counts), which is exactly the
   property they encode.
2. **Concatenated tables.** ``atom_counts`` repeats each unit's ``N`` over its
   frames; ``group_codes`` is the owning unit row offset by ``unit_ptr[b]``, hence
   injective over ``range(5)``; ``task_names`` is the **first-appearance merge**
   ``("equilibrium", "relax", "torsion")`` of the per-block dictionaries, under
   which block C's local codes ``(0, 1)`` become global ``(1, 2)`` — a
   non-identity remap, so an identity bug cannot pass.
3. **Cross-block payload.** For a frame list, ``atom_ptr`` is the exclusive cumsum
   of the selected ``atom_counts`` **in caller order**, the graph-level
   ``energy`` leaf is ``(k + offset + 3) * 0.5`` for the ``k``-th frame of its
   unit, ``Z`` is ``(8,1,1)`` for a 3-atom unit and ``(6,6,8,1)`` for a 4-atom
   one, the membership vector repeats the caller slot ``atom_counts`` times, and
   the first ``pos`` row of a sample is ``(3*(k*N) + offset + j) * 0.5``. Three
   requests are pinned: the cross-block ``[12, 1, 7]`` of the spec, its reverse
   (equal atom counts, so only the per-frame ``energy`` can witness a lost order),
   and the **interleaved** ``[12, 1, 13, 7, 0]`` — blocks ``(2, 0, 2, 1, 0)``, a
   frame list whose restore permutation is non-identity in both directions, which
   the two contiguous requests cannot exercise.
4. **Sampler epoch.** ``max_atoms=8`` with unit costs ``n_frames*N`` = mol-a 6,
   mol-b 12, scan-c 20, mol-d 3, scan-e 16: mixture ``{equilibrium: 0.0,
   relax: 0.5, torsion: 0.5}`` drops equilibrium; relax packs mol-b (oversize on
   its own, kept whole) and mol-d into **2** batches; torsion's two units are both
   oversize, so **2** batches. Natural epoch ``min(floor(2/0.5), floor(2/0.5)) =
   4`` slots, apportioned ``{relax: 2, torsion: 2}``, which consumes the whole
   supply — so the batch *contents* are permutation-independent, ``{2,3,4}``,
   ``{10}``, ``{5..9}``, ``{11..14}``, and the per-task frame totals are
   ``relax = 3 + 1 = 4`` and ``torsion = 5 + 4 = 9``. Only the order of the four
   batches is the seeded permutation, so it is asserted *stable*, not pinned.
5. **Refusals and bounded K.** A ``valid``-split block mixed into the plan raises
   :class:`~molix.store.BlockSplitMismatchError`, an unsealed block raises
   :class:`~molix.store.BlockNotReadyError` (never skipped, never substituted),
   and ``view.edge_counts`` raises the block store's own ``ValueError`` naming
   ``NeighborList`` — blocks carry no neighbour graph, so a fabricated zero would
   be a false budget. Results are independent of the reader-pool bound ``K``:
   ``max_open=1`` and ``max_open=8`` produce leaf-identical payloads while
   ``open_blocks`` never exceeds the cap.

    capture: derived by hand from the fixture table above — every golden is
             closed-form integer / halves arithmetic over a deterministic
             fixture, re-derived in ``_check_fixture_conventions``. No tool was
             run to produce a number; nothing was captured from an external
             implementation.
    verify:  PYTHONPATH=src python \\
             regressions/streaming-data-tiers-04-scheduler-integration-02-view.py
    commit:  57f2adc (+ working tree implementing
             streaming-data-tiers-04-scheduler-integration-02-view)
    torch:   2.12.1+cpu
    device:  cpu (the index algebra is integer-only; the float32 payload is exact
             halves)
    date:    2026-08-14
    oracle:  none — no third-party package is imported, subprocessed or consulted,
             at capture time or at run time. ``torch`` / ``tensordict`` are the
             tensor and batch containers the feature is used *with*, and
             ``collate_molecules`` over ``view[i]`` is this repo's own
             single-data-route equivalence check, not an external reference.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import torch
from tensordict import TensorDict

from molix.batch.collate import collate_molecules, collate_packed
from molix.batch.sampler import TaskMixtureBatchSampler
from molix.store import (
    BlockDataset,
    BlockNotReadyError,
    BlockReader,
    BlockSplitMismatchError,
    BlockWriter,
    PackedView,
    SamplingUnit,
)

# ---------------------------------------------------------------------------
# Fixture tables (see module docstring)
# ---------------------------------------------------------------------------

#: ``(unit key, task, n_atoms, n_frames, halves offset)`` per block, in append
#: order; the mapping order is the **view order** of the blocks.
_BLOCK_UNITS: Mapping[str, tuple[tuple[str, str, int, int, int], ...]] = {
    "A": (("mol-a", "equilibrium", 3, 2, 0), ("mol-b", "relax", 4, 3, 40)),
    "B": (("scan-c", "torsion", 4, 5, 100),),
    "C": (("mol-d", "relax", 3, 1, 200), ("scan-e", "torsion", 4, 4, 300)),
}
#: Opaque, caller-supplied provenance — fixed literals, presence-validated only.
_PROVENANCE: Mapping[str, object] = {
    "source": {"name": "molnex-regression", "coordinate": "n/a"},
    "generator": {
        "tool": "regressions/streaming-data-tiers-04-scheduler-integration-02-view.py",
        "version": "1",
        "commit": "57f2adc",
    },
    "created_at": "2026-08-14T00:00:00+00:00",
    "units": {"pos": "angstrom", "energy": "kcal/mol", "forces": "kcal/mol/angstrom"},
}
#: Atomic numbers of a unit, keyed by its atom count — one heavy-atom motif each.
_ATOMIC_NUMBERS: Mapping[int, tuple[int, ...]] = {3: (8, 1, 1), 4: (6, 6, 8, 1)}

# ---------------------------------------------------------------------------
# Hard-coded goldens — leg 1, the global index map
# ---------------------------------------------------------------------------

#: ``len(view)`` = ``frame_ptr[B]`` (hard-coded golden).
_LEN = 15
#: Exclusive cumsum of the per-block frame counts ``F = (5, 5, 5)``.
_FRAME_PTR: tuple[int, ...] = (0, 5, 10, 15)
#: Exclusive cumsum of the per-block unit counts ``M = (2, 1, 2)``.
_UNIT_PTR: tuple[int, ...] = (0, 2, 3, 5)
#: Block identity strings, in view order — the producer's directory basenames.
_BLOCK_IDS: tuple[str, ...] = ("block-A", "block-B", "block-C")
#: The single manifest split every planned block carries (invariant I7).
_SPLIT = "train"
#: ``max_b max(N_m)`` over the planned blocks.
_VIEW_MAX_ATOMS = 4
#: The six ``locate`` probes of the spec, and their hand-computed answers.
_LOCATE_FRAMES: tuple[int, ...] = (0, 4, 5, 9, 10, 14)
_LOCATE_BLOCKS: tuple[int, ...] = (0, 0, 1, 1, 2, 2)
_LOCATE_LOCALS: tuple[int, ...] = (0, 4, 0, 4, 0, 4)

# ---------------------------------------------------------------------------
# Hard-coded goldens — leg 1, the concatenated sampler tables
# ---------------------------------------------------------------------------

#: Atoms per frame, ``(15,) torch.long``.
_ATOM_COUNTS: tuple[int, ...] = (3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4)
#: Globally unique group id per frame = owning unit row + ``unit_ptr[b]``.
_GROUP_CODES: tuple[int, ...] = (0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4)
#: Task code per frame in the merged dictionary; block C remaps (0,1) -> (1,2).
_TASK_CODES: tuple[int, ...] = (0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2)
#: The merged task dictionary, in first-appearance order across blocks.
_TASK_NAMES: tuple[str, ...] = ("equilibrium", "relax", "torsion")
#: Units of the whole view = ``unit_ptr[B]``; the group codes partition it.
_TOTAL_UNITS = 5

# ---------------------------------------------------------------------------
# Hard-coded goldens — leg 2, the cross-block payload requests
# ---------------------------------------------------------------------------


class _Request(NamedTuple):
    """One ``frame_payload`` request with its hand-derived batch goldens.

    Attributes:
        label: Human name of the request, for the failure messages.
        frames: Global frame rows, in the order they must appear in the batch.
        atom_ptr: Exclusive cumsum ``(B+1,)`` of the selected atom counts.
        batch_sizes: Per-namespace ``batch_size`` of the collated batch.
        energy: ``graphs.energy`` ``(B,)`` in kcal mol⁻¹, caller order.
        atomic_numbers: ``atoms.Z`` ``(N,)``, caller order.
        membership: ``atoms.batch`` ``(N,)`` — the caller slot of every atom row.
        head_pos: First ``pos`` row ``(3,)`` in Å of each sample, caller order.
    """

    label: str
    frames: tuple[int, ...]
    atom_ptr: tuple[int, ...]
    batch_sizes: Mapping[str, tuple[int, ...]]
    energy: tuple[float, ...]
    atomic_numbers: tuple[int, ...]
    membership: tuple[int, ...]
    head_pos: tuple[tuple[float, float, float], ...]


#: The spec's cross-block request: frames (12, 1, 7) -> blocks (2, 0, 1),
#: units scan-e / mol-a / scan-c, atom counts 4, 3, 4.
_CROSS = _Request(
    label="cross-block [12, 1, 7]",
    frames=(12, 1, 7),
    atom_ptr=(0, 4, 7, 11),
    batch_sizes={
        "atoms": (11,),
        "edges": (0,),
        "graphs": (3,),
        "bonds": (),
        "angles": (5,),
        "propers": (2,),
    },
    energy=(152.0, 2.0, 52.5),
    atomic_numbers=(6, 6, 8, 1, 8, 1, 1, 6, 6, 8, 1),
    membership=(0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2),
    head_pos=((156.0, 156.5, 157.0), (4.5, 5.0, 5.5), (62.0, 62.5, 63.0)),
)
#: The same frames reversed. Atom counts are symmetric (4, 3, 4), so ``Z`` and the
#: membership vector are *identical* to the forward request — only the per-frame
#: ``energy`` and the ``pos`` heads can witness that the order was preserved.
_REVERSED = _Request(
    label="reversed [7, 1, 12]",
    frames=(7, 1, 12),
    atom_ptr=(0, 4, 7, 11),
    batch_sizes=_CROSS.batch_sizes,
    energy=(52.5, 2.0, 152.0),
    atomic_numbers=(6, 6, 8, 1, 8, 1, 1, 6, 6, 8, 1),
    membership=(0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2),
    head_pos=((62.0, 62.5, 63.0), (4.5, 5.0, 5.5), (156.0, 156.5, 157.0)),
)
#: Interleaved request: frames (12, 1, 13, 7, 0) -> blocks (2, 0, 2, 1, 0), i.e.
#: two blocks are re-entered after another block was visited. Block-major
#: materialisation order is C(12, 13), A(1, 0), B(7), so the restore permutation
#: is non-identity in both directions — the case neither contiguous request above
#: exercises, and the one where a merge that forgot ``perm`` still looks right on
#: three samples.
_INTERLEAVED = _Request(
    label="interleaved [12, 1, 13, 7, 0]",
    frames=(12, 1, 13, 7, 0),
    atom_ptr=(0, 4, 7, 11, 15, 18),
    batch_sizes={
        "atoms": (18,),
        "edges": (0,),
        "graphs": (5,),
        "bonds": (),
        "angles": (8,),
        "propers": (3,),
    },
    energy=(152.0, 2.0, 152.5, 52.5, 1.5),
    atomic_numbers=(6, 6, 8, 1, 8, 1, 1, 6, 6, 8, 1, 6, 6, 8, 1, 8, 1, 1),
    membership=(0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4),
    head_pos=(
        (156.0, 156.5, 157.0),
        (4.5, 5.0, 5.5),
        (162.0, 162.5, 163.0),
        (62.0, 62.5, 63.0),
        (0.0, 0.5, 1.0),
    ),
)
_REQUESTS: tuple[_Request, ...] = (_CROSS, _REVERSED, _INTERLEAVED)

# ---------------------------------------------------------------------------
# Hard-coded goldens — leg 3, the sampler epoch
# ---------------------------------------------------------------------------

#: Batch shares; equilibrium is configured to 0.0 and must never be planned.
_MIXTURE: Mapping[str, float] = {"equilibrium": 0.0, "relax": 0.5, "torsion": 0.5}
#: Per-batch atom-token budget. No edge budget exists over a block by design.
_SAMPLER_MAX_ATOMS = 8
#: Plan seed.
_SAMPLER_SEED = 42
#: Batch *contents*, sorted — permutation-independent because every batch is one
#: whole unit (mol-b, mol-d, scan-c, scan-e); only their order is seeded.
_GOLDEN_EPOCH_BATCHES: tuple[tuple[int, ...], ...] = (
    (2, 3, 4),
    (5, 6, 7, 8, 9),
    (10,),
    (11, 12, 13, 14),
)
#: Apportioned batch shares over the 4-slot natural epoch.
_GOLDEN_BATCHES_PER_TASK: Mapping[str, int] = {"relax": 2, "torsion": 2}
#: Frames each task contributes; the mixture governs batch shares, not these.
_GOLDEN_FRAMES_PER_TASK: Mapping[str, int] = {"relax": 4, "torsion": 9}

# ---------------------------------------------------------------------------
# Hard-coded goldens — leg 5, the reader-pool bound
# ---------------------------------------------------------------------------

#: The two pool bounds compared for payload identity: below and above ``B = 3``.
_BOUNDED_K: tuple[int, ...] = (1, 8)


def _require(condition: bool, message: str) -> None:
    """Raise :class:`AssertionError` with *message* unless *condition* holds."""
    if not condition:
        raise AssertionError(message)


def _require_equal_tensors(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    """Assert bit-exact tensor equality (dtype included) — never a tolerance."""
    _require(
        actual.dtype == expected.dtype,
        f"{what}: dtype drift {actual.dtype} vs {expected.dtype}",
    )
    _require(
        tuple(actual.shape) == tuple(expected.shape),
        f"{what}: shape drift {tuple(actual.shape)} vs {tuple(expected.shape)}",
    )
    _require(torch.equal(actual, expected), f"{what}: values drift")


def _require_td_equal(actual: TensorDict, expected: TensorDict, path: str = "") -> None:
    """Assert two batches are leaf-for-leaf identical: keys, batch_size, dtype, values."""
    _require(
        set(actual.keys()) == set(expected.keys()),
        f"key sets differ at {path or '/'}: {sorted(actual.keys())} vs {sorted(expected.keys())}",
    )
    _require(
        tuple(actual.batch_size) == tuple(expected.batch_size),
        f"batch_size differs at {path or '/'}: "
        f"{tuple(actual.batch_size)} vs {tuple(expected.batch_size)}",
    )
    for key in sorted(actual.keys()):
        left, right = actual[key], expected[key]
        if isinstance(left, TensorDict) and isinstance(right, TensorDict):
            _require_td_equal(left, right, f"{path}/{key}")
            continue
        _require_equal_tensors(left, right, f"leaf {path}/{key}")


def _require_payload_equal(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    what: str,
    path: str = "",
) -> None:
    """Assert two packed-cache-v1 payload mappings are leaf-for-leaf identical."""
    _require(
        set(actual) == set(expected),
        f"{what}: payload key sets differ at {path or '/'}: {sorted(actual)} vs {sorted(expected)}",
    )
    for key in sorted(actual):
        left, right = actual[key], expected[key]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            _require_payload_equal(left, right, what, f"{path}/{key}")
        elif isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            _require_equal_tensors(left, right, f"{what} leaf {path}/{key}")
        else:
            _require(
                left.__class__ is right.__class__ and left == right,
                f"{what}: leaf drift at {path}/{key}: {left!r} vs {right!r}",
            )


def _cumsum(counts: Sequence[int]) -> tuple[int, ...]:
    """Exclusive cumsum with the total appended — the pointer rule, by hand."""
    out = [0]
    for count in counts:
        out.append(out[-1] + count)
    return tuple(out)


# ---------------------------------------------------------------------------
# Deterministic fixture (no RNG — closed-form integers and halves)
# ---------------------------------------------------------------------------


def _halves(count: int, start: int) -> torch.Tensor:
    """``count`` consecutive multiples of 0.5 — exactly representable in float32.

    Args:
        count: Number of values.
        start: Integer offset applied before scaling, so each unit occupies its
            own distinguishable value range.

    Returns:
        ``(count,)`` float32 tensor ``[start, start+1, ...] * 0.5``.
    """
    return (torch.arange(count, dtype=torch.float32) + float(start)) * 0.5


def _i32(values: Sequence[int]) -> torch.Tensor:
    """``(len(values),)`` int32 column — the block store's topology dtype."""
    return torch.tensor(list(values), dtype=torch.int32)


def _valence(n_atoms: int) -> dict[str, torch.Tensor]:
    """Covalent topology of a linear chain over *n_atoms* atoms.

    Args:
        n_atoms: 3 or 4 — the two unit sizes of the fixture.

    Returns:
        The bond / angle / proper columns, all **molecule-local** indices, so the
        collate's own rebase is the only thing that ever offsets them.
    """
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


def _unit(key: str, task: str, n_atoms: int, n_frames: int, offset: int) -> SamplingUnit:
    """Build one sampling unit: one topology plus *n_frames* frames of it.

    Args:
        key: Unit identity, stored UTF-8 in the block's unit-key dictionary.
        task: Task label, stored as a dictionary code in the ``task`` column.
        n_atoms: Atoms ``N`` of the topology.
        n_frames: Frames ``n`` of that topology.
        offset: Integer offset feeding :func:`_halves`, distinct per unit, so a
            frame's payload identifies the unit it came from by value alone.

    Returns:
        The validated :class:`~molix.store.SamplingUnit`: ``pos`` ``(n*N, 3)`` Å,
        ``forces`` ``(n*N, 3)`` kcal mol⁻¹ Å⁻¹ and ``energy`` ``(n,)`` kcal mol⁻¹.
    """
    rows = n_frames * n_atoms
    return SamplingUnit(
        key=key,
        task=task,
        topology={
            "Z": torch.tensor(_ATOMIC_NUMBERS[n_atoms], dtype=torch.int16),
            **_valence(n_atoms),
        },
        frames={
            "pos": _halves(rows * 3, offset).reshape(rows, 3),
            "forces": -_halves(rows * 3, offset + 7).reshape(rows, 3),
            "energy": _halves(n_frames, offset + 3),
        },
    )


def _write_block(
    root: Path,
    units: Sequence[SamplingUnit],
    *,
    split: str = "train",
    seal: bool = True,
) -> Path:
    """Write *units* into *root* one unit at a time; seal unless told otherwise.

    Args:
        root: Block directory to create.
        units: Sampling units, in append order.
        split: The block's single manifest split.
        seal: Whether to finalize the block. An unsealed block is **not** READY,
            which is exactly what leg 4 needs.

    Returns:
        *root*.
    """
    writer = BlockWriter(root, split=split, provenance=_PROVENANCE)
    for unit in units:
        writer.append_unit(unit)
    if seal:
        writer.seal()
    return root


def _write_fixture(workdir: Path) -> list[Path]:
    """Write the three sealed A/B/C blocks and return their roots in view order."""
    return [
        _write_block(workdir / f"block-{name}", [_unit(*spec) for spec in specs])
        for name, specs in _BLOCK_UNITS.items()
    ]


# ---------------------------------------------------------------------------
# Fixture derivations — the goldens must mean what the header says
# ---------------------------------------------------------------------------


class _Frame(NamedTuple):
    """One global frame, derived in closed form from :data:`_BLOCK_UNITS`."""

    block: int
    local: int
    unit: int
    n_atoms: int
    task: str
    energy: float
    head_pos: tuple[float, float, float]


def _frame_table() -> tuple[_Frame, ...]:
    """Re-derive the whole per-frame index of the fixture from the unit table.

    Returns:
        One :class:`_Frame` per global frame, in view order. ``unit`` is the
        running unit row across blocks (= the globally unique group code),
        ``energy`` is ``(k + offset + 3) * 0.5`` for the ``k``-th frame of its
        unit and ``head_pos`` is ``(3*k*N + offset + j) * 0.5`` for ``j`` in
        ``(0, 1, 2)`` — the first ``pos`` row of that frame.
    """
    frames: list[_Frame] = []
    unit_row = 0
    for block, specs in enumerate(_BLOCK_UNITS.values()):
        local = 0
        for _key, task, n_atoms, n_frames, offset in specs:
            for k in range(n_frames):
                head = 3 * k * n_atoms + offset
                frames.append(
                    _Frame(
                        block=block,
                        local=local,
                        unit=unit_row,
                        n_atoms=n_atoms,
                        task=task,
                        energy=(k + offset + 3) * 0.5,
                        head_pos=(head * 0.5, (head + 1) * 0.5, (head + 2) * 0.5),
                    )
                )
                local += 1
            unit_row += 1
    return tuple(frames)


def _check_fixture_conventions() -> None:
    """Pin every literal to the unit table before it is used as a golden.

    A golden compares numbers against numbers, so it cannot catch a literal that
    was typed wrong in the first place, nor a fixture that quietly stopped being
    exact in ``float32``. Both are re-derived here from :data:`_BLOCK_UNITS`
    alone, in closed form.
    """
    for specs in _BLOCK_UNITS.values():
        for spec in specs:
            unit = _unit(*spec)
            for name, tensor in unit.frames.items():
                doubled = tensor * 2.0
                _require(
                    torch.equal(doubled, doubled.round()),
                    f"{unit.key}.{name} is not a multiple of 0.5 — the fixture must be "
                    "exactly representable in float32 so no tolerance is ever needed",
                )
            for name, tensor in unit.topology.items():
                if not name.startswith(("bond_atom", "angle_atom", "proper_atom")):
                    continue
                if tensor.numel() == 0:
                    continue
                _require(
                    int(tensor.min()) >= 0 and int(tensor.max()) < unit.n_atoms,
                    f"{unit.key}.{name} leaves the molecule-local range [0, {unit.n_atoms}) — "
                    "block topology indices are never global",
                )

    table = _frame_table()
    per_block_frames = [sum(spec[3] for spec in specs) for specs in _BLOCK_UNITS.values()]
    per_block_units = [len(specs) for specs in _BLOCK_UNITS.values()]
    names: list[str] = []
    for frame in table:
        if frame.task not in names:
            names.append(frame.task)

    typo = "a typo in the literal must not redefine the golden it is compared against"
    for label, count, derived_count in (
        ("_LEN", _LEN, len(table)),
        ("_TOTAL_UNITS", _TOTAL_UNITS, sum(per_block_units)),
        ("_VIEW_MAX_ATOMS", _VIEW_MAX_ATOMS, max(frame.n_atoms for frame in table)),
    ):
        _require(
            count == derived_count,
            f"{label} = {count} is not what the unit table implies ({derived_count}) — {typo}",
        )
    for label, vector, derived_vector in (
        ("_FRAME_PTR", _FRAME_PTR, _cumsum(per_block_frames)),
        ("_UNIT_PTR", _UNIT_PTR, _cumsum(per_block_units)),
        ("_ATOM_COUNTS", _ATOM_COUNTS, tuple(frame.n_atoms for frame in table)),
        ("_GROUP_CODES", _GROUP_CODES, tuple(frame.unit for frame in table)),
        ("_TASK_CODES", _TASK_CODES, tuple(names.index(frame.task) for frame in table)),
        ("_LOCATE_BLOCKS", _LOCATE_BLOCKS, tuple(table[f].block for f in _LOCATE_FRAMES)),
        ("_LOCATE_LOCALS", _LOCATE_LOCALS, tuple(table[f].local for f in _LOCATE_FRAMES)),
        ("_TASK_NAMES", _TASK_NAMES, tuple(names)),
        ("_BLOCK_IDS", _BLOCK_IDS, tuple(f"block-{name}" for name in _BLOCK_UNITS)),
    ):
        _require(
            vector == derived_vector,
            f"{label} = {vector} is not what the unit table implies ({derived_vector}) — {typo}",
        )

    for request in _REQUESTS:
        counts = [table[frame].n_atoms for frame in request.frames]
        derived_request = {
            "atom_ptr": _cumsum(counts),
            "energy": tuple(table[frame].energy for frame in request.frames),
            "atomic_numbers": tuple(
                z for frame in request.frames for z in _ATOMIC_NUMBERS[table[frame].n_atoms]
            ),
            "membership": tuple(slot for slot, count in enumerate(counts) for _ in range(count)),
            "head_pos": tuple(table[frame].head_pos for frame in request.frames),
        }
        for field, derived_value in derived_request.items():
            literal_value = tuple(getattr(request, field))
            _require(
                literal_value == derived_value,
                f"{request.label}: {field} literal {literal_value} is not what the unit table "
                f"implies ({derived_value})",
            )
        _require(
            request.batch_sizes["atoms"] == (sum(counts),),
            f"{request.label}: atoms batch_size {request.batch_sizes['atoms']} disagrees with "
            f"the selected atom counts {counts}",
        )
        _require(
            request.batch_sizes["graphs"] == (len(request.frames),),
            f"{request.label}: graphs batch_size {request.batch_sizes['graphs']} disagrees with "
            f"the {len(request.frames)} requested frames",
        )

    planned = tuple(sorted(frame for batch in _GOLDEN_EPOCH_BATCHES for frame in batch))
    _require(
        len(planned) == len(set(planned)),
        f"_GOLDEN_EPOCH_BATCHES repeats a frame: {planned}",
    )
    per_task: dict[str, int] = {}
    for frame in planned:
        per_task[table[frame].task] = per_task.get(table[frame].task, 0) + 1
    _require(
        per_task == dict(_GOLDEN_FRAMES_PER_TASK),
        f"_GOLDEN_FRAMES_PER_TASK {dict(_GOLDEN_FRAMES_PER_TASK)} disagrees with the golden "
        f"batches ({per_task})",
    )
    _require(
        all(
            sum(table[frame].n_atoms for frame in batch) <= _SAMPLER_MAX_ATOMS
            or len({table[frame].unit for frame in batch}) == 1
            for batch in _GOLDEN_EPOCH_BATCHES
        ),
        "a golden batch exceeds the atom budget without being a single whole unit — the "
        "oversize-singleton rule is the only reason a batch may cross the budget",
    )


# ---------------------------------------------------------------------------
# Leg 1 — the global index map and the concatenated sampler tables
# ---------------------------------------------------------------------------


def _derived_pointers(view: BlockDataset) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Re-derive ``(frame_ptr, unit_ptr)`` from the view's **public** surface.

    Neither pointer is a public attribute — and it need not be: ``locate`` over
    every frame yields the per-block frame counts, and the group codes seen in a
    block yield its unit count. Deriving them this way asserts the property they
    encode (a contiguous, gap-free block partition of both spaces) rather than
    reading a field.

    Args:
        view: The multi-block view.

    Returns:
        ``(frame_ptr, unit_ptr)``, both ``(B+1,)`` exclusive cumsums.

    Raises:
        AssertionError: A block's frames are not the contiguous global rows
            ``[frame_ptr[b], frame_ptr[b+1])``, or its group codes are not the
            contiguous unit rows ``[unit_ptr[b], unit_ptr[b+1])``.
    """
    blocks, locals_ = view.locate(list(range(len(view))))
    rows = blocks.tolist()
    group_codes = view.group_codes.tolist()

    frames_of_block: list[list[int]] = [[] for _ in view.blocks]
    groups_of_block: list[set[int]] = [set() for _ in view.blocks]
    for frame, block in enumerate(rows):
        frames_of_block[block].append(frame)
        groups_of_block[block].add(int(group_codes[frame]))

    frame_ptr = _cumsum([len(frames) for frames in frames_of_block])
    unit_ptr = _cumsum([len(groups) for groups in groups_of_block])
    for block in range(len(view.blocks)):
        span = list(range(frame_ptr[block], frame_ptr[block + 1]))
        _require(
            frames_of_block[block] == span,
            f"block {block} owns global frames {frames_of_block[block]}, not the contiguous "
            f"span {span} — the index map is not a block partition",
        )
        _require(
            [int(locals_[frame]) for frame in span] == list(range(len(span))),
            f"block {block} local rows are not 0..{len(span) - 1} in global frame order",
        )
        units = list(range(unit_ptr[block], unit_ptr[block + 1]))
        _require(
            sorted(groups_of_block[block]) == units,
            f"block {block} carries group codes {sorted(groups_of_block[block])}, not the "
            f"offset image {units} — group ids would collide across blocks",
        )
    return frame_ptr, unit_ptr


def _check_index_and_tables(view: BlockDataset, roots: Sequence[Path]) -> None:
    """Leg 1: pointers, ``len``, ``locate`` and the three tables equal the goldens.

    Args:
        view: The multi-block view.
        roots: The planned block roots, re-opened one at a time as an independent
            per-block count source.
    """
    _require(len(view) == _LEN, f"len(view) = {len(view)}, hand-computed {_LEN}")
    frame_ptr, unit_ptr = _derived_pointers(view)
    _require(frame_ptr == _FRAME_PTR, f"frame_ptr = {frame_ptr}, hand-computed {_FRAME_PTR}")
    _require(unit_ptr == _UNIT_PTR, f"unit_ptr = {unit_ptr}, hand-computed {_UNIT_PTR}")

    # Same two pointers, from the per-block readers: the view's index algebra is
    # the cumsum of the block facts, not a second bookkeeping of them.
    per_block = [(BlockReader(root).frame_count, BlockReader(root).unit_count) for root in roots]
    _require(
        _cumsum([count for count, _ in per_block]) == _FRAME_PTR,
        f"per-block frame counts {[c for c, _ in per_block]} do not cumsum to {_FRAME_PTR}",
    )
    _require(
        _cumsum([count for _, count in per_block]) == _UNIT_PTR,
        f"per-block unit counts {[c for _, c in per_block]} do not cumsum to {_UNIT_PTR}",
    )
    _require(
        frame_ptr[-1] == len(view) and unit_ptr[-1] == _TOTAL_UNITS,
        f"pointer totals {frame_ptr[-1]} / {unit_ptr[-1]} disagree with len(view) = "
        f"{len(view)} and {_TOTAL_UNITS} units",
    )

    blocks, locals_ = view.locate(list(_LOCATE_FRAMES))
    _require_equal_tensors(
        blocks, torch.tensor(_LOCATE_BLOCKS, dtype=torch.long), "locate block ordinals"
    )
    _require_equal_tensors(
        locals_, torch.tensor(_LOCATE_LOCALS, dtype=torch.long), "locate local rows"
    )

    for name, actual, expected in (
        ("atom_counts", view.atom_counts, _ATOM_COUNTS),
        ("group_codes", view.group_codes, _GROUP_CODES),
        ("task_codes", view.task_codes, _TASK_CODES),
    ):
        _require_equal_tensors(actual, torch.tensor(expected, dtype=torch.long), f"view.{name}")

    _require(
        view.task_names == _TASK_NAMES,
        f"task_names = {view.task_names}, hand-computed merge {_TASK_NAMES}",
    )
    # The merge is non-identity for block C: its manifest dictionary is
    # ("relax", "torsion") -> local codes (0, 1), which become global (1, 2).
    _require(
        BlockReader(roots[2]).task_names == ("relax", "torsion"),
        f"block C dictionary {BlockReader(roots[2]).task_names} is not the non-identity case "
        "this golden exists to exercise",
    )
    _require(
        BlockReader(roots[2]).task_codes.tolist() == [0, 1, 1, 1, 1]
        and view.task_codes[_FRAME_PTR[2] :].tolist() == [1, 2, 2, 2, 2],
        "block C's local task codes were not remapped into the merged dictionary",
    )
    _require(
        set(view.group_codes.tolist()) == set(range(_TOTAL_UNITS)),
        f"group codes {sorted(set(view.group_codes.tolist()))} do not partition "
        f"range({_TOTAL_UNITS}) — a group id would collide across blocks",
    )
    _require(view.split == _SPLIT, f"split = {view.split!r}, expected {_SPLIT!r}")
    _require(
        view.max_atoms == _VIEW_MAX_ATOMS,
        f"max_atoms = {view.max_atoms}, hand-computed {_VIEW_MAX_ATOMS}",
    )
    _require(
        view.block_ids == _BLOCK_IDS,
        f"block_ids = {view.block_ids}, expected the directory basenames {_BLOCK_IDS}",
    )
    _require(
        tuple(view.blocks) == tuple(Path(root) for root in roots),
        f"blocks = {view.blocks} is not the given plan order {list(roots)}",
    )


# ---------------------------------------------------------------------------
# Leg 2 — cross-block payload parity, in caller order
# ---------------------------------------------------------------------------


def _check_request(view: BlockDataset, request: _Request) -> None:
    """Leg 2: one ``frame_payload`` request equals its goldens and the oracle.

    Args:
        view: The multi-block view.
        request: The frame list plus its hand-derived batch goldens.
    """
    frames = list(request.frames)
    payload = view.frame_payload(frames)
    _require(
        payload["n_samples"] == len(frames),
        f"{request.label}: n_samples {payload['n_samples']} != {len(frames)} requested frames",
    )
    _require(
        tuple(payload["atom_ptr"].tolist()) == request.atom_ptr,
        f"{request.label}: atom_ptr {payload['atom_ptr'].tolist()}, hand-computed "
        f"{list(request.atom_ptr)}",
    )

    batch = collate_packed(PackedView(payload), range(len(frames)))
    _require(
        tuple(batch.batch_size) == (),
        f"{request.label}: top-level batch_size {tuple(batch.batch_size)} != ()",
    )
    for namespace, expected in request.batch_sizes.items():
        got = tuple(batch[namespace].batch_size)
        _require(
            got == expected,
            f"{request.label}: {namespace}.batch_size {got} != hand-computed {expected}",
        )

    _require_equal_tensors(
        batch["graphs"]["energy"],
        torch.tensor(request.energy, dtype=torch.float32),
        f"{request.label}: graphs.energy (kcal/mol, caller order)",
    )
    _require_equal_tensors(
        batch["atoms"]["Z"],
        torch.tensor(request.atomic_numbers, dtype=torch.long),
        f"{request.label}: atoms.Z",
    )
    _require_equal_tensors(
        batch["atoms"]["batch"],
        torch.tensor(request.membership, dtype=torch.long),
        f"{request.label}: atoms.batch (graph membership)",
    )
    heads = batch["atoms"]["pos"][torch.tensor(request.atom_ptr[:-1], dtype=torch.long)]
    _require_equal_tensors(
        heads,
        torch.tensor(request.head_pos, dtype=torch.float32),
        f"{request.label}: first pos row (angstrom) of every sample",
    )

    # The single-data-route oracle: the flat sample dicts of the same frames,
    # collated the ordinary way, must be the identical batch leaf for leaf.
    _require_td_equal(batch, collate_molecules([view[frame] for frame in frames]))


# ---------------------------------------------------------------------------
# Leg 3 — the sampler epoch over the view
# ---------------------------------------------------------------------------


def _sampler(view: BlockDataset) -> TaskMixtureBatchSampler:
    """Build the epoch of the spec: block-native code tables, atoms-only budget.

    Args:
        view: The multi-block view, whose concatenated tables are handed over
            verbatim — no per-frame Python label object is materialized.

    Returns:
        The eagerly planned sampler. ``max_edges`` is ``None`` because a block
        carries no neighbour graph; budgeting on fabricated zeros would be a lie.
    """
    return TaskMixtureBatchSampler(
        view,
        indices=list(range(len(view))),
        task_labels=view.task_codes,
        group_ids=view.group_codes,
        task_names=view.task_names,
        mixture=_MIXTURE,
        max_atoms=_SAMPLER_MAX_ATOMS,
        max_edges=None,
        seed=_SAMPLER_SEED,
    )


def _check_sampler_epoch(view: BlockDataset) -> None:
    """Leg 3: single-task batches, whole groups, determinism, per-task totals.

    Args:
        view: The multi-block view.
    """
    sampler = _sampler(view)
    plan = [list(batch) for batch in sampler]
    contents = sorted(tuple(batch) for batch in plan)
    _require(
        contents == sorted(_GOLDEN_EPOCH_BATCHES),
        f"epoch contents {contents}, hand-computed {sorted(_GOLDEN_EPOCH_BATCHES)}",
    )
    _require(
        len(sampler) == len(_GOLDEN_EPOCH_BATCHES),
        f"epoch length {len(sampler)}, hand-computed {len(_GOLDEN_EPOCH_BATCHES)}",
    )

    task_codes = view.task_codes.tolist()
    group_codes = view.group_codes.tolist()
    frames_of_group: dict[int, set[int]] = {}
    for frame, group in enumerate(group_codes):
        frames_of_group.setdefault(int(group), set()).add(frame)

    seen: list[int] = []
    for batch in plan:
        _require(bool(batch), "every batch must be non-empty")
        tasks = {task_codes[frame] for frame in batch}
        _require(len(tasks) == 1, f"batch {batch} mixes task codes {sorted(tasks)}")
        _require(
            view.task_names[tasks.pop()] != "equilibrium",
            f"batch {batch} carries equilibrium, whose mixture share is 0.0",
        )
        for group in {int(group_codes[frame]) for frame in batch}:
            missing = sorted(frames_of_group[group] - set(batch))
            _require(
                not missing,
                f"group {group} straddles batch {batch}; frames {missing} landed elsewhere — a "
                "molecule must never be split across batches",
            )
        seen.extend(batch)
    _require(len(seen) == len(set(seen)), f"an epoch must not repeat a frame: {seen}")

    _require(
        sampler.batches_per_task() == dict(_GOLDEN_BATCHES_PER_TASK),
        f"batches_per_task {sampler.batches_per_task()}, hand-computed "
        f"{dict(_GOLDEN_BATCHES_PER_TASK)}",
    )
    _require(
        sampler.frames_per_task() == dict(_GOLDEN_FRAMES_PER_TASK),
        f"frames_per_task {sampler.frames_per_task()}, hand-computed "
        f"{dict(_GOLDEN_FRAMES_PER_TASK)}",
    )

    # Determinism: a second, independently constructed sampler at the same seed
    # plans the same epoch, batch order included.
    _require(
        [list(batch) for batch in _sampler(view)] == plan,
        "two identical constructions at the same seed disagree — the plan is not deterministic",
    )
    _require(
        [list(batch) for batch in sampler] == plan,
        "iterating the same sampler twice must yield the same plan",
    )


# ---------------------------------------------------------------------------
# Leg 4 — the three typed refusals
# ---------------------------------------------------------------------------


def _check_refusals(workdir: Path, roots: Sequence[Path], view: BlockDataset) -> None:
    """Leg 4: mixed split, unsealed block and ``edge_counts``, each by named type.

    Args:
        workdir: Scratch directory for the two guard blocks.
        roots: The three sealed fixture roots.
        view: A constructed view over them, for the ``edge_counts`` refusal.
    """
    valid = _write_block(
        workdir / "block-valid",
        [_unit("mol-v", "relax", 3, 2, 500)],
        split="valid",
    )
    try:
        BlockDataset([roots[0], valid])
    except BlockSplitMismatchError as exc:
        message = str(exc)
        for expected in (str(roots[0]), str(valid), "train", "valid"):
            _require(
                expected in message,
                f"the split refusal does not name {expected!r}: {message}",
            )
    else:
        raise AssertionError(
            "a train block and a valid block built one view — splits must never be mixed"
        )

    unsealed = _write_block(
        workdir / "block-unsealed",
        [_unit("mol-u", "relax", 3, 2, 400)],
        seal=False,
    )
    planned = [roots[0], roots[1], unsealed]
    try:
        BlockDataset(planned)
    except BlockNotReadyError as exc:
        message = str(exc)
        _require(
            str(unsealed) in message and "unsealed" in message.lower(),
            f"the readiness refusal does not name the unsealed block and its state: {message}",
        )
    else:
        raise AssertionError(
            "an unsealed block was admitted — a planned block is never skipped and never "
            "substituted, so the view must stall instead of silently shortening the epoch"
        )
    # The refusal is about that block, not about the plan around it: the two
    # sealed blocks of the same plan still build a view of their own.
    _require(
        len(BlockDataset(planned[:2])) == _FRAME_PTR[2],
        "the two sealed blocks of the refused plan do not form a view of their own",
    )

    try:
        _ = view.edge_counts
    except ValueError as exc:
        _require(
            "NeighborList" in str(exc),
            f"edge_counts raised without pointing at the downstream builder: {exc}",
        )
    else:
        raise AssertionError(
            "edge_counts returned a value — blocks carry no neighbour-graph columns, and a "
            "fabricated zero would hand a token-budget sampler a false budget"
        )

    for guard in (BlockNotReadyError, BlockSplitMismatchError):
        _require(
            issubclass(guard, ValueError),
            f"{guard.__name__} must subclass ValueError so existing handlers keep working",
        )


# ---------------------------------------------------------------------------
# Leg 5 — the bounded reader pool
# ---------------------------------------------------------------------------


def _check_bounded_pool(roots: Sequence[Path]) -> None:
    """Leg 5: ``open_blocks`` stays under ``K`` and the payload does not move.

    Args:
        roots: The three sealed fixture roots.
    """
    payloads: list[Mapping[str, object]] = []
    for max_open in _BOUNDED_K:
        bounded = BlockDataset(roots, max_open=max_open)
        _require(
            bounded.max_open == max_open,
            f"max_open reports {bounded.max_open} for a view built with {max_open}",
        )
        for frame in range(len(bounded)):
            bounded.frame_payload([frame])
            _require(
                len(bounded.open_blocks) <= max_open,
                f"open_blocks {bounded.open_blocks} exceeds the pool bound {max_open} after "
                f"reading frame {frame}",
            )
        payloads.append(bounded.frame_payload(list(_INTERLEAVED.frames)))

    _require_payload_equal(
        payloads[0],
        payloads[1],
        f"max_open={_BOUNDED_K[0]} vs max_open={_BOUNDED_K[1]}",
    )


# ---------------------------------------------------------------------------


def main() -> int:
    _check_fixture_conventions()

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        roots = _write_fixture(workdir)
        view = BlockDataset(roots)

        _check_index_and_tables(view, roots)
        for request in _REQUESTS:
            _check_request(view, request)
        _check_sampler_epoch(view)
        _check_refusals(workdir, roots, view)
        _check_bounded_pool(roots)

    print(
        "streaming-data-tiers-04-scheduler-integration-02-view: three sealed train blocks "
        "(5+5+5 frames, 2+1+2 units) -> one BlockDataset with frame_ptr "
        f"{_FRAME_PTR}, unit_ptr {_UNIT_PTR}, len {_LEN}, atom_counts {_ATOM_COUNTS}, "
        f"group_codes {_GROUP_CODES}, task_names {_TASK_NAMES}, task_codes {_TASK_CODES}; "
        f"cross-block frame_payload {list(_CROSS.frames)}, its reverse and the interleaved "
        f"{list(_INTERLEAVED.frames)} collate leaf-for-leaf against collate_molecules "
        "(atoms 11/11/18, graphs 3/3/5, edges 0); TaskMixtureBatchSampler(mixture "
        f"0.0/0.5/0.5, max_atoms={_SAMPLER_MAX_ATOMS}, max_edges=None, seed={_SAMPLER_SEED}) "
        f"plans {list(_GOLDEN_EPOCH_BATCHES)} — single-task, group-integral, deterministic, "
        f"frames_per_task {dict(_GOLDEN_FRAMES_PER_TASK)}; mixed split, unsealed block and "
        f"edge_counts all refused by type; max_open {_BOUNDED_K[0]} vs {_BOUNDED_K[1]} "
        "payload-identical with open_blocks never above the cap OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
