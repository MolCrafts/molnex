"""Public-API regression for the rehomed transforms — graph, valence, extxyz.

Spec: ``molix-data-dissolution-03-transforms`` (acceptance ac-008).

Three modules left the acquisition layer without changing a line of behaviour:
the neighbour-list transform and the connectivity column adapters moved into
``molix.batch``, the extended-XYZ codec into ``molix.io``. This script pins the
move at the *user* surface — it imports only ``molix.batch`` /
``molix.io.extxyz`` and never a private module, so it fails the moment a public
re-export or an encoded convention drifts.

**Part A — the edge convention.** Three atoms on the x axis one Angstrom apart
at ``cutoff = 1.5 A``. The separations are 1.0 / 1.0 / 2.0 A, so the pair *set*
sits half an Angstrom clear of the cutoff either way and cannot flip on
arithmetic noise. The C++ kernel returns upper-triangle half pairs
(``edge_index[:, 0] > edge_index[:, 1]``) and ``symmetry=True`` appends the
reverse of each, in order, so the whole tensor is fixed by the documented
contract::

    row  edge_index  meaning            edge_diff = pos[target] - pos[source]
    ---  ----------  -----------------  -------------------------------------
    0    [1, 0]      half pair (1,0)    pos[0] - pos[1] = [-1, 0, 0]
    1    [2, 1]      half pair (2,1)    pos[1] - pos[2] = [-1, 0, 0]
    2    [0, 1]      reverse of row 0   pos[1] - pos[0] = [ 1, 0, 0]
    3    [1, 2]      reverse of row 1   pos[2] - pos[1] = [ 1, 0, 0]

``edge_dist = |edge_diff| = 1.0`` on every row, exact in binary (unit vector
along x). ``edge_diff`` is re-derived from ``pos`` and ``edge_index`` in closed
form before it is used as an oracle, so a typo in the table cannot quietly
redefine the golden. The ``task_id`` string is pinned byte-for-byte at the
default construction — it is the cache key every on-disk pipeline cache was
built under, so a single character of drift silently changes what a stored
cache means.

**Part B — valence columns to COO.** A hard-coded two-angle namespace in
column form (``atomi`` / ``atomj`` central / ``atomk``) stacks to ``[3, 2]``
with the columns as *rows* — the transpose is the whole point of the helper,
and getting it backwards would feed a kernel the atom triples of a different
angle.

**Part C — the extxyz codec, both halves.** The writer emits its two frames as
literal text (``Properties=species:S:1:pos:R:3``, ``energy=`` at ten decimals,
positions at eight), pinned line by line. The parser is pinned on a
hand-written two-frame buffer carrying ``Lattice="..."``: the recovered cell is
the row-major ``diag(10, 11, 12)`` it declares and the positions come back
exactly as written.

    Known gap (not exercised here): ``write_extxyz_frames`` emits no
    ``Lattice`` tag, while ``parse_extxyz_frames`` raises when the tag is
    absent, so the writer's own output is not readable by its parser today —
    see ``src/molix/io/extxyz.py`` (writer docstring vs ``_parse_lattice``).
    That asymmetry predates this move and is reported for ``/mol:fix``; the
    two halves are pinned independently here rather than papered over.

**Part D — dissolution.** The old homes are *gone*, not shimmed:
``molix.data.tasks.neighbor`` and ``molix.datasets._extxyz`` no longer resolve.

    capture: none — closed-form literals. The edge table is hand geometry on
             three collinear atoms, the COO stack is a transpose of hand
             indices, and the extxyz text is the writer's documented format
             spelled out. Nothing was captured from a run, and no third-party
             software is imported, subprocessed or consulted, here or at
             authoring time.
    verify:  PYTHONPATH=src python regressions/molix-data-dissolution-03-transforms.py
    commit:  9d69ebd (+ working tree implementing molix-data-dissolution-03-transforms)
    torch:   2.12.1+cpu
    device:  cpu, float32 (exact equality throughout — no tolerance is used)
    date:    2026-08-14
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from molix.batch import NeighborList
from molix.batch.valence_columns import stack_angle_index
from molix.io.extxyz import parse_extxyz_frames, write_extxyz_frames

# ---------------------------------------------------------------------------
# Part A goldens — three collinear atoms, 1 A apart
# ---------------------------------------------------------------------------

#: Atoms on the x axis at 0, 1, 2 A; y = z = 0, so every separation reads off
#: the x column by eye.
_POS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]

#: Between the nearest-neighbour separation (1.0 A, in) and the end-to-end one
#: (2.0 A, out), half an Angstrom clear of both.
_CUTOFF = 1.5

#: Half pairs first (source > target), then the reverse of each, in order.
_EDGE_INDEX = [[1, 0], [2, 1], [0, 1], [1, 2]]

#: pos[target] - pos[source] — the half pairs point backwards along x, their
#: reverses forwards.
_EDGE_DIFF = [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

#: |edge_diff| on every row — exactly 1.0, no floating-point residue.
_EDGE_DIST = [1.0, 1.0, 1.0, 1.0]

#: Cache identity at the default construction, byte-for-byte.
_TASK_ID = "nlist:cut=5.0:max=512:pbc=False:sym=True"

# ---------------------------------------------------------------------------
# Part B goldens — two angles in column form
# ---------------------------------------------------------------------------

#: Angle 0 = (0, 1, 2), angle 1 = (1, 2, 3); ``atomj`` is the central atom.
_ANGLE_COLUMNS = {"atomi": [0, 1], "atomj": [1, 2], "atomk": [2, 3]}

#: Columns become rows: row 0 is every ``atomi``, row 1 every ``atomj``, ...
_ANGLE_COO = [[0, 1], [1, 2], [2, 3]]

# ---------------------------------------------------------------------------
# Part C goldens — the extxyz codec
# ---------------------------------------------------------------------------

#: Two symbols, held constant across frames by the writer's contract.
_SPECIES = ["O", "H"]

#: (T=2, N=2, 3) positions in A — frame 0 displaces H along x, frame 1 along y.
_FRAMES = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]

#: Per-frame total energies in eV.
_ENERGIES = [-1.5, -2.5]

#: The writer's documented emission: count / comment / one line per atom, with
#: ``energy`` at ten decimals and coordinates at eight.
_WRITTEN_LINES = [
    "2",
    "Properties=species:S:1:pos:R:3 energy=-1.5000000000",
    "O 0.00000000 0.00000000 0.00000000",
    "H 1.00000000 0.00000000 0.00000000",
    "2",
    "Properties=species:S:1:pos:R:3 energy=-2.5000000000",
    "O 0.00000000 0.00000000 0.00000000",
    "H 0.00000000 1.00000000 0.00000000",
]

#: Hand-written parser input — a two-frame buffer declaring an orthorhombic
#: cell, so the row-major reshape of ``Lattice`` is observable per axis.
_PARSE_SOURCE = (
    "2\n"
    'Lattice="10.0 0.0 0.0 0.0 11.0 0.0 0.0 0.0 12.0" '
    'Properties=species:S:1:pos:R:3 energy=-1.5 pbc="T T T"\n'
    "O 0.00000000 0.00000000 0.00000000\n"
    "H 1.00000000 0.00000000 0.00000000\n"
    "2\n"
    'Lattice="10.0 0.0 0.0 0.0 11.0 0.0 0.0 0.0 12.0" '
    'Properties=species:S:1:pos:R:3 energy=-2.5 pbc="T T T"\n'
    "O 0.00000000 0.00000000 0.00000000\n"
    "H 0.00000000 1.00000000 0.00000000\n"
)

#: Row-major (3, 3) cell the buffer above declares.
_PARSED_CELL = [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]]

# ---------------------------------------------------------------------------
# Part D goldens — the vacated module addresses
# ---------------------------------------------------------------------------

#: Fully-qualified names that must no longer resolve to anything importable.
_VACATED_MODULES = ("molix.data.tasks.neighbor", "molix.datasets._extxyz")


def _check_edge_goldens_are_the_convention() -> None:
    """Derive Part A's literals from ``pos`` alone before trusting them.

    A round-trip compares what came back against what went in, so the literals
    must be checked against the *convention* they claim to encode — otherwise a
    typo in this file would silently redefine the golden.

    Raises:
        AssertionError: A literal does not match its closed-form derivation.
    """
    pos = torch.tensor(_POS)
    edge_index = torch.tensor(_EDGE_INDEX, dtype=torch.long)
    diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]  # target - source
    if not torch.equal(diff, torch.tensor(_EDGE_DIFF)):
        raise AssertionError(
            f"_EDGE_DIFF is not pos[target] - pos[source]: {_EDGE_DIFF} vs {diff.tolist()}"
        )
    if not torch.equal(diff.norm(dim=-1), torch.tensor(_EDGE_DIST)):
        raise AssertionError(
            f"_EDGE_DIST is not |edge_diff|: {_EDGE_DIST} vs {diff.norm(dim=-1).tolist()}"
        )
    half = edge_index[: len(_EDGE_INDEX) // 2]
    if not bool((half[:, 0] > half[:, 1]).all()):
        raise AssertionError(f"_EDGE_INDEX does not lead with upper-triangle pairs: {_EDGE_INDEX}")


def _check_neighbor_list() -> None:
    """Part A: the chain's edge tensors and the cache identity.

    Raises:
        AssertionError: Any edge field or the ``task_id`` string drifted.
    """
    out = NeighborList(cutoff=_CUTOFF, max_num_pairs=8, symmetry=True).execute(
        {"pos": torch.tensor(_POS)}
    )
    expected = [
        ("edge_index", torch.tensor(_EDGE_INDEX, dtype=torch.long)),
        ("edge_diff", torch.tensor(_EDGE_DIFF)),
        ("edge_dist", torch.tensor(_EDGE_DIST)),
    ]
    for key, golden in expected:
        if not torch.equal(out[key], golden):
            raise AssertionError(
                f"NeighborList.{key} drift: {out[key].tolist()} vs {golden.tolist()}"
            )

    task_id = NeighborList().task_id
    if task_id != _TASK_ID:
        raise AssertionError(
            f"NeighborList().task_id drift: {task_id!r} vs {_TASK_ID!r} — every "
            "on-disk cache built under the old string now means something else"
        )


def _check_angle_stack() -> None:
    """Part B: the angles namespace stacks column-wise into ``[3, N]``.

    Raises:
        AssertionError: The COO table is not the transpose of the columns.
    """
    columns = {k: torch.tensor(v, dtype=torch.long) for k, v in _ANGLE_COLUMNS.items()}
    coo = stack_angle_index(columns)
    golden = torch.tensor(_ANGLE_COO, dtype=torch.long)
    if tuple(coo.shape) != (3, 2):
        raise AssertionError(f"stack_angle_index shape {tuple(coo.shape)} != (3, 2)")
    if not torch.equal(coo, golden):
        raise AssertionError(f"stack_angle_index drift: {coo.tolist()} vs {golden}")


def _check_extxyz(workdir: Path) -> None:
    """Part C: the writer's emission and the parser's recovery, pinned apart.

    Args:
        workdir: Directory for the two scratch ``.xyz`` files.

    Raises:
        AssertionError: A written line or a parsed field drifted.
    """
    written = workdir / "written.xyz"
    write_extxyz_frames(
        written,
        species=_SPECIES,
        positions=np.array(_FRAMES, dtype=np.float64),
        energies=np.array(_ENERGIES, dtype=np.float64),
    )
    lines = written.read_text().splitlines()
    if lines != _WRITTEN_LINES:
        raise AssertionError(f"write_extxyz_frames drift:\n{lines}\nvs\n{_WRITTEN_LINES}")

    source = workdir / "source.xyz"
    source.write_text(_PARSE_SOURCE)
    frames = parse_extxyz_frames(source)
    if len(frames) != 2:
        raise AssertionError(f"parse_extxyz_frames returned {len(frames)} frames, want 2")
    for idx, frame in enumerate(frames):
        if not np.array_equal(frame.cell, np.array(_PARSED_CELL)):
            raise AssertionError(
                f"frame {idx} cell drift: {frame.cell.tolist()} vs {_PARSED_CELL} "
                "(Lattice must reshape row-major)"
            )
        if not np.array_equal(frame.pos, np.array(_FRAMES[idx])):
            raise AssertionError(f"frame {idx} pos drift: {frame.pos.tolist()} vs {_FRAMES[idx]}")
        if frame.species != _SPECIES:
            raise AssertionError(f"frame {idx} species drift: {frame.species} vs {_SPECIES}")


def _resolves(module: str) -> bool:
    """Whether *module* can still be located by the import system.

    ``find_spec`` imports the parent package first, so once phase 04 deleted
    ``molix.data`` / ``molix.datasets`` entirely a submodule query raises
    instead of returning ``None``. A missing parent is the strongest form of
    "gone", so it is reported as "does not resolve" rather than propagated.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _check_dissolved() -> None:
    """Part D: the vacated module addresses are gone, not shimmed.

    Raises:
        AssertionError: An old address still resolves.
    """
    for module in _VACATED_MODULES:
        if _resolves(module):
            raise AssertionError(
                f"{module} is still importable — the transform must live only at "
                "its new address, with no shim left behind"
            )


def main() -> int:
    _check_dissolved()
    _check_edge_goldens_are_the_convention()
    _check_neighbor_list()
    _check_angle_stack()
    with tempfile.TemporaryDirectory() as tmp:
        _check_extxyz(Path(tmp))

    print(
        "molix-data-dissolution-03-transforms: molix.batch.NeighborList on the "
        "3-atom chain gives edge_index=[[1,0],[2,1],[0,1],[1,2]], "
        "edge_diff=[[-1,0,0],[-1,0,0],[1,0,0],[1,0,0]], edge_dist=[1,1,1,1] and "
        f"task_id={_TASK_ID!r}; stack_angle_index -> [3, 2] COO {_ANGLE_COO}; "
        "molix.io.extxyz writes 8 literal lines and parses diag(10,11,12) back; "
        "molix.data.tasks.neighbor and molix.datasets._extxyz absent OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
