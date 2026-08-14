"""Public-API regression for the READY-tier block store — column blocks on disk.

Spec: ``streaming-data-tiers-02-block-format`` (acceptance ac-010).

A **block** is one directory of raw little-endian column files plus a
``manifest.json`` and a ``commit.log``, written one :class:`SamplingUnit` at a
time and finalized by ``seal()``. This script drives that surface the way a
converter would — ``from molix.store import BlockManifest, BlockReader,
BlockWriter, SamplingUnit`` (plus the packed collate for the parity oracle) and
nothing private — so it fails the moment the public API, the on-disk byte
layout, or the payload route table drifts.

Fixture (deterministic — no RNG anywhere, and every float is a multiple of
``0.5``, hence exact in ``float32``; nothing here has an FP tolerance)::

    block A  split "train"   2 units, 5 frames, 11 frame-atoms
      unit 0  "mol-a"  task "equilibrium"  Z=[8,1,1]    N=3  n_frames=2
              2 bonds, 1 angle, 0 propers
      unit 1  "mol-b"  task "relax"        Z=[6,6,1,1]  N=4  n_frames=3
              3 bonds, 2 angles, 1 proper
    block B  split "valid"   1 unit, 5 frames, 20 frame-atoms
      unit 0  "scan-c" task "torsion"      Z=[6,6,8,1]  N=4  n_frames=5
              3 bonds, 2 angles, 1 proper

Block A is exactly the spec's hand-computed arithmetic fixture ``N = (3, 4)``,
``n_frames = (2, 3)``, so the derived (never stored) pointers are literals:
``mol_atom_ptr = (0, 3, 7)``, ``mol_frame_ptr = (0, 2, 5)``,
``mol_frame_atom_ptr = (0, 6, 18)``, ``frame_mol = (0, 0, 1, 1, 1)``,
``atom_counts = (3, 3, 4, 4, 4)``.

Legs, in order:

1. **Streaming append + torn-tail resume.** Block A is written twice: once
   uninterrupted, and once with a simulated crash after the last unit's bytes
   were fsynced — ``commit.log`` is truncated **mid-record** and 24 junk bytes
   are appended to ``frame_pos.bin`` through raw file ops. ``BlockWriter.resume``
   must roll both back to the last crc32-valid record; the re-appended, sealed
   block is then asserted **byte-identical** (per-column sha256 + ``block_digest``)
   to the uninterrupted one.
2. **Sealed digests.** Every per-column sha256 and both ``block_digest`` values
   are compared against the literal goldens below — exact string equality, no
   tolerance.
3. **mmap read-back.** Every mapped column equals the written tensor via
   ``torch.equal`` (same dtype, same inner shape), and unit integrity holds:
   ``unit_frames(u)`` partitions ``range(frame_count)`` exactly, contiguously,
   and ``group_codes[f]`` is the owning unit row.
4. **Relocation (path independence, I1).** The sealed block A is ``shutil.move``\\ d
   to an unrelated directory; ``block_digest`` is unchanged (it covers
   ``{columns, rows, split, units, digests}`` only — never a path or the
   provenance), ``verify()`` is clean, and every column reads identically.
5. **Collate parity.** The same frames are built three ways —
   ``collate_packed(PackedView(reader.frame_payload(frames)), ...)``, the
   ``PackedCache`` v1 payload, and ``collate_molecules`` over per-frame sample
   dicts — and compared leaf-for-leaf (key sets, ``batch_size``, dtype,
   ``torch.equal``) for a fixture carrying bonds **and** angles **and** propers.

    capture: goldens produced by this script itself on the commit below; the
             fixture is closed-form (``arange`` scaled by 0.5 plus a per-unit
             integer offset, hand-written valence chains), so "capture" means
             "ran the deterministic writer once and pinned its sha256 output".
             The block format has no timestamp, no path and no provenance inside
             the digested core, so the bytes are identical on every run and on
             every little-endian host. Third-party imports are ``torch`` and
             ``tensordict`` only — the latter is the container ``collate_packed``
             returns, not an oracle. No oracle is imported, subprocessed or
             consulted, here or at authoring time; no network, no subprocess.
    verify:  PYTHONPATH=src python regressions/streaming-data-tiers-02-block-format.py
    commit:  6d425f7 (+ working tree implementing streaming-data-tiers-02-block-format)
    torch:   2.12.1+cpu
    device:  cpu, float32 / int32 / int16 (exact equality throughout)
    date:    2026-08-14
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from tensordict import TensorDict

from molix.batch.collate import collate_molecules, collate_packed
from molix.store import (
    BlockManifest,
    BlockReader,
    BlockWriter,
    PackedCache,
    PackedView,
    SamplingUnit,
)

# ---------------------------------------------------------------------------
# Hard-coded goldens
# ---------------------------------------------------------------------------

#: Opaque, caller-supplied provenance — fixed literals so the manifest bytes are
#: reproducible. It is deliberately *not* covered by ``block_digest``.
_PROVENANCE: Mapping[str, object] = {
    "source": {"name": "molnex-regression", "coordinate": "n/a"},
    "generator": {
        "tool": "regressions/streaming-data-tiers-02-block-format.py",
        "version": "1",
        "commit": "6d425f7",
    },
    "created_at": "2026-08-14T00:00:00+00:00",
    "units": {
        "frame_pos": "angstrom",
        "frame_target_energy": "kcal/mol",
        "frame_atom_target_forces": "kcal/mol/angstrom",
    },
}

#: Columns frozen from the first unit: 5 block-index + atom_Z + frame_pos +
#: 2 frame targets + 3 valence families (their columns plus a derived ``_n``).
_N_COLUMNS = 24
#: Commit record: unit_index (i8) + n_columns (i4) + C x committed_len (i8) +
#: crc32 (u4) = 16 + 8*C bytes.
_RECORD_SIZE = 16 + 8 * _N_COLUMNS

#: sha256 of every ``<column>.bin`` of sealed block A (split "train"). Identical
#: for the uninterrupted and the crash-resumed write — that is leg 1's assertion.
_BLOCK_A_COLUMN_SHA256: Mapping[str, str] = {
    "angle_atomi": "d05796eb3a889d209cc6e6e6a1429bbd9d218df2b58146a51be1fcafc3164042",
    "angle_atomj": "ad5dc1478de06a4c2728ea528bd9361a4b945e92a414bf4d180cedaaeaa5f4cc",
    "angle_atomk": "79f2a4da15e31b933ef794b6f0a825d72d2bc94c887c03183baa43a1d09fdbff",
    "angle_n": "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f",
    "angle_type": "782486f75e626020735465b875a72f8f468aa5fd32d09f243ca88917f9ce26c1",
    "atom_Z": "41e4a02fbf3c479370753cff3429c47249e7c0501da052e2c089191ad81378ad",
    "bond_atomi": "396506ae7ff3bbf97f7d4f7aa9497f7ac898d2a06412fbe72000424ce96a312c",
    "bond_atomj": "5e60b07a30d0103e3f73d203b3f5d1a56f3d2fb6be3a3b3e0ab4cce1a7820810",
    "bond_n": "0c40fc912bea3d01b4dbad07de4c8cf177ac0c424bc11d622d2239c0e5988986",
    "bond_type": "c8c362c52a33bdd8c9fa09b44caf5284e23fd17e4976103c51b9f2fadf957033",
    "frame_atom_target_forces": "41ea8b1086ce68ba58a300e5d2b00ecd0a3ce3fd08f05b9a924f49678dcb82b0",
    "frame_pos": "5d065d70d3ea9644bab9dcf0165f57c5b873b88dda824cd1ccd814b13fa04a09",
    "frame_target_energy": "ee186fbbdcdc683c0bf77f3f28a9b4703b3f611a24cb44d4b0d5585b377b8a36",
    "mol_n_atoms": "8073c94ef47ecc86dcd78a8d9027a23484fadcd7cea37150319ba8cbf1c70b6b",
    "mol_n_frames": "0c40fc912bea3d01b4dbad07de4c8cf177ac0c424bc11d622d2239c0e5988986",
    "mol_task": "01acecb507abfe1a354aa8064f4af5d3f1acd019e37db3c11c97523b71c76e9d",
    "proper_atomi": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
    "proper_atomj": "67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450",
    "proper_atomk": "26b25d457597a7b0463f9620f666dd10aa2c4373a505967c7c8d70922a2d6ece",
    "proper_atoml": "9d9f290527a6be626a8f5985b26e19b237b44872b03631811df4416fc1713178",
    "proper_n": "01acecb507abfe1a354aa8064f4af5d3f1acd019e37db3c11c97523b71c76e9d",
    "proper_type": "9d9f290527a6be626a8f5985b26e19b237b44872b03631811df4416fc1713178",
    "unit_key_bytes": "89e43d72f79c075a8c4f8f929a416111221e10c73d80614d18dd979aed403b11",
    "unit_key_len": "a23a868b85672de9418b12652e4c1663cb38054c1206157fbf5826bf10f80b2d",
}

#: sha256 over the canonical JSON of ``{columns, rows, split, units, digests}`` —
#: no path, no timestamp, no provenance, which is why relocation cannot move it.
_BLOCK_A_DIGEST = "868910fd174f522ea96bee096bda648e5a3721dba8b45b80a20bdafb49150ce7"

#: sha256 of every ``<column>.bin`` of sealed block B (split "valid").
_BLOCK_B_COLUMN_SHA256: Mapping[str, str] = {
    "angle_atomi": "01acecb507abfe1a354aa8064f4af5d3f1acd019e37db3c11c97523b71c76e9d",
    "angle_atomj": "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f",
    "angle_atomk": "0c40fc912bea3d01b4dbad07de4c8cf177ac0c424bc11d622d2239c0e5988986",
    "angle_n": "26b25d457597a7b0463f9620f666dd10aa2c4373a505967c7c8d70922a2d6ece",
    "angle_type": "41d805e613efbe1acb36eaa0127a35da4a1faa1ac97985d078e50d1fd96055fd",
    "atom_Z": "952e778cb4a51fb98d685f6b6dde89c0e2787f9938eceab7f53944f7fb106fca",
    "bond_atomi": "ad5dc1478de06a4c2728ea528bd9361a4b945e92a414bf4d180cedaaeaa5f4cc",
    "bond_atomj": "4636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d",
    "bond_n": "9d9f290527a6be626a8f5985b26e19b237b44872b03631811df4416fc1713178",
    "bond_type": "11047585fe102fbb5cadb42446612a578d88c6ef5ed076bb7ac360c4f9e4373d",
    "frame_atom_target_forces": "0525c0dbdf6363a08d4f2aac4247891c9405dbeb9c5a567cc351258b4da29329",
    "frame_pos": "35c926608bbc6d350fc1b89e20accbb7356a0c4682ed6a3deb97582397191dbf",
    "frame_target_energy": "648e5ecb219b01d8d51cea6a16190a13f60544f6e34dc5957b0eef31512e311e",
    "mol_n_atoms": "fb5e512425fc9449316ec95969ebe71e2d576dbab833d61e2a5b9330fd70ee02",
    "mol_n_frames": "2594b6a92ebfb1c3312deb7d01c015fb95e9fbe9bd7bc6b527af07813ec7b910",
    "mol_task": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
    "proper_atomi": "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
    "proper_atomj": "67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450",
    "proper_atomk": "26b25d457597a7b0463f9620f666dd10aa2c4373a505967c7c8d70922a2d6ece",
    "proper_atoml": "9d9f290527a6be626a8f5985b26e19b237b44872b03631811df4416fc1713178",
    "proper_n": "67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450",
    "proper_type": "9d9f290527a6be626a8f5985b26e19b237b44872b03631811df4416fc1713178",
    "unit_key_bytes": "5e8a5d96be6b1d641d98a9895d1cf01657b95ea6868d009315344cc9ea11a34b",
    "unit_key_len": "7aa8ca4a02506da9133d8f889678b76f716ce45d02e22fdb7b70a15e56a0eff8",
}

_BLOCK_B_DIGEST = "17c809a7d1f7ee9d791b26575d5fc7de98222c603c1e1d43d9c2fe46ab6b93af"

# Derived, never stored — asserted against the spec's hand-computed literals.
_MOL_ATOM_PTR = [0, 3, 7]
_MOL_FRAME_PTR = [0, 2, 5]
_MOL_FRAME_ATOM_PTR = [0, 6, 18]
_FRAME_MOL = [0, 0, 1, 1, 1]
_ATOM_COUNTS = [3, 3, 4, 4, 4]
#: Pointer vectors that must never appear as a column file.
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

#: Parity batch: frames (3, 0, 2) of block A -> units (1, 0, 1).
_PARITY_FRAMES = [3, 0, 2]
#: Hand-computed bucket sizes: atoms 4+3+4, angles 2+1+2, propers 1+0+1, and
#: 0 edges because a block stores no neighbour graph by design.
_PARITY_BATCH_SIZES: Mapping[str, tuple[int, ...]] = {
    "atoms": (11,),
    "edges": (0,),
    "graphs": (3,),
    "bonds": (),
    "angles": (5,),
    "propers": (2,),
}


# ---------------------------------------------------------------------------
# Deterministic fixture (no RNG — closed-form integers and halves)
# ---------------------------------------------------------------------------


def _halves(count: int, start: int) -> torch.Tensor:
    """``count`` consecutive multiples of 0.5 — exactly representable in float32.

    Args:
        count: Number of values.
        start: Integer offset applied before scaling, so each unit gets its own
            distinguishable value range.

    Returns:
        ``(count,)`` float32 tensor ``[start, start+1, ...] * 0.5``.
    """
    return (torch.arange(count, dtype=torch.float32) + float(start)) * 0.5


def _i32(values: list[int]) -> torch.Tensor:
    """Return *values* as an ``int32`` column (empty lists keep shape ``(0,)``)."""
    return torch.tensor(values, dtype=torch.int32)


def _valence(n_atoms: int) -> dict[str, torch.Tensor]:
    """Covalent topology of a linear chain, with molecule-local atom indices.

    Args:
        n_atoms: 3 (bent triatomic: 2 bonds, 1 angle, no proper) or 4 (chain:
            3 bonds, 2 angles, 1 proper).

    Returns:
        The 12 valence columns of the three declared families, all ``int32``
        and all indexing atoms in ``[0, n_atoms)``.
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


def _unit(key: str, task: str, z: list[int], n_frames: int, offset: int) -> SamplingUnit:
    """Build one sampling unit: one topology plus *n_frames* frames of it.

    Args:
        key: Unit identity, stored UTF-8 in ``unit_key_bytes``.
        task: Task label, stored as a dictionary code in ``mol_task``.
        z: Atomic numbers, ``(N,)``.
        n_frames: Frames of this topology.
        offset: Integer offset feeding :func:`_halves`, distinct per unit.

    Returns:
        The validated :class:`SamplingUnit`.
    """
    n_atoms = len(z)
    rows = n_frames * n_atoms
    return SamplingUnit(
        key=key,
        task=task,
        topology={"atom_Z": torch.tensor(z, dtype=torch.int16), **_valence(n_atoms)},
        frames={
            "frame_pos": _halves(rows * 3, offset).reshape(rows, 3),
            "frame_atom_target_forces": -_halves(rows * 3, offset + 7).reshape(rows, 3),
            "frame_target_energy": _halves(n_frames, offset + 3),
        },
    )


def _block_a_units() -> list[SamplingUnit]:
    """The two units of block A — the spec's ``N=(3,4)``, ``n_frames=(2,3)`` fixture."""
    return [
        _unit("mol-a", "equilibrium", [8, 1, 1], 2, 0),
        _unit("mol-b", "relax", [6, 6, 1, 1], 3, 40),
    ]


def _block_b_units() -> list[SamplingUnit]:
    """The single 5-frame torsion scan of block B."""
    return [_unit("scan-c", "torsion", [6, 6, 8, 1], 5, 100)]


def _samples(units: Sequence[SamplingUnit]) -> list[dict[str, object]]:
    """Re-express the same data as flat per-frame sample dicts (the v1 shape).

    Topology is repeated per frame — exactly the duplication the two-level block
    layout removes — so the packed-cache v1 oracle sees the identical numbers.

    Args:
        units: Units in block order.

    Returns:
        One flat dict per frame, in frame order.
    """
    out: list[dict[str, object]] = []
    for unit in units:
        n_atoms = unit.n_atoms
        top = unit.topology
        for k in range(unit.n_frames):
            rows = slice(k * n_atoms, (k + 1) * n_atoms)
            out.append(
                {
                    "Z": top["atom_Z"].long(),
                    "pos": unit.frames["frame_pos"][rows].clone(),
                    "targets": {
                        "energy": unit.frames["frame_target_energy"][k : k + 1].clone(),
                        "forces": unit.frames["frame_atom_target_forces"][rows].clone(),
                    },
                    "bond_index": torch.stack([top["bond_atomi"], top["bond_atomj"]]).long(),
                    "bond_types": top["bond_type"].long(),
                    "angles": {
                        "atomi": top["angle_atomi"].long(),
                        "atomj": top["angle_atomj"].long(),
                        "atomk": top["angle_atomk"].long(),
                        "type": top["angle_type"].long(),
                    },
                    "propers": {
                        "atomi": top["proper_atomi"].long(),
                        "atomj": top["proper_atomj"].long(),
                        "atomk": top["proper_atomk"].long(),
                        "atoml": top["proper_atoml"].long(),
                        "type": top["proper_type"].long(),
                    },
                }
            )
    return out


def _expected_columns(units: Sequence[SamplingUnit]) -> dict[str, torch.Tensor]:
    """Concatenate the fixture's own tensors into the columns a block must hold."""
    parts: dict[str, list[torch.Tensor]] = {}
    for unit in units:
        for name, tensor in (*unit.topology.items(), *unit.frames.items()):
            parts.setdefault(name, []).append(tensor)
    return {name: torch.cat(chunk) for name, chunk in parts.items()}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _column_sha256(root: Path) -> dict[str, str]:
    """Return ``{column name: sha256 hex}`` over the actual ``<name>.bin`` bytes."""
    return {
        path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.glob("*.bin"))
    }


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
        if isinstance(left, TensorDict) or isinstance(right, TensorDict):
            _require_td_equal(left, right, f"{path}/{key}")
            continue
        _require_equal_tensors(left, right, f"leaf {path}/{key}")


# ---------------------------------------------------------------------------
# Leg 1 — streaming append, torn-tail resume, seal
# ---------------------------------------------------------------------------


def _write_block(root: Path, units: Sequence[SamplingUnit], *, split: str) -> Path:
    """Write and seal a block without interruption — the reference write."""
    writer = BlockWriter(root, split=split, provenance=_PROVENANCE)
    for unit in units:
        writer.append_unit(unit)
    writer.seal()
    return root


def _write_block_through_a_crash(root: Path, units: Sequence[SamplingUnit], *, split: str) -> Path:
    """Write the same units, but crash mid-commit and recover with ``resume``.

    Every unit's column bytes are appended and fsynced before its commit record,
    so the crash is simulated the only way that can actually happen on disk: the
    last commit record is cut **in half** and 24 junk bytes are appended to a
    column, leaving the last unit's payload durable but *uncommitted*.
    ``BlockWriter.resume`` must discard both and continue at the last
    crc32-valid record.

    Args:
        root: Block directory.
        units: Units to append, in order. At least two.
        split: The block's single split.

    Returns:
        *root*, sealed.
    """
    writer = BlockWriter(root, split=split, provenance=_PROVENANCE)
    for unit in units:
        writer.append_unit(unit)
    del writer  # the crash: no seal, no handles, nothing flushed beyond the log

    declared = len(BlockManifest.read(root).columns)
    _require(declared == _N_COLUMNS, f"block declares {declared} columns, expected {_N_COLUMNS}")
    log = root / "commit.log"
    size = log.stat().st_size
    _require(
        size == _RECORD_SIZE * len(units),
        f"commit.log is {size} B, expected {_RECORD_SIZE} B x {len(units)} committed units",
    )
    os.truncate(log, size - _RECORD_SIZE // 2)  # tear the last record in half
    with open(root / "frame_pos.bin", "ab") as handle:  # torn column tail
        handle.write(b"\x7f" * 24)

    resumed = BlockWriter.resume(root)
    _require(
        resumed.unit_count == len(units) - 1,
        f"resume kept {resumed.unit_count} units, expected {len(units) - 1} "
        "(the half-written record must be discarded)",
    )
    index = resumed.append_unit(units[-1])
    _require(
        index == len(units) - 1,
        f"resume re-appended at index {index}, expected {len(units) - 1}",
    )
    resumed.seal()
    return root


def _check_append_and_resume(workdir: Path) -> Path:
    """Leg 1: a resumed block seals byte-identically to an uninterrupted one.

    Args:
        workdir: Scratch directory.

    Returns:
        The sealed, resumed block A directory (the one every later leg uses —
        deliberately, so the recovered bytes are what gets read, relocated and
        collated).
    """
    units = _block_a_units()
    clean = _write_block(workdir / "block-a-clean", units, split="train")
    resumed = _write_block_through_a_crash(workdir / "block-a", units, split="train")

    clean_digests, resumed_digests = _column_sha256(clean), _column_sha256(resumed)
    _require(
        clean_digests == resumed_digests,
        "resumed block is not byte-identical to the uninterrupted write: "
        f"{sorted(k for k in clean_digests if clean_digests[k] != resumed_digests.get(k))}",
    )
    clean_manifest = BlockManifest.read(clean)
    resumed_manifest = BlockManifest.read(resumed)
    _require(
        clean_manifest.block_digest == resumed_manifest.block_digest,
        f"block_digest drift after resume: {resumed_manifest.block_digest} "
        f"!= {clean_manifest.block_digest}",
    )
    return resumed


# ---------------------------------------------------------------------------
# Leg 2 — sealed digests against the literal goldens
# ---------------------------------------------------------------------------


def _check_digests(root: Path, columns: Mapping[str, str], block_digest: str, label: str) -> None:
    """Leg 2: per-column sha256 and ``block_digest`` equal the pinned literals."""
    actual = _column_sha256(root)
    _require(
        set(actual) == set(columns),
        f"{label}: column file set drift — "
        f"extra {sorted(set(actual) - set(columns))}, missing {sorted(set(columns) - set(actual))}",
    )
    drifted = sorted(name for name in actual if actual[name] != columns[name])
    _require(
        not drifted,
        f"{label}: sha256 drift on {drifted} — "
        + ", ".join(f"{name}: {actual[name]} != {columns[name]}" for name in drifted),
    )
    manifest = BlockManifest.read(root)
    _require(
        manifest.sealed,
        f"{label}: manifest is not sealed — a sealed manifest is the only witness of READY",
    )
    _require(
        dict(manifest.digests) == dict(columns),
        f"{label}: the manifest's own digests disagree with the column files",
    )
    _require(
        manifest.block_digest == block_digest,
        f"{label}: block_digest drift {manifest.block_digest} != {block_digest}",
    )


# ---------------------------------------------------------------------------
# Leg 3 — mmap read-back and unit integrity
# ---------------------------------------------------------------------------


def _check_read_back(reader: BlockReader, units: Sequence[SamplingUnit], label: str) -> None:
    """Leg 3: mapped columns equal the written tensors, and units stay whole."""
    for name, expected in _expected_columns(units).items():
        _require_equal_tensors(reader.column(name), expected, f"{label} column {name!r}")

    _require_equal_tensors(
        reader.column("mol_n_atoms"),
        torch.tensor([unit.n_atoms for unit in units], dtype=torch.int32),
        f"{label} column 'mol_n_atoms'",
    )
    _require_equal_tensors(
        reader.column("mol_n_frames"),
        torch.tensor([unit.n_frames for unit in units], dtype=torch.int32),
        f"{label} column 'mol_n_frames'",
    )

    _require(
        reader.unit_count == len(units),
        f"{label}: unit_count {reader.unit_count} != {len(units)}",
    )
    _require(
        reader.frame_count == sum(unit.n_frames for unit in units),
        f"{label}: frame_count {reader.frame_count} != {sum(u.n_frames for u in units)}",
    )

    seen: list[int] = []
    group_codes = [int(g) for g in reader.group_codes.tolist()]
    for row, unit in enumerate(units):
        frames = [int(f) for f in reader.unit_frames(row).tolist()]
        _require(
            len(frames) == unit.n_frames,
            f"{label}: unit {row} exposes {len(frames)} frames, appended {unit.n_frames}",
        )
        _require(
            frames == list(range(frames[0], frames[0] + len(frames))),
            f"{label}: unit {row} frames {frames} are not contiguous — a unit must stay whole",
        )
        _require(
            all(group_codes[f] == row for f in frames),
            f"{label}: group_codes disagree with unit {row}; one unit is one group",
        )
        _require(
            reader.unit_key(row) == unit.key and reader.unit_index(unit.key) == row,
            f"{label}: unit key round-trip failed for row {row}",
        )
        seen.extend(frames)
    _require(
        seen == list(range(reader.frame_count)),
        f"{label}: unit_frames do not partition range({reader.frame_count}): {seen}",
    )

    labels = [reader.task_names[code] for code in reader.task_codes.tolist()]
    expected_labels = [unit.task for unit in units for _ in range(unit.n_frames)]
    _require(
        labels == expected_labels,
        f"{label}: decoded task labels {labels} != appended {expected_labels}",
    )


def _check_derived_pointers(reader: BlockReader, root: Path) -> None:
    """Leg 3 (block A only): the two-level arithmetic equals the spec's literals."""
    for name, actual, expected in (
        ("mol_atom_ptr", reader.mol_atom_ptr.tolist(), _MOL_ATOM_PTR),
        ("mol_frame_ptr", reader.mol_frame_ptr.tolist(), _MOL_FRAME_PTR),
        ("mol_frame_atom_ptr", reader.mol_frame_atom_ptr.tolist(), _MOL_FRAME_ATOM_PTR),
        ("frame_mol", reader.frame_mol.tolist(), _FRAME_MOL),
        ("atom_counts", reader.atom_counts.tolist(), _ATOM_COUNTS),
    ):
        _require(actual == expected, f"block A {name} = {actual}, hand-computed {expected}")

    # Frame 3 is unit 1, k = 3 - 2 = 1 -> frame-atom rows 6 + 1*4 = 10 .. 14 and
    # topology rows mol_atom_ptr[1] + [0, 4) = 3 .. 7.
    payload = reader.frame_payload([3])
    _require_equal_tensors(
        payload["atoms"]["pos"], reader.column("frame_pos")[10:14], "frame 3 frame-atom rows 10..14"
    )
    _require_equal_tensors(
        payload["atoms"]["Z"], reader.column("atom_Z")[3:7], "frame 3 topology rows 3..7"
    )

    declared = BlockManifest.read(root).columns
    for name in _POINTER_NAMES:
        _require(
            name not in declared and not (root / f"{name}.bin").exists(),
            f"{name} is stored on disk — every pointer is derived at open, never stored",
        )

    try:
        _ = reader.edge_counts
    except ValueError as exc:
        _require(
            "NeighborList" in str(exc),
            f"edge_counts raised without pointing at NeighborList: {exc}",
        )
    else:
        raise AssertionError("edge_counts returned a value — blocks carry no edge columns")


# ---------------------------------------------------------------------------
# Leg 4 — relocation
# ---------------------------------------------------------------------------


def _check_relocation(root: Path, destination: Path) -> Path:
    """Leg 4: move a sealed block; digest, verification and reads are unchanged.

    Args:
        root: Sealed block directory (consumed — it is moved away).
        destination: New path, on the same filesystem or not.

    Returns:
        The new block directory.
    """
    before = BlockReader(root)
    before.verify()
    snapshot = {name: before.column(name).clone() for name in BlockManifest.read(root).columns}
    digest_before = BlockManifest.read(root).block_digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root), str(destination))
    _require(not root.exists(), f"{root} survived the move")

    after = BlockReader(destination)
    after.verify()
    _require(
        BlockManifest.read(destination).block_digest == digest_before,
        "block_digest changed on relocation — it must cover no path and no provenance",
    )
    for name, expected in snapshot.items():
        _require_equal_tensors(after.column(name), expected, f"relocated column {name!r}")
    return destination


# ---------------------------------------------------------------------------
# Leg 5 — collate parity
# ---------------------------------------------------------------------------


def _check_collate_parity(
    reader: BlockReader,
    units: Sequence[SamplingUnit],
    cache_path: Path,
) -> None:
    """Leg 5: ``frame_payload`` -> ``collate_packed`` equals both v1 oracles.

    Args:
        reader: Sealed block A.
        units: The units that block holds, for the per-frame sample dicts.
        cache_path: Where to write the packed-cache v1 oracle.
    """
    samples = _samples(units)
    _require(
        len(samples) == reader.frame_count,
        f"oracle built {len(samples)} samples for {reader.frame_count} block frames",
    )

    actual = collate_packed(
        PackedView(reader.frame_payload(_PARITY_FRAMES)), range(len(_PARITY_FRAMES))
    )

    cache = PackedCache(cache_path)
    cache.save(samples)
    packed_v1 = collate_packed(PackedView(cache.load(mmap=False)), _PARITY_FRAMES)
    _require_td_equal(actual, packed_v1)

    from_dicts = collate_molecules([samples[f] for f in _PARITY_FRAMES])
    _require_td_equal(actual, from_dicts)

    _require(
        tuple(actual.batch_size) == (),
        f"top-level batch_size {tuple(actual.batch_size)} != ()",
    )
    for namespace, expected in _PARITY_BATCH_SIZES.items():
        got = tuple(actual[namespace].batch_size)
        _require(got == expected, f"{namespace}.batch_size {got} != hand-computed {expected}")


# ---------------------------------------------------------------------------
# Fixture self-check — the goldens must mean what the header says
# ---------------------------------------------------------------------------


def _cumsum(counts: list[int]) -> list[int]:
    """Exclusive cumsum with the total appended — the format's pointer rule, by hand."""
    out = [0]
    for count in counts:
        out.append(out[-1] + count)
    return out


def _check_fixture_conventions() -> None:
    """Pin the fixture and the literals to their meaning before they are an oracle.

    A digest golden compares bytes against bytes, so it cannot catch a fixture
    that quietly stopped being exact-in-float32 or started carrying a global atom
    index; and a pointer literal typed by hand could silently redefine what it
    claims to encode. Both are re-derived in closed form here, from the units
    alone.
    """
    for block_units in (_block_a_units(), _block_b_units()):
        for unit in block_units:
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

    units = _block_a_units()
    n_atoms = [unit.n_atoms for unit in units]
    n_frames = [unit.n_frames for unit in units]
    frame_mol = [m for m, count in enumerate(n_frames) for _ in range(count)]
    for name, literal, derived in (
        ("_MOL_ATOM_PTR", _MOL_ATOM_PTR, _cumsum(n_atoms)),
        ("_MOL_FRAME_PTR", _MOL_FRAME_PTR, _cumsum(n_frames)),
        (
            "_MOL_FRAME_ATOM_PTR",
            _MOL_FRAME_ATOM_PTR,
            _cumsum([f * n for f, n in zip(n_frames, n_atoms)]),
        ),
        ("_FRAME_MOL", _FRAME_MOL, frame_mol),
        ("_ATOM_COUNTS", _ATOM_COUNTS, [n_atoms[m] for m in frame_mol]),
    ):
        _require(
            literal == derived,
            f"{name} = {literal} is not what block A's units imply ({derived}) — "
            "a typo in the literal must not redefine the golden it is compared against",
        )

    _require(
        sum(_PARITY_BATCH_SIZES["atoms"]) == sum(_ATOM_COUNTS[f] for f in _PARITY_FRAMES),
        "the parity atom count literal disagrees with atom_counts on the selected frames",
    )


# ---------------------------------------------------------------------------


def main() -> int:
    _check_fixture_conventions()
    units_a, units_b = _block_a_units(), _block_b_units()

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        block_a = _check_append_and_resume(workdir)
        block_b = _write_block(workdir / "block-b", units_b, split="valid")

        _check_digests(block_a, _BLOCK_A_COLUMN_SHA256, _BLOCK_A_DIGEST, "block A")
        _check_digests(block_b, _BLOCK_B_COLUMN_SHA256, _BLOCK_B_DIGEST, "block B")

        for root, split in ((block_a, "train"), (block_b, "valid")):
            recorded = BlockManifest.read(root).split
            _require(recorded == split, f"{root.name}: manifest split {recorded!r} != {split!r}")

        reader_a, reader_b = BlockReader(block_a), BlockReader(block_b)
        _check_read_back(reader_a, units_a, "block A")
        _check_read_back(reader_b, units_b, "block B")
        _check_derived_pointers(reader_a, block_a)

        _check_collate_parity(reader_a, units_a, workdir / "oracle-v1.pt")

        relocated = _check_relocation(block_a, workdir / "elsewhere" / "moved-block-a")
        _check_digests(relocated, _BLOCK_A_COLUMN_SHA256, _BLOCK_A_DIGEST, "relocated block A")
        _check_read_back(BlockReader(relocated), units_a, "relocated block A")

    print(
        "streaming-data-tiers-02-block-format: 2 unit-aligned blocks "
        "(A: units mol-a/mol-b, N=(3,4), n_frames=(2,3), split 'train'; "
        "B: 5-frame scan-c, split 'valid') append -> torn-tail resume -> seal "
        "byte-identically; 24 column sha256 + block_digest per block match the "
        f"pinned goldens (A {_BLOCK_A_DIGEST[:12]}..., B {_BLOCK_B_DIGEST[:12]}...); "
        "derived pointers = (0,3,7)/(0,2,5)/(0,6,18), atom_counts=(3,3,4,4,4), no "
        "pointer stored; unit_frames partition 5 frames with group_codes = unit row; "
        "relocation leaves block_digest and every column identical with verify() clean; "
        "frame_payload -> collate_packed equals the PackedCache v1 and collate_molecules "
        "oracles leaf-for-leaf (atoms 11, angles 5, propers 2, edges 0) OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
