"""READY-tier block store — streaming-appendable raw column files.

A **block** is one self-contained directory holding 1–5 GB of unit-aligned
training data as raw little-endian column files plus a ``manifest.json`` and a
small ``commit.log``. It is the READY tier of the streaming-data-tiers strategy:
a converter writes it incrementally, one :class:`SamplingUnit` at a time; a
consumer mmaps it and builds batches with the existing packed collate.

Vocabulary guard — "block" here is a **storage block**. It is neither
``molpy.Block`` (a chemical block: a residue-like group of atoms) nor a molrep
``nn`` block (an interaction layer). No bare ``Block`` symbol is exported from
this module, so no import can collide with either.

On-disk layout
--------------

::

    <block-root>/
      manifest.json   # schema, split, provenance, task dictionary, digests, sealed
      commit.log      # fixed-size crc32 commit records (append-only)
      mol_n_atoms.bin mol_n_frames.bin mol_task.bin unit_key_len.bin unit_key_bytes.bin
      atom_Z.bin frame_pos.bin frame_target_energy.bin frame_atom_target_forces.bin
      bond_n.bin bond_atomi.bin bond_atomj.bin bond_type.bin  ...

Column files are raw payload only — no header, no padding, no holes. Rows grow
along dim 0; the inner shape (e.g. ``(3,)`` for ``frame_pos``) is declared once
in the manifest and never varies. A COO ``(2, N)`` layout is forbidden on disk
because growing its last axis is not an append, so bonds are stored as two 1-D
columns (``bond_atomi`` / ``bond_atomj``) and ``bond_index`` is stacked only at
payload-assembly time.

Storage is **two-level ragged**: molecular topology (atomic numbers, valence
columns, molecule-local atom indices) is stored **once per unit**, while
positions / energies / forces are stored per frame. For unit ``m`` with
``N_m = mol_n_atoms[m]`` atoms and ``n_m = mol_n_frames[m]`` frames::

    mol_atom_ptr[m]       = sum_{m'<m} N_m'
    mol_frame_ptr[m]      = sum_{m'<m} n_m'
    mol_frame_atom_ptr[m] = sum_{m'<m} n_m' * N_m'
    frame_mol             = repeat_interleave(arange(M), mol_n_frames)

and frame ``f`` of unit ``m`` with local ordinal ``k = f - mol_frame_ptr[m]``
occupies frame-atom rows ``mol_frame_atom_ptr[m] + k*N_m + [0, N_m)`` and
topology rows ``mol_atom_ptr[m] + [0, N_m)``. ``N_m`` is structurally constant
inside a unit (a unit is one topology plus its frames), which is what makes the
``k*N_m`` stride valid and removes any need for a per-frame atom pointer column.

Derived, never stored
---------------------

``mol_atom_ptr``, ``mol_frame_ptr``, ``mol_frame_atom_ptr``, every ``<family>_ptr``,
``frame_mol`` and ``unit_key_ptr`` are computed by cumsum / ``repeat_interleave``
at open (O(M + F) ints, milliseconds) from the stored per-unit **counts**
(``mol_n_atoms``, ``mol_n_frames``, ``<family>_n``, ``unit_key_len``). One source
of truth per fact removes the pointer/payload-disagreement class of corruption
entirely. ``block_digest`` follows the same rule: it is derived from the
canonical JSON of the manifest core rather than trusted as stored text.

Append / commit protocol
------------------------

:meth:`BlockWriter.append_unit` performs, in this order:

1. append every column's rows with ``O_APPEND`` writes (no seeks, no rewrite of
   any predecessor byte);
2. ``fsync`` every touched column file;
3. append one fixed-size commit record to ``commit.log`` and ``fsync`` it.

The record is ``unit_index (i8) + n_columns (i4) + n_columns * committed_len (i8)
+ crc32 (u4)`` = ``16 + 8*C`` bytes, with the lengths listed in **sorted column
name** order and the crc32 taken over all preceding bytes of the record. The
record is the single commit point: the ordering above means a record can only
exist once the bytes it counts are durable, so recovery
(:meth:`BlockWriter.resume`) scans the log for the last crc-valid record,
``os.truncate``\\ s every column to its recorded length, truncates the log past
that record, and continues at ``unit_index + 1``. Uncommitted tails — the bytes
a crash left behind — are therefore discarded, and a resumed block seals to
byte-identical columns. ``manifest.json`` is rewritten atomically (tmp → fsync →
``os.replace``) only at schema freeze, at the first appearance of a new task
label, and at :meth:`BlockWriter.seal` — never per unit.

The writer holds **no long-lived file descriptors**: every append opens, writes,
fsyncs and closes. Consequently :meth:`BlockWriter.resume` invalidates any
earlier writer instance for the same root (its cached lengths and unit count are
stale after truncation); drop the old instance instead of interleaving the two.

Why no 64-byte alignment field
------------------------------

The npy-append-array precedent pads its header to 64 bytes because payload
follows metadata *in the same file*. Here every column is its own file mapped at
**offset 0**, so each column's row 0 sits on a page boundary by construction;
intra-column rows (e.g. 12-byte ``frame_pos`` rows) need no alignment because
mmap correctness and zero-copy depend only on the mapping base pointer. No
padding, no header, no alignment field — and no reason to add one later.

Why no edge / neighbor-list columns
-----------------------------------

A 40-atom molecule is smaller than a 5 Å cutoff sphere pairwise, so ``G(r_c)``
is nearly complete: ``40*39 = 1560`` ordered pairs ≈ 1.2e3 directed edges per
frame. Storing only ``edge_index`` (2 × int32) costs ≈ 9.6 kB/frame ≈ **10×**
the ≈ 1.0 kB/frame energy+force payload; adding ``edge_diff`` + ``edge_dist``
costs ≈ 28.8 kB/frame ≈ **29×**. It would also pin the cutoff and the edge
symmetry mode into the artifact. Blocks therefore store geometry + topology +
targets only, and the neighbour graph is rebuilt downstream
(:class:`molix.batch.NeighborList` or the MD bind path);
:attr:`BlockReader.edge_counts` raises rather than inventing a count.

Zero-copy reads
---------------

Columns are mapped through the single :meth:`BlockReader._map_column` seam with
``torch.UntypedStorage.from_file(path, shared=False, nbytes=...)`` — MAP_PRIVATE
at offset 0, which is why one file per column was chosen over one packed file
with per-column offsets (``UntypedStorage.from_file`` has no byte-offset
parameter). Measured on torch 2.12.1 (2026-08-14): a ``shared=False`` mapping
leaves ``untyped_storage().filename`` as ``None`` and ``shared=True`` cannot open
a sealed ``0o444`` column read-write, so the witness that actually holds for this
mapping is **page-cache coherence** — a byte written to the file through a second
descriptor is observed through the mapped tensor, which is false for any copy.
The spec's documented contingency (``torch.from_numpy(np.memmap(path, mode="r"))``)
fits behind the same seam as a one-line change. Mapped tensors are read-only by
contract: writing to one dirties a private copy that no other reader sees.

Mapping is **lazy per column**, so a batch plan can be built from the count
columns alone without faulting in a single frame.

Measured on Lustre, 2026-08-14
------------------------------

``scripts/store/bench_block_read.py`` on ``arrhenius1.hpc.arrhenius.naiss.se``
(login node; project Lustre ``/nobackup/proj/disk``, PFL 1 stripe x 4 MiB;
torch 2.12.1+cpu, ``OMP_NUM_THREADS=1``), median of 3 runs over a 1.03 GB block
(256 units x 8192 frames x 40 atoms, 17 columns). One pass gathers 65536 frames
= 31.5 MB of useful ``frame_pos`` payload in 256-frame batches; ``sequential``
is frames ``[0, 65536)``, ``shuffled`` is a seeded permutation over all 2.1 M
frames, so both orders move identical payload bytes. ``cold`` is the first touch
after ``POSIX_FADV_DONTNEED`` (``mincore`` witnessed 0.000 page residency),
``warm`` the immediately repeated pass. MB/s counts useful payload only, so the
shuffled rows also pay for pulling a whole 4 kB page per 480 B frame::

    order       path    cold MB/s   warm MB/s
    sequential  mmap          104        1114
    sequential  pread         128         218
    shuffled    mmap          2.2         602
    shuffled    pread         1.6         178

Write side, same client: 1000 small units (12 atoms, 4 frames -> 18 fsyncs per
unit: 17 columns + the commit log) commit at **153 ms/unit = 6.5 units/s**, and
the ~1 MB-per-unit read block at 185 ms/unit = 21.8 MB/s. Both are fsync-latency
bound, not bandwidth bound, so unit size — not column count — is the knob.
:meth:`BlockWriter.seal` streams sha256 at **1.4 GB/s**, i.e. a 2 GB block seals
in ~1.4 s (the spec's 1 GB/s assumption was conservative).

**Verdict: keep the mmap seam.** Warm, the mmap gather wins by **5.1x**
sequential and **3.4x** shuffled — the regime a training epoch runs in once a
block is resident. Cold, the two paths are a wash: ``pread``'s best reproducible
edge is 1.23x (sequential; a 2.77x outlier did not reproduce), under the spec's
1.5x threshold, so no ``pread``-based column reader is warranted behind
:meth:`BlockReader._map_column`. The cliff is locality, not path — cold shuffled
frame access runs at ~2 MB/s (~3-5 k frames/s, latency-bound) on **both** paths,
~270x below warm mmap — so a consumer must pre-warm a block sequentially instead
of relying on random faults; that is a conveyor concern, not a reader one. Cold
cells vary up to ~3x run-to-run on a shared login node and should be re-measured
on the compute node before prefetch policy is frozen.

Layering: this module imports stdlib + torch only. It must not import
``tensordict`` or any collate symbol — batch construction is the caller's job,
which composes as ``collate_packed(PackedView(reader.frame_payload(frames)), ...)``.

References:
    Fey & Lenssen. "Fast Graph Representation Learning with PyTorch Geometric."
    arXiv:1903.02428 — concatenated per-key tensors + slice index.
    Apache Arrow columnar format, variable-size list layout (offsets + values).
    https://arrow.apache.org/docs/format/Columnar.html
    MosaicML Streaming (MDS) shard layout — sample count + offsets + raw bytes.
    https://github.com/mosaicml/streaming
    npy-append-array — append + recovery of an interrupted array file.
    https://github.com/xor2k/npy-append-array
    Wang, Fass & Chodera. "End-to-end differentiable construction of molecular
    mechanics force fields." Chem. Sci. 13, 12016 (2022). DOI:10.1039/D2SC02739A
    — molecule-level grouping before configuration sampling, the discipline the
    unit-aligned append protocol preserves.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NamedTuple

import torch

__all__ = ["BlockManifest", "BlockReader", "BlockWriter", "SamplingUnit"]

_FORMAT: Final[str] = "molix.block/1"
_SCHEMA_VERSION: Final[int] = 1
_MANIFEST_NAME: Final[str] = "manifest.json"
_COMMIT_LOG_NAME: Final[str] = "commit.log"
_HASH_CHUNK: Final[int] = 1 << 20  # 1 MiB streaming sha256 chunk

# Little-endian dtype strings, so a column's bytes — and therefore its sha256 —
# are host-independent. Big-endian hosts are rejected, never byteswapped.
_DTYPE_STRINGS: Final[dict[torch.dtype, str]] = {
    torch.float32: "<f4",
    torch.float64: "<f8",
    torch.int16: "<i2",
    torch.int32: "<i4",
    torch.int64: "<i8",
    torch.int8: "|i1",
    torch.uint8: "|u1",
}
_TORCH_DTYPES: Final[dict[str, torch.dtype]] = {v: k for k, v in _DTYPE_STRINGS.items()}

# Valence term families and the collate bucket each one feeds.
_FAMILIES: Final[tuple[str, ...]] = ("bond", "angle", "proper", "improper", "internal")
_FAMILY_BUCKETS: Final[dict[str, str]] = {
    "bond": "bonds",
    "angle": "angles",
    "proper": "propers",
    "improper": "impropers",
    "internal": "internals",
}
_ATOM_INDEX_COLUMNS: Final[tuple[str, ...]] = ("atomi", "atomj", "atomk", "atoml")

# Block-index columns synthesised by the writer for every block. They carry the
# per-unit counts every derived pointer is built from, plus the unit key
# dictionary; none of them is routed into a batch payload.
_META_ROUTE_PREFIX: Final[str] = "meta."
_META_COLUMNS: Final[dict[str, tuple[str, str]]] = {
    "mol_n_atoms": ("<i4", "molecule"),
    "mol_n_frames": ("<i4", "molecule"),
    "mol_task": ("<i4", "molecule"),
    "unit_key_len": ("<i4", "molecule"),
    "unit_key_bytes": ("|u1", "bytes"),
}

# Columns every block must declare; everything else is optional and declared by
# the first appended unit.
_REQUIRED_TOPOLOGY: Final[str] = "atom_Z"
_REQUIRED_FRAME: Final[str] = "frame_pos"

# Provenance is opaque, caller-supplied JSON (no dataset coordinate is ever
# hard-coded in molnex); only key presence is validated.
_PROVENANCE_KEYS: Final[tuple[str, ...]] = ("source", "generator", "created_at", "units")

# The location-independent core of the manifest that ``block_digest`` covers.
_DIGEST_FIELDS: Final[tuple[str, ...]] = ("columns", "rows", "split", "units", "digests")


class _Route(NamedTuple):
    """Where a declared column lives on disk and where it lands in a payload.

    Attributes:
        space: One of ``molecule``, ``molecule_atom``, ``molecule_term:<family>``,
            ``frame``, ``frame_atom``, ``bytes`` — the row space whose length the
            column tracks.
        target: Dotted payload route (e.g. ``atoms.pos``), or ``meta.<name>`` for
            a block-index column that no payload consumes.
    """

    space: str
    target: str


def _column_route(name: str) -> _Route:
    """Resolve a column name to its row space and payload route.

    Naming determines the route, table-driven the way
    :data:`molix.batch.collate.INDEX_KEYS` is declarative rather than hard-coded
    per call site, so an unroutable column can never enter a block and
    :meth:`BlockReader.frame_payload` can never silently drop data.

    Args:
        name: Declared column name, e.g. ``frame_atom_target_forces``.

    Returns:
        The :class:`_Route` for *name*.

    Raises:
        ValueError: *name* matches no route rule of ``molix.block/1``.
    """
    if name in _META_COLUMNS:
        return _Route(_META_COLUMNS[name][1], f"{_META_ROUTE_PREFIX}{name}")
    for family in _FAMILIES:
        if name == f"{family}_n":
            return _Route("molecule", f"{_META_ROUTE_PREFIX}{name}")
        bucket = _FAMILY_BUCKETS[family]
        for column in _ATOM_INDEX_COLUMNS:
            if name == f"{family}_{column}":
                target = "bonds.bond_index" if family == "bond" else f"{bucket}.{column}"
                return _Route(f"molecule_term:{family}", target)
        if name == f"{family}_type":
            target = "bonds.bond_types" if family == "bond" else f"{bucket}.type"
            return _Route(f"molecule_term:{family}", target)
    if name == _REQUIRED_FRAME:
        return _Route("frame_atom", "atoms.pos")
    if name.startswith("frame_atom_target_"):
        return _Route("frame_atom", f"atoms.targets.{name[len('frame_atom_target_') :]}")
    if name.startswith("frame_target_"):
        return _Route("frame", f"graphs.targets.{name[len('frame_target_') :]}")
    if name.startswith("mol_target_"):
        return _Route("molecule", f"graphs.targets.{name[len('mol_target_') :]}")
    if name in ("atom_Z", "atom_mass", "atom_type"):
        return _Route("molecule_atom", f"atoms.{name[len('atom_') :]}")
    raise ValueError(
        f"column {name!r} matches no route rule of {_FORMAT} — an unroutable column "
        "cannot enter a block because frame_payload would silently drop it. Use a "
        "routed name: atom_Z / atom_mass / atom_type, frame_pos, frame_target_<name>, "
        "frame_atom_target_<name>, mol_target_<name>, <family>_atom{i,j,k,l} or "
        f"<family>_type with <family> in {_FAMILIES}."
    )


def _require_little_endian() -> None:
    """Reject big-endian hosts instead of silently byteswapping column bytes.

    Raises:
        NotImplementedError: The host is big-endian.
    """
    if sys.byteorder != "little":
        raise NotImplementedError(
            f"{_FORMAT} column files are little-endian by definition and this host is "
            f"{sys.byteorder}-endian; byteswapping would break every stored sha256."
        )


def _dtype_string(tensor: torch.Tensor, name: str) -> str:
    """Return the little-endian dtype string of *tensor*.

    Args:
        tensor: Column payload.
        name: Column name, for the error message.

    Returns:
        A numpy-style dtype string such as ``"<f4"``.

    Raises:
        ValueError: ``tensor.dtype`` has no raw little-endian encoding here.
    """
    try:
        return _DTYPE_STRINGS[tensor.dtype]
    except KeyError as exc:
        raise ValueError(
            f"column {name!r} has dtype {tensor.dtype}, which {_FORMAT} cannot store "
            f"as raw little-endian bytes; supported dtypes are {sorted(_DTYPE_STRINGS.values())}."
        ) from exc


def _row_bytes(spec: Mapping[str, Any]) -> int:
    """Return the byte size of one row of the column described by *spec*.

    Args:
        spec: Manifest column spec with ``dtype`` and ``inner_shape``.

    Returns:
        ``itemsize * prod(inner_shape)`` in bytes.
    """
    itemsize = torch.empty(0, dtype=_TORCH_DTYPES[spec["dtype"]]).element_size()
    inner = 1
    for dim in spec["inner_shape"]:
        inner *= int(dim)
    return itemsize * inner


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    """Serialize *document* canonically (sorted keys, no whitespace, UTF-8)."""
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    """Stream *path* in 1 MiB chunks and return its hex sha256 digest."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def _exclusive_cumsum(counts: torch.Tensor) -> torch.Tensor:
    """Return the exclusive cumsum of *counts* with its total appended.

    Args:
        counts: Non-negative per-unit counts ``(M,)``.

    Returns:
        ``(M + 1,)`` ``torch.long`` pointer vector, ``ptr[0] == 0`` and
        ``ptr[-1] == counts.sum()``.
    """
    total = counts.to(torch.long)
    return torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(total, 0)])


# ---------------------------------------------------------------------------
# SamplingUnit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SamplingUnit:
    """One topology plus one or more frames of it — the only ingestion unit.

    A unit is the atom of both sampling and storage: because
    :meth:`BlockWriter.append_unit` is the only ingestion method, a unit can
    never straddle a block boundary. All shape validation happens here, in
    ``__post_init__``, so a malformed unit cannot reach a column file; the writer
    only adds the cross-unit checks (frozen schema, dtypes, seal state).

    Topology tensors are stored once per unit; frame tensors are stored per
    frame with a ``k * n_atoms`` stride. Atom indices in valence columns are
    **molecule-local** ``[0, n_atoms)`` — the global offsetting is the collate's
    job, unchanged.

    Instances are frozen and their two mappings are wrapped in
    :class:`types.MappingProxyType`, so a validated unit stays validated.
    Equality is identity (``eq=False``): tensor payloads make value equality
    ill-defined.

    Args:
        key: Unit identity, e.g. a molecule id or scan id. Stored UTF-8 encoded
            in the ``unit_key_bytes`` / ``unit_key_len`` columns.
        task: Physical task label (``"equilibrium"``, ``"torsion"``, …). Stored
            as a dictionary code in ``mol_task``.
        topology: Molecule-space columns, ``{name: tensor}``. Must contain
            ``atom_Z`` ``(n_atoms,)``; may contain ``atom_mass`` ``(n_atoms,)``
            in amu and valence columns ``<family>_atom{i,j,k,l}`` /
            ``<family>_type`` ``(n_terms,)``.
        frames: Frame-space columns, ``{name: tensor}``. Must contain
            ``frame_pos`` ``(n_frames * n_atoms, 3)`` in Å; may contain
            ``frame_target_<name>`` ``(n_frames, ...)`` and
            ``frame_atom_target_<name>`` ``(n_frames * n_atoms, ...)`` (e.g.
            ``frame_target_energy`` in kcal mol⁻¹, ``frame_atom_target_forces``
            in kcal mol⁻¹ Å⁻¹).

    Raises:
        ValueError: A required column is missing, a column name is unroutable, a
            column sits in the wrong mapping, a row count contradicts
            ``n_atoms`` / ``n_frames``, or a valence atom index falls outside
            ``[0, n_atoms)``. The message names the offending column.
        NotImplementedError: The host is big-endian.
    """

    key: str
    task: str
    topology: Mapping[str, torch.Tensor]
    frames: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        _require_little_endian()
        object.__setattr__(self, "topology", MappingProxyType(dict(self.topology)))
        object.__setattr__(self, "frames", MappingProxyType(dict(self.frames)))
        if not self.key:
            raise ValueError(
                "SamplingUnit.key must be a non-empty string — it is the unit identity"
            )
        if not self.task:
            raise ValueError(
                f"SamplingUnit.task must be a non-empty string for unit {self.key!r} — "
                "it is the task dictionary label"
            )
        if _REQUIRED_TOPOLOGY not in self.topology:
            raise ValueError(
                f"SamplingUnit {self.key!r} is missing required topology column "
                f"{_REQUIRED_TOPOLOGY!r} — every block declares it."
            )
        if _REQUIRED_FRAME not in self.frames:
            raise ValueError(
                f"SamplingUnit {self.key!r} is missing required frame column "
                f"{_REQUIRED_FRAME!r} — a unit is one topology plus >= 1 frames of it."
            )
        self._validate_topology()
        self._validate_frames()

    @property
    def n_atoms(self) -> int:
        """Atoms in this unit's topology, ``N = len(atom_Z)``."""
        return int(self.topology[_REQUIRED_TOPOLOGY].shape[0])

    @property
    def n_frames(self) -> int:
        """Frames of this unit, ``len(frame_pos) // n_atoms`` (exact by validation)."""
        return int(self.frames[_REQUIRED_FRAME].shape[0]) // self.n_atoms

    def _validate_topology(self) -> None:
        """Check molecule-space row counts and molecule-local atom indices."""
        n_atoms = int(self.topology[_REQUIRED_TOPOLOGY].shape[0])
        if n_atoms <= 0:
            raise ValueError(
                f"column {_REQUIRED_TOPOLOGY!r} of unit {self.key!r} has 0 rows — "
                "a unit needs at least one atom."
            )
        term_rows: dict[str, int] = {}
        for name, tensor in self.topology.items():
            space = _column_route(name).space
            if space.startswith("frame"):
                raise ValueError(
                    f"column {name!r} of unit {self.key!r} is a {space}-space column and "
                    "belongs in SamplingUnit.frames, not in topology."
                )
            rows = int(tensor.shape[0])
            if space == "molecule_atom" and rows != n_atoms:
                raise ValueError(
                    f"column {name!r} of unit {self.key!r} has {rows} rows but "
                    f"molecule_atom space needs n_atoms = {n_atoms}."
                )
            if not space.startswith("molecule_term:"):
                continue
            family = space.split(":", 1)[1]
            if family in term_rows and term_rows[family] != rows:
                raise ValueError(
                    f"column {name!r} of unit {self.key!r} has {rows} rows but family "
                    f"{family!r} already declared {term_rows[family]} terms."
                )
            term_rows[family] = rows
            if rows and name.rsplit("_", 1)[-1] in _ATOM_INDEX_COLUMNS:
                low, high = int(tensor.min()), int(tensor.max())
                if low < 0 or high >= n_atoms:
                    raise ValueError(
                        f"column {name!r} of unit {self.key!r} carries atom indices "
                        f"[{low}, {high}] outside the molecule-local range "
                        f"[0, {n_atoms}) — block topology indices are never global."
                    )

    def _validate_frames(self) -> None:
        """Check frame-space and frame-atom-space row counts against ``k * N``."""
        n_atoms = int(self.topology[_REQUIRED_TOPOLOGY].shape[0])
        pos_rows = int(self.frames[_REQUIRED_FRAME].shape[0])
        if pos_rows == 0 or pos_rows % n_atoms:
            raise ValueError(
                f"column {_REQUIRED_FRAME!r} of unit {self.key!r} has {pos_rows} rows, "
                f"not a positive multiple of n_atoms = {n_atoms}: frame_atom rows must "
                "equal n_frames * n_atoms."
            )
        n_frames = pos_rows // n_atoms
        for name, tensor in self.frames.items():
            space = _column_route(name).space
            rows = int(tensor.shape[0])
            if space == "frame":
                if rows != n_frames:
                    raise ValueError(
                        f"column {name!r} of unit {self.key!r} has {rows} rows but frame "
                        f"space needs n_frames = {n_frames}."
                    )
            elif space == "frame_atom":
                if rows != n_frames * n_atoms:
                    raise ValueError(
                        f"column {name!r} of unit {self.key!r} has {rows} rows but "
                        f"frame_atom space needs n_frames * n_atoms = {n_frames * n_atoms}."
                    )
            else:
                raise ValueError(
                    f"column {name!r} of unit {self.key!r} is a {space}-space column and "
                    "belongs in SamplingUnit.topology, not in frames."
                )


# ---------------------------------------------------------------------------
# BlockManifest
# ---------------------------------------------------------------------------


class BlockManifest:
    """The ``manifest.json`` contract of one block.

    Holds the parsed JSON document and owns its two rules: the schema version a
    reader accepts, and the canonical-JSON ``block_digest``. The digest is
    **derived**, never trusted as stored text — it is recomputed on construction
    from ``{columns, rows, split, units, digests}`` (sorted keys, ``(",", ":")``
    separators, UTF-8), which makes it independent of provenance and of where the
    block directory lives, so a sealed block can be copied or moved and still
    verify.

    Args:
        document: Manifest fields — ``format``, ``schema_version``, ``columns``,
            ``split``, ``task_names``, ``units``, ``rows``, ``provenance``,
            ``digests``, ``sealed``. Copied, not aliased.

    Raises:
        ValueError: ``schema_version`` is absent or not the version this module
            writes, or a sealed document is missing a digest field.
    """

    __slots__ = ("_document",)

    def __init__(self, document: Mapping[str, Any]) -> None:
        version = document.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"manifest declares schema_version={version!r} but this molnex writes "
                f"and reads {_SCHEMA_VERSION} ({_FORMAT}); the block was produced by a "
                "different block-format version and must be re-converted."
            )
        self._document = dict(document)
        self._document["block_digest"] = self._derive_block_digest()

    @classmethod
    def read(cls, root: str | os.PathLike[str]) -> BlockManifest:
        """Read the manifest of the block directory *root*.

        Args:
            root: Block directory holding ``manifest.json``.

        Returns:
            The parsed :class:`BlockManifest`.

        Raises:
            ValueError: No manifest at *root*, unparsable JSON, or a
                ``schema_version`` this module does not implement.
        """
        path = Path(root) / _MANIFEST_NAME
        try:
            raw = path.read_text()
        except OSError as exc:
            raise ValueError(
                f"no readable {_MANIFEST_NAME} at {path} — a block directory is only "
                "identified by its manifest."
            ) from exc
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        try:
            return cls(document)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc

    def write_atomic(self, root: str | os.PathLike[str]) -> None:
        """Write this manifest into *root* atomically.

        Writes ``manifest.json.tmp``, fsyncs it, then ``os.replace``\\ s it onto
        ``manifest.json`` — the POSIX-atomic discipline
        :meth:`molix.store.cache.PackedCache.save` already uses, so a reader never
        observes a partial manifest and no temporary file survives the call.

        Args:
            root: Destination block directory; created if absent.
        """
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f"{_MANIFEST_NAME}.tmp"
        with open(tmp, "w") as handle:
            handle.write(json.dumps(self._document, sort_keys=True, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, directory / _MANIFEST_NAME)

    def column(self, name: str) -> Mapping[str, Any]:
        """Return the declared spec of one column.

        Args:
            name: Column name, e.g. ``frame_pos``.

        Returns:
            Mapping with ``dtype`` (little-endian string), ``inner_shape``
            (row shape after dim 0), ``space`` and ``route``.

        Raises:
            KeyError: *name* is not declared by this block.
        """
        try:
            return self._document["columns"][name]
        except KeyError as exc:
            raise KeyError(
                f"block declares no column {name!r}; declared columns are "
                f"{sorted(self._document['columns'])}."
            ) from exc

    @property
    def columns(self) -> Mapping[str, Any]:
        """All declared column specs, ``{name: spec}``."""
        return self._document["columns"]

    @property
    def schema_version(self) -> int:
        """The block format version this manifest was written with."""
        return int(self._document["schema_version"])

    @property
    def sealed(self) -> bool:
        """Whether the block is sealed — the only witness of READY."""
        return bool(self._document.get("sealed", False))

    @property
    def split(self) -> str:
        """The single split this block holds (``"train"`` / ``"valid"`` / …)."""
        return str(self._document["split"])

    @property
    def task_names(self) -> tuple[str, ...]:
        """The bounded task-label dictionary, indexed by ``mol_task`` codes."""
        return tuple(self._document.get("task_names", ()))

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Opaque caller-supplied provenance (source, generator, created_at, units)."""
        return self._document.get("provenance", {})

    @property
    def digests(self) -> Mapping[str, str]:
        """Per-column sha256 hex digests; empty until the block is sealed."""
        return self._document.get("digests", {})

    @property
    def block_digest(self) -> str | None:
        """sha256 over the canonical JSON core, or ``None`` while unsealed."""
        return self._document["block_digest"]

    @property
    def document(self) -> Mapping[str, Any]:
        """The manifest fields as a read-only mapping."""
        return MappingProxyType(self._document)

    def _derive_block_digest(self) -> str | None:
        """Recompute ``block_digest`` from the location-independent core."""
        if not self.sealed:
            return None
        try:
            core = {field: self._document[field] for field in _DIGEST_FIELDS}
        except KeyError as exc:
            raise ValueError(
                f"sealed manifest is missing field {exc.args[0]!r}; block_digest covers "
                f"{list(_DIGEST_FIELDS)}."
            ) from exc
        return hashlib.sha256(_canonical_json(core)).hexdigest()


# ---------------------------------------------------------------------------
# BlockWriter
# ---------------------------------------------------------------------------


class BlockWriter:
    """Append units to one block directory, commit them, and seal it.

    Lifecycle: construct → :meth:`append_unit` per unit → read
    :attr:`committed_bytes` against your own size target → :meth:`seal`. The
    store ships no size policy and no driver: when a block is "full" is the
    caller's decision, and composition (which units go where) lives in scripts
    and regressions.

    The schema is frozen from the first unit — its columns, dtypes and inner
    shapes become the block's contract, plus the synthesised block-index columns
    (``mol_n_atoms``, ``mol_n_frames``, ``mol_task``, ``unit_key_len``,
    ``unit_key_bytes``) and one ``<family>_n`` count column per valence family.
    Every later unit must carry exactly those columns with exactly those dtypes.

    Split purity is structural: ``split`` is a constructor argument recorded once
    in the manifest, :meth:`append_unit` takes no split parameter, and no
    per-unit or per-frame split field exists anywhere in the format.

    Args:
        root: Block directory; created if absent. Must not already contain a
            manifest — use :meth:`resume` to continue an interrupted write.
        split: The single split this block holds, e.g. ``"train"``.
        provenance: Opaque JSON-serializable provenance; only the presence of
            ``source`` / ``generator`` / ``created_at`` / ``units`` is validated,
            so no dataset coordinate is ever hard-coded in molnex.

    Raises:
        ValueError: *root* already holds a manifest, or *provenance* is missing a
            required key.
        NotImplementedError: The host is big-endian.
    """

    __slots__ = (
        "_columns",
        "_lengths",
        "_order",
        "_provenance",
        "_root",
        "_rows",
        "_sealed",
        "_split",
        "_task_names",
        "_unit_count",
    )

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        split: str,
        provenance: Mapping[str, Any],
    ) -> None:
        directory = Path(root)
        if (directory / _MANIFEST_NAME).exists():
            raise ValueError(
                f"block at {directory} already holds {_MANIFEST_NAME}; refusing to freeze "
                "a second schema over committed columns — call BlockWriter.resume(root) "
                "to continue an interrupted write."
            )
        self._configure(directory, split=split, provenance=provenance)

    @classmethod
    def resume(cls, root: str | os.PathLike[str]) -> BlockWriter:
        """Recover an interrupted write and continue at the next unit index.

        Reads the frozen schema from the manifest, scans ``commit.log`` for the
        last crc32-valid fixed-size record, truncates every column to the length
        that record witnesses, truncates the log past it, and returns a writer
        whose :attr:`unit_count` is ``unit_index + 1``. Uncommitted tails — a torn
        column write, a half-written record — are discarded, so the block seals
        byte-identically to an uninterrupted write.

        This invalidates any earlier writer instance for the same directory:
        after truncation its cached lengths and unit count are stale. The writer
        holds no long-lived descriptors, so dropping the old instance is enough.

        Args:
            root: Block directory holding a manifest and (usually) a commit log.

        Returns:
            A writer positioned after the last committed unit.

        Raises:
            ValueError: No readable manifest, or the block is already sealed.
        """
        manifest = BlockManifest.read(root)
        if manifest.sealed:
            raise ValueError(
                f"block at {root} is sealed and cannot be resumed — sealed columns are "
                "immutable; open a new block for further units."
            )
        writer = cls.__new__(cls)
        writer._configure(Path(root), split=manifest.split, provenance=manifest.provenance)
        writer._columns = {name: dict(spec) for name, spec in manifest.columns.items()}
        writer._order = tuple(sorted(writer._columns))
        writer._task_names = list(manifest.task_names)
        writer._truncate_to_last_commit()
        return writer

    # -- construction ------------------------------------------------------

    def _configure(
        self,
        root: Path,
        *,
        split: str,
        provenance: Mapping[str, Any],
    ) -> None:
        """Initialize the empty writer state shared by ``__init__`` and :meth:`resume`."""
        _require_little_endian()
        missing = [key for key in _PROVENANCE_KEYS if key not in provenance]
        if missing:
            raise ValueError(
                f"provenance is missing required key(s) {missing}; {_FORMAT} validates "
                f"presence only — {list(_PROVENANCE_KEYS)} must exist, their contents are "
                "the producer's business."
            )
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._split = str(split)
        self._provenance = dict(provenance)
        self._columns: dict[str, dict[str, Any]] = {}
        self._order: tuple[str, ...] = ()
        self._lengths: dict[str, int] = {}
        self._rows: dict[str, int] = {}
        self._task_names: list[str] = []
        self._unit_count = 0
        self._sealed = False

    # -- accessors ---------------------------------------------------------

    @property
    def unit_count(self) -> int:
        """Number of committed units in this block."""
        return self._unit_count

    @property
    def committed_bytes(self) -> int:
        """Total committed column payload in bytes (no manifest, no commit log).

        This is the number to compare against a 1–5 GB block target before
        deciding to :meth:`seal` and open the next block.
        """
        return sum(self._lengths.values())

    # -- append ------------------------------------------------------------

    def append_unit(self, unit: SamplingUnit) -> int:
        """Append one unit's columns and commit them.

        Freezes the block schema on the first call. Then: validate *unit* against
        the frozen schema, register a new task label (rewriting the manifest
        before any byte is written, so a crash can never leave a committed unit
        whose task code has no name), append every column tail with ``O_APPEND``,
        fsync each touched file, and finally append + fsync the crc32 commit
        record that makes the unit durable.

        Args:
            unit: The unit to append. It carries its own split-free identity —
                there is deliberately no ``split`` parameter here, which is what
                makes split purity structural.

        Returns:
            The unit index just committed (``0``-based, sequential).

        Raises:
            ValueError: The block is sealed, *unit* is missing a declared column,
                declares an undeclared one, or contradicts a frozen dtype / inner
                shape. The message names the offending column.
        """
        if self._sealed:
            raise ValueError(
                f"block at {self._root} is sealed — append_unit is rejected; a sealed "
                "block is immutable, so open a new block for further units."
            )
        froze_now = not self._columns
        if froze_now:
            self._freeze_schema(unit)

        payload = self._unit_columns(unit)
        self._validate_against_schema(unit, payload)

        new_task = unit.task not in self._task_names
        if new_task:
            self._task_names.append(unit.task)
        if froze_now or new_task:
            self._write_manifest()

        for name in self._order:
            self._append_column(name, payload[name])
        self._recompute_rows()

        index = self._unit_count
        self._append_commit_record(index)
        self._unit_count += 1
        return index

    def seal(self) -> None:
        """Finalize the block: per-column sha256, ``block_digest``, ``0o444``.

        Streams every column in 1 MiB chunks, writes the digests plus
        ``sealed: true`` into the manifest atomically, then drops the column files
        to read-only. The sealed, digest-carrying manifest is the **only** witness
        of READY — there is no ``_READY`` sentinel and no half-ready state a
        consumer can read. Every later :meth:`append_unit` raises.

        Raises:
            ValueError: The block is already sealed, or no unit was ever appended.
        """
        if self._sealed:
            raise ValueError(f"block at {self._root} is already sealed")
        if not self._unit_count:
            raise ValueError(
                f"block at {self._root} has no committed unit — seal() would produce a "
                "READY witness for an empty block; append at least one unit first."
            )
        digests = {name: _sha256_file(self._root / f"{name}.bin") for name in self._order}
        self._sealed = True
        self._write_manifest(digests)
        for name in self._order:
            os.chmod(self._root / f"{name}.bin", 0o444)

    # -- schema ------------------------------------------------------------

    def _freeze_schema(self, unit: SamplingUnit) -> None:
        """Derive the block's column contract from the first unit and create files."""
        columns: dict[str, dict[str, Any]] = {}
        for name, (dtype, space) in _META_COLUMNS.items():
            columns[name] = {
                "dtype": dtype,
                "inner_shape": [],
                "space": space,
                "route": _column_route(name).target,
            }
        families: list[str] = []
        for name, tensor in (*unit.topology.items(), *unit.frames.items()):
            route = _column_route(name)
            columns[name] = {
                "dtype": _dtype_string(tensor, name),
                "inner_shape": [int(dim) for dim in tensor.shape[1:]],
                "space": route.space,
                "route": route.target,
            }
            if route.space.startswith("molecule_term:"):
                family = route.space.split(":", 1)[1]
                if family not in families:
                    families.append(family)
        for family in families:
            name = f"{family}_n"
            columns[name] = {
                "dtype": "<i4",
                "inner_shape": [],
                "space": "molecule",
                "route": _column_route(name).target,
            }
        self._columns = columns
        self._order = tuple(sorted(columns))
        self._lengths = dict.fromkeys(self._order, 0)
        for name in self._order:
            (self._root / f"{name}.bin").touch()

    def _unit_columns(self, unit: SamplingUnit) -> dict[str, torch.Tensor]:
        """Return every declared column of *unit*, block-index columns included.

        The block-index columns are synthesised here: per-unit atom / frame
        counts, the task dictionary code, the UTF-8 unit key and its length, and
        one ``<family>_n`` term count per declared valence family. Their pointer
        vectors are derived at read time and never stored.
        """
        payload: dict[str, torch.Tensor] = {**unit.topology, **unit.frames}
        key_bytes = unit.key.encode("utf-8")
        task_code = (
            self._task_names.index(unit.task)
            if unit.task in self._task_names
            else len(self._task_names)
        )
        payload["mol_n_atoms"] = torch.tensor([unit.n_atoms], dtype=torch.int32)
        payload["mol_n_frames"] = torch.tensor([unit.n_frames], dtype=torch.int32)
        payload["mol_task"] = torch.tensor([task_code], dtype=torch.int32)
        payload["unit_key_len"] = torch.tensor([len(key_bytes)], dtype=torch.int32)
        payload["unit_key_bytes"] = torch.frombuffer(bytearray(key_bytes), dtype=torch.uint8)
        for name in self._order:
            family = name[: -len("_n")]
            if not name.endswith("_n") or family not in _FAMILIES:
                continue
            members = [
                column
                for column, spec in self._columns.items()
                if spec["space"] == f"molecule_term:{family}"
            ]
            present = [unit.topology[column] for column in members if column in unit.topology]
            rows = int(present[0].shape[0]) if present else 0
            payload[name] = torch.tensor([rows], dtype=torch.int32)
        return payload

    def _validate_against_schema(
        self,
        unit: SamplingUnit,
        payload: Mapping[str, torch.Tensor],
    ) -> None:
        """Check *payload* against the frozen schema, naming the first offender."""
        declared = set(self._order)
        given = set(payload)
        missing = sorted(declared - given)
        if missing:
            raise ValueError(
                f"unit {unit.key!r} is missing column(s) {missing} declared by the frozen "
                "block schema — every unit must carry every declared column (pass a "
                "zero-row tensor for an empty valence family)."
            )
        extra = sorted(given - declared)
        if extra:
            raise ValueError(
                f"unit {unit.key!r} declares column(s) {extra} absent from the frozen block "
                "schema — the schema is frozen by the first unit and never widened."
            )
        for name in self._order:
            tensor = payload[name]
            spec = self._columns[name]
            if tensor.device.type != "cpu":
                raise ValueError(
                    f"column {name!r} of unit {unit.key!r} lives on device "
                    f"{tensor.device} — block columns are written from CPU tensors."
                )
            dtype = _dtype_string(tensor, name)
            if dtype != spec["dtype"]:
                raise ValueError(
                    f"column {name!r} of unit {unit.key!r} has dtype {dtype} but the frozen "
                    f"schema declares {spec['dtype']} — the store performs no silent dtype "
                    "narrowing or widening."
                )
            inner = [int(dim) for dim in tensor.shape[1:]]
            if inner != [int(dim) for dim in spec["inner_shape"]]:
                raise ValueError(
                    f"column {name!r} of unit {unit.key!r} has inner shape {inner} but the "
                    f"frozen schema declares {list(spec['inner_shape'])}."
                )

    # -- manifest ----------------------------------------------------------

    def _write_manifest(self, digests: Mapping[str, str] | None = None) -> None:
        """Write the current schema / task dictionary / row counts atomically."""
        document: dict[str, Any] = {
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "columns": self._columns,
            "split": self._split,
            "task_names": list(self._task_names),
            "units": self._unit_count,
            "rows": dict(self._rows),
            "provenance": self._provenance,
            "digests": dict(digests or {}),
            "sealed": self._sealed,
        }
        BlockManifest(document).write_atomic(self._root)

    def _recompute_rows(self) -> None:
        """Refresh the per-space row counts from the committed column lengths."""
        for name in self._order:
            spec = self._columns[name]
            self._rows[spec["space"]] = self._lengths[name] // _row_bytes(spec)

    # -- column IO ---------------------------------------------------------

    def _append_column(self, name: str, tensor: torch.Tensor) -> None:
        """Append one column tail with ``O_APPEND`` and fsync it.

        No descriptor outlives the call and no predecessor byte is ever rewritten,
        which is what lets :meth:`resume` treat file length as the only mutable
        state.
        """
        data = memoryview(tensor.contiguous().numpy().tobytes())
        fd = os.open(self._root / f"{name}.bin", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            while data:
                data = data[os.write(fd, data) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        self._lengths[name] += tensor.numel() * tensor.element_size()

    def _append_commit_record(self, index: int) -> None:
        """Append + fsync the crc32 record that makes unit *index* durable.

        Layout (little-endian, ``16 + 8*C`` bytes): ``unit_index`` ``i8``,
        ``n_columns`` ``i4``, one ``committed_len`` ``i8`` per column in sorted
        column-name order, then ``crc32`` ``u4`` over all preceding bytes.
        """
        body = struct.pack("<qi", index, len(self._order))
        body += b"".join(struct.pack("<q", self._lengths[name]) for name in self._order)
        record = body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
        fd = os.open(self._root / _COMMIT_LOG_NAME, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, record)
            os.fsync(fd)
        finally:
            os.close(fd)

    def _truncate_to_last_commit(self) -> None:
        """Roll every column back to the last crc32-valid commit record."""
        size = 16 + 8 * len(self._order)
        log = self._root / _COMMIT_LOG_NAME
        raw = log.read_bytes() if log.exists() else b""
        lengths = dict.fromkeys(self._order, 0)
        last_index = -1
        valid_bytes = 0
        for start in range(0, len(raw) - size + 1, size):
            record = raw[start : start + size]
            body, checksum = record[:-4], struct.unpack("<I", record[-4:])[0]
            if zlib.crc32(body) & 0xFFFFFFFF != checksum:
                break
            unit_index, n_columns = struct.unpack("<qi", body[:12])
            if n_columns != len(self._order):
                break
            last_index = unit_index
            lengths = dict(zip(self._order, struct.unpack(f"<{n_columns}q", body[12:])))
            valid_bytes = start + size
        self._unit_count = last_index + 1
        self._lengths = lengths
        for name in self._order:
            path = self._root / f"{name}.bin"
            path.touch()
            os.truncate(path, lengths[name])
        log.touch()
        os.truncate(log, valid_bytes)
        self._recompute_rows()


# ---------------------------------------------------------------------------
# BlockReader
# ---------------------------------------------------------------------------


class BlockReader:
    """Read accessors and payload views over one sealed block.

    Refuses an unsealed block, so a consumer structurally cannot read a partially
    written one. Columns are mapped lazily and zero-copy through
    :meth:`_map_column`; every pointer vector is derived at open from the stored
    per-unit counts. The accessor surface mirrors what the loader already
    consumes — ``atom_counts`` pointer arithmetic for
    :class:`~molix.batch.sampler.TaskMixtureBatchSampler` and a
    :meth:`frame_payload` adapter for the packed collate — so a batch plan can be
    built without unpacking a single frame.

    Args:
        root: Sealed block directory.

    Raises:
        ValueError: No readable manifest, an unsupported schema version, or the
            block is not sealed.
        NotImplementedError: The host is big-endian.
    """

    __slots__ = (
        "_frame_mol",
        "_manifest",
        "_mapped",
        "_mol_atom_ptr",
        "_mol_frame_atom_ptr",
        "_mol_frame_ptr",
        "_mol_n_atoms",
        "_mol_n_frames",
        "_root",
        "_term_ptr",
        "_unit_keys",
    )

    def __init__(self, root: str | os.PathLike[str]) -> None:
        _require_little_endian()
        self._root = Path(root)
        self._manifest = BlockManifest.read(self._root)
        if not self._manifest.sealed:
            raise ValueError(
                f"block at {self._root} is not sealed — a sealed, digest-carrying manifest "
                "is the only witness of READY, so a partially written block cannot be read."
            )
        self._mapped: dict[str, torch.Tensor] = {}
        self._unit_keys: dict[str, int] | None = None
        self._mol_n_atoms = self._map_column("mol_n_atoms").to(torch.long)
        self._mol_n_frames = self._map_column("mol_n_frames").to(torch.long)
        self._mol_atom_ptr = _exclusive_cumsum(self._mol_n_atoms)
        self._mol_frame_ptr = _exclusive_cumsum(self._mol_n_frames)
        self._mol_frame_atom_ptr = _exclusive_cumsum(self._mol_n_frames * self._mol_n_atoms)
        self._frame_mol = torch.repeat_interleave(
            torch.arange(self._mol_n_atoms.numel()), self._mol_n_frames
        )
        self._term_ptr: dict[str, torch.Tensor] = {
            family: _exclusive_cumsum(self._map_column(f"{family}_n").to(torch.long))
            for family in _FAMILIES
            if f"{family}_n" in self._manifest.columns
        }

    # -- mapping seam ------------------------------------------------------

    def _map_column(self, name: str) -> torch.Tensor:
        """Map one column file zero-copy and cache the tensor.

        The single mapping seam of the reader (see the module docstring for the
        measured zero-copy witness and the documented numpy contingency): the
        whole file is mapped MAP_PRIVATE at offset 0 with
        ``torch.UntypedStorage.from_file`` and reshaped to
        ``(rows, *inner_shape)``. A zero-byte column — a valence family with no
        terms in any unit — cannot be mmap'd and yields an empty tensor of the
        declared dtype and shape instead.

        Args:
            name: Declared column name.

        Returns:
            The column as a ``(rows, *inner_shape)`` tensor of the declared dtype.
            Read-only by contract: writing to it dirties a private page nobody
            else observes.
        """
        cached = self._mapped.get(name)
        if cached is not None:
            return cached
        spec = self._manifest.column(name)
        path = self._root / f"{name}.bin"
        dtype = _TORCH_DTYPES[spec["dtype"]]
        shape = [0, *(int(dim) for dim in spec["inner_shape"])]
        nbytes = path.stat().st_size
        shape[0] = nbytes // _row_bytes(spec)
        if nbytes == 0:
            tensor = torch.empty(shape, dtype=dtype)
        else:
            storage = torch.UntypedStorage.from_file(str(path), shared=False, nbytes=nbytes)
            tensor = torch.empty(0, dtype=dtype).set_(storage, 0, tuple(shape))
        self._mapped[name] = tensor
        return tensor

    # -- columns and counts ------------------------------------------------

    def column(self, name: str) -> torch.Tensor:
        """Return one declared column as a mmap'd tensor.

        Args:
            name: Declared column name, e.g. ``frame_pos``.

        Returns:
            ``(rows, *inner_shape)`` tensor in the block's stored dtype and units
            (Å for ``frame_pos``, kcal mol⁻¹ for ``frame_target_energy``, …).

        Raises:
            KeyError: *name* is not declared by this block.
        """
        return self._map_column(name)

    @property
    def unit_count(self) -> int:
        """Number of sampling units (molecules / scans) in the block, ``M``."""
        return int(self._mol_n_atoms.numel())

    @property
    def frame_count(self) -> int:
        """Number of frames in the block, ``F = sum(mol_n_frames)``."""
        return int(self._mol_n_frames.sum())

    def __len__(self) -> int:
        """Frame count — a block indexes frames, which is what samplers count."""
        return self.frame_count

    @property
    def max_atoms(self) -> int:
        """Largest unit atom count ``max(N_m)`` — a token-budget input."""
        return int(self._mol_n_atoms.max())

    @property
    def atom_counts(self) -> torch.Tensor:
        """Atoms per frame ``(F,)`` ``torch.long`` — ``mol_n_atoms[frame_mol]``.

        Pointer arithmetic only: mirrors
        :attr:`molix.store.dataset.MmapDataset.atom_counts` so a token-budget
        sampler consumes a block without unpacking any frame.
        """
        return self._mol_n_atoms[self._frame_mol]

    @property
    def edge_counts(self) -> torch.Tensor:
        """Never returns — blocks carry no edge columns by design.

        Raises:
            ValueError: Always, naming the deliberate absence and the downstream
                builder.
        """
        raise ValueError(
            f"a {_FORMAT} block stores no edge columns by design: for a 40-atom molecule "
            "the cutoff graph is nearly complete (~1.2e3 directed edges/frame), so "
            "edge_index alone costs ~10x the energy+force frame payload (~29x with "
            "edge_diff + edge_dist) and would pin the cutoff and symmetry mode into the "
            "artifact. edge_counts is therefore unavailable — build the neighbour graph "
            "downstream with molix.batch.NeighborList (or the MD bind path) and budget "
            "batches on atom_counts."
        )

    # -- derived pointers --------------------------------------------------

    @property
    def mol_atom_ptr(self) -> torch.Tensor:
        """``(M+1,)`` ``torch.long`` exclusive cumsum of ``mol_n_atoms``.

        Topology rows of unit ``m`` are ``mol_atom_ptr[m] + [0, N_m)``. Derived at
        open, never stored.
        """
        return self._mol_atom_ptr

    @property
    def mol_frame_ptr(self) -> torch.Tensor:
        """``(M+1,)`` ``torch.long`` exclusive cumsum of ``mol_n_frames``.

        Frame rows of unit ``m`` are ``mol_frame_ptr[m] + [0, n_m)``. Derived at
        open, never stored.
        """
        return self._mol_frame_ptr

    @property
    def mol_frame_atom_ptr(self) -> torch.Tensor:
        """``(M+1,)`` ``torch.long`` exclusive cumsum of ``n_m * N_m``.

        Frame ``k`` of unit ``m`` occupies frame-atom rows
        ``mol_frame_atom_ptr[m] + k*N_m + [0, N_m)``. Derived at open, never
        stored.
        """
        return self._mol_frame_atom_ptr

    @property
    def frame_mol(self) -> torch.Tensor:
        """``(F,)`` ``torch.long`` owning unit row of every frame."""
        return self._frame_mol

    def unit_frames(self, unit: int) -> torch.Tensor:
        """Return the frame rows of one unit.

        Args:
            unit: Unit row in ``[0, unit_count)``.

        Returns:
            ``(n_m,)`` ``torch.long`` frame indices, contiguous and ascending —
            the frames appended together, which a sampler must keep together.
        """
        start = int(self._mol_frame_ptr[unit])
        stop = int(self._mol_frame_ptr[unit + 1])
        return torch.arange(start, stop, dtype=torch.long)

    # -- identity and labels -----------------------------------------------

    def unit_key(self, index: int) -> str:
        """Return the caller-supplied key of one unit.

        Args:
            index: Unit row in ``[0, unit_count)``.

        Returns:
            The UTF-8 decoded unit key, sliced out of ``unit_key_bytes`` with the
            pointer derived from ``unit_key_len``.
        """
        offsets = _exclusive_cumsum(self._map_column("unit_key_len").to(torch.long))
        raw = self._map_column("unit_key_bytes")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        return bytes(raw[start:stop].tolist()).decode("utf-8")

    def unit_index(self, key: str) -> int:
        """Return the unit row carrying *key* (reverse map, built once, lazily).

        Args:
            key: A unit key previously appended to this block.

        Returns:
            The unit row.

        Raises:
            KeyError: No unit of this block carries *key*.
        """
        if self._unit_keys is None:
            self._unit_keys = {self.unit_key(i): i for i in range(self.unit_count)}
        try:
            return self._unit_keys[key]
        except KeyError as exc:
            raise KeyError(f"block at {self._root} has no unit with key {key!r}") from exc

    @property
    def task_names(self) -> tuple[str, ...]:
        """The bounded task-label dictionary; index it with :attr:`task_codes`."""
        return self._manifest.task_names

    @property
    def task_codes(self) -> torch.Tensor:
        """``(F,)`` ``torch.int32`` per-frame task dictionary code.

        Decoding to the ``list[str]`` a sampler accepts is a caller one-liner:
        ``[reader.task_names[c] for c in reader.task_codes.tolist()]``.
        """
        return self._map_column("mol_task")[self._frame_mol].to(torch.int32)

    @property
    def group_codes(self) -> torch.Tensor:
        """``(F,)`` ``torch.int32`` group id — the owning unit row.

        One unit is one group, so no string column and no per-frame group table
        exists: grouping is the storage layout itself.
        """
        return self._frame_mol.to(torch.int32)

    # -- integrity ---------------------------------------------------------

    def verify(self) -> None:
        """Recompute every column sha256 and compare against the manifest.

        Returns:
            ``None`` when every column matches — verification is an assertion,
            not a verdict object.

        Raises:
            ValueError: The first column (in sorted name order) whose bytes do not
                match its recorded digest, named in the message.
        """
        digests = self._manifest.digests
        for name in sorted(digests):
            actual = _sha256_file(self._root / f"{name}.bin")
            if actual != digests[name]:
                raise ValueError(
                    f"column {name!r} of block {self._root} does not match its manifest "
                    f"sha256 ({actual} != {digests[name]}) — the block is corrupt; refetch "
                    "or re-convert it."
                )

    # -- payload adapter ---------------------------------------------------

    def frame_payload(self, frames: Sequence[int]) -> dict[str, Any]:
        """Assemble a packed-cache-v1-shaped payload for exactly *frames*.

        This is an adapter to an existing contract, not a second collate: the
        result is what :class:`molix.store.dataset.PackedView` wraps and
        ``collate_packed`` consumes, so batch construction stays the caller's
        composition::

            collate_packed(PackedView(reader.frame_payload(frames)), range(len(frames)))

        Molecule-level topology is expanded to the selected frames only (the same
        copy the collate would have made), and topology atom indices stay
        **molecule-local** ``[0, N_m)`` so the collate's own ``rebase`` performs
        the global offsetting unchanged. Blocks carry no edges, so the payload
        deliberately exercises the collate's empty-edge fallback.

        Args:
            frames: Frame rows to materialise, in the order they should appear in
                the batch.

        Returns:
            Payload mapping with ``n_samples``, ``atom_ptr`` ``(B+1,)``, an
            ``atoms`` bucket (``Z`` ``(N,)``, ``pos`` ``(N, 3)`` in Å, dotted
            ``targets.<name>`` leaves), a ``graphs`` bucket (dotted
            ``targets.<name>`` leaves), and one bucket + ``<family>_ptr`` per
            declared valence family (``bonds`` carries the stacked ``(2, T)``
            ``bond_index`` and ``bond_types``; other families carry 1-D
            ``atomi`` / ``atomj`` / ``atomk`` / ``atoml`` / ``type`` columns).
            Every topology column is cast to ``torch.int64`` so the payload
            matches the packed-cache v1 dtypes leaf for leaf.

        Raises:
            ValueError: *frames* is empty or names a frame outside
                ``[0, frame_count)``.
        """
        selected = torch.as_tensor([int(f) for f in frames], dtype=torch.long)
        if selected.numel() == 0:
            raise ValueError("frame_payload needs at least one frame — cannot build an empty batch")
        if int(selected.min()) < 0 or int(selected.max()) >= self.frame_count:
            raise ValueError(
                f"frame_payload got frame rows in [{int(selected.min())}, "
                f"{int(selected.max())}] but this block holds {self.frame_count} frames"
            )

        units = self._frame_mol[selected]
        counts = self._mol_n_atoms[units]
        ordinal = selected - self._mol_frame_ptr[units]
        atom_ptr = _exclusive_cumsum(counts)
        segment = torch.repeat_interleave(torch.arange(selected.numel()), counts)
        within = torch.arange(int(atom_ptr[-1])) - atom_ptr[segment]
        frame_atom_rows = (
            self._mol_frame_atom_ptr[units][segment] + ordinal[segment] * counts[segment] + within
        )
        topology_rows = self._mol_atom_ptr[units][segment] + within

        atoms: dict[str, torch.Tensor] = {}
        graphs: dict[str, torch.Tensor] = {}
        for name, spec in self._manifest.columns.items():
            target = str(spec["route"])
            if target.startswith(_META_ROUTE_PREFIX):
                continue
            space = str(spec["space"])
            if space.startswith("molecule_term:"):
                continue
            leaf = target.split(".", 1)[1]
            column = self._map_column(name)
            if space == "molecule_atom":
                atoms[leaf] = column[topology_rows]
            elif space == "frame_atom":
                atoms[leaf] = column[frame_atom_rows]
            elif space == "frame":
                graphs[leaf] = column[selected]
            elif space == "molecule":
                graphs[leaf] = column[units]

        payload: dict[str, Any] = {
            "n_samples": int(selected.numel()),
            "atom_ptr": atom_ptr,
            "atoms": atoms,
            "graphs": graphs,
        }
        for family, term_ptr in self._term_ptr.items():
            payload.update(self._family_payload(family, term_ptr, units))
        return payload

    def _family_payload(
        self,
        family: str,
        term_ptr: torch.Tensor,
        units: torch.Tensor,
    ) -> dict[str, Any]:
        """Gather one valence family's terms for the selected units.

        Args:
            family: Family name, e.g. ``"angle"``.
            term_ptr: ``(M+1,)`` derived term pointer of *family*.
            units: ``(B,)`` owning unit row per selected frame.

        Returns:
            ``{bucket: {column: tensor}, "<family>_ptr": (B+1,)}`` with every
            column cast to ``torch.int64`` and atom indices left molecule-local.
        """
        counts = term_ptr[units + 1] - term_ptr[units]
        ptr = _exclusive_cumsum(counts)
        segment = torch.repeat_interleave(torch.arange(units.numel()), counts)
        within = torch.arange(int(ptr[-1])) - ptr[segment]
        rows = term_ptr[units][segment] + within

        bucket: dict[str, torch.Tensor] = {}
        if family == "bond":
            bucket["bond_index"] = torch.stack(
                [
                    self._map_column("bond_atomi")[rows].to(torch.int64),
                    self._map_column("bond_atomj")[rows].to(torch.int64),
                ],
                dim=0,
            )
            if "bond_type" in self._manifest.columns:
                bucket["bond_types"] = self._map_column("bond_type")[rows].to(torch.int64)
        else:
            for column in (*_ATOM_INDEX_COLUMNS, "type"):
                name = f"{family}_{column}"
                if name in self._manifest.columns:
                    bucket[column] = self._map_column(name)[rows].to(torch.int64)
        return {_FAMILY_BUCKETS[family]: bucket, f"{family}_ptr": ptr}
