#!/usr/bin/env python3
"""Measure READY-tier block IO on the filesystem a training job will actually use.

Answers the two questions ``streaming-data-tiers-02-block-format`` § "Measured
decisions" leaves open before the hot read path is frozen:

1. **Which read path wins** — the :class:`molix.store.BlockReader` mmap gather or
   an ``os.pread`` of exactly the same byte ranges — under sequential and under
   shuffled frame order. The Lustre client's mmap page-fault readahead was
   historically poor, so this is measured per client, never assumed.
2. **What one unit costs to append** — ``append + fsync(columns) + crc32 commit
   record + fsync(log)`` in units/s and ms/unit, plus the ``seal()`` sha256
   streaming rate the spec's byte accounting assumed at 1 GB/s.

The script writes a disposable block, measures, prints three tables and deletes
the block again (``--keep`` retains it). It imports stdlib + numpy + torch only.

Method
------

* A pass gathers ``--pass-frames`` frames of one ``frame_atom``-space column in
  ``--batch-frames``-sized batches. ``sequential`` order is frames
  ``[0, pass_frames)`` (contiguous byte ranges); ``shuffled`` order is the first
  ``pass_frames`` entries of ``torch.randperm(F)`` under ``--seed``, so both
  orders move identical payload bytes and only locality differs. The reported
  MB/s is always **useful payload** — under shuffled mmap access the kernel may
  fault in a whole 4 kB page per 480 B frame, and that amplification is exactly
  what the number is meant to expose.
* Both paths compute the same checksum (int64 sum over the gathered int32 bit
  patterns) and the script asserts they agree, which is the proof that the two
  paths read the same bytes and that no lazy mapping escaped being touched.
* **Cold vs warm.** Every trial runs twice: once after ``posix_fadvise
  (POSIX_FADV_DONTNEED)`` on the column file (first touch) and once immediately
  after (warm page cache). ``DONTNEED`` is advisory — ``mincore`` residency is
  reported before and after so the cold column is a witnessed fact rather than a
  hope; on a client that ignores the hint both numbers converge and only the warm
  row is meaningful. Dropping the whole node's cache needs root and is never
  attempted.

Usage::

    OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/store/bench_block_read.py
    OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/store/bench_block_read.py \\
        --root /scratch/bench --units 64 --frames-per-unit 1024 --keep

Default block: 256 units x 8192 frames x 40 atoms -> ~1.01 GB on disk.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import mmap
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from molix.store import BlockReader, BlockWriter, SamplingUnit  # noqa: E402

_BENCH_COLUMN = "frame_pos"
_MB = 1e6


# ---------------------------------------------------------------------------
# page cache
# ---------------------------------------------------------------------------


class PageCache:
    """Advisory page-cache control and a residency witness for one file.

    ``drop`` is ``posix_fadvise(POSIX_FADV_DONTNEED)``, which needs no privilege
    but is a hint: a client may keep the pages. ``resident_fraction`` calls
    ``mincore(2)`` over a private read-only mapping, so "the column was cold" is
    an observation instead of an assumption.

    Args:
        path: File to advise on and inspect.
    """

    __slots__ = ("_libc", "_path")

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        self._libc.mmap.restype = ctypes.c_void_p
        self._libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        self._libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._libc.mincore.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
        ]

    def drop(self) -> None:
        """Ask the kernel to evict this file's clean pages (best effort)."""
        if not hasattr(os, "posix_fadvise"):
            return
        fd = os.open(self._path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def resident_fraction(self) -> float:
        """Return the fraction of this file's pages currently in page cache.

        Returns:
            Fraction in ``[0, 1]``, or ``nan`` when ``mincore`` is unavailable.
        """
        size = self._path.stat().st_size
        if size == 0:
            return float("nan")
        fd = os.open(self._path, os.O_RDONLY)
        try:
            addr = self._libc.mmap(None, size, 1, 2, fd, 0)  # PROT_READ, MAP_PRIVATE
        finally:
            os.close(fd)
        if addr in (None, ctypes.c_void_p(-1).value):
            return float("nan")
        try:
            pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            vector = (ctypes.c_ubyte * pages)()
            if self._libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vector) != 0:
                return float("nan")
            return sum(entry & 1 for entry in vector) / pages
        finally:
            self._libc.munmap(ctypes.c_void_p(addr), size)


# ---------------------------------------------------------------------------
# byte layout shared by both read paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameLayout:
    """Per-frame byte ranges of one ``frame_atom``-space column.

    Derived from a block's count columns only (``mol_n_atoms`` / ``mol_n_frames``),
    so building it does not fault in a single payload page.

    Attributes:
        starts: ``(F,)`` first frame-atom row of every frame.
        counts: ``(F,)`` atoms per frame.
        row_bytes: Bytes of one frame-atom row (``12`` for ``frame_pos``).
    """

    starts: torch.Tensor
    counts: torch.Tensor
    row_bytes: int

    @classmethod
    def of(cls, reader: BlockReader, column: str) -> FrameLayout:
        """Build the layout of *column* from *reader*'s derived pointers.

        Args:
            reader: Sealed-block reader.
            column: A ``frame_atom``-space column name.

        Returns:
            The :class:`FrameLayout` of *column*.
        """
        units = reader.frame_mol
        counts = reader.atom_counts
        ordinal = torch.arange(reader.frame_count) - reader.mol_frame_ptr[units]
        starts = reader.mol_frame_atom_ptr[units] + ordinal * counts
        spec = reader.column(column)
        return cls(starts, counts, spec.element_size() * int(spec.shape[1:].numel()))

    def payload_bytes(self, frames: torch.Tensor) -> int:
        """Return the useful payload of a pass over *frames*, in bytes."""
        return int(self.counts[frames].sum()) * self.row_bytes


class MmapGather:
    """Gather frame rows through the :class:`BlockReader` mmap column.

    Opens its own reader, so constructing one is what makes a cold pass cold: a
    reader caches its mapping for the life of the instance.

    Args:
        root: Sealed block directory.
        column: Column to gather.
        layout: Byte layout of *column*.
    """

    __slots__ = ("_column", "_layout")

    def __init__(self, root: Path, column: str, layout: FrameLayout) -> None:
        self._column = BlockReader(root).column(column)
        self._layout = layout

    def read(self, frames: torch.Tensor, batch: int) -> int:
        """Gather *frames* in *batch*-sized groups and return the checksum.

        Args:
            frames: Frame rows to gather, in access order.
            batch: Frames per gather call.

        Returns:
            int64 sum of the gathered values' int32 bit patterns.
        """
        checksum = 0
        for start in range(0, frames.numel(), batch):
            chunk = frames[start : start + batch]
            counts = self._layout.counts[chunk]
            segment = torch.repeat_interleave(torch.arange(chunk.numel()), counts)
            offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)])
            rows = self._layout.starts[chunk][segment] + (
                torch.arange(int(offsets[-1])) - offsets[segment]
            )
            checksum += int(self._column[rows].view(torch.int32).sum(dtype=torch.int64))
        return checksum


class PreadGather:
    """Gather exactly the same byte ranges with one ``os.pread`` per frame.

    Args:
        root: Sealed block directory.
        column: Column to gather.
        layout: Byte layout of *column*.
    """

    __slots__ = ("_fd", "_layout")

    def __init__(self, root: Path, column: str, layout: FrameLayout) -> None:
        self._fd = os.open(root / f"{column}.bin", os.O_RDONLY)
        self._layout = layout

    def read(self, frames: torch.Tensor, batch: int) -> int:
        """Read *frames* in *batch*-sized groups and return the checksum.

        Args:
            frames: Frame rows to read, in access order.
            batch: Frames per assembled buffer.

        Returns:
            int64 sum of the read values' int32 bit patterns.
        """
        row_bytes = self._layout.row_bytes
        starts = self._layout.starts
        counts = self._layout.counts
        checksum = 0
        for start in range(0, frames.numel(), batch):
            chunk = frames[start : start + batch].tolist()
            pieces = [
                os.pread(
                    self._fd,
                    int(counts[frame]) * row_bytes,
                    int(starts[frame]) * row_bytes,
                )
                for frame in chunk
            ]
            checksum += int(np.frombuffer(b"".join(pieces), dtype=np.int32).sum(dtype=np.int64))
        return checksum

    def close(self) -> None:
        """Close the file descriptor."""
        os.close(self._fd)


@dataclass(frozen=True)
class ReadResult:
    """One (path, order) trial: a first-touch pass and a warm pass.

    Attributes:
        path: ``"mmap"`` or ``"pread"``.
        order: ``"sequential"`` or ``"shuffled"``.
        cold_seconds: Wall time of the pass right after ``POSIX_FADV_DONTNEED``.
        warm_seconds: Wall time of the immediately repeated pass.
        payload_bytes: Useful bytes gathered per pass.
        checksum: Gathered-data checksum, identical across paths by construction.
        resident_before: Page-cache residency of the column after the drop.
        resident_after: Page-cache residency after the first-touch pass.
    """

    path: str
    order: str
    cold_seconds: float
    warm_seconds: float
    payload_bytes: int
    checksum: int
    resident_before: float
    resident_after: float

    @property
    def cold_mbps(self) -> float:
        """First-touch payload throughput in MB/s."""
        return self.payload_bytes / self.cold_seconds / _MB

    @property
    def warm_mbps(self) -> float:
        """Warm payload throughput in MB/s."""
        return self.payload_bytes / self.warm_seconds / _MB


# ---------------------------------------------------------------------------
# the disposable block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WriteCost:
    """Cost of producing one block.

    Attributes:
        units: Units appended.
        append_seconds: Wall time of all ``append_unit`` calls.
        seal_seconds: Wall time of ``seal()`` (per-column sha256 streaming).
        payload_bytes: Committed column payload.
        columns: Declared column count (fsyncs per unit are ``columns + 1``).
    """

    units: int
    append_seconds: float
    seal_seconds: float
    payload_bytes: int
    columns: int

    @property
    def units_per_second(self) -> float:
        """Committed units per second."""
        return self.units / self.append_seconds

    @property
    def ms_per_unit(self) -> float:
        """Milliseconds per append + fsync + commit record + fsync."""
        return 1e3 * self.append_seconds / self.units

    @property
    def append_mbps(self) -> float:
        """Committed payload throughput in MB/s."""
        return self.payload_bytes / self.append_seconds / _MB

    @property
    def seal_mbps(self) -> float:
        """sha256 streaming rate of ``seal()`` in MB/s."""
        return self.payload_bytes / self.seal_seconds / _MB


class ScratchBlock:
    """Write one disposable sealed block of synthetic units.

    Every unit is one topology (``atom_Z`` plus bonds and angles as int64
    columns) and ``frames_per_unit`` frames of ``frame_pos`` — the ``frame_atom``
    float32 bulk this benchmark reads — with a float32 energy and an int64 step
    counter in frame space. Positions are deterministic ramps: the benchmark
    measures bytes moving, not chemistry.

    Args:
        root: Directory to create the block in.
        units: Number of sampling units.
        frames_per_unit: Frames per unit.
        atoms: Atoms per unit.
    """

    __slots__ = ("_atoms", "_frames_per_unit", "_root", "_units")

    def __init__(self, root: Path, *, units: int, frames_per_unit: int, atoms: int) -> None:
        self._root = Path(root)
        self._units = int(units)
        self._frames_per_unit = int(frames_per_unit)
        self._atoms = int(atoms)

    @property
    def root(self) -> Path:
        """The block directory."""
        return self._root

    def write(self) -> WriteCost:
        """Append every unit, seal, and return the measured cost."""
        writer = BlockWriter(
            self._root,
            split="train",
            provenance={
                "source": {"synthetic": "bench_block_read"},
                "generator": {"tool": "bench_block_read.py", "version": "1", "commit": "n/a"},
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "units": {"frame_pos": "angstrom", "frame_target_energy": "kcal/mol"},
            },
        )
        append_seconds = 0.0
        for index in range(self._units):
            unit = self._unit(index)
            start = time.perf_counter()
            writer.append_unit(unit)
            append_seconds += time.perf_counter() - start
        payload_bytes = writer.committed_bytes
        columns = len(list(Path(self._root).glob("*.bin")))
        start = time.perf_counter()
        writer.seal()
        seal_seconds = time.perf_counter() - start
        return WriteCost(self._units, append_seconds, seal_seconds, payload_bytes, columns)

    def _unit(self, index: int) -> SamplingUnit:
        """Build unit *index*: one topology plus ``frames_per_unit`` frames."""
        atoms = self._atoms
        rows = self._frames_per_unit * atoms
        bond_i = torch.arange(atoms - 1, dtype=torch.int64)
        angle_i = torch.arange(atoms - 2, dtype=torch.int64)
        return SamplingUnit(
            key=f"unit-{index:06d}",
            task="torsion" if index % 2 else "equilibrium",
            topology={
                "atom_Z": torch.full((atoms,), 6, dtype=torch.int16),
                "bond_atomi": bond_i,
                "bond_atomj": bond_i + 1,
                "bond_type": torch.zeros(atoms - 1, dtype=torch.int64),
                "angle_atomi": angle_i,
                "angle_atomj": angle_i + 1,
                "angle_atomk": angle_i + 2,
            },
            frames={
                "frame_pos": (torch.arange(rows * 3, dtype=torch.float32) * 0.5).reshape(rows, 3),
                "frame_target_energy": torch.arange(
                    self._frames_per_unit, dtype=torch.float32
                ).mul_(0.25),
                "frame_target_step": torch.arange(self._frames_per_unit, dtype=torch.int64),
            },
        )


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _table(title: str, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Print one compact fixed-width table.

    Args:
        title: Table caption.
        header: Column headings.
        rows: Pre-formatted cells, one tuple per row.
    """
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]
    line = "  ".join("-" * width for width in widths)
    print(f"\n{title}")
    print("  ".join(cell.ljust(width) for cell, width in zip(header, widths)))
    print(line)
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def _verdict(results: list[ReadResult]) -> str:
    """Return the one-line read-path verdict over every measured trial.

    Args:
        results: Every (path, order) trial.

    Returns:
        A sentence naming the winner per order and the largest ratio, in the
        form the spec asks to be recorded when ``pread`` wins by > 1.5x.
    """
    parts = []
    for order in ("sequential", "shuffled"):
        for phase in ("cold", "warm"):
            speed = {
                result.path: (result.cold_mbps if phase == "cold" else result.warm_mbps)
                for result in results
                if result.order == order
            }
            winner = max(speed, key=lambda path: speed[path])
            loser = min(speed, key=lambda path: speed[path])
            ratio = speed[winner] / speed[loser] if speed[loser] else float("inf")
            parts.append(f"{order}/{phase}: {winner} {ratio:.2f}x")
    return "read-path verdict — " + "; ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO / ".bench-scratch",
        help="scratch directory for the disposable blocks (default: <repo>/.bench-scratch)",
    )
    parser.add_argument("--units", type=int, default=256, help="units in the read block")
    parser.add_argument("--frames-per-unit", type=int, default=8192, help="frames per unit")
    parser.add_argument("--atoms", type=int, default=40, help="atoms per unit")
    parser.add_argument(
        "--pass-frames", type=int, default=65536, help="frames gathered per read pass"
    )
    parser.add_argument("--batch-frames", type=int, default=256, help="frames per gather call")
    parser.add_argument(
        "--append-units", type=int, default=1000, help="small units for the append trial"
    )
    parser.add_argument(
        "--append-frames", type=int, default=4, help="frames per unit in the append trial"
    )
    parser.add_argument(
        "--append-atoms", type=int, default=12, help="atoms per unit in the append trial"
    )
    parser.add_argument("--seed", type=int, default=20260814, help="permutation seed")
    parser.add_argument("--keep", action="store_true", help="keep the scratch blocks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write, measure, print and clean up.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    # A per-process session directory: the benchmark only ever deletes what it
    # created, never the --root the caller pointed at.
    root = Path(args.root) / f"bench-{os.getpid()}"
    root.mkdir(parents=True)

    print(f"host        {platform.node()}")
    print(f"root        {root}")
    print(f"torch       {torch.__version__}")
    print(f"python      {platform.python_version()}")
    print(f"date        {time.strftime('%Y-%m-%d %H:%M:%S %z')}")

    try:
        read_block = ScratchBlock(
            root / "read-block",
            units=args.units,
            frames_per_unit=args.frames_per_unit,
            atoms=args.atoms,
        )
        write_cost = read_block.write()
        append_block = ScratchBlock(
            root / "append-block",
            units=args.append_units,
            frames_per_unit=args.append_frames,
            atoms=args.append_atoms,
        )
        append_cost = append_block.write()
        results = _measure_reads(read_block.root, args)
        _report(write_cost, append_cost, results, args)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    return 0


def _measure_reads(block: Path, args: argparse.Namespace) -> list[ReadResult]:
    """Run the four (path, order) trials against *block*.

    Args:
        block: Sealed block directory.
        args: Parsed command line.

    Returns:
        One :class:`ReadResult` per trial.

    Raises:
        SystemExit: The two read paths disagree on the gathered checksum.
    """
    reader = BlockReader(block)
    layout = FrameLayout.of(reader, _BENCH_COLUMN)
    total = reader.frame_count
    count = min(args.pass_frames, total)
    generator = torch.Generator().manual_seed(args.seed)
    orders = {
        "sequential": torch.arange(count),
        "shuffled": torch.randperm(total, generator=generator)[:count],
    }
    cache = PageCache(block / f"{_BENCH_COLUMN}.bin")

    results: list[ReadResult] = []
    for order, frames in orders.items():
        payload = layout.payload_bytes(frames)
        for path in ("mmap", "pread"):
            gather = (
                MmapGather(block, _BENCH_COLUMN, layout)
                if path == "mmap"
                else PreadGather(block, _BENCH_COLUMN, layout)
            )
            cache.drop()
            before = cache.resident_fraction()
            start = time.perf_counter()
            checksum = gather.read(frames, args.batch_frames)
            cold = time.perf_counter() - start
            after = cache.resident_fraction()
            start = time.perf_counter()
            gather.read(frames, args.batch_frames)
            warm = time.perf_counter() - start
            if isinstance(gather, PreadGather):
                gather.close()
            results.append(ReadResult(path, order, cold, warm, payload, checksum, before, after))
        pair = [result.checksum for result in results[-2:]]
        if pair[0] != pair[1]:
            raise SystemExit(f"{order}: mmap and pread read different bytes {pair}")
    return results


def _report(
    write_cost: WriteCost,
    append_cost: WriteCost,
    results: list[ReadResult],
    args: argparse.Namespace,
) -> None:
    """Print the three benchmark tables and the verdict line.

    Args:
        write_cost: Cost of producing the read block.
        append_cost: Cost of the small-unit append trial.
        results: Read trials.
        args: Parsed command line.
    """
    payload = results[0].payload_bytes
    _table(
        f"column read — {_BENCH_COLUMN}, {args.pass_frames} frames/pass "
        f"({payload / _MB:.1f} MB payload), batch {args.batch_frames}",
        ("order", "path", "cold s", "cold MB/s", "warm s", "warm MB/s", "resident pre/post"),
        [
            (
                result.order,
                result.path,
                f"{result.cold_seconds:.3f}",
                f"{result.cold_mbps:.1f}",
                f"{result.warm_seconds:.3f}",
                f"{result.warm_mbps:.1f}",
                f"{result.resident_before:.3f}/{result.resident_after:.3f}",
            )
            for result in results
        ],
    )
    _table(
        "append + commit + fsync",
        ("block", "units", "cols", "ms/unit", "units/s", "MB/s", "seal s", "seal MB/s"),
        [
            (
                name,
                str(cost.units),
                str(cost.columns),
                f"{cost.ms_per_unit:.3f}",
                f"{cost.units_per_second:.1f}",
                f"{cost.append_mbps:.1f}",
                f"{cost.seal_seconds:.3f}",
                f"{cost.seal_mbps:.1f}",
            )
            for name, cost in (("read-block", write_cost), ("append-block", append_cost))
        ],
    )
    print()
    print(_verdict(results))


if __name__ == "__main__":
    raise SystemExit(main())
