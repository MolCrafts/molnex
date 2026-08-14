"""Public-API regression for ``molix.store`` — packed round-trip + dissolution.

Spec: ``molix-data-dissolution-01-store`` (acceptance ac-007).

The storage half of ``molix.data`` moved verbatim into ``molix.store``. This
script pins the move at the *user* surface: it imports only
``from molix.store import CachedDataset, MmapDataset, PackedCache`` and never
touches a private module, so it fails the moment the public re-export or the
on-disk layout drifts.

**Part A — packed round-trip.** Three hard-coded two-atom samples are written
into a :class:`tempfile.TemporaryDirectory` with :meth:`PackedCache.save` and
read back through **both** readers (``MmapDataset`` = mmap'd storages,
``CachedDataset`` = resident copy). Every sample is the same rigid geometry —
H at the origin, C at +1 A along x, the single pair stored in both directions —
so the goldens are literals, not measurements::

    key           value                                   bucket
    ------------  --------------------------------------  ------
    Z             [1, 6]                                   atom
    pos           [[0, 0, 0], [1, 0, 0]]                   atom
    edge_index    [[0, 1], [1, 0]]                         edge
    edge_diff     [[1, 0, 0], [-1, 0, 0]]                  edge
    edge_dist     [1.0, 1.0]                               edge
    targets.U0    [float(i)]  (i = sample index)           graph

``edge_diff`` follows the repo convention ``pos[target] - pos[source]``, hence
the sign flip on the reverse edge; ``edge_dist = |edge_diff| = 1.0`` exactly
(unit vector along x, no floating-point residue). Comparison is
``torch.equal`` — bit-exact, no tolerance — because a byte-preserving move must
not perturb a single mantissa bit. The literals are first re-derived from
``pos`` / ``edge_index`` in closed form, so a typo in the table cannot quietly
redefine the golden it is compared against.

``n_edges == n_atoms == 2`` in every sample is deliberate: the leading dim
cannot distinguish a per-atom key from a per-edge key, so a schema that decided
buckets by numerology would swallow the edge keys into the atoms bucket, drop
``edge_ptr`` and silently report ``avg_num_neighbors == 0.0``. The golden is::

    avg_num_neighbors = total_edges / total_atoms = (3 x 2) / (3 x 2) = 1.0

exact in binary (6 / 6), so it is asserted with ``==``.

**Part B — dissolution.** The old home must be *gone*, not shimmed:
``importlib.util.find_spec("molix.data.cache")`` is ``None`` and
``from molix.data import PesBundle`` raises ``ImportError`` (the orphan
container was deleted outright, stage: experimental, no deprecation path).

    capture: none — closed-form literals. Every golden above is hand-written
             geometry plus one exact integer ratio; nothing was captured from a
             run, and no third-party oracle is imported, subprocessed or
             consulted, here or at authoring time.
    verify:  PYTHONPATH=src python regressions/molix-data-dissolution-01-store.py
    commit:  8ed6b96 (+ working tree implementing molix-data-dissolution-01-store)
    torch:   2.12.1+cpu
    device:  cpu, float32 (exact equality throughout — no tolerance is used)
    date:    2026-08-14
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

from molix.store import CachedDataset, MmapDataset, PackedCache

# ---------------------------------------------------------------------------
# Hard-coded goldens — one rigid 2-atom geometry, repeated three times
# ---------------------------------------------------------------------------

#: Number of samples written into the cache.
_N_SAMPLES = 3
#: Atomic numbers: H, C.
_Z_GOLDEN = [1, 6]
#: Positions in A — atom 1 sits at +1 A along x from atom 0.
_POS_GOLDEN = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
#: The single pair in both directions ([:, 0] = source, [:, 1] = target).
_EDGE_INDEX_GOLDEN = [[0, 1], [1, 0]]
#: pos[target] - pos[source], so the reverse edge flips sign.
_EDGE_DIFF_GOLDEN = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
#: |edge_diff| — exactly 1.0, no floating-point residue.
_EDGE_DIST_GOLDEN = [1.0, 1.0]
#: (3 samples x 2 edges) / (3 samples x 2 atoms) = 1.0, exact in binary.
_AVG_NUM_NEIGHBORS_GOLDEN = 1.0


def _make_samples() -> list[dict]:
    """Build the three golden samples as flat cache-input dicts.

    Returns:
        Flat sample dicts with per-atom ``Z`` ``(2,)`` / ``pos`` ``(2, 3)``,
        per-edge ``edge_index`` ``(2, 2)`` / ``edge_diff`` ``(2, 3)`` /
        ``edge_dist`` ``(2,)`` and graph-level ``targets.U0`` ``(1,)``
        carrying the identity ``float(i)``.
    """
    return [
        {
            "Z": torch.tensor(_Z_GOLDEN, dtype=torch.long),
            "pos": torch.tensor(_POS_GOLDEN),
            "edge_index": torch.tensor(_EDGE_INDEX_GOLDEN, dtype=torch.long),
            "edge_diff": torch.tensor(_EDGE_DIFF_GOLDEN),
            "edge_dist": torch.tensor(_EDGE_DIST_GOLDEN),
            "targets": {"U0": torch.tensor([float(i)])},
        }
        for i in range(_N_SAMPLES)
    ]


def _tensor(sample: dict, *path: str) -> torch.Tensor:
    """Return ``sample[path...]``, asserting it survived as a tensor.

    Args:
        sample: Unpacked sample dict from a reader's ``__getitem__``.
        path: Nested key path, e.g. ``("targets", "U0")``.

    Returns:
        The tensor stored at *path*.

    Raises:
        AssertionError: The key is missing or no longer holds a tensor.
    """
    node: object = sample
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise AssertionError(f"missing key {'.'.join(path)} in unpacked sample")
        node = node[key]
    if not isinstance(node, torch.Tensor):
        raise AssertionError(f"{'.'.join(path)} unpacked as {type(node).__name__}, not Tensor")
    return node


def _check_sample(sample: dict, idx: int, reader: str) -> None:
    """Assert one unpacked sample equals the literal goldens bit-for-bit."""
    expected: list[tuple[tuple[str, ...], torch.Tensor]] = [
        (("Z",), torch.tensor(_Z_GOLDEN, dtype=torch.long)),
        (("pos",), torch.tensor(_POS_GOLDEN)),
        (("edge_index",), torch.tensor(_EDGE_INDEX_GOLDEN, dtype=torch.long)),
        (("edge_diff",), torch.tensor(_EDGE_DIFF_GOLDEN)),
        (("edge_dist",), torch.tensor(_EDGE_DIST_GOLDEN)),
        (("targets", "U0"), torch.tensor([float(idx)])),
    ]
    for path, golden in expected:
        got = _tensor(sample, *path)
        if not torch.equal(got, golden):
            raise AssertionError(
                f"{reader}[{idx}].{'.'.join(path)} drift: "
                f"{got.tolist()} vs golden {golden.tolist()}"
            )


def _check_reader(dataset: CachedDataset | MmapDataset, reader: str) -> None:
    """Assert a reader exposes all three samples and the neighbour statistic."""
    if len(dataset) != _N_SAMPLES:
        raise AssertionError(f"{reader}: len {len(dataset)} != {_N_SAMPLES}")
    for idx in range(_N_SAMPLES):
        _check_sample(dataset[idx], idx, reader)

    avg = dataset.avg_num_neighbors
    if avg != _AVG_NUM_NEIGHBORS_GOLDEN:
        raise AssertionError(
            f"{reader}.avg_num_neighbors drift: {avg} vs {_AVG_NUM_NEIGHBORS_GOLDEN} "
            "(0.0 means the edge keys were filed into the atoms bucket and "
            "edge_ptr was dropped)"
        )


def _check_golden_conventions() -> None:
    """Pin the literals to their meaning before they are used as an oracle.

    A round-trip compares what came back against what went in, so the literals
    must be checked against the *convention* they claim to encode — otherwise a
    typo in this file would silently redefine the golden. Closed-form: derives
    ``edge_diff`` / ``edge_dist`` from ``pos`` and ``edge_index`` alone.
    """
    pos = torch.tensor(_POS_GOLDEN)
    edge_index = torch.tensor(_EDGE_INDEX_GOLDEN, dtype=torch.long)
    diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]  # target - source
    if not torch.equal(diff, torch.tensor(_EDGE_DIFF_GOLDEN)):
        raise AssertionError(
            f"_EDGE_DIFF_GOLDEN is not pos[target] - pos[source]: "
            f"{_EDGE_DIFF_GOLDEN} vs {diff.tolist()}"
        )
    if not torch.equal(diff.norm(dim=-1), torch.tensor(_EDGE_DIST_GOLDEN)):
        raise AssertionError(
            f"_EDGE_DIST_GOLDEN is not |edge_diff|: "
            f"{_EDGE_DIST_GOLDEN} vs {diff.norm(dim=-1).tolist()}"
        )
    n_atoms, n_edges = len(_Z_GOLDEN), len(_EDGE_INDEX_GOLDEN)
    if n_edges / n_atoms != _AVG_NUM_NEIGHBORS_GOLDEN:
        raise AssertionError(
            f"_AVG_NUM_NEIGHBORS_GOLDEN is not n_edges / n_atoms: "
            f"{_AVG_NUM_NEIGHBORS_GOLDEN} vs {n_edges / n_atoms}"
        )


def _check_roundtrip(workdir: Path) -> None:
    """Part A: write the goldens once, read them back through both readers."""
    sink = workdir / "store.pt"
    PackedCache(sink).save(_make_samples())

    _check_reader(MmapDataset(sink), "MmapDataset")
    _check_reader(CachedDataset(sink), "CachedDataset")


def _resolves(module: str) -> bool:
    """Whether *module* can still be located by the import system.

    ``find_spec`` imports the parent package first, so once phase 04 deleted
    ``molix.data`` entirely a submodule query raises instead of returning
    ``None``. A missing parent is the strongest form of "gone", so it is
    reported as "does not resolve" rather than propagated.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _check_dissolved() -> None:
    """Part B: the old ``molix.data`` storage surface is gone, not shimmed."""
    if _resolves("molix.data.cache"):
        raise AssertionError(
            "molix.data.cache is still importable — the storage layer must live "
            "only in molix.store, with no shim left behind"
        )

    try:
        from molix.data import PesBundle  # noqa: F401  (import must fail)
    except ImportError:
        return
    raise AssertionError("molix.data.PesBundle is still importable — it was deleted outright")


def main() -> int:
    _check_dissolved()
    _check_golden_conventions()
    with tempfile.TemporaryDirectory() as tmp:
        _check_roundtrip(Path(tmp))

    print(
        "molix-data-dissolution-01-store: PackedCache -> MmapDataset / CachedDataset "
        "round-trips 3 samples exactly (Z=[1,6], pos=[[0,0,0],[1,0,0]], "
        "edge_index=[[0,1],[1,0]], edge_diff=[[1,0,0],[-1,0,0]], edge_dist=[1,1], "
        "U0=[0,1,2]), avg_num_neighbors = 1.0, molix.data.cache absent and "
        "molix.data.PesBundle gone OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
