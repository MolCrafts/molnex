"""Public-API regression for the ETL exit — acquisition is gone, training is not.

Spec: ``molix-data-dissolution-04-etl-exit`` (acceptance ac-008).

Phase 04 deletes the acquisition layer (``DataSource``, the pipeline DAG, the
task hierarchy and the five dataset adapters) outright: MolHub owns dataset
acquisition, molnex reads packed caches. This script is the exit proof. It
imports only public entry points — ``molix.store``, ``molix.batch``,
``molix.io`` — and never touches a private module, so it fails the moment the
surviving surface stops composing or a shim reappears under the old names.

**Part A — the packages are gone, not shimmed.** ``molix.data`` and
``molix.datasets`` must not resolve (``importlib.util.find_spec(...) is
None``, which also catches a bare directory left behind as a namespace
package) and importing them must raise :class:`ImportError`. Stage:
experimental — deleted with no deprecation path and no alias.

**Part B — the surviving surface composes.** Two hard-coded molecules flow
``NeighborList`` -> ``PackedCache`` -> ``MmapDataset`` -> ``DataModule``, which
is the whole molnex-side data path once acquisition has left::

    mol  atoms  Z              pos (A)                            n_half_pairs
    ---  -----  -------------  ---------------------------------  ------------
    A    3      [8, 1, 1]      (0,0,0), (3,0,0), (0,4,0)          3
    B    4      [1, 6, 6, 1]   chain on x at 0, 1, 2, 3           6

Both geometries are stylized, not experimental. Molecule A is a 3-4-5 right
triangle, so its three pair distances are exactly 3, 4 and 5 A; molecule B is
a unit-spaced chain, so its six pair distances are exactly 1, 1, 1, 2, 2 and 3
A. Every coordinate and every distance is therefore exactly representable in
binary and the goldens are literals compared with ``torch.equal`` — bit-exact,
no tolerance. ``cutoff=5.5 A`` admits every pair in both molecules, so the edge
budget is hand-checkable: ``C(3,2) + C(4,2) = 3 + 6 = 9`` half pairs, doubled
by ``NeighborList(symmetry=True)`` into ``E = 18`` directed edges.

The goldens are::

    graphs.num_atoms      [3, 4]                        (per-graph atom counts)
    atoms.Z               [8, 1, 1, 1, 6, 6, 1]         A then B, concatenated
    atoms.batch           [0, 0, 0, 1, 1, 1, 1]         graph membership
    atoms.pos.shape       (7, 3)
    edges.edge_index.shape (18, 2)
    sorted(edges.edge_dist)  six 1s, four 2s, four 3s, two 4s, two 5s

``edge_dist`` is pinned as a *sorted* vector because the pair enumeration order
is the C++ kernel's, not this file's; the per-edge correspondence is pinned
separately and exactly by re-deriving ``edge_dist`` from the collated
``atoms.pos`` and ``edges.edge_index`` in closed form (``|pos[target] -
pos[source]|``, the repo Edge Convention). That second check is what makes the
sorted multiset non-vacuous: it fails if collate rebases an index wrongly, so
molecule B's edges would gather molecule A's coordinates.

    capture: none — closed-form literals. Every golden above is hand-written
             geometry (a 3-4-5 triangle, a unit chain) plus integer counting;
             nothing was captured from a run, and no third-party oracle is
             imported, subprocessed or consulted, here or at authoring time.
    verify:  PYTHONPATH=src python regressions/molix-data-dissolution-04-etl-exit.py
    commit:  30d1c61 (+ working tree implementing molix-data-dissolution-04-etl-exit)
    torch:   2.12.1+cpu
    device:  cpu, float32 (exact equality throughout — no tolerance is used)
    date:    2026-08-14
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

from molix.batch import DataModule, NeighborList, TargetSchema, collate_molecules
from molix.io import parse_extxyz_frames
from molix.store import MmapDataset, PackedCache

# ---------------------------------------------------------------------------
# Hard-coded goldens — two literal molecules, all-pairs inside the cutoff
# ---------------------------------------------------------------------------

#: Packages deleted in this phase; neither may resolve or import.
_DISSOLVED_PACKAGES = ("molix.data", "molix.datasets")

#: Cutoff in A — larger than every pair distance below, so both molecules are
#: complete graphs and the edge budget is pure combinatorics.
_CUTOFF = 5.5

#: Molecule A — 3-4-5 right triangle: pair distances are exactly 3, 4, 5 A.
_Z_A = [8, 1, 1]
_POS_A = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
#: Molecule B — unit-spaced chain on x: pair distances are exactly 1, 1, 1, 2, 2, 3 A.
_Z_B = [1, 6, 6, 1]
_POS_B = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
#: Graph-level target, one scalar per molecule.
_ENERGY = [-10.0, -20.0]

#: Per-graph atom counts, in dataset order.
_NUM_ATOMS_GOLDEN = [3, 4]
#: Atomic numbers of the collated batch: A concatenated with B.
_Z_GOLDEN = [8, 1, 1, 1, 6, 6, 1]
#: Graph membership of every atom.
_BATCH_GOLDEN = [0, 0, 0, 1, 1, 1, 1]
#: (N_total, 3) — 3 + 4 atoms.
_POS_SHAPE_GOLDEN = (7, 3)
#: (E, 2) — (C(3,2) + C(4,2)) x 2 directions = 18 directed edges.
_EDGE_INDEX_SHAPE_GOLDEN = (18, 2)
#: Every directed edge distance, sorted: A contributes 3,3,4,4,5,5; B
#: contributes 1,1,1,1,1,1,2,2,2,2,3,3.
_EDGE_DIST_SORTED_GOLDEN = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    2.0,
    2.0,
    2.0,
    3.0,
    3.0,
    3.0,
    3.0,
    4.0,
    4.0,
    5.0,
    5.0,
]


def _check_golden_conventions() -> None:
    """Pin the literals to their meaning before they are used as an oracle.

    A typo in the tables above would otherwise silently redefine the golden it
    is compared against. Closed-form: derives the atom counts, the membership
    vector, the edge budget and the full sorted distance multiset from the two
    coordinate tables and the cutoff alone.
    """
    counts = [len(_Z_A), len(_Z_B)]
    if counts != _NUM_ATOMS_GOLDEN:
        raise AssertionError(f"_NUM_ATOMS_GOLDEN is not the atom counts: {counts}")
    if _Z_A + _Z_B != _Z_GOLDEN:
        raise AssertionError("_Z_GOLDEN is not molecule A concatenated with molecule B")
    membership = [0] * len(_Z_A) + [1] * len(_Z_B)
    if membership != _BATCH_GOLDEN:
        raise AssertionError(f"_BATCH_GOLDEN is not graph membership: {membership}")
    if _POS_SHAPE_GOLDEN != (len(_Z_A) + len(_Z_B), 3):
        raise AssertionError("_POS_SHAPE_GOLDEN is not (N_A + N_B, 3)")

    distances: list[float] = []
    for positions in (_POS_A, _POS_B):
        pos = torch.tensor(positions)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = float((pos[j] - pos[i]).norm())
                if dist >= _CUTOFF:
                    raise AssertionError(
                        f"pair ({i},{j}) at {dist} A is outside cutoff {_CUTOFF} — the "
                        "complete-graph edge budget below would be wrong"
                    )
                distances.extend((dist, dist))  # symmetry=True: both directions
    if (len(distances), 2) != _EDGE_INDEX_SHAPE_GOLDEN:
        raise AssertionError(
            f"_EDGE_INDEX_SHAPE_GOLDEN is not the directed edge count: {len(distances)}"
        )
    if sorted(distances) != _EDGE_DIST_SORTED_GOLDEN:
        raise AssertionError(
            f"_EDGE_DIST_SORTED_GOLDEN is not the pair distances: {sorted(distances)}"
        )


def _check_packages_dissolved() -> None:
    """Part A: neither deleted package resolves or imports."""
    resolvable = [
        name for name in _DISSOLVED_PACKAGES if importlib.util.find_spec(name) is not None
    ]
    if resolvable:
        raise AssertionError(
            f"{resolvable} still resolve — dataset acquisition belongs to MolHub and "
            "these packages were deleted outright (no shim, no alias, and no bare "
            "directory left behind as a namespace package)"
        )

    for name in _DISSOLVED_PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        raise AssertionError(f"import {name} still succeeds — it was deleted outright")


def _make_samples() -> list[dict]:
    """Build the two molecules as flat sample dicts with their edge fields.

    Returns:
        One dict per molecule with per-atom ``Z`` / ``pos``, the per-edge
        fields written by :class:`NeighborList`, and a graph-level
        ``targets.energy``.
    """
    neighbors = NeighborList(cutoff=_CUTOFF, symmetry=True)
    samples = []
    for z, positions, energy in ((_Z_A, _POS_A, _ENERGY[0]), (_Z_B, _POS_B, _ENERGY[1])):
        sample = {
            "Z": torch.tensor(z, dtype=torch.long),
            "pos": torch.tensor(positions),
            "targets": {"energy": torch.tensor([energy])},
        }
        samples.append(neighbors.execute(sample))
    return samples


def _check_batch(batch, origin: str) -> None:
    """Assert one collated batch reproduces every literal golden.

    Args:
        batch: Post-collate nested ``TensorDict``.
        origin: Which producer built it, named in failures.
    """
    checks: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        (
            "graphs.num_atoms",
            batch["graphs", "num_atoms"],
            torch.tensor(_NUM_ATOMS_GOLDEN, dtype=batch["graphs", "num_atoms"].dtype),
        ),
        ("atoms.Z", batch["atoms", "Z"], torch.tensor(_Z_GOLDEN, dtype=batch["atoms", "Z"].dtype)),
        (
            "atoms.batch",
            batch["atoms", "batch"],
            torch.tensor(_BATCH_GOLDEN, dtype=batch["atoms", "batch"].dtype),
        ),
    ]
    for name, got, golden in checks:
        if not torch.equal(got, golden):
            raise AssertionError(f"{origin}: {name} drift: {got.tolist()} vs {golden.tolist()}")

    shapes = [
        ("atoms.pos", tuple(batch["atoms", "pos"].shape), _POS_SHAPE_GOLDEN),
        ("edges.edge_index", tuple(batch["edges", "edge_index"].shape), _EDGE_INDEX_SHAPE_GOLDEN),
    ]
    for name, got_shape, golden_shape in shapes:
        if got_shape != golden_shape:
            raise AssertionError(f"{origin}: {name} shape {got_shape} != {golden_shape}")

    edge_dist = batch["edges", "edge_dist"]
    sorted_dist = torch.sort(edge_dist).values
    if not torch.equal(sorted_dist, torch.tensor(_EDGE_DIST_SORTED_GOLDEN, dtype=edge_dist.dtype)):
        raise AssertionError(
            f"{origin}: sorted edges.edge_dist drift: {sorted_dist.tolist()} vs "
            f"{_EDGE_DIST_SORTED_GOLDEN}"
        )

    # Per-edge correspondence, in closed form: a mis-rebased edge_index would
    # make molecule B's edges gather molecule A's coordinates while leaving the
    # sorted multiset above intact.
    pos = batch["atoms", "pos"]
    edge_index = batch["edges", "edge_index"]
    derived = (pos[edge_index[:, 1]] - pos[edge_index[:, 0]]).norm(dim=-1)
    if not torch.equal(derived, edge_dist):
        raise AssertionError(
            f"{origin}: edges.edge_dist is not |pos[target] - pos[source]| — the "
            "rebase or the edge convention drifted"
        )

    energy = batch["graphs", "energy"]
    if not torch.equal(energy, torch.tensor(_ENERGY, dtype=energy.dtype)):
        raise AssertionError(f"{origin}: graphs.energy drift: {energy.tolist()} vs {_ENERGY}")


def _check_surviving_surface(workdir: Path) -> None:
    """Part B: NeighborList -> PackedCache -> MmapDataset -> DataModule."""
    samples = _make_samples()

    # The direct collate of the same samples is the oracle the loader path is
    # held against — both must produce the identical batch.
    schema = TargetSchema(graph_level=frozenset({"energy"}), atom_level=frozenset())
    _check_batch(collate_molecules(samples, target_schema=schema), "collate_molecules")

    sink = workdir / "exit.pt"
    PackedCache(sink).save(samples)
    dataset = MmapDataset(sink)
    if len(dataset) != len(samples):
        raise AssertionError(f"MmapDataset: len {len(dataset)} != {len(samples)}")

    module = DataModule(
        dataset,
        dataset,
        target_schema=schema,
        batch_size=len(samples),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    module.setup("fit")

    # The val loader keeps dataset order (no shuffle), so the goldens above —
    # which are written A-then-B — apply verbatim.
    batches = list(module.val_dataloader())
    if len(batches) != 1:
        raise AssertionError(f"DataModule: {len(batches)} val batches, expected 1")
    _check_batch(batches[0], "DataModule.val_dataloader")


def _check_io_entry_point() -> None:
    """The extxyz codec stayed behind when the dataset adapters left."""
    if not callable(parse_extxyz_frames):
        raise AssertionError("molix.io.parse_extxyz_frames is not callable")


def main() -> int:
    _check_packages_dissolved()
    _check_golden_conventions()
    _check_io_entry_point()
    with tempfile.TemporaryDirectory() as tmp:
        _check_surviving_surface(Path(tmp))

    print(
        "molix-data-dissolution-04-etl-exit: molix.data / molix.datasets neither "
        "resolve nor import; NeighborList -> PackedCache -> MmapDataset -> DataModule "
        "reproduces num_atoms=[3,4], Z=[8,1,1,1,6,6,1], batch=[0,0,0,1,1,1,1], "
        "pos (7,3), edge_index (18,2) and edge_dist {1x6, 2x4, 3x4, 4x2, 5x2} exactly OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
