"""Public-API regression for ``molix.batch`` — collate + padding + task cut.

Spec: ``molix-data-dissolution-02-batch`` (acceptance ac-008).

The batch-construction half of ``molix.data`` moved into ``molix.batch``, and
the one coupling from training-side code into the ETL pipeline DSL was cut:
``DataModule`` no longer takes ``batch_nodes: Sequence[Node]``, it takes
``batch_tasks: Sequence[BatchTask]``. This script pins both at the *user*
surface — it imports only ``molix.batch`` and ``molix.store`` public names and
never touches a private module, so it fails the moment the re-export, the
collated batch schema, or the post-collate transform contract drifts.

**Part A — collation goldens.** Two hard-coded molecules are collated with
:func:`collate_molecules`::

    mol  atoms  Z              pos (A)                          edge_index
    ---  -----  -------------  -------------------------------  ------------------------
    A    3      [8, 1, 1]      O at origin, H at +x, H at +y     [[0,1],[0,2]]
    B    4      [1, 6, 6, 1]   linear chain on x at 0,1,2,3      [[0,1],[1,2],[2,3]]

Both geometries are stylized (unit spacing, right-angle "water"), not
experimental: every coordinate, displacement and distance is then exactly
representable in binary, so the goldens are literals compared with
``torch.equal`` — bit-exact, no tolerance. Edges are stored as half pairs
(``NeighborList(symmetry=False)`` form, source→target only) purely to keep the
edge budget hand-checkable; the rebase arithmetic under test is
direction-agnostic. ``edge_diff`` follows the repo convention
``pos[target] - pos[source]``, and is re-derived here from ``pos`` /
``edge_index`` in closed form so a typo in the table cannot quietly redefine
the golden it is compared against.

The collation goldens are::

    graphs.num_atoms   [3, 4]                      (graph count is never padded)
    atoms.Z            [8, 1, 1, 1, 6, 6, 1]       A then B, concatenated
    atoms.batch        [0, 0, 0, 1, 1, 1, 1]       graph membership
    edges.edge_index   [[0,1],[0,2],[3,4],[4,5],[5,6]]

Molecule B's local ``[[0,1],[1,2],[2,3]]`` becomes ``[[3,4],[4,5],[5,6]]``:
every index shifted by +3, molecule A's atom count. That offset is the whole
point of the ``INDEX_KEYS`` rebase registry — un-rebased connectivity would
silently make molecule B a second copy of molecule A's neighbourhood.

**Part B — padding through ``batch_tasks``.** The same two molecules are
written to a :class:`~molix.store.PackedCache` and read back through
``MmapDataset`` into a :class:`~molix.batch.DataModule` configured with
``batch_tasks=[PadMolecularBatch(max_atoms=8, max_edges=8)]`` — the parameter
that replaced ``batch_nodes``. One loader batch must come back at fixed shapes
(8 atoms, 8 edges; 7 real atoms + 1 ghost, 5 real edges + 3 padding edges)
with::

    atoms.mask         [1,1,1,1,1,1,1,0]   True = real atom
    atoms.Z[7]         1                   ghost gets pad_atom_type (embeddable)
    atoms.pos[7]       [0, 0, 0]           ghost payload is zeroed
    edges.edge_index[5:]  [[7,7],[7,7],[7,7]]  self-loops on the FIRST ghost

and every real row bitwise identical to Part A's unpadded batch. Those three
facts are the zero-contribution contract: the mask zeroes ghost energy before
the per-graph scatter-sum, and self-loops on a ghost mean no padding edge ever
gathers from or scatters to a real atom, so real atoms' energy and forces are
unchanged by padding. A padding edge pointing at a real atom, or a mask one
element short, would corrupt forces on a CUDA-graph training run with no error.

**Part C — the coupling is cut.** No module may claim post-collate routing any
more: while the acquisition pipeline still did, a ``PadMolecularBatch`` on the
``molix.batch`` base matched neither its cacheable nor its batch-node branch,
so the padding task was silently dropped — a wrong-shape training run with no
error. Post-collate configuration belongs to the DataModule alone. Phase 04
deleted the pipeline outright, so the check is now that the module holding the
routing no longer resolves at all.

    capture: none — closed-form literals. Every golden above is hand-written
             geometry plus integer index arithmetic; nothing was captured from
             a run, and no third-party oracle is imported, subprocessed or
             consulted, here or at authoring time.
    verify:  PYTHONPATH=src python regressions/molix-data-dissolution-02-batch.py
    commit:  eeed814 (+ working tree implementing molix-data-dissolution-02-batch)
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
from tensordict import TensorDict

from molix.batch import DataModule, PadMolecularBatch, TargetSchema, collate_molecules
from molix.store import MmapDataset, PackedCache

# ---------------------------------------------------------------------------
# Hard-coded goldens — two stylized molecules, exact binary coordinates
# ---------------------------------------------------------------------------

#: Targets: ``energy`` per graph, ``forces`` per atom (padding must zero the latter).
SCHEMA = TargetSchema(graph_level=frozenset({"energy"}), atom_level=frozenset({"forces"}))

#: Molecule A — O at the origin, H at +1 A on x, H at +1 A on y (right angle).
_A_Z = [8, 1, 1]
_A_POS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
_A_EDGE_INDEX = [[0, 1], [0, 2]]
_A_EDGE_DIFF = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
_A_EDGE_DIST = [1.0, 1.0]
_A_ENERGY = -1.0
_A_FORCES = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]

#: Molecule B — linear H-C-C-H chain along x with unit spacing.
_B_Z = [1, 6, 6, 1]
_B_POS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
_B_EDGE_INDEX = [[0, 1], [1, 2], [2, 3]]
_B_EDGE_DIFF = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
_B_EDGE_DIST = [1.0, 1.0, 1.0]
_B_ENERGY = -2.0
_B_FORCES = [[0.25, 0.0, 0.0], [-0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]]

#: Collated goldens (A then B).
_NUM_ATOMS_GOLDEN = [3, 4]
_Z_GOLDEN = [8, 1, 1, 1, 6, 6, 1]
_BATCH_VECTOR_GOLDEN = [0, 0, 0, 1, 1, 1, 1]
#: Molecule B's indices shifted by +3 = len(_A_Z).
_EDGE_INDEX_GOLDEN = [[0, 1], [0, 2], [3, 4], [4, 5], [5, 6]]
_ENERGY_GOLDEN = [_A_ENERGY, _B_ENERGY]
_ATOM_OFFSET_GOLDEN = 3

#: Fixed shapes requested from ``PadMolecularBatch``.
_MAX_ATOMS = 8
_MAX_EDGES = 8
#: 7 real atoms + 1 ghost; True = real.
_MASK_GOLDEN = [True] * 7 + [False]
#: Ghost atomic number (``PadMolecularBatch`` default ``pad_atom_type``).
_GHOST_Z_GOLDEN = 1
#: Index of the first ghost atom — every padding edge self-loops here.
_GHOST0_GOLDEN = 7
#: 3 padding edges, each a (ghost0, ghost0) self-loop.
_PAD_EDGE_INDEX_GOLDEN = [[7, 7], [7, 7], [7, 7]]


def _molecule_a() -> dict:
    """Molecule A as a flat sample dict (3 atoms, 2 half-pair edges)."""
    return {
        "Z": torch.tensor(_A_Z, dtype=torch.long),
        "pos": torch.tensor(_A_POS),
        "edge_index": torch.tensor(_A_EDGE_INDEX, dtype=torch.long),
        "edge_diff": torch.tensor(_A_EDGE_DIFF),
        "edge_dist": torch.tensor(_A_EDGE_DIST),
        "targets": {
            "energy": torch.tensor([_A_ENERGY]),
            "forces": torch.tensor(_A_FORCES),
        },
    }


def _molecule_b() -> dict:
    """Molecule B as a flat sample dict (4 atoms, 3 half-pair edges)."""
    return {
        "Z": torch.tensor(_B_Z, dtype=torch.long),
        "pos": torch.tensor(_B_POS),
        "edge_index": torch.tensor(_B_EDGE_INDEX, dtype=torch.long),
        "edge_diff": torch.tensor(_B_EDGE_DIFF),
        "edge_dist": torch.tensor(_B_EDGE_DIST),
        "targets": {
            "energy": torch.tensor([_B_ENERGY]),
            "forces": torch.tensor(_B_FORCES),
        },
    }


def _assert_equal(got: torch.Tensor, golden: torch.Tensor, what: str) -> None:
    """Assert two tensors are bit-identical, naming *what* drifted."""
    if got.dtype != golden.dtype:
        raise AssertionError(f"{what} dtype drift: {got.dtype} vs golden {golden.dtype}")
    if not torch.equal(got, golden):
        raise AssertionError(f"{what} drift: {got.tolist()} vs golden {golden.tolist()}")


def _check_golden_conventions() -> None:
    """Pin the literals to their meaning before they are used as an oracle.

    Derives ``edge_diff`` / ``edge_dist`` from ``pos`` and ``edge_index`` in
    closed form (``pos[target] - pos[source]``), and re-derives the rebased
    ``edge_index`` golden from molecule A's atom count — so a typo in the
    tables above cannot silently redefine the golden it is compared against.
    """
    for tag, pos_l, idx_l, diff_l, dist_l in (
        ("A", _A_POS, _A_EDGE_INDEX, _A_EDGE_DIFF, _A_EDGE_DIST),
        ("B", _B_POS, _B_EDGE_INDEX, _B_EDGE_DIFF, _B_EDGE_DIST),
    ):
        pos = torch.tensor(pos_l)
        edge_index = torch.tensor(idx_l, dtype=torch.long)
        diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]  # target - source
        _assert_equal(diff, torch.tensor(diff_l), f"molecule {tag}: edge_diff literal")
        _assert_equal(diff.norm(dim=-1), torch.tensor(dist_l), f"molecule {tag}: edge_dist literal")

    if _ATOM_OFFSET_GOLDEN != len(_A_Z):
        raise AssertionError(
            f"_ATOM_OFFSET_GOLDEN is not molecule A's atom count: "
            f"{_ATOM_OFFSET_GOLDEN} vs {len(_A_Z)}"
        )
    rebased_b = [[i + _ATOM_OFFSET_GOLDEN, j + _ATOM_OFFSET_GOLDEN] for i, j in _B_EDGE_INDEX]
    derived = _A_EDGE_INDEX + rebased_b
    if derived != _EDGE_INDEX_GOLDEN:
        raise AssertionError(
            f"_EDGE_INDEX_GOLDEN is not molecule B rebased by +{_ATOM_OFFSET_GOLDEN}: "
            f"{_EDGE_INDEX_GOLDEN} vs {derived}"
        )
    n_real_atoms = len(_A_Z) + len(_B_Z)
    if _MASK_GOLDEN != [True] * n_real_atoms + [False] * (_MAX_ATOMS - n_real_atoms):
        raise AssertionError(f"_MASK_GOLDEN does not mark exactly {n_real_atoms} real atoms")
    if _GHOST0_GOLDEN != n_real_atoms:
        raise AssertionError(
            f"_GHOST0_GOLDEN is not the first ghost slot: {_GHOST0_GOLDEN} vs {n_real_atoms}"
        )


def _check_collate() -> TensorDict:
    """Part A: collate the two molecules and pin every literal golden.

    Returns:
        The unpadded batch, reused in Part B as the real-row oracle.
    """
    batch = collate_molecules([_molecule_a(), _molecule_b()], SCHEMA)

    if list(batch["atoms"].batch_size) != [len(_Z_GOLDEN)]:
        raise AssertionError(f"atoms batch_size drift: {list(batch['atoms'].batch_size)}")
    if list(batch["edges"].batch_size) != [len(_EDGE_INDEX_GOLDEN)]:
        raise AssertionError(f"edges batch_size drift: {list(batch['edges'].batch_size)}")
    if list(batch["graphs"].batch_size) != [len(_NUM_ATOMS_GOLDEN)]:
        raise AssertionError(f"graphs batch_size drift: {list(batch['graphs'].batch_size)}")

    _assert_equal(
        batch["graphs", "num_atoms"],
        torch.tensor(_NUM_ATOMS_GOLDEN, dtype=torch.long),
        "graphs.num_atoms",
    )
    _assert_equal(batch["atoms", "Z"], torch.tensor(_Z_GOLDEN, dtype=torch.long), "atoms.Z")
    _assert_equal(
        batch["atoms", "batch"],
        torch.tensor(_BATCH_VECTOR_GOLDEN, dtype=torch.long),
        "atoms.batch",
    )
    _assert_equal(batch["atoms", "pos"], torch.tensor(_A_POS + _B_POS), "atoms.pos")
    _assert_equal(batch["atoms", "forces"], torch.tensor(_A_FORCES + _B_FORCES), "atoms.forces")
    _assert_equal(
        batch["edges", "edge_index"],
        torch.tensor(_EDGE_INDEX_GOLDEN, dtype=torch.long),
        "edges.edge_index (rebased)",
    )
    _assert_equal(
        batch["edges", "edge_diff"], torch.tensor(_A_EDGE_DIFF + _B_EDGE_DIFF), "edges.edge_diff"
    )
    _assert_equal(
        batch["edges", "edge_dist"], torch.tensor(_A_EDGE_DIST + _B_EDGE_DIST), "edges.edge_dist"
    )
    _assert_equal(batch["graphs", "energy"], torch.tensor(_ENERGY_GOLDEN), "graphs.energy")
    return batch


def _check_padding(workdir: Path, unpadded: TensorDict) -> None:
    """Part B: drive ``PadMolecularBatch`` through ``DataModule(batch_tasks=...)``.

    Args:
        workdir: Directory for the packed cache holding the two molecules.
        unpadded: Part A's batch — the oracle for the real (non-ghost) rows.
    """
    sink = workdir / "two_molecules.pt"
    PackedCache(sink).save([_molecule_a(), _molecule_b()])
    dataset = MmapDataset(sink)

    datamodule = DataModule(
        dataset,
        dataset,
        target_schema=SCHEMA,
        batch_tasks=[PadMolecularBatch(max_atoms=_MAX_ATOMS, max_edges=_MAX_EDGES)],
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    # The val loader never shuffles, so this batch is [molecule A, molecule B].
    batch = next(iter(datamodule.val_dataloader()))

    # --- fixed shapes ---
    if list(batch["atoms"].batch_size) != [_MAX_ATOMS]:
        raise AssertionError(
            f"padded atoms batch_size drift: {list(batch['atoms'].batch_size)} vs [{_MAX_ATOMS}] "
            "— batch_tasks was not applied (the batch_nodes coupling cut silently dropping it?)"
        )
    if list(batch["edges"].batch_size) != [_MAX_EDGES]:
        raise AssertionError(
            f"padded edges batch_size drift: {list(batch['edges'].batch_size)} vs [{_MAX_EDGES}]"
        )
    # Graphs are NOT padded — pair with drop_last=True for the third static dim.
    _assert_equal(
        batch["graphs", "num_atoms"],
        torch.tensor(_NUM_ATOMS_GOLDEN, dtype=torch.long),
        "padded graphs.num_atoms",
    )

    # --- ghost-atom layout ---
    _assert_equal(
        batch["atoms", "mask"], torch.tensor(_MASK_GOLDEN, dtype=torch.bool), "padded atoms.mask"
    )
    ghost_z = int(batch["atoms", "Z"][_GHOST0_GOLDEN].item())
    if ghost_z != _GHOST_Z_GOLDEN:
        raise AssertionError(f"ghost atom Z drift: {ghost_z} vs golden {_GHOST_Z_GOLDEN}")
    _assert_equal(
        batch["atoms", "pos"][_GHOST0_GOLDEN:], torch.zeros(1, 3), "ghost atoms.pos (must be zero)"
    )
    _assert_equal(
        batch["atoms", "forces"][_GHOST0_GOLDEN:],
        torch.zeros(1, 3),
        "ghost atoms.forces (must be zero)",
    )

    # --- padding edges are self-loops on the FIRST ghost atom ---
    n_real_edges = len(_EDGE_INDEX_GOLDEN)
    _assert_equal(
        batch["edges", "edge_index"][n_real_edges:],
        torch.tensor(_PAD_EDGE_INDEX_GOLDEN, dtype=torch.long),
        "padding edges (must self-loop on the first ghost atom)",
    )
    _assert_equal(
        batch["edges", "edge_dist"][n_real_edges:],
        torch.zeros(_MAX_EDGES - n_real_edges),
        "padding edges.edge_dist (must be zero)",
    )

    # --- real rows survive padding bit-for-bit ---
    n_real_atoms = len(_Z_GOLDEN)
    for namespace, key, n_real in (
        ("atoms", "Z", n_real_atoms),
        ("atoms", "pos", n_real_atoms),
        ("atoms", "batch", n_real_atoms),
        ("atoms", "forces", n_real_atoms),
        ("edges", "edge_index", n_real_edges),
        ("edges", "edge_diff", n_real_edges),
        ("edges", "edge_dist", n_real_edges),
    ):
        _assert_equal(
            batch[namespace, key][:n_real],
            unpadded[namespace, key],
            f"padded {namespace}.{key} real rows",
        )


def _check_pipeline_no_longer_routes_batch_tasks() -> None:
    """Part C: nothing in molnex routes post-collate tasks through a pipeline.

    Phase 02 moved ``BatchTask`` onto ``DataModule(batch_tasks=...)``; phase 04
    deleted the acquisition layer that used to own the routing, so the strongest
    form of the check is that the module which held it no longer resolves.
    """
    if importlib.util.find_spec("molix.data") is not None:
        raise AssertionError(
            "molix.data still resolves — post-collate tasks belong to "
            "DataModule(batch_tasks=...); a pipeline that keeps routing them would "
            "silently drop a PadMolecularBatch built on the molix.batch base"
        )


def main() -> int:
    _check_golden_conventions()
    unpadded = _check_collate()
    with tempfile.TemporaryDirectory() as tmp:
        _check_padding(Path(tmp), unpadded)
    _check_pipeline_no_longer_routes_batch_tasks()

    print(
        "molix-data-dissolution-02-batch: collate_molecules packs 3+4 atoms exactly "
        "(num_atoms=[3,4], Z=[8,1,1,1,6,6,1], batch=[0,0,0,1,1,1,1], "
        "edge_index=[[0,1],[0,2],[3,4],[4,5],[5,6]] — molecule B rebased by +3); "
        "DataModule(batch_tasks=[PadMolecularBatch(max_atoms=8, max_edges=8)]) pads to "
        "8 atoms / 8 edges with mask=[1]*7+[0], ghost Z=1 at slot 7, padding edges "
        "[[7,7],[7,7],[7,7]] and bitwise-unchanged real rows; PipelineSpec.batch_nodes gone OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
