"""Public-API regression for TaskMixtureBatchSampler (hierarchical task mixture).

Spec: ``universal-class-i-10-sampler`` (acceptance ac-013).

One minimal scenario, public API only: a 24-sample :class:`~molix.store.PackedCache`
is written into a :class:`tempfile.TemporaryDirectory`, read back through
:class:`~molix.store.MmapDataset`, and handed to
:class:`~molix.batch.TaskMixtureBatchSampler` with the ByteFF-derived **Stage B**
mixture (``equilibrium 0.35 / torsion 0.45 / relax 0.20``, arXiv:2408.12817),
``max_atoms=24``, ``indices=range(20)`` (the electrostatic block is excluded from
this split) and ``seed=20260814``. No third-party oracle is imported,
subprocessed, or consulted — here or at capture time.

Fixture table — sample ``i`` carries ``_ATOM_COUNTS[i]`` atoms and
``_ATOM_COUNTS[i] + 2`` edges, so a unit's aggregate atom cost is the plain sum
over its member frames::

    idx     task           group    frames  atoms/frame  Sum atoms
    ---------------------------------------------------------------
    0..5    equilibrium    eq0..eq5      1            5          5
    6..9    torsion        tor_a         4            6         24
    10..12  torsion        tor_b         3            6         18
    13,14   torsion        tor_c         2            7         14
    15..17  relax          rel_a         3            4         12
    18,19   relax          rel_b         2            4          8
    20..23  electrostatic  ele_a/ele_b   2            5         10   (not in split)

Everything below the index literals is hand-derivable from that table:

1. **Per-task supply under ``max_atoms=24``** (next-fit, one open bin, whole
   units only) — equilibrium: 4 units (20 atoms) then 2 units (10) = **2**
   batches; torsion: 24 / 18 / 14, no two aggregate to <= 24 = **3** batches;
   relax: 12 + 8 = 20 = **1** batch.
2. **Natural epoch length** ``min_{p>0} floor(len(batches_t) / p_t)``
   = ``min(floor(2/0.35)=5, floor(3/0.45)=6, floor(1/0.20)=5)`` = **5**.
3. **Largest-remainder apportionment** over 5 slots: shares are equilibrium
   1.75, relax 1.00, torsion 2.25 -> bases (1, 1, 2), one leftover to the
   largest remainder (equilibrium, 0.75) -> ``{equilibrium: 2, relax: 1,
   torsion: 2}``.
4. **Frame counts** follow from which packed batches those slots consume:
   equilibrium 4 + 2 = 6, relax 5, torsion 3 + 2 = 5. The configured mixture
   governs *batch* shares only; frame shares are reported for audit.

Only the *index literals* in ``_GOLDEN_PLAN`` come from running this repo's own
deterministic plan; the counts above are hand-computed, so a mismatch means the
sampler deviated from the spec algorithm, not that the goldens went stale.

    capture: PYTHONPATH=src python regressions/universal-class-i-10-sampler.py
    commit: 01eb7d6 (+ working tree implementing universal-class-i-10-sampler)
    torch: 2.12.1+cpu
    device: cpu, float32 (index plan is integer-only; no FP tolerance applies)
    date: 2026-08-14
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from molix.batch import TaskMixtureBatchSampler
from molix.store import MmapDataset, PackedCache

# ---------------------------------------------------------------------------
# Fixture tables (see module docstring)
# ---------------------------------------------------------------------------

#: Per-sample atom counts of the 24-sample task fixture.
_ATOM_COUNTS: list[int] = (
    [5] * 6  # 0..5    equilibrium singletons
    + [6] * 4  # 6..9    torsion scan tor_a
    + [6] * 3  # 10..12  torsion scan tor_b
    + [7] * 2  # 13,14   torsion scan tor_c
    + [4] * 3  # 15..17  relax trajectory rel_a
    + [4] * 2  # 18,19   relax trajectory rel_b
    + [5] * 4  # 20..23  electrostatic ele_a / ele_b
)
#: Per-sample task labels, parent space, ``len == len(dataset)``.
_TASK_LABELS: list[str] = (
    ["equilibrium"] * 6 + ["torsion"] * 9 + ["relax"] * 5 + ["electrostatic"] * 4
)
#: Per-sample group ids, parent space, ``len == len(dataset)``.
_GROUP_IDS: list[str] = (
    [f"eq{i}" for i in range(6)]
    + ["tor_a"] * 4
    + ["tor_b"] * 3
    + ["tor_c"] * 2
    + ["rel_a"] * 3
    + ["rel_b"] * 2
    + ["ele_a"] * 2
    + ["ele_b"] * 2
)
#: Split index list — the electrostatic block (20..23) is outside this split.
_INDICES: list[int] = list(range(20))
#: ByteFF-derived Stage B mixture (arXiv:2408.12817), sums to 1.0.
_STAGE_B: dict[str, float] = {"equilibrium": 0.35, "torsion": 0.45, "relax": 0.20}
#: Per-batch atom-token budget.
_MAX_ATOMS = 24
#: Plan seed.
_SEED = 20260814

# ---------------------------------------------------------------------------
# Hard-coded goldens
# ---------------------------------------------------------------------------

#: ``list(sampler)`` for Stage B / max_atoms=24 / seed 20260814 (hard-coded golden).
_GOLDEN_PLAN: list[list[int]] = [
    [3, 2, 5, 1],  # equilibrium: 4 units x 5 atoms = 20 (a 5th would be 25 > 24)
    [15, 16, 17, 18, 19],  # relax: rel_a (12) + rel_b (8) = 20
    [10, 11, 12],  # torsion: tor_b, 18 atoms
    [13, 14],  # torsion: tor_c, 14 atoms
    [0, 4],  # equilibrium: the 2 leftover units, 10 atoms
]
#: Largest-remainder apportionment of the 5 slots (hard-coded golden).
_GOLDEN_BATCHES_PER_TASK: dict[str, int] = {"equilibrium": 2, "relax": 1, "torsion": 2}
#: Frame counts implied by ``_GOLDEN_PLAN`` (hard-coded golden).
_GOLDEN_FRAMES_PER_TASK: dict[str, int] = {"equilibrium": 6, "relax": 5, "torsion": 5}
#: Natural epoch length (hard-coded golden, step 2 of the derivation).
_GOLDEN_EPOCH_LEN = 5


def _make_sized_samples(atom_counts: list[int]) -> list[dict[str, object]]:
    """Build samples with the exact per-sample atom counts given, RNG-free.

    Args:
        atom_counts: Per-sample atom count. The edge count is ``atoms + 2`` so
            the packed-cache schema inference can never confuse a per-atom key
            with a per-edge key.

    Returns:
        Flat sample dicts in the pipeline's raw-sample shape. Geometry is a
        deterministic ladder along x (no RNG anywhere); only the *counts* are
        read by the sampler, which never unpacks a sample.
    """
    samples: list[dict[str, object]] = []
    for i, na in enumerate(atom_counts):
        ne = na + 2
        pos = torch.zeros(na, 3)
        pos[:, 0] = torch.arange(na, dtype=torch.float32)
        source = torch.arange(ne, dtype=torch.long) % na
        target = (torch.arange(ne, dtype=torch.long) + 1) % na
        diff = pos[target] - pos[source]
        samples.append(
            {
                "Z": torch.ones(na, dtype=torch.long),
                "pos": pos,
                "edge_index": torch.stack([source, target], dim=1),
                "edge_diff": diff,
                "edge_dist": diff.norm(dim=-1),
                "targets": {"U0": torch.tensor([float(i)])},
            }
        )
    return samples


def _build_sampler(workdir: Path) -> TaskMixtureBatchSampler:
    """Write the fixture cache under *workdir* and bind a Stage B sampler to it."""
    sink = workdir / "task_fixture.pt"
    PackedCache(sink).save(_make_sized_samples(_ATOM_COUNTS))
    dataset = MmapDataset(sink)
    if len(dataset) != len(_ATOM_COUNTS):
        raise AssertionError(f"cache round-trip lost samples: {len(dataset)} vs 24")
    return TaskMixtureBatchSampler(
        dataset,
        indices=_INDICES,
        task_labels=_TASK_LABELS,
        group_ids=_GROUP_IDS,
        mixture=_STAGE_B,
        max_atoms=_MAX_ATOMS,
        seed=_SEED,
    )


def _check_golden_plan(sampler: TaskMixtureBatchSampler) -> None:
    """Assertion 1: the plan and its length match the hard-coded goldens."""
    plan = list(sampler)
    if plan != _GOLDEN_PLAN:
        raise AssertionError(f"plan drift: {plan} vs {_GOLDEN_PLAN}")
    if len(sampler) != _GOLDEN_EPOCH_LEN:
        raise AssertionError(f"epoch length drift: {len(sampler)} vs {_GOLDEN_EPOCH_LEN}")
    if list(sampler) != plan:
        raise AssertionError("iterating twice must yield the same plan")


def _check_audit_dicts(sampler: TaskMixtureBatchSampler) -> None:
    """Assertion 2: the audit accessors match the hard-coded goldens."""
    batches = sampler.batches_per_task()
    frames = sampler.frames_per_task()
    if batches != _GOLDEN_BATCHES_PER_TASK:
        raise AssertionError(f"batches_per_task drift: {batches} vs {_GOLDEN_BATCHES_PER_TASK}")
    if frames != _GOLDEN_FRAMES_PER_TASK:
        raise AssertionError(f"frames_per_task drift: {frames} vs {_GOLDEN_FRAMES_PER_TASK}")
    if sum(batches.values()) != len(sampler):
        raise AssertionError(f"batch shares {batches} must sum to len(sampler) = {len(sampler)}")
    batches["equilibrium"] = -1  # caller mutation must not corrupt the plan
    if sampler.batches_per_task() != _GOLDEN_BATCHES_PER_TASK:
        raise AssertionError("batches_per_task must return a fresh dict per call")


def _check_structure(sampler: TaskMixtureBatchSampler) -> None:
    """Assertion 3: single-task batches, budget compliance and scan integrity."""
    plan = list(sampler)

    flat = [i for batch in plan for i in batch]
    if len(flat) != len(set(flat)):
        raise AssertionError(f"an epoch must not repeat a frame: {flat}")
    if not set(flat) <= set(_INDICES):
        raise AssertionError(f"plan reached outside the split: {sorted(set(flat) - set(_INDICES))}")

    for batch in plan:
        if not batch:
            raise AssertionError("every batch must be non-empty")
        tasks = {_TASK_LABELS[i] for i in batch}
        if len(tasks) != 1:
            raise AssertionError(f"batch {batch} mixes tasks {sorted(tasks)}")
        total = sum(_ATOM_COUNTS[i] for i in batch)
        if total > _MAX_ATOMS and len(batch) > 1:
            raise AssertionError(f"batch {batch} sums to {total} > budget {_MAX_ATOMS}")

    # Scan integrity: a unit is (task label, group id); each unit that appears
    # at all appears whole, inside exactly one batch — a torsion scan is never
    # split across two batches and never partially sampled.
    units: dict[tuple[str, str], list[int]] = {}
    for i in _INDICES:
        units.setdefault((_TASK_LABELS[i], _GROUP_IDS[i]), []).append(i)
    for unit, members in units.items():
        holding = [b for b, batch in enumerate(plan) if set(members) & set(batch)]
        if not holding:
            continue  # unit not drawn this epoch — allowed, integrity is per draw
        if len(holding) != 1:
            raise AssertionError(f"unit {unit} straddles batches {holding}")
        batch = plan[holding[0]]
        if not set(members) <= set(batch):
            missing = sorted(set(members) - set(batch))
            raise AssertionError(f"unit {unit} sampled partially; missing frames {missing}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        sampler = _build_sampler(Path(tmp))
        _check_golden_plan(sampler)
        _check_audit_dicts(sampler)
        _check_structure(sampler)

    print(
        "universal-class-i-10-sampler: Stage B (0.35/0.45/0.20), max_atoms=24, "
        "seed=20260814 -> plan [[3,2,5,1],[15,16,17,18,19],[10,11,12],[13,14],[0,4]], "
        "batches_per_task {equilibrium:2, relax:1, torsion:2}, frames_per_task "
        "{equilibrium:6, relax:5, torsion:5}, every batch single-task, every drawn "
        "unit whole in exactly one batch OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
