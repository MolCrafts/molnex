"""Public-API regression for TokenBudgetPacker + the DataModule batch-sampler seam.

Spec: ``universal-class-i-09-packer`` (acceptance ac-010).

Two independent parts, both RNG-free and both hard-coded:

**Part A — packing rule.** :class:`~molix.data.TokenBudgetPacker` is handed one
hard-coded ``(budget, counts)`` pair and one hard-coded *explicit* visit order,
so the emitted plan is a pure function of the literals in this file — no
permutation, no seed, no dataset. The expected plan below was derived by hand on
2026-08-14 (no third-party oracle was imported, subprocessed, or consulted, here
or at capture time).

Count table (dimensionless atom counts) and budget ``12``::

    index  |  0   1   2   3    4   5
    atoms  |  7   6   5   20   4   8

Visit order ``[2, 0, 1, 3, 4, 5]`` under **next-fit** (exactly one open bin, a
closed bin is never reopened)::

    idx 2 (5)   open bin empty          -> bin = [2],    total  5
    idx 0 (7)   5 + 7 = 12 <= 12        -> bin = [2, 0], total 12   (exact fit)
    idx 1 (6)   12 + 6 = 18  > 12       -> close [2, 0]; bin = [1], total  6
    idx 3 (20)  6 + 20 = 26  > 12       -> close [1];    bin = [3], total 20
                20 > 12 (oversize)      -> close [3] immediately (singleton)
    idx 4 (4)   open bin empty          -> bin = [4],    total  4
    idx 5 (8)   4 + 8 = 12 <= 12        -> bin = [4, 5], total 12
    end         flush                   -> close [4, 5]

    => [[2, 0], [1], [3], [4, 5]]

This order is chosen to separate next-fit from first-fit-over-all-open-bins:
a first-fit policy would put idx 4 (4 atoms) back into the *closed* bin ``[1]``
(6 + 4 = 10 <= 12) and emit ``[[2, 0], [1, 4], [3], [5]]`` instead. Sample 3
(20 atoms) exceeds the budget on its own and must survive as a singleton batch,
never dropped — concatenating the plan reproduces the visit order exactly.

**Part B — injection seam.** A tiny :class:`~molix.data.PackedCache` is written
into a :class:`tempfile.TemporaryDirectory`, a module-level stub
``BatchSamplerSource`` is injected into ``DataModule(num_workers=0)``, and
``on_epoch_start(2)`` is called with the default ``seed=42``. The stub must be
handed ``seed == 42 + 2 == 44`` (the ``seed + epoch`` reseed convention that
makes epoch *k* re-derivable on resume), and the collated batches must carry
exactly the injected index groups — proven through the per-graph identity
``U0 == float(index)``, so both the seam and the packed collate path are
covered without any RNG.

    capture: PYTHONPATH=src python regressions/universal-class-i-09-packer.py
    date: 2026-08-14
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from molix.data import DataModule, TokenBudgetPacker
from molix.store import CachedDataset, PackedCache

# ---------------------------------------------------------------------------
# Part A — hard-coded packing goldens
# ---------------------------------------------------------------------------

#: Dimensionless per-sample atom counts, index-aligned (see module docstring).
_ATOM_COUNTS = [7, 6, 5, 20, 4, 8]
#: Per-batch cap on the summed atom count.
_MAX_ATOMS = 12
#: Explicit visit order — no permutation, no seed.
_ORDER = [2, 0, 1, 3, 4, 5]
#: Hand-derived next-fit plan (2026-08-14). First-fit would give
#: [[2, 0], [1, 4], [3], [5]].
_PLAN_GOLDEN = [[2, 0], [1], [3], [4, 5]]
#: Only sample 3 (20 atoms) exceeds the budget on its own.
_OVERSIZE_GOLDEN = [False, False, False, True, False, False]

# ---------------------------------------------------------------------------
# Part B — injection goldens
# ---------------------------------------------------------------------------

#: Index groups the stub source hands the train loader; covers the 6-sample
#: train split exactly once, in a non-sorted order so the yielded identities
#: cannot match by accident.
_INJECTED_PLAN = [[3, 0], [5], [1, 4, 2]]
#: ``U0`` identities those groups must produce (``U0 == float(index)``).
_INJECTED_IDENTITIES = [[3.0, 0.0], [5.0], [1.0, 4.0, 2.0]]
#: ``DataModule`` default seed plus the epoch passed to ``on_epoch_start``.
_EPOCH = 2
_SEED_GOLDEN = 44

_N_TRAIN = 6
_N_VAL = 2


class _FixedPlanSource:
    """``BatchSamplerSource`` stub returning one fixed, hard-coded plan.

    Records every ``seed`` it is handed so the ``seed + epoch`` convention is
    observable, and returns the *same* list object every call so the loader's
    ``batch_sampler`` can be checked by identity.
    """

    def __init__(self, plan: list[list[int]]) -> None:
        self.plan = [list(group) for group in plan]
        self.seen_seeds: list[int] = []

    def batch_sampler(self, dataset: object, *, seed: int) -> list[list[int]]:
        """Record *seed* and return the fixed index plan."""
        del dataset
        self.seen_seeds.append(seed)
        return self.plan


def _make_samples(n: int) -> list[dict]:
    """Build *n* two-atom samples with fully deterministic geometry.

    Args:
        n: Number of samples. ``targets.U0`` carries the identity
            ``float(i)`` read back from the collated batch.

    Returns:
        Flat sample dicts with per-atom (``Z``, ``pos``), per-edge
        (``edge_index``, ``edge_diff``, ``edge_dist``) and graph-level
        (``targets.U0``) keys. Separations vary with ``i`` but no RNG is
        involved anywhere.
    """
    samples: list[dict] = []
    for i in range(n):
        sep = 1.0 + 0.1 * i  # A
        pos = torch.tensor([[0.0, 0.0, 0.0], [sep, 0.0, 0.0]])
        # edge_index [[0, 1]] -> source 0, target 1; edge_diff follows the
        # repo convention pos[target] - pos[source].
        diff = (pos[1] - pos[0]).unsqueeze(0)
        samples.append(
            {
                "Z": torch.tensor([1, 6], dtype=torch.long),
                "pos": pos,
                "edge_index": torch.tensor([[0, 1]], dtype=torch.long),
                "edge_diff": diff,
                "edge_dist": diff.norm(dim=-1),
                "targets": {"U0": torch.tensor([float(i)])},
            }
        )
    return samples


def _check_packer() -> None:
    """Part A: the packing rule reproduces the hand-derived plan exactly."""
    packer = TokenBudgetPacker([(_MAX_ATOMS, torch.tensor(_ATOM_COUNTS, dtype=torch.long))])

    mask = packer.oversize_mask()
    if mask.dtype is not torch.bool or tuple(mask.shape) != (len(_ATOM_COUNTS),):
        raise AssertionError(
            f"oversize_mask must be bool (n_samples,); got {mask.dtype} {tuple(mask.shape)}"
        )
    if mask.tolist() != _OVERSIZE_GOLDEN:
        raise AssertionError(f"oversize mask drift: {mask.tolist()} vs {_OVERSIZE_GOLDEN}")

    plan = packer.pack(_ORDER)
    if plan != _PLAN_GOLDEN:
        raise AssertionError(
            f"next-fit plan drift: {plan} vs {_PLAN_GOLDEN} "
            "(first-fit over all open bins would give [[2, 0], [1, 4], [3], [5]])"
        )

    flat = [idx for batch in plan for idx in batch]
    if flat != _ORDER:
        raise AssertionError(f"packing must partition the visit order: {flat} vs {_ORDER}")

    for batch in plan:
        total = sum(_ATOM_COUNTS[idx] for idx in batch)
        if total > _MAX_ATOMS and len(batch) > 1:
            raise AssertionError(f"batch {batch} sums to {total} > budget {_MAX_ATOMS}")

    if packer.pack([]) != []:
        raise AssertionError(f"empty order must pack to []; got {packer.pack([])}")


def _check_injection_seam(workdir: Path) -> None:
    """Part B: the injected source drives the train loader and sees ``seed + epoch``."""
    samples = _make_samples(_N_TRAIN + _N_VAL)
    train_sink = workdir / "train.pt"
    val_sink = workdir / "val.pt"
    PackedCache(train_sink).save(samples[:_N_TRAIN])
    PackedCache(val_sink).save(samples[_N_TRAIN:])

    source = _FixedPlanSource(_INJECTED_PLAN)
    dm = DataModule(
        CachedDataset(train_sink),
        CachedDataset(val_sink),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        batch_sampler_source=source,
    )
    if dm.seed != 42:
        raise AssertionError(f"DataModule default seed must be 42; got {dm.seed}")

    dm.on_epoch_start(_EPOCH)
    loader = dm.train_dataloader()

    if source.seen_seeds != [_SEED_GOLDEN]:
        raise AssertionError(
            f"source must be called once with seed {_SEED_GOLDEN} "
            f"(= {dm.seed} + {_EPOCH}); saw {source.seen_seeds}"
        )
    if loader.batch_sampler is not source.plan:
        raise AssertionError("the injected plan must be handed straight to the DataLoader")
    if loader.batch_size is not None:
        raise AssertionError(
            f"batch_size must be None on the injected path; got {loader.batch_size}"
        )

    yielded = [[float(u) for u in batch["graphs", "U0"]] for batch in loader]
    if yielded != _INJECTED_IDENTITIES:
        raise AssertionError(f"collated groups drift: {yielded} vs {_INJECTED_IDENTITIES}")


def main() -> int:
    _check_packer()
    with tempfile.TemporaryDirectory() as tmp:
        _check_injection_seam(Path(tmp))

    print(
        "universal-class-i-09-packer: pack([2,0,1,3,4,5]) = [[2,0],[1],[3],[4,5]] "
        "(next-fit closure + oversize singleton [3]), oversize_mask = "
        "[F,F,F,T,F,F], injected source saw seed 44 after on_epoch_start(2) and "
        "yielded U0 groups [[3,0],[5],[1,4,2]] OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
