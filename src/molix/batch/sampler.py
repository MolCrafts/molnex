"""Token-budget batch sampling over :class:`~molix.store.cache.PackedCache` pointers.

Molecular samples vary widely in size (QM9 spans 3–29 atoms), so a fixed
``batch_size`` makes per-batch compute and memory swing with batch
composition — GNN encoder cost scales with the total edge count O(E), not
the sample count O(B). :class:`TokenBudgetBatchSampler` instead packs each
batch next-fit (single open bin) under an atom and/or edge budget, keeping
per-batch work approximately constant. Per-sample sizes come from the packed
cache's cumsum pointers (``ptr[idx+1] - ptr[idx]``) — no sample is ever
unpacked during sampling.

Four layers, deliberately separated:

* :class:`TokenBudgetPacker` — the packing rule alone (visit order in,
  batch plan out); no dataset, no RNG, no warning policy.
* :class:`TokenBudgetBatchSampler` — seeded shuffle + oversize warning on
  top of the packer, over a packed-cache-backed dataset.
* :class:`TaskMixtureBatchSampler` — hierarchical task → group →
  configuration sampling: one physical task per batch drawn from a caller
  supplied mixture, whole sampling units inside it, packed by the same
  packer under the same token budgets.
* :class:`BatchSamplerSource` — the protocol a
  :class:`~molix.batch.datamodule.DataModule` calls once per epoch to obtain
  a batch sampler, so alternative batching policies can be injected.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch

from molix.core.seed import make_generator
from molix.store.dataset import BaseDataset

__all__ = [
    "BatchSamplerSource",
    "TaskMixtureBatchSampler",
    "TokenBudgetBatchSampler",
    "TokenBudgetPacker",
]


class TokenBudgetPacker:
    """Group a visit order into index batches that stay under budget.

    Next-fit (single open bin) packing: indices are consumed in the given
    order and accumulated into exactly one open batch, which is closed as
    soon as adding the next index would push any budget over its cap. A
    closed batch is never reopened, so the plan is a pure function of
    ``(counts, budgets, visit order)``: counts ``[7, 6, 5]`` under budget
    ``12`` pack as ``[[0], [1, 2]]``, never ``[[0, 2], [1]]`` — the latter
    is what a policy that kept several bins open would return.

    An index whose own count exceeds a budget cannot fit anywhere: it is
    emitted as a singleton batch — never dropped — and closes a non-empty
    open batch first rather than merging into it. The packer stays silent
    about such indices; query :meth:`oversize_mask` and decide (warn,
    drop, route elsewhere) in the caller. Owning no dataset, no RNG and no
    warning policy is what lets one packer serve both
    :class:`TokenBudgetBatchSampler` and any other sampler that already
    knows its own visit order.

    Args:
        budgets: One ``(budget, counts)`` pair per active budget, where
            ``counts`` is an integer ``(n_samples,)`` vector of
            per-sample sizes and ``budget`` caps their per-batch sum. Both
            are dimensionless counts (atoms, edges, …) in the same
            counting unit — no physical units are involved. Every vector
            is materialised once as Python ints, so a later mutation (or a
            closed mmap) cannot change the plan.

    Raises:
        ValueError: *budgets* is empty — at least one ``(budget, counts)``
            pair is required, since packing without a cap is undefined.
    """

    def __init__(self, budgets: Sequence[tuple[int, torch.Tensor]]) -> None:
        if not budgets:
            raise ValueError(
                "TokenBudgetPacker needs at least one (budget, counts) pair, "
                "e.g. [(max_atoms, atom_counts)] with counts an integer "
                "(n_samples,) vector; got an empty budgets sequence."
            )
        self._budgets = [budget for budget, _ in budgets]
        self._counts = [[int(v) for v in counts.tolist()] for _, counts in budgets]

    def oversize_mask(self) -> torch.Tensor:
        """Flag samples whose own count exceeds a budget.

        Returns:
            Bool tensor ``(n_samples,)`` — the elementwise OR over budgets
            of ``counts > budget``. It covers the **full** count vectors
            given at construction, so a caller that packs only a subset of
            the samples must index this mask by that subset.
        """
        mask = torch.zeros(len(self._counts[0]), dtype=torch.bool)
        for budget, counts in zip(self._budgets, self._counts):
            mask |= torch.tensor(counts, dtype=torch.long) > budget
        return mask

    def pack(self, order: Sequence[int]) -> list[list[int]]:
        """Pack *order* into consecutive batches under every active budget.

        Args:
            order: Sample indices in the order they should be visited
                (e.g. a shuffled permutation). Indices address the
                ``(n_samples,)`` count vectors given at construction.

        Returns:
            The batch plan as ``list[list[int]]``. Concatenating it
            reproduces *order* exactly — packing is a partition, nothing is
            dropped or duplicated, including oversize singletons. An empty
            *order* yields ``[]``.
        """
        batches: list[list[int]] = []
        current: list[int] = []
        totals = [0] * len(self._budgets)

        for idx in order:
            sizes = [counts[idx] for counts in self._counts]
            oversize = any(s > budget for s, budget in zip(sizes, self._budgets))
            fits = all(t + s <= budget for t, s, budget in zip(totals, sizes, self._budgets))
            if current and not fits:
                batches.append(current)
                current = []
                totals = [0] * len(self._budgets)
            current.append(idx)
            totals = [t + s for t, s in zip(totals, sizes)]
            if oversize:
                batches.append(current)
                current = []
                totals = [0] * len(self._budgets)
        if current:
            batches.append(current)
        return batches


class TokenBudgetBatchSampler:
    """Yield index batches whose total atom/edge counts stay under budget.

    Next-fit (single open bin) packing in seeded-shuffle order: samples
    are visited in a ``randperm`` permutation drawn from
    :func:`~molix.core.seed.make_generator` and handed to a
    :class:`TokenBudgetPacker`, which closes the open batch whenever
    adding the next sample would exceed any active budget. A sample whose
    own count exceeds the budget forms a singleton batch — it is never
    dropped — and one :class:`UserWarning` is emitted per sampler
    instance when the dataset contains any such sample.

    The batch list is computed once in ``__init__`` (O(n) pointer
    arithmetic) and cached, so ``__iter__`` / ``__len__`` are free.
    ``len(sampler)`` is the batch count for *this* permutation — it varies
    with ``seed``, so different epochs may have different lengths. Pass a
    per-epoch seed (e.g. ``seed + epoch``) to reshuffle between epochs
    while keeping epoch *k*'s composition re-derivable on resume.

    The instance holds only the dataset reference, integer budgets, the
    seed, and the computed ``list[list[int]]`` — all picklable. Note that
    ``DataLoader`` consumes a ``batch_sampler`` in the main process;
    workers receive index lists, so the sampler itself never crosses a
    process boundary unless the whole DataModule is shipped.

    Args:
        dataset: A packed-cache-backed dataset exposing ``atom_counts`` /
            ``edge_counts`` ``(n_samples,)`` count vectors —
            :class:`~molix.store.dataset.MmapDataset`,
            :class:`~molix.store.dataset.CachedDataset`, or a
            :class:`~molix.store.dataset.SubsetDataset` wrapping either.
        max_atoms: Maximum total atom count per batch, or ``None``.
        max_edges: Maximum total edge count per batch, or ``None``.
            At least one budget is required; both apply simultaneously
            when given.
        seed: Seed for the shuffle permutation.

    Raises:
        ValueError: No budget given, a budget is ``<= 0``, or *dataset*
            does not expose the pointer-derived count accessors.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        *,
        max_atoms: int | None = None,
        max_edges: int | None = None,
        seed: int = 42,
    ) -> None:
        if max_atoms is None and max_edges is None:
            raise ValueError(
                "TokenBudgetBatchSampler needs at least one budget: pass "
                "max_atoms and/or max_edges (total per-batch atom/edge caps)."
            )
        for name, value in (("max_atoms", max_atoms), ("max_edges", max_edges)):
            if value is not None and value <= 0:
                raise ValueError(
                    f"{name} must be > 0, got {value}. Pass a positive "
                    f"per-batch budget or omit {name} to disable that cap."
                )

        self._dataset = dataset
        self._max_atoms = max_atoms
        self._max_edges = max_edges
        self._seed = seed
        self._batches = self._compute_batches()

    def _budget_counts(self) -> list[tuple[int, torch.Tensor]]:
        """Resolve ``(budget, counts)`` pairs for every active budget."""
        pairs: list[tuple[int, torch.Tensor]] = []
        try:
            if self._max_atoms is not None:
                pairs.append((self._max_atoms, self._dataset.atom_counts))
            if self._max_edges is not None:
                pairs.append((self._max_edges, self._dataset.edge_counts))
        except AttributeError as e:
            raise ValueError(
                "TokenBudgetBatchSampler requires a packed-cache-backed "
                "dataset exposing atom_counts/edge_counts — use MmapDataset, "
                "CachedDataset, or a SubsetDataset wrapping one of them; got "
                f"{type(self._dataset).__name__}."
            ) from e
        return pairs

    def _compute_batches(self) -> list[list[int]]:
        """Next-fit (single open bin) packing of the shuffled indices under budget.

        Builds a transient :class:`TokenBudgetPacker` over the active
        ``(budget, counts)`` pairs — transient so the instance state stays
        the five picklable fields and never holds mmap-backed count
        vectors — warns once if any sample is oversize, and packs the
        seeded permutation.

        Returns:
            The batch plan for this ``seed`` as ``list[list[int]]``.
        """
        packer = TokenBudgetPacker(self._budget_counts())
        if bool(packer.oversize_mask().any()):
            warnings.warn(
                "TokenBudgetBatchSampler: at least one sample exceeds "
                "the per-batch budget on its own; such samples are kept "
                "as singleton batches (never dropped). Raise the budget "
                "to avoid singletons.",
                UserWarning,
                stacklevel=3,
            )
        n = len(self._dataset)
        perm = torch.randperm(n, generator=make_generator(self._seed)).tolist()
        return packer.pack(perm)

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate the cached batches, each a ``list[int]`` of sample indices."""
        return iter(self._batches)

    def __len__(self) -> int:
        """Number of batches for this permutation.

        Varies with ``seed`` (and therefore epoch when seeded as
        ``seed + epoch``), since next-fit packing of a different shuffle
        order can produce a different batch count.
        """
        return len(self._batches)


class TaskMixtureBatchSampler:
    """Yield single-task index batches drawn from a task mixture.

    A universal Class-I training collection mixes physically different
    data — equilibrium snapshots, torsion scans, near-equilibrium
    relaxations, electrostatic probes — whose natural frame counts are
    wildly unbalanced, and a torsion scan only carries its torsion
    profile when its frames are compared *within one batch*. This sampler
    therefore batches hierarchically: **task → group → configuration**.

    1. Frames of the split are bucketed into **sampling units** keyed
       ``(task_labels[i], group_ids[i])``, in first-appearance order.
    2. Units of one task are shuffled and handed to a transient
       :class:`TokenBudgetPacker` over their *aggregate* atom/edge costs,
       so a unit is packed whole — a scan never straddles two batches, and
       a unit larger than the budget becomes a whole singleton batch
       (warned about once, never split, never dropped).
    3. The epoch's slots are apportioned across tasks by the mixture and
       interleaved, so every batch carries exactly one task label. The
       flat frame pool is structurally unreachable.

    **There is no published precedent for a task → group → configuration
    hierarchical batch sampler**, so there is no reference implementation
    to validate against: correctness here is *structural* (mixture
    fidelity, unit integrity, determinism), while the scientific claim —
    that per-task gradient contributions track the configured mixture —
    is measured downstream by the gradient audit. Two published results
    are cited for the pieces this sampler borrows, and for nothing more:

    * **Stage structure only** — the default Stage A/B/C mixtures below
      are constructor *data* derived from ByteFF's three-stage curriculum
      over increasingly diverse PES regions; the mechanism (per-batch
      mixture draw, unit integrity, token packing) is not from ByteFF.
    * **Grouping discipline only** — espaloma splits at the molecule
      level so conformers never straddle a split; this sampler inherits
      that discipline one level down, making the group id the atomic unit
      of *sampling* exactly as it is the atomic unit of *splitting* in
      :func:`~molix.batch.group_split.group_split_indices`.

    Stage mixtures (plain ``dict`` constructor data — no ``Stage`` enum,
    no schedule object; choosing the active stage at epoch *e* stays with
    the caller):

    ===== =========== ======= ===== =============
    Stage equilibrium torsion relax electrostatic
    ===== =========== ======= ===== =============
    A     0.70        —       0.30  —
    B     0.35        0.45    0.20  —
    C     0.25        0.30    0.35  0.10
    ===== =========== ======= ===== =============

    **Stratified, not i.i.d.** Slots are apportioned by deterministic
    largest-remainder (Hamilton) rather than drawn per batch from a
    multinomial: ``base_t = floor(p_t · n)``, and the ``n − Σ base_t``
    leftover slots go to the largest fractional remainders, ties broken by
    ascending task name. An **anti-starvation** rule then guarantees that
    every task with ``p_t > 0`` and at least one available batch receives
    at least one slot; each such deficit is donated by the task with the
    largest surplus ``count_t − p_t·n`` (ties: larger count first, then
    ascending name), taken only from tasks that can spare it (``count ≥
    2``). The bound is therefore hard rather than statistical:
    ``|count_t − p_t·n| ≤ 1`` for every task when no override fires, and
    ``≤ 1 + K`` for donors when ``K`` rare tasks were rescued. Same
    expected mixture as a per-batch draw, zero sampling variance, exactly
    reproducible audit numbers.

    **The mixture governs batch shares only.** Frame shares follow from
    unit sizes and token packing; :meth:`frames_per_task` reports them for
    audit but does not guarantee them. Comparing either share against a
    calibration tolerance is the caller's job — this class ships no
    verdict façade.

    **Determinism.** Every random choice comes from one
    :func:`~molix.core.seed.make_generator` seeded with ``seed``: tasks
    are visited in ``sorted(mixture)`` order (independent of ``dict``
    insertion order), each drawing one unit permutation, and the slot
    order is one final permutation of the apportioned task multiset. The
    whole plan is built eagerly in ``__init__``, so ``len()`` and
    iteration are free and repeatable. The instance holds only the
    dataset reference, the budgets, the two label tuples, the mixture and
    the computed ``list[list[int]]`` — no packer and no mmap-backed count
    vector — so it pickles to an identical plan. Reseed per epoch with
    ``seed + epoch`` (the convention
    :class:`~molix.batch.datamodule.DataModule` already uses) to reshuffle
    between epochs while keeping epoch *k* re-derivable on resume.

    Args:
        dataset: A packed-cache-backed dataset exposing ``atom_counts`` /
            ``edge_counts`` ``(n_samples,)`` count vectors —
            :class:`~molix.store.dataset.MmapDataset`,
            :class:`~molix.store.dataset.CachedDataset`, or a
            :class:`~molix.store.dataset.SubsetDataset` wrapping either.
            Unit costs are pointer arithmetic over those vectors; no
            sample is ever unpacked.
        indices: The split's dataset indices, e.g. the train list from
            :func:`~molix.batch.group_split.group_split_indices`. Consumed
            as given — this sampler never splits and never re-derives
            groups. Yielded batches contain these parent-space indices.
        task_labels: Per-sample physical task label, ``len(dataset)``
            long (parent space, the shape
            :func:`~molix.batch.group_split.group_split_indices` already
            accepts). Labels are opaque strings; molnex never infers them.
        group_ids: Per-sample group identity, ``len(dataset)`` long. Two
            scans of one molecule sharing a label *and* a group id merge
            into one (coarser, still integral) unit — pass a finer id such
            as ``"<mol>:<dihedral>"`` to keep them separate. A molecule
            contributing to several tasks yields one unit per task, which
            is normal: cross-split leakage was already prevented upstream.
        mixture: Task → probability, summing to 1 within ``1e-9``. A label
            present in the split must appear here; give it ``0.0`` to
            exclude it deliberately.
        max_atoms: Maximum total atom count per batch, or ``None``.
        max_edges: Maximum total edge count per batch, or ``None``. At
            least one budget is required; both apply simultaneously when
            given.
        batches_per_epoch: Force the epoch length. Default ``None`` picks
            the longest epoch in which no task is exhausted,
            ``min over {t : p_t > 0} of floor(len(batches_t) / p_t)``.
            A larger forced value can exhaust a task: it is then dropped,
            the remaining probabilities are renormalized, one
            :class:`UserWarning` fires, and the plan ends at its realized
            length.
        seed: Seed for the unit permutations and the slot order.

    Raises:
        ValueError: A budget is missing or ``<= 0``,
            ``batches_per_epoch <= 0``, *mixture* is empty / does not sum
            to 1 within ``1e-9`` / carries a probability outside
            ``[0, 1]``, a label of the split is absent from *mixture*, a
            task with ``p_t > 0`` has no unit in the split, a label table
            length differs from ``len(dataset)``, *indices* is empty or
            addresses outside the dataset, or *dataset* does not expose
            the pointer-derived count accessors.

    Reference:
        Zheng et al. "ByteFF: A Machine Learning Force Field for
        Drug-like Molecules" arXiv:2408.12817 (2024) — stage structure
        only. https://arxiv.org/abs/2408.12817

        Wang, Fass, Chodera. "End-to-End Differentiable Construction of
        Molecular Mechanics Force Fields" Chem. Sci. 13, 12016 (2022) —
        grouping discipline only. DOI:10.1039/D2SC02739A, https://doi.org/10.1039/D2SC02739A
    """

    def __init__(
        self,
        dataset: BaseDataset,
        *,
        indices: Sequence[int],
        task_labels: Sequence[str],
        group_ids: Sequence[str | int],
        mixture: Mapping[str, float],
        max_atoms: int | None = None,
        max_edges: int | None = None,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        self._validate_sizes(max_atoms, max_edges, batches_per_epoch)
        self._validate_mixture(mixture)
        self._validate_tables(len(dataset), indices, task_labels, group_ids)

        self._dataset = dataset
        self._max_atoms = max_atoms
        self._max_edges = max_edges
        self._task_labels = tuple(task_labels)
        self._group_ids = tuple(group_ids)
        self._mixture = dict(mixture)
        self._batches_per_epoch = batches_per_epoch
        self._seed = seed
        self._batches = self._compute_batches(tuple(indices))

    @staticmethod
    def _validate_sizes(
        max_atoms: int | None, max_edges: int | None, batches_per_epoch: int | None
    ) -> None:
        """Reject missing / non-positive budget and epoch-length knobs."""
        if max_atoms is None and max_edges is None:
            raise ValueError(
                "TaskMixtureBatchSampler needs at least one budget: pass "
                "max_atoms and/or max_edges (total per-batch atom/edge caps)."
            )
        for name, value in (("max_atoms", max_atoms), ("max_edges", max_edges)):
            if value is not None and value <= 0:
                raise ValueError(
                    f"{name} must be > 0, got {value}. Pass a positive "
                    f"per-batch budget or omit {name} to disable that cap."
                )
        if batches_per_epoch is not None and batches_per_epoch <= 0:
            raise ValueError(
                f"batches_per_epoch must be > 0, got {batches_per_epoch}. "
                "Pass a positive epoch length or omit it to use the "
                "no-exhaustion default."
            )

    @staticmethod
    def _validate_mixture(mixture: Mapping[str, float]) -> None:
        """Reject an empty, out-of-range or non-normalized *mixture*."""
        if not mixture:
            raise ValueError(
                "mixture is empty: pass at least one task -> probability "
                "entry, e.g. {'equilibrium': 0.7, 'relax': 0.3}."
            )
        total = math.fsum(mixture.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"mixture probabilities sum to {total}, not 1.0 (tolerance "
                f"1e-9); got {dict(mixture)}. Rescale so probabilities sum "
                "to 1.0."
            )
        for task, probability in sorted(mixture.items()):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"mixture probability for task {task!r} is {probability}, "
                    "outside the valid range [0.0, 1.0]. Pass a probability "
                    "in [0.0, 1.0], using 0.0 to exclude the task."
                )

    @staticmethod
    def _validate_tables(
        n_samples: int,
        indices: Sequence[int],
        task_labels: Sequence[str],
        group_ids: Sequence[str | int],
    ) -> None:
        """Reject mis-sized label tables and empty / out-of-range *indices*."""
        for name, table in (("task_labels", task_labels), ("group_ids", group_ids)):
            if len(table) != n_samples:
                raise ValueError(
                    f"len({name}) is {len(table)} but len(dataset) is "
                    f"{n_samples}: {name} is a parent-space per-sample "
                    f"sequence, so fix the {name} accessor that produced it."
                )
        if len(indices) == 0:
            raise ValueError(
                "indices is empty: pass the split's dataset indices (e.g. the "
                "train list of group_split_indices); no batch can be formed "
                "from an empty split."
            )
        for index in indices:
            if not 0 <= index < n_samples:
                raise ValueError(
                    f"indices contains {index}, outside the valid range "
                    f"[0, {n_samples}) for this dataset. Pass split indices "
                    "addressing the dataset given to this sampler."
                )

    def _validate_tasks(self, task_units: dict[str, list[tuple[str | int, list[int]]]]) -> None:
        """Reject a split label missing from the mixture, or a supply-less task."""
        for task in sorted(task_units):
            if task not in self._mixture:
                n_frames = sum(len(frames) for _, frames in task_units[task])
                raise ValueError(
                    f"task_labels carries {n_frames} sample(s) labelled "
                    f"{task!r} in the split, but {task!r} is absent from "
                    "mixture. Add it with probability 0.0 to exclude it "
                    "deliberately."
                )
        for task, probability in sorted(self._mixture.items()):
            if probability > 0 and not task_units.get(task):
                raise ValueError(
                    f"mixture task {task!r} has probability {probability} but "
                    "zero sampling units in the split. Set its probability to "
                    "0.0 or widen the split."
                )

    def _frame_budgets(self) -> list[tuple[int, torch.Tensor]]:
        """Resolve ``(budget, per-frame counts)`` pairs for every active budget."""
        pairs: list[tuple[int, torch.Tensor]] = []
        try:
            if self._max_atoms is not None:
                pairs.append((self._max_atoms, self._dataset.atom_counts))
            if self._max_edges is not None:
                pairs.append((self._max_edges, self._dataset.edge_counts))
        except AttributeError as e:
            raise ValueError(
                "TaskMixtureBatchSampler requires a packed-cache-backed "
                "dataset exposing atom_counts/edge_counts — use MmapDataset, "
                "CachedDataset, or a SubsetDataset wrapping one of them; got "
                f"{type(self._dataset).__name__}."
            ) from e
        return pairs

    def _bucket_units(
        self, indices: tuple[int, ...]
    ) -> dict[str, list[tuple[str | int, list[int]]]]:
        """Bucket *indices* into sampling units keyed ``(task_label, group_id)``.

        Args:
            indices: The split's dataset indices, walked in the given order.

        Returns:
            Task label → its units as ``(group_id, member frame indices)``
            pairs, both levels in first-appearance order.
        """
        unit_frames: dict[tuple[str, str | int], list[int]] = {}
        for index in indices:
            key = (self._task_labels[index], self._group_ids[index])
            unit_frames.setdefault(key, []).append(index)

        task_units: dict[str, list[tuple[str | int, list[int]]]] = {}
        for (task, group), frames in unit_frames.items():
            task_units.setdefault(task, []).append((group, frames))
        return task_units

    def _pack_tasks(
        self,
        task_units: dict[str, list[tuple[str | int, list[int]]]],
        generator: torch.Generator,
    ) -> tuple[dict[str, list[list[int]]], list[str]]:
        """Shuffle and pack each task's units into frame-index batches.

        One transient :class:`TokenBudgetPacker` per task over that task's
        *aggregate* unit costs — transient so no mmap-backed count vector
        reaches the instance state, and per task because the packer binds
        its cost vectors at construction. Packed unit positions are
        expanded into their member frame indices, which is what makes unit
        integrity structural rather than merely checked.

        Args:
            task_units: Output of :meth:`_bucket_units`.
            generator: The single seeded generator; tasks are visited in
                ``sorted(mixture)`` order and a task with no unit in the
                split draws nothing.

        Returns:
            ``(task_batches, oversize)`` — packed frame-index batches per
            task, and the ``"<task>/<group>"`` names of units whose own
            aggregate cost exceeds a budget (the packer is silent about
            them by design; warning is this caller's job).
        """
        frame_budgets = self._frame_budgets()
        task_batches: dict[str, list[list[int]]] = {}
        oversize: list[str] = []
        for task in sorted(self._mixture):
            units = task_units.get(task)
            if not units:
                continue
            packer = TokenBudgetPacker(
                [
                    (budget, torch.tensor([int(counts[frames].sum()) for _, frames in units]))
                    for budget, counts in frame_budgets
                ]
            )
            oversize.extend(
                f"{task}/{units[position][0]}"
                for position, flagged in enumerate(packer.oversize_mask().tolist())
                if flagged
            )
            order = torch.randperm(len(units), generator=generator).tolist()
            task_batches[task] = [
                [index for position in group for index in units[position][1]]
                for group in packer.pack(order)
            ]
        return task_batches, oversize

    def _epoch_length(self, task_batches: dict[str, list[list[int]]]) -> int:
        """Number of slots to plan: the forced length, or the no-exhaustion floor.

        Args:
            task_batches: Packed batches per task, from :meth:`_pack_tasks`.

        Returns:
            ``batches_per_epoch`` when given, else ``min over {t : p_t > 0}
            of floor(len(batches_t) / p_t)`` — the longest epoch in which
            no task runs out. Since every task with ``p_t > 0`` has at
            least one unit (validated) and ``p_t <= 1``, this is ``>= 1``.
        """
        if self._batches_per_epoch is not None:
            return self._batches_per_epoch
        return min(
            math.floor(len(task_batches[task]) / probability)
            for task, probability in self._mixture.items()
            if probability > 0
        )

    def _apportion(self, n: int, task_batches: dict[str, list[list[int]]]) -> dict[str, int]:
        """Split *n* slots across tasks by largest remainder, then rescue rare tasks.

        Args:
            n: Epoch length in batches.
            task_batches: Packed batches per task, used only to tell a
                supply-less task (never rescued) from a starved one.

        Returns:
            Task → slot count, summing to *n*.
        """
        counts = {task: math.floor(probability * n) for task, probability in self._mixture.items()}
        remainders = {
            task: probability * n - counts[task] for task, probability in self._mixture.items()
        }
        leftover = n - sum(counts.values())
        for task in sorted(counts, key=lambda name: (-remainders[name], name))[:leftover]:
            counts[task] += 1

        for task in sorted(counts):
            starved = self._mixture[task] > 0 and counts[task] == 0
            if not starved or not task_batches.get(task):
                continue
            donors = [name for name, count in counts.items() if count >= 2]
            if not donors:
                continue
            donor = min(
                donors,
                key=lambda name: (-(counts[name] - self._mixture[name] * n), -counts[name], name),
            )
            counts[donor] -= 1
            counts[task] += 1
        return counts

    def _interleave(
        self,
        counts: dict[str, int],
        n: int,
        task_batches: dict[str, list[list[int]]],
        generator: torch.Generator,
    ) -> tuple[list[list[int]], tuple[str, int] | None]:
        """Order the apportioned slots and fill each from its task's supply.

        The slot labels are the multiset ``{t repeated counts[t]}`` in
        sorted task order, permuted once by the shared generator, so
        mixture fidelity holds per window rather than only per epoch.

        Args:
            counts: Apportioned slot count per task.
            n: Epoch length in batches (``sum(counts.values())``).
            task_batches: Packed batches per task, consumed in order.
            generator: The single seeded generator.

        Returns:
            ``(plan, shortfall)`` — the epoch's frame-index batches, and
            ``(task, slot)`` of the first slot whose task ran out of
            supply (``None`` when every slot was filled).
        """
        multiset: list[str] = []
        for task in sorted(counts):
            multiset.extend([task] * counts[task])

        cursors = dict.fromkeys(counts, 0)
        plan: list[list[int]] = []
        shortfall: tuple[str, int] | None = None
        for slot, position in enumerate(torch.randperm(n, generator=generator).tolist()):
            task = multiset[position]
            available = task_batches.get(task, [])
            if cursors[task] >= len(available):
                if shortfall is None:
                    shortfall = (task, slot)
                continue
            plan.append(available[cursors[task]])
            cursors[task] += 1
        return plan, shortfall

    def _compute_batches(self, indices: tuple[int, ...]) -> list[list[int]]:
        """Build the whole epoch plan eagerly, and own the warning policy.

        Args:
            indices: The split's dataset indices.

        Returns:
            The epoch's batches as ``list[list[int]]`` of parent-space
            dataset indices, every batch single-task by construction.
        """
        task_units = self._bucket_units(indices)
        self._validate_tasks(task_units)

        generator = make_generator(self._seed)
        task_batches, oversize = self._pack_tasks(task_units, generator)
        if oversize:
            warnings.warn(
                "TaskMixtureBatchSampler: sampling unit(s) "
                f"{', '.join(oversize)} (task/group) exceed the per-batch "
                "budget on their own; each is kept whole as a singleton "
                "batch — never split, never dropped. Raise max_atoms/"
                "max_edges to avoid singletons.",
                UserWarning,
                stacklevel=3,
            )

        n = self._epoch_length(task_batches)
        plan, shortfall = self._interleave(
            self._apportion(n, task_batches), n, task_batches, generator
        )
        if shortfall is not None:
            task, slot = shortfall
            warnings.warn(
                f"TaskMixtureBatchSampler: task {task!r} ran out of packed "
                f"batches at slot {slot} of {n}; its remaining slots are "
                "dropped, the mixture is renormalized over the surviving "
                "tasks, and len(sampler) reports the realized length. Lower "
                "batches_per_epoch (or omit it) to keep every task in supply.",
                UserWarning,
                stacklevel=3,
            )
        return plan

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate the eager plan, each item a ``list[int]`` of dataset indices."""
        return iter(self._batches)

    def __len__(self) -> int:
        """Realized number of batches in the epoch.

        Equals ``batches_per_epoch`` when that forced length was in
        supply, the no-exhaustion floor when it was not given, and the
        shorter realized length when a forced length exhausted a task.
        """
        return len(self._batches)

    def batches_per_task(self) -> dict[str, int]:
        """Count the planned batches carried by each task.

        Free (the plan is eager) and read-only: the returned ``dict`` is a
        fresh copy, so caller mutation cannot corrupt the plan. This is
        the realized batch share — the quantity the mixture governs.

        Returns:
            Task label → number of batches in this epoch's plan; tasks
            with no realized batch are absent.
        """
        counts: dict[str, int] = {}
        for batch in self._batches:
            task = self._task_labels[batch[0]]
            counts[task] = counts.get(task, 0) + 1
        return counts

    def frames_per_task(self) -> dict[str, int]:
        """Count the planned frames carried by each task.

        Frame shares follow from unit sizes and token packing; they are
        reported for audit, **not** guaranteed by the mixture (which
        governs batch shares only). Comparing them against a calibration
        tolerance is the caller's job.

        Returns:
            Task label → number of frames in this epoch's plan; tasks with
            no realized batch are absent.
        """
        counts: dict[str, int] = {}
        for batch in self._batches:
            task = self._task_labels[batch[0]]
            counts[task] = counts.get(task, 0) + len(batch)
        return counts


@runtime_checkable
class BatchSamplerSource(Protocol):
    """Per-epoch provider of a train batch sampler.

    Implemented by callers that want to drive batch composition
    themselves; a :class:`~molix.batch.datamodule.DataModule` constructed
    with ``batch_sampler_source=`` calls :meth:`batch_sampler` once per
    epoch, immediately before building the train
    :class:`~torch.utils.data.DataLoader`.

    A *provider* rather than a sampler instance, because the seed is only
    known per epoch: the DataModule passes ``seed + epoch``, which keeps
    epoch *k*'s composition re-derivable on resume without checkpointing
    RNG state (the same reseed convention
    :class:`TokenBudgetBatchSampler` gets on the budget path).
    """

    def batch_sampler(self, dataset: BaseDataset, *, seed: int) -> Iterable[list[int]]:
        """Return this epoch's batch sampler over *dataset*.

        The returned object is handed straight to
        :class:`~torch.utils.data.DataLoader` as ``batch_sampler=``, so it
        must be iterable **and** ``Sized`` — each iteration yields one
        ``list[int]`` of sample indices, and ``len(...)`` is the number of
        batches. A plain ``list[list[int]]`` qualifies, as does
        :class:`TokenBudgetBatchSampler`.

        Args:
            dataset: The train dataset the indices address; its
                ``__len__`` bounds the valid index range.
            seed: Per-epoch seed (``DataModule.seed + epoch``). Deriving
                every random choice from it keeps the epoch reproducible.

        Returns:
            An iterable, sized batch plan of sample-index lists.
        """
        ...
