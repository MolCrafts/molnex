"""Token-budget batch sampling over :class:`~molix.data.cache.PackedCache` pointers.

Molecular samples vary widely in size (QM9 spans 3–29 atoms), so a fixed
``batch_size`` makes per-batch compute and memory swing with batch
composition — GNN encoder cost scales with the total edge count O(E), not
the sample count O(B). :class:`TokenBudgetBatchSampler` instead packs each
batch next-fit (single open bin) under an atom and/or edge budget, keeping
per-batch work approximately constant. Per-sample sizes come from the packed
cache's cumsum pointers (``ptr[idx+1] - ptr[idx]``) — no sample is ever
unpacked during sampling.

Three layers, deliberately separated:

* :class:`TokenBudgetPacker` — the packing rule alone (visit order in,
  batch plan out); no dataset, no RNG, no warning policy.
* :class:`TokenBudgetBatchSampler` — seeded shuffle + oversize warning on
  top of the packer, over a packed-cache-backed dataset.
* :class:`BatchSamplerSource` — the protocol a
  :class:`~molix.data.datamodule.DataModule` calls once per epoch to obtain
  a batch sampler, so alternative batching policies can be injected.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol, runtime_checkable

import torch

from molix.core.seed import make_generator
from molix.data.dataset import BaseDataset

__all__ = ["BatchSamplerSource", "TokenBudgetBatchSampler", "TokenBudgetPacker"]


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
            :class:`~molix.data.dataset.MmapDataset`,
            :class:`~molix.data.dataset.CachedDataset`, or a
            :class:`~molix.data.dataset.SubsetDataset` wrapping either.
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


@runtime_checkable
class BatchSamplerSource(Protocol):
    """Per-epoch provider of a train batch sampler.

    Implemented by callers that want to drive batch composition
    themselves; a :class:`~molix.data.datamodule.DataModule` constructed
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
