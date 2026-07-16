"""Token-budget batch sampling over :class:`~molix.data.cache.PackedCache` pointers.

Molecular samples vary widely in size (QM9 spans 3–29 atoms), so a fixed
``batch_size`` makes per-batch compute and memory swing with batch
composition — GNN encoder cost scales with the total edge count O(E), not
the sample count O(B). :class:`TokenBudgetBatchSampler` instead packs each
batch greedily under an atom and/or edge budget, keeping per-batch work
approximately constant. Per-sample sizes come from the packed cache's
cumsum pointers (``ptr[idx+1] - ptr[idx]``) — no sample is ever unpacked
during sampling.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import torch

from molix.core.seed import make_generator
from molix.data.dataset import BaseDataset

__all__ = ["TokenBudgetBatchSampler"]


class TokenBudgetBatchSampler:
    """Yield index batches whose total atom/edge counts stay under budget.

    Greedy first-fit in seeded-shuffle order: samples are visited in a
    ``randperm`` permutation drawn from
    :func:`~molix.core.seed.make_generator`; the current batch is closed
    whenever adding the next sample would exceed any active budget. A
    sample whose own count exceeds the budget forms a singleton batch —
    it is never dropped — and one :class:`UserWarning` is emitted per
    sampler instance when this first happens.

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
        """Greedy first-fit packing of the shuffled indices under budget."""
        pairs = self._budget_counts()
        n = len(self._dataset)
        perm = torch.randperm(n, generator=make_generator(self._seed)).tolist()

        batches: list[list[int]] = []
        current: list[int] = []
        totals = [0] * len(pairs)
        warned_oversize = False

        for idx in perm:
            sizes = [int(counts[idx].item()) for _, counts in pairs]
            oversize = any(s > budget for s, (budget, _) in zip(sizes, pairs))
            if oversize and not warned_oversize:
                warnings.warn(
                    "TokenBudgetBatchSampler: at least one sample exceeds "
                    "the per-batch budget on its own; such samples are kept "
                    "as singleton batches (never dropped). Raise the budget "
                    "to avoid singletons.",
                    UserWarning,
                    stacklevel=3,
                )
                warned_oversize = True
            fits = all(t + s <= budget for t, s, (budget, _) in zip(totals, sizes, pairs))
            if current and not fits:
                batches.append(current)
                current = []
                totals = [0] * len(pairs)
            current.append(idx)
            totals = [t + s for t, s in zip(totals, sizes)]
            if oversize:
                batches.append(current)
                current = []
                totals = [0] * len(pairs)
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate the cached batches, each a ``list[int]`` of sample indices."""
        return iter(self._batches)

    def __len__(self) -> int:
        """Number of batches for this permutation.

        Varies with ``seed`` (and therefore epoch when seeded as
        ``seed + epoch``), since greedy packing of a different shuffle
        order can produce a different batch count.
        """
        return len(self._batches)
