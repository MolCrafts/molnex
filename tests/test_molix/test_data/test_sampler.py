"""RED tests for the token-budget dynamic batch sampler stack.

Covers spec ``dynamic-batching-packed-collate-01-sampler``:

* GROUP A — pointer-derived count accessors (``atom_counts`` /
  ``pair_counts``) on ``_CacheBacked`` datasets and ``SubsetDataset``.
* GROUP B — ``molix.data.sampler.TokenBudgetBatchSampler`` behaviour.
* GROUP C — ``DataModule`` opt-in wiring via ``max_atoms_per_batch`` /
  ``max_pairs_per_batch`` (translated to the sampler's ``budgets`` table).

Spec ``universal-class-i-09-packer`` adds GROUP D: characterization
goldens pinning the pre-refactor batch plans plus RED tests for the
extracted ``molix.data.TokenBudgetPacker``.

Spec ``universal-class-i-10-sampler`` adds GROUP E: RED tests for
``molix.data.sampler.TaskMixtureBatchSampler`` — hierarchical
task → group → configuration batch sampling over the same packed caches.

Spec ``streaming-data-tiers-04-scheduler-integration-01-tables`` adds
GROUP F: RED tests for ``molix.data.sampler.FrameLabelTable`` and for
planning the GROUP E fixture through *either* label representation —
legacy Python string sequences or the block-native ``(F,)`` integer code
columns — with one plan, one sampler and no downstream fork.

Fixture geometry (hand-checkable): sample ``i`` has ``3 + (i % 8)`` atoms
(cycling 3..10) and ``n_atoms + 2`` edges (cycling 5..12). Edge counts
never equal atom counts, so the packed-cache schema inference cannot
misclassify per-edge keys as per-atom. GROUP E's fixtures use explicit
per-sample atom-count tables instead (see ``TASK_ATOM_COUNTS``), so every
unit aggregate is a hand-derivable integer.
"""

from __future__ import annotations

import inspect
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import BatchSampler, Dataset, RandomSampler

from molix.core.seed import make_generator
from molix.data import TokenBudgetPacker
from molix.data.datamodule import DataModule
from molix.data.dataset import BaseDataset, SubsetDataset
from molix.data.sampler import TaskMixtureBatchSampler, TokenBudgetBatchSampler
from tests.conftest import frame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SAMPLES = 64
#: Hand-computed per-sample atom counts: 3, 4, ..., 10, 3, 4, ... (cycle of 8).
ATOM_COUNTS = [3 + (i % 8) for i in range(N_SAMPLES)]
#: Hand-computed per-sample edge counts: always atoms + 2 (5..12).
EDGE_COUNTS = [c + 2 for c in ATOM_COUNTS]


def _make_varied_samples(n: int) -> list:
    """Build ``n`` frames whose atom counts follow the module tables.

    Args:
        n: Number of frames (``ATOM_COUNTS`` prefix used).

    Returns:
        molpy ``Frame`` objects carrying the identity ``energy = float(i)``.
    """
    torch.manual_seed(0)
    samples = []
    for i in range(n):
        n_atoms = 3 + (i % 8)
        samples.append(
            frame(
                Z=[1] * n_atoms,
                pos=torch.randn(n_atoms, 3),
                energy=float(i),
                terms={"pairs": {"i": np.zeros(n_atoms + 2, dtype=np.int64)}},
            )
        )
    return samples


def _make_edgeless_samples(n: int) -> list[dict]:
    """Build ``n`` samples with per-atom keys only — packed cache gets no ``edge_ptr``."""
    torch.manual_seed(0)
    samples = []
    for i in range(n):
        na = 3 + (i % 8)
        samples.append(
            {
                "Z": torch.ones(na, dtype=torch.long),
                "pos": torch.randn(na, 3),
                "targets": {"U0": torch.tensor([float(i)])},
            }
        )
    return samples


class _CountsDataset(BaseDataset):
    """Frames plus per-frame ``n_atoms`` / ``n_pairs`` counts.

    Planning reads those columns and never opens a frame. Missing ``pairs``
    on the Frame means no ``n_pairs`` column — :func:`edge_nrows` raises
    like a block store.
    """

    def __init__(self, samples: list) -> None:
        self._samples = samples
        self.getitem_calls = 0
        self._n_atoms = torch.tensor(
            [int(sample["atoms"].nrows) for sample in samples], dtype=torch.long
        )
        try:
            self._n_edges: torch.Tensor | None = torch.tensor(
                [int(sample["pairs"].nrows) for sample in samples], dtype=torch.long
            )
        except KeyError:
            self._n_edges = None
        self.frame_mol = torch.arange(len(samples), dtype=torch.long)

    def column(self, name: str) -> torch.Tensor:
        if name == "n_atoms":
            return self._n_atoms
        if name == "n_pairs":
            if self._n_edges is None:
                raise KeyError(name)
            return self._n_edges
        raise KeyError(name)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        self.getitem_calls += 1
        return self._samples[idx]

    def __getstate__(self) -> dict:
        """Pickle the counts, not the frames: molpy ``Frame`` is not picklable.

        The sampler plans from ``n_atoms[frame_mol]`` alone, so a round-tripped
        stub still reproduces its plan — which is exactly what the pickle
        tests assert.
        """
        state = self.__dict__.copy()
        state["_samples"] = [None] * len(self._samples)
        return state


@pytest.fixture
def varied_ds() -> _CountsDataset:
    """A dataset with the hand-checkable count distribution."""
    return _CountsDataset(_make_varied_samples(N_SAMPLES))


def _make_budget_dm(tmp_path: Path, tag: str, *, seed: int = 42, **dm_kwargs) -> DataModule:
    """Build a DataModule over varied-count caches (num_workers=0, no pinning).

    Args:
        tmp_path: Pytest temp directory for the packed caches.
        tag: Unique filename tag so multiple modules can share ``tmp_path``.
        seed: DataModule shuffle seed.
        **dm_kwargs: Extra DataModule kwargs (e.g. ``max_atoms_per_batch``).

    Returns:
        A non-DDP DataModule whose train split is the full 64-sample
        varied-count dataset.
    """
    samples = _make_varied_samples(N_SAMPLES + 4)
    train = _CountsDataset(samples[:N_SAMPLES])
    val = _CountsDataset(samples[N_SAMPLES:])
    dm_kwargs.setdefault("batch_size", 4)
    dm_kwargs.setdefault("num_workers", 0)
    dm_kwargs.setdefault("pin_memory", False)
    return DataModule(train, val, seed=seed, **dm_kwargs)


class _NonPackedDataset(Dataset):
    """Minimal torch dataset with no packed-cache backing (no count accessors)."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        return {"Z": torch.ones(2, dtype=torch.long), "pos": torch.zeros(2, 3)}


# ---------------------------------------------------------------------------
# GROUP A — pointer-derived count accessors
# ---------------------------------------------------------------------------


class TestSubsetCounts:
    """``SubsetDataset`` counts must be gathered at packed indices, not forwarded."""

    def test_subset_atom_counts_remap(self, varied_ds):
        """Subset counts are the parent counts at the subset's packed indices."""
        subset = SubsetDataset(varied_ds, [5, 2, 9])
        counts = BaseDataset.nrows_of(subset, "atoms")
        # Shape guard: __getattr__ forwarding would return the FULL (64,) vector.
        assert counts.shape == (3,)
        assert counts.dtype == torch.long
        assert counts.tolist() == [ATOM_COUNTS[5], ATOM_COUNTS[2], ATOM_COUNTS[9]]

    def test_subset_edge_counts_remap(self, varied_ds):
        """Same remap semantics for pair_counts."""
        subset = SubsetDataset(varied_ds, [7, 0, 31, 12])
        counts = BaseDataset.nrows_of(subset, "pairs")
        assert counts.shape == (4,)
        assert counts.tolist() == [EDGE_COUNTS[i] for i in [7, 0, 31, 12]]

    def test_nested_subset_counts(self, varied_ds):
        """A subset of a subset composes the index remap correctly."""
        outer = SubsetDataset(varied_ds, [5, 2, 9, 40])
        inner = SubsetDataset(outer, [2, 0])  # packed indices 9, 5
        assert BaseDataset.nrows_of(inner, "atoms").tolist() == [ATOM_COUNTS[9], ATOM_COUNTS[5]]
        assert BaseDataset.nrows_of(inner, "pairs").tolist() == [EDGE_COUNTS[9], EDGE_COUNTS[5]]

    def test_split_subset_counts(self, varied_ds):
        """``split()`` views report counts at their own packed indices."""
        train, rest = varied_ds.split(sizes=(48, 16), seed=7)
        expected = [ATOM_COUNTS[i] for i in train.indices]
        assert BaseDataset.nrows_of(train, "atoms").tolist() == expected
        assert BaseDataset.nrows_of(rest, "atoms").shape == (16,)


# ---------------------------------------------------------------------------
# GROUP B — TokenBudgetBatchSampler
# ---------------------------------------------------------------------------


class TestSamplerBudgetCompliance:
    """Every yielded batch must respect the configured budgets."""

    def test_atom_budget_respected(self, varied_ds):
        """Atom-only budget: counts[batch].sum() <= max_atoms for every batch."""
        sampler = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16}, seed=0)
        counts = torch.tensor(ATOM_COUNTS, dtype=torch.long)
        for batch in sampler:
            assert len(batch) >= 1
            assert int(counts[batch].sum()) <= 16

    def test_edge_budget_respected(self, varied_ds):
        """Edge-only budget: counts[batch].sum() <= max_pairs for every batch."""
        sampler = TokenBudgetBatchSampler(varied_ds, budgets={"pairs": 20}, seed=0)
        counts = torch.tensor(EDGE_COUNTS, dtype=torch.long)
        for batch in sampler:
            assert int(counts[batch].sum()) <= 20

    def test_both_budgets_respected_simultaneously(self, varied_ds):
        """Dual budget: each batch satisfies BOTH limits at once."""
        sampler = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16, "pairs": 20}, seed=0)
        atom_counts = torch.tensor(ATOM_COUNTS, dtype=torch.long)
        pair_counts = torch.tensor(EDGE_COUNTS, dtype=torch.long)
        for batch in sampler:
            assert int(atom_counts[batch].sum()) <= 16
            assert int(pair_counts[batch].sum()) <= 20

    def test_injected_atom_counts_are_used_instead_of_nrows(self) -> None:
        """A caller-supplied vector is the budget; the dataset is not scanned."""

        class _Boom:
            def __len__(self) -> int:
                return 3

            def __getitem__(self, idx: int) -> int:
                raise AssertionError("packing must not read samples")

        counts = torch.tensor([2, 3, 4], dtype=torch.long)
        sampler = TokenBudgetBatchSampler(
            _Boom(),  # type: ignore[arg-type]
            budgets={"atoms": 6},
            counts={"atoms": counts},
            seed=0,
        )
        for batch in sampler:
            assert int(counts[batch].sum()) <= 6 or len(batch) == 1


class TestSamplerCoverage:
    """Exact-once coverage and ``__len__`` consistency."""

    def test_exact_once_coverage(self, varied_ds):
        """Concatenated batch indices over one epoch == sorted range(n)."""
        sampler = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16}, seed=3)
        seen = [i for batch in sampler for i in batch]
        assert sorted(seen) == list(range(N_SAMPLES))

    def test_len_matches_yielded_batches(self, varied_ds):
        """len(sampler) equals the number of batches actually yielded."""
        sampler = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16}, seed=3)
        assert len(sampler) == len(list(sampler))


class TestSamplerOversize:
    """Samples exceeding the budget on their own become kept singletons."""

    def test_oversize_sample_is_singleton_not_dropped(self, varied_ds):
        """Each over-budget sample sits alone in its batch; coverage stays exact."""
        # Budget 8 < max sample count 10 → samples with 9 or 10 atoms oversize.
        with pytest.warns(UserWarning):
            sampler = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 8}, seed=1)
            batches = list(sampler)
        oversize = {i for i, c in enumerate(ATOM_COUNTS) if c > 8}
        assert oversize  # fixture sanity: 16 oversize samples exist
        for batch in batches:
            for idx in batch:
                if idx in oversize:
                    assert batch == [idx]
        seen = sorted(i for batch in batches for i in batch)
        assert seen == list(range(N_SAMPLES))

    def test_oversize_warns_exactly_once_per_instance(self, varied_ds):
        """2+ oversize samples still produce exactly one warning per sampler."""
        with pytest.warns(UserWarning) as record:
            sampler = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 8}, seed=1)
            list(sampler)
        budget_warnings = [w for w in record if "budget" in str(w.message).lower()]
        assert len(budget_warnings) == 1


class TestSamplerDeterminism:
    """Batch composition is a pure function of (dataset, budgets, seed)."""

    def test_same_seed_same_batches(self, varied_ds):
        """Two instances with identical config yield identical batch lists."""
        a = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16}, seed=123)
        b = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16}, seed=123)
        assert list(a) == list(b)

    def test_different_seed_different_batches(self, varied_ds):
        """seed vs seed+1 produce different batch composition."""
        a = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16}, seed=123)
        b = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16}, seed=124)
        # 64! permutations — a seeded collision is astronomically unlikely.
        assert list(a) != list(b)

    def test_iteration_stable_within_instance(self, varied_ds):
        """Iterating the same instance twice yields the cached composition."""
        sampler = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16}, seed=5)
        assert list(sampler) == list(sampler)


class TestSamplerValidation:
    """Eager (``__init__``-time) validation with actionable messages."""

    def test_no_budget_raises(self, varied_ds):
        """An empty budget table is rejected eagerly."""
        with pytest.raises(ValueError, match="budgets"):
            TokenBudgetBatchSampler(varied_ds, budgets={}, seed=0)

    def test_zero_budget_raises(self, varied_ds):
        """A zero budget is rejected eagerly, naming the table."""
        with pytest.raises(ValueError, match="atoms"):
            TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 0}, seed=0)

    def test_negative_budget_raises(self, varied_ds):
        """A negative budget is rejected eagerly, naming the table."""
        with pytest.raises(ValueError, match="pairs"):
            TokenBudgetBatchSampler(varied_ds, budgets={"pairs": -5}, seed=0)

    def test_counts_must_cover_every_budgeted_table(self, varied_ds):
        """A partial counts table is refused, never completed from the dataset."""
        with pytest.raises(ValueError, match="pairs"):
            TokenBudgetBatchSampler(
                varied_ds,
                budgets={"atoms": 16, "pairs": 20},
                counts={"atoms": torch.tensor(ATOM_COUNTS)},
                seed=0,
            )

    def test_counts_for_an_unbudgeted_table_are_refused(self, varied_ds):
        with pytest.raises(ValueError, match="no budget"):
            TokenBudgetBatchSampler(
                varied_ds,
                budgets={"atoms": 16},
                counts={"atoms": torch.tensor(ATOM_COUNTS), "pairs": torch.tensor(EDGE_COUNTS)},
                seed=0,
            )

    def test_non_packed_dataset_raises_naming_supported_types(self):
        """Datasets that are not Frame-shaped raise, naming nrows."""
        with pytest.raises(ValueError) as excinfo:
            TokenBudgetBatchSampler(_NonPackedDataset(), budgets={"atoms": 16}, seed=0)
        message = str(excinfo.value)
        assert "atoms" in message
        assert "_NonPackedDataset" in message


class TestSamplerNoUnpack:
    """The sampler must never unpack samples — pointer arithmetic only."""

    def test_zero_getitem_calls_during_construction_and_iteration(self, tmp_path):
        """__getitem__ is called 0 times by __init__ + one full epoch."""
        ds = _CountsDataset(_make_varied_samples(N_SAMPLES))
        sampler = TokenBudgetBatchSampler(ds, budgets={"atoms": 16, "pairs": 24}, seed=0)
        list(sampler)
        assert ds.getitem_calls == 0


class TestSamplerPickle:
    """Sampler must survive pickling (spawn-safe DataModule transfer)."""

    def test_pickle_round_trip_same_batches(self, tmp_path):
        """pickle.loads(pickle.dumps(s)) yields the identical batch list."""
        ds = _CountsDataset(_make_varied_samples(N_SAMPLES))
        sampler = TokenBudgetBatchSampler(ds, budgets={"atoms": 16}, seed=11)
        restored = pickle.loads(pickle.dumps(sampler))
        assert list(restored) == list(sampler)


class TestSamplerGraphsBudgetAndOrder:
    """``graphs`` counts one per sample; ``shuffle=False`` keeps index order; ``reseed`` replans."""

    def test_graphs_budget_caps_molecules_per_batch(self, varied_ds):
        sampler = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 64, "graphs": 3}, seed=0)
        assert all(len(batch) <= 3 for batch in sampler)
        assert sorted(i for batch in sampler for i in batch) == list(range(N_SAMPLES))

    def test_shuffle_false_packs_in_index_order(self, varied_ds):
        sampler = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16}, seed=0, shuffle=False)
        flat = [i for batch in sampler for i in batch]
        assert flat == list(range(N_SAMPLES))
        counts = torch.tensor(ATOM_COUNTS, dtype=torch.long)
        assert all(int(counts[batch].sum()) <= 16 for batch in sampler)

    def test_reseed_replans_like_a_fresh_instance(self, varied_ds):
        sampler = TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16}, seed=1)
        before = list(sampler)
        sampler.reseed(2)
        assert list(sampler) == list(
            TokenBudgetBatchSampler(varied_ds, budgets={"atoms": 16}, seed=2)
        )
        assert list(sampler) != before


# ---------------------------------------------------------------------------
# GROUP C — DataModule wiring
# ---------------------------------------------------------------------------


class TestDataModuleBudgetOptIn:
    """``max_atoms_per_batch`` switches the train loader to the budget sampler."""

    def test_budget_loader_uses_token_sampler(self, tmp_path):
        """train_dataloader().batch_sampler is a TokenBudgetBatchSampler."""
        dm = _make_budget_dm(tmp_path, "optin", max_atoms_per_batch=16)
        loader = dm.train_dataloader()
        assert isinstance(loader.batch_sampler, TokenBudgetBatchSampler)
        # DataLoader normalises batch_size to None when batch_sampler is given.
        assert loader.batch_size is None

    def test_budget_loader_iterates_and_respects_budget(self, tmp_path):
        """The opt-in loader collates fine and every batch obeys the budget."""
        dm = _make_budget_dm(tmp_path, "iter", max_atoms_per_batch=16)
        loader = dm.train_dataloader()
        total_graphs = 0
        for batch in loader:
            n_atoms = batch["atoms", "Z"].shape[0]
            assert 0 < n_atoms <= 16
            total_graphs += batch["graphs", "energy"].shape[0]
        assert total_graphs == N_SAMPLES


class TestDataModuleBackwardCompat:
    """Without the new kwargs the DataLoader config is byte-for-byte legacy."""

    def test_default_train_loader_config_unchanged(self, tmp_path):
        """Fixed batch_size, RandomSampler-backed BatchSampler, seeded generator."""
        dm = _make_budget_dm(tmp_path, "compat", batch_size=4)
        loader = dm.train_dataloader()
        assert loader.batch_size == 4
        assert isinstance(loader.sampler, RandomSampler)
        assert isinstance(loader.batch_sampler, BatchSampler)
        assert loader.generator is not None
        assert loader.drop_last is False


class TestDataModuleDDP:
    """Budgeted batching is mutually exclusive with DDP for now."""

    def test_budget_with_ddp_raises(self, tmp_path, monkeypatch):
        """_is_distributed() True + budget → eager ValueError mentioning DDP."""
        monkeypatch.setattr("molix.data.datamodule._is_distributed", lambda: True)
        dm = _make_budget_dm(tmp_path, "ddp", max_atoms_per_batch=16)
        with pytest.raises(ValueError, match="(?i)ddp|distributed"):
            dm.train_dataloader()


class TestDataModuleEpochRederivation:
    """Epoch k's batch composition is derivable from (seed, epoch) alone."""

    def test_epoch_reshuffle_differs(self, tmp_path):
        """Two consecutive epochs yield different batch compositions."""
        dm = _make_budget_dm(tmp_path, "reshuffle", seed=777, max_atoms_per_batch=16)
        dm.on_epoch_start(0)
        batches_epoch0 = list(dm.train_dataloader().batch_sampler)
        dm.on_epoch_start(1)
        batches_epoch1 = list(dm.train_dataloader().batch_sampler)
        # Same multiset of indices either way — only the composition changes.
        flat0 = sorted(i for b in batches_epoch0 for i in b)
        flat1 = sorted(i for b in batches_epoch1 for i in b)
        assert flat0 == flat1 == list(range(N_SAMPLES))
        assert batches_epoch0 != batches_epoch1

    def test_epoch_composition_resume_derivable(self, tmp_path):
        """Fresh DataModule + on_epoch_start(1) reproduces continuous epoch 1."""
        dm_cont = _make_budget_dm(tmp_path, "cont", seed=777, max_atoms_per_batch=16)
        dm_cont.on_epoch_start(0)
        batches_epoch0 = list(dm_cont.train_dataloader().batch_sampler)
        dm_cont.on_epoch_start(1)
        batches_epoch1_cont = list(dm_cont.train_dataloader().batch_sampler)

        dm_fresh = _make_budget_dm(tmp_path, "fresh", seed=777, max_atoms_per_batch=16)
        dm_fresh.on_epoch_start(1)
        batches_epoch1_fresh = list(dm_fresh.train_dataloader().batch_sampler)

        # (a) Resume reproducibility: derivable from (seed, epoch) alone.
        assert batches_epoch1_fresh == batches_epoch1_cont
        # (b) Guard against a degenerate single-permutation world.
        assert batches_epoch1_cont != batches_epoch0


# ---------------------------------------------------------------------------
# GROUP D — characterization goldens + TokenBudgetPacker
# (spec universal-class-i-09-packer)
# ---------------------------------------------------------------------------

#: Sample count of the golden characterization cache.
GOLDEN_N = 16
#: Default ``seed`` of ``TokenBudgetBatchSampler.__init__`` (src/molix/data/sampler.py).
GOLDEN_SEED = 42

# --- Captured from the PRE-REFACTOR tree on 2026-08-14, torch 2.12.1+cpu ----
# Capture one-liner (run from the repo root, with the molcrafts PYTHONPATH):
#
#   python -c "
#   import sys, tempfile, warnings, torch; from pathlib import Path
#   sys.path.insert(0, '.')
#   from tests.test_molix.test_data.test_sampler import _write_varied_cache
#   from molix.core.seed import make_generator
#   from molix.data.sampler import TokenBudgetBatchSampler as S
#   print(torch.randperm(16, generator=make_generator(42)).tolist())
#   with tempfile.TemporaryDirectory() as td:
#       ds = _CountsDataset(_make_varied_samples(16))
#       print(list(S(ds, max_atoms=16)))
#       print(list(S(ds, max_atoms=16, max_pairs=20)))
#       warnings.simplefilter('ignore'); print(list(S(ds, max_atoms=8)))
#   "
#
#: ``torch.randperm(16, generator=make_generator(42)).tolist()``.
GOLDEN_PERM = [6, 3, 0, 7, 10, 15, 2, 12, 14, 9, 8, 5, 4, 13, 11, 1]
#: Plan for ``max_atoms=16`` over the 16-sample varied cache.
GOLDEN_PLAN_ATOMS16 = [[6, 3], [0, 7], [10, 15], [2, 12], [14, 9, 8], [5, 4], [13, 11], [1]]
#: Plan for ``max_atoms=16, max_pairs=20`` — the edge cap splits ``[14, 9, 8]``.
GOLDEN_PLAN_ATOMS16_EDGES20 = [
    [6, 3],
    [0, 7],
    [10, 15],
    [2, 12],
    [14, 9],
    [8, 5],
    [4, 13],
    [11, 1],
]
#: Plan for ``max_atoms=8`` — samples with 9/10 atoms are kept as singletons.
GOLDEN_PLAN_ATOMS8 = [
    [6],
    [3],
    [0],
    [7],
    [10],
    [15],
    [2],
    [12],
    [14],
    [9, 8],
    [5],
    [4],
    [13],
    [11],
    [1],
]
#: Verbatim oversize warning text (src/molix/data/sampler.py).
GOLDEN_OVERSIZE_WARNING = (
    "TokenBudgetBatchSampler: at least one sample exceeds the per-batch "
    "budget on its own; such samples are kept as singleton batches (never "
    "dropped). Raise the budget to avoid singletons."
)
#: Instance state that must stay picklable and packer-free after the refactor.
GOLDEN_SAMPLER_FIELDS = ["_batches", "_budgets", "_counts", "_dataset", "_seed", "_shuffle"]


@pytest.fixture
def golden_ds() -> _CountsDataset:
    """16-sample varied-count dataset backing the characterization goldens.

    Counts are a pure function of the sample index (``3 + i % 8`` atoms,
    ``atoms + 2`` edges), so rebuilding it per test reproduces the captured
    plans exactly.
    """
    return _CountsDataset(_make_varied_samples(GOLDEN_N))


class TestSamplerGoldenPlan:
    """Characterization: the pre-refactor plans, pinned as literals.

    Green on the pre-refactor tree by construction — this class exists to
    prove the ``TokenBudgetPacker`` extraction is behavior-preserving.
    """

    def test_golden_counts_are_deterministic(self, golden_ds):
        """The fixture cache reproduces the hand-checkable count tables."""
        assert BaseDataset.nrows_of(golden_ds, "atoms").tolist() == ATOM_COUNTS[:GOLDEN_N]
        assert BaseDataset.nrows_of(golden_ds, "pairs").tolist() == EDGE_COUNTS[:GOLDEN_N]

    def test_permutation_matches_golden(self):
        """The visit order is ``randperm(n, generator=make_generator(seed))``."""
        perm = torch.randperm(GOLDEN_N, generator=make_generator(GOLDEN_SEED))
        assert perm.tolist() == GOLDEN_PERM

    def test_atom_budget_plan_matches_golden(self, golden_ds):
        """max_atoms=16 reproduces the captured batch plan."""
        sampler = TokenBudgetBatchSampler(golden_ds, budgets={"atoms": 16})
        assert list(sampler) == GOLDEN_PLAN_ATOMS16

    def test_dual_budget_plan_matches_golden(self, golden_ds):
        """max_atoms=16 + max_pairs=20 reproduces the captured batch plan."""
        sampler = TokenBudgetBatchSampler(golden_ds, budgets={"atoms": 16, "pairs": 20})
        assert list(sampler) == GOLDEN_PLAN_ATOMS16_EDGES20

    def test_oversize_plan_matches_golden(self, golden_ds):
        """max_atoms=8 reproduces the captured singleton-heavy batch plan."""
        with pytest.warns(UserWarning):
            sampler = TokenBudgetBatchSampler(golden_ds, budgets={"atoms": 8})
        assert list(sampler) == GOLDEN_PLAN_ATOMS8


class TestSamplerSurfacePreservation:
    """Observable sampler surface pinned across the packer extraction.

    Also green pre-refactor: the packer must stay transient (never stored
    on the instance) and the warning text/count must not drift.
    """

    def test_instance_fields_match_golden_list(self, golden_ds):
        """``vars(sampler)`` keeps exactly its six picklable fields."""
        sampler = TokenBudgetBatchSampler(golden_ds, budgets={"atoms": 16})
        assert sorted(vars(sampler)) == GOLDEN_SAMPLER_FIELDS

    def test_oversize_warning_text_verbatim_once(self, golden_ds):
        """Exactly one warning per instance, with the unchanged message."""
        with pytest.warns(UserWarning) as record:
            TokenBudgetBatchSampler(golden_ds, budgets={"atoms": 8})
        assert [str(w.message) for w in record] == [GOLDEN_OVERSIZE_WARNING]


class TestTokenBudgetPacker:
    """``molix.data.TokenBudgetPacker`` — next-fit packing, no dataset/RNG.

    ``TokenBudgetPacker`` is imported inside each test so this class fails
    at call time (RED) without breaking collection of the characterization
    class above.
    """

    def test_next_fit_closes_the_single_open_bin(self):
        """counts [7, 6, 5] / budget 12 → [[0], [1, 2]] (not first-fit [[0, 2], [1]])."""

        packer = TokenBudgetPacker([(12, torch.tensor([7, 6, 5]))])
        assert packer.pack([0, 1, 2]) == [[0], [1, 2]]

    def test_oversize_singleton_is_emitted_without_warning(self):
        """counts [3, 20, 4] / budget 10 → singletons, and the packer stays silent."""

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            packer = TokenBudgetPacker([(10, torch.tensor([3, 20, 4]))])
            assert packer.pack([0, 1, 2]) == [[0], [1], [2]]

    def test_oversize_closes_a_non_empty_bin_first(self):
        """An oversize index never merges into the open bin — it closes it."""

        packer = TokenBudgetPacker([(10, torch.tensor([2, 3, 25, 4]))])
        assert packer.pack([0, 1, 2, 3]) == [[0, 1], [2], [3]]

    def test_edge_budget_closes_batch_before_atom_budget(self):
        """Dual budgets: the binding (edge) cap drives closure, hand-computed."""

        atom_counts = torch.tensor([5, 5, 5, 5])
        pair_counts = torch.tensor([4, 4, 4, 4])
        dual = TokenBudgetPacker([(100, atom_counts), (10, pair_counts)])
        assert dual.pack([0, 1, 2, 3]) == [[0, 1], [2, 3]]
        # Atoms alone (20 <= 100) would keep all four together — so the edge
        # budget, not the atom budget, is what closed each batch.
        atoms_only = TokenBudgetPacker([(100, atom_counts)])
        assert atoms_only.pack([0, 1, 2, 3]) == [[0, 1, 2, 3]]

    def test_pack_empty_order_returns_empty_plan(self):
        """An empty visit order packs to no batches at all."""

        packer = TokenBudgetPacker([(10, torch.tensor([3, 4]))])
        assert packer.pack([]) == []

    def test_empty_budgets_raises_naming_the_pair(self):
        """``budgets=[]`` is rejected eagerly, naming ``(budget, counts)``."""

        with pytest.raises(ValueError, match=r"\(budget, counts\)"):
            TokenBudgetPacker(budgets=[])

    def test_oversize_mask_values_single_budget(self):
        """``counts > budget`` elementwise, as a bool ``(n_samples,)`` tensor."""

        mask = TokenBudgetPacker([(10, torch.tensor([3, 20, 4]))]).oversize_mask()
        assert mask.dtype == torch.bool
        assert mask.tolist() == [False, True, False]

    def test_oversize_mask_is_elementwise_or_across_budgets(self):
        """A sample oversize under *any* budget is masked."""

        packer = TokenBudgetPacker([(10, torch.tensor([3, 20, 4])), (5, torch.tensor([1, 1, 9]))])
        assert packer.oversize_mask().tolist() == [False, True, True]

    def test_constructs_from_bare_tensors_without_a_dataset(self):
        """No dataset, no RNG: bare ``(budget, counts)`` tensors are enough."""

        packer = TokenBudgetPacker([(9, torch.tensor([4, 4, 4, 4], dtype=torch.long))])
        assert not any("dataset" in name for name in vars(packer))
        assert packer.pack([3, 1, 0, 2]) == [[3, 1], [0, 2]]

    def test_golden_atom_budget_plan_via_packer(self, golden_ds):
        """Packing the golden permutation reproduces the max_atoms=16 plan."""

        packer = TokenBudgetPacker([(16, BaseDataset.nrows_of(golden_ds, "atoms"))])
        assert packer.pack(GOLDEN_PERM) == GOLDEN_PLAN_ATOMS16

    def test_golden_dual_budget_plan_via_packer(self, golden_ds):
        """Packing the golden permutation reproduces the dual-budget plan."""

        packer = TokenBudgetPacker(
            [
                (16, BaseDataset.nrows_of(golden_ds, "atoms")),
                (20, BaseDataset.nrows_of(golden_ds, "pairs")),
            ]
        )
        assert packer.pack(GOLDEN_PERM) == GOLDEN_PLAN_ATOMS16_EDGES20

    def test_golden_oversize_plan_via_packer(self, golden_ds):
        """Packing the golden permutation reproduces the max_atoms=8 plan."""

        packer = TokenBudgetPacker([(8, BaseDataset.nrows_of(golden_ds, "atoms"))])
        assert packer.pack(GOLDEN_PERM) == GOLDEN_PLAN_ATOMS8


# ---------------------------------------------------------------------------
# GROUP E — TaskMixtureBatchSampler
# (spec universal-class-i-10-sampler)
# ---------------------------------------------------------------------------
#
# One deliberate fixture, hand-derived end to end.  Sample ``i`` carries
# ``TASK_ATOM_COUNTS[i]`` atoms and ``TASK_ATOM_COUNTS[i] + 2`` edges, so a
# unit's aggregate cost is ``sum(atoms)`` / ``sum(atoms) + 2 * n_frames``:
#
#   idx     task           group    frames  atoms/frame  Σatoms  Σedges
#   -----------------------------------------------------------------------
#   0..5    equilibrium    eq0..eq5      1            5       5       7
#   6..9    torsion        tor_a         4            6      24      32
#   10..12  torsion        tor_b         3            6      18      24
#   13,14   torsion        tor_c         2            7      14      18
#   15..17  relax          rel_a         3            4      12      18
#   18,19   relax          rel_b         2            4       8      12
#   20,21   electrostatic  ele_a         2            5      10      14
#   22,23   electrostatic  ele_b         2            5      10      14
#
# Per-task next-fit supply under ``max_atoms=24`` (or 20 — same counts):
#   equilibrium  6 units × 5 atoms → 4 units (20) + 2 units (10)      = 2 batches
#   torsion      24 / 18 / 14 — no two aggregate ≤ 24                 = 3 batches
#   relax        12 + 8 = 20 ≤ 24                                     = 1 batch
#   electrostatic 10 + 10 = 20 ≤ 24                                   = 1 batch
#
# ``max_atoms=20`` additionally makes ``tor_a`` (24 atoms) an oversize unit,
# kept whole as a singleton batch and warned about exactly once.

#: Per-sample atom counts of the 24-sample task fixture (table above).
TASK_ATOM_COUNTS = (
    [5] * 6  # 0..5    equilibrium singletons
    + [6] * 4  # 6..9    torsion scan tor_a  (oversize at max_atoms=20)
    + [6] * 3  # 10..12  torsion scan tor_b
    + [7] * 2  # 13,14   torsion scan tor_c
    + [4] * 3  # 15..17  relax trajectory rel_a
    + [4] * 2  # 18,19   relax trajectory rel_b
    + [5] * 4  # 20..23  electrostatic ele_a / ele_b
)
#: Per-sample task labels, parent space, ``len == len(dataset)``.
TASK_LABELS = ["equilibrium"] * 6 + ["torsion"] * 9 + ["relax"] * 5 + ["electrostatic"] * 4
#: Per-sample group ids, parent space, ``len == len(dataset)``.
GROUP_IDS = (
    [f"eq{i}" for i in range(6)]
    + ["tor_a"] * 4
    + ["tor_b"] * 3
    + ["tor_c"] * 2
    + ["rel_a"] * 3
    + ["rel_b"] * 2
    + ["ele_a"] * 2
    + ["ele_b"] * 2
)
#: Split index list without the electrostatic samples (Stage A/B splits).
TASK_INDICES_NO_ELECTROSTATIC = list(range(20))
#: Split index list covering the whole fixture (Stage C splits).
TASK_INDICES_ALL = list(range(24))

#: ByteFF-derived Stage B mixture (arXiv:2408.12817), sums to 1.0.
STAGE_B = {"equilibrium": 0.35, "torsion": 0.45, "relax": 0.20}
#: ByteFF-derived Stage C mixture (arXiv:2408.12817), sums to 1.0.
STAGE_C = {"equilibrium": 0.25, "torsion": 0.30, "relax": 0.35, "electrostatic": 0.10}
#: Mixture whose ``electrostatic`` share is small enough that largest-remainder
#: alone starves it, so the anti-starvation override must fire. Sums to 1.0.
SCARCE_MIXTURE = {
    "equilibrium": 0.30,
    "torsion": 0.35,
    "relax": 0.33,
    "electrostatic": 0.02,
}
#: Seed for the hard-coded golden plan (spec ``universal-class-i-10-sampler``).
TASK_SEED = 20260814

# --- Golden plan, captured 2026-08-14 from the scratchpad reference ---------
# prototype ``task_mixture_prototype.build_plan`` (torch 2.12.1+cpu, CPU,
# float32), which implements the spec's eager-plan steps 1-7 verbatim:
#   1. bucket ``indices`` (given order) into units keyed (task_label, group_id)
#   2. per-unit aggregate cost from ``dataset.atom_counts`` pointer arithmetic
#   3. ONE ``make_generator(TASK_SEED)``; tasks visited in ``sorted(mixture)``
#      order — here equilibrium, relax, torsion — each consuming one
#      ``torch.randperm(n_units, generator=g)``
#   4. one transient ``TokenBudgetPacker`` per task, ``.pack(shuffled_order)``,
#      unit positions expanded to member frame indices
#   5. natural epoch length ``min_{p>0} floor(len(batches_t) / p_t)``
#      = min(floor(2/0.35)=5, floor(3/0.45)=6, floor(1/0.20)=5) = 5
#   6. largest-remainder apportionment: base = (1, 1, 2) for
#      (equilibrium 1.75, relax 1.00, torsion 2.25); leftover 1 goes to the
#      largest remainder (equilibrium, 0.75) → equilibrium 2, relax 1, torsion 2
#   7. interleave: multiset ["equilibrium"×2, "relax"×1, "torsion"×2] in sorted
#      task order, permuted by ``torch.randperm(5, generator=g)`` from the SAME
#      generator; each slot consumes its task's next unused packed batch
#
# Only the *index literals* below come from the prototype — every count in the
# derivation above is hand-computed from the fixture table. A post-GREEN
# mismatch therefore means the implementation deviated from the spec algorithm,
# not that the goldens were stale.
#
#: ``list(TaskMixtureBatchSampler(...))`` for Stage B / max_atoms=24 / seed 20260814.
GOLDEN_TASK_PLAN = [
    [3, 2, 5, 1],  # equilibrium: 4 units × 5 atoms = 20 (a 5th would be 25 > 24)
    [15, 16, 17, 18, 19],  # relax: rel_a (12) + rel_b (8) = 20
    [10, 11, 12],  # torsion: tor_b, 18 atoms
    [13, 14],  # torsion: tor_c, 14 atoms
    [0, 4],  # equilibrium: the 2 leftover units, 10 atoms
]
#: Hand-derived apportionment of the 5 golden slots (step 6 above).
GOLDEN_BATCHES_PER_TASK = {"equilibrium": 2, "relax": 1, "torsion": 2}
#: Frame counts implied by ``GOLDEN_TASK_PLAN`` (6 + 5 + 3 + 2).
GOLDEN_FRAMES_PER_TASK = {"equilibrium": 6, "relax": 5, "torsion": 5}

#: Units per task in the wide fixture used for the ≥ 200-batch mixture bounds.
BIG_UNITS_PER_TASK = 96
#: Atom budget of the wide fixture — every unit costs ≤ 6 and no two fit, so
#: ``n_batches == n_units`` per task and no unit is oversize.
BIG_MAX_ATOMS = 6
#: Forced epoch length with a non-integral apportionment for every task.
BIG_BATCHES_PER_EPOCH = 203


def _make_sized_samples(atom_counts: list[int]) -> list:
    """Build frames with the exact per-sample atom counts given.

    Args:
        atom_counts: Per-frame atom count.

    Returns:
        molpy ``Frame`` objects.
    """
    torch.manual_seed(0)
    return [
        frame(
            Z=[1] * na,
            pos=torch.randn(na, 3),
            energy=float(i),
            terms={"pairs": {"i": np.zeros(na + 2, dtype=np.int64)}},
        )
        for i, na in enumerate(atom_counts)
    ]


def _task_ds() -> "_CountsDataset":
    """The 24-sample task fixture dataset."""
    return _CountsDataset(_make_sized_samples(TASK_ATOM_COUNTS))


def _big_task_tables() -> tuple[list[int], list[str], list[str]]:
    """Wide fixture: ``BIG_UNITS_PER_TASK`` units for each of the four tasks.

    Equilibrium and electrostatic units are single frames (5 / 4 atoms);
    torsion and relax units are two frames of 3 atoms (6 atoms aggregate).
    Under ``BIG_MAX_ATOMS`` every unit is its own batch — no two units fit
    together (5+4 = 9 > 6) and none is oversize (max aggregate 6 == budget) —
    so each task supplies exactly ``BIG_UNITS_PER_TASK`` batches.

    Returns:
        ``(atom_counts, task_labels, group_ids)`` in parent index order:
        equilibrium block, torsion block, relax block, electrostatic block.
    """
    atoms: list[int] = []
    labels: list[str] = []
    groups: list[str] = []
    for k in range(BIG_UNITS_PER_TASK):
        atoms.append(5)
        labels.append("equilibrium")
        groups.append(f"eq{k:03d}")
    for task, per_frame in (("torsion", 3), ("relax", 3)):
        for k in range(BIG_UNITS_PER_TASK):
            for _ in range(2):
                atoms.append(per_frame)
                labels.append(task)
                groups.append(f"{task[:3]}{k:03d}")
    for k in range(BIG_UNITS_PER_TASK):
        atoms.append(4)
        labels.append("electrostatic")
        groups.append(f"ele{k:03d}")
    return atoms, labels, groups


@pytest.fixture
def task_ds() -> _CountsDataset:
    """The 24-sample hand-checkable task/group fixture."""
    return _task_ds()


@pytest.fixture(scope="session")
def big_task_fixture() -> tuple[_CountsDataset, list[str], list[str]]:
    """Wide read-only fixture: 4 tasks × 96 units → ≥ 200 batches available."""
    atoms, labels, groups = _big_task_tables()
    return (
        _CountsDataset(_make_sized_samples(atoms)),
        labels,
        groups,
    )


def _units_in(indices: list[int]) -> dict[tuple[str, str], list[int]]:
    """Group *indices* into ``(task, group) -> frame indices`` (fixture space)."""
    units: dict[tuple[str, str], list[int]] = {}
    for i in indices:
        units.setdefault((TASK_LABELS[i], GROUP_IDS[i]), []).append(i)
    return units


class TestTaskMixturePlan:
    """Eager plan: single-task batches and the hard-coded golden."""

    def test_every_batch_carries_exactly_one_task_label(self, task_ds):
        """The flat frame pool is structurally unreachable."""

        sampler = TaskMixtureBatchSampler(
            task_ds,
            indices=TASK_INDICES_ALL,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_C,
            max_atoms=24,
            batches_per_epoch=4,
            seed=TASK_SEED,
        )
        for batch in sampler:
            assert len(batch) >= 1
            assert len({TASK_LABELS[i] for i in batch}) == 1

    def test_plan_indices_are_a_duplicate_free_subset_of_the_split(self, task_ds):
        """An epoch never repeats a frame nor reaches outside ``indices``."""

        sampler = TaskMixtureBatchSampler(
            task_ds,
            indices=TASK_INDICES_ALL,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_C,
            max_atoms=24,
            batches_per_epoch=4,
            seed=TASK_SEED,
        )
        flat = [i for batch in sampler for i in batch]
        assert len(flat) == len(set(flat))
        assert set(flat) <= set(TASK_INDICES_ALL)

    def test_natural_epoch_length_is_the_no_exhaustion_floor(self, task_ds):
        """Step 5: ``min(floor(2/0.35), floor(3/0.45), floor(1/0.20)) == 5``."""

        sampler = TaskMixtureBatchSampler(
            task_ds,
            indices=TASK_INDICES_NO_ELECTROSTATIC,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_B,
            max_atoms=24,
            seed=TASK_SEED,
        )
        assert len(sampler) == 5

    def test_hard_coded_golden_plan(self, task_ds):
        """The captured Stage B plan at ``seed=20260814`` reproduces."""

        sampler = TaskMixtureBatchSampler(
            task_ds,
            indices=TASK_INDICES_NO_ELECTROSTATIC,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_B,
            max_atoms=24,
            seed=TASK_SEED,
        )
        assert list(sampler) == GOLDEN_TASK_PLAN

    def test_hard_coded_golden_audit_dicts(self, task_ds):
        """The audit accessors match the hand-derived apportionment."""

        sampler = TaskMixtureBatchSampler(
            task_ds,
            indices=TASK_INDICES_NO_ELECTROSTATIC,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_B,
            max_atoms=24,
            seed=TASK_SEED,
        )
        assert sampler.batches_per_task() == GOLDEN_BATCHES_PER_TASK
        assert sampler.frames_per_task() == GOLDEN_FRAMES_PER_TASK


class TestTaskMixtureIntegrity:
    """Unit integrity, budget compliance and the reuse boundary."""

    def test_scan_units_never_straddle_batches(self, task_ds):
        """Each unit's frames land in exactly one batch, whole."""

        with pytest.warns(UserWarning):
            sampler = TaskMixtureBatchSampler(
                task_ds,
                indices=TASK_INDICES_NO_ELECTROSTATIC,
                task_labels=TASK_LABELS,
                group_ids=GROUP_IDS,
                mixture=STAGE_B,
                max_atoms=20,
                batches_per_epoch=6,
                seed=TASK_SEED,
            )
        plan = list(sampler)
        # Apportionment at n=6 is (equilibrium 2, torsion 3, relax 1) == supply,
        # so every one of the 8 units in the split appears exactly once.
        for key, frames in _units_in(TASK_INDICES_NO_ELECTROSTATIC).items():
            hosts = [b for b in plan if set(frames) & set(b)]
            assert len(hosts) == 1, f"unit {key} straddles {len(hosts)} batches"
            assert set(frames) <= set(hosts[0])

    def test_oversize_unit_is_kept_whole_as_a_singleton_batch(self, task_ds):
        """``tor_a`` (24 atoms > 20) is one whole batch, never split."""

        with pytest.warns(UserWarning):
            sampler = TaskMixtureBatchSampler(
                task_ds,
                indices=TASK_INDICES_NO_ELECTROSTATIC,
                task_labels=TASK_LABELS,
                group_ids=GROUP_IDS,
                mixture=STAGE_B,
                max_atoms=20,
                batches_per_epoch=6,
                seed=TASK_SEED,
            )
        assert [6, 7, 8, 9] in list(sampler)

    def test_oversize_unit_warns_exactly_once_naming_task_and_group(self, task_ds):
        """One sampler-owned UserWarning naming ``torsion`` + ``tor_a``."""

        with pytest.warns(UserWarning) as record:
            TaskMixtureBatchSampler(
                task_ds,
                indices=TASK_INDICES_NO_ELECTROSTATIC,
                task_labels=TASK_LABELS,
                group_ids=GROUP_IDS,
                mixture=STAGE_B,
                max_atoms=20,
                batches_per_epoch=6,
                seed=TASK_SEED,
            )
        oversize = [w for w in record if "tor_a" in str(w.message)]
        assert len(oversize) == 1
        assert "torsion" in str(oversize[0].message)

    def test_every_batch_respects_both_budgets(self, task_ds):
        """No non-oversize batch exceeds ``max_atoms`` or ``max_pairs``."""

        sampler = TaskMixtureBatchSampler(
            task_ds,
            indices=TASK_INDICES_NO_ELECTROSTATIC,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_B,
            max_atoms=24,
            max_pairs=34,
            seed=TASK_SEED,
        )
        atoms = BaseDataset.nrows_of(task_ds, "atoms")
        pairs = BaseDataset.nrows_of(task_ds, "pairs")
        for batch in sampler:
            # Largest unit costs 24 atoms / 32 edges, so nothing is oversize here.
            assert int(atoms[batch].sum()) <= 24
            assert int(pairs[batch].sum()) <= 34

    def test_edge_budget_closes_batches_the_atom_budget_would_not(self, task_ds):
        """The edge cap is honoured, not silently dropped.

        All 6 equilibrium units cost 30 atoms — one batch under
        ``max_atoms=100``. Their 42 edges exceed ``max_pairs=32`` after the
        4th unit (28 edges; a 5th makes 35), so the edge cap alone must split
        them into 2 batches.
        """

        sampler = TaskMixtureBatchSampler(
            task_ds,
            indices=TASK_INDICES_NO_ELECTROSTATIC,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_B,
            max_atoms=100,
            max_pairs=32,
            batches_per_epoch=6,
            seed=TASK_SEED,
        )
        assert sampler.batches_per_task()["equilibrium"] == 2
        pairs = BaseDataset.nrows_of(task_ds, "pairs")
        for batch in sampler:
            assert int(pairs[batch].sum()) <= 32

    def test_single_next_fit_loop_lives_in_the_packer(self):
        """The one open-bin accumulator appears once, inside 09's packer.

        Characterization — green already, like ``TestSamplerGoldenPlan``. It
        exists so that adding a second next-fit loop while implementing
        ``TaskMixtureBatchSampler`` turns this test red.
        """
        from molix.data import sampler as sampler_module

        source = Path(inspect.getsourcefile(sampler_module)).read_text()
        assert source.count("current.append(") == 1
        assert "current.append(" in inspect.getsource(TokenBudgetPacker)

    def test_task_mixture_sampler_has_no_second_packing_loop(self):
        """The sampler delegates instead of re-implementing next-fit."""

        source = inspect.getsource(TaskMixtureBatchSampler)
        for marker in ("current.append(", "batches.append(current)", "totals ="):
            assert marker not in source, f"second packing loop marker {marker!r}"
        assert "TokenBudgetPacker(" in source
        assert ".pack(" in source

    def test_zero_getitem_calls_during_construction_and_iteration(self, tmp_path):
        """Unit costs come from pointers only — no sample is unpacked."""

        ds = _task_ds()
        sampler = TaskMixtureBatchSampler(
            ds,
            indices=TASK_INDICES_ALL,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_C,
            max_atoms=24,
            max_pairs=34,
            batches_per_epoch=4,
            seed=TASK_SEED,
        )
        list(sampler)
        assert ds.getitem_calls == 0


class TestTaskMixtureMixture:
    """Largest-remainder apportionment and anti-starvation."""

    def test_stage_b_apportionment_bound_over_203_batches(self, big_task_fixture):
        """``|count_t - p_t·n| <= 1`` for Stage B, no override fired.

        Deterministic apportionment makes this a hard bound, strictly stronger
        than a χ² test on a zero-variance realization. Hand-derived at n=203:
        equilibrium 71.05 → 71, torsion 91.35 → 91, relax 40.6 → 40 + the single
        leftover (largest remainder 0.6) = 41; all ≤ the 96-unit supply.
        """

        ds, labels, groups = big_task_fixture
        non_electrostatic = [i for i, t in enumerate(labels) if t != "electrostatic"]
        sampler = TaskMixtureBatchSampler(
            ds,
            indices=non_electrostatic,
            task_labels=labels,
            group_ids=groups,
            mixture=STAGE_B,
            max_atoms=BIG_MAX_ATOMS,
            batches_per_epoch=BIG_BATCHES_PER_EPOCH,
            seed=TASK_SEED,
        )
        assert len(sampler) == BIG_BATCHES_PER_EPOCH
        counts = sampler.batches_per_task()
        for task, p in STAGE_B.items():
            assert abs(counts[task] - p * len(sampler)) <= 1

    def test_stage_c_apportionment_bound_over_203_batches(self, big_task_fixture):
        """Same hard bound for the four-task Stage C mixture."""

        ds, labels, groups = big_task_fixture
        sampler = TaskMixtureBatchSampler(
            ds,
            indices=list(range(len(labels))),
            task_labels=labels,
            group_ids=groups,
            mixture=STAGE_C,
            max_atoms=BIG_MAX_ATOMS,
            batches_per_epoch=BIG_BATCHES_PER_EPOCH,
            seed=TASK_SEED,
        )
        assert len(sampler) == BIG_BATCHES_PER_EPOCH
        counts = sampler.batches_per_task()
        for task, p in STAGE_C.items():
            assert abs(counts[task] - p * len(sampler)) <= 1

    def test_scarce_stage_c_task_receives_at_least_one_batch(self, task_ds):
        """Electrostatic (p=0.10, 1 unit) is never dropped at small n."""

        sampler = TaskMixtureBatchSampler(
            task_ds,
            indices=TASK_INDICES_ALL,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_C,
            max_atoms=24,
            batches_per_epoch=4,
            seed=TASK_SEED,
        )
        assert sampler.batches_per_task()["electrostatic"] >= 1

    def test_anti_starvation_donor_stays_within_one_plus_k(self, big_task_fixture):
        """The override fires for p=0.02 and the donor deviates ≤ 1+K.

        Hand-derived at n=10 for ``SCARCE_MIXTURE``: bases are (3, 3, 3, 0) for
        (equilibrium 3.0, torsion 3.5, relax 3.3, electrostatic 0.2); the single
        leftover goes to torsion (remainder 0.5) → torsion 4. electrostatic is
        starved, so K=1 and the largest-surplus donor (torsion, 4 - 3.5 = 0.5)
        gives one back: |4-1 - 3.5| = 0.5 ≤ 1 + K = 2.
        """

        ds, labels, groups = big_task_fixture
        n = 10
        sampler = TaskMixtureBatchSampler(
            ds,
            indices=list(range(len(labels))),
            task_labels=labels,
            group_ids=groups,
            mixture=SCARCE_MIXTURE,
            max_atoms=BIG_MAX_ATOMS,
            batches_per_epoch=n,
            seed=TASK_SEED,
        )
        counts = sampler.batches_per_task()
        assert counts["electrostatic"] >= 1
        rescued = sum(
            1
            for task, p in SCARCE_MIXTURE.items()
            if p > 0 and math.floor(p * n) == 0 and counts.get(task, 0) >= 1
        )
        assert rescued >= 1
        for task, p in SCARCE_MIXTURE.items():
            assert abs(counts.get(task, 0) - p * n) <= 1 + rescued


class TestTaskMixtureDeterminism:
    """The plan is a pure function of the constructor arguments."""

    def _build(self, dataset, seed: int):
        """Stage B sampler over the fixture split, at *seed*."""

        return TaskMixtureBatchSampler(
            dataset,
            indices=TASK_INDICES_NO_ELECTROSTATIC,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_B,
            max_atoms=24,
            seed=seed,
        )

    def test_same_arguments_give_identical_plans(self, task_ds):
        """Two instances with identical config agree."""
        assert list(self._build(task_ds, TASK_SEED)) == list(self._build(task_ds, TASK_SEED))

    def test_seed_plus_one_gives_a_different_plan(self, task_ds):
        """Reseeding actually reshuffles units and slots."""
        assert list(self._build(task_ds, TASK_SEED)) != list(self._build(task_ds, TASK_SEED + 1))

    def test_iterating_one_instance_twice_is_stable(self, task_ds):
        """``__iter__`` replays the cached eager plan."""
        sampler = self._build(task_ds, TASK_SEED)
        assert list(sampler) == list(sampler)

    def test_pickle_round_trip_gives_an_identical_plan(self, tmp_path):
        """No dataset-sized count tensor is stored on the instance."""
        ds = _task_ds()
        sampler = self._build(ds, TASK_SEED)
        restored = pickle.loads(pickle.dumps(sampler))
        assert list(restored) == list(sampler)
        assert restored.batches_per_task() == sampler.batches_per_task()

    def test_epoch_one_plan_is_derivable_from_seed_plus_one(self, task_ds):
        """Resume needs ``seed + epoch`` only, no carried RNG state."""
        epoch0 = list(self._build(task_ds, TASK_SEED))
        epoch1 = list(self._build(task_ds, TASK_SEED + 1))
        fresh_epoch1 = list(self._build(task_ds, TASK_SEED + 1))
        assert fresh_epoch1 == epoch1
        assert epoch1 != epoch0


class TestTaskMixtureValidation:
    """Eager ``ValueError``s naming argument, value and fix."""

    def _kwargs(self, **overrides):
        """Valid Stage B constructor kwargs, with *overrides* applied."""
        kwargs = dict(
            indices=TASK_INDICES_NO_ELECTROSTATIC,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_B,
            max_atoms=24,
            seed=TASK_SEED,
        )
        kwargs.update(overrides)
        return kwargs

    def test_no_budget_raises_naming_both_arguments(self, task_ds):
        """Neither budget given → the message names ``max_atoms`` and ``max_pairs``."""

        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(task_ds, **self._kwargs(max_atoms=None))
        message = str(excinfo.value)
        assert "max_atoms" in message
        assert "max_pairs" in message

    def test_zero_max_atoms_raises_naming_the_value(self, task_ds):
        """``max_atoms=0`` → names the argument and the offending value."""

        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(task_ds, **self._kwargs(max_atoms=0))
        message = str(excinfo.value)
        assert "max_atoms" in message
        assert "0" in message

    def test_negative_max_edges_raises_naming_the_value(self, task_ds):
        """``max_pairs=-5`` → names the argument and the offending value."""

        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(task_ds, **self._kwargs(max_pairs=-5))
        message = str(excinfo.value)
        assert "max_pairs" in message
        assert "-5" in message

    def test_mixture_not_summing_to_one_raises_naming_the_sum(self, task_ds):
        """0.5 + 0.25 + 0.5 = 1.25 exactly (binary-exact) → the sum is reported."""

        bad = {"equilibrium": 0.5, "torsion": 0.25, "relax": 0.5}
        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(task_ds, **self._kwargs(mixture=bad))
        message = str(excinfo.value)
        assert "mixture" in message
        assert "1.25" in message

    def test_mixture_sum_within_one_e_minus_nine_is_accepted(self, task_ds):
        """The tolerance is ``1e-9`` (``group_split.py:30``), not exact equality."""

        near = {"equilibrium": 0.35, "torsion": 0.45, "relax": 0.20 + 1e-12}
        sampler = TaskMixtureBatchSampler(task_ds, **self._kwargs(mixture=near))
        assert len(sampler) >= 1

    def test_negative_probability_raises_naming_task_and_value(self, task_ds):
        """``equilibrium=-0.1`` (sum still 1.0) → names the task and value."""

        bad = {"equilibrium": -0.1, "torsion": 0.6, "relax": 0.5}
        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(task_ds, **self._kwargs(mixture=bad))
        message = str(excinfo.value)
        assert "equilibrium" in message
        assert "-0.1" in message

    def test_probability_above_one_raises_naming_task_and_value(self, task_ds):
        """``equilibrium=1.5`` (sum still 1.0) → names the task and value."""

        bad = {"equilibrium": 1.5, "torsion": -0.5, "relax": 0.0}
        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(task_ds, **self._kwargs(mixture=bad))
        message = str(excinfo.value)
        assert "equilibrium" in message or "torsion" in message
        assert "1.5" in message or "-0.5" in message

    def test_empty_mixture_raises_naming_the_argument(self, task_ds):
        """``mixture={}`` has no task to draw → eager ``ValueError``."""

        with pytest.raises(ValueError, match="mixture"):
            TaskMixtureBatchSampler(task_ds, **self._kwargs(mixture={}))

    def test_split_label_absent_from_mixture_raises(self, task_ds):
        """4 electrostatic samples in the split, no such key in Stage B."""

        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(task_ds, **self._kwargs(indices=TASK_INDICES_ALL))
        message = str(excinfo.value)
        assert "electrostatic" in message
        assert "4" in message

    def test_task_with_probability_but_no_units_raises(self, task_ds):
        """Stage C over an electrostatic-free split → names task and probability."""

        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(task_ds, **self._kwargs(mixture=STAGE_C))
        message = str(excinfo.value)
        assert "electrostatic" in message
        assert "0.1" in message

    def test_task_labels_length_mismatch_raises_naming_both_lengths(self, task_ds):
        """``len(task_labels)`` must equal ``len(dataset)``."""

        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(task_ds, **self._kwargs(task_labels=TASK_LABELS[:20]))
        message = str(excinfo.value)
        assert "task_labels" in message
        assert "20" in message
        assert "24" in message

    def test_group_ids_length_mismatch_raises_naming_both_lengths(self, task_ds):
        """``len(group_ids)`` must equal ``len(dataset)``."""

        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(task_ds, **self._kwargs(group_ids=GROUP_IDS[:20]))
        message = str(excinfo.value)
        assert "group_ids" in message
        assert "20" in message
        assert "24" in message

    def test_empty_indices_raises_naming_the_argument(self, task_ds):
        """``indices=[]`` cannot form a single batch."""

        with pytest.raises(ValueError, match="indices"):
            TaskMixtureBatchSampler(task_ds, **self._kwargs(indices=[]))

    def test_out_of_range_index_raises_naming_the_value(self, task_ds):
        """An index ≥ ``len(dataset)`` is reported with the valid range.

        The other four indices keep one unit per Stage B task, so the
        out-of-range index is the *only* invalid thing about this call.
        """

        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(task_ds, **self._kwargs(indices=[0, 1, 6, 15, 99]))
        message = str(excinfo.value)
        assert "indices" in message
        assert "99" in message


class TestTaskMixtureAudit:
    """Read-only audit accessors and the exhaustion path."""

    def _sampler(self, dataset, **overrides):
        """Stage C sampler covering all four tasks (4 slots, no exhaustion)."""

        kwargs = dict(
            indices=TASK_INDICES_ALL,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_C,
            max_atoms=24,
            batches_per_epoch=4,
            seed=TASK_SEED,
        )
        kwargs.update(overrides)
        return TaskMixtureBatchSampler(dataset, **kwargs)

    def test_batches_per_task_sums_to_len(self, task_ds):
        """Every planned batch is attributed to exactly one task."""
        sampler = self._sampler(task_ds)
        assert sum(sampler.batches_per_task().values()) == len(sampler)

    def test_frames_per_task_sums_to_the_planned_frame_count(self, task_ds):
        """Every planned frame is attributed to exactly one task."""
        sampler = self._sampler(task_ds)
        total = sum(len(batch) for batch in sampler)
        assert sum(sampler.frames_per_task().values()) == total

    def test_audit_dicts_match_a_direct_recount_of_the_plan(self, task_ds):
        """The accessors report the realized plan, not the request."""
        sampler = self._sampler(task_ds)
        batches: dict[str, int] = {}
        frames: dict[str, int] = {}
        for batch in sampler:
            task = TASK_LABELS[batch[0]]
            batches[task] = batches.get(task, 0) + 1
            frames[task] = frames.get(task, 0) + len(batch)
        assert sampler.batches_per_task() == batches
        assert sampler.frames_per_task() == frames

    def test_mutating_a_returned_dict_does_not_corrupt_the_plan(self, task_ds):
        """Accessors hand back fresh copies, not internal state."""
        sampler = self._sampler(task_ds)
        snapshot = sampler.batches_per_task()
        plan = list(sampler)
        sampler.batches_per_task()["equilibrium"] = 999
        sampler.frames_per_task().clear()
        assert sampler.batches_per_task() == snapshot
        assert list(sampler) == plan

    def test_exhaustion_warns_exactly_once_naming_the_exhausted_task(self, task_ds):
        """``batches_per_epoch=7`` demands 3 equilibrium batches of 2."""

        with pytest.warns(UserWarning) as record:
            TaskMixtureBatchSampler(
                task_ds,
                indices=TASK_INDICES_NO_ELECTROSTATIC,
                task_labels=TASK_LABELS,
                group_ids=GROUP_IDS,
                mixture=STAGE_B,
                max_atoms=24,
                batches_per_epoch=7,
                seed=TASK_SEED,
            )
        exhausted = [w for w in record if "equilibrium" in str(w.message)]
        assert len(exhausted) == 1

    def test_exhausted_plan_length_and_composition_are_hand_computed(self, task_ds):
        """Apportionment (3, 3, 1) vs supply (2, 3, 1) → 6 realized slots.

        At n=7 Stage B apportions equilibrium 2.45 → 2 + the leftover (largest
        remainder 0.45) = 3, torsion 3.15 → 3, relax 1.4 → 1. Equilibrium only
        supplies 2 batches, and neither torsion (3 of 3) nor relax (1 of 1) can
        absorb the renormalized slot, so the plan ends one short at 6 and every
        unit of the split is used exactly once.
        """

        with pytest.warns(UserWarning):
            sampler = TaskMixtureBatchSampler(
                task_ds,
                indices=TASK_INDICES_NO_ELECTROSTATIC,
                task_labels=TASK_LABELS,
                group_ids=GROUP_IDS,
                mixture=STAGE_B,
                max_atoms=24,
                batches_per_epoch=7,
                seed=TASK_SEED,
            )
        assert len(sampler) == 6
        assert sampler.batches_per_task() == {"equilibrium": 2, "torsion": 3, "relax": 1}
        assert sampler.frames_per_task() == {"equilibrium": 6, "torsion": 9, "relax": 5}


# ---------------------------------------------------------------------------
# GROUP F — FrameLabelTable and the two sampler table forms
# (spec streaming-data-tiers-04-scheduler-integration-01-tables)
# ---------------------------------------------------------------------------
#
# The block store already emits its scheduler metadata in bounded integer form
# (per-frame ``task_codes`` / ``group_codes`` ``(F,) int32`` plus the
# ``task_names`` dictionary), so the GROUP E fixture is written here a second
# time as codes.  Both code tables are hand-transcribed from the GROUP E table
# above, in first-appearance order — the order ``FrameLabelTable.from_labels``
# must reproduce from the string tables:
#
#   task codes    equilibrium 0 (×6), torsion 1 (×9), relax 2 (×5),
#                 electrostatic 3 (×4)
#   group codes   eq0..eq5 → 0..5, tor_a → 6, tor_b → 7, tor_c → 8,
#                 rel_a → 9, rel_b → 10, ele_a → 11, ele_b → 12

#: Bounded task dictionary; the index *is* the task code stored in a block.
TASK_NAMES = ("equilibrium", "torsion", "relax", "electrostatic")
#: Bounded group dictionary, same first-appearance order as ``GROUP_IDS``.
GROUP_NAMES = tuple(f"eq{i}" for i in range(6)) + (
    "tor_a",
    "tor_b",
    "tor_c",
    "rel_a",
    "rel_b",
    "ele_a",
    "ele_b",
)
#: Group code of ``tor_a`` — the unit that is oversize at ``max_atoms=20``.
TOR_A_GROUP_CODE = 6
#: ``(F,) int32`` task codes: the block-native form of ``TASK_LABELS``.
TASK_CODES = torch.tensor([0] * 6 + [1] * 9 + [2] * 5 + [3] * 4, dtype=torch.int32)
#: ``(F,) int32`` group codes: the block-native form of ``GROUP_IDS``.
GROUP_CODES = torch.tensor(
    [0, 1, 2, 3, 4, 5] + [6] * 4 + [7] * 3 + [8] * 2 + [9] * 3 + [10] * 2 + [11] * 2 + [12] * 2,
    dtype=torch.int32,
)
#: Every parent-space index, for whole-table ``codes_at`` round-trips.
ALL_FRAMES = list(range(len(TASK_LABELS)))


def _frame_label_table() -> type:
    """Return ``molix.data.sampler.FrameLabelTable``, imported at call time.

    Deferred on purpose: while the type does not exist yet the RED failure is
    an ``ImportError`` *inside* the GROUP F tests, so collection of this module
    — and therefore every pre-existing GROUP A-E test — stays unaffected.
    """
    from molix.data.sampler import FrameLabelTable

    return FrameLabelTable


class TestFrameLabelTable:
    """Per-frame ``(task, group)`` identity of one parent index space."""

    def test_from_labels_factorizes_in_first_appearance_order(self):
        """The string tables factorize into exactly the block-native codes."""

        table = _frame_label_table().from_labels(TASK_LABELS, GROUP_IDS)
        tasks, groups = table.codes_at(ALL_FRAMES)
        assert tasks == TASK_CODES.tolist()
        assert groups == GROUP_CODES.tolist()

    def test_from_labels_keeps_the_task_dictionary_in_first_appearance_order(self):
        """``task_names`` is the bounded dictionary, not a sorted set."""

        table = _frame_label_table().from_labels(TASK_LABELS, GROUP_IDS)
        assert table.task_names == TASK_NAMES

    def test_len_is_the_frame_count(self):
        """``len(table)`` is ``F``, the parent index space size."""

        table = _frame_label_table().from_labels(TASK_LABELS, GROUP_IDS)
        assert len(table) == 24

    def test_task_name_round_trips_every_frame_label(self):
        """Decoding every task code reproduces ``TASK_LABELS``."""

        table = _frame_label_table().from_labels(TASK_LABELS, GROUP_IDS)
        tasks, _ = table.codes_at(ALL_FRAMES)
        assert [table.task_name(code) for code in tasks] == TASK_LABELS

    def test_group_label_round_trips_every_frame_group_id(self):
        """Decoding every group code reproduces ``GROUP_IDS``."""

        table = _frame_label_table().from_labels(TASK_LABELS, GROUP_IDS)
        _, groups = table.codes_at(ALL_FRAMES)
        assert [table.group_label(code) for code in groups] == GROUP_IDS

    def test_canonical_constructor_takes_the_block_native_int32_columns(self):
        """``(F,) int32`` columns + a dictionary are accepted verbatim."""

        table = _frame_label_table()(
            task_codes=TASK_CODES,
            group_codes=GROUP_CODES,
            task_names=TASK_NAMES,
        )
        assert len(table) == 24
        assert table.task_name(1) == "torsion"

    def test_group_label_falls_back_to_str_of_the_code_without_a_dictionary(self):
        """No ``group_names`` → the code renders as ``str(code)``."""

        table = _frame_label_table()(
            task_codes=TASK_CODES,
            group_codes=GROUP_CODES,
            task_names=TASK_NAMES,
        )
        assert table.group_label(TOR_A_GROUP_CODE) == "6"

    def test_group_label_uses_the_dictionary_when_group_names_are_given(self):
        """With ``group_names`` the original label comes back."""

        table = _frame_label_table()(
            task_codes=TASK_CODES,
            group_codes=GROUP_CODES,
            task_names=TASK_NAMES,
            group_names=GROUP_NAMES,
        )
        assert table.group_label(TOR_A_GROUP_CODE) == "tor_a"

    def test_codes_at_returns_python_int_pairs_in_the_given_order(self):
        """One gather per table, in the caller's order, Python ints out."""

        table = _frame_label_table().from_labels(TASK_LABELS, GROUP_IDS)
        tasks, groups = table.codes_at([13, 0, 20, 6])
        assert tasks == [1, 0, 3, 1]
        assert groups == [8, 0, 11, 6]
        assert all(type(code) is int for code in (*tasks, *groups))

    def test_unequal_table_lengths_raise_naming_both_arguments_and_lengths(self):
        """24 task codes vs 20 group codes → both names, both lengths."""

        with pytest.raises(ValueError) as excinfo:
            _frame_label_table()(
                task_codes=TASK_CODES,
                group_codes=GROUP_CODES[:20],
                task_names=TASK_NAMES,
            )
        message = str(excinfo.value)
        assert "task_codes" in message
        assert "group_codes" in message
        assert "24" in message
        assert "20" in message

    def test_two_dimensional_code_tensor_raises_naming_the_argument(self):
        """The code tables are ``(F,)``, never ``(4, 6)``."""

        with pytest.raises(ValueError) as excinfo:
            _frame_label_table()(
                task_codes=TASK_CODES.reshape(4, 6),
                group_codes=GROUP_CODES,
                task_names=TASK_NAMES,
            )
        message = str(excinfo.value)
        assert "task_codes" in message
        assert "1-D" in message or "(F,)" in message

    def test_floating_code_tensor_raises_naming_the_dtype(self):
        """A float table cannot index a dictionary — reject, never round."""

        with pytest.raises(ValueError) as excinfo:
            _frame_label_table()(
                task_codes=TASK_CODES,
                group_codes=GROUP_CODES.to(torch.float32),
                task_names=TASK_NAMES,
            )
        message = str(excinfo.value)
        assert "group_codes" in message
        assert "integ" in message.lower() or "dtype" in message.lower()

    def test_non_cpu_code_tensor_raises_naming_cpu(self):
        """Tables live on CPU (a ``meta`` tensor stands in for CUDA)."""

        elsewhere = torch.zeros(len(TASK_LABELS), dtype=torch.int32, device="meta")
        with pytest.raises(ValueError) as excinfo:
            _frame_label_table()(
                task_codes=elsewhere,
                group_codes=GROUP_CODES,
                task_names=TASK_NAMES,
            )
        message = str(excinfo.value)
        assert "task_codes" in message
        assert "cpu" in message.lower()

    def test_negative_code_raises_naming_the_argument(self):
        """Codes index dictionaries, so ``-1`` is a table bug, not a wrap."""

        codes = GROUP_CODES.clone()
        codes[3] = -1
        with pytest.raises(ValueError) as excinfo:
            _frame_label_table()(
                task_codes=TASK_CODES,
                group_codes=codes,
                task_names=TASK_NAMES,
            )
        message = str(excinfo.value)
        assert "group_codes" in message
        assert "-1" in message or "negative" in message.lower()

    def test_task_code_outside_the_dictionary_raises_naming_task_names(self):
        """Code 3 with a 2-entry dictionary → names the dictionary and both."""

        with pytest.raises(ValueError) as excinfo:
            _frame_label_table()(
                task_codes=TASK_CODES,
                group_codes=GROUP_CODES,
                task_names=TASK_NAMES[:2],
            )
        message = str(excinfo.value)
        assert "task_names" in message
        assert "3" in message
        assert "2" in message


class TestTaskMixtureTableForms:
    """One sampler, two label representations."""

    def _string_form(self, dataset, **overrides) -> TaskMixtureBatchSampler:
        """The legacy Stage B sampler over the fixture split, string tables."""

        kwargs = dict(
            indices=TASK_INDICES_NO_ELECTROSTATIC,
            task_labels=TASK_LABELS,
            group_ids=GROUP_IDS,
            mixture=STAGE_B,
            max_atoms=24,
            seed=TASK_SEED,
        )
        kwargs.update(overrides)
        return TaskMixtureBatchSampler(dataset, **kwargs)

    def _code_form(self, dataset, **overrides) -> TaskMixtureBatchSampler:
        """The same sampler over the block-native ``(F,)`` code columns."""

        kwargs = dict(
            indices=TASK_INDICES_NO_ELECTROSTATIC,
            task_labels=TASK_CODES,
            group_ids=GROUP_CODES,
            task_names=TASK_NAMES,
            mixture=STAGE_B,
            max_atoms=24,
            seed=TASK_SEED,
        )
        kwargs.update(overrides)
        return TaskMixtureBatchSampler(dataset, **kwargs)

    def test_code_form_plan_equals_the_string_form_plan(self, task_ds):
        """Two representations, one plan, at ``seed=TASK_SEED``."""

        assert list(self._code_form(task_ds)) == list(self._string_form(task_ds))

    def test_code_form_batches_per_task_equals_the_string_form(self, task_ds):
        """The batch-share audit dict is representation-independent."""

        code_form = self._code_form(task_ds).batches_per_task()
        assert code_form == self._string_form(task_ds).batches_per_task()

    def test_code_form_frames_per_task_equals_the_string_form(self, task_ds):
        """The frame-share audit dict is representation-independent."""

        code_form = self._code_form(task_ds).frames_per_task()
        assert code_form == self._string_form(task_ds).frames_per_task()

    def test_oversize_warning_names_the_group_code_without_a_name_dictionary(self, task_ds):
        """No group dictionary → the warning names ``torsion/6``."""

        with pytest.warns(UserWarning) as record:
            self._code_form(task_ds, max_atoms=20, batches_per_epoch=6)
        oversize = [w for w in record if f"torsion/{TOR_A_GROUP_CODE}" in str(w.message)]
        assert len(oversize) == 1

    def test_oversize_warning_names_the_group_label_when_group_names_are_given(self, task_ds):
        """With a group dictionary the code form names ``torsion/tor_a``."""

        with pytest.warns(UserWarning) as record:
            self._code_form(
                task_ds,
                max_atoms=20,
                batches_per_epoch=6,
                group_names=GROUP_NAMES,
            )
        oversize = [w for w in record if "torsion/tor_a" in str(w.message)]
        assert len(oversize) == 1

    def test_mutating_the_caller_code_tensors_changes_neither_plan_nor_audit(self, task_ds):
        """The sampler copies the tables it plans from."""

        task_codes = TASK_CODES.clone()
        group_codes = GROUP_CODES.clone()
        sampler = self._code_form(task_ds, task_labels=task_codes, group_ids=group_codes)
        plan = list(sampler)
        batches = sampler.batches_per_task()
        frames = sampler.frames_per_task()

        task_codes.fill_(0)
        group_codes.fill_(0)

        assert list(sampler) == plan
        assert sampler.batches_per_task() == batches
        assert sampler.frames_per_task() == frames

    def test_no_parent_space_table_is_retained_on_the_instance(self, task_ds):
        """The R1 win — no ``(F,)``-length object survives construction.

        The dataset reference is exempt (it *is* ``len(dataset)`` long by
        definition); every other retained attribute must be shorter.
        """

        sampler = self._code_form(task_ds)
        slots = getattr(type(sampler), "__slots__", ())
        retained = list(vars(sampler).items()) if hasattr(sampler, "__dict__") else []
        retained += [(name, getattr(sampler, name)) for name in slots]
        for name, value in retained:
            if value is task_ds:
                continue
            try:
                size = len(value)
            except TypeError:
                continue
            assert size != len(task_ds), f"sampler retains a parent-space table in {name!r}"

    def test_integer_task_codes_without_task_names_raise_naming_task_names(self, task_ds):
        """Integer codes are undecodable without their dictionary."""

        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(
                task_ds,
                indices=TASK_INDICES_NO_ELECTROSTATIC,
                task_labels=TASK_CODES,
                group_ids=GROUP_CODES,
                mixture=STAGE_B,
                max_atoms=24,
                seed=TASK_SEED,
            )
        message = str(excinfo.value)
        assert "task_names" in message
        assert "task_labels" in message

    def test_task_names_with_string_labels_raise_naming_both_arguments(self, task_ds):
        """String labels already carry their names — a dictionary is a bug."""

        with pytest.raises(ValueError) as excinfo:
            TaskMixtureBatchSampler(
                task_ds,
                indices=TASK_INDICES_NO_ELECTROSTATIC,
                task_labels=TASK_LABELS,
                group_ids=GROUP_IDS,
                task_names=TASK_NAMES,
                mixture=STAGE_B,
                max_atoms=24,
                seed=TASK_SEED,
            )
        message = str(excinfo.value)
        assert "task_names" in message
        assert "task_labels" in message

    def test_espaloma_shaped_construction_yields_a_non_empty_plan(self, task_ds):
        """``EnergyTaskSamplerSource.batch_sampler``'s exact call shape.

        Tuple string tables, ``indices=range(len(dataset))`` and a single-task
        mixture at probability 1.0 (``scripts/mm_param_learning/
        espaloma_mm_protocol_train.py:560-568``) must keep working verbatim.
        """

        sampler = TaskMixtureBatchSampler(
            task_ds,
            indices=range(len(task_ds)),
            task_labels=("energy",) * len(task_ds),
            group_ids=tuple(GROUP_IDS),
            mixture={"energy": 1.0},
            max_atoms=24,
            seed=TASK_SEED,
        )
        assert len(sampler) >= 1
        assert all(len(batch) >= 1 for batch in sampler)
