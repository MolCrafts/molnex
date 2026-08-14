"""RED tests for the token-budget dynamic batch sampler stack.

Covers spec ``dynamic-batching-packed-collate-01-sampler``:

* GROUP A — pointer-derived count accessors (``atom_counts`` /
  ``edge_counts``) on ``_CacheBacked`` datasets and ``SubsetDataset``.
* GROUP B — ``molix.data.sampler.TokenBudgetBatchSampler`` behaviour.
* GROUP C — ``DataModule`` opt-in wiring via ``max_atoms_per_batch`` /
  ``max_edges_per_batch``.

Spec ``universal-class-i-09-packer`` adds GROUP D: characterization
goldens pinning the pre-refactor batch plans plus RED tests for the
extracted ``molix.data.TokenBudgetPacker``.

Fixture geometry (hand-checkable): sample ``i`` has ``3 + (i % 8)`` atoms
(cycling 3..10) and ``n_atoms + 2`` edges (cycling 5..12). Edge counts
never equal atom counts, so the packed-cache schema inference cannot
misclassify per-edge keys as per-atom.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.data import BatchSampler, Dataset, RandomSampler

from molix.core.seed import make_generator
from molix.data import TokenBudgetPacker
from molix.data.cache import PackedCache
from molix.data.datamodule import DataModule
from molix.data.dataset import CachedDataset, MmapDataset, SubsetDataset
from molix.data.sampler import TokenBudgetBatchSampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SAMPLES = 64
#: Hand-computed per-sample atom counts: 3, 4, ..., 10, 3, 4, ... (cycle of 8).
ATOM_COUNTS = [3 + (i % 8) for i in range(N_SAMPLES)]
#: Hand-computed per-sample edge counts: always atoms + 2 (5..12).
EDGE_COUNTS = [c + 2 for c in ATOM_COUNTS]


def _make_varied_samples(n: int) -> list[dict]:
    """Build ``n`` samples whose atom/edge counts follow the module tables.

    Args:
        n: Number of samples (``ATOM_COUNTS`` / ``EDGE_COUNTS`` prefix used).

    Returns:
        Flat sample dicts with per-atom (``Z``, ``pos``), per-edge
        (``edge_index``, ``edge_diff``, ``edge_dist``) and graph-level
        (``targets.U0`` — unique identity ``float(i)``) keys.
    """
    torch.manual_seed(0)
    samples = []
    for i in range(n):
        na = 3 + (i % 8)
        ne = na + 2
        samples.append(
            {
                "Z": torch.ones(na, dtype=torch.long),
                "pos": torch.randn(na, 3),
                "edge_index": torch.randint(0, na, (ne, 2), dtype=torch.long),
                "edge_diff": torch.randn(ne, 3),
                "edge_dist": torch.rand(ne) + 0.5,
                "targets": {"U0": torch.tensor([float(i)])},
            }
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


def _write_varied_cache(tmp_path: Path, tag: str, n: int = N_SAMPLES) -> Path:
    """Save a varied-count packed cache under *tmp_path* and return its sink."""
    sink = tmp_path / f"varied_{tag}.pt"
    PackedCache(sink).save(_make_varied_samples(n))
    return sink


@pytest.fixture
def varied_ds(tmp_path) -> MmapDataset:
    """An mmap-backed dataset with the hand-checkable count distribution."""
    return MmapDataset(_write_varied_cache(tmp_path, "main"))


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
    train_sink = tmp_path / f"budget_train_{tag}.pt"
    val_sink = tmp_path / f"budget_val_{tag}.pt"
    PackedCache(train_sink).save(samples[:N_SAMPLES])
    PackedCache(val_sink).save(samples[N_SAMPLES:])
    dm_kwargs.setdefault("batch_size", 4)
    dm_kwargs.setdefault("num_workers", 0)
    dm_kwargs.setdefault("pin_memory", False)
    return DataModule(MmapDataset(train_sink), MmapDataset(val_sink), seed=seed, **dm_kwargs)


class _NonPackedDataset(Dataset):
    """Minimal torch dataset with no packed-cache backing (no count accessors)."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        return {"Z": torch.ones(2, dtype=torch.long), "pos": torch.zeros(2, 3)}


class _CountingDataset(MmapDataset):
    """MmapDataset that counts ``__getitem__`` calls (no-unpack constraint)."""

    def __init__(self, sink: Path) -> None:
        super().__init__(sink)
        self.getitem_calls = 0

    def __getitem__(self, idx: int) -> dict:
        self.getitem_calls += 1
        return super().__getitem__(idx)


# ---------------------------------------------------------------------------
# GROUP A — pointer-derived count accessors
# ---------------------------------------------------------------------------


class TestCountAccessors:
    """``atom_counts`` / ``edge_counts`` on ``_CacheBacked`` datasets."""

    def test_atom_counts_values_shape_dtype(self, varied_ds):
        """atom_counts == atom_ptr[1:] - atom_ptr[:-1], long dtype, shape (n,)."""
        counts = varied_ds.atom_counts
        assert isinstance(counts, torch.Tensor)
        assert counts.dtype == torch.long
        assert counts.shape == (N_SAMPLES,)
        assert counts.tolist() == ATOM_COUNTS

    def test_edge_counts_values_shape_dtype(self, varied_ds):
        """edge_counts == edge_ptr[1:] - edge_ptr[:-1], long dtype, shape (n,)."""
        counts = varied_ds.edge_counts
        assert isinstance(counts, torch.Tensor)
        assert counts.dtype == torch.long
        assert counts.shape == (N_SAMPLES,)
        assert counts.tolist() == EDGE_COUNTS

    def test_counts_match_pointer_diff(self, varied_ds):
        """Accessors agree with the raw cumsum pointers in the payload."""
        atom_ptr = varied_ds._payload["atom_ptr"]
        edge_ptr = varied_ds._payload["edge_ptr"]
        assert torch.equal(varied_ds.atom_counts, atom_ptr[1:] - atom_ptr[:-1])
        assert torch.equal(varied_ds.edge_counts, edge_ptr[1:] - edge_ptr[:-1])

    def test_cached_dataset_counts_match(self, tmp_path):
        """CachedDataset (mmap=False) exposes the same counts as MmapDataset."""
        sink = _write_varied_cache(tmp_path, "cached")
        ds = CachedDataset(sink)
        assert ds.atom_counts.tolist() == ATOM_COUNTS
        assert ds.edge_counts.tolist() == EDGE_COUNTS


class TestSubsetCounts:
    """``SubsetDataset`` counts must be gathered at packed indices, not forwarded."""

    def test_subset_atom_counts_remap(self, varied_ds):
        """Subset counts are the parent counts at the subset's packed indices."""
        subset = SubsetDataset(varied_ds, [5, 2, 9])
        counts = subset.atom_counts
        # Shape guard: __getattr__ forwarding would return the FULL (64,) vector.
        assert counts.shape == (3,)
        assert counts.dtype == torch.long
        assert counts.tolist() == [ATOM_COUNTS[5], ATOM_COUNTS[2], ATOM_COUNTS[9]]

    def test_subset_edge_counts_remap(self, varied_ds):
        """Same remap semantics for edge_counts."""
        subset = SubsetDataset(varied_ds, [7, 0, 31, 12])
        counts = subset.edge_counts
        assert counts.shape == (4,)
        assert counts.tolist() == [EDGE_COUNTS[i] for i in [7, 0, 31, 12]]

    def test_nested_subset_counts(self, varied_ds):
        """A subset of a subset composes the index remap correctly."""
        outer = SubsetDataset(varied_ds, [5, 2, 9, 40])
        inner = SubsetDataset(outer, [2, 0])  # packed indices 9, 5
        assert inner.atom_counts.tolist() == [ATOM_COUNTS[9], ATOM_COUNTS[5]]
        assert inner.edge_counts.tolist() == [EDGE_COUNTS[9], EDGE_COUNTS[5]]

    def test_split_subset_counts(self, varied_ds):
        """``split()`` views report counts at their own packed indices."""
        train, rest = varied_ds.split(sizes=(48, 16), seed=7)
        expected = [ATOM_COUNTS[i] for i in train._indices]
        assert train.atom_counts.tolist() == expected
        assert rest.atom_counts.shape == (16,)


class TestMissingPointer:
    """Caches without per-edge keys have no ``edge_ptr`` — actionable error."""

    def test_edge_counts_missing_edge_ptr_raises(self, tmp_path):
        """edge_counts on an edge-less cache raises ValueError naming the pointer."""
        sink = tmp_path / "edgeless.pt"
        PackedCache(sink).save(_make_edgeless_samples(8))
        ds = MmapDataset(sink)
        with pytest.raises(ValueError, match="edge_ptr"):
            _ = ds.edge_counts

    def test_atom_counts_still_work_on_edgeless_cache(self, tmp_path):
        """atom_counts is independent of edge keys."""
        sink = tmp_path / "edgeless2.pt"
        PackedCache(sink).save(_make_edgeless_samples(8))
        ds = MmapDataset(sink)
        assert ds.atom_counts.tolist() == ATOM_COUNTS[:8]


# ---------------------------------------------------------------------------
# GROUP B — TokenBudgetBatchSampler
# ---------------------------------------------------------------------------


class TestSamplerBudgetCompliance:
    """Every yielded batch must respect the configured budgets."""

    def test_atom_budget_respected(self, varied_ds):
        """Atom-only budget: counts[batch].sum() <= max_atoms for every batch."""
        sampler = TokenBudgetBatchSampler(varied_ds, max_atoms=16, seed=0)
        counts = torch.tensor(ATOM_COUNTS, dtype=torch.long)
        for batch in sampler:
            assert len(batch) >= 1
            assert int(counts[batch].sum()) <= 16

    def test_edge_budget_respected(self, varied_ds):
        """Edge-only budget: counts[batch].sum() <= max_edges for every batch."""
        sampler = TokenBudgetBatchSampler(varied_ds, max_edges=20, seed=0)
        counts = torch.tensor(EDGE_COUNTS, dtype=torch.long)
        for batch in sampler:
            assert int(counts[batch].sum()) <= 20

    def test_both_budgets_respected_simultaneously(self, varied_ds):
        """Dual budget: each batch satisfies BOTH limits at once."""
        sampler = TokenBudgetBatchSampler(varied_ds, max_atoms=16, max_edges=20, seed=0)
        atom_counts = torch.tensor(ATOM_COUNTS, dtype=torch.long)
        edge_counts = torch.tensor(EDGE_COUNTS, dtype=torch.long)
        for batch in sampler:
            assert int(atom_counts[batch].sum()) <= 16
            assert int(edge_counts[batch].sum()) <= 20


class TestSamplerCoverage:
    """Exact-once coverage and ``__len__`` consistency."""

    def test_exact_once_coverage(self, varied_ds):
        """Concatenated batch indices over one epoch == sorted range(n)."""
        sampler = TokenBudgetBatchSampler(varied_ds, max_atoms=16, seed=3)
        seen = [i for batch in sampler for i in batch]
        assert sorted(seen) == list(range(N_SAMPLES))

    def test_len_matches_yielded_batches(self, varied_ds):
        """len(sampler) equals the number of batches actually yielded."""
        sampler = TokenBudgetBatchSampler(varied_ds, max_atoms=16, seed=3)
        assert len(sampler) == len(list(sampler))


class TestSamplerOversize:
    """Samples exceeding the budget on their own become kept singletons."""

    def test_oversize_sample_is_singleton_not_dropped(self, varied_ds):
        """Each over-budget sample sits alone in its batch; coverage stays exact."""
        # Budget 8 < max sample count 10 → samples with 9 or 10 atoms oversize.
        with pytest.warns(UserWarning):
            sampler = TokenBudgetBatchSampler(varied_ds, max_atoms=8, seed=1)
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
            sampler = TokenBudgetBatchSampler(varied_ds, max_atoms=8, seed=1)
            list(sampler)
        budget_warnings = [w for w in record if "budget" in str(w.message).lower()]
        assert len(budget_warnings) == 1


class TestSamplerDeterminism:
    """Batch composition is a pure function of (dataset, budgets, seed)."""

    def test_same_seed_same_batches(self, varied_ds):
        """Two instances with identical config yield identical batch lists."""
        a = TokenBudgetBatchSampler(varied_ds, max_atoms=16, seed=123)
        b = TokenBudgetBatchSampler(varied_ds, max_atoms=16, seed=123)
        assert list(a) == list(b)

    def test_different_seed_different_batches(self, varied_ds):
        """seed vs seed+1 produce different batch composition."""
        a = TokenBudgetBatchSampler(varied_ds, max_atoms=16, seed=123)
        b = TokenBudgetBatchSampler(varied_ds, max_atoms=16, seed=124)
        # 64! permutations — a seeded collision is astronomically unlikely.
        assert list(a) != list(b)

    def test_iteration_stable_within_instance(self, varied_ds):
        """Iterating the same instance twice yields the cached composition."""
        sampler = TokenBudgetBatchSampler(varied_ds, max_atoms=16, seed=5)
        assert list(sampler) == list(sampler)


class TestSamplerValidation:
    """Eager (``__init__``-time) validation with actionable messages."""

    def test_no_budget_raises(self, varied_ds):
        """At least one of max_atoms / max_edges must be given."""
        with pytest.raises(ValueError, match="max_atoms|max_edges"):
            TokenBudgetBatchSampler(varied_ds, seed=0)

    def test_zero_budget_raises(self, varied_ds):
        """max_atoms=0 is rejected eagerly, naming the parameter."""
        with pytest.raises(ValueError, match="max_atoms"):
            TokenBudgetBatchSampler(varied_ds, max_atoms=0, seed=0)

    def test_negative_budget_raises(self, varied_ds):
        """max_edges<0 is rejected eagerly, naming the parameter."""
        with pytest.raises(ValueError, match="max_edges"):
            TokenBudgetBatchSampler(varied_ds, max_edges=-5, seed=0)

    def test_non_packed_dataset_raises_naming_supported_types(self):
        """Datasets without count accessors raise, listing supported types."""
        with pytest.raises(ValueError) as excinfo:
            TokenBudgetBatchSampler(_NonPackedDataset(), max_atoms=16, seed=0)
        message = str(excinfo.value)
        assert "MmapDataset" in message
        assert "CachedDataset" in message
        assert "SubsetDataset" in message


class TestSamplerNoUnpack:
    """The sampler must never unpack samples — pointer arithmetic only."""

    def test_zero_getitem_calls_during_construction_and_iteration(self, tmp_path):
        """__getitem__ is called 0 times by __init__ + one full epoch."""
        ds = _CountingDataset(_write_varied_cache(tmp_path, "counting"))
        sampler = TokenBudgetBatchSampler(ds, max_atoms=16, max_edges=24, seed=0)
        list(sampler)
        assert ds.getitem_calls == 0


class TestSamplerPickle:
    """Sampler must survive pickling (spawn-safe DataModule transfer)."""

    def test_pickle_round_trip_same_batches(self, tmp_path):
        """pickle.loads(pickle.dumps(s)) yields the identical batch list."""
        ds = CachedDataset(_write_varied_cache(tmp_path, "pickle"))
        sampler = TokenBudgetBatchSampler(ds, max_atoms=16, seed=11)
        restored = pickle.loads(pickle.dumps(sampler))
        assert list(restored) == list(sampler)


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
            total_graphs += batch["graphs", "U0"].shape[0]
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
#   from molix.data.dataset import MmapDataset
#   from molix.data.sampler import TokenBudgetBatchSampler as S
#   print(torch.randperm(16, generator=make_generator(42)).tolist())
#   with tempfile.TemporaryDirectory() as td:
#       ds = MmapDataset(_write_varied_cache(Path(td), 'golden', n=16))
#       print(list(S(ds, max_atoms=16)))
#       print(list(S(ds, max_atoms=16, max_edges=20)))
#       warnings.simplefilter('ignore'); print(list(S(ds, max_atoms=8)))
#   "
#
#: ``torch.randperm(16, generator=make_generator(42)).tolist()``.
GOLDEN_PERM = [6, 3, 0, 7, 10, 15, 2, 12, 14, 9, 8, 5, 4, 13, 11, 1]
#: Plan for ``max_atoms=16`` over the 16-sample varied cache.
GOLDEN_PLAN_ATOMS16 = [[6, 3], [0, 7], [10, 15], [2, 12], [14, 9, 8], [5, 4], [13, 11], [1]]
#: Plan for ``max_atoms=16, max_edges=20`` — the edge cap splits ``[14, 9, 8]``.
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
#: Verbatim oversize warning text (src/molix/data/sampler.py:125-132).
GOLDEN_OVERSIZE_WARNING = (
    "TokenBudgetBatchSampler: at least one sample exceeds the per-batch "
    "budget on its own; such samples are kept as singleton batches (never "
    "dropped). Raise the budget to avoid singletons."
)
#: Instance state that must stay picklable and packer-free after the refactor.
GOLDEN_SAMPLER_FIELDS = ["_batches", "_dataset", "_max_atoms", "_max_edges", "_seed"]


@pytest.fixture
def golden_ds(tmp_path) -> MmapDataset:
    """16-sample varied-count dataset backing the characterization goldens.

    Counts are a pure function of the sample index (``3 + i % 8`` atoms,
    ``atoms + 2`` edges), so rebuilding the cache per test in ``tmp_path``
    reproduces the captured plans exactly.
    """
    return MmapDataset(_write_varied_cache(tmp_path, "golden", n=GOLDEN_N))


class TestSamplerGoldenPlan:
    """Characterization: the pre-refactor plans, pinned as literals.

    Green on the pre-refactor tree by construction — this class exists to
    prove the ``TokenBudgetPacker`` extraction is behavior-preserving.
    """

    def test_golden_counts_are_deterministic(self, golden_ds):
        """The fixture cache reproduces the hand-checkable count tables."""
        assert golden_ds.atom_counts.tolist() == ATOM_COUNTS[:GOLDEN_N]
        assert golden_ds.edge_counts.tolist() == EDGE_COUNTS[:GOLDEN_N]

    def test_permutation_matches_golden(self):
        """The visit order is ``randperm(n, generator=make_generator(seed))``."""
        perm = torch.randperm(GOLDEN_N, generator=make_generator(GOLDEN_SEED))
        assert perm.tolist() == GOLDEN_PERM

    def test_atom_budget_plan_matches_golden(self, golden_ds):
        """max_atoms=16 reproduces the captured batch plan."""
        sampler = TokenBudgetBatchSampler(golden_ds, max_atoms=16)
        assert list(sampler) == GOLDEN_PLAN_ATOMS16

    def test_dual_budget_plan_matches_golden(self, golden_ds):
        """max_atoms=16 + max_edges=20 reproduces the captured batch plan."""
        sampler = TokenBudgetBatchSampler(golden_ds, max_atoms=16, max_edges=20)
        assert list(sampler) == GOLDEN_PLAN_ATOMS16_EDGES20

    def test_oversize_plan_matches_golden(self, golden_ds):
        """max_atoms=8 reproduces the captured singleton-heavy batch plan."""
        with pytest.warns(UserWarning):
            sampler = TokenBudgetBatchSampler(golden_ds, max_atoms=8)
        assert list(sampler) == GOLDEN_PLAN_ATOMS8


class TestSamplerSurfacePreservation:
    """Observable sampler surface pinned across the packer extraction.

    Also green pre-refactor: the packer must stay transient (never stored
    on the instance) and the warning text/count must not drift.
    """

    def test_instance_fields_match_golden_list(self, golden_ds):
        """``vars(sampler)`` keeps exactly its five picklable fields."""
        sampler = TokenBudgetBatchSampler(golden_ds, max_atoms=16)
        assert sorted(vars(sampler)) == GOLDEN_SAMPLER_FIELDS

    def test_oversize_warning_text_verbatim_once(self, golden_ds):
        """Exactly one warning per instance, with the unchanged message."""
        with pytest.warns(UserWarning) as record:
            TokenBudgetBatchSampler(golden_ds, max_atoms=8)
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
        edge_counts = torch.tensor([4, 4, 4, 4])
        dual = TokenBudgetPacker([(100, atom_counts), (10, edge_counts)])
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

        packer = TokenBudgetPacker([(16, golden_ds.atom_counts)])
        assert packer.pack(GOLDEN_PERM) == GOLDEN_PLAN_ATOMS16

    def test_golden_dual_budget_plan_via_packer(self, golden_ds):
        """Packing the golden permutation reproduces the dual-budget plan."""

        packer = TokenBudgetPacker([(16, golden_ds.atom_counts), (20, golden_ds.edge_counts)])
        assert packer.pack(GOLDEN_PERM) == GOLDEN_PLAN_ATOMS16_EDGES20

    def test_golden_oversize_plan_via_packer(self, golden_ds):
        """Packing the golden permutation reproduces the max_atoms=8 plan."""

        packer = TokenBudgetPacker([(8, golden_ds.atom_counts)])
        assert packer.pack(GOLDEN_PERM) == GOLDEN_PLAN_ATOMS8
