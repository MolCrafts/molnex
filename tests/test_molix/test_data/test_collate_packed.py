"""Tests for the packed-aware collate fast path.

Covers spec ``dynamic-batching-packed-collate-02-collate``:

* GROUP 1 — equivalence oracle: ``collate_packed(view, indices, schema)``
  must equal ``collate_molecules([dataset[i] for i in indices], schema)``
  leaf-for-leaf (keys, ``torch.equal`` values, dtypes, ``batch_size`` at
  every level). Acceptance ac-001..004.
* GROUP 2 — eager actionable errors on empty indices / missing Z|pos.
  Acceptance ac-010.
* GROUP 3 — layering guard: ``cache.py`` imports no ``tensordict`` /
  ``TargetSchema``. Acceptance ac-005.
* GROUP 4 — ``DataModule`` fast-path routing, spawn workers, batch_nodes +
  ftype post-steps, pickle without captured payload tensors. Acceptance
  ac-006..009.

Imports of the production surface (``molix.data.collate.collate_packed``,
``packed_view()`` / ``PackedView``, ``_PackedCollateFn`` / ``_IndexDataset``)
are kept inside test bodies, a holdover from the RED phase that also keeps
each test's dependency explicit at its use site.

Fixture geometry (hand-checkable, mirrors ``test_sampler.py``): sample
``i`` has ``3 + (i % 8)`` atoms (cycling 3..10) and ``n_atoms + 2`` edges
(cycling 5..12). Edge counts never equal atom counts, and ``U0`` is shape
``(1,)`` (never 1 atom/edge), so the packed schema routes ``U0`` to graph,
``forces`` to atom, and ``n_heavy`` to scalar with no leading-dim
ambiguity.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from molix.data.cache import PackedCache
from molix.data.collate import TargetSchema, collate_molecules
from molix.data.dataset import CachedDataset, MmapDataset, SubsetDataset

# ---------------------------------------------------------------------------
# Fixture sample builders
# ---------------------------------------------------------------------------

#: Schema covering both target routings plus a python-scalar target.
SCHEMA = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset({"forces"}))


def _atom_count(i: int) -> int:
    """Per-sample atom count: 3, 4, ..., 10 cycling (never 1)."""
    return 3 + (i % 8)


def _make_varied_samples(n: int, *, with_scalar: bool = True) -> list[dict]:
    """Build ``n`` edged samples with atom/graph/(scalar) targets.

    Args:
        n: Number of samples.
        with_scalar: Include a python-scalar ``n_heavy`` target.

    Returns:
        Flat sample dicts with per-atom (``Z``, ``pos``, ``targets.forces``),
        per-edge (``edge_index`` ``(E, 2)``, ``edge_diff`` ``(E, 3)``,
        ``edge_dist`` ``(E,)``) and graph-level (``targets.U0`` shape
        ``(1,)``, unique identity ``float(i)``) keys.
    """
    torch.manual_seed(0)
    samples: list[dict] = []
    for i in range(n):
        na = _atom_count(i)
        ne = na + 2
        targets: dict = {
            "U0": torch.tensor([float(i)]),
            "forces": torch.randn(na, 3),
        }
        if with_scalar:
            targets["n_heavy"] = int(i % 4)
        samples.append(
            {
                "Z": torch.ones(na, dtype=torch.long),
                "pos": torch.randn(na, 3),
                "edge_index": torch.randint(0, na, (ne, 2), dtype=torch.long),
                "edge_diff": torch.randn(ne, 3),
                "edge_dist": torch.rand(ne) + 0.5,
                "targets": targets,
            }
        )
    return samples


def _make_mixed_edge_samples(n: int) -> list[dict]:
    """Build ``n`` samples where every other sample has zero edges.

    Per-sample zero edges are a supported packed-cache state (``edge_ptr``
    stays flat across the empty sample); the resulting batch must walk the
    normal concatenation branch, matching the oracle.
    """
    torch.manual_seed(1)
    samples: list[dict] = []
    for i in range(n):
        na = _atom_count(i)
        ne = 0 if i % 2 == 1 else na + 2
        samples.append(
            {
                "Z": torch.ones(na, dtype=torch.long),
                "pos": torch.randn(na, 3),
                "edge_index": torch.randint(0, na, (ne, 2), dtype=torch.long),
                "edge_diff": torch.randn(ne, 3),
                "edge_dist": torch.rand(ne),
                "targets": {"U0": torch.tensor([float(i)])},
            }
        )
    return samples


def _make_edgeless_samples(n: int) -> list[dict]:
    """Build ``n`` samples with per-atom keys only — packed cache has no edge keys.

    Exercises the all-edge-less fallback: ``collate_molecules`` emits the
    canonical empty-edge TensorDict (``edge_index`` ``zeros(0, 2)`` long,
    ``edge_diff`` ``zeros(0, 3)``, ``edge_dist`` ``zeros(0)``,
    ``batch_size=[0]``).
    """
    torch.manual_seed(2)
    samples: list[dict] = []
    for i in range(n):
        na = _atom_count(i)
        samples.append(
            {
                "Z": torch.ones(na, dtype=torch.long),
                "pos": torch.randn(na, 3),
                "targets": {"U0": torch.tensor([float(i)])},
            }
        )
    return samples


def _save(tmp_path: Path, tag: str, samples: list[dict]) -> Path:
    """Save *samples* to a packed cache under *tmp_path* and return the sink."""
    sink = tmp_path / f"{tag}.pt"
    PackedCache(sink).save(samples)
    return sink


# ---------------------------------------------------------------------------
# Equivalence oracle + leaf-equality asserter
# ---------------------------------------------------------------------------


def _oracle(dataset, indices: list[int], schema: TargetSchema) -> TensorDict:
    """Ground-truth batch: collate per-sample dicts via the slow path."""
    return collate_molecules([dataset[i] for i in indices], schema)


def _assert_td_equal(got: TensorDict, want: TensorDict) -> None:
    """Assert two (nested) TensorDicts are leaf-for-leaf identical.

    Checks identical key sets at every nesting level, ``torch.equal`` and
    identical dtype per leaf tensor, and identical ``batch_size`` on each
    nested TensorDict and the top level.

    Args:
        got: Output under test (fast path).
        want: Oracle output (slow path).
    """
    assert list(got.batch_size) == list(want.batch_size), (
        f"top-level batch_size: {list(got.batch_size)} != {list(want.batch_size)}"
    )
    assert set(got.keys()) == set(want.keys()), (
        f"top-level keys: {set(got.keys())} != {set(want.keys())}"
    )

    for level in ("atoms", "edges", "graphs"):
        sub_got = got[level]
        sub_want = want[level]
        assert isinstance(sub_got, TensorDict), f"{level!r} is not a TensorDict"
        assert list(sub_got.batch_size) == list(sub_want.batch_size), (
            f"{level} batch_size: {list(sub_got.batch_size)} != {list(sub_want.batch_size)}"
        )
        assert set(sub_got.keys()) == set(sub_want.keys()), (
            f"{level} keys: {set(sub_got.keys())} != {set(sub_want.keys())}"
        )
        for key in sub_want.keys():
            g = sub_got[key]
            w = sub_want[key]
            assert g.dtype == w.dtype, f"{level}.{key} dtype: {g.dtype} != {w.dtype}"
            assert torch.equal(g, w), f"{level}.{key} values differ"


def _packed_collate(dataset, indices: list[int], schema: TargetSchema) -> TensorDict:
    """Run the fast path: ``collate_packed(dataset.packed_view(), indices, schema)``.

    Deferred import keeps module collection clean while the production
    surface is still absent (RED).
    """
    from molix.data.collate import collate_packed

    return collate_packed(dataset.packed_view(), indices, schema)


# ===========================================================================
# GROUP 1 — equivalence (ac-001..004)
# ===========================================================================


class TestEquivalenceWithEdges:
    """Multi-sample edged batches match the oracle leaf-for-leaf (ac-001)."""

    def test_multi_sample_with_edges(self, tmp_path):
        """A >=2-sample batch with varied atom/edge counts equals the oracle."""
        ds = MmapDataset(_save(tmp_path, "edged", _make_varied_samples(8)))
        indices = [0, 3, 5]
        _assert_td_equal(_packed_collate(ds, indices, SCHEMA), _oracle(ds, indices, SCHEMA))

    def test_cached_dataset_backing(self, tmp_path):
        """CachedDataset (mmap=False) packed view collates identically."""
        ds = CachedDataset(_save(tmp_path, "edged_cached", _make_varied_samples(8)))
        indices = [1, 2, 6, 7]
        _assert_td_equal(_packed_collate(ds, indices, SCHEMA), _oracle(ds, indices, SCHEMA))

    def test_singleton_batch(self, tmp_path):
        """A length-1 index list collates equivalently (ac-004)."""
        ds = MmapDataset(_save(tmp_path, "single", _make_varied_samples(8)))
        indices = [4]
        _assert_td_equal(_packed_collate(ds, indices, SCHEMA), _oracle(ds, indices, SCHEMA))


class TestEquivalenceEdgeless:
    """Edge-less handling matches the oracle, incl. empty-edge fallback (ac-002)."""

    def test_mixed_zero_and_edged_samples(self, tmp_path):
        """A batch mixing zero-edge and edged samples equals the oracle."""
        ds = MmapDataset(_save(tmp_path, "mixed", _make_mixed_edge_samples(8)))
        # indices 0,2 have edges; 1,3 have zero edges.
        indices = [0, 1, 2, 3]
        schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset())
        _assert_td_equal(_packed_collate(ds, indices, schema), _oracle(ds, indices, schema))

    def test_all_edgeless_fallback_matches_oracle(self, tmp_path):
        """All-edge-less schema → fast-path edges equals the oracle fallback."""
        ds = MmapDataset(_save(tmp_path, "edgeless", _make_edgeless_samples(8)))
        indices = [0, 1, 2]
        schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset())
        _assert_td_equal(_packed_collate(ds, indices, schema), _oracle(ds, indices, schema))

    def test_all_edgeless_fallback_exact_fields(self, tmp_path):
        """Fast-path empty-edge TensorDict is byte-identical to collate.py:149-156."""
        ds = MmapDataset(_save(tmp_path, "edgeless2", _make_edgeless_samples(8)))
        schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset())
        batch = _packed_collate(ds, [0, 1, 2], schema)
        edges = batch["edges"]
        assert list(edges.batch_size) == [0]
        assert torch.equal(edges["edge_index"], torch.zeros(0, 2, dtype=torch.long))
        assert edges["edge_index"].dtype == torch.long
        assert torch.equal(edges["edge_diff"], torch.zeros(0, 3))
        assert torch.equal(edges["edge_dist"], torch.zeros(0))


class TestTargetSchemaRouting:
    """atom_level / graph_level / scalar targets route exactly as the oracle (ac-003)."""

    def test_atom_and_graph_targets(self, tmp_path):
        """forces under atoms, U0 reshape(-1) under graphs, equal to oracle."""
        ds = MmapDataset(_save(tmp_path, "routing", _make_varied_samples(8, with_scalar=False)))
        indices = [0, 2, 4]
        schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset({"forces"}))
        got = _packed_collate(ds, indices, schema)
        want = _oracle(ds, indices, schema)
        _assert_td_equal(got, want)
        # Spell out the routing the asserter folds in, so a regression is legible.
        assert "forces" in got["atoms"].keys()
        assert "U0" in got["graphs"].keys()
        assert got["graphs", "U0"].shape == (len(indices),)

    def test_scalar_target_routed(self, tmp_path):
        """A python-scalar target (n_heavy) routes per the oracle conversion path."""
        ds = MmapDataset(_save(tmp_path, "scalar", _make_varied_samples(8, with_scalar=True)))
        indices = [1, 3, 6]
        schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset({"forces"}))
        _assert_td_equal(_packed_collate(ds, indices, schema), _oracle(ds, indices, schema))


class TestSubsetViews:
    """SubsetDataset packed views remap local→packed indices (ac-004)."""

    def test_subset_explicit_indices(self, tmp_path):
        """Explicit shuffled SubsetDataset view equals the oracle over global samples."""
        ds = MmapDataset(_save(tmp_path, "subset", _make_varied_samples(16)))
        subset = SubsetDataset(ds, [9, 2, 14, 5])
        # Local indices into the subset → packed indices [9, 2, 14, 5].
        local = [0, 1, 2, 3]
        got = _packed_collate(subset, local, SCHEMA)
        want = _oracle(subset, local, SCHEMA)
        _assert_td_equal(got, want)

    def test_split_shuffled_view(self, tmp_path):
        """A split()-produced shuffled view collates equivalently."""
        ds = MmapDataset(_save(tmp_path, "split", _make_varied_samples(16)))
        train, _val = ds.split(sizes=(12, 4), seed=7)
        local = [0, 5, 11, 3]
        _assert_td_equal(_packed_collate(train, local, SCHEMA), _oracle(train, local, SCHEMA))


# ===========================================================================
# GROUP 2 — eager errors (ac-010)
# ===========================================================================


class TestEagerErrors:
    """collate_packed raises actionable ValueError eagerly (ac-010)."""

    def test_empty_indices_raises(self, tmp_path):
        """Empty index list → ValueError, matching the oracle's 'empty' wording."""
        from molix.data.collate import collate_packed

        ds = MmapDataset(_save(tmp_path, "empty", _make_varied_samples(8)))
        with pytest.raises(ValueError, match="(?i)empty"):
            collate_packed(ds.packed_view(), [], SCHEMA)

    def test_missing_pos_raises_naming_key(self, tmp_path):
        """A view whose schema lacks pos → ValueError naming the missing key."""
        from molix.data.collate import collate_packed

        # Cache with Z only (no pos) — PackedCache accepts this layout.
        samples = [{"Z": torch.ones(_atom_count(i), dtype=torch.long)} for i in range(4)]
        ds = MmapDataset(_save(tmp_path, "nopos", samples))
        with pytest.raises(ValueError, match="pos"):
            collate_packed(ds.packed_view(), [0, 1], SCHEMA)

    def test_missing_z_raises_naming_key(self, tmp_path):
        """A view whose schema lacks Z → ValueError naming the missing key."""
        from molix.data.collate import collate_packed

        # Cache with pos only (no Z). Uniform atom count so pos packs (without
        # Z there is no per-atom ref key, so a varying-shape pos can't be
        # classified); collate_packed must still reject the missing Z.
        torch.manual_seed(3)
        samples = [{"pos": torch.randn(5, 3)} for _ in range(4)]
        ds = MmapDataset(_save(tmp_path, "noz", samples))
        with pytest.raises(ValueError, match="Z"):
            collate_packed(ds.packed_view(), [0, 1], SCHEMA)


# ===========================================================================
# GROUP 3 — layering guard (ac-005)
# ===========================================================================


class TestCacheLayering:
    """cache.py stays IO-only — no tensordict / TargetSchema import (ac-005)."""

    def test_cache_has_no_collate_imports(self):
        """No import line in cache.py references tensordict or TargetSchema."""
        import molix.data.cache as cache_mod

        source = Path(cache_mod.__file__).read_text()
        import_lines = [
            line for line in source.splitlines() if line.lstrip().startswith(("import ", "from "))
        ]
        joined = "\n".join(import_lines)
        assert "tensordict" not in joined
        assert "TargetSchema" not in joined


# ===========================================================================
# GROUP 4 — DataModule integration (ac-006..009)
# ===========================================================================


class _MarkerNode:
    """Minimal batch node that stamps a marker leaf onto the graphs namespace."""

    def apply(self, batch: TensorDict) -> TensorDict:
        """Write a constant ``marker`` leaf of shape ``(B,)`` under graphs."""
        b = batch["graphs"].batch_size[0]
        batch["graphs", "marker"] = torch.ones(b, dtype=torch.long)
        return batch


class _NonPackedDataset(torch.utils.data.Dataset):
    """Minimal map-style dataset with no packed-cache backing (no packed_view)."""

    def __init__(self, samples: list[dict]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


def _make_dm(tmp_path: Path, tag: str, *, samples=None, **kwargs):
    """Build a DataModule over a packed varied-count cache (num_workers=0).

    Args:
        tmp_path: Pytest temp directory for the packed caches.
        tag: Unique filename tag.
        samples: Override sample list (defaults to varied edged samples).
        **kwargs: Extra DataModule kwargs.

    Returns:
        A non-DDP DataModule whose train/val splits back the fast path.
    """
    from molix.data.datamodule import DataModule

    if samples is None:
        samples = _make_varied_samples(12)
    train_sink = _save(tmp_path, f"{tag}_train", samples[:8])
    val_sink = _save(tmp_path, f"{tag}_val", samples[8:])
    kwargs.setdefault("target_schema", SCHEMA)
    kwargs.setdefault("batch_size", 4)
    kwargs.setdefault("num_workers", 0)
    kwargs.setdefault("pin_memory", False)
    return DataModule(MmapDataset(train_sink), MmapDataset(val_sink), **kwargs)


def _batch_order(loader) -> list[float]:
    """Per-sample ``U0`` identities across an entire loader, in yield order."""
    return [float(u) for batch in loader for u in batch["graphs", "U0"]]


def _assert_no_unpack(monkeypatch, body) -> None:
    """Run *body* with PackedCache.unpack_sample counted; assert zero calls.

    This is the code-level proof the fast path is engaged: the slow path
    reconstructs each sample via ``unpack_sample``, so a fast-path epoch
    must register zero calls. Used to keep the integration tests honestly
    RED until the fast path is wired (without it they pass vacuously over
    the existing slow path).

    Args:
        monkeypatch: pytest monkeypatch fixture.
        body: Zero-arg callable that drives one or more loader epochs.
    """
    calls = {"n": 0}
    orig = PackedCache.unpack_sample

    def _counting(payload, idx):
        calls["n"] += 1
        return orig(payload, idx)

    monkeypatch.setattr(PackedCache, "unpack_sample", staticmethod(_counting))
    body()
    assert calls["n"] == 0, f"fast path expected, but unpack_sample ran {calls['n']}x"


class TestDataModuleFastPathRouting:
    """Packed datasets route through the fast path; non-packed fall back (ac-006)."""

    def test_packed_uses_packed_collate_fn(self, tmp_path):
        """With a packed dataset, the train collate_fn is the packed fast-path type."""
        from molix.data.datamodule import _PackedCollateFn

        dm = _make_dm(tmp_path, "route")
        loader = dm.train_dataloader()
        assert isinstance(loader.collate_fn, _PackedCollateFn)

    def test_fast_path_does_not_unpack(self, tmp_path, monkeypatch):
        """PackedCache.unpack_sample is never called during a fast-path epoch."""
        calls = {"n": 0}
        orig = PackedCache.unpack_sample

        def _counting(payload, idx):
            calls["n"] += 1
            return orig(payload, idx)

        monkeypatch.setattr(PackedCache, "unpack_sample", staticmethod(_counting))

        dm = _make_dm(tmp_path, "nounpack")
        for _ in dm.train_dataloader():
            pass
        assert calls["n"] == 0

    def test_fast_path_batches_equal_slow_path(self, tmp_path, monkeypatch):
        """Fast-path batches equal collate_molecules for the same index order."""
        dm = _make_dm(tmp_path, "equal", batch_size=4)
        # Deterministic order: iterate the val loader (never shuffled).
        want = _oracle(dm.val_dataset, list(range(len(dm.val_dataset))), SCHEMA)
        captured: dict = {}

        def _run():
            captured["batch"] = next(iter(dm.val_dataloader()))

        # Engaging the fast path is part of the contract — zero unpack calls.
        _assert_no_unpack(monkeypatch, _run)
        batch = captured["batch"]
        # ftype cast is the only intended transform on the fast path with no
        # batch nodes; default ftype is float32 so leaves stay identical.
        _assert_td_equal(batch.to(torch.float32), want.to(torch.float32))

    def test_non_packed_dataset_falls_back(self, tmp_path):
        """A non-packed dataset uses _CollateFn + collate_molecules."""
        from molix.data.datamodule import DataModule, _CollateFn

        samples = _make_varied_samples(6)
        ds = _NonPackedDataset(samples)
        dm = DataModule(
            ds,
            ds,
            target_schema=SCHEMA,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        loader = dm.train_dataloader()
        assert isinstance(loader.collate_fn, _CollateFn)
        batch = next(iter(loader))
        assert isinstance(batch, TensorDict)
        assert "U0" in batch["graphs"].keys()


class TestDataModulePostSteps:
    """batch_nodes and batch_to(ftype) apply on the fast path (ac-008)."""

    def test_batch_node_applied_on_fast_path(self, tmp_path, monkeypatch):
        """A registered marker batch node stamps its leaf on fast-path batches."""
        dm = _make_dm(tmp_path, "marker", batch_nodes=[_MarkerNode()])
        captured: dict = {}

        def _run():
            captured["batch"] = next(iter(dm.val_dataloader()))

        _assert_no_unpack(monkeypatch, _run)
        batch = captured["batch"]
        assert "marker" in batch["graphs"].keys()
        assert torch.all(batch["graphs", "marker"] == 1)

    def test_ftype_cast_applied_on_fast_path(self, tmp_path, monkeypatch):
        """Float leaves cast to the captured non-default ftype on the fast path."""
        from molix.config import config

        prev = config["ftype"]
        try:
            config["ftype"] = torch.float64
            dm = _make_dm(tmp_path, "ftype")
            captured: dict = {}

            def _run():
                captured["batch"] = next(iter(dm.val_dataloader()))

            _assert_no_unpack(monkeypatch, _run)
            batch = captured["batch"]
            assert batch["atoms", "pos"].dtype == torch.float64
            assert batch["graphs", "U0"].dtype == torch.float64
            # Integer leaves are untouched by batch_to.
            assert batch["atoms", "Z"].dtype == torch.long
        finally:
            config["ftype"] = prev


class TestDataModulePickle:
    """The packed collate callable pickles without capturing payload tensors (ac-009)."""

    def test_packed_collate_fn_pickles_and_collates(self, tmp_path):
        """pickle round-trip of _PackedCollateFn still produces correct batches."""
        from molix.data.datamodule import _PackedCollateFn

        dm = _make_dm(tmp_path, "pickle")
        loader = dm.train_dataloader()
        fn = loader.collate_fn
        assert isinstance(fn, _PackedCollateFn)
        restored = pickle.loads(pickle.dumps(fn))
        # _IndexDataset hands the collate fn a list of int indices.
        indices = [0, 1, 2]
        out1 = fn(indices)
        out2 = restored(indices)
        _assert_td_equal(out2, out1)

    def test_pickled_state_excludes_lazy_view(self, tmp_path):
        """__getstate__ of _PackedCollateFn omits the lazily built PackedView field."""
        from molix.data.datamodule import _PackedCollateFn

        dm = _make_dm(tmp_path, "state")
        fn = dm.train_dataloader().collate_fn
        assert isinstance(fn, _PackedCollateFn)
        # Force lazy view construction so the exclusion is meaningful.
        fn([0, 1])
        state = fn.__getstate__()
        # No state entry may hold a PackedView (the lazy field is dropped).
        from molix.data.dataset import PackedView

        assert not any(isinstance(v, PackedView) for v in state.values())


class TestDataModuleSpawnWorkers:
    """Spawn num_workers=2 end-to-end matches num_workers=0 fast path (ac-007)."""

    def test_spawn_workers_match_sync_fast_path(self, tmp_path):
        """One epoch with spawn workers yields the same U0 multiset as sync."""
        samples = _make_varied_samples(12)
        train_sink = _save(tmp_path, "spawn_train", samples[:8])
        val_sink = _save(tmp_path, "spawn_val", samples[8:])

        from molix.data.datamodule import DataModule, _PackedCollateFn

        # Val loader never shuffles → deterministic order across worker counts.
        dm_sync = DataModule(
            MmapDataset(train_sink),
            MmapDataset(val_sink),
            target_schema=SCHEMA,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        # Both worker counts must take the packed fast path.
        assert isinstance(dm_sync.val_dataloader().collate_fn, _PackedCollateFn)
        sync_order = _batch_order(dm_sync.val_dataloader())

        dm_spawn = DataModule(
            MmapDataset(train_sink),
            MmapDataset(val_sink),
            target_schema=SCHEMA,
            batch_size=2,
            num_workers=2,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=2,
        )
        seen = 0
        for batch in dm_spawn.val_dataloader():
            assert isinstance(batch, TensorDict)
            assert batch["atoms", "Z"].ndim == 1
            assert batch["edges", "edge_index"].shape[1] == 2
            seen += 1
        assert seen >= 1
        spawn_order = _batch_order(dm_spawn.val_dataloader())
        assert sorted(spawn_order) == sorted(sync_order)
