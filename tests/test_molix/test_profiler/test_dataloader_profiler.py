"""Tests for :class:`molix.profiler.DataLoaderProfiler`, incl. worker picklability."""

from __future__ import annotations

import pickle

from molix.profiler import DataLoaderProfiler, MockSource
from molix.profiler.dataloader import _ProfilerCollate


def test_run_with_zero_workers():
    """Single-process path produces a result with positive throughput."""
    prof = DataLoaderProfiler(batch_size=8, num_workers=0)
    result = prof.run(MockSource(n_samples=200, n_atoms=(5, 15)), n_batches=10, n_warmup=2)
    assert result.num_workers == 0
    assert result.throughput_graphs_per_sec > 0
    assert result.batch_graph_stats.mean > 0


def test_collate_is_picklable_for_spawn_workers():
    """The collate callable must pickle (closures broke num_workers>0)."""
    from molix.data.collate import DEFAULT_TARGET_SCHEMA

    fn = _ProfilerCollate(DEFAULT_TARGET_SCHEMA, ())
    restored = pickle.loads(pickle.dumps(fn))
    assert isinstance(restored, _ProfilerCollate)


def test_run_with_spawn_workers():
    """num_workers>0 (spawn) no longer raises a PicklingError on collate_fn."""
    prof = DataLoaderProfiler(batch_size=8, num_workers=2)
    result = prof.run(MockSource(n_samples=200, n_atoms=(5, 15)), n_batches=10, n_warmup=2)
    assert result.num_workers == 2
    assert result.throughput_graphs_per_sec > 0
