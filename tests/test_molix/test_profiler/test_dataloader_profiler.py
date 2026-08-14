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
    from molix.batch.collate import DEFAULT_TARGET_SCHEMA

    fn = _ProfilerCollate(DEFAULT_TARGET_SCHEMA, ())
    restored = pickle.loads(pickle.dumps(fn))
    assert isinstance(restored, _ProfilerCollate)


def test_run_with_spawn_workers():
    """num_workers>0 (spawn) no longer raises a PicklingError on collate_fn."""
    prof = DataLoaderProfiler(batch_size=8, num_workers=2)
    result = prof.run(MockSource(n_samples=200, n_atoms=(5, 15)), n_batches=10, n_warmup=2)
    assert result.num_workers == 2
    assert result.throughput_graphs_per_sec > 0


def test_run_applies_batch_task():
    """``batch_tasks=(...)`` runs on every profiled batch (post-``batch_nodes``).

    The profiler no longer takes a ``PipelineSpec``; post-collate transforms
    arrive as ``BatchTask`` instances. Locals (task class + imports) keep this
    test self-contained; with ``num_workers=0`` the very instance passed in is
    the one ``_ProfilerCollate`` calls, so its counter is observable here.
    """
    import torch
    from tensordict import TensorDict

    from molix.batch.task import BatchTask

    class _MarkerTask(BatchTask):
        """Stamps a ``graphs.marker`` leaf and records what it stamped."""

        def __init__(self) -> None:
            self.calls = 0
            self.last_marker: torch.Tensor | None = None

        def execute(self, batch: TensorDict) -> TensorDict:
            """Write ``marker`` ``(B,)`` under graphs, then read it back."""
            self.calls += 1
            n_graphs = int(batch["graphs"].batch_size[0])
            batch["graphs", "marker"] = torch.full((n_graphs,), 7, dtype=torch.long)
            self.last_marker = batch["graphs", "marker"]
            return batch

    task = _MarkerTask()
    prof = DataLoaderProfiler(batch_size=8, num_workers=0, batch_tasks=[task])
    result = prof.run(MockSource(n_samples=40, n_atoms=(5, 15)), n_batches=3, n_warmup=1)

    assert result.n_batches == 3
    assert result.throughput_graphs_per_sec > 0
    # One collate per consumed batch: n_warmup + n_batches, none dropped.
    assert task.calls == 4
    assert task.last_marker is not None
    assert task.last_marker.tolist() == [7] * 8
