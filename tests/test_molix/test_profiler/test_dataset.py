"""Tests for :class:`molix.profiler.dataset.DatasetProfiler`.

Every fixture is built from **literal** sample dicts (no RNG), so the sizes
asserted below are analytic constants of the fixture, not measured values.
The only timing assertions are ``> 0`` liveness checks — no wall-clock
magnitude is pinned.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from molix.data.cache import PackedCache, _flatten
from molix.data.dataset import CachedDataset
from molix.profiler.dataset import (
    DatasetProfiler,
    DatasetResult,
    FieldSpec,
    TargetStat,
    _flatten_leaves,
)

# ---------------------------------------------------------------------------
# Literal fixtures — no RNG, so every expectation below is analytic
# ---------------------------------------------------------------------------

_N_RECORDS = 100
_ATOM_MEAN = 3.5  # atom counts cycle 2,3,4,5 → 350 atoms over 100 records
_EDGE_MEAN = 7.0  # bidirectional ring → 2N edges per record → 700 total
_MAX_ATOMS = 5
_MAX_EDGES = 10
_AVG_NUM_NEIGHBORS = 2.0  # 700 edges / 350 atoms, exactly


def _record(i: int, n: int | None = None) -> dict:
    """One literal sample dict: a ring molecule of ``n`` (default ``2 + i % 4``) atoms.

    Keys follow the raw-sample tier of the two-tier data contract:
    ``Z`` ``(N,)``, ``pos`` ``(N, 3)``, ``edge_index`` ``(2N, 2)``,
    ``edge_dist`` ``(2N,)``, and the scalar target ``targets.U0`` ``(1,)``.

    Args:
        i: Record index; drives the atom count and the target value.
        n: Fixed atom count, overriding the ``2 + i % 4`` cycle.

    Returns:
        A flat sample dict suitable for :meth:`PackedCache.save`.
    """
    n_atoms = 2 + i % 4 if n is None else n
    src = torch.arange(n_atoms, dtype=torch.long)
    dst = (src + 1) % n_atoms
    edge_index = torch.cat([torch.stack([src, dst], dim=1), torch.stack([dst, src], dim=1)], dim=0)
    return {
        "Z": torch.full((n_atoms,), 6, dtype=torch.long),
        "pos": torch.arange(3 * n_atoms, dtype=torch.float32).reshape(n_atoms, 3),
        "edge_index": edge_index,
        "edge_dist": torch.full((2 * n_atoms,), 1.5, dtype=torch.float32),
        "targets": {"U0": torch.tensor([float(i)], dtype=torch.float32)},
    }


def _write_cache(tmp_path: Path, samples: list[dict]) -> Path:
    """Pack *samples* into a cache file under *tmp_path* and return the sink path."""
    sink = tmp_path / "cache.pt"
    PackedCache(sink).save(samples)
    return sink


class _CountingCachedDataset(CachedDataset):
    """:class:`CachedDataset` that counts ``__getitem__`` calls.

    Subclassing keeps the packed-pointer fast path (``atom_counts`` /
    ``max_atoms`` / ``packed_view``) intact, so the counter observes only
    the profiler's sampled slow path.
    """

    def __init__(self, sink: Path) -> None:
        super().__init__(sink)
        self.n_getitem = 0

    def __getitem__(self, idx: int) -> dict:
        self.n_getitem += 1
        return super().__getitem__(idx)


@pytest.fixture
def ring_dataset(tmp_path: Path) -> CachedDataset:
    """Cache-backed dataset over the 100 literal ring records."""
    return CachedDataset(_write_cache(tmp_path, [_record(i) for i in range(_N_RECORDS)]))


# ---------------------------------------------------------------------------
# DatasetProfiler
# ---------------------------------------------------------------------------


class TestDatasetProfiler:
    """Config in ``__init__``, data in ``run()``; exact cache stats + sampled access."""

    def test_run_uses_exact_cache_counts(self, ring_dataset):
        """Size stats come from the packed pointers — all 100 records, not the 5 sampled."""
        result = DatasetProfiler(n_samples=5).run(ring_dataset)

        assert isinstance(result, DatasetResult)
        assert result.counts_exact is True
        assert result.n_total == _N_RECORDS
        assert result.max_atoms == ring_dataset.max_atoms == _MAX_ATOMS
        assert result.max_edges == ring_dataset.max_edges == _MAX_EDGES
        assert result.avg_num_neighbors == ring_dataset.avg_num_neighbors
        assert result.avg_num_neighbors == pytest.approx(_AVG_NUM_NEIGHBORS, rel=1e-12)
        assert result.atom_stats.mean == float(ring_dataset.atom_counts.double().mean())
        assert result.atom_stats.mean == pytest.approx(_ATOM_MEAN, rel=1e-12)
        assert result.edge_stats is not None
        assert result.edge_stats.mean == pytest.approx(_EDGE_MEAN, rel=1e-12)

    def test_run_samples_only_n_samples_for_access(self, tmp_path):
        """The access loop is bounded by ``n_samples`` (+ warmup), never ``n_total``."""
        ds = _CountingCachedDataset(_write_cache(tmp_path, [_record(i) for i in range(_N_RECORDS)]))

        result = DatasetProfiler(n_samples=5, n_warmup=3).run(ds)

        assert result.n_sampled == 5
        assert result.cold_access_ms > 0.0
        assert result.access_ms.mean_ms > 0.0
        # at most one cold access + 3 discarded warmups + 5 measured accesses
        assert 5 <= ds.n_getitem <= 1 + 3 + 5

    def test_run_fields_come_from_packed_schema(self, ring_dataset):
        """Field layout is read off ``payload["schema"]`` verbatim, hence exact."""
        result = DatasetProfiler(n_samples=5).run(ring_dataset)
        schema = ring_dataset.packed_view().payload["schema"]

        assert result.fields_exact is True
        assert all(isinstance(f, FieldSpec) for f in result.fields)
        assert {(f.key, f.axis, f.dtype, f.extra_shape) for f in result.fields} == {
            (key, *spec) for key, spec in schema.items()
        }

    def test_run_on_plain_sequence_falls_back(self):
        """A ``list[dict]`` has no packed pointers: degrade audibly, never raise."""
        records = [_record(i, n=4) for i in range(10)]

        result = DatasetProfiler(n_samples=5).run(records)

        assert isinstance(result, DatasetResult)
        assert result.counts_exact is False
        assert result.fields_exact is False
        assert result.n_total == 10
        assert result.n_sampled == 5
        assert result.atom_stats.mean == pytest.approx(4.0, rel=1e-12)
        assert result.fields  # inferred from the sampled records

    def test_run_without_edge_ptr_warns_not_raises(self, tmp_path):
        """A cache built without per-edge keys degrades to ``edge_stats=None`` + [WARN]."""
        samples = [
            {
                "Z": torch.full((4,), 8, dtype=torch.long),
                "pos": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            }
            for _ in range(6)
        ]
        ds = CachedDataset(_write_cache(tmp_path, samples))
        with pytest.raises(ValueError):
            _ = ds.edge_counts  # precondition: the exact fast path really does raise

        result = DatasetProfiler(n_samples=3).run(ds)

        assert result.edge_stats is None
        assert result.warnings

    def test_run_rejects_empty_dataset(self):
        """Empty input is a user error, and the message must say what to pass instead."""
        with pytest.raises(ValueError, match="CachedDataset"):
            DatasetProfiler().run([])

    def test_init_rejects_degenerate_config(self):
        """``n_samples <= 0`` / ``stride <= 0`` are rejected at construction."""
        with pytest.raises(ValueError, match="n_samples"):
            DatasetProfiler(n_samples=0)
        with pytest.raises(ValueError, match="stride"):
            DatasetProfiler(stride=0)


# ---------------------------------------------------------------------------
# DatasetResult
# ---------------------------------------------------------------------------


class TestDatasetResult:
    """The report is a sectioned frame; diagnostics are ``[WARN]`` lines, never raises."""

    def test_print_report_sections(self, ring_dataset, capsys):
        """All five sections and the 72-char rule are present."""
        DatasetProfiler(n_samples=5).run(ring_dataset).print_report()

        out = capsys.readouterr().out
        assert "─" * 72 in out
        for section in ("Size", "Access", "Footprint", "Fields", "Targets"):
            assert section in out

    def test_print_report_flags_nonfinite_targets(self, tmp_path, capsys):
        """A NaN target column is counted and warned about, not raised on."""
        samples = [_record(i, n=4) for i in range(8)]
        samples[3]["targets"]["U0"] = torch.tensor([float("nan")], dtype=torch.float32)
        ds = CachedDataset(_write_cache(tmp_path, samples))

        result = DatasetProfiler(n_samples=8).run(ds)
        result.print_report()

        assert all(isinstance(t, TargetStat) for t in result.targets)
        assert any(t.n_nonfinite == 1 for t in result.targets)
        # uniform sizes + cache-backed input ⇒ the NaN column is the only warning source
        assert "[WARN]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _flatten_leaves
# ---------------------------------------------------------------------------


def test_flatten_leaves_does_not_raise_on_reserved_key():
    """Deliberate divergence from :func:`molix.data.cache._flatten`, which validates.

    The packing-time flattener rejects a sample key colliding with a reserved
    packed-cache key; the profiler's read-only walker must survive any malformed
    sample, because diagnostics never raise.
    """
    sample = {
        "schema": torch.zeros(2),  # collides with a reserved packed-cache key
        "Z": torch.ones(2, dtype=torch.long),
        "targets": {"U0": torch.tensor([1.0])},
    }

    with pytest.raises(ValueError):
        _flatten(sample)

    leaves = _flatten_leaves(sample)

    assert set(leaves) == {"schema", "Z", "targets.U0"}
    assert leaves["targets.U0"] is sample["targets"]["U0"]
