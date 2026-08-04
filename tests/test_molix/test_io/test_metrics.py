"""MolRec metrics JSONL binding (provisional in molix.io.metrics)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molix.io.metrics import MetricsWriter, read_metrics, rebuild_metrics_index
from molix.io.run_record import ensure_run_package, meta_path, status_path


class TestMetricsWriter:
    def test_scalar_append_and_index(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        writer = MetricsWriter(root)
        writer.scalar("train/loss", 0.5, step=1)
        writer.scalar("train/loss", 0.25, step=2)
        writer.scalar("eval/MAE", 0.1, step=2)
        writer.flush()

        metrics_file = root / "metrics" / "metrics.jsonl"
        index_file = root / "metrics" / "index.json"
        assert metrics_file.is_file()
        assert index_file.is_file()

        lines = [json.loads(line) for line in metrics_file.read_text().splitlines() if line.strip()]
        assert [row["k"] for row in lines] == ["train/loss", "train/loss", "eval/MAE"]
        assert lines[0]["t"] == "scalar"
        assert lines[0]["v"] == 0.5
        assert "w" in lines[0]

        index = json.loads(index_file.read_text())
        assert index["line_count"] == 3
        assert index["series_count"] == 2
        assert index["series"]["train/loss"]["count"] == 2
        assert index["series"]["train/loss"]["latest_step"] == 2

    def test_rejects_nan_scalar(self, tmp_path: Path) -> None:
        writer = MetricsWriter(tmp_path)
        with pytest.raises(ValueError, match="scalar metric value"):
            writer.scalar("train/loss", float("nan"), step=1)

    def test_read_filters_and_skips_bad_lines(self, tmp_path: Path) -> None:
        writer = MetricsWriter(tmp_path)
        writer.scalar("train/loss", 0.3, step=1)
        writer.text("note", "warmup", step=1)
        path = tmp_path / "metrics" / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{bad\n")
            fh.write(json.dumps({"t": "scalar", "k": "train/loss", "s": 2, "w": "t", "v": 0.2}))
            fh.write("\n")

        result = read_metrics(tmp_path, metric_type="scalar", key="train/loss", since_line=1)
        assert result.parse_errors == 1
        assert len(result.records) == 1
        assert result.records[0]["v"] == 0.2


class TestRunPackage:
    def test_ensure_run_package_writes_meta_status(self, tmp_path: Path) -> None:
        root = ensure_run_package(tmp_path / "pkg", state="running", stage="train")
        meta = json.loads(meta_path(root).read_text())
        status = json.loads(status_path(root).read_text())
        assert meta["record_schema_version"] == 1
        assert meta["format_name"] == "molrec"
        assert status["state"] == "running"
        assert status["stage"] == "train"

    def test_round_trip_matches_fixture_shape(self, tmp_path: Path) -> None:
        """Same relative paths as molrec fixtures/run-minimal."""
        root = ensure_run_package(tmp_path / "pkg")
        writer = MetricsWriter(root)
        writer.scalar("train/loss", 0.5, step=1)
        writer.flush()
        rebuild_metrics_index(root)

        assert (root / "meta" / "meta.json").is_file()
        assert (root / "status" / "status.json").is_file()
        assert (root / "metrics" / "metrics.jsonl").is_file()
        assert (root / "metrics" / "index.json").is_file()
