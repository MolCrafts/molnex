"""RED tests for ``molix.io.JournalReader`` (yields plain ``dict``).

Acceptance traces:
    ac-005 — JournalReader yields plain dicts (not dataclass instances).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _write_records(tmp_path: Path, records: list[dict[str, Any]]) -> None:
    from molix.io import JournalWriter

    with JournalWriter(tmp_path, run_id="run0") as writer:
        for r in records:
            writer.append(**r)


def test_reader_importable() -> None:
    from molix.io import JournalReader  # noqa: F401


def test_records_yields_plain_dicts(tmp_path: Path) -> None:
    """ac-005: ``next(reader.records())`` returns a plain ``dict`` instance."""
    from molix.io import JournalReader

    _write_records(
        tmp_path,
        [
            {
                "type": "scalar",
                "key": "train/loss",
                "step": 0,
                "wall_time_ns": 1,
                "value": 0.42,
            }
        ],
    )

    reader = JournalReader(tmp_path, run_id="run0")
    record = next(iter(reader.records()))
    assert isinstance(record, dict)
    assert type(record).__name__ == "dict"
    assert record["type"] == "scalar"
    assert record["key"] == "train/loss"
    assert record["step"] == 0
    assert record["value"] == pytest.approx(0.42)


def test_five_types_roundtrip_as_dict(tmp_path: Path) -> None:
    """All five wire-format types round-trip into the dict shape."""
    from molix.io import JournalReader

    records = [
        {"type": "scalar", "key": "train/loss", "step": 0, "wall_time_ns": 1, "value": 0.42},
        {
            "type": "histogram",
            "key": "weights/embed",
            "step": 0,
            "wall_time_ns": 2,
            "value": {"bins": [0.0, 0.5, 1.0], "counts": [3.0, 5.0]},
        },
        {"type": "text", "key": "note/run", "step": 0, "wall_time_ns": 3, "value": "ok"},
        {
            "type": "image_ref",
            "key": "viz/sample",
            "step": 0,
            "wall_time_ns": 4,
            "value": {"path": "viz/0.png", "caption": "first"},
        },
        {"type": "json", "key": "hparams", "step": 0, "wall_time_ns": 5, "value": {"lr": 1e-3}},
    ]
    _write_records(tmp_path, records)

    read = list(JournalReader(tmp_path, run_id="run0").records())
    assert len(read) == len(records)

    for orig, got in zip(records, read, strict=True):
        assert isinstance(got, dict)
        assert got["type"] == orig["type"]
        assert got["key"] == orig["key"]
        assert got["step"] == orig["step"]

        if orig["type"] == "scalar":
            assert got["value"] == pytest.approx(orig["value"])
        elif orig["type"] == "histogram":
            np.testing.assert_array_equal(
                np.asarray(got["value"]["bins"]), np.asarray(orig["value"]["bins"])
            )
            np.testing.assert_array_equal(
                np.asarray(got["value"]["counts"]), np.asarray(orig["value"]["counts"])
            )
        else:
            assert got["value"] == orig["value"]


def test_keys_enumerate_distinct(tmp_path: Path) -> None:
    """Reader exposes a ``keys()`` API that lists distinct keys in append order."""
    from molix.io import JournalReader

    _write_records(
        tmp_path,
        [
            {"type": "scalar", "key": "train/loss", "step": 0, "wall_time_ns": 1, "value": 0.1},
            {"type": "scalar", "key": "eval/MAE", "step": 0, "wall_time_ns": 2, "value": 0.2},
            {"type": "scalar", "key": "train/loss", "step": 1, "wall_time_ns": 3, "value": 0.3},
        ],
    )

    reader = JournalReader(tmp_path, run_id="run0")
    keys = reader.keys()
    assert keys == ["train/loss", "eval/MAE"]


def test_scalars_returns_step_value_arrays(tmp_path: Path) -> None:
    """Scalar series read returns ``(steps, values)`` parallel ndarrays."""
    from molix.io import JournalReader

    _write_records(
        tmp_path,
        [
            {"type": "scalar", "key": "train/loss", "step": i, "wall_time_ns": i, "value": float(i)}
            for i in range(5)
        ],
    )

    reader = JournalReader(tmp_path, run_id="run0")
    steps, values = reader.scalars("train/loss")

    assert isinstance(steps, np.ndarray)
    assert isinstance(values, np.ndarray)
    np.testing.assert_array_equal(steps, np.arange(5))
    np.testing.assert_array_equal(values, np.arange(5, dtype=np.float64))
