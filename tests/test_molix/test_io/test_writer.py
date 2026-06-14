"""RED tests for ``molix.io.JournalWriter`` (kwargs API).

Acceptance traces:
    ac-004 — kwargs API, dispatches five reserved types, ValueError on unknown.
    ac-006 — Zarr v3 sharded layout (chunks=1024, shards=1_048_576) preserved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest


def _append_scalar(
    writer: Any, step: int, key: str = "train/loss", value: float | None = None
) -> None:
    writer.append(
        type="scalar",
        key=key,
        step=step,
        wall_time_ns=1_700_000_000_000_000_000 + step,
        value=float(step) if value is None else value,
    )


def test_writer_importable() -> None:
    """`from molix.io import JournalWriter` succeeds."""
    from molix.io import JournalWriter  # noqa: F401


def test_append_kwargs_api(tmp_path: Path) -> None:
    """ac-004: ``append`` accepts kwargs (no MetricRecord), persists a scalar."""
    from molix.io import JournalWriter

    with JournalWriter(tmp_path, run_id="run0") as writer:
        writer.append(
            type="scalar",
            key="train/loss",
            step=0,
            wall_time_ns=1_700_000_000_000_000_000,
            value=1.23,
        )

    import zarr

    grp = zarr.open(str(tmp_path), mode="r")["metrics/records/run0"]
    assert str(grp["type"][0]) == "scalar"
    assert str(grp["key"][0]) == "train/loss"
    assert int(grp["step"][0]) == 0
    assert float(grp["value_scalar"][0]) == pytest.approx(1.23)


@pytest.mark.parametrize(
    "type_name, value",
    [
        ("scalar", 0.42),
        ("histogram", {"bins": [0.0, 0.5, 1.0], "counts": [3.0, 5.0]}),
        ("text", "first epoch ok"),
        ("image_ref", {"path": "viz/0.png", "caption": "first"}),
        ("json", {"lr": 1e-3, "model": "mace"}),
    ],
)
def test_five_reserved_types_dispatch(tmp_path: Path, type_name: str, value: Any) -> None:
    """ac-004: All five wire-format types dispatch correctly inside ``append``."""
    from molix.io import JournalWriter

    with JournalWriter(tmp_path, run_id="run0") as writer:
        writer.append(
            type=type_name,
            key="some/key",
            step=0,
            wall_time_ns=1,
            value=value,
        )

    import zarr

    grp = zarr.open(str(tmp_path), mode="r")["metrics/records/run0"]
    assert str(grp["type"][0]) == type_name


def test_append_unknown_type_raises(tmp_path: Path) -> None:
    """ac-004: unknown ``type`` raises ValueError (replaces dataclass validation)."""
    from molix.io import JournalWriter

    with JournalWriter(tmp_path, run_id="run0") as writer:
        with pytest.raises(ValueError):
            writer.append(
                type="bogus",
                key="x",
                step=0,
                wall_time_ns=1,
                value=0.0,
            )


def test_zarr_layout_chunks_and_shards(tmp_path: Path) -> None:
    """ac-006: ``chunks=(1024,)``, ``shards=(1_048_576,)`` preserved verbatim.

    HPC-inode discipline relies on these exact values per the original
    ``recorder/__init__.py`` rationale (mirrors ``PackedCache``).
    """
    from molix.io import JournalWriter

    with JournalWriter(tmp_path, run_id="run0") as writer:
        _append_scalar(writer, step=0)

    import zarr

    grp = zarr.open(str(tmp_path), mode="r")["metrics/records/run0"]

    arr = grp["type"]
    assert arr.chunks == (1024,), f"chunks must be (1024,); got {arr.chunks}"

    shards = getattr(arr, "shards", None)
    assert shards == (1_048_576,), f"shards must be (1_048_576,); got {shards}"


def test_sharding_keeps_file_count_small(tmp_path: Path) -> None:
    """ac-006: 100k records produce O(50) files (PackedCache-equivalent inode discipline)."""
    from molix.io import JournalWriter

    keys = [f"train/m{i}" for i in range(5)]
    with JournalWriter(tmp_path, run_id="run0") as writer:
        for step in range(20_000):
            for k in keys:
                _append_scalar(writer, step=step, key=k, value=float(step))

    run_dir = tmp_path / "metrics" / "records" / "run0"
    files = []
    for root, _, fnames in os.walk(run_dir):
        for f in fnames:
            if f.endswith(".json"):
                continue
            files.append(os.path.join(root, f))

    assert len(files) <= 50, (
        f"Sharding regression: 100k records produced {len(files)} files (expected ≤ 50)."
    )


def test_append_after_close_raises(tmp_path: Path) -> None:
    """Late-write rejection contract preserved from ZarrMetricStore."""
    from molix.io import JournalWriter

    writer = JournalWriter(tmp_path, run_id="run0")
    _append_scalar(writer, step=0)
    writer.close()

    with pytest.raises(RuntimeError):
        _append_scalar(writer, step=1)


def test_empty_key_rejected(tmp_path: Path) -> None:
    """Non-empty key contract preserved."""
    from molix.io import JournalWriter

    with JournalWriter(tmp_path, run_id="run0") as writer:
        with pytest.raises(ValueError):
            writer.append(
                type="scalar",
                key="",
                step=0,
                wall_time_ns=1,
                value=0.0,
            )


# ---------------------------------------------------------------------------
# Durability flags — autoflush (SIGKILL-safe) + fsync (crash-safe).
# A logging.FileHandler-style line-buffered sink: each append reaches the OS
# at once, so a hard kill that skips close() still leaves the records on disk.
# ---------------------------------------------------------------------------


def _read_scalars(tmp_path: Path, run_id: str = "run0") -> list[float]:
    """Read back value_scalar without going through close()/the index group."""
    import zarr

    grp = zarr.open(str(tmp_path), mode="r")[f"metrics/records/{run_id}"]
    return [float(v) for v in grp["value_scalar"][:]]


def test_default_buffers_until_flush(tmp_path: Path) -> None:
    """Without autoflush, appended records are NOT on disk until flush/close."""
    from molix.io import JournalWriter

    writer = JournalWriter(tmp_path, run_id="run0")  # no context manager → no close
    _append_scalar(writer, step=0, value=1.0)
    _append_scalar(writer, step=1, value=2.0)
    assert _read_scalars(tmp_path) == []  # still buffered in memory

    writer.flush()
    assert _read_scalars(tmp_path) == [1.0, 2.0]


def test_autoflush_persists_without_close(tmp_path: Path) -> None:
    """autoflush=True: records survive even when close() never runs (SIGKILL)."""
    from molix.io import JournalWriter

    writer = JournalWriter(tmp_path, run_id="run0", autoflush=True)
    _append_scalar(writer, step=0, value=10.0)
    _append_scalar(writer, step=1, value=20.0)
    # Deliberately DO NOT call writer.close() — simulate a killed process.
    assert _read_scalars(tmp_path) == [10.0, 20.0]


def test_autoflush_reader_recovers_without_index(tmp_path: Path) -> None:
    """JournalReader reads an autoflushed, never-closed run (rebuilds the key index)."""
    from molix.io import JournalReader, JournalWriter

    writer = JournalWriter(tmp_path, run_id="run0", autoflush=True)
    _append_scalar(writer, step=0, key="train/loss", value=0.5)
    _append_scalar(writer, step=1, key="eval/E_MAE", value=0.1)
    # no close()

    reader = JournalReader(tmp_path, run_id="run0")
    assert len(reader) == 2
    assert set(reader.keys()) == {"train/loss", "eval/E_MAE"}


def test_fsync_flag_persists_and_is_safe(tmp_path: Path) -> None:
    """autoflush + fsync: still correct, and fsync over the store does not error."""
    from molix.io import JournalWriter

    writer = JournalWriter(tmp_path, run_id="run0", autoflush=True, fsync=True)
    _append_scalar(writer, step=0, value=42.0)
    _append_scalar(writer, step=1, value=43.0)
    assert _read_scalars(tmp_path) == [42.0, 43.0]
