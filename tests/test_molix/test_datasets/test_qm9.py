"""Tests for QM9Source + integration with pipeline.materialize.

Uses a synthetic QM9 tarball placed in tmp_path to avoid network downloads.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import torch

from molix.data import MmapDataset, pipeline
from molix.datasets.qm9 import QM9_TARGET_SCHEMA, QM9Source, download_qm9


# ---------------------------------------------------------------------------
# Fake QM9 tarball fixture
# ---------------------------------------------------------------------------


def _xyz_record(
    index: int,
    *,
    natoms: int = 3,
    U0: float = 1.0,
    gap: float = 0.5,
) -> str:
    """Produce a QM9-style .xyz body for one molecule (H2O-like, 3 atoms).

    QM9 property line has these columns (in order):
      tag index A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv
    """
    props = [
        "gdb",
        str(index),
        "1.0", "1.0", "1.0",           # A B C
        "0.0", "0.0",                  # mu alpha
        "-0.1", "0.1",                 # homo lumo
        str(gap),
        "0.0", "0.0",                  # r2 zpve
        str(U0), "0.0", "0.0", "0.0",  # U0 U H G
        "0.0",                          # Cv
    ]
    lines = [
        str(natoms),
        " ".join(props),
        "O   0.0  0.0  0.0",
        "H   0.76 0.59 0.0",
        "H  -0.76 0.59 0.0",
        "",                             # freq line (ignored by _parse_xyz)
    ]
    return "\n".join(lines) + "\n"


@pytest.fixture
def fake_qm9_root(tmp_path: Path) -> Path:
    """Create a directory with a 3-molecule QM9 tarball + empty exclusion list."""
    root = tmp_path / "qm9"
    root.mkdir()

    # Exclusion list: QM9 exclusion format has 9 header lines + trailing line
    # that _load_exclusion_list skips. Provide 10 filler lines + one data line
    # that excludes no real index (we use indices 1-3 below).
    excl = ["header"] * 9 + ["9999 dummy", "trailer"]
    (root / "qm9_exclude.txt").write_text("\n".join(excl) + "\n")

    # Tarball with 3 molecules (indices 1..3 in filename suffix)
    tarball_path = root / "qm9.tar.bz2"
    with tarfile.open(tarball_path, "w:bz2") as tar:
        for i in (1, 2, 3):
            body = _xyz_record(i, U0=float(i)).encode("utf-8")
            name = f"dsgdb9nsd_{i:06d}.xyz"
            info = tarfile.TarInfo(name=name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDownloadQm9:
    def test_noop_when_files_present(self, fake_qm9_root):
        # Files already exist; should not raise, not modify.
        before = (fake_qm9_root / "qm9.tar.bz2").stat().st_mtime_ns
        download_qm9(fake_qm9_root)
        after = (fake_qm9_root / "qm9.tar.bz2").stat().st_mtime_ns
        assert before == after


class TestQM9Source:
    def test_basic_indexing(self, fake_qm9_root):
        src = QM9Source(fake_qm9_root, download=False)
        assert len(src) == 3
        sample = src[0]
        assert "Z" in sample
        assert "pos" in sample
        assert "targets" in sample
        assert sample["Z"].shape[0] == 3        # O + H + H
        assert sample["pos"].shape == (3, 3)

    def test_source_id_stable(self, fake_qm9_root):
        a = QM9Source(fake_qm9_root, download=False).source_id
        b = QM9Source(fake_qm9_root, download=False).source_id
        assert a == b

    def test_source_id_reflects_targets_filter(self, fake_qm9_root):
        full = QM9Source(fake_qm9_root, download=False).source_id
        filtered = QM9Source(
            fake_qm9_root, download=False, targets=["U0", "gap"]
        ).source_id
        assert full != filtered
        assert "targets=" not in full
        assert "targets=U0+gap" in filtered

    def test_targets_filter_shrinks_sample(self, fake_qm9_root):
        src = QM9Source(fake_qm9_root, download=False, targets=["U0"])
        assert set(src[0]["targets"].keys()) == {"U0"}

    def test_unknown_target_rejected(self, fake_qm9_root):
        with pytest.raises(ValueError, match="Unknown QM9 targets"):
            QM9Source(fake_qm9_root, download=False, targets=["not_a_qm9_prop"])

    def test_offline_without_tarball_raises(self, tmp_path):
        (tmp_path / "qm9").mkdir()
        with pytest.raises(FileNotFoundError, match="QM9 tarball not found"):
            QM9Source(tmp_path / "qm9", download=False)

    def test_targets_record_ordered_for_stable_id(self, fake_qm9_root):
        # order of the targets list must not affect source_id
        a = QM9Source(fake_qm9_root, download=False, targets=["gap", "U0"]).source_id
        b = QM9Source(fake_qm9_root, download=False, targets=["U0", "gap"]).source_id
        assert a == b


class TestQM9SourceWithMaterialize:
    """End-to-end: QM9Source → pipeline.materialize → MmapDataset.from_cache."""

    def test_materialize_roundtrip(self, fake_qm9_root, tmp_path):
        src = QM9Source(fake_qm9_root, download=False, targets=["U0"])
        spec = pipeline("qm9-test").build()       # no-op pipeline
        sink = tmp_path / "prepared"

        spec.materialize(src, sink=sink)
        ds = MmapDataset.from_cache(sink)

        assert len(ds) == 3
        assert ds.meta["source_id"] == src.source_id
        # U0 targets survive the round-trip
        u0s = sorted(float(ds[i]["targets"]["U0"].item()) for i in range(3))
        assert u0s == [1.0, 2.0, 3.0]

    def test_qm9_schema_exported(self):
        """QM9_TARGET_SCHEMA is exported as a module-level constant for workflows."""
        assert "U0" in QM9_TARGET_SCHEMA.graph_level
        assert "gap" in QM9_TARGET_SCHEMA.graph_level
        assert QM9_TARGET_SCHEMA.atom_level == frozenset()
