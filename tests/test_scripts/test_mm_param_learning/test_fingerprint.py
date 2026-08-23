"""Fingerprint fail-closed gates (no molpy)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.mm_param_learning import verify_fingerprint
from scripts.mm_param_learning.fingerprint import (
    PROD_MODELS,
    QUASI_PROD,
    attach_fingerprint,
    fingerprint,
    require_fingerprint_file,
    require_not_production_go,
    require_power_of_two,
    write_fingerprint,
)


class TestRequirePowerOfTwo:
    def test_rejects_three(self):
        with pytest.raises(ValueError, match="power of two"):
            require_power_of_two("hidden", 3)

    def test_accepts_sixty_four(self):
        require_power_of_two("hidden", 64)


class TestFingerprint:
    def test_writes_hash(self, tmp_path: Path):
        blob = fingerprint()
        assert len(blob["fingerprint"]) == 64
        assert blob["config"]["hidden"] == 64
        assert blob["config"]["val_split"] == "omol25-small-molecule-8-2"
        assert blob["config"]["packed_max_atoms"] == 16384
        path = write_fingerprint(tmp_path / "fp.json", blob)
        loaded = require_fingerprint_file(path)
        assert loaded["fingerprint"] == blob["fingerprint"]

    def test_missing_file_fails_closed(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="fail-closed"):
            require_fingerprint_file(tmp_path / "absent.json")

    def test_quasi_prod_sizes_are_powers_of_two(self):
        for key in ("hidden", "n_blocks", "n_torsion_terms", "packed_max_atoms"):
            require_power_of_two(key, int(QUASI_PROD[key]))

    def test_prod_models_are_three_powers_of_two(self):
        assert set(PROD_MODELS) == {"small", "medium", "large"}
        for name, spec in PROD_MODELS.items():
            for key in ("hidden", "n_blocks", "n_torsion_terms"):
                require_power_of_two(f"{name}.{key}", int(spec[key]))


class TestRequireNotProductionGo:
    def test_go_without_quasi_tag_raises(self, tmp_path: Path):
        path = tmp_path / "signoff.yaml"
        path.write_text("decision: GO\n")
        with pytest.raises(RuntimeError, match="not a production gate"):
            require_not_production_go(path)

    def test_missing_is_ok(self, tmp_path: Path):
        require_not_production_go(tmp_path / "none.yaml")


class TestAttachFingerprint:
    def test_records_hash_first(self):
        assert attach_fingerprint({"a": 1}, "abc") == {"fingerprint": "abc", "a": 1}


class TestVerifyFingerprint:
    def _write_fp(self, run_dir: Path, fp: str) -> None:
        (run_dir / "fingerprint.json").write_text(json.dumps({"fingerprint": fp}))

    def test_all_match(self, tmp_path: Path):
        self._write_fp(tmp_path, "f" * 64)
        (tmp_path / "split.json").write_text(json.dumps({"fingerprint": "f" * 64}))
        (tmp_path / "metrics.json").write_text(json.dumps({"fingerprint": "f" * 64}))
        assert verify_fingerprint.main(["verify_fingerprint.py", str(tmp_path)]) == 0

    def test_mismatch_fails(self, tmp_path: Path):
        self._write_fp(tmp_path, "f" * 64)
        (tmp_path / "split.json").write_text(json.dumps({"fingerprint": "e" * 64}))
        assert verify_fingerprint.main(["verify_fingerprint.py", str(tmp_path)]) == 1

    def test_absent_field_fails(self, tmp_path: Path):
        self._write_fp(tmp_path, "f" * 64)
        (tmp_path / "split.json").write_text(json.dumps({"coverage": {}}))
        assert verify_fingerprint.main(["verify_fingerprint.py", str(tmp_path)]) == 1

    def test_missing_fingerprint_file_fails(self, tmp_path: Path):
        assert verify_fingerprint.main(["verify_fingerprint.py", str(tmp_path)]) == 1

    def test_no_artifacts_fails(self, tmp_path: Path):
        self._write_fp(tmp_path, "f" * 64)
        assert verify_fingerprint.main(["verify_fingerprint.py", str(tmp_path)]) == 1
