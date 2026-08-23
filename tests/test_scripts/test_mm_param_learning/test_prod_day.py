"""Unit tests for the one-day Class-I launcher config."""

from __future__ import annotations

import pytest
from scripts.mm_param_learning.configs import production_calibration_drift
from scripts.mm_param_learning.prod_day import config_for


class TestConfigFor:
    def test_unknown_size_raises(self):
        with pytest.raises(ValueError, match="unknown model size"):
            config_for("tiny")

    def test_three_sizes_differ_in_hidden(self):
        small = config_for("small")
        medium = config_for("medium")
        large = config_for("large")
        assert small["hidden"] < medium["hidden"] < large["hidden"]
        assert small["n_blocks"] < medium["n_blocks"] < large["n_blocks"]
        assert small["kind"] == "prod-day1"
        assert medium["packed_max_atoms"] == 16384
        assert large["omol_role"] == "4m"
        assert large["val_role"] == "val-neutral"
        assert large["slurm_account"] == "naiss2026-4-715-gpu"


class TestCalibrationIdentity:
    def test_production_matches_calibration_except_duration_and_output(self):
        """Checklist §0: science pins are identical; only enumerated keys differ."""
        assert production_calibration_drift() == []
