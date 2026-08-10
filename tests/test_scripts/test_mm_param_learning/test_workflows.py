"""Unit tests for MmParamWorkflows stubs."""

from __future__ import annotations

import pytest

pytest.importorskip("molexp")

from scripts.mm_param_learning.constants import EXPERIMENT_SLUGS
from scripts.mm_param_learning.workflows import MmParamWorkflows


class TestMmParamWorkflows:
    def test_three_tasks_per_slug(self):
        wf = MmParamWorkflows()
        for slug in EXPERIMENT_SLUGS:
            names = wf.task_names(slug)
            assert names == MmParamWorkflows.TASK_NAMES
            assert "resolve_molhub_coordinates" in names
            assert "record_run_params" in names
            assert "write_placeholder_metrics" in names

    def test_unknown_slug_raises(self):
        with pytest.raises(ValueError, match="Unknown experiment slug"):
            MmParamWorkflows().build("not-a-real-experiment")
