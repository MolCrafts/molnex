"""Unit tests for MmParamWorkflows and ClassIV0Workflow."""

from __future__ import annotations

import pytest

pytest.importorskip("molexp")

from scripts.mm_param_learning.constants import EXPERIMENT_SLUGS, V0_EXPERIMENT_SLUG
from scripts.mm_param_learning.workflows import ClassIV0Workflow, MmParamWorkflows


class TestMmParamWorkflows:
    def test_legacy_slugs_keep_resolve_tasks(self):
        wf = MmParamWorkflows()
        for slug in EXPERIMENT_SLUGS:
            if slug == V0_EXPERIMENT_SLUG:
                continue
            names = wf.task_names(slug)
            assert names == MmParamWorkflows.TASK_NAMES

    def test_v0_slug_is_the_class_i_workflow(self):
        names = MmParamWorkflows().task_names(V0_EXPERIMENT_SLUG)
        assert names == ("resolve_collection", "fit")

    def test_unknown_slug_raises(self):
        with pytest.raises(ValueError, match="Unknown experiment slug"):
            MmParamWorkflows().build("not-a-real-experiment")


class TestClassIV0Workflow:
    def test_task_names(self):
        assert ClassIV0Workflow().task_names() == ("resolve_collection", "fit")

    def test_fit_does_not_load_phalkethoh(self):
        from pathlib import Path

        text = Path("scripts/mm_param_learning/workflows.py").read_text()
        assert "load_phalkethoh_val" not in text
        assert "load_omol25_split" in text


class TestTrainEntryIsDeclarationOnly:
    def test_no_argparse_or_main(self):
        from pathlib import Path

        text = Path("scripts/mm_param_learning/train.py").read_text()
        assert "import argparse" not in text
        assert "def main(" not in text
        assert "ClassIV0Workflow" in text
