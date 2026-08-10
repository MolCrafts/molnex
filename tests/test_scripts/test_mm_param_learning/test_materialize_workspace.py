"""Unit tests for MmParamLearningWorkspace.materialize (temp root only)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("molexp")

from scripts.mm_param_learning.constants import (
    DEFAULT_WORKSPACE_ROOT,
    EXPERIMENT_SLUGS,
    KNOWLEDGE_NOTE_NAME,
    MOLHUB_COORDINATES,
    PROGRAM,
    PROJECT_SLUG,
)
from scripts.mm_param_learning.materialize_workspace import MmParamLearningWorkspace


class TestMmParamLearningWorkspace:
    def test_materialize_temp_root(self, tmp_path: Path):
        root = tmp_path / "ws"
        summary = MmParamLearningWorkspace(root=root).materialize()
        assert (root / "workspace.json").is_file()
        assert summary["project"] == PROJECT_SLUG
        assert set(summary["experiments"]) == set(EXPERIMENT_SLUGS)
        # Project + four experiments
        project_dir = root / "projects" / PROJECT_SLUG
        assert project_dir.is_dir()
        exp_dir = project_dir / "experiments"
        assert set(p.name for p in exp_dir.iterdir() if p.is_dir()) == set(EXPERIMENT_SLUGS)
        # Seed run params (molexp may prefix run ids with ``run-``).
        for slug in EXPERIMENT_SLUGS:
            runs_dir = exp_dir / slug / "runs"
            run_dirs = [p for p in runs_dir.iterdir() if p.is_dir() and "seed" in p.name]
            assert len(run_dirs) == 1, run_dirs
            run_json = run_dirs[0] / "run.json"
            assert run_json.is_file()
            text = run_json.read_text()
            assert "dataset_coordinate" in text
            assert PROGRAM in text or "mm-param-learning-baseline" in text
            assert MOLHUB_COORDINATES[slug] in text or slug in text

    def test_idempotent(self, tmp_path: Path):
        root = tmp_path / "ws"
        MmParamLearningWorkspace(root=root).materialize()
        MmParamLearningWorkspace(root=root).materialize()
        exp_dir = root / "projects" / PROJECT_SLUG / "experiments"
        assert len([p for p in exp_dir.iterdir() if p.is_dir()]) == 4
        # Knowledge note not duplicated (single slug dir)
        notes = list(root.rglob(KNOWLEDGE_NOTE_NAME))
        # at most one concept directory with that name
        note_dirs = [p for p in notes if p.is_dir()]
        assert len(note_dirs) == 1

    def test_knowledge_note_content(self, tmp_path: Path):
        root = tmp_path / "ws"
        MmParamLearningWorkspace(root=root).materialize()
        bodies = list(root.rglob("index.md"))
        assert bodies, "expected Knowledge Note index.md"
        body = "\n".join(p.read_text() for p in bodies)
        for num in ("1", "2", "3", "4", "5", "6"):
            assert f"Stage {num}" in body
        for coord in MOLHUB_COORDINATES.values():
            assert coord in body

    def test_explicit_root_not_default(self, tmp_path: Path):
        root = tmp_path / "only-here"
        MmParamLearningWorkspace(root=root).materialize()
        assert (root / "workspace.json").is_file()
        # Must not touch operator default from this call.
        assert root.resolve() != DEFAULT_WORKSPACE_ROOT.resolve()
