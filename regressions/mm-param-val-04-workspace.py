#!/usr/bin/env python
"""Regression: mm-param-learning workspace layout goldens (temp root only)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mm_param_learning.constants import (  # noqa: E402
    DEFAULT_WORKSPACE_ROOT,
    EXPERIMENT_SLUGS,
    MOLHUB_COORDINATES,
    PROJECT_SLUG,
)
from scripts.mm_param_learning.materialize_workspace import (  # noqa: E402
    MmParamLearningWorkspace,
)
from scripts.mm_param_learning.workflows import MmParamWorkflows  # noqa: E402


def main() -> None:
    try:
        import molexp  # noqa: F401
    except ImportError:
        print("mm-param-val-04-workspace: SKIP (molexp not installed)")
        return

    assert set(EXPERIMENT_SLUGS) == {
        "potential-parity",
        "zinc-typing-recovery",
        "phalkethoh-mm-energy",
        "latent-analysis",
    }
    wf = MmParamWorkflows()
    for slug in EXPERIMENT_SLUGS:
        assert wf.task_names(slug) == MmParamWorkflows.TASK_NAMES

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ws"
        assert root.resolve() != DEFAULT_WORKSPACE_ROOT.resolve()
        s1 = MmParamLearningWorkspace(root=root).materialize()
        s2 = MmParamLearningWorkspace(root=root).materialize()
        assert s1["project"] == PROJECT_SLUG
        assert set(s2["experiments"]) == set(EXPERIMENT_SLUGS)
        exp_dir = root / "projects" / PROJECT_SLUG / "experiments"
        assert len([p for p in exp_dir.iterdir() if p.is_dir()]) == 4
        for coord in MOLHUB_COORDINATES.values():
            assert coord.startswith("dataset:")
    print("mm-param-val-04-workspace: OK")


if __name__ == "__main__":
    main()
