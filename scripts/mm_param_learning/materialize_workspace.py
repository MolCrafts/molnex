"""Idempotent materialization of the MM parameter-learning molexp workspace.

Creates ``Workspace → Project → Experiment → Run`` under an explicit *root*
(temp in CI) or :data:`DEFAULT_WORKSPACE_ROOT` for operators. Never fetches
MolHub artifacts; seed run params only record coordinates and program ids.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.mm_param_learning.constants import (
    DEFAULT_WORKSPACE_ROOT,
    EXPERIMENT_SLUGS,
    KNOWLEDGE_NOTE_NAME,
    MOLHUB_COORDINATES,
    PROGRAM,
    PROJECT_SLUG,
    VALIDATION_STAGES,
)
from scripts.mm_param_learning.workflows import MmParamWorkflows


def _knowledge_body() -> str:
    lines = [
        "# MM Parameter-Learning Validation — Stages 1–6",
        "",
        f"Program: `{PROGRAM}`",
        "",
        "## Stages",
        "",
    ]
    for num, title in VALIDATION_STAGES:
        lines.append(f"### Stage {num} — {title}")
        lines.append("")
    lines.extend(
        [
            "## MolHub coordinates",
            "",
            "Data is loaded via the **Python molhub SDK** only (no molhub MCP plane).",
            "Workspace orchestration uses **molexp**.",
            "",
        ]
    )
    for slug, coord in MOLHUB_COORDINATES.items():
        lines.append(f"- `{slug}` → `{coord}`")
    lines.append("")
    return "\n".join(lines)


class MmParamLearningWorkspace:
    """Materialize the milestone-1 validation workspace tree.

    Idempotent: a second call keeps experiment count at four and does not
    duplicate the validation Knowledge Note.

    Args:
        root: Workspace root directory. Defaults to
            :data:`DEFAULT_WORKSPACE_ROOT`. Tests **must** pass a temp path.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_WORKSPACE_ROOT

    def materialize(self) -> dict[str, Any]:
        """Create workspace layout, four experiments, seed runs, and note.

        Returns:
            Summary dict with ``root``, ``project``, ``experiments``, ``note``.

        Raises:
            ImportError: If molexp is not installed.
        """
        try:
            from molexp import Workspace
            from molexp.workspace.bundle import Bundle
        except ImportError as exc:  # pragma: no cover - soft dep
            raise ImportError(
                "molexp is required for MmParamLearningWorkspace.materialize"
            ) from exc

        self.root.mkdir(parents=True, exist_ok=True)
        ws = Workspace(self.root, name=PROJECT_SLUG)
        ws.materialize()
        project = ws.add_project(PROJECT_SLUG)
        project.materialize()

        workflows = MmParamWorkflows()
        experiments: list[str] = []
        for slug in EXPERIMENT_SLUGS:
            coord = MOLHUB_COORDINATES[slug]
            exp = project.add_experiment(
                slug,
                params={
                    "dataset_coordinate": coord,
                    "milestone": 1,
                    "program": PROGRAM,
                },
                description=f"MM param validation experiment {slug}",
                tags=["mm-param-learning", "milestone-1"],
            )
            exp.materialize()
            # Seed run — fixed id for idempotency.
            run = exp.add_run(
                params={
                    "dataset_coordinate": coord,
                    "milestone": 1,
                    "program": PROGRAM,
                    "experiment": slug,
                },
                id=f"seed-{slug}",
            )
            run.materialize()
            # Compile workflow for side-effect validation (not executed).
            workflows.build(slug)
            experiments.append(slug)

        bundle = Bundle(ws.root)
        note = bundle.create_note(KNOWLEDGE_NOTE_NAME, body=_knowledge_body())

        return {
            "root": str(self.root),
            "project": PROJECT_SLUG,
            "experiments": experiments,
            "note": KNOWLEDGE_NOTE_NAME,
            "workspace": ws,
            "note_obj": note,
        }


def main(argv: list[str] | None = None) -> int:
    """CLI entry: materialize the operator workspace (or ``--root`` override)."""
    parser = argparse.ArgumentParser(
        description="Materialize mm-param-learning molexp workspace (idempotent)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workspace root (default: operator DEFAULT_WORKSPACE_ROOT).",
    )
    args = parser.parse_args(argv)
    summary = MmParamLearningWorkspace(root=args.root).materialize()
    print(f"materialized project={summary['project']} root={summary['root']}")
    print(f"experiments={summary['experiments']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
