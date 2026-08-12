"""Idempotent materialization of the MM parameter-learning molexp workspace.

Creates ``Workspace → Project → Experiment → Run`` under an explicit *root*
(temp in CI) or :data:`DEFAULT_WORKSPACE_ROOT` for operators. Never fetches
MolHub artifacts; seed run params only record coordinates and program ids.

Knowledge is written as sourced :class:`~molexp.workspace.KnowledgeItem`
objects (OKF), not unsourced free Notes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.mm_param_learning.constants import (
    DEFAULT_NORMALIZED_DATA_ROOT,
    DEFAULT_WORKSPACE_ROOT,
    EXPERIMENT_SLUGS,
    KNOWLEDGE_NOTE_NAME,
    MOLHUB_COORDINATES,
    PROGRAM,
    PROJECT_SLUG,
    PROTOCOL_EXPERIMENT_SLUG,
    VALIDATION_STAGES,
)
from scripts.mm_param_learning.workflows import MmParamWorkflows


def _knowledge_body() -> str:
    lines = [
        "# MM Parameter-Learning Validation — Stages",
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
            "## Hierarchy",
            "",
            "Managed by **molexp**: `Workspace → Project → Experiment → Run`.",
            "Datasets/models coordinates by **molhub** (Python SDK).",
            "",
            "## MolHub coordinates",
            "",
            "Data is loaded via the **Python molhub SDK** only (no molhub MCP plane).",
            "Workspace orchestration uses **molexp**; agents discover tools on **molmcp molexp** plane.",
            "",
            f"Default normalized PhAlkEthOH root: `{DEFAULT_NORMALIZED_DATA_ROOT}`",
            "",
        ]
    )
    for slug, coord in MOLHUB_COORDINATES.items():
        lines.append(f"- `{slug}` → `{coord}`")
    lines.extend(
        [
            "",
            "## Protocol train entry",
            "",
            "```bash",
            "python scripts/mm_param_learning/setup_class_i_protocol.py",
            "python scripts/mm_param_learning/espaloma_mm_protocol_train.py \\",
            f"  --data-root {DEFAULT_NORMALIZED_DATA_ROOT} --energy-scope full \\",
            f"  --workspace {DEFAULT_WORKSPACE_ROOT} \\",
            f"  --experiment {PROTOCOL_EXPERIMENT_SLUG} --run-id protocol-full-energy-mini-001",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


class MmParamLearningWorkspace:
    """Materialize the milestone-1 validation workspace tree.

    Idempotent: a second call keeps experiment count stable and refreshes
    the validation KnowledgeItem body.

    Args:
        root: Workspace root directory. Defaults to
            :data:`DEFAULT_WORKSPACE_ROOT`. Tests **must** pass a temp path.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_WORKSPACE_ROOT

    def materialize(self) -> dict[str, Any]:
        """Create workspace layout, experiments, seed runs, and knowledge.

        Returns:
            Summary dict with ``root``, ``project``, ``experiments``, ``note``.

        Raises:
            ImportError: If molexp is not installed.
        """
        try:
            from molexp import Workspace
            from molexp.workspace import SourceRef, write_knowledge_item
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

        sources = [SourceRef(kind="experiment", ref=slug) for slug in experiments]
        if not sources:
            sources = [SourceRef(kind="experiment", ref=PROTOCOL_EXPERIMENT_SLUG)]
        sources.extend(
            SourceRef(kind="file", ref=coord)
            for coord in sorted(set(MOLHUB_COORDINATES.values()))
        )
        write_knowledge_item(
            project,
            name=KNOWLEDGE_NOTE_NAME,
            kind="ProtocolNote",
            sources=sources,
            created_by="mm-param-learning/materialize_workspace",
            body=_knowledge_body(),
            title="MM param validation stages",
        )

        return {
            "root": str(self.root),
            "project": PROJECT_SLUG,
            "experiments": experiments,
            "note": KNOWLEDGE_NOTE_NAME,
            "workspace": ws,
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
