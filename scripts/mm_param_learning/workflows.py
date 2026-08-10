"""Stub WorkflowCompiler skeletons for MM parameter-learning experiments.

Each experiment compiles a three-task workflow:

1. ``resolve_molhub_coordinates`` — shape-check ``dataset:`` coordinates (no network)
2. ``record_run_params`` — echo seed params for audit
3. ``write_placeholder_metrics`` — write ``scaffold_ok=1.0`` via RegisterMetric

These stubs intentionally do **not** fetch datasets or run science kernels.
"""

from __future__ import annotations

from typing import Any

from scripts.mm_param_learning.constants import EXPERIMENT_SLUGS, MOLHUB_COORDINATES


class MmParamWorkflows:
    """Factory for per-experiment milestone-1 workflow stubs.

    Args:
        None — all configuration comes from :mod:`constants`.
    """

    TASK_NAMES: tuple[str, ...] = (
        "resolve_molhub_coordinates",
        "record_run_params",
        "write_placeholder_metrics",
    )

    def build(self, experiment_slug: str) -> Any:
        """Compile a three-task workflow for *experiment_slug*.

        Args:
            experiment_slug: One of :data:`EXPERIMENT_SLUGS`.

        Returns:
            A molexp ``CompiledWorkflow`` with :attr:`TASK_NAMES` in order.

        Raises:
            ValueError: If *experiment_slug* is unknown.
            ImportError: If molexp is not installed.
        """
        if experiment_slug not in MOLHUB_COORDINATES:
            raise ValueError(
                f"Unknown experiment slug {experiment_slug!r}. Known: {list(EXPERIMENT_SLUGS)}"
            )
        try:
            from molexp.workflow import RegisterMetric, WorkflowCompiler
        except ImportError as exc:  # pragma: no cover - soft dep
            raise ImportError(
                "molexp is required to build MmParamWorkflows; install molexp "
                "or skip these tests when molexp is absent."
            ) from exc

        coordinate = MOLHUB_COORDINATES[experiment_slug]
        wf = WorkflowCompiler(name=f"mm-param-{experiment_slug}")

        @wf.task
        async def resolve_molhub_coordinates() -> dict[str, str]:
            """Shape-check the experiment's MolHub coordinate (no network)."""
            if not coordinate.startswith("dataset:"):
                raise ValueError(f"coordinate must start with 'dataset:': {coordinate!r}")
            return {"dataset_coordinate": coordinate}

        @wf.task(depends_on=["resolve_molhub_coordinates"])
        async def record_run_params(
            resolve_molhub_coordinates: dict[str, str],
        ) -> dict[str, str]:
            """Echo resolved coordinate for run-params audit."""
            return dict(resolve_molhub_coordinates)

        @wf.task(depends_on=["record_run_params"])
        async def write_placeholder_metrics(
            record_run_params: dict[str, str],
        ) -> dict[str, Any]:
            """Write placeholder scaffold metric (no science)."""
            return {
                "scaffold_ok": RegisterMetric(key="scaffold_ok", value=1.0),
                "dataset_coordinate": record_run_params["dataset_coordinate"],
            }

        return wf.compile()

    def task_names(self, experiment_slug: str) -> tuple[str, ...]:
        """Return the three stub task names after a successful :meth:`build`."""
        compiled = self.build(experiment_slug)
        return tuple(compiled.graph.task_names)
