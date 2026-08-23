"""molexp workflows for the MM / Class-I program.

v0 science lives on :class:`ClassIV0Workflow`: resolve the MolHub collection,
load OMol25-small, molecule-level 8:2 self-split. Legacy experiment slugs
keep a coordinate resolve workflow so ``MmParamLearningWorkspace`` still
compiles one workflow per experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.mm_param_learning.constants import (
    COLLECTION_COORDINATE,
    EXPERIMENT_SLUGS,
    MOLHUB_COORDINATES,
    TRAIN_COORDINATE,
    V0_EXPERIMENT_SLUG,
    VAL_COORDINATE,
)


class ClassIV0Workflow:
    """OMol25-small train / OMol25-small molecule-level 8:2 val."""

    name: str = "class-i-v0"

    def compiler(self) -> Any:
        """Return an uncompiled :class:`~molexp.workflow.WorkflowCompiler`."""
        from molexp.workflow import RegisterMetric, WorkflowCompiler

        wf = WorkflowCompiler(name=self.name)

        @wf.task
        async def resolve_collection() -> dict[str, str]:
            """Read the v0 active roles. No fetch, no THEMol."""
            from molhub.dataset import Class1Universal2026

            collection = Class1Universal2026().load()
            if collection.coordinate != COLLECTION_COORDINATE:
                raise ValueError(f"unexpected collection {collection.coordinate}")
            train = collection.role("train").source_coordinates[0]
            if train != TRAIN_COORDINATE:
                raise ValueError(f"v0 train pin drifted: train={train}")
            if any("themol" in c for c in collection.source_coordinates):
                raise ValueError("THEMol leaked into the v0 resolve set")
            return {
                "collection": collection.coordinate,
                "train": train,
                "validation": VAL_COORDINATE,
            }

        @wf.task(depends_on=["resolve_collection"])
        async def fit(
            train: str,
            validation: str,
            collection: str = "",
            packed_train: str = "",
            omol_role: str = "val-neutral",
            n_train: int = 0,
            max_atoms: int = 32,
            packed_max_atoms: int = 16384,
            train_frac: float = 0.8,
            hidden: int = 64,
            n_blocks: int = 4,
            n_torsion_terms: int = 8,
            epochs: int = 2,
            lr: float = 1e-4,
            seed: int = 2666,
            device: str = "cpu",
            energy_scope: str = "full",
            record_root: str = "",
            eval_every_epochs: int = 1,
        ) -> dict[str, float | str]:
            """Load OMol25-small and fit on a molecule-level self-split."""
            from scripts.mm_param_learning.class_i_fit import ClassIFit, load_omol25_split

            if train != TRAIN_COORDINATE or validation != VAL_COORDINATE:
                raise ValueError(f"fit pins drifted: train={train} val={validation}")
            del collection
            train_records, val_records = load_omol25_split(
                packed=packed_train or None,
                role=omol_role,
                max_atoms=max_atoms,
                n_keep=n_train,
                train_frac=train_frac,
                seed=seed,
            )
            metrics = ClassIFit(
                train_records,
                val_records,
                hidden=hidden,
                n_blocks=n_blocks,
                n_torsion_terms=n_torsion_terms,
                max_atoms=packed_max_atoms,
                epochs=epochs,
                lr=lr,
                seed=seed,
                device=device,
                record_root=Path(record_root) if record_root else None,
                energy_scope=energy_scope,
                eval_every_epochs=eval_every_epochs,
            ).run()
            return {
                "train_loss": RegisterMetric(key="train_loss", value=float(metrics["train_loss"])),
                "val_mae_kcal": RegisterMetric(
                    key="val_mae_kcal", value=float(metrics["val_mae_kcal"])
                ),
                "n_train": metrics["n_train"],
                "n_val": metrics["n_val"],
                "epochs": metrics["epochs"],
                "step_per_second": metrics.get("step_per_second", float("nan")),
            }

        return wf

    def compile(self) -> Any:
        """Compile the v0 workflow."""
        return self.compiler().compile()

    def attach(self, experiment: Any, *, params: dict[str, list] | None = None) -> Any:
        """Bind this workflow onto a molexp experiment and seed runs."""
        import molexp  # noqa: F401 — registers the workflow executor

        return experiment.define(self.compile(), params=params or {"seed": [2666]})

    def task_names(self) -> tuple[str, ...]:
        """Compiled task names in graph order."""
        return tuple(self.compile().graph.task_names)


class MmParamWorkflows:
    """One compiled workflow per experiment slug."""

    TASK_NAMES: tuple[str, ...] = (
        "resolve_molhub_coordinates",
        "record_run_params",
        "write_placeholder_metrics",
    )

    def build(self, experiment_slug: str) -> Any:
        """Compile the workflow for *experiment_slug*."""
        if experiment_slug not in MOLHUB_COORDINATES:
            raise ValueError(
                f"Unknown experiment slug {experiment_slug!r}. Known: {list(EXPERIMENT_SLUGS)}"
            )
        if experiment_slug == V0_EXPERIMENT_SLUG:
            return ClassIV0Workflow().compile()
        return self._legacy_resolve(experiment_slug)

    def _legacy_resolve(self, experiment_slug: str) -> Any:
        from molexp.workflow import RegisterMetric, WorkflowCompiler

        coordinate = MOLHUB_COORDINATES[experiment_slug]
        wf = WorkflowCompiler(name=f"mm-param-{experiment_slug}")

        @wf.task
        async def resolve_molhub_coordinates() -> dict[str, str]:
            if not coordinate.startswith("dataset:"):
                raise ValueError(f"coordinate must start with 'dataset:': {coordinate!r}")
            return {"dataset_coordinate": coordinate}

        @wf.task(depends_on=["resolve_molhub_coordinates"])
        async def record_run_params(
            resolve_molhub_coordinates: dict[str, str],
        ) -> dict[str, str]:
            return dict(resolve_molhub_coordinates)

        @wf.task(depends_on=["record_run_params"])
        async def write_placeholder_metrics(
            record_run_params: dict[str, str],
        ) -> dict[str, Any]:
            return {
                "scaffold_ok": RegisterMetric(key="scaffold_ok", value=1.0),
                "dataset_coordinate": record_run_params["dataset_coordinate"],
            }

        return wf.compile()

    def task_names(self, experiment_slug: str) -> tuple[str, ...]:
        """Task names of the compiled workflow for *experiment_slug*."""
        compiled = self.build(experiment_slug)
        return tuple(compiled.graph.task_names)
