"""Eval report for a potential: forces, energy per atom, virial, outliers.

The numbers are :class:`~molix.core.metrics.PotentialReport`. This hook
reads them off the batch and the step's predictions and writes the
resulting scalars. It does not look at a loss weight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from molix.core.hook import ScalarHook
from molix.core.metrics import PotentialReport
from molix.core.state import Path


def _leaf(container: Any, namespace: str, key: str) -> torch.Tensor | None:
    """Read ``container[namespace][key]`` or a flat ``container[key]``."""
    if container is None or not hasattr(container, "__getitem__"):
        return None
    try:
        namespace_value = container[namespace]
    except (KeyError, IndexError, TypeError):
        namespace_value = None
    if namespace_value is not None:
        try:
            value = namespace_value[key]
        except (KeyError, IndexError, TypeError):
            value = None
        if torch.is_tensor(value):
            return value
    try:
        value = container[key]
    except (KeyError, IndexError, TypeError):
        return None
    if torch.is_tensor(value):
        return value
    return None


def _optional(container: Any, namespace: str, key: str) -> torch.Tensor | None:
    try:
        value = container[namespace][key]
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return value if torch.is_tensor(value) else None


class PotentialReportHook(ScalarHook):
    """Publish per-element force error, energy per atom, virial, and outliers.

    Predictions are read from ``outputs["predictions"]`` (the step's
    return). Targets are read from the batch. A missing quantity is
    skipped, so a force-only batch does not invent an energy. Padding
    masks (``atoms.mask``, ``graphs.mask``) drop ghost rows. The loss's
    inlier weight is not read.

    Train values are this batch, stored as device tensors. Eval values
    accumulate across the phase, are all-reduced when a process group
    exists, and are published as Python floats from
    ``on_eval_step_complete``.

    Args:
        elements: Atomic numbers with their own force MAE.
        outlier_force: Atoms whose force-error norm exceeds this count
            as outliers. The unit is the force column's unit.
        pred_namespace: Where the model writes ``forces`` / ``energy`` /
            ``virial`` when predictions are a nested batch. A flat
            predictions dict is read by leaf name.
    """

    def __init__(
        self,
        *,
        elements: Sequence[int] = (),
        outlier_force: float | None = None,
        pred_namespace: str = "graphs",
    ) -> None:
        self.elements = tuple(int(element) for element in elements)
        self.outlier_force = None if outlier_force is None else float(outlier_force)
        self.pred_namespace = pred_namespace
        self.train = PotentialReport(self.elements, self.outlier_force)
        self.val = PotentialReport(self.elements, self.outlier_force)

    @property
    def scalar_keys(self) -> tuple[Path, ...]:
        names = ["force_mae", "energy_per_atom_mae", "virial_mae"]
        names.extend(f"force_mae_Z{element}" for element in self.elements)
        if self.outlier_force is not None:
            names.extend(("force_outlier_count", "force_outlier_fraction"))
        return tuple((phase, name) for phase in ("train", "eval") for name in names)

    def _fill(self, report: PotentialReport, batch: Any, outputs: Mapping[str, Any]) -> None:
        predictions = outputs.get("predictions", outputs)
        forces_pred = _leaf(predictions, "atoms", "forces")
        forces_ref = _optional(batch, "atoms", "forces")
        mask = _optional(batch, "atoms", "mask")
        if forces_pred is not None and forces_ref is not None and forces_pred is not forces_ref:
            z = _optional(batch, "atoms", "Z")
            report.update_forces(forces_pred, forces_ref, z=z, mask=mask)
            report.update_outliers(forces_pred, forces_ref, mask=mask)
        energy_pred = _leaf(predictions, self.pred_namespace, "energy")
        energy_ref = _optional(batch, "graphs", "energy")
        num_atoms = _optional(batch, "graphs", "num_atoms")
        graph_mask = _optional(batch, "graphs", "mask")
        if (
            energy_pred is not None
            and energy_ref is not None
            and num_atoms is not None
            and energy_pred is not energy_ref
        ):
            report.update_energy(energy_pred, energy_ref, num_atoms, mask=graph_mask)
        virial_pred = _leaf(predictions, self.pred_namespace, "virial")
        virial_ref = _optional(batch, "graphs", "virial")
        if virial_pred is not None and virial_ref is not None and virial_pred is not virial_ref:
            report.update_virial(virial_pred, virial_ref, mask=graph_mask)

    def on_eval_phase_start(self, trainer, state) -> None:
        del trainer, state
        self.val.reset()

    def on_train_batch_end(self, trainer, state, batch, outputs) -> None:
        del trainer
        self.train.reset()
        self._fill(self.train, batch, outputs)
        namespace = state["train"]
        for name, value in self.train.scalars().items():
            namespace[name] = value

    def on_eval_batch_end(self, trainer, state, batch, outputs) -> None:
        del trainer, state
        self._fill(self.val, batch, outputs)

    def on_eval_step_complete(self, trainer, state) -> None:
        device = None
        try:
            device = next(trainer.model.parameters()).device
        except (AttributeError, StopIteration, RuntimeError):
            device = None
        self.val.sync(device)
        namespace = state["eval"]
        for name, value in self.val.scalars().items():
            namespace[name] = float(value)
