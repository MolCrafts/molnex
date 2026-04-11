"""Atomic baseline subtraction task."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from molix.data.task import DatasetTask


class AtomicDress(DatasetTask):
    """Subtract element-dependent atomic baseline from a scalar target.

    Two-phase: :meth:`fit` computes per-element energies via least-squares
    on the full training set, then :meth:`execute` subtracts the baseline
    from each sample.
    """

    def __init__(
        self,
        elements: list[int] | tuple[int, ...],
        target_key: str = "U0",
        output_key: str = "U0",
    ) -> None:
        self.elements = tuple(elements)
        self.target_key = target_key
        self.output_key = output_key
        self.atomic_energies: dict[int, float] = {}

    @property
    def task_id(self) -> str:
        elems = ",".join(str(e) for e in sorted(self.elements))
        return f"dress:{self.target_key}->{self.output_key}:e={elems}"

    # -- DatasetTask contract -----------------------------------------------

    def fit(self, samples: list[dict]) -> None:
        x_rows: list[list[float]] = []
        y_vals: list[float] = []

        for sample in samples:
            z = sample["Z"]
            row = [float((z == elem).sum().item()) for elem in self.elements]
            x_rows.append(row)
            targets = sample.get("targets", {})
            if self.target_key not in targets:
                raise KeyError(f"Missing target '{self.target_key}'")
            y_vals.append(float(targets[self.target_key].reshape(-1)[0].item()))

        x = np.asarray(x_rows, dtype=np.float64)
        y = np.asarray(y_vals, dtype=np.float64)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)

        self.atomic_energies = {
            elem: float(beta[idx]) for idx, elem in enumerate(self.elements)
        }

    def execute(self, sample: dict) -> dict:
        if not self.atomic_energies:
            raise RuntimeError("AtomicDress.fit() must be called before execute()")

        targets = dict(sample.get("targets", {}))
        if self.target_key not in targets:
            raise KeyError(f"Missing target '{self.target_key}'")

        baseline = sum(
            self.atomic_energies.get(int(z.item()), 0.0) for z in sample["Z"]
        )
        value = targets[self.target_key].reshape(-1)[0]
        corrected = value - torch.tensor(baseline, dtype=value.dtype, device=value.device)
        targets[self.output_key] = corrected.reshape(1)

        return {**sample, "targets": targets}

    def state_dict(self) -> dict[str, Any]:
        elems = sorted(self.atomic_energies.keys())
        return {
            "elements": torch.tensor(elems, dtype=torch.long),
            "energies": torch.tensor(
                [self.atomic_energies[e] for e in elems], dtype=torch.float64
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        elems = state["elements"].tolist()
        energies = state["energies"].tolist()
        self.atomic_energies = dict(zip(elems, energies))
