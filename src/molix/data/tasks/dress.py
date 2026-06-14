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
        """Cache-key identity ``dress:<target>-><output>:e=<sorted elements>``."""
        elems = ",".join(str(e) for e in sorted(self.elements))
        return f"dress:{self.target_key}->{self.output_key}:e={elems}"

    # -- DatasetTask contract -----------------------------------------------

    def fit(self, samples: list[dict]) -> None:
        """Fit per-element baseline energies by least squares over *samples*.

        Builds a count matrix (one row per sample, one column per element
        in :attr:`elements`, entries = number of atoms of that element)
        and solves ``counts @ beta ≈ target`` via
        :func:`numpy.linalg.lstsq`. The solution ``beta`` is stored as
        :attr:`atomic_energies` (``{element: energy}``).

        Args:
            samples: Training sample dicts; each must hold ``Z`` (atomic
                numbers) and ``targets[target_key]``.

        Raises:
            KeyError: A sample is missing ``target_key`` in its targets.
        """
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

        self.atomic_energies = {elem: float(beta[idx]) for idx, elem in enumerate(self.elements)}

    def execute(self, data: dict) -> dict:
        """Subtract the fitted atomic baseline from one sample's target.

        Sums the per-element baseline over the sample's atoms and writes
        ``target - baseline`` to ``targets[output_key]`` as a ``(1,)``
        tensor (matching the input dtype/device).

        Args:
            data: A sample dict with ``Z`` and ``targets[target_key]``.

        Returns:
            A new sample dict with the baseline-subtracted target.

        Raises:
            RuntimeError: Called before :meth:`fit` (no fitted energies).
            KeyError: ``target_key`` is absent from the sample's targets.
        """
        if not self.atomic_energies:
            raise RuntimeError("AtomicDress.fit() must be called before execute()")

        targets = dict(data.get("targets", {}))
        if self.target_key not in targets:
            raise KeyError(f"Missing target '{self.target_key}'")

        baseline = sum(self.atomic_energies.get(int(z.item()), 0.0) for z in data["Z"])
        value = targets[self.target_key].reshape(-1)[0]
        corrected = value - torch.tensor(baseline, dtype=value.dtype, device=value.device)
        targets[self.output_key] = corrected.reshape(1)

        return {**data, "targets": targets}

    def state_dict(self) -> dict[str, Any]:
        """Serialise the fitted baseline for caching.

        Returns:
            ``{"elements": LongTensor(n_elements),
            "energies": Float64Tensor(n_elements)}`` with elements sorted
            ascending and aligned to their fitted energies.
        """
        elems = sorted(self.atomic_energies.keys())
        return {
            "elements": torch.tensor(elems, dtype=torch.long),
            "energies": torch.tensor([self.atomic_energies[e] for e in elems], dtype=torch.float64),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the fitted baseline produced by :meth:`state_dict`.

        Args:
            state: Mapping with aligned ``"elements"`` and ``"energies"``
                tensors; rebuilds :attr:`atomic_energies`.
        """
        elems = state["elements"].tolist()
        energies = state["energies"].tolist()
        self.atomic_energies = dict(zip(elems, energies))
