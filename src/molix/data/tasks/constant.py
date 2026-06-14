"""Inject a constant scalar label into ``sample["targets"][key]``.

For datasets where a downstream model requires a per-graph scalar that
the source does not natively provide. The canonical use case is feeding
a uniformly-neutral molecular dataset (e.g. QM9) into a model that
requires a ``total_charge`` target — the model fail-fasts on the missing
key, and this task is the one-line opt-in that says "yes, this dataset
is uniformly ``value``".

Example::

    Pipeline("qm9-multipole")
        .add(UnitConvert({"U0": ("hartree", "eV")}))
        .add(ConstantLabel(key="total_charge", value=0.0))
        .add(NeighborList(cutoff=5.0))
        .build()
"""

from __future__ import annotations

import torch

from molix.data.task import SampleTask


class ConstantLabel(SampleTask):
    """Write a constant scalar to ``sample["targets"][key]``.

    Args:
        key: Target name. After collate the value lives at
            ``batch["graphs", key]`` as a ``(B,)`` tensor.
        value: The constant scalar value (one per graph).

    Notes:
        Runs once at cache time, so the constant is baked into the packed
        cache and the runtime hot path is zero-overhead.
    """

    def __init__(self, key: str, value: float) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("ConstantLabel: 'key' must be a non-empty string.")
        self.key = key
        self.value = float(value)

    @property
    def task_id(self) -> str:
        """Cache-key identity ``const_label:<key>=<value>``."""
        return f"const_label:{self.key}={self.value!r}"

    def execute(self, data: dict) -> dict:
        """Write the constant under ``sample["targets"][key]``.

        Args:
            data: A raw sample dict; its ``targets`` sub-dict is copied
                (not mutated in place).

        Returns:
            A new sample dict with ``targets[key]`` set to a ``(1,)``
            tensor holding the configured value (default dtype).
        """
        targets = dict(data.get("targets", {}))
        targets[self.key] = torch.tensor([self.value], dtype=torch.get_default_dtype())
        return {**data, "targets": targets}
