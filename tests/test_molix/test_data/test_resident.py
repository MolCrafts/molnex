"""ResidentValidation — mirrors ``src/molix/data/resident.py``."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molix.data import ResidentValidation


class _Counting:
    def __init__(self) -> None:
        self.val_calls = 0

    def train_dataloader(self):
        return iter([TensorDict({"x": torch.zeros(1)}, batch_size=[1])])

    def val_dataloader(self):
        self.val_calls += 1
        return iter(
            [TensorDict({"x": torch.full((1,), float(i))}, batch_size=[1]) for i in range(3)]
        )


class TestResidentValidation:
    def test_validation_is_built_once_and_replayed(self) -> None:
        inner = _Counting()
        resident = ResidentValidation(inner, "cpu")
        first = [float(b["x"]) for b in resident.val_dataloader()]
        second = [float(b["x"]) for b in resident.val_dataloader()]
        assert first == second == [0.0, 1.0, 2.0]
        assert inner.val_calls == 1
