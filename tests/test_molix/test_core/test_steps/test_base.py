"""Tests for the eager step prologue :func:`molix.core.steps.base._mark_leaves`."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.autograd import Derivative, Potential
from molix.core.steps.base import _mark_leaves


class _Potential(Potential):
    """A force potential in the molnex shape, so the prologue has an op to mark."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self._derivative = Derivative(self.energy, wrt=("atoms.pos",))

    def energy(self, frame: TensorDict) -> TensorDict:
        frame["graphs", "energy"] = self.linear(frame["atoms", "pos"]).sum().reshape(1)
        return frame


class TestMarkLeaves:
    def test_reads_the_op_through_a_compile_wrapper(self) -> None:
        """``compile(step=False)`` wraps the model; the prologue still marks its leaves."""
        frame = TensorDict(
            atoms=TensorDict(pos=torch.zeros(3, 3), batch_size=[3]),
            graphs=TensorDict(batch_size=[1]),
            batch_size=[],
        )

        _mark_leaves(torch.compile(_Potential()), frame)

        pos = frame["atoms", "pos"]
        assert pos.is_leaf and pos.requires_grad

    def test_a_plain_module_is_a_no_op(self) -> None:
        _mark_leaves(nn.Linear(3, 1), {})
