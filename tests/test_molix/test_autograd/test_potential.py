"""Tests for :class:`molix.autograd.potential.Potential` — the two-method contract on a stub."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.autograd import Derivative, Potential
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, FORCES_KEY

FORCES = ("atoms.pos",)


class _Squares(Potential):
    """``E_g = s Σ_{i∈g} ‖r_i‖²``: writes both energy columns; one parameter ``s``."""

    def __init__(self, *, wrt: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), dtype=torch.float64))
        self._derivative = Derivative(self.energy, wrt=wrt)

    def energy(self, frame: TensorDict) -> TensorDict:
        pos = frame["atoms", "pos"]
        atomic = self.scale * (pos * pos).sum(dim=-1)
        total = torch.zeros(frame["graphs"].batch_size[0], dtype=pos.dtype)
        frame[ATOMIC_ENERGY_KEY] = atomic
        frame[ENERGY_KEY] = total.index_add(0, frame["atoms", "batch"], atomic)
        return frame


def _frame() -> TensorDict:
    pos = torch.arange(12, dtype=torch.float64).reshape(4, 3) * 0.1
    return TensorDict(
        atoms=TensorDict(pos=pos, batch=torch.tensor([0, 0, 1, 1]), batch_size=[4]),
        graphs=TensorDict(batch_size=[2]),
        batch_size=[],
    )


class TestPotential:
    def test_forward_is_the_derivative_op(self) -> None:
        pot = _Squares(wrt=FORCES)
        frame = _frame()
        out = pot(frame)
        assert out is frame
        torch.testing.assert_close(out[FORCES_KEY], (-2.0 * frame["atoms", "pos"]).detach())
        torch.testing.assert_close(pot.derivative(_frame())[FORCES_KEY], out[FORCES_KEY])

    def test_empty_wrt_runs_the_energy_only(self) -> None:
        out = _Squares()(_frame())
        assert FORCES_KEY not in out.keys(include_nested=True)
        torch.testing.assert_close(out[ENERGY_KEY], torch.tensor([0.55, 4.51], dtype=torch.float64))

    def test_derivative_reads_back_the_variables(self) -> None:
        assert _Squares(wrt=FORCES).derivative.wrt == (("atoms", "pos"),)

    def test_state_dict_is_flat_with_no_derivative_keys(self) -> None:
        assert list(_Squares(wrt=FORCES).state_dict()) == ["scale"]
