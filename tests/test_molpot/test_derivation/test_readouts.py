"""In-place EnergyReadout / ForceReadout (no user-facing Derivative)."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.derivation import EnergyReadout, ForceReadout
from molpot.derivation.protocol import ENERGY_KEY, FORCES_KEY, write_energy


class _ToyPotential(nn.Module):
    """E = s · Σ ‖pos‖² written onto the batch; counts forward calls."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.n_forward = 0

    def forward(self, batch: TensorDict) -> TensorDict:
        self.n_forward += 1
        pos = batch["atoms", "pos"]
        atom_batch = batch["atoms", "batch"]
        atom_e = self.scale * (pos * pos).sum(dim=-1)
        n_graphs = int(batch["graphs"].batch_size[0])
        energy = torch.zeros(n_graphs, dtype=pos.dtype, device=pos.device)
        energy.scatter_add_(0, atom_batch, atom_e)
        write_energy(batch, energy, atomic_energy=atom_e)
        return batch


def _batch(n_atoms: int = 4, n_graphs: int = 2) -> TensorDict:
    per = n_atoms // n_graphs
    pos = torch.arange(n_atoms * 3, dtype=torch.float32).reshape(n_atoms, 3) * 0.1
    return TensorDict(
        atoms=TensorDict(
            pos=pos.clone(),
            batch=torch.arange(n_graphs).repeat_interleave(per),
            batch_size=[n_atoms],
        ),
        graphs=TensorDict(batch_size=[n_graphs]),
        batch_size=[],
    )


def _model(*, train: bool = True) -> _ToyPotential:
    m = _ToyPotential()
    m.train(train)
    return m


class TestGradMode:
    def test_energy_then_force_is_single_forward(self):
        model = _model()
        batch = _batch()
        batch = EnergyReadout(model, method="grad", backward=True)(batch)
        batch = ForceReadout(model, method="grad")(batch)
        assert model.n_forward == 1
        assert batch[FORCES_KEY].shape == (4, 3)
        assert batch[ENERGY_KEY].shape == (2,)

    def test_forces_match_analytic(self):
        model = _model()
        batch = _batch()
        pos0 = batch["atoms", "pos"].clone()
        batch = EnergyReadout(model, method="grad", backward=True)(batch)
        batch = ForceReadout(model, method="grad")(batch)
        expected = -2.0 * model.scale.detach() * pos0
        assert torch.allclose(batch[FORCES_KEY], expected, atol=1e-5)

    def test_force_loss_reaches_parameters(self):
        model = _model(train=True)
        batch = _batch()
        batch = EnergyReadout(model, method="grad", backward=True)(batch)
        batch = ForceReadout(model, method="grad")(batch)
        batch[FORCES_KEY].pow(2).mean().backward()
        assert model.scale.grad is not None
        assert model.scale.grad.abs().item() > 0

    def test_energy_only(self):
        model = _model()
        batch = EnergyReadout(model, method="grad", backward=False)(_batch())
        assert model.n_forward == 1
        assert ENERGY_KEY[1] in batch["graphs"].keys()


class TestFuncMode:
    def test_energy_then_force_is_single_forward(self):
        model = _model()
        batch = _batch()
        batch = EnergyReadout(model, method="func", backward=True)(batch)
        assert model.n_forward == 0  # lazy
        batch = ForceReadout(model, method="func")(batch)
        assert model.n_forward == 1
        assert batch[FORCES_KEY].shape == (4, 3)

    def test_forces_match_analytic(self):
        model = _model()
        batch = _batch()
        pos0 = batch["atoms", "pos"].clone()
        batch = EnergyReadout(model, method="func", backward=True)(batch)
        batch = ForceReadout(model, method="func")(batch)
        expected = -2.0 * model.scale.detach() * pos0
        assert torch.allclose(batch[FORCES_KEY], expected, atol=1e-5)

    def test_force_loss_reaches_parameters(self):
        model = _model(train=True)
        batch = _batch()
        batch = EnergyReadout(model, method="func", backward=True)(batch)
        batch = ForceReadout(model, method="func")(batch)
        batch[FORCES_KEY].pow(2).mean().backward()
        assert model.scale.grad is not None
        assert model.scale.grad.abs().item() > 0

    def test_energy_only(self):
        model = _model()
        batch = EnergyReadout(model, method="func", backward=False)(_batch())
        assert model.n_forward == 1
        assert batch[ENERGY_KEY].shape == (2,)

    def test_force_alone_is_single_forward(self):
        model = _model()
        batch = ForceReadout(model, method="func")(_batch())
        assert model.n_forward == 1
        assert FORCES_KEY[1] in batch["atoms"].keys()
