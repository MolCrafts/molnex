"""Unit tests for ``molpot.composition.energy_force.EnergyForceModel``.

Backends are exercised separately — functorch paths never call autograd
helpers and vice versa.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.composition.energy_force import EnergyForceModel


class _ToyEF(EnergyForceModel):
    """Minimal energy: ``E = s · Σ ‖pos‖²`` (analytic ``F = -2 s pos``).

    Counts ``energy_forward`` so the 1-pass contract is enforceable without
    mocking internals.
    """

    def __init__(self, *, force_method: str = "functorch", compute_forces: bool = True) -> None:
        super().__init__(force_method=force_method, compute_forces=compute_forces)  # type: ignore[arg-type]
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.n_energy_calls = 0

    def energy_forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        self.n_energy_calls += 1
        pos = batch["atoms", "pos"]
        atom_batch = batch["atoms", "batch"]
        atom_e = self.scale * (pos * pos).sum(dim=-1)
        n_graphs = int(batch["graphs"].batch_size[0])
        energy = torch.zeros(n_graphs, dtype=pos.dtype, device=pos.device)
        energy.scatter_add_(0, atom_batch, atom_e)
        return {"energy": energy, "atomic_energy": atom_e}


def _batch(n_atoms: int = 4, n_graphs: int = 2) -> TensorDict:
    assert n_atoms % n_graphs == 0
    per = n_atoms // n_graphs
    pos = torch.arange(n_atoms * 3, dtype=torch.float32).reshape(n_atoms, 3) * 0.1
    return TensorDict(
        atoms=TensorDict(
            pos=pos,
            batch=torch.arange(n_graphs).repeat_interleave(per),
            batch_size=[n_atoms],
        ),
        graphs=TensorDict(batch_size=[n_graphs]),
        batch_size=[],
    )


class TestFunctorchBackend:
    """``force_method='functorch'`` — has_aux only, never autograd helpers."""

    def test_train_is_single_energy_pass(self):
        model = _ToyEF(force_method="functorch")
        model.train()
        model.n_energy_calls = 0
        out = model(_batch(), compute_forces=True)
        assert model.n_energy_calls == 1
        assert out["forces"].shape == (4, 3)

    def test_eval_is_single_energy_pass(self):
        model = _ToyEF(force_method="functorch")
        model.eval()
        model.n_energy_calls = 0
        model(_batch(), compute_forces=True)
        assert model.n_energy_calls == 1

    def test_forces_match_analytic(self):
        model = _ToyEF(force_method="functorch")
        model.train()
        batch = _batch()
        out = model(batch.clone(), compute_forces=True)
        expected = -2.0 * model.scale.detach() * batch["atoms", "pos"]
        assert torch.allclose(out["forces"], expected, atol=1e-6)

    def test_force_loss_reaches_parameters(self):
        model = _ToyEF(force_method="functorch")
        model.train()
        out = model(_batch().clone(), compute_forces=True)
        out["forces"].pow(2).mean().backward()
        assert model.scale.grad is not None
        assert model.scale.grad.abs().item() > 0

    def test_forces_from_energy_rejected(self):
        model = _ToyEF(force_method="functorch")
        pos = torch.zeros(2, 3, requires_grad=True)
        energy = pos.pow(2).sum()
        try:
            model.force_derivation.forces_from_energy(energy, pos)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "autograd-only" in str(e)


class TestAutogradBackend:
    """``force_method='autograd'`` — energy once + forces_from_energy only."""

    def test_train_is_single_energy_pass(self):
        model = _ToyEF(force_method="autograd")
        model.train()
        model.n_energy_calls = 0
        model(_batch(), compute_forces=True)
        assert model.n_energy_calls == 1

    def test_eval_is_single_energy_pass(self):
        model = _ToyEF(force_method="autograd")
        model.eval()
        model.n_energy_calls = 0
        model(_batch(), compute_forces=True)
        assert model.n_energy_calls == 1

    def test_forces_match_analytic(self):
        model = _ToyEF(force_method="autograd")
        model.train()
        batch = _batch()
        out = model(batch.clone(), compute_forces=True)
        expected = -2.0 * model.scale.detach() * batch["atoms", "pos"]
        assert torch.allclose(out["forces"], expected, atol=1e-6)

    def test_force_loss_reaches_parameters(self):
        model = _ToyEF(force_method="autograd")
        model.train()
        out = model(_batch().clone(), compute_forces=True)
        out["forces"].pow(2).mean().backward()
        assert model.scale.grad is not None
        assert model.scale.grad.abs().item() > 0

    def test_has_aux_rejected(self):
        model = _ToyEF(force_method="autograd")

        def energy_fn_aux(p: torch.Tensor):
            return p.pow(2).sum(), {"x": p}

        try:
            model.force_derivation(energy_fn_aux, torch.zeros(2, 3), has_aux=True)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "functorch-only" in str(e)
