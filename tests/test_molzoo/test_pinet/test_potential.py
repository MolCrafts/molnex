"""Unit tests for molzoo.pinet.potential.PiNetPotential."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.compile import Compiler
from molzoo.pinet import PiNetPotential, PiNetSpec
from tests.conftest import make_graph_batch
from tests.test_molzoo.test_pinet.conftest import PAIRS, TINY

FORCES = ("atoms.pos",)


def _model(*, wrt=FORCES, seed: int = 0, **overrides) -> PiNetPotential:
    torch.manual_seed(seed)
    return PiNetPotential(PiNetSpec(**{**TINY, "rank": 3, **overrides}), wrt=wrt)


def _moved(batch: TensorDict, pos: torch.Tensor) -> TensorDict:
    """``batch`` with its pair geometry rebuilt from ``pos``."""
    return make_graph_batch(
        pos, batch["atoms", "Z"], batch["pairs", "pair_index"], batch["atoms", "batch"]
    )


class TestPiNetPotential:
    def test_writes_energy_atomic_energy_and_forces(self, cluster):
        out = _model()(cluster)
        assert out["graphs", "energy"].shape == (1,)
        assert out["atoms", "energy"].shape == (4,)
        assert out["atoms", "forces"].shape == (4, 3)
        torch.testing.assert_close(out["atoms", "energy"].sum(), out["graphs", "energy"][0])

    def test_state_dict_is_flat(self):
        assert not any(key.startswith("encoder.") for key in _model().state_dict())

    def test_forces_match_central_differences(self, cluster):
        model = _model(wrt=())
        forces = _model()(cluster.clone())["atoms", "forces"]
        pos = cluster["atoms", "pos"]
        h = 1e-5
        atoms = (0, 3)
        fd = torch.zeros(len(atoms), 3, dtype=pos.dtype)
        with torch.no_grad():
            for row, i in enumerate(atoms):
                for k in range(3):
                    plus, minus = pos.clone(), pos.clone()
                    plus[i, k] += h
                    minus[i, k] -= h
                    fd[row, k] = -(
                        model.energy(_moved(cluster, plus))["graphs", "energy"]
                        - model.energy(_moved(cluster, minus))["graphs", "energy"]
                    ).sum() / (2 * h)
        torch.testing.assert_close(forces[list(atoms)], fd, atol=1e-7, rtol=1e-6)

    def test_atomic_energies_add_the_per_element_dress(self, cluster):
        dress = [-0.5, -37.8, -54.6, -75.1]
        bare = _model(wrt=())(cluster.clone())["graphs", "energy"]
        dressed = _model(wrt=(), atomic_energies=dress)(cluster.clone())["graphs", "energy"]
        torch.testing.assert_close(dressed - bare, torch.tensor([sum(dress)], dtype=torch.float64))

    def test_scale_multiplies_the_network_energy(self, cluster):
        one = _model(wrt=())(cluster.clone())["graphs", "energy"]
        three = _model(wrt=(), scale=3.0)(cluster.clone())["graphs", "energy"]
        torch.testing.assert_close(three, 3.0 * one)

    def test_out_activation_defaults_to_pinn_linear(self):
        assert not any(isinstance(m, torch.nn.Tanh) for m in _model().out_layers.modules())

    def test_a_padded_batch_returns_the_unpadded_energy(self, cluster):
        model = _model(wrt=())
        ref = model(cluster.clone())["graphs", "energy"]
        pos = torch.cat([cluster["atoms", "pos"], torch.full((2, 3), 50.0, dtype=torch.float64)])
        padded = make_graph_batch(
            pos,
            torch.tensor([1, 6, 7, 8, 1, 1]),
            torch.tensor(PAIRS),
            torch.zeros(6, dtype=torch.long),
        )
        padded["atoms", "mask"] = torch.tensor([True] * 4 + [False] * 2)
        torch.testing.assert_close(model(padded)["graphs", "energy"], ref)

    def test_force_loss_reaches_the_parameters(self, cluster):
        model = _model()
        model(cluster)["atoms", "forces"].pow(2).mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    @pytest.mark.slow
    def test_forward_compiles_to_one_graph_in_inference(self, cluster):
        model = _model()
        model(cluster.clone())  # spend the eager element gate
        with torch.no_grad():
            explanation = Compiler.explain(model, cluster)
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            explanation.break_sites
        )
