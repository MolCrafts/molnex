"""Unit tests for molecular loss presets in :mod:`molix.core.losses.molecular`."""

from __future__ import annotations

import torch

from molix.core.losses import energy_force_mse, energy_mse
from molix.data.collate import collate_molecules


def _sample(Z, pos, *, U0=None, forces=None):
    s = {
        "Z": torch.as_tensor(Z, dtype=torch.long),
        "pos": torch.as_tensor(pos, dtype=torch.float32),
        "edge_index": torch.zeros(0, 2, dtype=torch.long),
        "bond_diff": torch.zeros(0, 3),
        "bond_dist": torch.zeros(0),
        "targets": {},
    }
    if U0 is not None:
        s["targets"]["U0"] = torch.tensor([float(U0)])
    if forces is not None:
        s["targets"]["forces"] = torch.as_tensor(forces, dtype=torch.float32)
    return s


class TestEnergyMSE:
    def test_matches_handwritten_mse_on_graph_energy(self):
        samples = [
            _sample([1, 6], [[0, 0, 0], [1, 0, 0]], U0=1.0),
            _sample([6, 8], [[0, 0, 0], [1.2, 0, 0]], U0=2.5),
        ]
        batch = collate_molecules(samples)
        preds = {"energy": torch.tensor([0.8, 2.7])}

        loss_fn = energy_mse("U0")
        got = loss_fn(preds, batch)

        expected = torch.nn.functional.mse_loss(
            preds["energy"], batch["graphs", "U0"].view_as(preds["energy"])
        )
        assert torch.allclose(got, expected)

    def test_reduction_none_returns_per_sample(self):
        samples = [
            _sample([1], [[0, 0, 0]], U0=1.0),
            _sample([6], [[0, 0, 0]], U0=3.0),
        ]
        batch = collate_molecules(samples)
        preds = {"energy": torch.tensor([1.5, 2.5])}

        loss_fn = energy_mse("U0", reduction="none")
        got = loss_fn(preds, batch)
        assert got.shape == preds["energy"].shape
        assert torch.allclose(got, torch.tensor([0.25, 0.25]))

    def test_custom_pred_key(self):
        batch = collate_molecules([_sample([1], [[0, 0, 0]], U0=2.0)])
        preds = {"E_pred": torch.tensor([1.0])}
        loss_fn = energy_mse("U0", pred_key="E_pred")
        assert torch.allclose(loss_fn(preds, batch), torch.tensor(1.0))


class TestEnergyForceMSE:
    def test_combines_energy_and_force_terms(self):
        forces_a = torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        forces_b = torch.tensor([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])
        samples = [
            _sample([1, 6], [[0, 0, 0], [1, 0, 0]], U0=1.0, forces=forces_a),
            _sample([6, 8], [[0, 0, 0], [1.2, 0, 0]], U0=2.0, forces=forces_b),
        ]
        from molix.data.collate import TargetSchema

        schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset({"forces"}))
        batch = collate_molecules(samples, schema)

        preds = {
            "energy": torch.tensor([0.5, 2.5]),
            "forces": torch.zeros_like(batch["atoms", "forces"]),
        }

        loss_fn = energy_force_mse(
            energy_target_key="U0",
            force_target_key="forces",
            lambda_F=0.5,
        )
        got = loss_fn(preds, batch)

        e_true = batch["graphs", "U0"].view_as(preds["energy"])
        f_true = batch["atoms", "forces"]
        expected = torch.nn.functional.mse_loss(
            preds["energy"], e_true
        ) + 0.5 * torch.nn.functional.mse_loss(preds["forces"], f_true)
        assert torch.allclose(got, expected)

    def test_lambda_zero_matches_energy_only(self):
        forces = torch.zeros(1, 3)
        samples = [_sample([1], [[0, 0, 0]], U0=1.0, forces=forces)]
        from molix.data.collate import TargetSchema

        schema = TargetSchema(graph_level=frozenset({"U0"}), atom_level=frozenset({"forces"}))
        batch = collate_molecules(samples, schema)

        preds = {
            "energy": torch.tensor([2.0]),
            "forces": torch.ones(1, 3),
        }
        joint = energy_force_mse(
            energy_target_key="U0",
            force_target_key="forces",
            lambda_F=0.0,
        )(preds, batch)
        energy_only = energy_mse("U0")(preds, batch)
        assert torch.allclose(joint, energy_only)
