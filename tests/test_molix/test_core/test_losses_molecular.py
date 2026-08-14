"""Unit tests for molecular loss presets in :mod:`molix.core.losses.molecular`."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.config import config
from molix.core.losses import WeightedLoss, energy_force_mse, energy_mse
from molix.core.losses.molecular import (
    ChargeMSELoss,
    ProjectedInternalHessianLoss,
    ScaleNormalizedLoss,
    TorsionProfileLoss,
    center_by_group,
    molecule_centered_energy_mse,
    parameter_bag_mse,
)
from molix.data.collate import collate_molecules


def _sample(Z, pos, *, U0=None, forces=None):
    s = {
        "Z": torch.as_tensor(Z, dtype=torch.long),
        "pos": torch.as_tensor(pos, dtype=torch.float32),
        "edge_index": torch.zeros(0, 2, dtype=torch.long),
        "edge_diff": torch.zeros(0, 3),
        "edge_dist": torch.zeros(0),
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


class TestCenterByGroup:
    def test_exact_centering(self):
        values = torch.tensor([1.0, 3.0, 10.0, 14.0])
        groups = torch.tensor([0, 0, 1, 1])
        c = center_by_group(values, groups)
        assert torch.allclose(c, torch.tensor([-1.0, 1.0, -2.0, 2.0]))


class TestMoleculeCenteredEnergyMse:
    def test_zero_when_identical_after_center(self):
        pred = torch.tensor([1.0, 2.0, 5.0, 7.0])
        true = torch.tensor([10.0, 11.0, 0.0, 2.0])
        groups = torch.tensor([0, 0, 1, 1])
        batch = TensorDict(
            {
                "graphs": TensorDict(
                    {
                        "mm_energy": true,
                        "molecule_id_index": groups,
                    },
                    batch_size=[4],
                )
            },
            batch_size=[],
        )
        loss = molecule_centered_energy_mse()({"energy": pred}, batch)
        assert float(loss) < 1e-10


class TestParameterBagMse:
    def test_zero_and_positive(self):
        a = {"k": torch.tensor([1.0, 2.0])}
        assert float(parameter_bag_mse(a, a)) == 0.0
        b = {"k": torch.tensor([1.0, 4.0])}
        assert float(parameter_bag_mse(a, b)) == ((0.0 + 4.0) / 2.0)


# Analytic Wilson GF goldens (Wilson 1941): 2-atom bond along x, k=2 kcal/mol/Å².
# H_x = k B^T B; H_q = (B^+)^T H_x B^+ = [[2.0]]. Hard-coded 2026-08-13.
_WILSON_B = torch.tensor([[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
_WILSON_H_X = torch.tensor(
    [
        [2.0, 0.0, 0.0, -2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
_H_Q_ATOL = 1e-6
_ENERGY_ATOL = 1e-10


class _FixedLoss(torch.nn.Module):
    """Constant scalar stand-in for wrapping in ScaleNormalizedLoss."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = float(value)

    def forward(self, pred: object, target: object) -> torch.Tensor:
        return torch.tensor(self._value)


class TestProjectedInternalHessianLoss:
    def test_project_hardcoded_wilson_hq_is_two(self):
        h_q = ProjectedInternalHessianLoss.project(_WILSON_H_X, _WILSON_B)
        assert torch.allclose(h_q, torch.tensor([[2.0]]), atol=_H_Q_ATOL)

    def test_diagonal_mse_vs_k_two_is_zero(self):
        loss_fn = ProjectedInternalHessianLoss(target_kind="diagonal")
        pred = {"hessian": _WILSON_H_X}
        target = {"B": _WILSON_B, "diagonal": torch.tensor([2.0])}
        got = loss_fn(pred, target)
        assert torch.allclose(got, torch.tensor(0.0), atol=_ENERGY_ATOL)

    def test_diagonal_mse_vs_k_zero_is_four(self):
        loss_fn = ProjectedInternalHessianLoss(target_kind="diagonal")
        pred = {"hessian": _WILSON_H_X}
        target = {"B": _WILSON_B, "diagonal": torch.tensor([0.0])}
        got = loss_fn(pred, target)
        assert torch.allclose(got, torch.tensor(4.0), atol=_ENERGY_ATOL)

    def test_hq_ref_mse_against_projected_is_zero(self):
        loss_fn = ProjectedInternalHessianLoss(target_kind="hq_ref")
        pred = {"hessian": _WILSON_H_X}
        target = {"B": _WILSON_B, "hq_ref": torch.tensor([[2.0]])}
        got = loss_fn(pred, target)
        assert torch.allclose(got, torch.tensor(0.0), atol=_ENERGY_ATOL)

    def test_nonsquare_hessian_raises(self):
        with pytest.raises(ValueError):
            ProjectedInternalHessianLoss.project(torch.zeros(6, 5), _WILSON_B)

    def test_b_columns_not_matching_3n_raises(self):
        with pytest.raises(ValueError):
            ProjectedInternalHessianLoss.project(_WILSON_H_X, torch.zeros(1, 5))

    def test_hx_yy_noise_leaves_hq_unchanged(self):
        noisy = _WILSON_H_X.clone()
        noisy[1, 1] = noisy[1, 1] + 10.0
        h_q = ProjectedInternalHessianLoss.project(_WILSON_H_X, _WILSON_B)
        h_q_noisy = ProjectedInternalHessianLoss.project(noisy, _WILSON_B)
        assert torch.allclose(h_q, h_q_noisy, atol=_H_Q_ATOL)
        assert torch.allclose(h_q_noisy, torch.tensor([[2.0]]), atol=_H_Q_ATOL)

    def test_diagonal_kind_does_not_penalize_off_diagonals(self):
        b = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        h_q_full = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        h_x = b.T @ h_q_full @ b
        loss_fn = ProjectedInternalHessianLoss(target_kind="diagonal")
        got = loss_fn({"hessian": h_x}, {"B": b, "diagonal": torch.tensor([2.0, 3.0])})
        assert torch.allclose(got, torch.tensor(0.0), atol=_ENERGY_ATOL)

    def test_wilson_identity_hq_equals_k(self):
        k = 2.0
        b = _WILSON_B
        h_x = k * (b.T @ b)
        h_q = ProjectedInternalHessianLoss.project(h_x, b)
        assert torch.allclose(h_q, torch.tensor([[k]]), atol=_H_Q_ATOL)


class TestTorsionProfileLoss:
    def test_centered_profile_mse_is_zero(self):
        pred = {"energy": torch.tensor([1.0, 2.0, 5.0, 7.0])}
        target = {
            "energy": torch.tensor([10.0, 11.0, 0.0, 2.0]),
            "molecule_id_index": torch.tensor([0, 0, 1, 1]),
        }
        got = TorsionProfileLoss()(pred, target)
        assert torch.allclose(got, torch.tensor(0.0), atol=_ENERGY_ATOL)

    def test_zero_barrier_weight_profile_mse_is_one(self):
        pred = {"energy": torch.tensor([0.0, 4.0])}
        target = {
            "energy": torch.tensor([0.0, 2.0]),
            "molecule_id_index": torch.tensor([0, 0]),
        }
        got = TorsionProfileLoss(barrier_weight=0.0)(pred, target)
        assert torch.allclose(got, torch.tensor(1.0), atol=_ENERGY_ATOL)

    def test_barrier_weight_half_gives_three(self):
        pred = {"energy": torch.tensor([0.0, 4.0])}
        target = {
            "energy": torch.tensor([0.0, 2.0]),
            "molecule_id_index": torch.tensor([0, 0]),
        }
        got = TorsionProfileLoss(barrier_weight=0.5)(pred, target)
        assert torch.allclose(got, torch.tensor(3.0), atol=_ENERGY_ATOL)

    def test_singleton_group_barrier_is_zero(self):
        pred = {"energy": torch.tensor([3.0])}
        target = {
            "energy": torch.tensor([9.0]),
            "molecule_id_index": torch.tensor([0]),
        }
        got = TorsionProfileLoss(barrier_weight=1.0)(pred, target)
        assert torch.allclose(got, torch.tensor(0.0), atol=_ENERGY_ATOL)

    def test_empty_group_barrier_is_zero(self):
        pred = {"energy": torch.tensor([1.0, 2.0, 5.0, 7.0])}
        target = {
            "energy": torch.tensor([10.0, 11.0, 0.0, 2.0]),
            "molecule_id_index": torch.tensor([0, 0, 2, 2]),
        }
        got = TorsionProfileLoss(barrier_weight=1.0)(pred, target)
        assert torch.allclose(got, torch.tensor(0.0), atol=_ENERGY_ATOL)

    def test_constant_energy_shift_is_invariant(self):
        pred = {
            "energy": torch.tensor([1.0, 2.0, 5.0, 7.0])
            + torch.tensor([100.0, 100.0, -50.0, -50.0])
        }
        target = {
            "energy": torch.tensor([10.0, 11.0, 0.0, 2.0]) + torch.tensor([-3.0, -3.0, 8.0, 8.0]),
            "molecule_id_index": torch.tensor([0, 0, 1, 1]),
        }
        got = TorsionProfileLoss()(pred, target)
        assert torch.allclose(got, torch.tensor(0.0), atol=_ENERGY_ATOL)


class TestChargeMSELoss:
    def test_mse_is_one_hundredth(self):
        pred = {"q": torch.tensor([0.2, -0.2])}
        target = {"q": torch.tensor([0.1, -0.1])}
        got = ChargeMSELoss()(pred, target)
        assert torch.allclose(got, torch.tensor(0.01), atol=_ENERGY_ATOL)

    def test_forward_does_not_require_energy_keys(self):
        pred = {"q": torch.tensor([0.2, -0.2])}
        target = {"q": torch.tensor([0.1, -0.1])}
        got = ChargeMSELoss()(pred, target)
        assert "energy" not in pred
        assert "energy" not in target
        assert torch.allclose(got, torch.tensor(0.01), atol=_ENERGY_ATOL)


class TestScaleNormalizedLoss:
    def test_inner_four_scale_two_is_one(self):
        loss_fn = ScaleNormalizedLoss(_FixedLoss(4.0), scale=2.0)
        got = loss_fn({}, {})
        assert torch.allclose(got, torch.tensor(1.0), atol=_ENERGY_ATOL)

    def test_scale_is_buffer_not_parameter(self):
        loss_fn = ScaleNormalizedLoss(_FixedLoss(4.0), scale=2.0)
        buffers = dict(loss_fn.named_buffers())
        assert "scale" in buffers
        assert buffers["scale"].dtype == config.ftype
        assert "scale" not in {name for name, _ in loss_fn.named_parameters()}

    def test_zero_scale_raises(self):
        with pytest.raises(ValueError):
            ScaleNormalizedLoss(_FixedLoss(4.0), scale=0.0)

    def test_weighted_loss_combo_on_charge_golden(self):
        loss_fn = WeightedLoss([(1.0, ScaleNormalizedLoss(ChargeMSELoss(), scale=2.0))])
        pred = {"q": torch.tensor([0.2, -0.2])}
        target = {"q": torch.tensor([0.1, -0.1])}
        got = loss_fn(pred, target)
        assert torch.allclose(got, torch.tensor(0.0025), atol=_ENERGY_ATOL)
