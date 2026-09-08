"""Unit tests for molecular loss presets in :mod:`molix.core.losses.molecular`."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.batch.collate import collate_molecules
from molix.config import config
from molix.core.losses import WeightedLoss, energy_force_mse, energy_mse
from molix.core.losses.molecular import (
    ChargeMSELoss,
    ParsleyEnergyWeight,
    ProjectedInternalHessianLoss,
    ScaleNormalizedLoss,
    TorsionProfileLoss,
    center_by_group,
    molecule_centered_energy_mse,
    parameter_bag_mse,
)
from tests.conftest import frame


def _sample(Z, pos, *, U0=None, forces=None):
    """One molpy ``Frame``; ``U0`` becomes a graph leaf, forces a per-atom leaf."""
    meta = {} if U0 is None else {"U0": float(U0)}
    return frame(Z=Z, pos=pos, forces=forces, **meta)


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
        batch = collate_molecules(samples)

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
        batch = collate_molecules(samples)

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


# ---------------------------------------------------------------------------
# universal-class-i-12-torsion-weights — Parsley per-configuration energy weights
#
# Domain basis: Qiu et al., "Development and Benchmarking of Open Force Field
# v1.0.0 — the Parsley Small-Molecule Force Field", JCTC 17, 6262-6280 (2021),
# §2.1.7 eqs. 3-5, https://doi.org/10.1021/acs.jctc.1c00571
#
#     L_g = Σ_{i∈g} w(ΔE_i)·[ (E_MM,i - c_MM) - (E_QM,i - c_QM) ]² / Σ_{i∈g} w(ΔE_i)
#     ΔE_i = E_QM,i - min_j∈g E_QM,j                      (kcal/mol)
#     w(ΔE) = 1                                            ΔE < dE
#     w(ΔE) = dE / sqrt(dE² + (ΔE - dE)²)                  dE ≤ ΔE < ΔE_max
#     w(ΔE) = 0                                            ΔE ≥ ΔE_max
#
# The trailing "−½" of the requirement text is read as the *exponent*
# (w = (1 + (ΔE-1)²)^(-1/2)); the literal "minus one half" reading is
# discontinuous at ΔE = dE and negative for ΔE > 2 (see spec §Domain basis).
# c_MM / c_QM are the *weighted* group means (argmin_c Σ w (d-c)² = Σwd/Σw).
#
# All goldens below are analytic closed forms, hard-coded, float64, atol 1e-12.
# No oracle software of any kind is involved.
# ---------------------------------------------------------------------------

_PARSLEY_ATOL = 1e-12

# sqrt(2) and 1/sqrt(2) as exact float64 literals (avoids a new module import).
_SQRT2 = 1.4142135623730951
_INV_SQRT2 = 0.7071067811865476

# Fixture P — one scan, exercises all three weight branches at once.
#   e_true = [0.0, 0.5, 2.0, 7.0]            → ΔE = [0.0, 0.5, 2.0, 7.0]
#   w      = [1, 1, 1/sqrt(2), 0]            (branch 1, 1, 2, 3)
#   d      = e_pred - e_true = [-0.5, -0.5, sqrt(2), 50.0]
#   Σwd    = -0.5 - 0.5 + (1/sqrt(2))·sqrt(2) + 0 = 0  → weighted mean 0
#   Σwd²   = 0.25 + 0.25 + (1/sqrt(2))·2 + 0 = 0.5 + sqrt(2)
#   Σw     = 2 + 1/sqrt(2)
#   L      = (0.5 + sqrt(2)) / (2 + 1/sqrt(2)) = (1 + 2sqrt(2))/(4 + sqrt(2))
#          = 1/sqrt(2)                        [since (4+sqrt(2))/sqrt(2) = 1+2sqrt(2)]
_P_E_TRUE = torch.tensor([0.0, 0.5, 2.0, 7.0], dtype=torch.float64)
_P_E_PRED = torch.tensor([-0.5, 0.0, 2.0 + _SQRT2, 57.0], dtype=torch.float64)
_P_D = torch.tensor([-0.5, -0.5, _SQRT2, 50.0], dtype=torch.float64)
_P_WEIGHTS = torch.tensor([1.0, 1.0, _INV_SQRT2, 0.0], dtype=torch.float64)
_P_GROUP = torch.tensor([0, 0, 0, 0])
_P_LOSS = 0.7071067811865476  # = 1/sqrt(2), independently re-derived above

# Fixture Q — same scan, uniform +1 offset plus a 50 kcal/mol error parked on
# the ΔE = 7 (w = 0) configuration:
#   d = [1, 1, 1, 50]; Σwd/Σw = (2 + 1/sqrt(2))/(2 + 1/sqrt(2)) = 1
#   r = d - 1 = [0, 0, 0, 49] and w[3] = 0 → L = 0 (analytically exact).
_Q_E_PRED = torch.tensor([1.0, 1.5, 3.0, 57.0], dtype=torch.float64)
# Barrier term (unweighted, per-group max-min): pred 57-1 = 56, ref 7-0 = 7,
# → (56 - 7)² = 2401.
_Q_BARRIER = 2401.0

# Fixture E — uniform weights (both ΔE < 1 kcal/mol), so the weighted objective
# must degenerate to the plain mean-centered MSE:
#   d = [0, 2]; mean 1; r = [-1, 1]; L = (1 + 1)/2 = 1.0 exactly.
_E_E_TRUE = torch.tensor([0.0, 0.5], dtype=torch.float64)
_E_E_PRED = torch.tensor([0.0, 2.5], dtype=torch.float64)
_E_GROUP = torch.tensor([0, 0])


class TestParsleyEnergyWeight:
    def test_branch_table_default_policy(self):

        # denom = 1, upper = 5 kcal/mol.
        # ΔE = 2 → 1/sqrt(1 + 1) = 1/sqrt(2); ΔE = 3 → 1/sqrt(1 + 4) = 1/sqrt(5).
        relative = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0], dtype=torch.float64)
        expected = torch.tensor(
            [1.0, 1.0, 1.0, 0.7071067811865476, 0.4472135954999579, 0.0, 0.0],
            dtype=torch.float64,
        )
        got = ParsleyEnergyWeight()(relative)
        assert torch.allclose(got, expected, atol=_PARSLEY_ATOL)

    def test_output_dtype_matches_input(self):

        relative = torch.tensor([0.0, 2.0], dtype=torch.float64)
        assert ParsleyEnergyWeight()(relative).dtype == torch.float64

    def test_continuous_at_lower_boundary(self):

        # The rejected "minus one half" reading jumps 1.0 → 0.5 here.
        policy = ParsleyEnergyWeight()
        below = policy(torch.tensor([1.0 - 1e-9], dtype=torch.float64))
        at = policy(torch.tensor([1.0], dtype=torch.float64))
        assert torch.allclose(below, at, atol=1e-6)
        assert torch.allclose(below, torch.tensor([1.0], dtype=torch.float64), atol=1e-6)

    def test_non_negative_on_dense_grid(self):

        relative = torch.arange(0.0, 8.0, 0.01, dtype=torch.float64)
        got = ParsleyEnergyWeight()(relative)
        assert bool((got >= 0.0).all())

    def test_monotone_non_increasing_on_dense_grid(self):

        # Second guard on the rejected reading, which increases again past ΔE = 2.
        relative = torch.arange(0.0, 8.0, 0.01, dtype=torch.float64)
        got = ParsleyEnergyWeight()(relative)
        assert bool((torch.diff(got) <= 0.0).all())

    def test_branch_table_non_default_policy(self):

        # denom = 2, upper = 6 kcal/mol.
        # ΔE = 2 → 2/sqrt(4 + 0) = 1; ΔE = 4 → 2/sqrt(4 + 4) = 1/sqrt(2).
        relative = torch.tensor([1.0, 2.0, 4.0, 6.0, 10.0], dtype=torch.float64)
        expected = torch.tensor(
            [1.0, 1.0, 0.7071067811865475, 0.0, 0.0],
            dtype=torch.float64,
        )
        got = ParsleyEnergyWeight(denom=2.0, upper=6.0)(relative)
        assert torch.allclose(got, expected, atol=_PARSLEY_ATOL)

    def test_zero_denom_raises(self):

        with pytest.raises(ValueError):
            ParsleyEnergyWeight(denom=0.0)

    def test_upper_below_denom_raises(self):

        with pytest.raises(ValueError):
            ParsleyEnergyWeight(denom=1.0, upper=0.5)

    def test_negative_relative_energy_raises(self):

        # Guards a caller passing absolute instead of relative QM energies.
        with pytest.raises(ValueError):
            ParsleyEnergyWeight()(torch.tensor([0.0, -0.25], dtype=torch.float64))

    def test_output_is_detached(self):

        relative = torch.tensor([0.0, 2.0], dtype=torch.float64, requires_grad=True)
        got = ParsleyEnergyWeight()(relative)
        assert got.requires_grad is False

    def test_denom_and_upper_are_buffers_at_config_ftype(self):

        policy = ParsleyEnergyWeight()
        buffers = dict(policy.named_buffers())
        assert "denom" in buffers
        assert "upper" in buffers
        assert buffers["denom"].dtype == config.ftype
        assert buffers["upper"].dtype == config.ftype
        names = {name for name, _ in policy.named_parameters()}
        assert "denom" not in names
        assert "upper" not in names

    def test_buffers_appear_in_state_dict(self):

        state = ParsleyEnergyWeight(denom=2.0, upper=6.0).state_dict()
        assert set(state) == {"denom", "upper"}
        assert float(state["denom"]) == 2.0
        assert float(state["upper"]) == 6.0

    def test_buffers_follow_module_to(self):

        policy = ParsleyEnergyWeight().to(torch.float64)
        assert policy.denom.dtype == torch.float64
        assert policy.upper.dtype == torch.float64


class TestCenterByGroupWeights:
    def test_weighted_centering_leaves_zero_weighted_mean_unchanged(self):
        # Fixture P residuals: Σwd = 0, so weighted centering is the identity.
        got = center_by_group(_P_D, _P_GROUP, weights=_P_WEIGHTS)
        assert torch.allclose(got, _P_D, atol=_PARSLEY_ATOL)

    def test_weights_none_matches_plain_mean_golden(self):
        # Same inputs and golden as TestCenterByGroup.test_exact_centering.
        values = torch.tensor([1.0, 3.0, 10.0, 14.0])
        groups = torch.tensor([0, 0, 1, 1])
        got = center_by_group(values, groups, weights=None)
        assert torch.equal(got, torch.tensor([-1.0, 1.0, -2.0, 2.0]))


class TestTorsionProfileLossParsley:
    def test_fixture_p_weighted_profile_is_inverse_sqrt_two(self):

        loss_fn = TorsionProfileLoss(energy_weight=ParsleyEnergyWeight())
        got = loss_fn(
            {"energy": _P_E_PRED},
            {"energy": _P_E_TRUE, "molecule_id_index": _P_GROUP},
        )
        assert torch.allclose(got, torch.tensor(_P_LOSS, dtype=torch.float64), atol=_PARSLEY_ATOL)

    def test_fixture_q_zero_weight_configuration_is_excluded(self):

        # Analytic value is exactly 0; float64 weighted centering leaves ~1e-32.
        loss_fn = TorsionProfileLoss(energy_weight=ParsleyEnergyWeight())
        got = loss_fn(
            {"energy": _Q_E_PRED},
            {"energy": _P_E_TRUE, "molecule_id_index": _P_GROUP},
        )
        assert torch.allclose(got, torch.tensor(0.0, dtype=torch.float64), atol=_PARSLEY_ATOL)

    def test_fixture_q_barrier_term_stays_unweighted(self):

        loss_fn = TorsionProfileLoss(
            energy_weight=ParsleyEnergyWeight(),
            barrier_weight=1.0,
        )
        got = loss_fn(
            {"energy": _Q_E_PRED},
            {"energy": _P_E_TRUE, "molecule_id_index": _P_GROUP},
        )
        assert torch.allclose(
            got, torch.tensor(_Q_BARRIER, dtype=torch.float64), atol=_PARSLEY_ATOL
        )

    def test_zero_weight_configuration_gets_zero_gradient(self):

        # dL/dE_MM,j = 2 w_j r_j / Σw → [-0.3694, -0.3694, 0.7388, 0.0].
        e_pred = _P_E_PRED.clone().requires_grad_()
        loss_fn = TorsionProfileLoss(energy_weight=ParsleyEnergyWeight())
        loss_fn(
            {"energy": e_pred},
            {"energy": _P_E_TRUE, "molecule_id_index": _P_GROUP},
        ).backward()
        grad = e_pred.grad
        assert grad is not None
        assert float(grad[3]) == 0.0
        assert bool(torch.isfinite(grad[:3]).all())
        assert bool((grad[:3] != 0.0).all())

    def test_reduction_none_is_per_profile_and_grid_count_invariant(self):

        # Group 1 is fixture P with every grid point duplicated (8 points):
        # numerator and denominator both double → identical per-profile loss.
        e_pred = torch.cat([_P_E_PRED, _P_E_PRED.repeat_interleave(2)])
        e_true = torch.cat([_P_E_TRUE, _P_E_TRUE.repeat_interleave(2)])
        groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        loss_fn = TorsionProfileLoss(energy_weight=ParsleyEnergyWeight(), reduction="none")
        got = loss_fn({"energy": e_pred}, {"energy": e_true, "molecule_id_index": groups})
        assert got.shape == (2,)
        assert torch.allclose(
            got,
            torch.tensor([_P_LOSS, _P_LOSS], dtype=torch.float64),
            atol=_PARSLEY_ATOL,
        )

    def test_reduction_sum_adds_profiles(self):

        e_pred = torch.cat([_P_E_PRED, _P_E_PRED.repeat_interleave(2)])
        e_true = torch.cat([_P_E_TRUE, _P_E_TRUE.repeat_interleave(2)])
        groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        loss_fn = TorsionProfileLoss(energy_weight=ParsleyEnergyWeight(), reduction="sum")
        got = loss_fn({"energy": e_pred}, {"energy": e_true, "molecule_id_index": groups})
        # 2 / sqrt(2) = sqrt(2)
        assert torch.allclose(got, torch.tensor(_SQRT2, dtype=torch.float64), atol=_PARSLEY_ATOL)

    def test_reduction_mean_averages_profiles(self):

        e_pred = torch.cat([_P_E_PRED, _P_E_PRED.repeat_interleave(2)])
        e_true = torch.cat([_P_E_TRUE, _P_E_TRUE.repeat_interleave(2)])
        groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        loss_fn = TorsionProfileLoss(energy_weight=ParsleyEnergyWeight(), reduction="mean")
        got = loss_fn({"energy": e_pred}, {"energy": e_true, "molecule_id_index": groups})
        assert torch.allclose(got, torch.tensor(_P_LOSS, dtype=torch.float64), atol=_PARSLEY_ATOL)

    def test_equal_weights_match_default_path(self):

        pred = {"energy": _E_E_PRED}
        target = {"energy": _E_E_TRUE, "molecule_id_index": _E_GROUP}
        weighted = TorsionProfileLoss(energy_weight=ParsleyEnergyWeight())(pred, target)
        default = TorsionProfileLoss()(pred, target)
        assert float(weighted) == 1.0
        assert float(default) == 1.0
        assert float(weighted) == float(default)

    def test_singleton_group_with_weights_is_zero(self):

        loss_fn = TorsionProfileLoss(energy_weight=ParsleyEnergyWeight())
        got = loss_fn(
            {"energy": torch.tensor([9.0], dtype=torch.float64)},
            {
                "energy": torch.tensor([3.0], dtype=torch.float64),
                "molecule_id_index": torch.tensor([0]),
            },
        )
        assert torch.allclose(got, torch.tensor(0.0, dtype=torch.float64), atol=_PARSLEY_ATOL)

    def test_default_off_equivalence_is_bit_for_bit(self):
        # Pre-existing fixture from TestTorsionProfileLoss (pred [0, 4], true [0, 2]).
        pred = {"energy": torch.tensor([0.0, 4.0])}
        target = {
            "energy": torch.tensor([0.0, 2.0]),
            "molecule_id_index": torch.tensor([0, 0]),
        }
        implicit = TorsionProfileLoss()(pred, target)
        explicit = TorsionProfileLoss(energy_weight=None)(pred, target)
        assert float(implicit) == 1.0
        assert float(explicit) == float(implicit)
