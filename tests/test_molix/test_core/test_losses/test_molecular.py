"""Unit tests for molecular loss presets in :mod:`molix.core.losses.molecular`."""

from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from molix.core.losses import (
    MeanTermEnergyLoss,
    WeightedLoss,
    energy_force_mse,
    energy_loss,
    energy_mse,
    force_loss,
)
from molix.core.losses.molecular import (
    AtomicChargeLoss,
    BoltzmannProfileLoss,
    ChargeMSELoss,
    LogRatioPrior,
    NormPrior,
    ParsleyEnergyWeight,
    PartialHessianLoss,
    ProjectedInternalHessianLoss,
    ReferenceShiftPrior,
    ScaleNormalizedLoss,
    StationaryForceLoss,
    TorsionProfileLoss,
    center_by_group,
    log_knee,
    molecule_centered_energy_mse,
    parameter_bag_mse,
)
from molix.data.collate import collate_molecules
from tests.conftest import frame


def _sample(Z, pos, *, U0=None, forces=None):
    """One molpy ``Frame``; ``U0`` becomes a graph leaf, forces a per-atom leaf.

    The batches below are collated at fp32 — the dtype a ``DataModule`` serves
    a model at — so the fp32 predictions the tests hand the losses meet targets
    of the same dtype, as they do in a pipeline.
    """
    meta = {} if U0 is None else {"U0": float(U0)}
    return frame(Z=Z, pos=pos, forces=forces, **meta)


class TestEnergyMSE:
    def test_matches_handwritten_mse_on_graph_energy(self):
        samples = [
            _sample([1, 6], [[0, 0, 0], [1, 0, 0]], U0=1.0),
            _sample([6, 8], [[0, 0, 0], [1.2, 0, 0]], U0=2.5),
        ]
        batch = collate_molecules(samples, dtype=torch.float32)
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
        batch = collate_molecules(samples, dtype=torch.float32)
        preds = {"energy": torch.tensor([1.5, 2.5])}

        loss_fn = energy_mse("U0", reduction="none")
        got = loss_fn(preds, batch)
        assert got.shape == preds["energy"].shape
        assert torch.allclose(got, torch.tensor([0.25, 0.25]))

    def test_custom_pred_key(self):
        batch = collate_molecules([_sample([1], [[0, 0, 0]], U0=2.0)], dtype=torch.float32)
        preds = {"E_pred": torch.tensor([1.0])}
        loss_fn = energy_mse("U0", pred_key="E_pred")
        assert torch.allclose(loss_fn(preds, batch), torch.tensor(1.0))


class TestEnergyLossGraphWeight:
    """``weight_key`` on ``graphs`` — a padded batch's dead slots drop out of the mean."""

    def _batch(self):
        samples = [
            _sample([1, 6], [[0, 0, 0], [1, 0, 0]], U0=1.0),
            _sample([6, 8], [[0, 0, 0], [1.2, 0, 0]], U0=2.5),
            _sample([1], [[0, 0, 0]], U0=0.0),  # a dead slot: label 0, garbage prediction
        ]
        return collate_molecules(samples, dtype=torch.float32)

    def test_graphs_mask_weights_the_mean(self):
        batch = self._batch()
        batch["graphs", "mask"] = torch.tensor([True, True, False])
        preds = {"energy": torch.tensor([0.8, 2.7, 100.0])}
        got = energy_loss("U0")(preds, batch)
        expected = torch.tensor(((0.8 - 1.0) ** 2 + (2.7 - 2.5) ** 2) / 2)
        assert torch.allclose(got, expected)

    def test_absent_weight_column_is_unweighted(self):
        batch = self._batch()
        preds = {"energy": torch.tensor([0.8, 2.7, 1.0])}
        got = energy_loss("U0")(preds, batch)
        expected = torch.nn.functional.mse_loss(preds["energy"], batch["graphs", "U0"])
        assert torch.allclose(got, expected)

    def test_weight_key_none_ignores_the_mask(self):
        batch = self._batch()
        batch["graphs", "mask"] = torch.tensor([True, True, False])
        preds = {"energy": torch.tensor([0.8, 2.7, 1.0])}
        got = energy_loss("U0", weight_key=None)(preds, batch)
        expected = torch.nn.functional.mse_loss(preds["energy"], batch["graphs", "U0"])
        assert torch.allclose(got, expected)

    def test_sum_and_none_reductions_apply_the_weight(self):
        batch = self._batch()
        batch["graphs", "mask"] = torch.tensor([True, True, False])
        preds = {"energy": torch.tensor([0.8, 2.7, 100.0])}
        per_graph = energy_loss("U0", reduction="none")(preds, batch)
        assert torch.allclose(per_graph, torch.tensor([0.04, 0.04, 0.0]))
        total = energy_loss("U0", reduction="sum")(preds, batch)
        assert torch.allclose(total, torch.tensor(0.08))

    def test_per_atom_and_mask_compose(self):
        batch = self._batch()
        batch["graphs", "mask"] = torch.tensor([True, True, False])
        preds = {"energy": torch.tensor([0.8, 2.7, 100.0])}
        got = energy_loss("U0", per_atom=True)(preds, batch)
        expected = torch.tensor(((0.8 - 1.0) ** 2 / 4 + (2.7 - 2.5) ** 2 / 4) / 2)
        assert torch.allclose(got, expected)

    def test_energy_mse_preset_forwards_the_weight(self):
        batch = self._batch()
        batch["graphs", "mask"] = torch.tensor([True, True, False])
        preds = {"energy": torch.tensor([0.8, 2.7, 100.0])}
        assert torch.allclose(energy_mse("U0")(preds, batch), energy_loss("U0")(preds, batch))


class TestEnergyForceMSE:
    def test_combines_energy_and_force_terms(self):
        forces_a = torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        forces_b = torch.tensor([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])
        samples = [
            _sample([1, 6], [[0, 0, 0], [1, 0, 0]], U0=1.0, forces=forces_a),
            _sample([6, 8], [[0, 0, 0], [1.2, 0, 0]], U0=2.0, forces=forces_b),
        ]
        batch = collate_molecules(samples, dtype=torch.float32)

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
        batch = collate_molecules(samples, dtype=torch.float32)

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

    def test_masked_force_term_averages_over_the_real_atoms_only(self):
        """On a padded batch the force MSE denominator is ``atoms.mask.sum()``.

        ``MoleculePadder`` appends ghost atoms whose force target and force
        prediction are both zero, so an unmasked mean silently divides the real
        residual by ``max_atoms`` instead of the real atom count — the same
        loss value then depends on how much padding a batch happened to carry.
        The energy term is untouched (``graphs`` is never padded).
        """
        from molix.data import MoleculePadder

        forces_a = torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        forces_b = torch.tensor([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])
        samples = [
            _sample([1, 6], [[0, 0, 0], [1, 0, 0]], U0=1.0, forces=forces_a),
            _sample([6, 8], [[0, 0, 0], [1.2, 0, 0]], U0=2.0, forces=forces_b),
        ]
        batch = collate_molecules(samples, dtype=torch.float32)
        n_real = int(batch["atoms"].batch_size[0])
        padded = MoleculePadder(max_atoms=n_real + 6, max_pairs=8, cutoff=1.0)(batch.clone())
        preds = {
            "energy": torch.tensor([0.5, 2.5]),
            "forces": torch.zeros_like(padded["atoms", "forces"]),
        }

        got = energy_force_mse(energy_target_key="U0", force_target_key="forces", lambda_F=0.5)(
            preds, padded
        )

        e_true = padded["graphs", "U0"].view_as(preds["energy"])
        f_true = padded["atoms", "forces"][:n_real]
        expected = torch.nn.functional.mse_loss(
            preds["energy"], e_true
        ) + 0.5 * torch.nn.functional.mse_loss(preds["forces"][:n_real], f_true)
        assert torch.allclose(got, expected)
        unmasked = torch.nn.functional.mse_loss(preds["forces"], padded["atoms", "forces"])
        assert not torch.allclose(
            0.5 * unmasked, got - torch.nn.functional.mse_loss(preds["energy"], e_true)
        )


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
        pred = {"charge": torch.tensor([0.2, -0.2])}
        target = {"charge": torch.tensor([0.1, -0.1])}
        got = ChargeMSELoss()(pred, target)
        assert torch.allclose(got, torch.tensor(0.01), atol=_ENERGY_ATOL)

    def test_forward_does_not_require_energy_keys(self):
        pred = {"charge": torch.tensor([0.2, -0.2])}
        target = {"charge": torch.tensor([0.1, -0.1])}
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
        loss_fn = ScaleNormalizedLoss(_FixedLoss(4.0), scale=2.0, dtype=torch.float64)
        buffers = dict(loss_fn.named_buffers())
        assert "scale" in buffers
        assert buffers["scale"].dtype == torch.float64
        assert "scale" not in {name for name, _ in loss_fn.named_parameters()}

    def test_zero_scale_raises(self):
        with pytest.raises(ValueError):
            ScaleNormalizedLoss(_FixedLoss(4.0), scale=0.0)

    def test_weighted_loss_combo_on_charge_golden(self):
        loss_fn = WeightedLoss([(1.0, ScaleNormalizedLoss(ChargeMSELoss(), scale=2.0))])
        pred = {"charge": torch.tensor([0.2, -0.2])}
        target = {"charge": torch.tensor([0.1, -0.1])}
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

    def test_denom_and_upper_are_buffers_at_the_given_dtype(self):
        policy = ParsleyEnergyWeight(dtype=torch.float64)
        buffers = dict(policy.named_buffers())
        assert "denom" in buffers
        assert "upper" in buffers
        assert buffers["denom"].dtype == torch.float64
        assert buffers["upper"].dtype == torch.float64
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


class TestComposableLosses:
    """The three axes vary independently: measure, weight, combination."""

    @staticmethod
    def _batch() -> TensorDict:
        return TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "forces": torch.zeros(3, 3),
                        "inlier": torch.tensor([1.0, 0.0, 1.0]),
                    },
                    batch_size=[3],
                ),
                "graphs": TensorDict(
                    {"energy": torch.zeros(1), "num_atoms": torch.tensor([3])},
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )

    def test_the_measure_is_the_callers(self) -> None:
        """A Huber past its delta grows linearly where a squared error grows squared."""
        preds = {"energy": torch.tensor([100.0])}
        batch = self._batch()

        squared = energy_loss("energy")(preds, batch)
        huber = energy_loss("energy", elementwise=nn.HuberLoss(delta=1.0, reduction="none"))(
            preds, batch
        )

        assert float(squared) == pytest.approx(100.0**2)
        assert float(huber) == pytest.approx(100.0 - 0.5)

    def test_a_per_atom_weight_drops_a_row_without_gathering_it(self) -> None:
        """The masked row contributes nothing, and the mean divides by what it kept."""
        preds = {"forces": torch.tensor([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0], [1.0, 0.0, 0.0]])}
        batch = self._batch()

        weighted = force_loss("forces", weight_key="inlier")(preds, batch)
        unweighted = force_loss("forces", weight_key=None)(preds, batch)

        # Two kept rows, one squared unit each, over 2*3 weighted elements.
        assert float(weighted) == pytest.approx(2.0 / 6.0)
        assert float(unweighted) > float(weighted)

    def test_the_shape_does_not_move_with_how_many_rows_are_masked(self) -> None:
        """A weight is a multiply, never a gather — the output stays a scalar."""
        preds = {"forces": torch.ones(3, 3)}
        batch = self._batch()
        batch["atoms", "inlier"] = torch.zeros(3)

        assert force_loss("forces", weight_key="inlier")(preds, batch).shape == ()


class TestPartialHessianLoss:
    """Trace-relative MAE on the partial Hessian — ByteFF's stage-1 objective."""

    @staticmethod
    def _batch(label: torch.Tensor, mask: torch.Tensor | None = None):
        namespace = {"block": label}
        if mask is not None:
            namespace["mask"] = mask
        return TensorDict(
            {"hessian_pairs": TensorDict(namespace, batch_size=[label.shape[0]])},
            batch_size=[],
        )

    def test_a_perfect_prediction_scores_zero(self) -> None:
        """The residual, not the magnitude, is what is measured."""
        label = torch.arange(18, dtype=torch.float64).reshape(2, 9) * 7.0
        loss = PartialHessianLoss()({"partial_hessian": label.clone()}, self._batch(label))
        assert float(loss) == 0.0

    def test_each_block_is_divided_by_its_own_traces_magnitude(self) -> None:
        """A stiff pair and a soft one must weigh the same, which is why it is relative.

        Two blocks with the same absolute residual but traces of 100 and 1000 must
        contribute in the ratio 10:1. An absolute residual would fit the stiff pair
        and ignore the soft one — curvature spans orders of magnitude within one
        molecule.
        """
        label = torch.zeros(2, 9, dtype=torch.float64)
        label[0, 0] = label[0, 4] = label[0, 8] = 100.0 / 3.0
        label[1, 0] = label[1, 4] = label[1, 8] = 1000.0 / 3.0
        pred = label.clone()
        pred[:, 1] += 5.0  # the same off-diagonal error in both
        residual = ((pred - label).abs() / torch.tensor([[100.0], [1000.0]])).sum(dim=-1)
        expected = residual.sum() / (2 * 9)
        torch.testing.assert_close(
            PartialHessianLoss()({"partial_hessian": pred}, self._batch(label)), expected
        )

    def test_a_nearly_flat_block_cannot_dominate_by_dividing_by_almost_zero(self) -> None:
        """The floor is why: without it a trace of 1e-6 makes one pair the whole loss."""
        label = torch.zeros(1, 9, dtype=torch.float64)
        pred = torch.full((1, 9), 1.0, dtype=torch.float64)
        floored = PartialHessianLoss(trace_floor=10.0)
        torch.testing.assert_close(
            floored({"partial_hessian": pred}, self._batch(label)),
            torch.tensor(1.0 / 10.0, dtype=torch.float64),
        )

    def test_a_blown_up_predicted_block_drops_its_row_instead_of_swamping_the_mean(
        self,
    ) -> None:
        """One absurd constant early in a fit must not steer the batch.

        Clipping would keep its gradient; letting it through would make the mean its
        own. ByteFF drops the row. With the only bad row dropped, the loss is the
        good row's alone.
        """
        label = torch.zeros(2, 9, dtype=torch.float64)
        pred = torch.zeros(2, 9, dtype=torch.float64)
        pred[0, 0] = 5.0  # a real, modest residual
        pred[1, 0] = 1e9  # blown up
        loss = PartialHessianLoss(blowup=1e4)({"partial_hessian": pred}, self._batch(label))
        torch.testing.assert_close(loss, torch.tensor(5.0 / 10.0 / 9.0, dtype=torch.float64))

    def test_dead_pair_rows_are_excluded_from_the_mean_not_merely_zeroed(self) -> None:
        """Otherwise the loss shrinks with however empty the batch happened to be.

        A padded row has a zero label and a zero prediction, so it adds nothing to
        the numerator — but counting it in the denominator would scale the whole
        loss by the fill fraction, making a run's numbers depend on its packing.
        """
        label = torch.zeros(4, 9, dtype=torch.float64)
        pred = torch.zeros(4, 9, dtype=torch.float64)
        pred[0, 0] = 5.0
        mask = torch.tensor([True, False, False, False])
        dense = PartialHessianLoss()({"partial_hessian": pred[:1]}, self._batch(label[:1]))
        padded = PartialHessianLoss()({"partial_hessian": pred}, self._batch(label, mask))
        torch.testing.assert_close(padded, dense)

    def test_a_batch_without_the_namespace_is_refused(self) -> None:
        """There is nothing to fit, and a zero loss would read as converged."""
        with pytest.raises(ValueError, match="no hessian_pairs namespace"):
            PartialHessianLoss()(
                {"partial_hessian": torch.zeros(1, 9)}, TensorDict({}, batch_size=[])
            )

    def test_a_shape_disagreement_is_refused(self) -> None:
        """Both sides are ``(n_pairs, 9)`` in the same order, or the rows are mismatched."""
        label = torch.zeros(3, 9, dtype=torch.float64)
        with pytest.raises(ValueError, match="does not match the label"):
            PartialHessianLoss()({"partial_hessian": torch.zeros(2, 9)}, self._batch(label))


class TestReferenceShiftPrior:
    def test_is_the_mean_squared_shift_over_live_rows(self):
        """A padded row neither scores nor counts."""
        batch = TensorDict(
            {"angles": TensorDict({"mask": torch.tensor([True, True, False])}, batch_size=[3])},
            batch_size=[],
        )
        pred = {"angle_theta0_residual": torch.tensor([0.1, -0.3, 5.0])}
        got = ReferenceShiftPrior()(pred, batch)
        assert torch.allclose(got, torch.tensor((0.01 + 0.09) / 2))

    def test_a_missing_prediction_is_an_error(self):
        batch = TensorDict({"angles": TensorDict({}, batch_size=[0])}, batch_size=[])
        with pytest.raises(KeyError, match="angle_theta0_residual"):
            ReferenceShiftPrior()({}, batch)


class TestLogRatioPrior:
    """Mean squared log-ratio of a parameter to its reference, over live rows."""

    @staticmethod
    def _batch(value: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor | None = None):
        atoms = {"sigma": value, "sigma_ref": reference}
        if mask is not None:
            atoms["mask"] = mask
        return TensorDict({"atoms": TensorDict(atoms, batch_size=[value.shape[0]])}, batch_size=[])

    @staticmethod
    def _prior() -> LogRatioPrior:
        return LogRatioPrior(("atoms", "sigma"), ("atoms", "sigma_ref"), namespace="atoms")

    def test_is_zero_on_the_reference_and_the_mean_squared_log_ratio_off_it(self) -> None:
        reference = torch.tensor([3.4, 2.6, 1.9], dtype=torch.float64)
        on = self._batch(reference.clone(), reference)
        assert float(self._prior()(on, on)) == 0.0
        value = reference * torch.tensor([2.0, 0.5, 1.0], dtype=torch.float64)
        off = self._batch(value, reference)
        expected = (2 * math.log(2.0) ** 2) / 3
        torch.testing.assert_close(
            self._prior()(off, off), torch.tensor(expected, dtype=torch.float64)
        )

    def test_a_ghost_row_with_a_zero_reference_poisons_neither_value_nor_gradient(self) -> None:
        """Dead rows carry ``sigma_ref = 0`` and ``sigma = 0``; ``0 · log(0/0)`` is NaN."""
        raw = torch.tensor([0.1, -0.2, 0.0], requires_grad=True)
        reference = torch.tensor([3.4, 2.6, 0.0])
        value = reference * torch.exp(raw)
        mask = torch.tensor([1.0, 1.0, 0.0])
        padded = self._batch(value, reference, mask)
        dense = self._batch(value[:2], reference[:2])
        got = self._prior()(padded, padded)
        torch.testing.assert_close(got, self._prior()(dense, dense))
        got.backward()
        assert torch.isfinite(raw.grad).all()
        assert float(raw.grad[2]) == 0.0

    def test_a_missing_column_is_refused(self) -> None:
        batch = TensorDict(
            {"atoms": TensorDict({"sigma": torch.ones(2)}, batch_size=[2])}, batch_size=[]
        )
        with pytest.raises(KeyError, match="sigma_ref"):
            self._prior()(batch, batch)


class TestNormPrior:
    """ByteFF's ``l1_norm``: the mean |k| over every element of the live rows."""

    @staticmethod
    def _batch(mask):
        return TensorDict(
            {"propers": TensorDict({"mask": torch.tensor(mask)}, batch_size=[len(mask)])},
            batch_size=[],
        )

    def test_is_the_mean_absolute_value_over_live_elements(self):
        """A padded row neither scores nor counts; every term of a live row does."""
        k = torch.tensor([[1.0, -2.0, 0.0], [0.5, 0.5, -0.5], [9.0, 9.0, 9.0]])
        got = NormPrior(("propers", "k"))({("propers", "k"): k}, self._batch([True, True, False]))
        assert torch.allclose(got, torch.tensor((3.0 + 1.5) / 6))

    def test_p_two_is_the_mean_square(self):
        k = torch.tensor([[1.0, -2.0], [3.0, 0.0]])
        got = NormPrior(("propers", "k"), p=2)({("propers", "k"): k}, self._batch([True, True]))
        assert torch.allclose(got, torch.tensor((1.0 + 4.0 + 9.0) / 4))

    def test_without_a_mask_every_row_is_live(self):
        k = torch.tensor([-1.0, 3.0])
        batch = TensorDict({"propers": TensorDict({}, batch_size=[2])}, batch_size=[])
        prior = NormPrior(("propers", "k"))
        assert torch.allclose(prior({("propers", "k"): k}, batch), torch.tensor(2.0))

    def test_pulls_every_live_element_toward_zero_by_the_same_amount(self):
        """The L1 gradient is ``sign(k) / (live rows · terms)`` — a constant pull."""
        k = torch.tensor([[1.0, -2.0], [0.5, 0.25], [7.0, 7.0]], requires_grad=True)
        prior = NormPrior(("propers", "k"))
        prior({("propers", "k"): k}, self._batch([True, True, False])).backward()
        expected = torch.tensor([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]) / 4
        assert torch.allclose(k.grad, expected)

    def test_a_missing_prediction_is_an_error(self):
        with pytest.raises(KeyError, match="propers"):
            NormPrior(("propers", "k"))({}, self._batch([True]))

    def test_p_is_one_or_two(self):
        with pytest.raises(ValueError, match="norm's power"):
            NormPrior(("propers", "k"), p=3)


class TestAtomicChargeLoss:
    """Charge MSE on a packed batch, where the ghosts must not count."""

    @staticmethod
    def _batch(reference: torch.Tensor, mask: torch.Tensor | None = None):
        atoms = {"atomic_charge": reference}
        if mask is not None:
            atoms["mask"] = mask
        return TensorDict(
            {"atoms": TensorDict(atoms, batch_size=[reference.shape[0]])}, batch_size=[]
        )

    def test_it_is_the_mean_squared_error_over_the_live_atoms(self) -> None:
        """The plain case, so the padded case below has something to equal."""
        reference = torch.tensor([0.31, -0.62, 0.31], dtype=torch.float64)
        pred = reference + torch.tensor([0.1, -0.2, 0.0], dtype=torch.float64)
        got = AtomicChargeLoss()({"charge": pred}, self._batch(reference))
        torch.testing.assert_close(got, torch.tensor((0.01 + 0.04) / 3, dtype=torch.float64))

    def test_ghost_atoms_are_excluded_from_the_mean_not_merely_zeroed(self) -> None:
        """A ghost's label is zero while the charge solve still answers for it.

        Counting the ghosts in the denominator would scale the loss by the fill
        fraction; counting them in the numerator would pull every predicted charge
        toward zero. Neither is a property of the chemistry.
        """
        reference = torch.tensor([0.31, -0.62, 0.31, 0.0, 0.0], dtype=torch.float64)
        pred = torch.tensor([0.41, -0.82, 0.31, 0.77, -0.5], dtype=torch.float64)
        mask = torch.tensor([True, True, True, False, False])
        padded = AtomicChargeLoss()({"charge": pred}, self._batch(reference, mask))
        dense = AtomicChargeLoss()({"charge": pred[:3]}, self._batch(reference[:3]))
        torch.testing.assert_close(padded, dense)

    def test_a_batch_without_the_label_column_is_refused(self) -> None:
        """A zero loss on a set that carries no charges would read as converged."""
        batch = TensorDict(
            {"atoms": TensorDict({"Z": torch.tensor([1, 1])}, batch_size=[2])}, batch_size=[]
        )
        with pytest.raises(ValueError, match="no 'atomic_charge' column"):
            AtomicChargeLoss()({"charge": torch.zeros(2)}, batch)

    def test_a_shape_disagreement_is_refused(self) -> None:
        """One value per atom of the batch, on both sides."""
        reference = torch.zeros(3, dtype=torch.float64)
        with pytest.raises(ValueError, match="do not match the label"):
            AtomicChargeLoss()({"charge": torch.zeros(2)}, self._batch(reference))


class TestBoltzmannProfileLoss:
    """ByteFF's torsion-stage objective: relative, weighted, offset-free."""

    @staticmethod
    def _batch(reference, profile, mask=None):
        graphs = {
            "energy": torch.tensor(reference, dtype=torch.float64),
            "profile": torch.tensor(profile, dtype=torch.long),
        }
        if mask is not None:
            graphs["mask"] = torch.tensor(mask)
        return TensorDict(
            {"graphs": TensorDict(graphs, batch_size=[len(reference)])}, batch_size=[]
        )

    def test_the_gradient_is_the_gradient_of_the_loss(self) -> None:
        """The detached weight must be a constant of the data, not of the prediction.

        ByteFF weights by the lower of the label's and the prediction's aligned
        energy and detaches the result, which drops ``dw/dE``: the number reported
        is then not the function whose gradient is followed, and the objective
        moves under the optimizer. Measured consequence — a torsion-scan overfit
        reaches 0.27 by step 1000 and walks to 74 or 2755 depending on the run, at
        the same learning rate.

        A central difference is the whole check: it is insensitive to how the loss
        is implemented and it fails on a gradient that is not one.
        """
        # The inputs are the whole test. The weight depends on the prediction only
        # where two things hold at once: the prediction's relative energy is below
        # the label's, so ``min`` picks it, *and* above ``clamp``, so the
        # exponential is not saturated at 1. Without such a conformer a finite
        # difference agrees on both weightings and proves nothing — measured, on
        # the obvious input set. Conformer 2 is built to bind it: label relative
        # 6.0 against predicted relative 3.0, with ``clamp`` at 2.0. On the
        # ``min(label, prediction)`` form the two gradients differ by 58 % there.
        reference = [0.0, 0.0, 6.0, 0.3, 1.1]
        profile = [0, 0, 0, 1, 1]
        pred = torch.tensor([0.0, 0.0, 3.0, 1.4, 0.2], dtype=torch.float64)
        objective = BoltzmannProfileLoss()
        batch = self._batch(reference, profile)

        variable = pred.clone().requires_grad_(True)
        objective({"energy": variable}, batch).backward()
        analytic = variable.grad.clone()

        step = 1e-6
        numeric = torch.zeros_like(pred)
        for i in range(pred.numel()):
            up, down = pred.clone(), pred.clone()
            up[i] += step
            down[i] -= step
            numeric[i] = (
                float(objective({"energy": up}, batch)) - float(objective({"energy": down}, batch))
            ) / (2.0 * step)
        torch.testing.assert_close(analytic, numeric, atol=1e-6, rtol=1e-5)

    def test_the_weight_does_not_move_when_only_the_prediction_does(self) -> None:
        """A profile's weights are its label's, so two predictions share them.

        The discrete half of the same defect: the prediction's per-profile minimum
        is an argmin, and a weight that depends on it jumps for every conformer of
        that profile the moment it moves to another one.
        """
        reference = [0.0, 0.5, 4.0]
        batch = self._batch(reference, [0, 0, 0])
        objective = BoltzmannProfileLoss()
        # Same residual shape, different argmin: conformer 0 is the predicted
        # minimum in one and conformer 2 in the other.
        a = torch.tensor([0.0, 1.5, 5.0], dtype=torch.float64)
        b = torch.tensor([5.0, 6.5, 0.0], dtype=torch.float64) - 5.0 + a[0]
        first = float(objective({"energy": a}, batch))
        second = float(objective({"energy": b}, batch))
        assert first != pytest.approx(second), "the two predictions differ; this is a sanity check"
        # And the label-only weight makes each a plain function of its residual:
        # shifting a prediction by a constant cannot change the loss at all.
        shifted = float(objective({"energy": a + 137.0}, batch))
        assert shifted == pytest.approx(first, rel=1e-12)

    def test_a_constant_offset_per_profile_costs_nothing(self) -> None:
        """The whole reason this loss exists.

        THEMol publishes absolute energies near -503 415 kcal/mol against a
        ~10 kcal/mol barrier. A profile predicted with the right shape and the
        wrong constant is a correct torsion term, and must score zero.
        """
        reference = [0.0, 1.0, 2.0]
        pred = torch.tensor([1.0e5, 1.0e5 + 1.0, 1.0e5 + 2.0], dtype=torch.float64)
        loss = BoltzmannProfileLoss()({"energy": pred}, self._batch(reference, [0, 0, 0]))
        assert float(loss) == pytest.approx(0.0, abs=1e-12)

    def test_profiles_weigh_equally_regardless_of_their_conformer_count(self) -> None:
        """A 24-point scan and a 12-point one are one datum each, not 24 and 12.

        Two profiles here, of three and two conformers: the first is exact up to
        a constant (0), the second carries a residual of 1 after its own optimal
        offset, so the mean over profiles is 1/2 — not the per-conformer mean.
        """
        reference = [0.0, 1.0, 2.0, 0.0, 1.0]
        pred = torch.tensor([10.0, 11.0, 12.0, 0.0, 3.0], dtype=torch.float64)
        loss = BoltzmannProfileLoss()({"energy": pred}, self._batch(reference, [0, 0, 0, 1, 1]))
        torch.testing.assert_close(loss, torch.tensor(0.5, dtype=torch.float64))

    def test_an_error_high_above_the_minimum_costs_less_than_the_same_error_near_it(
        self,
    ) -> None:
        """The Boltzmann weight's whole job.

        A barrier top that only one of the two curves reaches would otherwise
        dominate a fit whose business is the shape near the accessible minima.
        Same absolute error, moved from 0.5 to 10 kcal/mol above the minimum.
        """
        delta = 0.4
        low = torch.tensor([0.0, 0.5, 0.5 + delta], dtype=torch.float64)
        high = torch.tensor([0.0, 0.5, 10.0 + delta], dtype=torch.float64)
        loss_low = BoltzmannProfileLoss()({"energy": low}, self._batch([0.0, 0.5, 0.5], [0, 0, 0]))
        loss_high = BoltzmannProfileLoss()(
            {"energy": high}, self._batch([0.0, 0.5, 10.0], [0, 0, 0])
        )
        assert float(loss_high) < float(loss_low)

    def test_dead_graph_slots_enter_neither_the_minimum_nor_the_mean(self) -> None:
        """A padded slot holds a zero energy, which would win every minimum.

        It carries the last profile index — the padder's ghost sink — so it can
        never join a live profile, and an unoccupied index contributes nothing to
        the average over profiles.
        """
        reference = [0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0]
        pred = torch.tensor([10.0, 11.0, 12.0, 0.0, 3.0, 0.0, 0.0], dtype=torch.float64)
        mask = [True] * 5 + [False, False]
        padded = BoltzmannProfileLoss()(
            {"energy": pred}, self._batch(reference, [0, 0, 0, 1, 1, 6, 6], mask)
        )
        dense = BoltzmannProfileLoss()(
            {"energy": pred[:5]}, self._batch(reference[:5], [0, 0, 0, 1, 1])
        )
        torch.testing.assert_close(padded, dense)

    def test_a_batch_without_the_profile_index_is_refused(self) -> None:
        """Without it there is nothing to be relative *to*, and the loss is a lie."""
        batch = TensorDict(
            {"graphs": TensorDict({"energy": torch.zeros(2)}, batch_size=[2])}, batch_size=[]
        )
        with pytest.raises(ValueError, match="ProfileIndex"):
            BoltzmannProfileLoss()({"energy": torch.zeros(2)}, batch)

    def test_a_non_positive_decay_is_refused(self) -> None:
        """It divides an energy; zero would make every weight 0 or inf."""
        with pytest.raises(ValueError, match="decay must be > 0"):
            BoltzmannProfileLoss(decay=0.0)


class TestBoltzmannProfileSaturation:
    """``soft_max`` — ByteFF's ``boltzmann_soft_mse``, for off-equilibrium frames."""

    @staticmethod
    def _batch(reference, profile):
        return TensorDict(
            {
                "graphs": TensorDict(
                    {
                        "energy": torch.tensor(reference, dtype=torch.float64),
                        "profile": torch.tensor(profile, dtype=torch.long),
                    },
                    batch_size=[len(reference)],
                )
            },
            batch_size=[],
        )

    def test_past_the_knee_the_loss_grows_logarithmically_not_quadratically(self) -> None:
        """The property, not the formula: doubling a profile's loss costs ``m ln 2``.

        That is what ``m (1 + log(L/m))`` means, and its gradient ``m / L`` is what
        bounds one frame against a repulsive wall. Asserted by a *difference* so the
        test cannot pass by restating the implementation. With this fixture
        ``L = X**2 / 4``, so scaling the residual by ``sqrt(2)`` doubles it.
        """
        knee = 4.0
        reference = [0.0, 0.0]
        loss = BoltzmannProfileLoss(soft_max=knee)
        first = loss(
            {"energy": torch.tensor([0.0, 1.0e3], dtype=torch.float64)},
            self._batch(reference, [0, 0]),
        )
        doubled = loss(
            {"energy": torch.tensor([0.0, 1.0e3 * math.sqrt(2.0)], dtype=torch.float64)},
            self._batch(reference, [0, 0]),
        )
        assert float(doubled - first) == pytest.approx(knee * math.log(2.0), rel=1e-9)

    def test_below_the_knee_the_loss_is_exactly_unchanged(self) -> None:
        """A Huber is the identity below its knee — not merely close to it.

        The regime the fit actually works in must not be distorted at all, which a
        saturating rescale would do everywhere.
        """
        reference = [0.0, 0.5]
        pred = torch.tensor([0.0, 0.6], dtype=torch.float64)
        plain = BoltzmannProfileLoss()({"energy": pred}, self._batch(reference, [0, 0]))
        soft = BoltzmannProfileLoss(soft_max=100.0)(
            {"energy": pred}, self._batch(reference, [0, 0])
        )
        torch.testing.assert_close(soft, plain)

    def test_the_gradient_is_attenuated_not_cut(self) -> None:
        """Bounded is not the same as ignored, and this is the difference.

        ``m/L`` past the knee keeps pulling an outlier back slowly. Applying
        ``m tanh(L/m)`` directly instead would give ``sech**2(L/m)``, which
        underflows to exactly zero — a frame against a wall would then never be
        corrected. A non-zero gradient here is the contract, not an accident.
        """
        reference = [0.0, 0.0]
        pred = torch.tensor([0.0, 50.0], dtype=torch.float64, requires_grad=True)
        BoltzmannProfileLoss(soft_max=4.0)(
            {"energy": pred}, self._batch(reference, [0, 0])
        ).backward()
        assert pred.grad is not None
        assert float(pred.grad[1]) != 0.0
        assert abs(float(pred.grad[1])) < 1.0  # heavily attenuated, not clipped

    def test_a_non_positive_soft_max_is_refused(self) -> None:
        """It is the knee and the only thing divided by; zero would make it inf."""
        with pytest.raises(ValueError, match="soft_max must be > 0"):
            BoltzmannProfileLoss(soft_max=0.0)


class TestForceLossOutlierBound:
    """``outlier_norm`` — the DFT tail a Class-I form cannot represent."""

    @staticmethod
    def _batch(reference, mask=None):
        atoms = {"forces": reference}
        if mask is not None:
            atoms["mask"] = torch.tensor(mask)
        return TensorDict(
            {"atoms": TensorDict(atoms, batch_size=[reference.shape[0]])}, batch_size=[]
        )

    def test_an_atom_whose_reference_force_exceeds_the_bound_stops_counting(self) -> None:
        """Fitting a 200-800 kcal/mol/A tail moves every parameter that can reach it.

        Dropping it must change the *denominator* too, or the loss would merely be
        diluted rather than the atom excluded — so the bounded loss has to equal
        the loss of the kept atoms alone.
        """
        reference = torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [900.0, 0.0, 0.0]], dtype=torch.float64
        )
        pred = torch.tensor(
            [[1.5, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        )
        bounded = force_loss("forces", outlier_norm=200.0)({"forces": pred}, self._batch(reference))
        kept = force_loss("forces")({"forces": pred[:2]}, self._batch(reference[:2]))
        torch.testing.assert_close(bounded, kept)

    def test_the_bound_multiplies_with_the_padding_mask(self) -> None:
        """Both are 'a row that should not count'; neither may cancel the other."""
        reference = torch.tensor(
            [[1.0, 0.0, 0.0], [900.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        )
        pred = torch.tensor(
            [[1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [7.0, 0.0, 0.0]], dtype=torch.float64
        )
        got = force_loss("forces", outlier_norm=200.0, weight_key="mask")(
            {"forces": pred}, self._batch(reference, [True, True, False])
        )
        only_first = force_loss("forces")({"forces": pred[:1]}, self._batch(reference[:1]))
        torch.testing.assert_close(got, only_first)

    def test_without_the_bound_every_atom_still_counts(self) -> None:
        """The default must not change: it is the behaviour every caller has."""
        reference = torch.tensor([[900.0, 0.0, 0.0]], dtype=torch.float64)
        pred = torch.zeros(1, 3, dtype=torch.float64)
        got = force_loss("forces")({"forces": pred}, self._batch(reference))
        torch.testing.assert_close(got, torch.tensor(900.0**2 / 3, dtype=torch.float64))

    def test_a_profile_that_already_fits_stays_at_zero_under_saturation(self) -> None:
        """The saturating rescale divides by the loss, so zero is its edge case.

        It is also the case that forced the denominator to be a `clamp` rather than
        a `where` against a Python float: that float becomes a host scalar tensor in
        the launch, and a CUDA kernel handed one dies in `cuPointerGetAttribute`.
        Invisible on CPU, which is why it is pinned by value here and by the
        stage-3 calibration on the device.
        """
        reference = [0.0, 1.0, 2.0]
        exact = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float64)
        loss = BoltzmannProfileLoss(soft_max=4.0)(
            {"energy": exact}, TestBoltzmannProfileSaturation._batch(reference, [0, 0, 0])
        )
        assert float(loss) == 0.0


class TestLogKnee:
    def test_below_the_knee_the_loss_and_its_gradient_are_unchanged(self) -> None:
        value = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        out = log_knee(value, 4.0)
        assert float(out.detach()) == pytest.approx(2.0)
        out.backward()
        assert float(value.grad) == pytest.approx(1.0)

    def test_above_the_knee_the_gradient_is_knee_over_loss(self) -> None:
        value = torch.tensor(16.0, dtype=torch.float64, requires_grad=True)
        out = log_knee(value, 4.0)
        out.backward()
        assert float(value.grad) == pytest.approx(4.0 / 16.0)
        doubled = torch.tensor(32.0, dtype=torch.float64)
        assert float(log_knee(doubled, 4.0) - out.detach()) == pytest.approx(4.0 * math.log(2.0))


class TestStationaryForceLoss:
    """ByteFF ``Force_MSE``: at a QM minimum the MM force is its own residual."""

    @staticmethod
    def _batch(forces: torch.Tensor, mask: torch.Tensor | None = None) -> TensorDict:
        atoms = {"forces_pred": forces}
        if mask is not None:
            atoms["mask"] = mask
        return TensorDict({"atoms": TensorDict(atoms, batch_size=[forces.shape[0]])}, batch_size=[])

    def test_a_zero_force_scores_zero(self) -> None:
        batch = self._batch(torch.zeros(3, 3))
        assert float(StationaryForceLoss()(batch, batch)) == 0.0

    def test_a_masked_ghost_row_does_not_count(self) -> None:
        """Excluded from numerator and divisor alike, by weight, never by gather."""
        forces = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [100.0, 100.0, 100.0]])
        dense = self._batch(forces[:2])
        padded = self._batch(forces, torch.tensor([1.0, 1.0, 0.0]))
        torch.testing.assert_close(
            StationaryForceLoss()(padded, padded), StationaryForceLoss()(dense, dense)
        )

    def test_without_a_mask_it_is_the_mean_squared_component(self) -> None:
        forces = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7.0
        batch = self._batch(forces)
        torch.testing.assert_close(StationaryForceLoss()(batch, batch), forces.pow(2).mean())

    def test_soft_max_knees_the_mean(self) -> None:
        forces = torch.full((2, 3), 3.0)  # mean squared component 9
        batch = self._batch(forces)
        torch.testing.assert_close(
            StationaryForceLoss(soft_max=4.0)(batch, batch), log_knee(forces.pow(2).mean(), 4.0)
        )

    def test_a_missing_prediction_is_refused(self) -> None:
        batch = TensorDict(
            {"atoms": TensorDict({"mask": torch.ones(2)}, batch_size=[2])}, batch_size=[]
        )
        with pytest.raises(KeyError, match="forces_pred"):
            StationaryForceLoss()(batch, batch)

    def test_skip_above_drops_the_whole_molecule_whose_nonbonded_force_is_large(self) -> None:
        """ByteFF ``skip_large_nb``: one clashing atom takes its molecule out, numerator
        and divisor alike; the other molecule scores as if alone."""
        forces = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [4.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        nonbonded = torch.zeros(4, 3)
        nonbonded[2, 1] = 100.0
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "forces_pred": forces,
                        "forces_nonbonded": nonbonded,
                        "batch": torch.tensor([0, 0, 1, 1]),
                        "mask": torch.ones(4),
                    },
                    batch_size=[4],
                ),
                "graphs": TensorDict({"num_atoms": torch.tensor([2, 2])}, batch_size=[2]),
            },
            batch_size=[],
        )
        alone = self._batch(forces[:2])
        torch.testing.assert_close(
            StationaryForceLoss(skip_above=50.0)(batch, batch),
            StationaryForceLoss()(alone, alone),
        )

    def test_skip_above_without_the_nonbonded_column_is_refused(self) -> None:
        batch = self._batch(torch.ones(2, 3))
        with pytest.raises(KeyError, match="forces_nonbonded"):
            StationaryForceLoss(skip_above=1.0)(batch, batch)


class TestMeanTermEnergyLoss:
    """ByteFF ``Bond_Energy`` / ``Angle_Energy``: bonded strain on a QM minimum, per live term."""

    @staticmethod
    def _batch(
        energy: torch.Tensor,
        bonds_mask: torch.Tensor,
        angles_mask: torch.Tensor,
        graphs_mask: torch.Tensor | None = None,
    ) -> TensorDict:
        graphs = {"energy_pred": energy}
        if graphs_mask is not None:
            graphs["mask"] = graphs_mask
        return TensorDict(
            {
                "graphs": TensorDict(graphs, batch_size=[energy.shape[0]]),
                "bonds": TensorDict({"mask": bonds_mask}, batch_size=[bonds_mask.shape[0]]),
                "angles": TensorDict({"mask": angles_mask}, batch_size=[angles_mask.shape[0]]),
            },
            batch_size=[],
        )

    def test_divides_the_energy_by_the_live_terms_of_every_namespace(self) -> None:
        batch = self._batch(
            torch.tensor([3.0, 5.0]), torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1.0, 0.0])
        )
        torch.testing.assert_close(MeanTermEnergyLoss()(batch, batch), torch.tensor(8.0 / 3.0))

    def test_a_dead_graph_slot_is_weighted_out(self) -> None:
        batch = self._batch(
            torch.tensor([3.0, 500.0]), torch.ones(2), torch.ones(2), torch.tensor([1.0, 0.0])
        )
        torch.testing.assert_close(MeanTermEnergyLoss()(batch, batch), torch.tensor(3.0 / 4.0))

    def test_without_masks_every_row_counts(self) -> None:
        batch = TensorDict(
            {
                "graphs": TensorDict({"energy_pred": torch.tensor([6.0])}, batch_size=[1]),
                "bonds": TensorDict({"k": torch.zeros(2)}, batch_size=[2]),
                "angles": TensorDict({"k": torch.zeros(1)}, batch_size=[1]),
            },
            batch_size=[],
        )
        torch.testing.assert_close(MeanTermEnergyLoss()(batch, batch), torch.tensor(2.0))

    def test_a_missing_namespace_is_refused_by_name(self) -> None:
        batch = TensorDict(
            {
                "graphs": TensorDict({"energy_pred": torch.tensor([1.0])}, batch_size=[1]),
                "bonds": TensorDict({"mask": torch.ones(1)}, batch_size=[1]),
            },
            batch_size=[],
        )
        with pytest.raises(KeyError, match="angles"):
            MeanTermEnergyLoss()(batch, batch)

    def test_a_missing_prediction_is_refused(self) -> None:
        batch = TensorDict(
            {"graphs": TensorDict({"num_atoms": torch.tensor([1])}, batch_size=[1])}, batch_size=[]
        )
        with pytest.raises(KeyError, match="energy_pred"):
            MeanTermEnergyLoss()(batch, batch)
