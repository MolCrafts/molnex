"""Tests for Polar one-body field update and field readout."""

from __future__ import annotations

import cuequivariance as cue
import pytest
import torch

from molix import config
from molrep.interaction.polar.field import (
    AgnosticEmbeddedOneBodyVariableUpdate,
    OneBodyMLPFieldReadout,
)
from molrep.utils.equivariance import rotate_irreps_features_simple


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


_FEATS = "4x0e+4x1o"
_ATTRS = "2x0e"
_CHARGES = "2x0e+2x1o"
_POTENTIAL = "2x0e+2x1o"
_FEAT_DIM = 16  # 4*1 + 4*3
_CHARGE_DIM = 8  # 2*1 + 2*3
_OUT_DIM = 10  # charges ⊕ 2x0e


def _update():
    return AgnosticEmbeddedOneBodyVariableUpdate(
        node_attrs_irreps=_ATTRS,
        node_feats_irreps=_FEATS,
        potential_irreps=_POTENTIAL,
        charges_irreps=_CHARGES,
    )


def _readout():
    return OneBodyMLPFieldReadout(
        node_feats_irreps=_FEATS,
        potential_irreps=_POTENTIAL,
        charges_irreps=_CHARGES,
    )


class TestAgnosticEmbeddedOneBodyVariableUpdate:
    """Test the Polar non-SCF one-body field update."""

    def test_output_is_charges_plus_two_scalars(self):
        """Dump: charges 2x0e+2x1o (8) ⊕ 2x0e Fukui → 10."""
        torch.manual_seed(0)
        n = 3
        out = _update()(
            torch.randn(n, 2, dtype=torch.float64),
            torch.randn(n, _FEAT_DIM, dtype=torch.float64),
            torch.randn(n, _CHARGE_DIM, dtype=torch.float64),
            torch.randn(n, _CHARGE_DIM, dtype=torch.float64),
        )
        assert out.shape == (n, _OUT_DIM)

    def test_submodule_names_match_official_remap(self):
        """Keys the 05 polar_remap will copy by name."""
        block = _update()
        assert {
            "potential_embedding",
            "source_embedding",
            "dot_products",
            "nonlinearity",
            "tp_out",
            "readout",
        } <= set(block._modules)
        assert {
            "potential_linear",
            "node_feats_linear",
            "charge_embedding",
        } <= set(block.potential_embedding._modules)
        assert {"linear_1", "linear_mid", "linear_2"} <= set(block.readout._modules)

    def test_radial_mlp_is_official_width(self):
        """RadialMLP([2 n_0e, 64, 64, 64, n_0e]); tiny n_0e=4."""
        assert _update().nonlinearity.hs == [8, 64, 64, 64, 4]

    def test_readout_gate_width_is_32_times_sh(self):
        """MLP = 32 × Y_{lmax=1} → 32x0e+32x1o (128); gate input 160."""
        readout = _update().readout
        assert cue.Irreps("O3", "32x0e+32x1o").dim == 128
        assert readout.linear_1.irreps_out.dim == 160

    def test_even_scalars_are_z_rotation_invariant(self):
        """0e slices of charges⊕Fukui stay put under a joint z-rotation."""
        torch.manual_seed(1)
        block = _update()
        n = 4
        node_attrs = torch.randn(n, 2, dtype=torch.float64)
        node_feats = torch.randn(n, _FEAT_DIM, dtype=torch.float64)
        potential = torch.randn(n, _CHARGE_DIM, dtype=torch.float64)
        charges = torch.randn(n, _CHARGE_DIM, dtype=torch.float64)
        out = block(node_attrs, node_feats, potential, charges)
        theta = 0.5
        c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        rot = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        out_rot = block(
            node_attrs,
            rotate_irreps_features_simple(node_feats, rot, _FEATS),
            rotate_irreps_features_simple(potential, rot, _POTENTIAL),
            rotate_irreps_features_simple(charges, rot, _CHARGES),
        )
        # unsimplified 2x0e+2x1o+2x0e: 0e at [0:2] and [8:10]
        assert torch.allclose(out[:, :2], out_rot[:, :2], atol=1e-8)
        assert torch.allclose(out[:, 8:], out_rot[:, 8:], atol=1e-8)


class TestOneBodyMLPFieldReadout:
    """Test the Polar post-induction field energy readout."""

    def test_output_is_per_atom_scalar(self):
        """Squeeze of RadialMLP(..., 1) → (N,) eV/atom."""
        torch.manual_seed(0)
        n = 3
        out = _readout()(
            torch.randn(n, _FEAT_DIM, dtype=torch.float64),
            torch.randn(n, _CHARGE_DIM, dtype=torch.float64),
            torch.randn(n, _CHARGE_DIM, dtype=torch.float64),
            torch.randn(n, _CHARGE_DIM, dtype=torch.float64),
        )
        assert out.shape == (n,)

    def test_uses_sum_of_induced_and_reference_charges(self):
        """q_in = charges_induced + charges_0; swapping them must not change E."""
        torch.manual_seed(2)
        block = _readout()
        n = 2
        feats = torch.randn(n, _FEAT_DIM, dtype=torch.float64)
        field = torch.randn(n, _CHARGE_DIM, dtype=torch.float64)
        q0 = torch.randn(n, _CHARGE_DIM, dtype=torch.float64)
        qi = torch.randn(n, _CHARGE_DIM, dtype=torch.float64)
        a = block(feats, field, q0, qi)
        b = block(feats, field, qi, q0)
        assert torch.allclose(a, b, atol=1e-12)

    def test_submodule_names_match_official_remap(self):
        """linear_up_q/v, SparseUvu dots, RadialMLP."""
        block = _readout()
        assert {
            "linear_up_q",
            "linear_up_v",
            "dot_products_q",
            "dot_products_v",
            "mlp",
        } <= set(block._modules)

    def test_mlp_is_official_width(self):
        """RadialMLP([2 n_0e, 128, 128, 128, 1]); tiny n_0e=4."""
        assert _readout().mlp.hs == [8, 128, 128, 128, 1]
