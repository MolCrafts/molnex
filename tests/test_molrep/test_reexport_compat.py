"""Back-compat guard for the mace-subpackage-restructure-01 re-export shims.

Every legacy import path must resolve to the *same object* as its new
``mace`` home (not a re-implementation), and the two package ``__all__``
lists must stay byte-identical to the pre-move lists so downstream
``from molrep.interaction import X`` keeps working.

Scope is the degraded chain step: ``residual`` / ``product_basis`` /
``element`` are deliberately NOT moved here (deferred to -01b), so they
are absent from these assertions on purpose.
"""

import importlib

import molrep.interaction
import molrep.interaction.density
import molrep.interaction.mace.conv
import molrep.interaction.mace.density
import molrep.interaction.product
import molrep.readout
import molrep.readout.mace
import molrep.readout.product
import molrep.readout.scalar

# Pre-move lists, copied verbatim from src/molrep/{interaction,readout}/__init__.py
# at parent commit cf60f99. The move must not add, drop, or reorder a name.
EXPECTED_INTERACTION_ALL = [
    "DensityInteraction",
    "DensityResidualInteraction",
    "GatedNonlinearity",
    "ResidualInteraction",
    "EquivariantProductBasis",
    "RadialMLP",
    "MessageAggregation",
    "MessageAggregationSpec",
    "EquivariantLinear",
    "ConvTP",
    "ConvTPSpec",
    "irreps_from_l_max",
    "sh_irreps_from_l_max",
    "SymmetricContraction",
    "SymmetricContractionSpec",
    "ElementUpdate",
    "ElementUpdateSpec",
    "RadialWeightMLP",
    "RadialWeightMLPSpec",
]

EXPECTED_READOUT_ALL = [
    "masked_sum_pooling",
    "masked_mean_pooling",
    "BasisProjection",
    "BasisProjectionSpec",
    "ProductHead",
    "ProductHeadSpec",
]


class TestInteractionDensityShim:
    """``molrep.interaction.density`` re-exports ``molrep.interaction.mace.density``."""

    def test_density_interaction_is_same_object(self):
        """Legacy ``DensityInteraction`` must be the moved class itself."""
        assert (
            molrep.interaction.density.DensityInteraction
            is molrep.interaction.mace.density.DensityInteraction
        )

    def test_density_residual_interaction_is_same_object(self):
        """Legacy ``DensityResidualInteraction`` must be the moved class itself."""
        assert (
            molrep.interaction.density.DensityResidualInteraction
            is molrep.interaction.mace.density.DensityResidualInteraction
        )

    def test_skip_tp_method_constant_is_same_value(self):
        """The ``skip_tp`` method constant travels with the module."""
        assert (
            molrep.interaction.density.SKIP_TP_METHOD
            is molrep.interaction.mace.density.SKIP_TP_METHOD
        )


class TestInteractionConvShim:
    """``molrep.interaction.product`` re-exports the split-out ConvTP pair."""

    def test_conv_tp_is_same_object(self):
        """Legacy ``ConvTP`` must be the class now living in ``mace.conv``."""
        assert molrep.interaction.product.ConvTP is molrep.interaction.mace.conv.ConvTP

    def test_conv_tp_spec_is_same_object(self):
        """Legacy ``ConvTPSpec`` must be the class now living in ``mace.conv``."""
        assert molrep.interaction.product.ConvTPSpec is molrep.interaction.mace.conv.ConvTPSpec


class TestReadoutScalarShim:
    """``molrep.readout.scalar`` re-exports ``molrep.readout.mace``."""

    def test_linear_readout_is_same_object(self):
        """Legacy ``LinearReadout`` must be the moved class itself."""
        assert molrep.readout.scalar.LinearReadout is molrep.readout.mace.LinearReadout

    def test_non_linear_readout_is_same_object(self):
        """Legacy ``NonLinearReadout`` must be the moved class itself."""
        assert molrep.readout.scalar.NonLinearReadout is molrep.readout.mace.NonLinearReadout

    def test_non_linear_bias_readout_is_same_object(self):
        """Legacy ``NonLinearBiasReadout`` must be the moved class itself."""
        assert (
            molrep.readout.scalar.NonLinearBiasReadout is molrep.readout.mace.NonLinearBiasReadout
        )


class TestReadoutProductShim:
    """``molrep.readout.product`` re-exports ``molrep.readout.mace``."""

    def test_product_head_is_same_object(self):
        """Legacy ``ProductHead`` must be the moved class itself."""
        assert molrep.readout.product.ProductHead is molrep.readout.mace.ProductHead

    def test_product_head_spec_is_same_object(self):
        """Legacy ``ProductHeadSpec`` must be the moved class itself."""
        assert molrep.readout.product.ProductHeadSpec is molrep.readout.mace.ProductHeadSpec


class TestPackageExports:
    """Package-level ``__all__`` lists are frozen across the move."""

    def test_interaction_all_is_unchanged(self):
        """``molrep.interaction.__all__`` must match the pre-move list verbatim."""
        assert molrep.interaction.__all__ == EXPECTED_INTERACTION_ALL

    def test_readout_all_is_unchanged(self):
        """``molrep.readout.__all__`` must match the pre-move list verbatim."""
        assert molrep.readout.__all__ == EXPECTED_READOUT_ALL

    def test_legacy_packages_import_clean(self):
        """A cold import of every touched package raises no ImportError."""
        for name in ("molrep", "molrep.interaction", "molrep.readout"):
            assert importlib.import_module(name) is not None
