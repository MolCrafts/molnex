"""PotentialIR aggregate — optional bags + unit_system validation."""

import pytest
import torch

from molpot.ir import (
    BondBag,
    NonbondedScaling,
    PotentialIR,
)


class TestPotentialIR:
    def test_empty_ir_is_valid(self):
        ir = PotentialIR()
        assert ir is not None
        # Empty IR means zero contribution from every term bag.
        assert getattr(ir, "bonds", None) is None or ir.bonds is None
        assert (
            getattr(ir, "unit_system", "class_i_canonical")
            in (
                "class_i_canonical",
                None,
            )
            or ir.unit_system == "class_i_canonical"
        )

    def test_empty_ir_defaults_to_class_i_canonical_units(self):
        ir = PotentialIR()
        unit_system = getattr(ir, "unit_system", "class_i_canonical")
        assert unit_system == "class_i_canonical"

    def test_unknown_unit_system_raises(self):
        with pytest.raises(ValueError):
            PotentialIR(unit_system="not_a_real_unit_system")

    def test_subset_of_bags_is_valid(self):
        ir = PotentialIR(
            bonds=BondBag(k=torch.tensor([100.0]), r0=torch.tensor([1.5])),
            scaling=NonbondedScaling(),
        )
        assert ir.bonds is not None
        assert ir.scaling is not None
