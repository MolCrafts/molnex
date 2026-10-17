"""The style registry: one potential class per molrs style."""

from __future__ import annotations

import pytest
import torch.nn as nn

from molpot.potentials.registry import register, registered


def test_the_molrs_class_i_styles_are_registered():
    assert set(registered()) >= {
        "bond:harmonic",
        "angle:harmonic",
        "dihedral:periodic",
        "dihedral:opls",
        "improper:cvff",
        "pair:lj/cut",
        "pair:coul/cut",
    }


def test_a_second_claim_on_a_style_is_refused():
    class Claimant(nn.Module):
        style = "bond:harmonic"

    with pytest.raises(ValueError, match="bond:harmonic"):
        register(Claimant)
