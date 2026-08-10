"""MACE-only interaction blocks.

Home for the interaction-layer building blocks that exist solely to serve the
MACE family of encoders (plain MACE, MACE-MP / MatPES, MACE-OMOL). These are
**pure geometric-convolution blocks**: they consume node/edge features and emit
node/edge features, and never compute an energy or a force — energy heads and
force derivation stay in ``molpot``.

This mirrors the role of :mod:`molrep.interaction.pinet` for the PiNet family:
one namespace per model family, with the genuinely shared blocks
(``contraction`` / ``radial`` / ``gate`` / ``linear`` / ``aggregation`` /
``product``) remaining at the :mod:`molrep.interaction` top level.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from molrep.interaction.mace.block import InteractionBlock, InteractionSpec
from molrep.interaction.mace.conv import ConvTP, ConvTPSpec
from molrep.interaction.mace.density import (
    SKIP_TP_METHOD,
    DensityInteraction,
    DensityResidualInteraction,
)

__all__ = [
    "ConvTP",
    "ConvTPSpec",
    "InteractionBlock",
    "InteractionSpec",
    "DensityInteraction",
    "DensityResidualInteraction",
    "SKIP_TP_METHOD",
]
