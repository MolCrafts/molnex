"""Validated configuration of the MACE conformer encoder.

Torch-free. :class:`MaceConformerEncoderSpec` is the MACE body's fields
(:class:`~molrep.encoders.mace.spec.MACEEncoderSpec`, a MACE-OFF-shaped stack:
an agnostic first layer, residual layers after it, no conditioning, no distance
transform) plus the readout that maps its invariants onto the force-field
heads' interface. Spec: ``.claude/specs/rondo-enc-02-mace.md``.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
    Kovács et al. "MACE-OFF: Short-Range Transferable Machine Learning Force
    Fields for Organic Molecules" JACS 2025. https://arxiv.org/abs/2312.15211
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from molrep.encoders.mace.spec import BodySchedule, MACEEncoderSpec

__all__ = ["MaceConformerEncoderSpec"]

#: THEMol's elements (H C N O F P S Cl Br I).
THEMOL_ELEMENTS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]


class MaceConformerEncoderSpec(MACEEncoderSpec):
    """A MACE-OFF-shaped body and the readout onto the heads' features.

    The body defaults are MACE-OFF24's small shape on THEMol's elements, at
    ``l_max = 2``: the encoder parameterizes a molecule once, so it trades
    angular order for nothing at MD time. ``avg_num_neighbors`` is the THEMol
    mean within ``r_max`` (spec 02 §3).

    Attributes:
        max_hidden_l: Highest ``l`` carried in the node state between layers.
        radial_mlp: Hidden widths of the radial weight MLP.
        width: Width of the atom features handed to the heads (ByteFF-26's 256).
        bond_width: Width of the bond features (ByteFF-26's ``edge_dim``, 256).
        hidden: Hidden width of the atom and bond readout MLPs.
        charge_dim: Width of the formal-charge embedding. Geometry does not
            determine a charge state; ByteFF-26 reads it from the graph too.
        bond_lengths: ``(low, high, count)`` of the Gaussian basis on the
            reference bond length, Å.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    atomic_numbers: list[int] = Field(default_factory=lambda: list(THEMOL_ELEMENTS), min_length=1)
    r_max: float = Field(5.0, gt=0.0)
    l_max: int = Field(2, ge=1)
    num_features: int = Field(128, gt=0)
    num_interactions: int = Field(2, gt=0)
    correlation: int = Field(3, ge=1)
    avg_num_neighbors: float | None = Field(16.0, gt=0.0)

    interaction: Literal["density", "residual", "agnostic"] = "agnostic"
    distance_transform: Literal["none", "agnesi"] = "none"
    conditioning: Literal["none", "charge_spin"] = "none"

    max_hidden_l: int = Field(1, ge=0)
    radial_mlp: tuple[int, ...] = (64, 64, 64)

    width: int = Field(256, gt=0)
    bond_width: int = Field(256, gt=0)
    hidden: int = Field(256, gt=0)
    charge_dim: int = Field(8, gt=0)
    bond_lengths: tuple[float, float, int] = (0.8, 2.4, 16)

    def body_schedule(self) -> BodySchedule:
        """MACE-OFF's schedule: ``max_hidden_l``, the radial MLP, per-element products."""
        return BodySchedule(
            hidden_l=self.max_hidden_l,
            edge_channels=self.num_features,
            radial_mlp=tuple(self.radial_mlp),
            keep_last_layer_irreps=False,
            shared_product=False,
            residual_first=False,
        )
