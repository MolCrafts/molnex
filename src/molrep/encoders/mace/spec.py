"""Validated configuration of the MACE encoder.

Torch-free: imports only ``typing`` and ``pydantic``. :class:`MACEEncoderSpec`
holds what :class:`~molrep.encoders.mace.encoder.MACEEncoder` reads; the irreps
schedule of the body (:class:`BodySchedule`) is stated by each shipped spec
(:mod:`molzoo.mace.spec`), which also adds the energy readout's fields.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["BodySchedule", "MACEEncoderSpec"]


class BodySchedule(NamedTuple):
    """Irreps schedule of one feature body, independent of any energy head.

    Attributes:
        hidden_l: Highest ``l`` kept in the node state between layers.
        edge_channels: Per-``l`` multiplicity of the mid-layer edge irreps on
            a residual stack. A density stack builds those irreps from
            ``num_features`` and ignores this count.
        radial_mlp: Hidden widths of the radial weight MLP.
        keep_last_layer_irreps: Last product keeps ``hidden`` irreps. When
            false, the last layer is scalar.
        shared_product: One symmetric-contraction weight set for every
            element. When false, the product carries one set per element.
        residual_first: The first layer is a residual block too, so its
            product carries a skip connection (MACE-OFF23, MACE-LES-OFF).
    """

    hidden_l: int
    edge_channels: int
    radial_mlp: tuple[int, ...]
    keep_last_layer_irreps: bool
    shared_product: bool
    residual_first: bool


class MACEEncoderSpec(BaseModel):
    """Configuration of the MACE body, independent of any energy head.

    The three ``Literal`` switches select which stack
    :class:`~molrep.encoders.mace.encoder.MACEEncoder` builds. They have no
    default on this class — a bare encoder spec describes no shipped model, so
    the caller must say which one it means.

    Attributes:
        atomic_numbers: Element table (z-table), strictly ascending. The
            encoder looks ``Z`` up with ``torch.searchsorted``, which silently
            snaps to a neighbouring row on an unordered table.
        r_max: Radial cutoff in Å.
        num_bessel: Number of Bessel radial basis functions.
        num_polynomial_cutoff: Polynomial cutoff exponent ``p``.
        l_max: Maximum spherical-harmonics order.
        num_features: Scalar channel multiplicity (``hidden_irreps`` 0e count).
        num_interactions: Number of interaction/product layers.
        correlation: Body-order correlation of the symmetric contraction.
        avg_num_neighbors: Message normalisation of the ``"agnostic"``
            interactions; required by them, ignored by the others.
        charge_classes: Embedding rows of the total-charge conditioning.
        charge_offset: Index offset applied to the total charge.
        spin_classes: Embedding rows of the total-spin conditioning.
        spin_offset: Index offset applied to the total spin.
        interaction: ``"density"`` (density-normalised interactions),
            ``"agnostic"`` (normalised by ``avg_num_neighbors``) or
            ``"residual"`` (non-linear residual ones).
        distance_transform: ``"agnesi"`` applies the Agnesi radial transform
            before the Bessel basis; ``"none"`` feeds raw distances.
        conditioning: ``"charge_spin"`` adds the total-charge / total-spin
            joint embedding to the initial node features; the four
            ``*_classes`` / ``*_offset`` fields are then required, and are
            rejected without it.
    """

    model_config = ConfigDict(extra="forbid")

    atomic_numbers: list[int] = Field(..., min_length=1)
    r_max: float = Field(6.0, gt=0.0)
    num_bessel: int = Field(8, gt=0)
    num_polynomial_cutoff: int = Field(5, gt=0)
    l_max: int = Field(3, ge=0)
    num_features: int = Field(128, gt=0)
    num_interactions: int = Field(2, gt=0)
    correlation: int = Field(2, ge=1)
    avg_num_neighbors: float | None = Field(None, gt=0.0)
    charge_classes: int | None = Field(None, gt=0)
    charge_offset: int | None = Field(None, ge=0)
    spin_classes: int | None = Field(None, gt=0)
    spin_offset: int | None = Field(None, ge=0)

    interaction: Literal["density", "residual", "agnostic"]
    distance_transform: Literal["none", "agnesi"]
    conditioning: Literal["none", "charge_spin"]

    @model_validator(mode="after")
    def _check_element_table(self) -> Self:
        """Reject element tables that would mis-index the one-hot.

        Raises:
            ValueError: If ``atomic_numbers`` is not strictly ascending
                (unordered or duplicated), or an ``"agnostic"`` stack has no
                ``avg_num_neighbors``.
        """
        pairs = zip(self.atomic_numbers, self.atomic_numbers[1:])
        if any(previous >= following for previous, following in pairs):
            raise ValueError(
                f"atomic_numbers must be strictly ascending (searchsorted lookup), "
                f"got {self.atomic_numbers}"
            )
        if self.interaction == "agnostic" and self.avg_num_neighbors is None:
            raise ValueError('interaction="agnostic" requires avg_num_neighbors')
        return self

    @model_validator(mode="after")
    def _check_conditioning_fields(self) -> Self:
        """The four conditioning sizes exist iff the stack is charge/spin conditioned.

        Raises:
            ValueError: If ``conditioning="charge_spin"`` leaves any of them
                ``None``, or ``conditioning="none"`` sets any of them.
        """
        sizes = (self.charge_classes, self.charge_offset, self.spin_classes, self.spin_offset)
        if self.conditioning == "charge_spin" and any(size is None for size in sizes):
            raise ValueError(
                'conditioning="charge_spin" needs charge_classes, charge_offset, '
                "spin_classes and spin_offset"
            )
        if self.conditioning == "none" and any(size is not None for size in sizes):
            raise ValueError(
                'conditioning="none" takes no charge_classes / charge_offset / '
                "spin_classes / spin_offset"
            )
        return self

    def body_schedule(self) -> BodySchedule:
        """The irreps schedule of this body — every shipped spec states its own.

        Raises:
            NotImplementedError: On a spec that describes no shipped model.
        """
        raise NotImplementedError(f"{type(self).__name__} states no body schedule")
