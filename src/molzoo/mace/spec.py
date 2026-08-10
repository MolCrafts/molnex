"""Validated configurations for the MACE foundation-model variants.

Three pydantic models: :class:`MACESpec` carries everything the two shipped
variants share, :class:`MACEMatpesSpec` and :class:`MACEOMolSpec` add the
variant-only fields and pin the defaults of the constructors they replace
(``MACEMatpes.__init__`` / ``MACEOMol.__init__``).

Units follow the rest of the repo: positions and ``r_max`` in Å, energies
(``atomic_energies``, ``shift``) in eV, ``atomic_energies`` per atom.

This module deliberately imports **nothing** but ``typing`` and ``pydantic``.
A configuration must stay readable — from a CLI, a checkpoint manifest, or a
test — without paying for torch and the cuEquivariance stack, so tensor
construction (``atomic_energies`` → buffer) belongs to
:class:`~molzoo.mace.encoder.MACEEncoder`, not here.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
    Batatia et al. "A foundation model for atomistic materials chemistry"
    (MACE-MP-0). https://arxiv.org/abs/2401.00096
    Kaplan et al. "A foundational potential energy surface dataset for
    materials" (MatPES). https://arxiv.org/abs/2503.04070
    Levine et al. "The Open Molecules 2025 (OMol25) Dataset, Evaluations, and
    Models" https://arxiv.org/abs/2505.08762
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MACESpec(BaseModel):
    """Shared configuration of the MACE foundation-model backbone.

    The five ``Literal`` switches at the bottom select which stack
    :class:`~molzoo.mace.encoder.MACEEncoder` builds. They have no default on
    this base class — a bare ``MACESpec`` describes no shipped model, so the
    caller must say which one it means (the subclasses do exactly that).

    Attributes:
        atomic_numbers: Element table (z-table), strictly ascending. The
            encoder looks ``Z`` up with ``torch.searchsorted``, which silently
            snaps to a neighbouring row on an unordered table.
        atomic_energies: Per-element reference energies ``E0`` in eV/atom, in
            ``atomic_numbers`` order. A plain ``list`` — see the module
            docstring for why the tensor is built by the encoder.
        r_max: Radial cutoff in Å.
        num_bessel: Number of Bessel radial basis functions.
        num_polynomial_cutoff: Polynomial cutoff exponent ``p`` (also the ZBL
            envelope exponent, matching MACE).
        l_max: Maximum spherical-harmonics order.
        num_features: Scalar channel multiplicity (``hidden_irreps`` 0e count).
        num_interactions: Number of interaction/product layers.
        correlation: Body-order correlation of the symmetric contraction.
        mlp_dim: Hidden width of the final non-linear readout.
        scale: ``atomic_inter_scale`` — multiplies the interaction energy (eV).
        shift: ``atomic_inter_shift`` — per-atom energy offset in eV.
        use_fallback: Pure-torch cuEquivariance path (``True``) or fused
            kernels (``False``). Fused kernels need a GPU and the
            ``cuequivariance-ops-torch`` wheel.
        interaction: ``"density"`` (MACE-MP/MatPES density-normalised
            interactions) or ``"residual"`` (OMOL non-linear residual ones).
        readout: ``"per_layer"`` (one readout per layer, summed) or
            ``"final"`` (a single readout on the last layer).
        distance_transform: ``"agnesi"`` applies the Agnesi radial transform
            before the Bessel basis; ``"none"`` feeds raw distances.
        pair_repulsion: ``"zbl"`` adds the ZBL short-range pair term.
        conditioning: ``"charge_spin"`` adds the total-charge / total-spin
            joint embedding to the initial node features.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    atomic_numbers: list[int] = Field(..., min_length=1)
    atomic_energies: list[float] = Field(..., min_length=1)

    r_max: float = Field(6.0, gt=0.0)
    num_bessel: int = Field(8, gt=0)
    num_polynomial_cutoff: int = Field(5, gt=0)
    l_max: int = Field(3, ge=0)
    num_features: int = Field(128, gt=0)
    num_interactions: int = Field(2, gt=0)
    correlation: int = Field(2, ge=1)
    mlp_dim: int = Field(16, gt=0)
    scale: float = 1.0
    shift: float = 0.0
    use_fallback: bool = False

    interaction: Literal["density", "residual"]
    readout: Literal["per_layer", "final"]
    distance_transform: Literal["none", "agnesi"]
    pair_repulsion: Literal["none", "zbl"]
    conditioning: Literal["none", "charge_spin"]

    @model_validator(mode="after")
    def _check_element_table(self) -> MACESpec:
        """Reject element tables that would mis-index ``E0`` or the one-hot.

        Raises:
            ValueError: If ``atomic_energies`` has a different length than
                ``atomic_numbers``, or ``atomic_numbers`` is not strictly
                ascending (unordered or duplicated).
        """
        if len(self.atomic_energies) != len(self.atomic_numbers):
            raise ValueError(
                f"atomic_energies has {len(self.atomic_energies)} entries but "
                f"atomic_numbers has {len(self.atomic_numbers)} — one E0 per element"
            )
        pairs = zip(self.atomic_numbers, self.atomic_numbers[1:])
        if any(previous >= following for previous, following in pairs):
            raise ValueError(
                f"atomic_numbers must be strictly ascending (searchsorted lookup), "
                f"got {self.atomic_numbers}"
            )
        return self


class MACEMatpesSpec(MACESpec):
    """Configuration of the MACE-MatPES foundation model.

    Defaults reproduce ``MACEMatpes.__init__`` of the pre-cutover flat module
    (``src/molzoo/mace_matpes.py``, deleted in 06-wire; see git history),
    including its materialised ``radial_mlp`` of ``[64, 64, 64]``.

    Attributes:
        max_hidden_l: Highest ``l`` carried in the node state between layers
            (1 for the shipped MatPES models, i.e. ``128x0e+128x1o``).
        radial_mlp: Hidden widths of the radial weight MLP.
    """

    max_hidden_l: int = Field(1, ge=0)
    radial_mlp: list[int] = Field(default_factory=lambda: [64, 64, 64], min_length=1)

    num_bessel: int = Field(10, gt=0)
    correlation: int = Field(3, ge=1)

    interaction: Literal["density", "residual"] = "density"
    readout: Literal["per_layer", "final"] = "per_layer"
    distance_transform: Literal["none", "agnesi"] = "agnesi"
    pair_repulsion: Literal["none", "zbl"] = "zbl"
    conditioning: Literal["none", "charge_spin"] = "none"

    @model_validator(mode="after")
    def _check_residual_layer(self) -> MACEMatpesSpec:
        """MatPES's stack is one density layer plus at least one residual one.

        Raises:
            ValueError: If ``num_interactions`` is below 2.
        """
        if self.num_interactions < 2:
            raise ValueError(f"num_interactions must be at least 2, got {self.num_interactions}")
        return self


class MACEOMolSpec(MACESpec):
    """Configuration of the MACE-OMOL foundation model.

    Defaults reproduce ``MACEOMol.__init__`` of the pre-cutover flat module
    (``src/molzoo/mace_omol.py``, deleted in 06-wire; see git history).
    ``use_fallback`` is not a constructor argument there — the flat model
    hard-codes the fused cuEquivariance path, i.e. ``False``.

    Attributes:
        edge_channels: Per-``l`` channel count of the mid-layer edge irreps and
            the radial-MLP hidden width (a deliberate bottleneck below
            ``num_features``).
        charge_classes: Embedding rows for the total-charge conditioning.
        charge_offset: Index offset applied to total charge (charge −100 → row 0).
        spin_classes: Embedding rows for the total-spin conditioning.
        spin_offset: Index offset applied to total spin.
    """

    edge_channels: int = Field(128, gt=0)
    charge_classes: int = Field(201, gt=0)
    charge_offset: int = Field(100, ge=0)
    spin_classes: int = Field(101, gt=0)
    spin_offset: int = Field(0, ge=0)

    num_features: int = Field(1024, gt=0)
    num_interactions: int = Field(3, gt=0)

    interaction: Literal["density", "residual"] = "residual"
    readout: Literal["per_layer", "final"] = "final"
    distance_transform: Literal["none", "agnesi"] = "none"
    pair_repulsion: Literal["none", "zbl"] = "none"
    conditioning: Literal["none", "charge_spin"] = "charge_spin"

    @model_validator(mode="after")
    def _check_angular_order(self) -> MACEOMolSpec:
        """OMOL's mid-layer edge irreps span ``range(l_max)`` — empty at ``l_max=0``.

        Raises:
            ValueError: If ``l_max`` is below 1.
        """
        if self.l_max < 1:
            raise ValueError(f"l_max must be at least 1, got {self.l_max}")
        return self
