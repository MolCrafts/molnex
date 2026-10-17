"""Validated configurations for the MACE foundation-model variants.

:class:`MACESpec` is the shared body configuration; every shipped product has
a subclass that pins its defaults (:class:`MACEMatpesSpec`,
:class:`MACEOMolSpec`, :class:`MACEOffSpec`, :class:`MACELesSpec`,
:class:`MACEPolarSpec`). A spec is the only argument a product constructor
takes — ``MaceMatpes(MACEMatpesSpec(...))`` — so a default exists in exactly
one place.

:class:`MaceOfficialConfig` is the typed reading of an official MACE archive's
hyper-parameters (what :meth:`molzoo.mace.utils.MaceArchive.read` returns next
to the weights). Each spec's ``from_official`` builds itself from one and holds
that family's rejection rules, so a checkpoint from another family cannot be
loaded onto a model whose frozen constants it was never fitted with.

Units follow the rest of the repo: positions and ``r_max`` in Å, energies
(``atomic_energies``, ``shift``) in eV, ``atomic_energies`` per atom.

This module deliberately imports **nothing** but ``re``, ``typing`` and
``pydantic``. A configuration must stay readable — from a CLI, a checkpoint
manifest, or a test — without paying for torch and the cuEquivariance stack.
The reference table becomes the ``atomic_energies`` buffer in
:class:`molzoo.mace.potential.MacePotential`, not here.

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
    Cheng, B. "Latent Ewald summation for machine learning of long-range
    interactions." npj Comput. Mater. 11, 80 (2025).
    https://doi.org/10.1038/s41524-025-01577-7
    Batatia et al. "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model
    for Molecular Chemistry" arXiv:2602.19411.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, NamedTuple, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

#: One term of an official irreps string, e.g. ``"128x1o"`` — multiplicity,
#: angular order ``l``, parity. Parity is fixed by ``l`` in every MACE
#: configuration, so only the first two groups are read.
_IRREPS_TERM = re.compile(r"(\d+)x(\d+)([eo])")


def parse_irreps(text: str, key: str) -> tuple[int, int]:
    """Scalar multiplicity and highest angular order of an official irreps string.

    ``"128x0e+128x1o"`` → ``(128, 1)``; ``"16x0e"`` → ``(16, 0)``. Upstream MACE
    states channel widths this way, so a checkpoint's ``num_features`` /
    ``max_hidden_l`` / ``mlp_dim`` are read from its config instead of being
    assumed — a differently sized checkpoint must never load into a default
    model.

    Args:
        text: Irreps string from the official config.
        key: Config key it came from, for the error message.

    Returns:
        ``(multiplicity of the scalar 0e term, highest l over all terms)``.

    Raises:
        ValueError: If a term is not ``<mul>x<l><parity>``, or the string has
            no scalar term — the channel width would then be a guess.
    """
    terms: list[tuple[int, int]] = []
    for term in str(text).split("+"):
        matched = _IRREPS_TERM.fullmatch(term.strip())
        if matched is None:
            raise ValueError(f"config key {key} is not an irreps string: {text!r} (term {term!r})")
        terms.append((int(matched.group(1)), int(matched.group(2))))
    scalars = [multiplicity for multiplicity, order in terms if order == 0]
    if not scalars:
        raise ValueError(
            f"config key {key} has no scalar (0e) term: {text!r} — the channel "
            "width is its multiplicity and there is no default"
        )
    return scalars[0], max(order for _, order in terms)


def _present(**fields: Any) -> dict[str, Any]:
    """The given fields without the ``None`` ones (an absent official field keeps the default)."""
    return {name: value for name, value in fields.items() if value is not None}


class LesOfficialConfig(BaseModel):
    """The LES block of an official MACE-LES archive.

    Attributes:
        ewald: Upstream class name of the Ewald summation (``"Ewald"`` is the
            charge-only legacy one this port implements).
        sigma: Gaussian width of the latent charges in Å.
        dl: Reciprocal-grid spacing in Å.
        norm_factor: LES's ``norm_factor`` in eV·Å/e².
        remove_self_interaction: Subtract the Gaussian self-energy.
        output_scale: Factor on the summed charge readouts.
        use_atomwise, use_fixed_atomic_charges, use_epsilon_r_scaling,
        use_dipoles, use_quads, use_induced_charges, use_induced_dipoles:
            Upstream switches this port does not implement; any ``True`` is
            rejected by :meth:`MACELesSpec.from_official`.
    """

    model_config = ConfigDict(extra="ignore")

    ewald: str
    sigma: float
    dl: float
    norm_factor: float
    remove_self_interaction: bool
    output_scale: float = 1.0
    use_atomwise: bool = False
    use_fixed_atomic_charges: bool = False
    use_epsilon_r_scaling: bool = False
    use_dipoles: bool = False
    use_quads: bool = False
    use_induced_charges: bool = False
    use_induced_dipoles: bool = False


class MaceOfficialConfig(BaseModel):
    """Hyper-parameters read from an official MACE archive, typed.

    Only the fields a ``from_official`` reads are declared; everything else in
    an archive's config is ignored. Upstream spellings are accepted as aliases
    (``max_ell`` / ``l_max``, ``atomic_inter_scale`` / ``scale``, ``radial_MLP``
    / ``radial_mlp``, ``include_electrostatic_self_interaction``, ``field_si``).
    A field the archive does not carry is ``None`` and leaves the spec's own
    default standing; a field it carries as ``None`` is in ``model_fields_set``
    and is a statement the family rules may reject.

    Attributes:
        atomic_numbers: Element table.
        atomic_energies: Per-element ``E0`` in eV/atom.
        r_max: Cutoff in Å.
        hidden_irreps: Node-state irreps string, e.g. ``"128x0e+128x1o"``.
        mlp_irreps: Last-readout hidden irreps string, e.g. ``"16x0e"``.
        num_features, max_hidden_l, mlp_dim: Explicit widths, used only when
            the irreps strings are absent (Polar, OMol dumps).
        pair_repulsion, distance_transform: Architecture switches of the
            density families.
        interaction_blocks: Upstream block class names of the stack, first to
            last (MACE-OFF reads which kind the first layer is). A pickled
            module reports the stack itself; a json written by upstream's
            ``extract_config_mace_model`` names it as ``interaction_cls_first``
            + ``interaction_cls`` × ``num_interactions - 1``, which is read
            into this field when the list is absent.
        avg_num_neighbors: Message normalisation of the agnostic interactions.
        les: The LES block of a MACE-LES archive.
        kspace_cutoff: Serialised reciprocal cutoff of a Polar dump; wins over
            the heuristic ``kspace_cutoff_factor`` when present.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    atomic_numbers: list[int]
    atomic_energies: list[float]
    r_max: float
    num_bessel: int | None = None
    num_polynomial_cutoff: int | None = None
    l_max: int | None = Field(None, validation_alias=AliasChoices("max_ell", "l_max"))
    correlation: int | None = None
    num_interactions: int | None = None
    hidden_irreps: str | None = None
    mlp_irreps: str | None = Field(None, validation_alias=AliasChoices("MLP_irreps", "mlp_irreps"))
    radial_mlp: list[int] | None = Field(
        None, validation_alias=AliasChoices("radial_MLP", "radial_mlp")
    )
    scale: float | None = Field(None, validation_alias=AliasChoices("atomic_inter_scale", "scale"))
    shift: float | None = Field(None, validation_alias=AliasChoices("atomic_inter_shift", "shift"))
    num_features: int | None = None
    max_hidden_l: int | None = None
    mlp_dim: int | None = None
    edge_channels: int | None = None
    pair_repulsion: bool | None = None
    distance_transform: str | None = None
    interaction_blocks: list[str] | None = None
    avg_num_neighbors: float | None = None
    les: LesOfficialConfig | None = None
    num_recursion_steps: int | None = None
    multipole_lmax: int | None = None
    multipole_sigma: float | None = None
    field_lmax: int | None = None
    field_widths: list[float] | None = None
    field_norms: list[float] | None = None
    add_local_electron_energy: bool | None = None
    kspace_cutoff_factor: float | None = None
    kspace_cutoff: float | None = None
    electrostatic_self_interaction: bool | None = Field(
        None,
        validation_alias=AliasChoices(
            "include_electrostatic_self_interaction", "electrostatic_self_interaction"
        ),
    )
    field_self_interaction: bool | None = Field(
        None, validation_alias=AliasChoices("field_si", "field_self_interaction")
    )
    keep_last_layer_irreps: bool | None = None
    quadrupole_feature_corrections: bool | None = None
    use_so3: bool | None = None
    use_agnostic_product: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def _stack_from_class_names(cls, data: Any) -> Any:
        """Read upstream's ``interaction_cls_first`` / ``interaction_cls`` pair as the stack.

        ``mace.tools.scripts_utils.extract_config_mace_model`` writes the two
        bare class names and ``num_interactions`` instead of the block list a
        pickled module carries; the list is the first name followed by the
        other ``num_interactions - 1`` times.
        """
        if not isinstance(data, Mapping) or "interaction_blocks" in data:
            return data
        first, rest, depth = (
            data.get(key)
            for key in ("interaction_cls_first", "interaction_cls", "num_interactions")
        )
        if first is None or rest is None or depth is None:
            return data
        return {**data, "interaction_blocks": [str(first), *([str(rest)] * (int(depth) - 1))]}

    def body_fields(self) -> dict[str, Any]:
        """The shared body fields every family reads, only those the archive carries."""
        return _present(
            r_max=self.r_max,
            num_bessel=self.num_bessel,
            num_polynomial_cutoff=self.num_polynomial_cutoff,
            l_max=self.l_max,
            num_interactions=self.num_interactions,
            correlation=self.correlation,
            scale=self.scale,
            shift=self.shift,
        )

    def irreps_widths(self) -> tuple[int, int, int]:
        """``(num_features, max_hidden_l, mlp_dim)`` from the two irreps strings.

        Raises:
            KeyError: If either string is absent — the widths have no default.
            ValueError: If a string is not an irreps string.
        """
        absent = [
            key
            for key, value in (
                ("hidden_irreps", self.hidden_irreps),
                ("MLP_irreps", self.mlp_irreps),
            )
            if value is None
        ]
        if absent:
            raise KeyError(
                f"official config is missing {absent}: "
                "num_features / max_hidden_l / mlp_dim are derived from the irreps "
                "strings and have no default"
            )
        num_features, max_hidden_l = parse_irreps(str(self.hidden_irreps), "hidden_irreps")
        mlp_dim, _ = parse_irreps(str(self.mlp_irreps), "MLP_irreps")
        return num_features, max_hidden_l, mlp_dim


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
            ``atomic_numbers`` order. Any iterable of numbers — a tensor
            included — is read into a plain ``list``.
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
        avg_num_neighbors: Message normalisation of the ``"agnostic"``
            interactions; required by them, ignored by the others.
        last_readout_bias: The last per-layer readout carries a bias
            (Polar's dump does).
        charge_classes: Embedding rows of the total-charge conditioning.
        charge_offset: Index offset applied to the total charge.
        spin_classes: Embedding rows of the total-spin conditioning.
        spin_offset: Index offset applied to the total spin.
        interaction: ``"density"`` (MatPES density-normalised interactions),
            ``"agnostic"`` (MACE-OFF / MACE-MP-0 interactions normalised by
            ``avg_num_neighbors``) or ``"residual"`` (OMOL non-linear residual
            ones).
        readout: ``"per_layer"`` (one readout per layer, summed) or
            ``"final"`` (a single readout on the last layer).
        distance_transform: ``"agnesi"`` applies the Agnesi radial transform
            before the Bessel basis; ``"none"`` feeds raw distances.
        pair_repulsion: ``"zbl"`` adds the ZBL short-range pair term.
        conditioning: ``"charge_spin"`` adds the total-charge / total-spin
            joint embedding to the initial node features; the four
            ``*_classes`` / ``*_offset`` fields are then required, and are
            rejected without it.
    """

    model_config = ConfigDict(extra="forbid")

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
    avg_num_neighbors: float | None = Field(None, gt=0.0)
    last_readout_bias: bool = False
    charge_classes: int | None = Field(None, gt=0)
    charge_offset: int | None = Field(None, ge=0)
    spin_classes: int | None = Field(None, gt=0)
    spin_offset: int | None = Field(None, ge=0)

    interaction: Literal["density", "residual", "agnostic"]
    readout: Literal["per_layer", "final"]
    distance_transform: Literal["none", "agnesi"]
    pair_repulsion: Literal["none", "zbl"]
    conditioning: Literal["none", "charge_spin"]

    @field_validator("atomic_energies", mode="before")
    @classmethod
    def _energies_as_list(cls, value: Any) -> Any:
        """Read any iterable of numbers — a tensor included — into ``list[float]``."""
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return [float(energy) for energy in value]
        return value

    @model_validator(mode="after")
    def _check_element_table(self) -> Self:
        """Reject element tables that would mis-index ``E0`` or the one-hot.

        Raises:
            ValueError: If ``atomic_energies`` has a different length than
                ``atomic_numbers``, or ``atomic_numbers`` is not strictly
                ascending (unordered or duplicated), or an ``"agnostic"``
                stack has no ``avg_num_neighbors``.
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
            NotImplementedError: On the bare base class, which describes no
                shipped model.
        """
        raise NotImplementedError(f"{type(self).__name__} states no body schedule")


class MACEMatpesSpec(MACESpec):
    """Configuration of the MACE-MatPES foundation model.

    Defaults reproduce the released MatPES architecture,
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
    def _check_residual_layer(self) -> Self:
        """MatPES's stack is one density layer plus at least one residual one.

        Raises:
            ValueError: If ``num_interactions`` is below 2.
        """
        if self.num_interactions < 2:
            raise ValueError(f"num_interactions must be at least 2, got {self.num_interactions}")
        return self

    def body_schedule(self) -> BodySchedule:
        """Density schedule: ``max_hidden_l``, the spec's radial MLP, per-element products."""
        return BodySchedule(
            hidden_l=self.max_hidden_l,
            edge_channels=self.num_features,
            radial_mlp=tuple(self.radial_mlp),
            keep_last_layer_irreps=False,
            shared_product=False,
            residual_first=False,
        )

    @classmethod
    def from_official(cls, cfg: MaceOfficialConfig) -> Self:
        """The spec of an official MatPES archive.

        Args:
            cfg: The archive's typed hyper-parameters.

        Returns:
            A spec whose widths come from the irreps strings.

        Raises:
            ValueError: If the archive states ``pair_repulsion`` off or a
                ``distance_transform`` other than Agnesi. Both are hard-wired
                here and their fitted constants are buffers, so the strict
                weight load would not object to the surplus term — the model
                would run, look sane and be quietly wrong.
            KeyError: If the irreps strings are absent.
        """
        if "pair_repulsion" in cfg.model_fields_set and not cfg.pair_repulsion:
            raise ValueError(
                f"official config sets pair_repulsion={cfg.pair_repulsion!r}, "
                "but MACEMatpesSpec hard-wires pair_repulsion='zbl' and the ZBL "
                "constants are buffers, so the load would not object to the surplus "
                "term"
            )
        if (
            "distance_transform" in cfg.model_fields_set
            and str(cfg.distance_transform).lower() != "agnesi"
        ):
            raise ValueError(
                f"official config sets distance_transform="
                f"{cfg.distance_transform!r}, but MACEMatpesSpec hard-wires "
                "distance_transform='agnesi' and the Agnesi constants are buffers, "
                "so the load would not object to the surplus transform"
            )
        num_features, max_hidden_l, mlp_dim = cfg.irreps_widths()
        return cls(
            atomic_numbers=cfg.atomic_numbers,
            atomic_energies=cfg.atomic_energies,
            num_features=num_features,
            max_hidden_l=max_hidden_l,
            mlp_dim=mlp_dim,
            **cfg.body_fields(),
            **_present(radial_mlp=cfg.radial_mlp),
        )


class MACEOMolSpec(MACESpec):
    """Configuration of the MACE-OMOL foundation model.

    Defaults reproduce the released OMOL architecture.

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
    def _check_angular_order(self) -> Self:
        """OMOL's mid-layer edge irreps span ``range(l_max)`` — empty at ``l_max=0``.

        Raises:
            ValueError: If ``l_max`` is below 1.
        """
        if self.l_max < 1:
            raise ValueError(f"l_max must be at least 1, got {self.l_max}")
        return self

    def body_schedule(self) -> BodySchedule:
        """Residual schedule: hidden order ``l_max - 1``, one shared product.

        Returns:
            The OMOL body schedule; the radial MLP is ``[edge_channels] * 3``.
        """
        return BodySchedule(
            hidden_l=self.l_max - 1,
            edge_channels=self.edge_channels,
            radial_mlp=(self.edge_channels, self.edge_channels, self.edge_channels),
            keep_last_layer_irreps=False,
            shared_product=True,
            residual_first=False,
        )

    @classmethod
    def from_official(cls, cfg: MaceOfficialConfig) -> Self:
        """The spec of an official OMol archive: the explicit widths it carries, defaults otherwise.

        Args:
            cfg: The archive's typed hyper-parameters.
        """
        return cls(
            atomic_numbers=cfg.atomic_numbers,
            atomic_energies=cfg.atomic_energies,
            **cfg.body_fields(),
            **_present(
                num_features=cfg.num_features,
                mlp_dim=cfg.mlp_dim,
                edge_channels=cfg.edge_channels,
            ),
        )


class MACEOffSpec(MACESpec):
    """Configuration of the MACE-OFF organic force fields.

    Standard MACE: a first ``RealAgnosticInteractionBlock``, then
    ``RealAgnosticResidualInteractionBlock`` layers, both normalised by
    ``avg_num_neighbors``; per-layer readouts; no ZBL term and no distance
    transform. Defaults are MACE-OFF24 medium (``128x0e+128x1o``, ``l_max=3``,
    ``r_max=6``); ``avg_num_neighbors`` has no default (it is the training
    set's).

    Attributes:
        max_hidden_l: Highest ``l`` carried in the node state between layers.
        radial_mlp: Hidden widths of the radial weight MLP.
        residual_first: The first layer is a residual block too (MACE-OFF23,
            MACE-LES-OFF), so its product carries a skip connection.
    """

    max_hidden_l: int = Field(1, ge=0)
    residual_first: bool = False
    radial_mlp: list[int] = Field(default_factory=lambda: [64, 64, 64], min_length=1)

    r_max: float = Field(6.0, gt=0.0)
    l_max: int = Field(3, ge=1)
    num_bessel: int = Field(8, gt=0)
    num_features: int = Field(128, gt=0)
    num_interactions: int = Field(2, gt=0)
    correlation: int = Field(3, ge=1)

    interaction: Literal["density", "residual", "agnostic"] = "agnostic"
    readout: Literal["per_layer", "final"] = "per_layer"
    distance_transform: Literal["none", "agnesi"] = "none"
    pair_repulsion: Literal["none", "zbl"] = "none"
    conditioning: Literal["none", "charge_spin"] = "none"

    def body_schedule(self) -> BodySchedule:
        """Standard MACE schedule: ``max_hidden_l``, the spec's radial MLP, per-element products."""
        return BodySchedule(
            hidden_l=self.max_hidden_l,
            edge_channels=self.num_features,
            radial_mlp=tuple(self.radial_mlp),
            keep_last_layer_irreps=False,
            shared_product=False,
            residual_first=self.residual_first,
        )

    @classmethod
    def from_official(cls, cfg: MaceOfficialConfig) -> Self:
        """The spec of an official MACE-OFF archive.

        Args:
            cfg: The archive's typed hyper-parameters.

        Raises:
            ValueError: If the interaction stack is not ``RealAgnostic``
                (``Residual``) blocks, or the archive carries a pair
                repulsion or a distance transform (MACE-OFF has neither).
            KeyError: If ``avg_num_neighbors`` or the irreps strings are
                absent.
        """
        blocks = list(cfg.interaction_blocks or [])
        residual = "RealAgnosticResidualInteractionBlock"
        if blocks[1:] != [residual] * (len(blocks) - 1) or blocks[:1] not in (
            ["RealAgnosticInteractionBlock"],
            [residual],
        ):
            raise ValueError(
                "MACE-OFF expects RealAgnostic(Residual)InteractionBlock first and "
                f"{residual} after it; the checkpoint has {blocks}"
            )
        for name, value in (
            ("pair_repulsion", cfg.pair_repulsion),
            ("distance_transform", cfg.distance_transform),
        ):
            if value:
                raise ValueError(f"MACE-OFF has no {name}; the checkpoint sets {value!r}")
        if cfg.avg_num_neighbors is None:
            raise KeyError("official MACE-OFF config is missing avg_num_neighbors")
        num_features, max_hidden_l, mlp_dim = cfg.irreps_widths()
        return cls(
            atomic_numbers=cfg.atomic_numbers,
            atomic_energies=cfg.atomic_energies,
            avg_num_neighbors=cfg.avg_num_neighbors,
            num_features=num_features,
            max_hidden_l=max_hidden_l,
            mlp_dim=mlp_dim,
            residual_first=blocks[0] == residual,
            **cfg.body_fields(),
            **_present(radial_mlp=cfg.radial_mlp),
            **cls._extra_official_fields(cfg),
        )

    @classmethod
    def _extra_official_fields(cls, cfg: MaceOfficialConfig) -> dict[str, Any]:
        """Fields a subclass reads from the archive beyond the MACE-OFF body."""
        del cfg
        return {}


class MACELesSpec(MACEOffSpec):
    """Configuration of MACE-LES: MACE-OFF plus the latent-charge long-range energy.

    Attributes:
        les_sigma: Gaussian width ``σ`` of the latent charges in Å.
        les_dl: Reciprocal-grid spacing in Å (cutoff ``2π / dl``).
        les_norm_factor: LES's ``norm_factor`` in eV·Å/e².
        les_output_scale: Factor on the summed charge readouts.
        les_remove_self_interaction: Subtract the Gaussian self-energy.
    """

    les_sigma: float = Field(1.0, gt=0.0)
    les_dl: float = Field(2.0, gt=0.0)
    les_norm_factor: float = 90.0474
    les_output_scale: float = 1.0
    les_remove_self_interaction: bool = True

    @classmethod
    def _extra_official_fields(cls, cfg: MaceOfficialConfig) -> dict[str, Any]:
        """The LES block, after the charge-only rule.

        Raises:
            KeyError: If the archive has no LES block.
            ValueError: If the archive uses another Ewald class or any of the
                latent-multipole / induced switches — only the charge-only
                legacy LES Ewald is ported.
        """
        les = cfg.les
        if les is None:
            raise KeyError("official MACE-LES config has no LES block")
        switches = (
            ("use_atomwise", les.use_atomwise),
            ("use_fixed_atomic_charges", les.use_fixed_atomic_charges),
            ("use_epsilon_r_scaling", les.use_epsilon_r_scaling),
            ("use_dipoles", les.use_dipoles),
            ("use_quads", les.use_quads),
            ("use_induced_charges", les.use_induced_charges),
            ("use_induced_dipoles", les.use_induced_dipoles),
        )
        unsupported = [name for name, on in switches if on]
        if les.ewald != "Ewald" or unsupported:
            raise ValueError(
                "only the charge-only legacy LES Ewald is ported; this checkpoint uses "
                f"{les.ewald} with {unsupported or 'no extra'} switches"
            )
        return {
            "les_sigma": les.sigma,
            "les_dl": les.dl,
            "les_norm_factor": les.norm_factor,
            "les_output_scale": les.output_scale,
            "les_remove_self_interaction": les.remove_self_interaction,
        }


class MACEPolarSpec(MACESpec):
    """Configuration of the MACE-POLAR-1-M foundation model, body and electrostatic head.

    Defaults pin the official ``MACE-POLAR-1-M.model`` dump of 2026-08-30,
    not PolarMACE constructor defaults.

    Attributes:
        max_hidden_l: Highest ``l`` in the node state and mid-layer edge
            irreps. Dump is ``512x0e+512x1o`` at ``l_max=3``, so this is 1,
            not the OMOL ``l_max - 1``.
        keep_last_layer_irreps: Last product keeps full ``hidden`` irreps
            (dump ``True``).
        edge_channels: Per-``l`` channel count of the mid-layer edge irreps.
        radial_mlp: Hidden widths of the radial weight MLP.
        num_recursion_steps: Field-dependent charge updates after the
            short-range stack.
        multipole_lmax: Highest multipole order of the GTO density. Independent
            of the body's ``l_max``.
        multipole_sigma: Gaussian width of that density, in Å.
        field_lmax: Highest order of the field features.
        field_widths: Gaussian widths of the field features, in Å.
        field_norms: Per-order normalisation of the field features.
        add_local_electron_energy: Add the local electron-energy readout.
        kspace_cutoff_factor: Reciprocal-space cutoff factor.
        kspace_cutoff: Explicit reciprocal cutoff. ``None`` derives it.
        electrostatic_self_interaction: On-site GTO overlap counts in the
            Hartree energy.
        field_self_interaction: An atom sees its own multipole in its field
            features.
    """

    max_hidden_l: int = Field(1, ge=0)
    keep_last_layer_irreps: bool = True
    last_readout_bias: bool = True
    edge_channels: int = Field(128, gt=0)
    radial_mlp: list[int] = Field(default_factory=lambda: [64, 64, 64], min_length=1)

    num_bessel: int = Field(8, gt=0)
    num_features: int = Field(512, gt=0)
    num_interactions: int = Field(2, gt=0)
    correlation: int = Field(3, ge=1)

    num_recursion_steps: int = Field(2, ge=1)
    multipole_lmax: int = Field(1, ge=0, le=1)
    multipole_sigma: float = Field(1.5, gt=0.0)
    field_lmax: int = Field(1, ge=0, le=1)
    field_widths: list[float] = Field(default_factory=lambda: [1.5, 3.0])
    field_norms: list[float] = Field(default_factory=lambda: [20.0, 20.0, 0.5, 0.5])
    add_local_electron_energy: bool = True
    kspace_cutoff_factor: float = Field(1.0, gt=0.0)
    kspace_cutoff: float | None = Field(None, gt=0.0)
    electrostatic_self_interaction: bool = True
    field_self_interaction: bool = False

    interaction: Literal["density", "residual"] = "residual"
    readout: Literal["per_layer", "final"] = "per_layer"
    distance_transform: Literal["none", "agnesi"] = "none"
    pair_repulsion: Literal["none", "zbl"] = "none"
    conditioning: Literal["none", "charge_spin"] = "none"

    def body_schedule(self) -> BodySchedule:
        """Residual schedule at ``max_hidden_l``, not at ``l_max - 1``.

        Returns:
            The polar body schedule; ``keep_last_layer_irreps`` is the spec's.
        """
        return BodySchedule(
            hidden_l=self.max_hidden_l,
            edge_channels=self.edge_channels,
            radial_mlp=tuple(self.radial_mlp),
            keep_last_layer_irreps=self.keep_last_layer_irreps,
            shared_product=True,
            residual_first=False,
        )

    @classmethod
    def from_official(cls, cfg: MaceOfficialConfig) -> Self:
        """The spec of an official Polar dump.

        Widths come from the irreps strings when present, else from the
        explicit ``num_features`` / ``max_hidden_l`` / ``mlp_dim``; a
        single-interaction model has no non-linear readout and so no
        ``MLP_irreps``.

        Args:
            cfg: The archive's typed hyper-parameters.

        Raises:
            ValueError: If the dump states a Polar architecture switch this
                port does not implement (``keep_last_layer_irreps`` off,
                quadrupole corrections, SO(3), a non-agnostic product).
            KeyError: If neither the irreps strings nor the explicit widths
                are present.
        """
        for name, value, supported in (
            ("keep_last_layer_irreps", cfg.keep_last_layer_irreps, True),
            ("quadrupole_feature_corrections", cfg.quadrupole_feature_corrections, False),
            ("use_so3", cfg.use_so3, False),
            ("use_agnostic_product", cfg.use_agnostic_product, True),
        ):
            if value is not None and value != supported:
                raise ValueError(f"unsupported Polar configuration: {name}={value!r}")
        num_features, max_hidden_l, mlp_dim = cfg.num_features, cfg.max_hidden_l, cfg.mlp_dim
        if cfg.hidden_irreps is not None:
            num_features, max_hidden_l = parse_irreps(cfg.hidden_irreps, "hidden_irreps")
        if cfg.mlp_irreps is not None:
            mlp_dim, _ = parse_irreps(cfg.mlp_irreps, "MLP_irreps")
        single = (cfg.num_interactions or 2) == 1
        if num_features is None or max_hidden_l is None or (mlp_dim is None and not single):
            raise KeyError(
                "official Polar config requires hidden_irreps and MLP_irreps (or explicit widths)"
            )
        return cls(
            atomic_numbers=cfg.atomic_numbers,
            atomic_energies=cfg.atomic_energies,
            num_features=num_features,
            max_hidden_l=max_hidden_l,
            **cfg.body_fields(),
            **_present(
                mlp_dim=mlp_dim,
                edge_channels=cfg.edge_channels,
                radial_mlp=cfg.radial_mlp,
                num_recursion_steps=cfg.num_recursion_steps,
                multipole_lmax=cfg.multipole_lmax,
                multipole_sigma=cfg.multipole_sigma,
                field_lmax=cfg.field_lmax,
                field_widths=cfg.field_widths,
                field_norms=cfg.field_norms,
                add_local_electron_energy=cfg.add_local_electron_energy,
                kspace_cutoff_factor=cfg.kspace_cutoff_factor,
                kspace_cutoff=cfg.kspace_cutoff,
                electrostatic_self_interaction=cfg.electrostatic_self_interaction,
                field_self_interaction=cfg.field_self_interaction,
            ),
        )
