"""Validated configuration for the Allegro family.

Torch-free: imports only ``typing`` and ``pydantic``. Field names follow the
upstream ``FullAllegroModel`` builder (``src/molzoo/specs/allegro.md`` §6) so
an official package's hyper-parameters map one to one; every cross-field rule
is checked here, once.

:class:`NequipConfig` is what the ``.nequip.zip`` reader
(:class:`~molzoo.allegro.utils.NequipPackage`) recovers from a package's
module tree, in upstream vocabulary; :meth:`AllegroSpec.from_package` turns it
into a spec without touching torch.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NequipMLP(BaseModel):
    """Shape of one upstream ``ScalarMLPFunction``: ``width`` is ``None`` at depth 0."""

    model_config = ConfigDict(extra="forbid")

    depth: int = Field(..., ge=0)
    width: int | None = Field(None, gt=0)
    nonlinearity: str | None = None


class NequipConfig(BaseModel):
    """Hyper-parameters an official ``.nequip.zip`` package describes.

    Filled by :meth:`~molzoo.allegro.utils.NequipPackage.read` from the
    package's module tree; only the fields this port reads are declared and
    anything else is ignored.
    """

    model_config = ConfigDict(extra="ignore")

    atomic_numbers: list[int] = Field(..., min_length=1)
    r_max: float = Field(..., gt=0.0)
    num_bessels: int = Field(..., gt=0)
    bessel_trainable: bool
    polynomial_cutoff_p: int = Field(..., gt=0)
    radial_chemical_embed_dim: int = Field(..., gt=0)
    scalar_embed_mlp: NequipMLP
    l_max: int = Field(..., ge=1)
    parity: bool
    num_layers: int = Field(..., gt=0)
    num_scalar_features: int = Field(..., gt=0)
    num_tensor_features: int = Field(..., gt=0)
    allegro_mlp: NequipMLP
    readout_mlp: NequipMLP
    avg_num_neighbors: float = Field(..., gt=0.0)
    per_type_energy_scales: list[float] | None = None
    per_type_energy_shifts: list[float] | None = None
    per_type_energy_scales_trainable: bool = False
    per_type_energy_shifts_trainable: bool = False
    pair_potential: Literal["ZBL"] | None = None
    zbl_polynomial_cutoff_p: int | None = Field(None, gt=0)


class AllegroSpec(BaseModel):
    """Configuration of :class:`~molzoo.allegro.encoder.AllegroEncoder` and its products."""

    model_config = ConfigDict(extra="forbid")

    #: Atom-type table: type ``t`` is ``atomic_numbers[t]`` (upstream ``type_names`` order).
    atomic_numbers: list[int] = Field(..., min_length=1)
    r_max: float = Field(..., gt=0.0)
    num_bessels: int = Field(8, gt=0)
    bessel_trainable: bool = False
    polynomial_cutoff_p: int = Field(6, gt=0)
    #: Width ``D`` of the two-body type embedding; ``None`` is ``num_scalar_features``.
    radial_chemical_embed_dim: int | None = Field(None, gt=0)
    scalar_embed_mlp_hidden_layers_depth: int = Field(1, ge=0)
    scalar_embed_mlp_hidden_layers_width: int = Field(64, gt=0)
    scalar_embed_mlp_nonlinearity: str | None = "silu"
    l_max: int = Field(2, ge=1)
    parity: bool = True
    num_layers: int = Field(2, gt=0)
    num_scalar_features: int = Field(64, gt=0)
    num_tensor_features: int = Field(16, gt=0)
    allegro_mlp_hidden_layers_depth: int = Field(1, ge=0)
    allegro_mlp_hidden_layers_width: int = Field(64, gt=0)
    allegro_mlp_nonlinearity: str | None = "silu"
    readout_mlp_hidden_layers_depth: int = Field(1, ge=0)
    readout_mlp_hidden_layers_width: int = Field(32, gt=0)
    readout_mlp_nonlinearity: str | None = "silu"
    #: Edge-sum normalisation ``1/√⟨N⟩``; ``None`` is no normalisation.
    avg_num_neighbors: float | None = Field(None, gt=0.0)
    #: Per-type energy scale / shift in eV, one per type after validation (a
    #: scalar or one-entry list is broadcast to every type); ``None`` is 1 / 0.
    per_type_energy_scales: list[float] | None = None
    per_type_energy_shifts: list[float] | None = None
    per_type_energy_scales_trainable: bool = False
    per_type_energy_shifts_trainable: bool = False
    #: Add the LAMMPS ``pair_zbl`` repulsion (upstream ``pair_potential: ZBL``).
    zbl: bool = False
    zbl_polynomial_cutoff_p: int = Field(6, gt=0)
    #: MolNex-only: keep the last layer's tensor irreps and write them (spec §4).
    expose_tensor_track: bool = False
    #: ``cuet.SegmentedPolynomial`` method of the tensor products.
    tp_method: Literal["uniform_1d", "fused_tp"] = "uniform_1d"

    @model_validator(mode="before")
    @classmethod
    def _broadcast_type_tables(cls, data: Any) -> Any:
        """A scalar or one-entry ``per_type_energy_*`` becomes one entry per type."""
        if not isinstance(data, dict) or not isinstance(data.get("atomic_numbers"), list):
            return data
        n_types = len(data["atomic_numbers"])
        data = dict(data)
        for name in ("per_type_energy_scales", "per_type_energy_shifts"):
            value = data.get(name)
            if isinstance(value, int | float) and not isinstance(value, bool):
                data[name] = [float(value)] * n_types
            elif isinstance(value, list) and len(value) == 1:
                data[name] = [float(value[0])] * n_types
        return data

    @model_validator(mode="after")
    def _check(self) -> AllegroSpec:
        if len(set(self.atomic_numbers)) != len(self.atomic_numbers):
            raise ValueError("atomic_numbers must be unique")
        if min(self.atomic_numbers) < 1:
            raise ValueError("atomic_numbers must be positive")
        for name in ("per_type_energy_scales", "per_type_energy_shifts"):
            value = getattr(self, name)
            if value is not None and len(value) != len(self.atomic_numbers):
                raise ValueError(f"{name} must have one entry or one per type")
        return self

    @property
    def embed_dim(self) -> int:
        """Two-body embedding width ``D``."""
        return self.radial_chemical_embed_dim or self.num_scalar_features

    @classmethod
    def from_package(cls, cfg: NequipConfig) -> AllegroSpec:
        """The spec of the model an official package describes.

        A depth-0 MLP leaves its ``*_hidden_layers_width`` at the spec default
        (upstream records no width for it); ``pair_potential: ZBL`` becomes
        ``zbl`` with the package's cutoff exponent.

        Args:
            cfg: What :meth:`~molzoo.allegro.utils.NequipPackage.read` recovered.

        Returns:
            A validated spec with the default ``tp_method``.
        """
        fields: dict[str, Any] = {
            "atomic_numbers": cfg.atomic_numbers,
            "r_max": cfg.r_max,
            "num_bessels": cfg.num_bessels,
            "bessel_trainable": cfg.bessel_trainable,
            "polynomial_cutoff_p": cfg.polynomial_cutoff_p,
            "radial_chemical_embed_dim": cfg.radial_chemical_embed_dim,
            "l_max": cfg.l_max,
            "parity": cfg.parity,
            "num_layers": cfg.num_layers,
            "num_scalar_features": cfg.num_scalar_features,
            "num_tensor_features": cfg.num_tensor_features,
            "avg_num_neighbors": cfg.avg_num_neighbors,
            "per_type_energy_scales": cfg.per_type_energy_scales,
            "per_type_energy_shifts": cfg.per_type_energy_shifts,
            "per_type_energy_scales_trainable": cfg.per_type_energy_scales_trainable,
            "per_type_energy_shifts_trainable": cfg.per_type_energy_shifts_trainable,
            "zbl": cfg.pair_potential == "ZBL",
        }
        if cfg.pair_potential == "ZBL" and cfg.zbl_polynomial_cutoff_p is not None:
            fields["zbl_polynomial_cutoff_p"] = cfg.zbl_polynomial_cutoff_p
        for prefix, mlp in (
            ("scalar_embed_mlp", cfg.scalar_embed_mlp),
            ("allegro_mlp", cfg.allegro_mlp),
            ("readout_mlp", cfg.readout_mlp),
        ):
            fields[f"{prefix}_hidden_layers_depth"] = mlp.depth
            fields[f"{prefix}_nonlinearity"] = mlp.nonlinearity
            if mlp.width is not None:
                fields[f"{prefix}_hidden_layers_width"] = mlp.width
        return cls(**fields)
