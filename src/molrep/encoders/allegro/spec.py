"""Validated configuration of the Allegro encoder.

Torch-free: imports only ``pydantic``. Field names follow the upstream
``FullAllegroModel`` builder (``src/molzoo/specs/allegro.md`` §6). A product
adds its readout fields by subclassing (:class:`molzoo.allegro.spec.AllegroSpec`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["AllegroEncoderSpec"]


class AllegroEncoderSpec(BaseModel):
    """Configuration of :class:`~molrep.encoders.allegro.encoder.AllegroEncoder`."""

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
    #: Edge-sum normalisation ``1/√⟨N⟩``; ``None`` is no normalisation.
    avg_num_neighbors: float | None = Field(None, gt=0.0)
    #: MolNex-only: keep the last layer's tensor irreps and write them (spec §4).
    expose_tensor_track: bool = False

    @model_validator(mode="after")
    def _check_types(self) -> AllegroEncoderSpec:
        if len(set(self.atomic_numbers)) != len(self.atomic_numbers):
            raise ValueError("atomic_numbers must be unique")
        if min(self.atomic_numbers) < 1:
            raise ValueError("atomic_numbers must be positive")
        return self

    @property
    def embed_dim(self) -> int:
        """Two-body embedding width ``D``."""
        return self.radial_chemical_embed_dim or self.num_scalar_features
