"""Validated configuration snapshot for :class:`~molzoo.chem.encoder.ChemPerception`."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChemPerceptionSpec(BaseModel):
    """Configuration for the continuous chemical-perception recipe.

    Attributes:
        atom_dim: Per-atom feature dimension.
        bond_dim: Per-bond feature dimension.
        angle_dim: Per-angle feature dimension.
        proper_dim: Per-proper-torsion feature dimension.
        improper_dim: Per-improper feature dimension.
        num_elements: Atomic-number embedding table size.
        hidden_dim: Optional shared MLP hidden width for context builders.
        num_bond_types: Bond-type table size; ``0`` disables type conditioning.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    atom_dim: int = Field(default=32, gt=0)
    bond_dim: int = Field(default=32, gt=0)
    angle_dim: int = Field(default=32, gt=0)
    proper_dim: int = Field(default=32, gt=0)
    improper_dim: int = Field(default=32, gt=0)
    num_elements: int = Field(default=119, gt=0)
    hidden_dim: int | None = None
    num_bond_types: int = Field(default=0, ge=0)
