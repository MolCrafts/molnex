"""Validated configuration snapshot for :class:`~molzoo.chem.encoder.ChemPerception`."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from molrep.chem.embed import DiscreteChannel


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
        atom_channels: Extra integer atom (or graph) columns embedded into the
            atom seed, e.g. ``formal_charge``; see
            :class:`~molrep.chem.embed.DiscreteChannel`.
        bond_channels: Extra integer bond columns embedded into the bond seed,
            e.g. ``bond_type``.
        n_blocks: Covalent-graph hops. The receptive field of an atom feature
            is ``n_blocks`` bonds, so this is the knob that decides how much
            of a molecule a predicted parameter can see.
        local_attention: Softmax-normalise each atom's aggregation over its
            own incident covalent bonds instead of summing them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    atom_dim: int = Field(default=32, gt=0)
    bond_dim: int = Field(default=32, gt=0)
    angle_dim: int = Field(default=32, gt=0)
    proper_dim: int = Field(default=32, gt=0)
    improper_dim: int = Field(default=32, gt=0)
    num_elements: int = Field(default=119, gt=0)
    hidden_dim: int | None = None
    atom_channels: tuple[DiscreteChannel, ...] = ()
    bond_channels: tuple[DiscreteChannel, ...] = ()
    n_blocks: int = Field(default=5, ge=0)
    local_attention: bool = False
