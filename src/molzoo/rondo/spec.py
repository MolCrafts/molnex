"""Validated configuration snapshot for the Rondo recipe.

Spec: ``src/molzoo/specs/rondo.md`` §4.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RondoSpec(BaseModel):
    """Configuration for :class:`~molzoo.rondo.parameterizer.Rondo`.

    Attributes:
        hidden: Shared feature width for the atom / bond / angle / proper /
            improper channels and for every head MLP. A power of two.
        n_blocks: Covalent hops ``L``. The receptive field of a predicted
            parameter is ``L`` bonds.
        n_torsion_terms: Fourier terms ``T`` per proper torsion. Real
            capacity, not an ignored argument — the torsion is the general
            cosine series.
        local_attention: Softmax over an atom's incident covalent bonds
            instead of a plain sum.
        num_elements: Atomic-number embedding table size.
        num_bond_types: Bond-type table size; ``0`` disables type
            conditioning.
        eem_max_atoms: Fixed per-molecule capacity of the dense charge-
            equilibration solve. Must cover the largest molecule in the train
            set. Sizing it from the batch instead is a host sync and a
            data-dependent shape, i.e. a graph break on a hot path.
        cutoff: Nonbonded radius in Å, declared on the field. These are
            isolated molecules with no cell, so the Coulomb term is the open
            ``1/r`` sum, which is unbounded unless a radius is stated —
            ``neighbor_list()`` refuses to invent one.
        scale_14_charge: 1-4 Coulomb exclusion scale. 1-2 and 1-3 are ``0``.
        scale_14_lj: 1-4 Lennard-Jones exclusion scale.
        mixing: Lennard-Jones combination rule.

    The energy unit and what the derivative differentiates against are
    :class:`~molzoo.rondo.potential.RondoPotential` construction arguments,
    not fields here: they describe what leaves the potential, not what the
    parameterizer predicts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hidden: int = Field(default=64, gt=0)
    n_blocks: int = Field(default=4, ge=0)
    n_torsion_terms: int = Field(default=8, gt=0)
    local_attention: bool = False
    num_elements: int = Field(default=119, gt=0)
    num_bond_types: int = Field(default=0, ge=0)
    eem_max_atoms: int = Field(default=512, ge=1)
    cutoff: float = Field(default=10.0, gt=0.0)
    scale_14_charge: float = Field(default=0.5, ge=0.0, le=1.0)
    scale_14_lj: float = Field(default=0.5, ge=0.0, le=1.0)
    mixing: str = "geometric"
