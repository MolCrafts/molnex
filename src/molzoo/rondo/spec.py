"""Validated configuration snapshot for the Rondo recipe.

Spec: ``src/molzoo/specs/rondo.md`` §4. The spec holds only what is Rondo's
own — the encoder and the heads. The functional form (styles, cutoff, mixing,
1-4 scales, Coulomb constant) is the force field handed to
:class:`~molzoo.rondo.model.Rondo` and is not restated here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from molrep.chem.embed import DiscreteChannel

__all__ = ["EncoderSpec", "HeadsSpec", "RondoSpec"]

#: ByteFF's atom features, as the batch columns molhub writes for THEMol
#: (``mol_data.py:174-189``): formal charge clamped to ±2, total molecular
#: charge clamped to ±5, the number of ring bonds on the atom, the smallest
#: ring through it (0 = none, rings above 8 read as 8).
DEFAULT_ATOM_CHANNELS: tuple[DiscreteChannel, ...] = (
    DiscreteChannel(name="formal_charge", num_classes=5, offset=2),
    DiscreteChannel(name="total_charge", num_classes=11, offset=5, per="graph"),
    DiscreteChannel(name="ring_connectivity", num_classes=5),
    DiscreteChannel(name="min_ring_size", num_classes=9),
)

#: ByteFF's bond features (``mol_data.py:192-201``): bond order as molpy's
#: ``bond_type`` (1 single, 2 double, 3 triple, 4 aromatic) and ring membership.
DEFAULT_BOND_CHANNELS: tuple[DiscreteChannel, ...] = (
    DiscreteChannel(name="bond_type", num_classes=5),
    DiscreteChannel(name="in_ring", num_classes=2),
)


class EncoderSpec(BaseModel):
    """The :class:`~molrep.chem.encoder.ChemEncoder` under Rondo.

    Attributes:
        width: Feature width of the atom / bond / angle / proper / improper
            channels and of the hop MLPs. A power of two.
        n_blocks: Covalent hops ``L``; the receptive field of a predicted
            parameter is ``L`` bonds.
        local_attention: Softmax over an atom's incident covalent bonds
            instead of a plain sum.
        num_elements: Atomic-number embedding table size.
        atom_channels: Extra integer atom (or graph) columns summed into the
            atom seed; every one is a required batch column.
        bond_channels: Extra integer bond columns summed into the bond seed.
    """

    model_config = ConfigDict(frozen=True)

    width: int = Field(default=64, gt=0)
    n_blocks: int = Field(default=4, ge=0)
    local_attention: bool = False
    num_elements: int = Field(default=119, gt=0)
    atom_channels: tuple[DiscreteChannel, ...] = DEFAULT_ATOM_CHANNELS
    bond_channels: tuple[DiscreteChannel, ...] = DEFAULT_BOND_CHANNELS


class HeadsSpec(BaseModel):
    """The parameter heads: one MLP per parameter, each starting at a reference.

    Force constants are LAMMPS ``K`` (no ½); angles are degrees.

    Attributes:
        hidden: Hidden width of every head MLP.
        bond_k_ref: Reference bond ``K``, kcal/mol/Å². GAFF2 bonds run
            300-700; THEMol's 1-2 Hessian blocks put the median at ``−2K =
            −932``.
        bond_r0_gain: Relative change of ``r0`` per unit of pre-activation.
        angle_k_ref: Reference angle ``K``, kcal/mol/rad² (GAFF2: 35-70).
        angle_theta0_gain_deg: Degrees of ``theta0`` per unit of pre-activation.
        proper_periodicity: Fourier terms ``n`` of a proper torsion. ``(1, 2,
            3, 4)`` is the OPLS / GAFF capacity.
        improper_periodicity: The ``n`` of the improper term (ByteFF: 2).
        improper_k_ref: Reference improper amplitude per row, kcal/mol (ByteFF's
            GAFF2 mean 4.29 per centre over its three rows). Positive by
            construction; the improper-strain objective is unbounded below for a
            signed amplitude.
        lj_sigma_gain: Relative change of σ per unit of pre-activation.
        lj_epsilon_gain: Log change of ε per unit of pre-activation.
        eem_max_atoms: Fixed per-molecule capacity of the dense charge-
            equilibration solve. Must cover the largest molecule in the train
            set; sizing it from the batch is a host sync and a graph break.
        eem_jitter: Added to every hardness before the solve.
    """

    model_config = ConfigDict(frozen=True)

    hidden: int = Field(default=64, gt=0)
    bond_k_ref: float = Field(default=450.0, gt=0.0)
    bond_r0_gain: float = Field(default=0.1, gt=0.0)
    angle_k_ref: float = Field(default=40.0, gt=0.0)
    angle_theta0_gain_deg: float = Field(default=20.0, gt=0.0)
    proper_periodicity: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    improper_periodicity: int = Field(default=2, ge=1)
    improper_k_ref: float = Field(default=4.291 / 3.0, gt=0.0)
    lj_sigma_gain: float = Field(default=0.1, gt=0.0)
    lj_epsilon_gain: float = Field(default=1.0, gt=0.0)
    eem_max_atoms: int = Field(default=512, ge=1)
    eem_jitter: float = Field(default=1e-4, ge=0.0)


class RondoSpec(BaseModel):
    """Configuration for :class:`~molzoo.rondo.model.Rondo`: encoder and heads.

    What a training stage trains is not a field here: one spec, one
    ``state_dict`` layout, every objective. The energy terms come from the
    force field the model is built on.
    """

    model_config = ConfigDict(frozen=True)

    encoder: EncoderSpec = EncoderSpec()
    heads: HeadsSpec = HeadsSpec()
