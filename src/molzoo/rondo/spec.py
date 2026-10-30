"""Validated configuration snapshot for the Rondo recipe.

Spec: ``src/molzoo/specs/rondo.md`` §4 and ``.claude/specs/rondo-enc-01-graph-encoder.md``
§3. The spec holds only what is Rondo's own — the encoder and the heads. The
functional form (styles, cutoff, mixing, 1-4 scales, Coulomb constant) is the
force field handed to :class:`~molzoo.rondo.model.Rondo` and is not restated
here. The defaults are the published model's sizes (the reference
``fftrainer_config_in_use.yaml``), except Rondo's own pair heads (D1, D2).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from molrep.encoders.covalent_graph.spec import CovalentGraphEncoderSpec

__all__ = ["HeadsSpec", "RondoSpec"]


class HeadsSpec(BaseModel):
    """The parameter heads.

    The bonded heads are the published model's (``MMBonded``: ``pre_mlp_dims
    (256, 256, 3)``, ``out_mlp_dims (256, 4)``, ``tanh_output 20``). Rondo's
    pair heads are its own: EEM charges (D1) and σ / ε around GAFF2 (D2), one
    MLP per parameter of width :attr:`hidden`.
    :class:`~molzoo.byteff.potential.ByteFF` uses the published pair heads
    instead (``LJEs``: ``pre_mlp_dims (512, 512, 4)``, ``out_mlp_dims (256,
    3)``, ``tanh_output 20``).

    Attributes:
        bonded_hidden: Hidden width of each bonded head's ``pre`` and ``post``.
        bonded_width: Output width of ``pre`` and ``post``.
        bonded_layers: Linears in ``pre`` and in ``post``.
        bonded_out_hidden: Hidden width of each bonded parameter MLP.
        bonded_out_layers: Linears in each bonded parameter MLP.
        tanh_output: Saturation of the published model's output transform.
        bond_k_floor: Lowest bond ``K`` the bond head puts out, kcal mol⁻¹ Å⁻²
            (LAMMPS, ``E = K (r - r0)²``) — the lower clamp of its window. The
            published window's is 40, which is soft enough for an
            out-of-distribution bond to collapse in MD (a zwitterion's C–N⁺, spec
            ``rondo-enc-08-bond-floor``); 80 is the 0.05th percentile of the DFT
            bond stiffness on THEMol's held-out Hessians.
            :meth:`~molzoo.byteff.potential.ByteFF.load` keeps the published 40.
        hidden: Hidden width of the pair heads.
        lj_sigma_gain: Relative change of σ per unit of pre-activation.
        lj_epsilon_gain: Log change of ε per unit of pre-activation.
        eem_max_atoms: Fixed per-molecule capacity of the dense charge-
            equilibration solve. Must cover the largest molecule in the train
            set; sizing it from the batch is a host sync and a graph break.
        eem_jitter: Added to every hardness before the solve.
        lj_direct_hidden: Hidden width of the published model's σ and ε MLPs
            (:class:`~molzoo.byteff.potential.ByteFF` only).
        lj_direct_layers: Linears in each of them.
        bcc_hidden: Hidden width of the published model's bond-charge ``pre`` MLP.
        bcc_width: Output width of ``pre``.
        bcc_layers: Linears in ``pre``.
        bcc_out_hidden: Hidden width of the bond-charge ``out`` MLP.
        bcc_out_layers: Linears in ``out``.
        pair_tanh_output: Saturation of the published model's σ, ε and charge transfer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bonded_hidden: int = Field(default=256, gt=0)
    bonded_width: int = Field(default=256, gt=0)
    bonded_layers: int = Field(default=3, ge=1)
    bonded_out_hidden: int = Field(default=256, gt=0)
    bonded_out_layers: int = Field(default=4, ge=1)
    tanh_output: float = Field(default=20.0, gt=0.0)
    bond_k_floor: float = Field(default=80.0, gt=0.0)
    hidden: int = Field(default=64, gt=0)
    lj_sigma_gain: float = Field(default=0.1, gt=0.0)
    lj_epsilon_gain: float = Field(default=1.0, gt=0.0)
    eem_max_atoms: int = Field(default=512, ge=1)
    eem_jitter: float = Field(default=1e-4, ge=0.0)
    lj_direct_hidden: int = Field(default=256, gt=0)
    lj_direct_layers: int = Field(default=3, ge=1)
    bcc_hidden: int = Field(default=512, gt=0)
    bcc_width: int = Field(default=512, gt=0)
    bcc_layers: int = Field(default=4, ge=1)
    bcc_out_hidden: int = Field(default=256, gt=0)
    bcc_out_layers: int = Field(default=3, ge=1)
    pair_tanh_output: float = Field(default=20.0, gt=0.0)


class RondoSpec(BaseModel):
    """Configuration for :class:`~molzoo.rondo.model.Rondo`: encoder and heads.

    What a training stage trains is not a field here: one spec, one
    ``state_dict`` layout, every objective. The energy terms come from the
    force field the model is built on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    encoder: CovalentGraphEncoderSpec = CovalentGraphEncoderSpec()
    heads: HeadsSpec = HeadsSpec()


def validate(**values: Any) -> RondoSpec:
    """Check constructor keywords and return the snapshot the classes keep.

    A caller passes ``width=``, ``n_layers=``, ``hidden=`` and the other fields
    of :class:`CovalentGraphEncoderSpec` and :class:`HeadsSpec`. The snapshot is not a
    constructor argument.
    """
    unknown = set(values) - set(CovalentGraphEncoderSpec.model_fields) - set(HeadsSpec.model_fields)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unknown setting {names}")
    encoder = {key: values[key] for key in values if key in CovalentGraphEncoderSpec.model_fields}
    heads = {key: values[key] for key in values if key in HeadsSpec.model_fields}
    return RondoSpec(encoder=CovalentGraphEncoderSpec(**encoder), heads=HeadsSpec(**heads))
