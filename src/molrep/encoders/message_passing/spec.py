"""Validated configuration of the message-passing encoder.

Torch-free: imports only ``typing`` and ``pydantic``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["DiscreteChannel", "MessagePassingEncoderSpec"]


class DiscreteChannel(BaseModel):
    """One integer column of the batch, embedded into a feature stream.

    Attributes:
        name: The column, under ``atoms`` (``per="node"``), ``graphs``
            (``per="graph"``) or ``bonds`` (a bond channel).
        num_classes: Table size. An index outside ``[0, num_classes)`` after
            the offset is clamped to the nearest class (ByteFF clamps its
            charges to ±5 / ±2 the same way).
        offset: Added before the lookup, so a signed column such as a formal
            charge of ``-2`` lands on row ``0``.
        per: ``"node"`` columns have one row per atom (or per bond); a
            ``"graph"`` column has one row per molecule and is broadcast to its
            atoms through ``atoms.batch``.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    num_classes: int = Field(gt=0)
    offset: int = 0
    per: Literal["node", "graph"] = "node"


class MessagePassingEncoderSpec(BaseModel):
    """Configuration of :class:`~molrep.encoders.message_passing.encoder.MessagePassingEncoder`.

    Attributes:
        atom_dim: Per-atom feature width.
        bond_dim: Per-bond feature width.
        num_elements: Atomic-number embedding table size.
        hidden_dim: Hidden width of the hop MLPs; ``None`` is ``max(atom_dim, bond_dim)``.
        atom_channels: Extra integer atom (or graph) columns summed into the
            atom seed, e.g. ``formal_charge``; each is a required batch column.
        bond_channels: Extra integer bond columns summed into the bond seed and
            every hop, e.g. ``bond_type``.
        n_blocks: Covalent hops; the receptive field of an atom feature is
            ``n_blocks`` bonds.
        local_attention: Softmax-normalise each atom's aggregation over its
            own incident covalent bonds instead of summing them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    atom_dim: int = Field(default=32, gt=0)
    bond_dim: int = Field(default=32, gt=0)
    num_elements: int = Field(default=119, gt=0)
    hidden_dim: int | None = Field(default=None, gt=0)
    atom_channels: tuple[DiscreteChannel, ...] = ()
    bond_channels: tuple[DiscreteChannel, ...] = ()
    n_blocks: int = Field(default=5, ge=0)
    local_attention: bool = False
