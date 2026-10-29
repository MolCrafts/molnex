"""Validated configuration for the PiNet2 products.

Torch-free: imports only ``typing``, ``pydantic`` and the encoder spec. The
encoder's fields and rules are :class:`~molrep.encoders.pinet.spec.PiNetEncoderSpec`;
this adds the energy readout.
"""

from __future__ import annotations

from pydantic import Field

from molrep.encoders.pinet.spec import PiNetEncoderSpec

__all__ = ["PiNetSpec"]


class PiNetSpec(PiNetEncoderSpec):
    """Configuration of the PiNet2 products: the encoder's fields and the readout.

    Field names follow ``src/molzoo/specs/pinet2.md`` §6 (PiNN names there).
    """

    #: Energy readout (``OutLayer``) widths and activation; ``None`` is PiNN's.
    out_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    out_activation: str | None = None
    #: Multiplier on the summed network output (PiNN ``1 / e_scale``). PiNN's
    #: ``e_dress`` is not a field: it is :class:`molix.data.AtomicDress`,
    #: outside the network (spec §4).
    scale: float = 1.0
