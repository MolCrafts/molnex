"""Validated configuration for the PiNet2 products.

Torch-free: imports only ``typing``, ``pydantic`` and the encoder spec. The
encoder's fields and rules are :class:`~molrep.encoders.pinet.spec.PiNetEncoderSpec`;
this adds the energy readout.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from molrep.encoders.pinet.spec import PiNetEncoderSpec

__all__ = ["PiNetSpec"]


class PiNetSpec(PiNetEncoderSpec):
    """Configuration of the PiNet2 products: the encoder's fields and the readout.

    Field names follow ``src/molzoo/specs/pinet2.md`` §6 (PiNN names there).
    """

    #: Energy readout (``OutLayer``) widths and activation; ``None`` is PiNN's.
    out_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    out_activation: str | None = None
    #: PiNN ``e_dress`` in ``atomic_numbers`` order (eV); ``None`` is all zero.
    atomic_energies: list[float] | None = None
    #: Multiplier on the summed network output (PiNN ``1 / e_scale``).
    scale: float = 1.0

    @model_validator(mode="after")
    def _check_readout(self) -> PiNetSpec:
        if self.atomic_energies is not None and len(self.atomic_energies) != len(
            self.atomic_numbers
        ):
            raise ValueError("atomic_energies must have one entry per atomic number")
        return self
