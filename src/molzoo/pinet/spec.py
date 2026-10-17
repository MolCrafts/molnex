"""Validated configuration for the PiNet2 family.

Torch-free: imports only ``typing`` and ``pydantic``. Every cross-field rule
is checked here, once, at construction; the modules built from a spec never
re-check it and never read it on a hot path.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PiNetSpec(BaseModel):
    """Configuration of :class:`~molzoo.pinet.encoder.PiNetEncoder` and its products.

    Field names follow ``src/molzoo/specs/pinet2.md`` §6 (PiNN names there).
    """

    model_config = ConfigDict(extra="forbid")

    atomic_numbers: list[int] = Field(default_factory=lambda: [1, 6, 7, 8], min_length=1)
    r_max: float = Field(default=4.0, gt=0.0)
    cutoff_type: Literal["f1", "f2", "hip"] = "f1"
    basis_type: Literal["polynomial", "gaussian"] = "polynomial"
    n_basis: int = Field(default=4, gt=0)
    gamma: float | list[float] = 3.0
    center: float | list[float] | None = None
    pp_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    pi_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    ii_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    depth: int = Field(default=4, gt=0)
    activation: str = "tanh"
    weighted: bool = False
    rank: Literal[1, 3, 5] = 3
    #: Write the per-block ``p3``/``p5`` and per-pair ``i1``/``i3``/``i5``
    #: histories in :meth:`PiNetEncoder.forward` (MolNex-only, spec §4).
    emit_property_features: bool = True
    #: Energy readout (``OutLayer``) widths and activation; ``None`` is PiNN's.
    out_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    out_activation: str | None = None
    #: PiNN ``e_dress`` in ``atomic_numbers`` order (eV); ``None`` is all zero.
    atomic_energies: list[float] | None = None
    #: Multiplier on the summed network output (PiNN ``1 / e_scale``).
    scale: float = 1.0

    @model_validator(mode="after")
    def _check(self) -> PiNetSpec:
        if any(b <= a for a, b in zip(self.atomic_numbers, self.atomic_numbers[1:])):
            raise ValueError("atomic_numbers must be strictly ascending")
        if self.pp_nodes[-1] != self.ii_nodes[-1]:
            raise ValueError("pp_nodes[-1] must equal ii_nodes[-1] (scalar gating of P3/P5)")
        if self.atomic_energies is not None and len(self.atomic_energies) != len(
            self.atomic_numbers
        ):
            raise ValueError("atomic_energies must have one entry per atomic number")
        return self
