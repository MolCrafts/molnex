"""Validated configuration snapshot for :class:`~molzoo.pinet.encoder.PiNet`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PiNetSpec(BaseModel):
    """Configuration snapshot for :class:`~molzoo.pinet.encoder.PiNet`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    atom_types: list[int] = Field(default_factory=lambda: [1, 6, 7, 8], min_length=1)
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
    #: Emit the per-block ``p3``/``p5`` and ``i1``/``i3``/``i5`` tracks that the
    #: PiNet property heads consume. See :class:`~molzoo.pinet.encoder.PiNet`.
    emit_property_features: bool = True
