"""Scale/shift modules for energy heads.

Two orthogonal primitives:

* :class:`GlobalRescale` — ``y = scale * x + shift`` applied to a scalar /
  tensor output (typically the per-graph total energy). Scale and shift are
  non-trainable buffers fit from training data.
* :class:`PerSpeciesScaleShift` — ``y_i = scale[Z_i] * x_i + shift[Z_i]``
  applied to a per-atom tensor. Matches ``nequip``'s ``PerSpeciesScaleShift``.

Neither module mutates the batch — they transform a tensor and return it.
Composition with :class:`molpot.heads.edge_energy.EdgeEnergyHead` is done by
the calling script (explicit wiring, no registry magic).

Reference:
    mir-group/allegro configs ``configs/minimal.yaml`` (``PerSpeciesRescale``
    + ``RescaleEnergyEtc`` builder chain).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalRescale(nn.Module):
    """Affine transform ``y = scale * x + shift`` with frozen buffers.

    Args:
        scale: Scalar multiplier. For Allegro QM9 this is typically the dataset
            ``total_energy_std`` (or per-atom energy std if the head emits
            per-atom energies).
        shift: Scalar additive term. Typically ``0.0`` when AtomicDress has
            already removed the per-species baseline from the target.
    """

    def __init__(self, *, scale: float = 1.0, shift: float = 0.0, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale), dtype=dtype))
        self.register_buffer("shift", torch.tensor(float(shift), dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``scale * x + shift`` (broadcast across ``x``)."""
        return self.scale * x + self.shift

    def extra_repr(self) -> str:
        """Render the scalar ``scale`` and ``shift`` for ``repr(module)``."""
        return f"scale={self.scale.item():.4g}, shift={self.shift.item():.4g}"


class PerSpeciesScaleShift(nn.Module):
    """Per-atom affine with species-indexed scale & shift.

    Args:
        num_species: Size of the lookup table (``Z`` is used directly as the
            index, so this must be ``> max(Z)`` in the dataset).
        scales: Tensor / sequence of length ``num_species`` giving the
            per-species multiplier. Use ``per_atom_energy_std`` per element
            for Allegro QM9.  Dead rows (species never present) may be
            ``1.0``.
        shifts: Tensor / sequence of length ``num_species`` giving the
            per-species additive. Use ``0.0`` when AtomicDress has already
            been applied to the target; otherwise use the ridge-regression
            atomic energies.
        scales_trainable: Make ``scales`` a learnable ``nn.Parameter``
            (otherwise a buffer).
        shifts_trainable: Same for ``shifts``.
        dtype: Storage dtype of the tables.
            Allegro passes ``torch.float64`` (upstream ``PerTypeScaleShift``
            always computes in float64, whatever the model dtype).

    The index need not be ``Z``: any per-atom species index into the table
    works (Allegro passes its type index).

    Shapes:
        * ``x``: per-atom quantity ``(N,)``.
        * ``Z``: atomic numbers ``(N,)``.
        * returns: ``(N,)``.
    """

    def __init__(
        self,
        *,
        num_species: int,
        scales: torch.Tensor | list[float] | None = None,
        shifts: torch.Tensor | list[float] | None = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        dtype: torch.dtype,
    ):
        super().__init__()
        if num_species <= 0:
            raise ValueError(f"num_species must be positive, got {num_species}")

        scales_t = self._as_1d_tensor(scales, num_species, fill=1.0, dtype=dtype)
        shifts_t = self._as_1d_tensor(shifts, num_species, fill=0.0, dtype=dtype)

        for name, value, trainable in (
            ("scales", scales_t, scales_trainable),
            ("shifts", shifts_t, shifts_trainable),
        ):
            if trainable:
                setattr(self, name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)

    @staticmethod
    def _as_1d_tensor(
        value: torch.Tensor | list[float] | None,
        length: int,
        fill: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if value is None:
            return torch.full((length,), fill, dtype=dtype)
        t = torch.as_tensor(value, dtype=dtype)
        if t.ndim != 1 or t.shape[0] != length:
            raise ValueError(f"expected a 1D tensor of length {length}, got shape {tuple(t.shape)}")
        return t

    def forward(self, x: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Return ``scales[Z] * x + shifts[Z]`` elementwise, in the tables' dtype."""
        return self.scales[Z] * x.to(self.scales.dtype) + self.shifts[Z]

    def extra_repr(self) -> str:
        """Render the species count and trainability flag for ``repr(module)``."""
        return (
            f"num_species={self.scales.shape[0]}, "
            f"scales_trainable={isinstance(self.scales, nn.Parameter)}, "
            f"shifts_trainable={isinstance(self.shifts, nn.Parameter)}"
        )
