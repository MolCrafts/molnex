"""A per-atom weight column that drops reference-force outliers.

``atoms.force_weight`` is ``1`` on atoms whose reference force magnitude is
within a declared bound and ``0`` elsewhere, times ``atoms.mask``. It is the
column :func:`molix.core.losses.force_loss` multiplies into the residual —
padding ghosts and outliers are the same thing to that function, a row that
should not count, so they share one column rather than one code path each.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase

from molix.data.task import BatchTask
from molix.frame import as_tensor_frame

__all__ = ["ForceInlierWeight"]


class ForceInlierWeight(BatchTask):
    """Write a per-atom ``0``/``1`` weight from a reference-force bound.

    For each atom ``i`` with structural mask ``m_i`` (``atoms.mask``, else
    all-True):

    ``w_i = m_i * [ ||F_ref,i|| <= threshold ]``

    **Why a weight and not a filter.** Selecting rows by a boolean is a
    data-dependent shape, which ``torch.compile`` can only graph-break on. A
    multiplicative weight keeps the shape static and costs one elementwise
    comparison, and :func:`~molix.core.losses.force_loss` divides a mean by
    the weight it kept, so the reported value does not move with how many rows
    a batch happens to drop.

    **Why the bound has no default.** It is in the unit of
    ``atoms.forces`` — kcal mol⁻¹ Å⁻¹ on a LAMMPS-real batch, eV Å⁻¹ on an
    OMol25 one — and it states a physical claim about the model being fitted:
    *this functional form cannot represent references above here, and chasing
    them costs more than it buys*. A default would be that claim made on the
    caller's behalf in a unit this class cannot see.

    Args:
        threshold: Reference-force magnitude bound, in the unit of
            ``atoms.forces``. Atoms at or below it keep weight ``1``.
        out_key: Column written under ``atoms``. Pass the same string as
            ``force_loss(weight_key=...)``.

    Raises:
        ValueError: ``threshold`` is not finite and positive.
    """

    def __init__(self, *, threshold: float, out_key: str = "force_weight") -> None:
        value = float(threshold)
        if not (value > 0.0) or value != value or value == float("inf"):
            raise ValueError(f"threshold must be finite and positive, got {threshold!r}")
        self.threshold = value
        self.out_key = str(out_key)
        self._task_id = f"force_inlier:{self.out_key}:{self.threshold:.6g}"

    @property
    def task_id(self) -> str:
        """Cache-key identity ``force_inlier:<out_key>:<threshold>``."""
        return self._task_id

    def execute(self, data: TensorDict | dict) -> TensorDict | dict:
        """Write ``atoms.<out_key>``.

        Args:
            data: Collated batch carrying ``atoms.forces`` ``(N, 3)``.
                Optional ``atoms.mask`` ``(N,)`` (True = real atom).

        Returns:
            The same batch, with the weight column written.

        Raises:
            TypeError: ``data`` is a plain ``dict`` — the write would land in
                a container the caller does not hold.
            KeyError: The batch carries no ``atoms.forces`` to bound.
        """
        if not isinstance(data, TensorDictBase):
            raise TypeError(
                f"{type(self).__name__} writes into the caller's batch and needs a "
                f"TensorDict, got {type(data).__name__}"
            )
        frame = as_tensor_frame(data)
        atoms = frame["atoms"]
        if "forces" not in atoms.keys():
            raise KeyError(
                f"{type(self).__name__} needs atoms.forces to bound; batch has "
                f"{sorted(atoms.keys())}"
            )
        reference: torch.Tensor = atoms["forces"]
        weight = (reference.norm(dim=-1) <= self.threshold).to(dtype=reference.dtype)
        if "mask" in atoms.keys():
            weight = weight * atoms["mask"].to(dtype=weight.dtype)
        atoms[self.out_key] = weight
        return data
