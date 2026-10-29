"""The frozen elemental energy table a training residual and a deployed model share.

An atomic dress is a constant per-element energy ``c[Z]``. Training subtracts
it from the label (:class:`~molix.data.EnergyResidual`, on the loader) so the
network learns the residual; a deployed or absolute evaluation adds it back
(:class:`~molix.deploy.DressedPotential`, inside the exported graph). Both read
the same :class:`AtomicDress`, so the subtract and the add are one set of
numbers. Because ``c[Z]`` does not depend on positions, forces and virial of
the residual equal those of the absolute energy.

How the numbers are obtained is a separate type per algorithm, each returning an
:class:`AtomicDress` (:class:`~molix.data.LeastSquaresDress`,
:class:`~molix.data.SuppliedDress`); the table does not fit.

Reference:
    Shao et al. "PiNN: A Python Library for Building Atomic Neural Networks of
    Molecules and Materials" J. Chem. Inf. Model. 60, 1184 (2020).
    https://doi.org/10.1021/acs.jcim.9b00994 (``pinn/utils.py``
    ``atomic_dress`` / ``get_atomic_dress``).
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn


class AtomicDress(nn.Module):
    """One frozen Z-indexed elemental energy set ``c[Z]``, plus a per-graph intercept ``b``.

    Applied to a graph ``g``: ``e0(g) = sum_{i in g} c[Z_i] + b``. The intercept
    is one constant per graph, not per atom, so it has no atom attribution; a
    dress with ``b != 0`` serves a residual label but not an exported atom-energy
    model (:class:`~molix.deploy.DressedPotential` refuses it).

    Both buffers are persistent CPU clones of the constructor arguments, so a
    later write to the caller's tensors is invisible and a DataLoader worker
    never unpickles a CUDA tensor. The digest is computed once here; an
    in-place write into a buffer, or a ``load_state_dict`` into a dress an
    :class:`~molix.data.EnergyResidual` already holds, is undefined — a new
    table is a new ``AtomicDress``. Move a deep copy, never the loader's
    instance, to put the table on a device.

    Args:
        e0: ``(Z_max + 1,)`` per-element energy, indexed by atomic number, in
            the unit of ``graphs.energy`` (eV). Stored as float64.
        support: ``(Z_max + 1,)`` bool. True where ``e0[Z]`` is defined; a zero
            coefficient with ``support[Z] == False`` is no coefficient.
        intercept: Per-graph constant ``b`` in the same unit. ``0.0`` is a
            purely elemental dress.

    Attributes:
        e0: The float64 table buffer.
        support: The bool support buffer.
        intercept: The float64 scalar buffer ``b``.

    Raises:
        ValueError: If either argument is not 1-D, the lengths differ, the
            table is empty, or ``support`` is not ``torch.bool``.
    """

    e0: torch.Tensor
    support: torch.Tensor
    intercept: torch.Tensor

    def __init__(self, *, e0: torch.Tensor, support: torch.Tensor, intercept: float = 0.0) -> None:
        super().__init__()
        if e0.ndim != 1:
            raise ValueError(
                f"AtomicDress e0 must be 1-D (got ndim={e0.ndim}, shape={tuple(e0.shape)})"
            )
        if support.ndim != 1:
            raise ValueError(
                "AtomicDress support must be 1-D "
                f"(got ndim={support.ndim}, shape={tuple(support.shape)})"
            )
        if e0.shape[0] != support.shape[0]:
            raise ValueError(
                "AtomicDress e0 and support must have the same length "
                f"(got {e0.shape[0]} and {support.shape[0]})"
            )
        if e0.shape[0] < 1:
            raise ValueError("AtomicDress needs at least one slot; no dress is no AtomicDress")
        if support.dtype != torch.bool:
            raise ValueError(
                f"AtomicDress support.dtype must be torch.bool (got {support.dtype}; "
                "uint8 is not bool)"
            )
        table = e0.detach().to(device="cpu", dtype=torch.float64).clone()
        mask = support.detach().to(device="cpu").clone()
        self.register_buffer("e0", table, persistent=True)
        self.register_buffer("support", mask, persistent=True)
        offset = torch.tensor(float(intercept), dtype=torch.float64)
        self.register_buffer("intercept", offset, persistent=True)
        table_bytes = table.contiguous().numpy().astype("<f8", copy=False).tobytes()
        mask_bytes = mask.contiguous().to(dtype=torch.uint8).numpy().tobytes()
        offset_bytes = offset.reshape(1).numpy().astype("<f8", copy=False).tobytes()
        self._digest = hashlib.sha256(table_bytes + mask_bytes + offset_bytes).hexdigest()[:16]

    @property
    def digest(self) -> str:
        """16 hex chars of SHA-256 over float64 ``e0``, uint8 ``support``, float64 ``intercept``.

        Little-endian bytes, in that order.

        Computed once at construction: the identity of the numbers, not of the
        algorithm that produced them.
        """
        return self._digest

    def per_atom(self, Z: torch.Tensor) -> torch.Tensor:
        """Gather ``c[Z]`` per atom, float64 ``(N,)``.

        Compile-clean: no host sync, no clamp, no ``support`` read. An atomic
        number outside the table is undefined here; the loader
        (:class:`~molix.data.EnergyResidual`) is where an unsupported real atom
        raises.

        Args:
            Z: Atomic numbers ``(N,)``.

        Returns:
            Per-atom dress ``(N,)`` float64, on the buffer's device.
        """
        return self.e0[Z.long()]

    def restore(self, residual: torch.Tensor, graph_e0: torch.Tensor) -> torch.Tensor:
        """Absolute energy ``residual + e0`` of an undressed forward, added in float64.

        ``graph_e0`` already holds the intercept (it is ``graphs.e0``).

        The inverse of :class:`~molix.data.EnergyResidual`'s subtract. Calling it
        on the output of :class:`~molix.deploy.DressedPotential` counts ``e0``
        twice.

        Args:
            residual: Per-graph residual energy ``(B,)``.
            graph_e0: Per-graph dress sum ``(B,)`` (``graphs.e0``).

        Returns:
            ``(B,)`` absolute energy in ``residual.dtype``.
        """
        total = residual.to(dtype=torch.float64) + graph_e0.to(dtype=torch.float64)
        return total.to(dtype=residual.dtype)
