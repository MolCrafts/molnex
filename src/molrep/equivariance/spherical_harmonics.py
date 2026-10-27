"""Real spherical harmonics in cuEquivariance's ``SO3`` convention.

The module owns the monomial table read off the descriptor
(:class:`.spherical_harmonics_terms.SphericalHarmonicsTerms`) and evaluates it
on the vectors it is given — normalisation, when a caller wants it, happens
before (:class:`molrep.embedding.angular.SphericalHarmonics`). Degrees are
``0..3``; a list outside that raises at construction.

On a CUDA tensor ``molnex_equivariance::harmonics`` evaluates the table, one
thread per vector, with the product rule as its first and second adjoint; on
a CPU tensor :mod:`.reference` does.

Reference:
    Real spherical harmonics in the cuEquivariance ``SO3`` convention.
"""

from __future__ import annotations

import torch
from torch import nn

from . import reference
from .backend import NativeBackend
from .spherical_harmonics_terms import SphericalHarmonicsTerms


class SphericalHarmonics(nn.Module):
    """Evaluate ``Y_lm`` on ``(vectors)`` of shape ``(N, 3)``.

    Args:
        ls: Strictly increasing degrees, each in ``0..3``.
        dtype: ``float32`` or ``float64``. Coefficients are stored in this dtype.
        device: Device of the term-table buffers.
    """

    def __init__(
        self, ls: list[int], *, dtype: torch.dtype, device: torch.device | str | None = None
    ) -> None:
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("native spherical harmonics requires float32 or float64")
        tables = SphericalHarmonicsTerms(ls)
        self.ls = tables.ls
        self.components = tables.components
        self.register_buffer(
            "terms", torch.tensor(tables.terms, dtype=torch.int16, device=device), persistent=False
        )
        self.register_buffer(
            "coefficients",
            torch.tensor(tables.coefficients, dtype=dtype, device=device),
            persistent=False,
        )
        NativeBackend.load()

    def extra_repr(self) -> str:
        return f"ls={self.ls}"

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Evaluate the polynomial.

        Args:
            vectors: ``(N, 3)`` Cartesian components.

        Returns:
            Spherical harmonics ``(N, components)`` in the descriptor's order.
        """
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError("spherical harmonics expects a matrix of shape (N, 3)")
        if vectors.dtype != self.coefficients.dtype:
            raise ValueError("vector dtype does not match the coefficient dtype")
        if vectors.is_cuda:
            return torch.ops.molnex_equivariance.harmonics(
                vectors.contiguous(), self.terms, self.coefficients, self.components
            )
        return reference.spherical_harmonics(
            vectors, self.coefficients, self.terms.long(), self.components
        )
