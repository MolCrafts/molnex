"""Chemical support indexing via an L2 k-nearest-neighbour bank.

Thin perception-side bookkeeping for learnable classical force fields:
store training embeddings (or discrete type-id points) and query whether
new vectors fall inside a support radius. No molpot imports; no active-
learning loop.

Reference:
    Spec: learnable-classical-ff-09-provenance
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

__all__ = ["ChemicalSupportIndex"]


class ChemicalSupportIndex:
    """L2 kNN bank of chemical-support reference vectors.

    Built from continuous training embeddings / fingerprints, or from a
    discrete type-id set via :meth:`from_type_ids` (1-D points). Membership
    is ``min L2 distance <= radius``.

    Args:
        bank: Support vectors ``(n_support, dim)``.
        radius: Inclusive L2 radius for :meth:`contains`.
        k: Number of nearest neighbours returned by :meth:`knn`.

    Raises:
        ValueError: If ``bank`` is not 2-D, empty, or ``k`` / ``radius``
            are invalid.
    """

    def __init__(
        self,
        bank: torch.Tensor,
        *,
        radius: float,
        k: int = 1,
    ) -> None:
        if bank.ndim != 2:
            raise ValueError(
                f"ChemicalSupportIndex bank must be 2-D (n, dim); got shape {tuple(bank.shape)}"
            )
        if bank.shape[0] == 0:
            raise ValueError("ChemicalSupportIndex bank must contain at least one vector")
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if radius < 0:
            raise ValueError(f"radius must be >= 0; got {radius}")
        # Detach + clone so the bank is a pure lookup table (no grad graph).
        self._bank = bank.detach().clone()
        self._radius = float(radius)
        self._k = min(int(k), int(self._bank.shape[0]))

    @classmethod
    def from_type_ids(
        cls,
        type_ids: Iterable[int],
        *,
        radius: float = 0.0,
        k: int = 1,
    ) -> ChemicalSupportIndex:
        """Build a 1-D L2 bank from discrete type identifiers.

        Each type id becomes a point ``[float(id)]``. With ``radius=0`` this
        is exact set membership under L2.

        Args:
            type_ids: Iterable of integer type ids in the known support.
            radius: Inclusive L2 radius (default exact match).
            k: Neighbours for :meth:`knn`.

        Returns:
            Index over unique type-id points.
        """
        unique = sorted({int(t) for t in type_ids})
        if not unique:
            raise ValueError("from_type_ids requires at least one type id")
        bank = torch.tensor(unique, dtype=torch.float64).unsqueeze(-1)
        return cls(bank, radius=radius, k=k)

    @property
    def bank(self) -> torch.Tensor:
        """Support vectors ``(n_support, dim)``."""
        return self._bank

    @property
    def radius(self) -> float:
        """Inclusive L2 support radius."""
        return self._radius

    @property
    def k(self) -> int:
        """Number of neighbours used by :meth:`knn`."""
        return self._k

    @property
    def n_support(self) -> int:
        """Number of bank vectors."""
        return int(self._bank.shape[0])

    def knn(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return L2 distances and indices of the ``k`` nearest bank members.

        Args:
            query: Query vectors ``(m, dim)`` matching bank feature dim.

        Returns:
            ``(distances, indices)`` each of shape ``(m, k)``. Distances are
            non-negative L2 norms; indices index into :attr:`bank`.
        """
        q = self._validate_query(query)
        if q.shape[0] == 0:
            return (
                torch.empty(0, self._k, dtype=self._bank.dtype, device=q.device),
                torch.empty(0, self._k, dtype=torch.long, device=q.device),
            )
        # (m, n) pairwise L2
        # cdist is the primitive for kNN L2 banks; keeps this module free of
        # third-party ANN deps.
        dists = torch.cdist(q, self._bank.to(device=q.device, dtype=q.dtype), p=2)
        k = self._k
        values, indices = torch.topk(dists, k=k, dim=-1, largest=False, sorted=True)
        return values, indices

    def min_distance(self, query: torch.Tensor) -> torch.Tensor:
        """L2 distance to the nearest bank member for each query row.

        Args:
            query: Query vectors ``(m, dim)``.

        Returns:
            Distances ``(m,)``.
        """
        dist, _ = self.knn(query)
        if dist.numel() == 0:
            return dist.reshape(0)
        return dist[:, 0]

    def contains(self, query: torch.Tensor) -> torch.Tensor:
        """Whether each query lies inside the support radius.

        Args:
            query: Query vectors ``(m, dim)``.

        Returns:
            Boolean mask ``(m,)`` — ``True`` where ``min L2 <= radius``.
        """
        return self.min_distance(query) <= self._radius

    def coverage_fraction(self, query: torch.Tensor) -> float:
        """Fraction of query rows inside the support radius.

        Empty queries are defined as fully covered (``1.0``) so callers can
        treat "no predictions" as no coverage gap.

        Args:
            query: Query vectors ``(m, dim)``.

        Returns:
            Scalar in ``[0, 1]``.
        """
        if query.shape[0] == 0:
            return 1.0
        mask = self.contains(query)
        return float(mask.float().mean().item())

    def contains_type_ids(self, type_ids: torch.Tensor) -> torch.Tensor:
        """Membership for integer type ids against a 1-D type-id bank.

        Args:
            type_ids: Integer ids ``(m,)``.

        Returns:
            Boolean mask ``(m,)``.
        """
        if type_ids.ndim != 1:
            raise ValueError(f"type_ids must be 1-D; got shape {tuple(type_ids.shape)}")
        query = type_ids.to(dtype=torch.float64).unsqueeze(-1)
        # Compare in bank dtype/device without forcing caller's long tensor.
        return self.contains(query.to(device=self._bank.device))

    def coverage_fraction_type_ids(self, type_ids: torch.Tensor) -> float:
        """Coverage fraction for integer type-id queries.

        Args:
            type_ids: Integer ids ``(m,)``.

        Returns:
            Scalar in ``[0, 1]``.
        """
        if type_ids.numel() == 0:
            return 1.0
        mask = self.contains_type_ids(type_ids)
        return float(mask.float().mean().item())

    def _validate_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 2:
            raise ValueError(f"query must be 2-D (m, dim); got shape {tuple(query.shape)}")
        if query.shape[-1] != self._bank.shape[-1]:
            raise ValueError(f"query dim {query.shape[-1]} != bank dim {self._bank.shape[-1]}")
        return query
