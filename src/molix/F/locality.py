"""Functional API for locality operations (molix).

Input validation (dtype, shape, device) is the caller's responsibility.
The C++ kernel trusts its inputs; these wrappers only prepare sentinel
tensors and forward to ``torch.ops.molix.*``.
"""

from __future__ import annotations

from molix import ensure_op_registered
from torch import Tensor, empty, ops


def get_neighbor_pairs(
    positions: Tensor,
    cutoff: float,
    max_num_pairs: int = -1,
    box_vectors: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return pair indices and geometry for atoms within ``cutoff``.

    Args:
        positions: ``(N, 3)`` float tensor, contiguous.
        cutoff: Distance cutoff. Must be positive.
        max_num_pairs: Fixed output width. ``-1`` returns all ``N*(N-1)/2``
            pairs padded with ``-1`` / ``NaN`` beyond cutoff; otherwise the
            output is padded/truncated to ``max_num_pairs``. Callers should
            compare the returned ``num_pairs`` against ``max_num_pairs``
            to detect overflow.
        box_vectors: ``(3, 3)`` upper-triangular cell for PBC, or ``None``
            for open boundaries.

    Returns:
        ``(neighbors, deltas, distances, num_pairs)`` where ``neighbors`` is
        ``(2, max_num_pairs)`` int32, ``deltas`` is ``(max_num_pairs, 3)``,
        ``distances`` is ``(max_num_pairs,)``, and ``num_pairs`` is a 1-elem
        int32 tensor holding the actual count within cutoff.
    """
    ensure_op_registered("get_neighbor_pairs")

    if box_vectors is None:
        box_vectors = empty((0, 0), device=positions.device, dtype=positions.dtype)

    return ops.molix.get_neighbor_pairs(positions, cutoff, max_num_pairs, box_vectors)
