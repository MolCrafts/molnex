"""
Functional API for locality operations (molix)
"""

from torch import empty, ops, Tensor


def get_neighbor_pairs(
    positions: Tensor,
    cutoff: float,
    max_num_pairs: int = -1,
    box_vectors: Tensor | None = None,
    check_errors: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns indices and distances of atom pairs within a given cutoff distance.

    Mirrors behavior of the prior molnex implementation; calls torch.ops.molnex backend.
    """
    if box_vectors is None:
        box_vectors = empty((0, 0), device=positions.device, dtype=positions.dtype)

    neighbors, deltas, distances, number_found_pairs = ops.neighbors.getNeighborPairs(
        positions=positions,
        cutoff=cutoff,
        max_num_neighbors=max_num_pairs,
        box_vectors=box_vectors,
        checkErrors=check_errors,
    )
    return neighbors, deltas, distances, number_found_pairs
