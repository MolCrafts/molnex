"""Radius graph neighbor search for molpot."""

import torch

from molix.F.locality import get_neighbor_pairs


def radius_graph(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute neighbor list for a set of positions.

    Args:
        pos: Atomic positions [N, 3]
        batch: Batch indices [N]
        cutoff: Cutoff radius

    Returns:
        edge_index: [num_edges, 2]
        edge_vec: [num_edges, 3] (pos_j - pos_i)
    """
    # Note: get_neighbor_pairs expects (positions, cutoff, ...)
    # It returns (neighbors, deltas, distances, number_found_pairs)
    # neighbors is [num_pairs, 2]
    # deltas is [num_pairs, 3] (pos_j - pos_i)

    # We currently don't support batching in get_neighbor_pairs directly if
    # it doesn't take batch index. But we can mask or handle it.
    # Actually, molix backend usually handles PBC but maybe not multiple molecules
    # unless they are separated by PBC.

    # For now, let's assume it handles the whole batch if positions are concatenated.
    # But we need to ensure we don't find neighbors across different molecules.

    # A simple but potentially slow way is to use torch_cluster.radius_graph if available,
    # but we should stick to molix as it's our backend.

    # Let's use get_neighbor_pairs and then mask pairs from different molecules.
    neighbors, deltas, _, _ = get_neighbor_pairs(pos, cutoff)

    # Mask pairs that are across different molecules
    # neighbors: [num_pairs, 2]
    node_i = neighbors[:, 0]
    node_j = neighbors[:, 1]

    mask = batch[node_i] == batch[node_j]

    edge_index = neighbors[mask]  # [num_edges, 2]
    edge_vec = deltas[mask]  # [num_edges, 3]

    return edge_index, edge_vec
