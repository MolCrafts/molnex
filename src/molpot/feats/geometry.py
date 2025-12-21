"""Geometry utilities for distance and angle calculations."""

import torch


def compute_distances(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Compute distances from positions and edge indices.
    
    Args:
        pos: Atomic positions [N, 3]
        edge_index: Edge connectivity [2, num_edges]
        
    Returns:
        Distances [num_edges]
    """
    pos_i = pos[edge_index[0]]  # [num_edges, 3]
    pos_j = pos[edge_index[1]]  # [num_edges, 3]
    
    edge_vec = pos_j - pos_i  # [num_edges, 3]
    distances = torch.norm(edge_vec, dim=-1)  # [num_edges]
    
    return distances


def compute_angles(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    angle_index: torch.Tensor,
) -> torch.Tensor:
    """Compute angles between edges.
    
    Args:
        pos: Atomic positions [N, 3]
        edge_index: Edge connectivity [2, num_edges]
        angle_index: Angle indices [2, num_angles] where each column is (edge_i, edge_j)
        
    Returns:
        Angles in radians [num_angles]
    """
    # Get edge vectors
    pos_i = pos[edge_index[0]]  # [num_edges, 3]
    pos_j = pos[edge_index[1]]  # [num_edges, 3]
    edge_vec = pos_j - pos_i  # [num_edges, 3]
    
    # Get pairs of edges for angles
    vec_i = edge_vec[angle_index[0]]  # [num_angles, 3]
    vec_j = edge_vec[angle_index[1]]  # [num_angles, 3]
    
    # Compute angles using dot product
    cos_angle = torch.sum(vec_i * vec_j, dim=-1) / (
        torch.norm(vec_i, dim=-1) * torch.norm(vec_j, dim=-1) + 1e-8
    )
    
    # Clamp to avoid numerical issues with acos
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    angles = torch.acos(cos_angle)
    
    return angles


def compute_dihedrals(
    pos: torch.Tensor,
    dihedral_index: torch.Tensor,
) -> torch.Tensor:
    """Compute dihedral angles.
    
    Args:
        pos: Atomic positions [N, 3]
        dihedral_index: Dihedral indices [4, num_dihedrals] where each column is (i, j, k, l)
        
    Returns:
        Dihedral angles in radians [num_dihedrals]
    """
    # Get positions for each dihedral
    pos_i = pos[dihedral_index[0]]  # [num_dihedrals, 3]
    pos_j = pos[dihedral_index[1]]
    pos_k = pos[dihedral_index[2]]
    pos_l = pos[dihedral_index[3]]
    
    # Compute vectors
    b1 = pos_j - pos_i
    b2 = pos_k - pos_j
    b3 = pos_l - pos_k
    
    # Compute normal vectors
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    
    # Normalize
    n1 = n1 / (torch.norm(n1, dim=-1, keepdim=True) + 1e-8)
    n2 = n2 / (torch.norm(n2, dim=-1, keepdim=True) + 1e-8)
    
    # Compute dihedral angle
    cos_dihedral = torch.sum(n1 * n2, dim=-1)
    cos_dihedral = torch.clamp(cos_dihedral, -1.0, 1.0)
    
    dihedrals = torch.acos(cos_dihedral)
    
    return dihedrals
