"""Utility functions for equivariance testing.

This module provides tools for testing SO(3) equivariance of neural network layers,
including rotation matrix generation and transformation of features.
"""

import math

import cuequivariance as cue
import torch


def random_rotation_matrix(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate a random 3x3 rotation matrix using QR decomposition.

    Args:
        dtype: Data type for the rotation matrix.

    Returns:
        Random 3x3 rotation matrix with determinant +1.
    """
    # Generate random matrix
    M = torch.randn(3, 3, dtype=dtype)

    # QR decomposition
    Q, R = torch.linalg.qr(M)

    # Ensure determinant is +1 (proper rotation)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q


def rotation_matrix_z(angle: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate rotation matrix around z-axis.

    Args:
        angle: Rotation angle in radians.
        dtype: Data type for the rotation matrix.

    Returns:
        3x3 rotation matrix for rotation around z-axis.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=dtype)


def rotation_matrix_x(angle: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate rotation matrix around x-axis.

    Args:
        angle: Rotation angle in radians.
        dtype: Data type for the rotation matrix.

    Returns:
        3x3 rotation matrix for rotation around x-axis.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return torch.tensor([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]], dtype=dtype)


def rotation_matrix_y(angle: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate rotation matrix around y-axis.

    Args:
        angle: Rotation angle in radians.
        dtype: Data type for the rotation matrix.

    Returns:
        3x3 rotation matrix for rotation around y-axis.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return torch.tensor([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=dtype)


def rotate_vectors(vectors: torch.Tensor, rot_matrix: torch.Tensor) -> torch.Tensor:
    """Rotate vectors using rotation matrix.

    Args:
        vectors: Input vectors with shape (..., 3).
        rot_matrix: 3x3 rotation matrix.

    Returns:
        Rotated vectors with same shape as input.
    """
    return vectors @ rot_matrix.T


def rotate_irreps_features_simple(
    features: torch.Tensor,
    rot_matrix: torch.Tensor,
    irreps: str,
) -> torch.Tensor:
    """Rotate features for simple irreps (scalars and vectors only).

    This is a simplified rotation function that handles l=0 (scalars) and l=1 (vectors).
    For higher-order irreps, use Wigner D matrices.

    Args:
        features: Input features with shape (n_nodes, irreps_dim) in ir_mul layout.
        rot_matrix: 3x3 rotation matrix.
        irreps: Irreps string (e.g., "64x0e + 32x1o").

    Returns:
        Rotated features with same shape as input.
    """
    # Parse irreps
    irreps_obj = cue.Irreps("O3", irreps)

    n_nodes = features.shape[0]
    rotated = features.clone()

    offset = 0
    for mul_irrep in irreps_obj:
        mul = mul_irrep.mul
        l = mul_irrep.ir.l
        dim_l = 2 * l + 1

        if l == 0:
            # Scalars are invariant under rotation
            offset += mul * dim_l
        elif l == 1:
            # Vectors: rotate each vector independently
            # In ir_mul layout: features are ordered as (ir_dim, mul)
            # So for l=1 with mul=M, we have shape (3, M) flattened to (3*M,)
            # which means: [x1, x2, ..., xM, y1, y2, ..., yM, z1, z2, ..., zM]
            vec_block = features[:, offset : offset + mul * 3]  # (n_nodes, 3*mul)
            vec_block_reshaped = vec_block.reshape(n_nodes, 3, mul)  # (n_nodes, 3, mul)

            # Apply rotation to xyz components
            vec_block_rotated = torch.einsum("ij,njm->nim", rot_matrix, vec_block_reshaped)

            # Reshape back
            rotated[:, offset : offset + mul * 3] = vec_block_rotated.reshape(n_nodes, 3 * mul)
            offset += mul * dim_l
        else:
            # Higher-order irreps: for proper equivariance, need Wigner D matrices
            # For now, we just copy them (not equivariant!)
            # This function should only be used for testing scalars and vectors
            offset += mul * dim_l

    return rotated


def check_equivariance(
    output1_rotated: torch.Tensor,
    output2: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> bool:
    """Check if two tensors are approximately equal for equivariance testing.

    This function verifies the equivariance property:
        f(Rx) = Rf(x)
    where R is a rotation, f is the layer, and x is the input.

    Args:
        output1_rotated: Rotated output from original input (R·f(x)).
        output2: Output from rotated input (f(R·x)).
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        True if tensors are approximately equal, False otherwise.
    """
    return torch.allclose(output1_rotated, output2, rtol=rtol, atol=atol)
