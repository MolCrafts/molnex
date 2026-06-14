"""Reciprocal-space vector generation for Ewald and mesh-based methods."""

import torch
from torch.nn.utils.rnn import pad_sequence


def get_ns_mesh(cell: torch.Tensor, mesh_spacing: float) -> torch.Tensor:
    """Compute mesh size from target spacing, rounded to powers of 2 for FFT.

    Args:
        cell: Unit cell matrix ``(3, 3)``.
        mesh_spacing: Target spacing in real space.

    Returns:
        Mesh dimensions ``(3,)`` as integer powers of 2.
    """
    basis_norms = torch.linalg.norm(cell, dim=1)
    ns_approx = basis_norms / mesh_spacing
    ns_actual_approx = 2 * ns_approx + 1
    return torch.tensor(2).pow(torch.ceil(torch.log2(ns_actual_approx)).long())


def _generate_kvectors(cell: torch.Tensor, ns: torch.Tensor, for_ewald: bool) -> torch.Tensor:
    """Generate k-vectors on a mesh grid for FFT or Ewald summation.

    Args:
        cell: Unit cell matrix ``(3, 3)``.
        ns: Mesh dimensions ``(3,)``.
        for_ewald: If True, return full explicit grid; otherwise FFT-optimized.

    Returns:
        K-vectors with shape ``(nx, ny, nz, 3)`` for mesh,
        or ``(n_kvecs, 3)`` for Ewald.
    """
    if cell.shape != (3, 3):
        raise ValueError(f"cell of shape {list(cell.shape)} should be of shape (3, 3)")

    if ns.shape != (3,):
        raise ValueError(f"ns of shape {list(ns.shape)} should be of shape (3, )")

    if ns.device != cell.device:
        raise ValueError(
            f"`ns` and `cell` are not on the same device, got {ns.device} and {cell.device}."
        )

    if cell.is_cuda:
        inverse_cell = torch.linalg.inv_ex(cell)[0]
    else:
        inverse_cell = torch.linalg.inv(cell)

    reciprocal_cell = 2 * torch.pi * inverse_cell.T
    bx = reciprocal_cell[0]
    by = reciprocal_cell[1]
    bz = reciprocal_cell[2]

    kxs = (bx * ns[0]) * torch.fft.fftfreq(ns[0], device=cell.device, dtype=cell.dtype).unsqueeze(
        -1
    )
    kys = (by * ns[1]) * torch.fft.fftfreq(ns[1], device=cell.device, dtype=cell.dtype).unsqueeze(
        -1
    )

    if for_ewald:
        kzs = (bz * ns[2]) * torch.fft.fftfreq(
            ns[2], device=cell.device, dtype=cell.dtype
        ).unsqueeze(-1)
    else:
        kzs = (bz * ns[2]) * torch.fft.rfftfreq(
            ns[2], device=cell.device, dtype=cell.dtype
        ).unsqueeze(-1)

    return kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]


def generate_kvectors_for_mesh(cell: torch.Tensor, ns: torch.Tensor) -> torch.Tensor:
    """Compute reciprocal-space vectors for FFT-based mesh calculators.

    Args:
        cell: Unit cell matrix ``(3, 3)``.
        ns: Mesh dimensions ``(3,)`` (preferably powers of 2).

    Returns:
        K-vectors ``(nx, ny, nz//2+1, 3)`` with ``k_vectors[0,0,0] = [0,0,0]``.
    """
    return _generate_kvectors(cell=cell, ns=ns, for_ewald=False)


def generate_kvectors_for_ewald(
    cell: torch.Tensor,
    ns: torch.Tensor,
) -> torch.Tensor:
    """Compute all reciprocal-space vectors for explicit Ewald summation.

    Args:
        cell: Unit cell matrix ``(3, 3)``.
        ns: Number of k-vectors along each axis ``(3,)``.

    Returns:
        K-vectors ``(n_kvecs, 3)`` with ``k_vectors[0] = [0,0,0]``.
    """
    return _generate_kvectors(cell=cell, ns=ns, for_ewald=True).reshape(-1, 3)


def compute_batched_kvectors(
    lr_wavelength: float,
    cells: torch.Tensor,
) -> torch.Tensor:
    """Generate k-vectors for multiple systems in a batch.

    Args:
        lr_wavelength: Spatial resolution for long-range part.
        cells: Unit cell matrices ``(B, 3, 3)``.

    Returns:
        Padded k-vectors ``(B, max_n_kvecs, 3)``.
    """
    all_kvectors = []
    k_cutoff = 2 * torch.pi / lr_wavelength
    for cell in cells:
        basis_norms = torch.linalg.norm(cell, dim=1)
        ns_float = k_cutoff * basis_norms / 2 / torch.pi
        ns = torch.ceil(ns_float).long()
        kvectors = generate_kvectors_for_ewald(ns=ns, cell=cell)
        all_kvectors.append(kvectors)
    return pad_sequence(all_kvectors, batch_first=True)
