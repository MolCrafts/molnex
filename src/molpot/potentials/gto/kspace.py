"""Official Polar GTO reciprocal-space electrostatics.

Not PME. Polar SI V.6 is the analytic GTO Fourier transform plus a **direct
reciprocal-lattice sum** (half-sphere, conjugate symmetry). This module is
PyTorch on purpose: ``E`` depends on ``cell`` through the k-vectors, so
``∂E/∂pos`` and the virial ``∂E/∂ε`` are ordinary autograd. A fused CUDA
kernel would need a custom ``autograd.Function`` for both.

Dump Gaussian convention ``erf(r/(2σ))/r`` (not paper Eq. 37). Two Gaussians
of width ``σ`` have form factor ``exp(-σ² k² / 2)`` each, so the pair kernel
in reciprocal space is ``exp(-σ² k²) / k²``. Energy in eV::

    E = eV_A · (2π / Ω) Σ_{k ≠ 0} exp(-σ² k²) / k² |S(k)|²
    S(k) = Σ_i (q_i + i k · μ_i) exp(i k · r_i)

Cutoff (training ``--kspace_cutoff_factor=κ``, SI typical ``κ = 1.5``)::

    k_cutoff = κ · 2.25 / σ_min · (l_max + 1)^{0.3}

Reference:
    Batatia et al. "MACE-POLAR-1" arXiv:2602.19411, §II.2.6b and SI V.6.
"""

from __future__ import annotations

import math

import torch

from molpot.potentials.elec.prefactors import eV_A

_TWOPI = 2.0 * math.pi
_FOURPI = 4.0 * math.pi


def kspace_cutoff(sigma: float, l_max: int, kappa: float = 1.0) -> float:
    """Polar SI V.6 reciprocal cutoff in Å⁻¹.

    Args:
        sigma: Smallest GTO width in Å.
        l_max: Multipole order entering the cutoff exponent.
        kappa: ``kspace_cutoff_factor``. Training dump default is ``1.0``.

    Returns:
        ``k_cutoff`` in Å⁻¹.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0 Å, got {sigma}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be > 0, got {kappa}")
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}")
    return float(kappa) * 2.25 / float(sigma) * (float(l_max) + 1.0) ** 0.3


def cartesian_dipole(feats: torch.Tensor) -> torch.Tensor:
    """``ir_mul`` ``(q, py, pz, px)`` → Cartesian ``μ = (px, py, pz)``.

    Args:
        feats: Multipoles ``(N, >= 4)``.

    Returns:
        Dipoles ``(N, 3)`` in e·Å.
    """
    return torch.stack((feats[:, 3], feats[:, 1], feats[:, 2]), dim=-1)


def _as_cell(cell: torch.Tensor) -> torch.Tensor:
    """Squeeze a ``(1, 3, 3)`` batch cell to ``(3, 3)``."""
    if cell.dim() == 3:
        if cell.shape[0] != 1:
            raise ValueError(
                f"GTO k-space is single-system: cell must be (3, 3) or (1, 3, 3), "
                f"got {tuple(cell.shape)}"
            )
        cell = cell[0]
    if tuple(cell.shape) != (3, 3):
        raise ValueError(f"cell must be (3, 3), got {tuple(cell.shape)}")
    return cell


def half_sphere_kvectors(
    cell: torch.Tensor,
    k_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integer reciprocal grid, half-sphere, ``k ≠ 0`` and ``|k| < k_cutoff``.

    Args:
        cell: Lattice vectors ``(3, 3)`` Å, one vector per row.
        k_cutoff: Radius in Å⁻¹.

    Returns:
        ``kvec (M, 3)`` Å⁻¹, ``k_sq (M,)``, conjugate-symmetry ``factors (M,)``
        (``2`` on a ±k pair, never ``k = 0``).
    """
    cell = _as_cell(cell)
    device = cell.device
    dtype = cell.dtype
    g = _TWOPI * torch.linalg.inv(cell).T
    k_cut = float(k_cutoff)
    nk = [max(1, int(torch.linalg.norm(cell[i]).item() * k_cut / _TWOPI) + 1) for i in range(3)]
    # A blown-up MD cell makes the mesh O(L³). 32³-odd n-range is already
    # millions of k-points; fail closed instead of allocating tens of GiB.
    _max_nk = 32
    if max(nk) > _max_nk:
        raise RuntimeError(
            f"GTO k-grid exploded: nk={tuple(nk)} for |cell|="
            f"{tuple(float(torch.linalg.norm(cell[i])) for i in range(3))} Å, "
            f"k_cutoff={k_cut} Å⁻¹. The lattice is unphysically large — fix the "
            f"MD (forces / barostat) rather than summing this mesh."
        )
    n1 = torch.arange(-nk[0], nk[0] + 1, device=device)
    n2 = torch.arange(-nk[1], nk[1] + 1, device=device)
    n3 = torch.arange(-nk[2], nk[2] + 1, device=device)
    nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
    kvec = nvec.to(dtype=dtype) @ g
    k_sq = (kvec * kvec).sum(dim=1)
    keep = (k_sq > 0.0) & (k_sq < k_cut * k_cut)
    nvec = nvec[keep]
    kvec = kvec[keep]
    k_sq = k_sq[keep]
    nonzero = nvec != 0
    first = torch.argmax(nonzero.to(torch.long), dim=1)
    sign = torch.gather(nvec, 1, first.unsqueeze(1)).squeeze(-1)
    hemisphere = sign > 0
    kvec = kvec[hemisphere]
    k_sq = k_sq[hemisphere]
    factors = torch.full((kvec.shape[0],), 2.0, dtype=dtype, device=device)
    return kvec, k_sq, factors


def structure_factor(
    pos: torch.Tensor,
    charges: torch.Tensor,
    kvec: torch.Tensor,
    dipoles: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``S(k) = Σ_i (q_i + i k·μ_i) exp(i k · r_i)``.

    Args:
        pos: Positions ``(N, 3)`` Å.
        charges: Monopoles ``(N,)`` e.
        kvec: Reciprocal vectors ``(M, 3)`` Å⁻¹.
        dipoles: Optional Cartesian dipoles ``(N, 3)`` e·Å.

    Returns:
        ``S_real (M,)``, ``S_imag (M,)``.
    """
    kr = pos @ kvec.T
    cos_kr = torch.cos(kr)
    sin_kr = torch.sin(kr)
    s_real = charges @ cos_kr
    s_imag = charges @ sin_kr
    if dipoles is not None:
        uk = dipoles @ kvec.T
        s_real = s_real - (uk * sin_kr).sum(dim=0)
        s_imag = s_imag + (uk * cos_kr).sum(dim=0)
    return s_real, s_imag


def hartree_energy(
    pos: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    *,
    sigma: float,
    k_cutoff: float,
    dipoles: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scalar GTO Hartree energy in eV for one periodic cell.

    Args:
        pos: ``(N, 3)`` Å.
        charges: ``(N,)`` e.
        cell: ``(3, 3)`` Å.
        sigma: GTO width Å (dump ``1.5``).
        k_cutoff: Reciprocal cutoff Å⁻¹.
        dipoles: Optional ``(N, 3)`` e·Å.

    Returns:
        Scalar energy ``()`` in eV.
    """
    cell = _as_cell(cell)
    volume = torch.linalg.det(cell).abs()
    kvec, k_sq, factors = half_sphere_kvectors(cell, k_cutoff)
    if kvec.numel() == 0:
        return charges.new_zeros(())
    s_real, s_imag = structure_factor(pos, charges, kvec, dipoles)
    s_sq = s_real * s_real + s_imag * s_imag
    kfac = torch.exp(-(sigma * sigma) * k_sq) / k_sq
    return eV_A * _TWOPI / volume * (factors * kfac * s_sq).sum()


def electrostatic_potential(
    pos: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    *,
    sigma_comb: torch.Tensor,
    k_cutoff: float,
    dipoles: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K-space potential and Cartesian gradient at each atom.

    ``σ_comb`` may be a vector of feature widths: the source structure factor
    is shared, the Gaussian form factor is per width.

    Args:
        pos: ``(N, 3)`` Å.
        charges: ``(N,)`` e.
        cell: ``(3, 3)`` Å.
        sigma_comb: Combined GTO widths ``(R,)`` Å.
        k_cutoff: Reciprocal cutoff Å⁻¹.
        dipoles: Optional ``(N, 3)`` e·Å.

    Returns:
        ``phi (N, R)`` eV/e, ``field (N, 3, R)`` eV/(e·Å) as ``+∇φ``.
    """
    cell = _as_cell(cell)
    volume = torch.linalg.det(cell).abs()
    kvec, k_sq, factors = half_sphere_kvectors(cell, k_cutoff)
    n = pos.shape[0]
    n_rad = int(sigma_comb.shape[0])
    if kvec.numel() == 0:
        zeros = charges.new_zeros(n, n_rad)
        return zeros, charges.new_zeros(n, 3, n_rad)
    s_real, s_imag = structure_factor(pos, charges, kvec, dipoles)
    kr = pos @ kvec.T
    cos_kr = torch.cos(kr)
    sin_kr = torch.sin(kr)
    term = s_real.unsqueeze(0) * cos_kr + s_imag.unsqueeze(0) * sin_kr
    grad_term = -s_real.unsqueeze(0) * sin_kr + s_imag.unsqueeze(0) * cos_kr
    sigma = sigma_comb.to(dtype=charges.dtype, device=charges.device).reshape(1, -1)
    kfac = torch.exp(-(sigma * sigma) * k_sq.unsqueeze(-1)) / k_sq.unsqueeze(-1)
    weight = factors.unsqueeze(-1) * kfac
    pre = eV_A * _FOURPI / volume
    phi = pre * (term @ weight)
    field = pre * torch.einsum("im,mr,mc->icr", grad_term, weight, kvec)
    return phi, field
