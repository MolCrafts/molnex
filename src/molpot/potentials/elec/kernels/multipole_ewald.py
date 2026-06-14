"""Multipole Ewald kernel — pure tensor math, no head logic.

The kernel implements the LES-style σ-Gaussian-screened multipole
electrostatic interaction as pure tensor functions: the five
real-space pair kernels :math:`f_{qq}, f_{q\\mu}, f_{\\mu\\mu},
f_{Q\\mu}, f_{QQ}` from successive differentiation of
:math:`\\operatorname{erf}(r/(\\sigma\\sqrt{2}))/r`, the
reciprocal-space scalar kernel :math:`\\exp(-\\sigma^2 k^2/2)/k^2`,
and the per-multipole self-correction constants.

Two single-graph compute paths are provided:

* :meth:`compute_realspace` — non-periodic O(N²) all-pairs path.
* :meth:`compute_reciprocal` — 3D-periodic half-k-sphere path with
  optional ``kvec_indices`` for FD-stress smoothness.

The kernel has **no notion** of "head", "potential", "per-graph
batching", or "induced response". Heads (e.g.
:class:`molpot.heads.BondChargeHead`, :class:`molpot.heads.DipoleHead`)
predict ``q_i / \\mu_i / Q_i`` and *call into* this kernel; the
LES-flavoured per-graph + induced-response wrapper lives in
:class:`molpot.potentials.elec.ewald_multipole.EwaldMultipoleEnergy`.

References:
    * Cheng B., *Latent Ewald summation for machine-learning potentials*,
      npj Comput. Mater. **11**, 80 (2025). DOI: 10.1038/s41524-025-01577-7.
    * Aguado A. & Madden P. A., *Ewald summation of electrostatic
      multipole interactions up to the quadrupolar level*,
      J. Chem. Phys. **119**, 7471 (2003). DOI: 10.1063/1.1605941.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["MultipoleEwaldKernel"]


class MultipoleEwaldKernel(nn.Module):
    """Pure-math multipole Ewald kernel.

    Args:
        sigma: Gaussian charge-smearing length in Å. Default 1.0.
        dl: Reciprocal-space grid resolution in Å (sets ``k_max =
            2π/dl``). Default 2.0.
        prefactor: Electrostatic prefactor ``1/(2 ε₀)`` in the project's
            unit system. Default ``90.4756`` (eV·Å·e⁻²).
        remove_self_interaction: When ``True`` (default), the realspace
            path already excludes ``i == j``; the reciprocal path
            subtracts the per-multipole self constants returned by
            :meth:`self_contribution` from both the energy and the
            per-atom potential / field. When ``False``, the realspace
            path adds the self constants back so the two paths give
            comparable totals.
    """

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        dl: float = 2.0,
        prefactor: float = 90.4756,
        remove_self_interaction: bool = True,
    ):
        super().__init__()
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma!r}")
        if dl <= 0.0:
            raise ValueError(f"dl must be > 0, got {dl!r}")
        if prefactor <= 0.0:
            raise ValueError(f"prefactor must be > 0, got {prefactor!r}")

        self.sigma = sigma
        self.dl = dl
        self.prefactor = prefactor
        self.remove_self_interaction = remove_self_interaction

        self._a = 1.0 / (sigma * (2.0**0.5))
        self._sigma_sq_half = sigma * sigma / 2.0
        # Widen k² cutoff by 1e-10 relative: a bare `k² ≤ k_max²` cutoff is
        # a step discontinuity at the boundary; under cell rotation the
        # rounded `nvec @ G @ R.T` k² lookup flips a few boundary
        # k-vectors on/off relative to the unrotated cell, breaking SO(3)
        # equivariance of the reciprocal sum and FD-stress smoothness.
        self._k_sq_max = (2.0 * torch.pi / dl) ** 2 * (1.0 + 1.0e-10)
        # `prefactor = 1/(2ε₀) = 2π · 1/(4πε₀)`; the bare per-pair
        # real-space kernels need `1/(4πε₀) = prefactor / (2π)`. Reciprocal
        # sums use `prefactor / V` directly per LES `ewald.py`.
        self._norm_const = prefactor / (2.0 * torch.pi)
        self._sc = self._compute_self_contribution()

    def from_dist(self, r_ij: torch.Tensor) -> dict[str, torch.Tensor]:
        """Real-space screened-Coulomb multipole kernels at pairwise displacements.

        Computes ``f_qq(r), f_qu(r), f_uu(r), f_Qu(r), f_QQ(r)`` at every
        ``r_ij[i, j] = r[j] - r[i]``, all multiplied by ``norm_const =
        prefactor / (2π)``. Diagonal (``i == j``) entries are zero.

        Args:
            r_ij: ``(N, N, 3)`` pairwise displacement tensor with the
                convention ``r_ij[i, j] = r[j] - r[i]``.

        Returns:
            Dict with keys ``"f_qq"`` ``(N, N)``, ``"f_qu"`` ``(N, N, 3)``,
            ``"f_uu"`` ``(N, N, 3, 3)``, ``"f_Qu"`` ``(N, N, 3, 3, 3)``,
            ``"f_QQ"`` ``(N, N, 3, 3, 3, 3)``.
        """
        n = r_ij.shape[0]
        device = r_ij.device
        dtype = r_ij.dtype
        a = self._a
        sqrt_pi = math.sqrt(math.pi)
        norm_const = self._norm_const

        eye_n = torch.eye(n, dtype=torch.bool, device=device)
        mask_off = ~eye_n  # (N, N) — True off-diagonal

        r_norm = torch.linalg.norm(r_ij, dim=-1)  # (N, N)

        rinv = torch.zeros_like(r_norm)
        rinv[mask_off] = 1.0 / r_norm[mask_off]

        erf_vals = torch.zeros_like(r_norm)
        erf_vals[mask_off] = torch.special.erf(r_norm[mask_off] * a)

        f_qq = erf_vals * rinv * norm_const  # (N, N)

        rinv2 = rinv * rinv
        rinv3 = rinv2 * rinv
        gauss = torch.exp(-((a * r_norm) ** 2)) * mask_off
        rhat = r_ij * rinv.unsqueeze(-1)
        eye3 = torch.eye(3, dtype=dtype, device=device)

        s1 = erf_vals * rinv3 - (2.0 * a / sqrt_pi) * gauss * rinv2
        s2 = (
            3.0 * erf_vals * rinv3
            - (6.0 * a / sqrt_pi) * gauss * rinv2
            - (4.0 * a**3 / sqrt_pi) * gauss
        )

        rr = rhat[..., :, None] * rhat[..., None, :]  # (N, N, 3, 3)
        f_uu = (s2[:, :, None, None] * rr - s1[:, :, None, None] * eye3[None, None]) * norm_const
        f_qu = s1[..., None] * r_ij * norm_const  # (N, N, 3)

        rinv4 = rinv3 * rinv
        rinv5 = rinv4 * rinv
        s3 = (
            15.0 * erf_vals * rinv4
            - (30.0 * a / sqrt_pi) * gauss * rinv3
            - (20.0 * a**3 / sqrt_pi) * gauss * rinv
            - (8.0 * a**5 / sqrt_pi) * gauss * r_norm
        )
        s4 = (
            105.0 * erf_vals * rinv5
            - (210.0 * a / sqrt_pi) * gauss * rinv4
            - (140.0 * a**3 / sqrt_pi) * gauss * rinv2
            - (56.0 * a**5 / sqrt_pi) * gauss
            - (16.0 * a**7 / sqrt_pi) * gauss * r_norm**2
        )

        rrr = torch.einsum("nmi,nmj,nmk->nmijk", rhat, rhat, rhat)
        term_delta_r = (
            torch.einsum("ab,ijc->ijabc", eye3, rhat)
            + torch.einsum("ac,ijb->ijabc", eye3, rhat)
            + torch.einsum("bc,ija->ijabc", eye3, rhat)
        )
        f_Qu = (
            s3[..., None, None, None] * rrr - (s2 * rinv)[..., None, None, None] * term_delta_r
        ) * norm_const

        rrrr = torch.einsum("ija,ijb,ijc,ijd->ijabcd", rhat, rhat, rhat, rhat)
        term_delta_rr = (
            torch.einsum("ab,ijc,ijd->ijabcd", eye3, rhat, rhat)
            + torch.einsum("ac,ijb,ijd->ijabcd", eye3, rhat, rhat)
            + torch.einsum("ad,ijb,ijc->ijabcd", eye3, rhat, rhat)
            + torch.einsum("bc,ija,ijd->ijabcd", eye3, rhat, rhat)
            + torch.einsum("bd,ija,ijc->ijabcd", eye3, rhat, rhat)
            + torch.einsum("cd,ija,ijb->ijabcd", eye3, rhat, rhat)
        )
        term_delta_delta = (
            torch.einsum("ab,cd->abcd", eye3, eye3)
            + torch.einsum("ac,bd->abcd", eye3, eye3)
            + torch.einsum("ad,bc->abcd", eye3, eye3)
        )[None, None]
        f_QQ = (
            s4[..., None, None, None, None] * rrrr
            - (s3 * rinv)[..., None, None, None, None] * term_delta_rr
            + (s2 * rinv2)[..., None, None, None, None] * term_delta_delta
        ) * norm_const

        return {"f_qq": f_qq, "f_qu": f_qu, "f_uu": f_uu, "f_Qu": f_Qu, "f_QQ": f_QQ}

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """Reciprocal-space scalar kernel ``exp(-σ² k²/2) / k²``.

        The k=0 term is not handled here; the caller is expected to
        mask out k=0 before invoking this method (Yeh-Berkowitz uniform
        compensating-background convention).
        """
        return torch.exp(-self._sigma_sq_half * k_sq) / k_sq

    def self_contribution(self) -> dict[str, float]:
        """Per-multipole self-correction constants (cached at construction)."""
        return self._sc

    def _compute_self_contribution(self) -> dict[str, float]:
        twopi32 = (2.0 * math.pi) ** 1.5
        sigma = self.sigma
        prefactor = self.prefactor
        return {
            "energy_q": prefactor / (sigma * twopi32),
            "phi_q": 2.0 * prefactor / (sigma * twopi32),
            "energy_u": prefactor / (3.0 * sigma**3 * twopi32),
            "energy_Q": prefactor / (10.0 * sigma**5 * twopi32),
            "field_u": (4.0 / (3.0 * math.sqrt(math.pi))) * self._a**3 * self._norm_const,
        }

    def _build_kvec_grid(
        self, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enumerate integer triplets, project to k-space, filter by k² cutoff.

        Returns ``(nvec, kvec, k_sq)`` with all three sliced to the keep-set
        ``(k² > 0) & (k² ≤ k_sq_max)``.
        """
        device = cell.device
        dtype = cell.dtype
        twopi = 2.0 * math.pi
        cell_inv = torch.linalg.inv(cell)
        G = twopi * cell_inv.T
        norms = torch.linalg.norm(cell, dim=1)
        Nk = [max(1, int(norms[i].item() / self.dl)) for i in range(3)]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)
        nvec = (
            torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3).to(dtype)
        )
        kvec = nvec @ G
        k_sq = (kvec * kvec).sum(dim=1)
        keep = (k_sq > 0.0) & (k_sq <= self._k_sq_max)
        return nvec[keep], kvec[keep], k_sq[keep]

    def enumerate_kvec_indices(self, cell: torch.Tensor) -> torch.Tensor:
        """Enumerate the integer-triplet ``(M, 3)`` array used by reciprocal.

        Returns the per-cell ``nvec`` already filtered by
        ``keep = (k² > 0) & (k² ≤ k_sq_max)``. Used by FD-stress
        validation to freeze the keep-set on the unperturbed cell.
        """
        nvec, _, _ = self._build_kvec_grid(cell)
        return nvec

    def compute_realspace(
        self,
        pos: torch.Tensor,
        q: torch.Tensor,
        mu: torch.Tensor | None = None,
        Q: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Non-periodic O(N²) all-pairs path.

        Args:
            pos: ``(N, 3)`` atomic positions.
            q: ``(N,)`` per-atom charges.
            mu: optional ``(N, 3)`` per-atom dipoles.
            Q: optional ``(N, 3, 3)`` per-atom quadrupoles (caller
                responsibility: symmetric traceless).

        Returns:
            Dict with keys ``"pot"`` (scalar energy), ``"phi"`` ``(N,)``
            (per-atom potential), ``"field"`` ``(N, 3)`` (per-atom field).
        """
        r_ij = pos.unsqueeze(0) - pos.unsqueeze(1)  # r_ij[i, j] = r[j] - r[i]
        kernels = self.from_dist(r_ij)
        f_qq = kernels["f_qq"]
        f_qu = kernels["f_qu"]
        f_uu = kernels["f_uu"]
        f_Qu = kernels["f_Qu"]
        f_QQ = kernels["f_QQ"]

        e_phi = torch.einsum("i,ij->j", q, f_qq)
        pot = 0.5 * torch.dot(e_phi, q)
        e_field = torch.einsum("i,ijc->jc", q, f_qu)

        if mu is not None:
            e_phi_u = torch.einsum("ic,ijc->j", mu, f_qu)
            e_phi = e_phi + e_phi_u
            pot = pot + torch.dot(e_phi_u, q)
            E_u = torch.einsum("ijcd,ic->jd", f_uu, mu)
            pot = pot - 0.5 * torch.einsum("ic,ic->", mu, E_u)
            e_field = e_field + E_u

        if Q is not None:
            e_phi_Q = 0.5 * torch.einsum("iab,ijab->j", Q, f_uu)
            E_Q = 0.5 * torch.einsum("iab,ijabc->jc", Q, f_Qu)
            e_phi = e_phi + e_phi_Q
            pot = pot + torch.dot(q, e_phi_Q)
            pot = pot + 0.125 * torch.einsum("iab,ijabcd,jcd->", Q, f_QQ, Q)
            e_field = e_field + E_Q
            if mu is not None:
                pot = pot - torch.einsum("ic,ic->", mu, E_Q)

        if not self.remove_self_interaction:
            sc = self._sc
            pot = pot + (q * q).sum() * sc["energy_q"]
            e_phi = e_phi + q * sc["phi_q"]
            if mu is not None:
                pot = pot + (mu * mu).sum() * sc["energy_u"]
                e_field = e_field - sc["field_u"] * mu
            if Q is not None:
                pot = pot + (Q * Q).sum() * sc["energy_Q"]

        return {"pot": pot, "phi": e_phi, "field": e_field}

    def compute_reciprocal(
        self,
        pos: torch.Tensor,
        q: torch.Tensor,
        cell: torch.Tensor,
        mu: torch.Tensor | None = None,
        Q: torch.Tensor | None = None,
        kvec_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """3D-periodic reciprocal half-k-sphere path.

        Args:
            pos: ``(N, 3)`` atomic positions.
            q: ``(N,)`` per-atom charges.
            cell: ``(3, 3)`` lattice vectors.
            mu: optional ``(N, 3)`` per-atom dipoles.
            Q: optional ``(N, 3, 3)`` per-atom quadrupoles.
            kvec_indices: optional ``(M, 3)`` integer triplet array from
                :meth:`enumerate_kvec_indices` for FD-stress smoothness
                (production callers must leave this ``None``).

        Returns:
            Dict with keys ``"pot"`` (scalar energy), ``"phi"`` ``(N,)``,
            ``"field"`` ``(N, 3)``.
        """
        device = pos.device
        dtype = pos.dtype
        twopi = 2.0 * math.pi
        sigma_sq_half = self._sigma_sq_half
        prefactor = self.prefactor

        volume = torch.linalg.det(cell).abs()

        if kvec_indices is None:
            nvec, kvec, k_sq = self._build_kvec_grid(cell)
        else:
            nvec = kvec_indices.to(dtype=dtype, device=device)
            kvec = nvec @ (twopi * torch.linalg.inv(cell).T)
            k_sq = (kvec * kvec).sum(dim=1)

        # Half-k-sphere: keep one of every ±k pair, factor=2 on non-axis.
        non_zero = (nvec != 0).to(torch.long)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze(-1)
        hemisphere = (sign > 0) | (nvec == 0).all(dim=1)
        kvec = kvec[hemisphere]
        k_sq = k_sq[hemisphere]
        factors = torch.where(
            (nvec[hemisphere] == 0).all(dim=1),
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(2.0, dtype=dtype, device=device),
        )

        k_dot_r = pos @ kvec.T  # (N, M)
        cos_kr = torch.cos(k_dot_r)
        sin_kr = torch.sin(k_dot_r)
        S_real = torch.einsum("i,im->m", q, cos_kr)
        S_imag = torch.einsum("i,im->m", q, sin_kr)

        if mu is not None:
            uk = mu @ kvec.T  # (N, M)
            S_real = S_real - torch.einsum("im,im->m", uk, sin_kr)
            S_imag = S_imag + torch.einsum("im,im->m", uk, cos_kr)

        if Q is not None:
            qk2 = torch.einsum("ma,iab,mb->im", kvec, Q, kvec)
            S_real = S_real - 0.5 * torch.einsum("im,im->m", qk2, cos_kr)
            S_imag = S_imag - 0.5 * torch.einsum("im,im->m", qk2, sin_kr)

        S_sq = S_real * S_real + S_imag * S_imag
        kfac = torch.exp(-sigma_sq_half * k_sq) / k_sq
        pot = (factors * kfac * S_sq).sum() / volume * prefactor

        prefactor_arr = factors * 2.0 * kfac / volume * prefactor  # (M,)
        term_real = S_real.unsqueeze(0) * cos_kr + S_imag.unsqueeze(0) * sin_kr  # (N, M)
        e_phi = torch.einsum("m,im->i", prefactor_arr, term_real)

        term_imag = S_real.unsqueeze(0) * sin_kr - S_imag.unsqueeze(0) * cos_kr  # (N, M)
        e_field = torch.einsum("m,im,mc->ic", prefactor_arr, term_imag, kvec)

        if self.remove_self_interaction:
            sc = self._sc
            pot = pot - (q * q).sum() * sc["energy_q"]
            e_phi = e_phi - q * sc["phi_q"]
            if mu is not None:
                pot = pot - (mu * mu).sum() * sc["energy_u"]
                e_field = e_field + sc["field_u"] * mu
            if Q is not None:
                pot = pot - (Q * Q).sum() * sc["energy_Q"]

        return {"pot": pot, "phi": e_phi, "field": e_field}
