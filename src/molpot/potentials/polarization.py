"""Induced-dipole polarization potential — self-consistent CG solve.

This module solves a self-consistent Thole-damped induced-dipole equation

    (α⁻¹ − T) · μ = E_perm

via conjugate gradient, where ``T`` is the (Coulomb-prefactored, Thole-damped)
dipole-dipole interaction tensor and ``E_perm`` is the permanent-charge
electric field at each atom. The final polarization energy is
``U_pol = -½ Σ μ · E_perm`` per molecule.

Units (must match across all of molpot to avoid dimensional inconsistency
when composing with other potentials):

* charges in ``e``, positions in ``Å``,
* polarizabilities in ``e²·Å²·eV⁻¹`` (matches ``ElementAlphaTable``'s
  default conversion factor),
* energy in ``eV``.

The Coulomb prefactor ``k_e = 1/(4πε₀) ≈ 14.3996 eV·Å·e⁻²`` is applied
to both the permanent-charge field ``E_perm`` and the dipole-dipole
tensor ``T`` so the equation is dimensionally homogeneous and the
returned energy is in ``eV``. Override via ``coulomb_prefactor`` to use
a different unit system.

This is **distinct** from
:class:`molpot.potentials.EwaldMultipoleEnergy`'s non-self-consistent
linear-response α-mode. The two implement different physical
approximations:

* :class:`Polarization` (this module) — **self-consistent**: ``μ`` is
  iteratively converged to be in equilibrium with the field it itself
  generates via ``T·μ`` coupling. CG-iterated.

* :class:`EwaldMultipoleEnergy` α-mode — **non-self-consistent**: ``μ``
  is computed once from the *fixed* permanent-multipole field
  (``u_induced = α · E^(0)``), no ``T·μ`` feedback, no iteration.
  This is the LES (Latent Ewald Summation) framework's prescription.

Composing both in a single :class:`molpot.composition.PotentialComposer`
would **double-count induction** — pick exactly one. Choose this module
for classical-polarizable force fields (AMOEBA-style); choose
:class:`EwaldMultipoleEnergy` for LES-trained ML interatomic potentials.

References:
    Thole, B. T., *Chem. Phys.* **59**, 341 (1981) — Thole exponential
    damping function ``λ(u) = 1 − (1 + u + ½u²) e^{−u}``.
    Ren, P. & Ponder, J. W., *J. Phys. Chem. B* **107**, 5933 (2003) —
    AMOEBA polarizable force field formulation.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import torch
import torch.nn as nn

# Module-level marker set to ``True`` by
# :class:`molpot.composition.Sonata`'s ``forward`` (and any future
# LES-style electrostatics composer) after it has evaluated the
# σ-screened multipole-Coulomb / Ewald energy on a batch. A subsequent
# call to :meth:`Polarization.forward` reads this flag and emits a
# ``UserWarning`` mentioning ``double-count`` so a user who hand-builds
# a parallel ``Sonata + Polarization`` graph does not silently
# double-count induction (Polarization's CG Thole solve is
# self-consistent; Sonata's LES α-mode is non-self-consistent linear
# response — composing both is physically wrong).
_LES_ELECTROSTATICS_PRECOMPUTED = False


def _set_les_electrostatics_precomputed(value: bool = True) -> None:
    """Mark whether LES-style electrostatics has been evaluated this session.

    Sonata's forward calls this with ``value=True`` at the end of its
    Ewald evaluation. Tests that need to restore the default can call
    with ``value=False``.
    """
    global _LES_ELECTROSTATICS_PRECOMPUTED
    _LES_ELECTROSTATICS_PRECOMPUTED = bool(value)


class Polarization(nn.Module):
    """Self-consistent induced-dipole polarization energy.

    Computes the polarization energy by:
      1. Computing the permanent electric field from partial charges:
         ``E_perm,i = k_e · Σⱼ qⱼ · r̂ᵢⱼ / rᵢⱼ²``  (Coulomb's law).
      2. Building a sparse dipole-dipole interaction tensor T from
         edge_index, with Thole damping and the same Coulomb prefactor:
         ``Tᵢⱼ = k_e · λ(uᵢⱼ) · (3 r̂r̂ − I) / rᵢⱼ³``.
      3. Solving ``(α⁻¹ − T) · μ = E_perm`` per molecule via CG.
      4. ``U_pol = -½ Σᵢ μᵢ · E_perm,i`` per molecule.

    See also:
        :class:`molpot.composition.Sonata` — the static permanent-
        electrostatics composer that wraps
        :class:`molpot.potentials.EwaldMultipoleEnergy` for the LES
        (Latent Ewald Summation) framework. Sonata's forthcoming
        ``LesPolarizable`` sibling adds inline non-self-consistent
        induced response via the ``α``-mode of
        :class:`EwaldMultipoleEnergy`. Pick exactly one model line:
        this :class:`Polarization` (classical-polarizable AMOEBA-style)
        OR the LES α-mode (ML interatomic potential). Composing both
        on the same atoms double-counts induction; calling
        :meth:`Polarization.forward` after a :class:`Sonata` forward in
        the same Python process emits a ``UserWarning`` matching
        ``double-count`` so the hazard surfaces at runtime.

    Args:
        damping_factor: Thole-style damping factor for the dipole-dipole
            tensor. Default ``0.39`` (AMOEBA literature value).
        coulomb_prefactor: Electrostatic prefactor ``k_e = 1/(4πε₀)``
            in the project's unit system. Default ``14.3996`` (eV·Å·e⁻²),
            matching ``EwaldMultipoleEnergy.norm_const``. Set to ``1.0``
            for atomic units.
    """

    def __init__(
        self,
        *,
        damping_factor: float = 0.39,
        coulomb_prefactor: float = 14.3996,
    ):
        super().__init__()
        self.damping_factor = damping_factor
        self.coulomb_prefactor = coulomb_prefactor

    def forward(
        self,
        *,
        pos: torch.Tensor,
        charge: torch.Tensor,
        alpha: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Compute induced-dipole polarization energy.

        Args:
            pos: Atom positions ``(N, 3)``.
            charge: Partial charges ``(N,)``.
            alpha: Isotropic polarizabilities ``(N,)``.
            batch: Graph index per atom ``(N,)``.
            edge_index: Neighbor pairs ``(E, 2)``.
            num_graphs: Number of graphs.

        Returns:
            Per-graph polarization energy ``(B,)`` or scalar.
        """
        if _LES_ELECTROSTATICS_PRECOMPUTED:
            warnings.warn(
                "Polarization.forward called after an LES-style "
                "electrostatics composer (e.g. molpot.composition.Sonata) "
                "has already evaluated electrostatics in this session. "
                "Combining this self-consistent CG Thole induced-dipole "
                "solve with the LES α-mode (non-self-consistent linear "
                "response) double-counts induction. Pick exactly one "
                "model line: Polarization (classical-polarizable "
                "AMOEBA-style) or Sonata's future LesPolarizable line "
                "(ML interatomic potential).",
                UserWarning,
                stacklevel=2,
            )
            # One-shot consumption: reset the flag so subsequent
            # ``Polarization.forward`` calls in unrelated test
            # contexts (or downstream pipelines that no longer
            # interleave Sonata) do not re-warn. A subsequent
            # ``Sonata.forward`` will set the flag again.
            _set_les_electrostatics_precomputed(False)

        N = pos.shape[0]
        device = pos.device
        dtype = pos.dtype

        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1

        # 1. Permanent electric field at each atom from charges (Coulomb's law).
        # E(r_src) = k_e · Σⱼ qⱼ · (r_src − r_j) / |r_src − r_j|³
        #          = k_e · Σⱼ qⱼ · r̂_{j→src} / r²
        # Implementation note: r_vec / r_norm² has magnitude 1/r (not 1/r²);
        # what we want is r_vec / r_norm³ = r̂ / r². Sign of r̂_{src→dst} vs
        # r̂_{dst→src} is irrelevant for the energy (μ flips with E so U_pol
        # = −½ μ·E_perm is sign-invariant); the chosen convention here
        # matches the edge_index direction.
        src, dst = edge_index[:, 0], edge_index[:, 1]
        r_vec = pos[dst] - pos[src]  # (E, 3)
        r_norm = r_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (E, 1)
        r_hat = r_vec / r_norm  # (E, 3) unit
        field_contrib = self.coulomb_prefactor * charge[dst].unsqueeze(-1) * r_hat / r_norm.pow(2)
        E_perm = torch.zeros(N, 3, dtype=dtype, device=device)
        E_perm.index_add_(0, src, field_contrib)

        # 2. Build dipole-dipole interaction tensor T (sparse, 3x3 blocks)
        # T_ij = k_e · λ(u_ij) · (3 r̂ r̂ᵀ − I) / r³  (Thole-damped).
        r3 = r_norm.squeeze(-1).pow(3)  # (E,)

        # Thole damping: λ(u) = 1 − (1 + u + ½u²) · exp(−u)
        # where u = damping_factor · r / (α_i α_j)^(1/6).
        alpha_src = alpha[src].clamp(min=1e-12)
        alpha_dst = alpha[dst].clamp(min=1e-12)
        alpha_prod_sixth = (alpha_src * alpha_dst).pow(1.0 / 6.0)
        u = self.damping_factor * r_norm.squeeze(-1) / alpha_prod_sixth.clamp(min=1e-12)
        damping = 1.0 - (1.0 + u + 0.5 * u.pow(2)) * torch.exp(-u)

        # T_ij @ v = k_e · damping_ij / r³ · (3 (r̂_ij · v) r̂_ij − v).
        # Built implicitly (matvec only, no full N×N×3×3 matrix).
        damped_inv_r3 = self.coulomb_prefactor * damping / r3.clamp(min=1e-12)  # (E,)

        def apply_T(v: torch.Tensor) -> torch.Tensor:
            """Apply dipole-dipole operator T @ v, where v is (N, 3)."""
            v_dst = v[dst]  # (E, 3)
            dot = (r_hat * v_dst).sum(dim=-1, keepdim=True)  # (E, 1)
            t_contrib = damped_inv_r3.unsqueeze(-1) * (3.0 * dot * r_hat - v_dst)  # (E, 3)
            result = torch.zeros_like(v)
            result.index_add_(0, src, t_contrib)
            return result

        def apply_A(v: torch.Tensor) -> torch.Tensor:
            """Apply (alpha_inv - T) @ v."""
            alpha_inv = 1.0 / alpha.clamp(min=1e-12)
            return alpha_inv.unsqueeze(-1) * v - apply_T(v)

        # 3. Solve (alpha_inv - T) @ mu = E_perm using conjugate gradient
        mu = self._cg_solve(apply_A, E_perm, max_iter=50, tol=1e-6)

        # 4. U_pol = -0.5 * sum(mu . E_perm) per molecule
        per_atom_energy = -0.5 * (mu * E_perm).sum(dim=-1)  # (N,)

        energy = torch.zeros(num_graphs, dtype=dtype, device=device)
        energy.index_add_(0, batch, per_atom_energy)
        return energy

    @staticmethod
    def _cg_solve(
        matvec: Callable,
        b: torch.Tensor,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Conjugate gradient solver for A @ x = b.

        Args:
            matvec: Function computing A @ x.
            b: Right-hand side ``(N, 3)``.
            max_iter: Maximum iterations.
            tol: Convergence tolerance on relative residual norm.

        Returns:
            Solution x ``(N, 3)``.
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = (r * r).sum()

        b_norm = b.norm()
        if b_norm < 1e-12:
            return x

        for _ in range(max_iter):
            Ap = matvec(p)
            pAp = (p * Ap).sum()
            if pAp.abs() < 1e-12:
                break
            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = (r * r).sum()
            if rs_new.sqrt() / b_norm < tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x
