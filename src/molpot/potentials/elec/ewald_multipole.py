"""Latent Ewald Summation (LES) screened-Coulomb multipole potential.

This module is a thin :class:`molpot.potentials.base.BasePotential`
wrapper around the pure kernel
:class:`molpot.potentials.elec.kernels.MultipoleEwaldKernel`. It adds
exactly four things the kernel deliberately omits:

1. **Per-graph dispatch** — iterates over ``batch`` to call the kernel
   on each graph and reassembles per-atom outputs.
2. **Realspace / reciprocal selection** — chooses between the kernel's
   non-periodic O(N²) realspace path and 3D-periodic reciprocal path
   based on whether ``cell`` is supplied and has non-zero determinant.
3. **External-field injection** — adds ``e_ext`` to the per-atom field
   returned by the kernel before the induced-response step.
4. **Non-self-consistent induced response** — applies
   ``q_induced = -κ·Φ`` / ``u_induced = α·E`` from the LES one-shot
   linear response when ``κ`` / ``α`` are supplied; energies
   ``U_iq = ½ Φ·q_induced`` and ``U_iu = -½ E·u_induced``.

The five pair kernels ``f_{qq}, f_{qu}, f_{uu}, f_{Qu}, f_{QQ}``, the
reciprocal-space scalar kernel ``exp(-σ² k²/2)/k²``, the per-multipole
self-correction constants, and both realspace / reciprocal compute
paths live in
:class:`molpot.potentials.elec.kernels.MultipoleEwaldKernel`. Heads
(e.g. :class:`molpot.heads.BondChargeHead`,
:class:`molpot.heads.DipoleHead`) can call the kernel directly; this
wrapper is for users who want the LES per-graph + induced-response
ergonomics. See the kernel module for the full reference list (Cheng
2025 npj CompMat; Aguado & Madden 2003 JCP).

Three flagged caveats requiring user awareness:

1. **🟡 Flag #1 — multipole-S(k) provenance.** The multipole structure
   factor and per-atom μ²/Q² self-corrections are NOT present in the
   v1 arXiv preprint of King et al. *Nat. Commun.* 16:8763 (2025)
   (arXiv:2412.15455v1); they appear only in the upstream LES code.
   The canonical published derivation is Aguado & Madden 2003 — see
   the kernel module.

2. **🟢 Flag #2 — Q tracelessness.** The quadrupole self-correction
   assumes ``Q`` symmetric and traceless. Upstream LES does not
   enforce this; here we trace-project at the readout layer (in
   :class:`molpot.heads.PolarizabilityHead`'s l=2 path and in
   :class:`molpot.heads.PermMultipoleHead`'s Θ output).

3. **🟢 Flag #3 — k=0 background.** The k=0 reciprocal-space term is
   excluded by the kernel. Mathematically this imposes a uniform
   compensating background charge (Yeh-Berkowitz tin-foil convention);
   charge non-neutrality is permitted but the boundary is documented.

What this is NOT:
    * **Not :class:`molpot.potentials.Polarization`.** That class
      solves a self-consistent CG Thole iteration; LES α-mode here is
      a **one-shot** non-self-consistent linear response. Composing
      both in one ``PotentialComposer`` would double-count induction.
    * **Not Qeq / CELLI.** No global Lagrangian solve with KKT or
      Hirshfeld supervision.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from molpot.potentials.base import BasePotential
from molpot.potentials.elec.kernels.multipole_ewald import MultipoleEwaldKernel

# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class EwaldMultipoleEnergySpec(BaseModel):
    """Configuration snapshot for :class:`EwaldMultipoleEnergy`.

    Frozen Pydantic model — every constructor argument is captured here
    so a trained checkpoint carries an exact, validated description of
    the potential it was built with. Defaults exactly match the LES
    upstream defaults so a freshly-constructed
    ``EwaldMultipoleEnergy()`` reproduces ``les.Les()`` numerically
    on the parity-test oracle path.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    sigma: float = Field(default=1.0, gt=0.0)
    dl: float = Field(default=2.0, gt=0.0)
    prefactor: float = Field(default=90.4756, gt=0.0)
    remove_self_interaction: bool = True
    use_epsilon_r_scaling: bool = False


# ---------------------------------------------------------------------------
# EwaldMultipoleEnergy — thin BasePotential wrapper over MultipoleEwaldKernel
# ---------------------------------------------------------------------------


class EwaldMultipoleEnergy(BasePotential):
    """LES-style σ-screened multipole electrostatic energy potential.

    Thin :class:`BasePotential` wrapper that owns a
    :class:`MultipoleEwaldKernel`, dispatches per graph and per path
    (realspace vs reciprocal), and applies the non-self-consistent
    induced response (κ for induced charges, α for induced dipoles).

    Args:
        sigma: Gaussian charge-smearing length in Å. Default 1.0.
        dl: Reciprocal-space grid resolution in Å (controls k-space
            cutoff ``k_max = 2π/dl``). Default 2.0.
        prefactor: Electrostatic prefactor ``1/(2 ε₀)`` in the project's
            unit system. Default ``90.4756`` (eV·Å·e⁻²).
        remove_self_interaction: If True (default), exclude the
            atom-on-itself Gaussian self-term from both the energy and
            the per-atom potential / field. Required for the
            linear-response path so ``Φ(r_i)`` is the field from
            *other* sources only.
        use_epsilon_r_scaling: If True, apply Clausius-Mossotti-style
            ``ε_r = (Σ_i α_i)/(V·ε₀) + 1`` scaling to the dielectric
            response. Default False (research-feature; off in v0).

    Forward inputs (passed as keyword arguments):
        q: ``(N,)`` per-atom charges; **required**.
        pos: ``(N, 3)`` per-atom positions; **required**.
        cell: ``(3, 3)`` lattice vectors for periodic systems; ``None``
            or zero-determinant ⇒ non-periodic O(N²) path.
        batch: ``(N,)`` int graph membership for batched inputs;
            ``None`` ⇒ single-graph.
        mu: ``(N, 3)`` per-atom dipoles; optional.
        Q: ``(N, 3, 3)`` per-atom quadrupoles; optional. Caller is
            responsible for tracelessness (Flag #2).
        kappa: ``(N,)`` per-atom hardness for the induced-charge
            response; optional. Triggers ``q_induced = -κ·Φ`` and
            ``U_iq = ½ Φ·q_induced``.
        alpha: ``(N,)`` (isotropic) or ``(N, 3, 3)`` (anisotropic)
            per-atom polarizability for the induced-dipole response;
            optional. Triggers ``u_induced = α·E`` and
            ``U_iu = -½ E·u_induced``.
        e_ext: ``(3,)`` external electric field; optional, added to
            the per-atom field before the polarizability response.
        compute_field: Reserved for API compatibility; the field is
            always populated. Default ``False``.

    Forward output:
        Dict with keys ``"pot"`` (per-graph energy), ``"phi"`` ``(N,)``,
        ``"field"`` ``(N, 3)``, ``"q_induced"`` ``(N,)``, ``"u_induced"``
        ``(N, 3)``.
    """

    name = "ewald_multipole"
    type = "long_range"

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        dl: float = 2.0,
        prefactor: float = 90.4756,
        remove_self_interaction: bool = True,
        use_epsilon_r_scaling: bool = False,
    ):
        super().__init__()
        self.config = EwaldMultipoleEnergySpec(
            sigma=sigma,
            dl=dl,
            prefactor=prefactor,
            remove_self_interaction=remove_self_interaction,
            use_epsilon_r_scaling=use_epsilon_r_scaling,
        )
        self.use_epsilon_r_scaling = self.config.use_epsilon_r_scaling
        self.kernel = MultipoleEwaldKernel(
            sigma=self.config.sigma,
            dl=self.config.dl,
            prefactor=self.config.prefactor,
            remove_self_interaction=self.config.remove_self_interaction,
        )

    @classmethod
    def from_spec(cls, spec: EwaldMultipoleEnergySpec) -> "EwaldMultipoleEnergy":
        """Construct from a frozen :class:`EwaldMultipoleEnergySpec` snapshot."""
        return cls(**spec.model_dump())

    def from_dist(self, r_ij: torch.Tensor) -> dict[str, torch.Tensor]:
        """Delegates to :meth:`MultipoleEwaldKernel.from_dist`."""
        return self.kernel.from_dist(r_ij)

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """Delegates to :meth:`MultipoleEwaldKernel.lr_from_k_sq`."""
        return self.kernel.lr_from_k_sq(k_sq)

    def self_contribution(self) -> dict[str, float]:
        """Delegates to :meth:`MultipoleEwaldKernel.self_contribution`."""
        return self.kernel.self_contribution()

    def enumerate_kvec_indices(self, cell: torch.Tensor) -> torch.Tensor:
        """Delegates to :meth:`MultipoleEwaldKernel.enumerate_kvec_indices`."""
        return self.kernel.enumerate_kvec_indices(cell)

    def forward(
        self,
        data: dict[str, Any] | None = None,
        *,
        q: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        Q: torch.Tensor | None = None,
        kappa: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        e_ext: torch.Tensor | None = None,
        compute_field: bool = False,
        kvec_indices: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute total LES electrostatic energy + per-atom Φ, E.

        Dispatches between the realspace (non-periodic) and reciprocal
        (periodic) kernel paths based on the presence of ``cell``;
        inlines the non-self-consistent induced response when ``κ``
        and/or ``α`` are passed.
        """
        if data is not None and q is None:
            q = data.get("q") if isinstance(data, dict) else None
        if data is not None and pos is None:
            pos = data.get("pos") if isinstance(data, dict) else None
        if q is None or pos is None:
            raise ValueError(
                "EwaldMultipoleEnergy.forward requires `q` and `pos` "
                "as keyword arguments (or as `data` keys)."
            )

        n = pos.shape[0]
        if batch is None:
            batch_eff = torch.zeros(n, dtype=torch.long, device=pos.device)
            single_graph = True
        else:
            batch_eff = batch
            single_graph = False

        phi_full = torch.zeros_like(q)
        field_full = torch.zeros(n, 3, dtype=pos.dtype, device=pos.device)
        q_ind_full = torch.zeros_like(q)
        u_ind_full = torch.zeros(n, 3, dtype=pos.dtype, device=pos.device)
        per_graph_pot: list[torch.Tensor] = []

        # Hoist per-cell work out of the per-graph loop: when the cell is
        # shared (`cell.dim() == 2`), the det/use_reciprocal check is a
        # single host sync; when per-graph (`cell.dim() == 3`), compute
        # the periodic mask once and index it.
        if cell is None:
            cell_shared = None
            shared_use_reciprocal = False
            per_graph_periodic: torch.Tensor | None = None
        elif cell.dim() == 2:
            cell_shared = cell
            shared_use_reciprocal = bool(torch.linalg.det(cell).abs() > 1e-6)
            per_graph_periodic = None
        else:
            cell_shared = None
            shared_use_reciprocal = False
            per_graph_periodic = torch.linalg.det(cell).abs() > 1e-6

        unique = torch.unique(batch_eff)
        for raw in unique:
            mask = batch_eff == raw
            pos_i = pos[mask]
            q_i = q[mask]
            mu_i = mu[mask] if mu is not None else None
            Q_i = Q[mask] if Q is not None else None
            kappa_i = kappa[mask] if kappa is not None else None
            alpha_i = alpha[mask] if alpha is not None else None

            if per_graph_periodic is not None:
                i = int(raw.item())
                cell_i = cell[i]
                use_reciprocal = bool(per_graph_periodic[i])
            else:
                cell_i = cell_shared
                use_reciprocal = shared_use_reciprocal

            if use_reciprocal:
                sub = self.kernel.compute_reciprocal(
                    pos=pos_i,
                    q=q_i,
                    cell=cell_i,
                    mu=mu_i,
                    Q=Q_i,
                    kvec_indices=kvec_indices,
                )
            else:
                sub = self.kernel.compute_realspace(pos=pos_i, q=q_i, mu=mu_i, Q=Q_i)

            sub = self._apply_external_and_induced(sub, e_ext, kappa_i, alpha_i)

            per_graph_pot.append(sub["pot"])
            phi_full[mask] = sub["phi"]
            field_full[mask] = sub["field"]
            if "q_induced" in sub:
                q_ind_full[mask] = sub["q_induced"]
            if "u_induced" in sub:
                u_ind_full[mask] = sub["u_induced"]

        if single_graph:
            pot_out = per_graph_pot[0]
        else:
            pot_out = torch.stack(per_graph_pot)

        return {
            "pot": pot_out,
            "phi": phi_full,
            "field": field_full,
            "q_induced": q_ind_full,
            "u_induced": u_ind_full,
        }

    def _apply_external_and_induced(
        self,
        kernel_out: dict[str, torch.Tensor],
        e_ext: torch.Tensor | None,
        kappa: torch.Tensor | None,
        alpha: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Add ``e_ext`` to the per-atom field and apply the LES linear response.

        Kept separate from the pure kernel so that other potentials /
        heads that don't need induced response can call the kernel
        directly with a clean tensor-in / tensor-out interface.
        """
        pot = kernel_out["pot"]
        e_phi = kernel_out["phi"]
        e_field = kernel_out["field"]

        if e_ext is not None:
            e_field = e_field + e_ext.unsqueeze(0)

        out: dict[str, torch.Tensor] = {"phi": e_phi, "field": e_field}
        if kappa is not None:
            q_induced = -kappa * e_phi
            pot = pot + 0.5 * torch.dot(e_phi, q_induced)
            out["q_induced"] = q_induced
        if alpha is not None:
            if alpha.dim() == 1:
                u_induced = alpha.unsqueeze(-1) * e_field
            else:
                u_induced = torch.einsum("icd,id->ic", alpha, e_field)
            pot = pot - 0.5 * torch.einsum("ic,ic->", e_field, u_induced)
            out["u_induced"] = u_induced
        out["pot"] = pot
        return out
