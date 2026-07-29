"""Sonata composer — static permanent-electrostatics model line.

Sonata is a sibling (not subclass) of :class:`PotentialComposer`,
specialised for the multipole-Ewald path. It owns three required
sub-modules — an Allegro-style encoder with ``expose_tensor_track=True``,
a :class:`PermMultipoleHead` predicting ``(q, μ, Θ)``, and an
:class:`EwaldMultipoleEnergy` evaluator — plus an optional short-range
energy head. Forward returns a flat dict ``{energy, energy_short,
energy_es, atomic_*, molecular_dipole, phi, field, [forces], [stress]}``
addressable by string keys (no nesting).

What this is NOT
    * **Not a composer for induced-response models.** The forthcoming
      ``LesPolarizable`` composer line wraps :class:`EwaldMultipoleEnergy`
      with ``kappa_head`` / ``alpha_head`` to add inline non-self-consistent
      linear response. Mixing those with the self-consistent CG Thole solve
      in :class:`molpot.potentials.Polarization` would double-count
      induction. Sonata refuses such kwargs at construction time.
    * **Not invoking** :class:`PotentialComposer` — the latter's
      ``mixing_fn(atom_params, edge_index)`` per-pair contract does not
      match the global multipole-Ewald signature ``(q, pos, cell, batch,
      mu, Q)``.
    * **Not for QEq / CELLI.** Charges come from the direct l=0 readout
      of :class:`PermMultipoleHead`, not from a charge-equilibration
      KKT solve.

References:
    * Cheng B., *Latent Ewald summation for machine-learning potentials*,
      npj Comput. Mater. **11**, 80 (2025).
      https://doi.org/10.1038/s41524-025-01577-7
    * King D. S. et al., *Latent equivariant ML force fields with
      long-range electrostatics*, Nat. Commun. **16**, 8763 (2025).
      https://doi.org/10.1038/s41467-025-63852-x
    * Aguado A. & Madden P. A., *Ewald summation of electrostatic
      multipole interactions up to the quadrupolar level*,
      J. Chem. Phys. **119**, 7471 (2003).
      https://doi.org/10.1063/1.1605941
    * Stone A. J., *The Theory of Intermolecular Forces*, 2nd ed.
      (Oxford, 2013), §3.
    * Upstream LES code: https://github.com/ChengUCB/les
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict

from molpot.derivation import ForceDerivation
from molpot.heads.multipole import PermMultipoleHead, PermMultipoleHeadSpec
from molpot.potentials.elec.ewald_multipole import (
    EwaldMultipoleEnergy,
    EwaldMultipoleEnergySpec,
)
from molpot.potentials.polarization import (
    Polarization,
    _set_les_electrostatics_precomputed,
)

# ---------------------------------------------------------------------------
# Refusal messages — used by the construction-time scope checks.
# ---------------------------------------------------------------------------

_INDUCED_REFUSE_MSG = (
    "Sonata is the *static permanent-electrostatics* composer; "
    "induced-response inputs ({name}={value!r}) belong to the future "
    "`LesPolarizable` composer line, which wraps `EwaldMultipoleEnergy` "
    "with κ / α heads for inline non-self-consistent linear response. "
    "Use that composer (when implemented) for induced-response training."
)

_POLARIZATION_REFUSE_MSG = (
    "Sonata refuses `Polarization` in `short_range_head`: it would "
    "double-count induction against any future inline LES α-mode. "
    "Polarization belongs to the `LesPolarizable` composer line; use "
    "an `EdgeEnergyHead` (or any head that writes a dict containing "
    "`'energy_short'`) here instead."
)


def _theta_to_cartesian_quadrupole(theta: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, 5)`` ℓ=2 real-spherical components to ``(N, 3, 3)``.

    The result is a symmetric, traceless 3×3 tensor that is rotation-
    equivariant with the input under cuequivariance's ``D⁽²⁾`` Wigner
    matrix: ``M(D⁽²⁾(R) · θ) = R · M(θ) · Rᵀ``.

    The cuequivariance ``2e`` real-spherical-harmonic convention
    (extracted from ``cuequivariance.group_theory.descriptors.
    spherical_harmonics_.sympy_spherical_harmonics``) maps the five
    components in ``ir_mul`` layout to these polynomial bases:

    * index 0 (m=-2): ``√15 · xz``
    * index 1 (m=-1): ``√15 · xy``
    * index 2 (m=0):  ``(√5/2) · (-x² + 2y² - z²)``
    * index 3 (m=+1): ``√15 · yz``
    * index 4 (m=+2): ``(√15/2) · (z² - x²)``

    Note that index 0 is ``xz`` (not ``xy``) and the m=0 diagonal is
    ``diag(-1, 2, -1)`` (not ``diag(-1, -1, 2)``) — both depart from
    the older "physics convention" that is wrong for cuequivariance.

    The basis tensors below are orthonormal in the Frobenius inner
    product (any consistent global scaling preserves the equivariance).

    Args:
        theta: ``(N, 5)`` spherical components in cuequivariance's
            ``2e`` ir_mul layout (``mul=1``, last axis = component).

    Returns:
        ``(N, 3, 3)`` symmetric traceless Cartesian quadrupole tensor.
    """
    s2 = 0.5**0.5
    s6 = 6.0**-0.5
    n = theta.shape[0]
    q = torch.zeros(n, 3, 3, dtype=theta.dtype, device=theta.device)
    # index 0 (m=-2 ↔ xz): xz off-diagonal
    q[:, 0, 2] = q[:, 2, 0] = theta[:, 0] * s2
    # index 1 (m=-1 ↔ xy): xy off-diagonal
    q[:, 0, 1] = q[:, 1, 0] = theta[:, 1] * s2
    # index 2 (m=0 ↔ -x²+2y²-z²): diag = (-1, 2, -1) / √6
    q[:, 0, 0] = -theta[:, 2] * s6
    q[:, 1, 1] = 2.0 * theta[:, 2] * s6
    q[:, 2, 2] = -theta[:, 2] * s6
    # index 3 (m=+1 ↔ yz): yz off-diagonal
    q[:, 1, 2] = q[:, 2, 1] = theta[:, 3] * s2
    # index 4 (m=+2 ↔ z²-x²): diag = (-1, 0, 1) / √2  (added on top of m=0)
    q[:, 0, 0] = q[:, 0, 0] - theta[:, 4] * s2
    q[:, 2, 2] = q[:, 2, 2] + theta[:, 4] * s2
    return q


# ---------------------------------------------------------------------------
# SonataSpec — frozen Pydantic snapshot.
# ---------------------------------------------------------------------------


class SonataSpec(BaseModel):
    """Frozen Pydantic snapshot of a :class:`Sonata` composer.

    Mirrors the :class:`PermMultipoleHeadSpec` /
    :class:`EwaldMultipoleEnergySpec` pattern: every constructor argument
    is captured here so a trained checkpoint carries an exact, validated
    description of the model line it was built with. ``head`` and
    ``ewald`` are nested sub-specs; ``short_range`` is an opaque dict
    (the head class is the user's choice and we deliberately do not
    take a head-spec dependency from this layer).

    Attributes:
        head: The :class:`PermMultipoleHeadSpec` for the multipole readout.
        ewald: The :class:`EwaldMultipoleEnergySpec` for the σ-screened
            multipole-Coulomb evaluator.
        short_range: ``None`` when no short-range head is wired, else an
            opaque metadata dict (``{"kind": "single", "type": ...}``
            for a single head, ``{"kind": "list", "types": [...]}``
            for an :class:`nn.ModuleList`).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    head: PermMultipoleHeadSpec
    ewald: EwaldMultipoleEnergySpec
    short_range: dict | None = None


# ---------------------------------------------------------------------------
# Sonata
# ---------------------------------------------------------------------------


class Sonata(nn.Module):
    """Static permanent-electrostatics composer.

    Args:
        encoder: Allegro-style encoder. Must be constructed with
            ``expose_tensor_track=True``; ``l_max ≥ 2`` is required when
            the head predicts dipoles or quadrupoles. Sonata reads
            ``encoder.tensor_track_irreps`` to wire the equivariant
            moment readouts.
        perm_multipole_head: :class:`PermMultipoleHead` predicting
            ``(q, μ, Θ)`` and writing them under ``("atoms", ...)``.
        ewald: :class:`EwaldMultipoleEnergy` evaluator. Sonata calls it
            directly (not via :class:`PotentialComposer`).
        short_range_head: Optional ``nn.Module`` (or
            ``list[nn.Module]``) that consumes ``batch`` and returns a
            dict containing ``"energy_short"`` ``(B,)``. Refused if it
            is, or contains, a :class:`Polarization` — that would
            double-count induction against any future inline LES α-mode.

    Refused construction kwargs (raise :class:`ValueError`):
        * ``kappa_head=`` — induced-charge response head (future
          ``LesPolarizable``).
        * ``alpha_head=`` — induced-dipole response head (future
          ``LesPolarizable``).
        * Any ``induced_*`` kwarg.
        * ``short_range_head`` containing a :class:`Polarization`.

    Forward output (flat dict):
        * ``energy``: ``(B,)`` total per-graph energy.
        * ``energy_short``: ``(B,)`` short-range energy contribution
          (zeros when ``short_range_head`` is ``None``).
        * ``energy_es``: ``(B,)`` electrostatic (Ewald) energy.
        * ``atomic_charges``: ``(N,)`` per-atom charges.
        * ``atomic_dipoles``: ``(N, 3)`` per-atom dipoles (only when
          the head predicts them).
        * ``atomic_quadrupoles``: ``(N, 5)`` per-atom quadrupoles in
          the traceless-symmetric basis (only when the head predicts
          them).
        * ``molecular_dipole``: ``(B, 3)`` molecular dipole moment.
        * ``phi``: ``(N,)`` per-atom electrostatic potential.
        * ``field``: ``(N, 3)`` per-atom electric field.
        * ``charge_sum_pre_proj`` / ``charge_sum_post_proj``: per-graph
          charge sums before and after the head's total-charge
          projection (only when the head has projection on).
        * ``forces``: ``(N, 3)`` only when ``compute_forces=True``.
        * ``stress``: ``(B, 3, 3)`` symmetric, only when
          ``compute_stress=True``.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        perm_multipole_head: PermMultipoleHead,
        ewald: EwaldMultipoleEnergy,
        short_range_head: nn.Module | list[nn.Module] | None = None,
        **kwargs: Any,
    ) -> None:
        # Scope-refusal checks fire BEFORE super().__init__() so a failed
        # construction does not leak partial state via registered modules.
        if "kappa_head" in kwargs:
            raise ValueError(
                _INDUCED_REFUSE_MSG.format(name="kappa_head", value=kwargs["kappa_head"])
            )
        if "alpha_head" in kwargs:
            raise ValueError(
                _INDUCED_REFUSE_MSG.format(name="alpha_head", value=kwargs["alpha_head"])
            )
        for name in kwargs:
            if name.startswith("induced_"):
                raise ValueError(_INDUCED_REFUSE_MSG.format(name=name, value=kwargs[name]))
        if kwargs:
            raise TypeError(f"Sonata.__init__ got unexpected kwargs: {sorted(kwargs)}")

        if isinstance(short_range_head, Polarization):
            raise ValueError(_POLARIZATION_REFUSE_MSG)
        if isinstance(short_range_head, (list, nn.ModuleList)) and any(
            isinstance(h, Polarization) for h in short_range_head
        ):
            raise ValueError(_POLARIZATION_REFUSE_MSG)

        super().__init__()
        self.encoder = encoder
        self.perm_multipole_head = perm_multipole_head
        self.ewald = ewald
        # Pure-torch multipole-Ewald graph → functorch force path.
        # Stress keeps a custom symmetric-ε post-process after torch.func.grad
        # (not a drop-in for StressDerivation's isotropic V scaling).
        self.force_derivation = ForceDerivation(method="functorch")

        # Wrap a list of heads in nn.ModuleList; otherwise store the single
        # head (or None) directly. Storing as None on the attribute keeps
        # forward branching simple.
        self.short_range_head: nn.Module | nn.ModuleList | None
        if short_range_head is None:
            self.short_range_head = None
        elif isinstance(short_range_head, nn.ModuleList):
            self.short_range_head = short_range_head
        elif isinstance(short_range_head, list):
            self.short_range_head = nn.ModuleList(short_range_head)
        else:
            self.short_range_head = short_range_head

        # Capture short_range head metadata opaquely (Sonata does not own
        # a head-spec dependency from this layer).
        short_range_meta: dict | None = None
        if isinstance(self.short_range_head, nn.ModuleList):
            short_range_meta = {
                "kind": "list",
                "types": [type(h).__name__ for h in self.short_range_head],
            }
        elif isinstance(self.short_range_head, nn.Module):
            short_range_meta = {
                "kind": "single",
                "type": type(self.short_range_head).__name__,
            }

        self.config = SonataSpec(
            head=perm_multipole_head.config,
            ewald=ewald.config,
            short_range=short_range_meta,
        )

    @classmethod
    def from_spec(cls, spec: SonataSpec, encoder: nn.Module) -> "Sonata":
        """Construct from a frozen :class:`SonataSpec` and an encoder.

        ``spec.short_range`` is opaque metadata only — the user must
        supply the actual short-range head externally if a round-trip
        wiring with the same head is required. The encoder is wiring
        (not a hyperparameter — it depends on the dataset and training
        script) and is therefore a positional argument, mirroring the
        :meth:`PermMultipoleHead.from_spec` pattern.

        Args:
            spec: The frozen configuration snapshot.
            encoder: Allegro-style encoder providing
                ``tensor_track_irreps``, ``output_dim``, ``l_max``,
                and ``expose_tensor_track``.

        Returns:
            A new :class:`Sonata` whose architecture matches ``spec``.
        """
        head = PermMultipoleHead.from_spec(spec.head, tensor_irreps=encoder.tensor_track_irreps)
        ewald = EwaldMultipoleEnergy.from_spec(spec.ewald)
        return cls(
            encoder=encoder,
            perm_multipole_head=head,
            ewald=ewald,
            short_range_head=None,
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: TensorDict,
        *,
        compute_forces: bool = False,
        compute_stress: bool = False,
        kvec_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run encoder → multipole head → Ewald → optional short-range.

        Args:
            batch: Post-collate :class:`TensorDict` carrying ``atoms``,
                ``edges``, and ``graphs`` sub-dicts. Periodic systems
                must include ``("graphs", "cell")``.
            compute_forces: Derive forces ``F = -∂U/∂pos`` via
                :class:`~molpot.derivation.ForceDerivation`
                (``method="functorch"``).
            compute_stress: Derive stress ``σ = (1/V) ∂U/∂ε`` via
                ``torch.func.grad`` through a differentiable strain
                perturbation. Requires ``("graphs", "cell")``.
            kvec_indices: optional ``(M, 3)`` precomputed integer
                triplet array forwarded to
                :meth:`EwaldMultipoleEnergy._compute_reciprocal` to
                freeze the reciprocal-space integer grid for
                finite-difference cell/strain validation. **Production
                callers must leave this ``None``** — the Ewald cutoff
                ``k² ≤ k_sq_max`` is a step discontinuity in cell
                strain and freezing the grid is *only* appropriate
                near a fixed reference cell where the keep-set would
                not change anyway. See
                :meth:`EwaldMultipoleEnergy.enumerate_kvec_indices`
                for the canonical FD-stress technique.

        Returns:
            Flat dict with the keys documented in the class docstring.
        """
        atom_batch = batch["atoms", "batch"]
        edge_index = batch["edges", "edge_index"]
        src, dst = edge_index[:, 0], edge_index[:, 1]
        head = self.perm_multipole_head

        pos0 = batch["atoms", "pos"].detach()
        cell0 = batch.get(("graphs", "cell"))
        if compute_stress and cell0 is None:
            raise ValueError("compute_stress=True requires `('graphs', 'cell')` in the batch.")

        def _run_pipeline(
            b: TensorDict, pos: torch.Tensor, cell: torch.Tensor | None
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """Encoder → multipole head → Ewald → short-range on a prepared batch.

            ``b`` must already carry ``pos`` and (for the grad paths) the edge
            geometry recomputed from ``pos``; ``pos`` / ``cell`` are passed
            explicitly to Ewald so the gradient flows through them.
            """
            # -- encoder writes edge_features (and edge_tensor_features) --
            self.encoder(b)
            # -- head writes per-atom moments + molecular dipole --
            head_out = head(b)

            # -- Ewald: read moments from the batch --
            q = b["atoms", head.out_charge_key]
            mu = b.get(("atoms", head.out_dipole_key)) if head.dipole else None
            Theta = b.get(("atoms", head.out_quadrupole_key)) if head.quadrupole else None

            # Theta lives in the (N, 5) real-spherical 2e basis (cuequivariance
            # output). EwaldMultipoleEnergy.forward expects Q in (N, 3, 3)
            # symmetric traceless Cartesian form; convert here at the composer
            # boundary so neither the head nor the potential needs to know
            # about the basis swap.
            Q_cart = _theta_to_cartesian_quadrupole(Theta) if Theta is not None else None

            ewald_out = self.ewald(
                q=q,
                pos=pos,
                cell=cell,
                batch=atom_batch,
                mu=mu,
                Q=Q_cart,
                kvec_indices=kvec_indices,
            )
            energy_es = ewald_out["pot"]
            if energy_es.dim() == 0:
                energy_es = energy_es.unsqueeze(0)

            n_graphs = int(energy_es.shape[0])

            # -- short-range head(s) --
            energy_short = torch.zeros(n_graphs, dtype=energy_es.dtype, device=energy_es.device)
            if self.short_range_head is not None:
                heads_iter = (
                    self.short_range_head
                    if isinstance(self.short_range_head, nn.ModuleList)
                    else [self.short_range_head]
                )
                for sr_head in heads_iter:
                    sr_out = sr_head(b)
                    if not isinstance(sr_out, dict):
                        raise TypeError(
                            f"Short-range head {type(sr_head).__name__} must return "
                            f"a dict; got {type(sr_out).__name__}."
                        )
                    if "energy_short" not in sr_out:
                        raise KeyError(
                            f"Short-range head {type(sr_head).__name__} must return "
                            f"a dict with key 'energy_short'; got keys {list(sr_out)}."
                        )
                    energy_short = energy_short + sr_out["energy_short"]

            energy = energy_short + energy_es

            out: dict[str, torch.Tensor] = {
                "energy": energy,
                "energy_short": energy_short,
                "energy_es": energy_es,
                "atomic_charges": q,
                "phi": ewald_out["phi"],
                "field": ewald_out["field"],
            }
            if mu is not None:
                out["atomic_dipoles"] = mu
            if Theta is not None:
                out["atomic_quadrupoles"] = Theta
            if "molecular_dipole" in head_out:
                out["molecular_dipole"] = head_out["molecular_dipole"]
            if "charge_sum_pre_proj" in head_out:
                out["charge_sum_pre_proj"] = head_out["charge_sum_pre_proj"]
            if "charge_sum_post_proj" in head_out:
                out["charge_sum_post_proj"] = head_out["charge_sum_post_proj"]
            return energy, out

        def _strained(
            pos_orig: torch.Tensor, strain: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Apply a symmetric strain to positions and cell (ε = 0 is identity)."""
            sym_eps = 0.5 * (strain + strain.transpose(-1, -2))
            eye_b = torch.eye(3, dtype=strain.dtype, device=strain.device).expand_as(strain)
            i_plus = eye_b + sym_eps
            pos = torch.einsum("aij,aj->ai", i_plus[atom_batch], pos_orig)
            cell = torch.einsum("bij,blj->bli", i_plus, cell0)
            return pos, cell

        def _prepare(b: TensorDict, pos: torch.Tensor, cell: torch.Tensor | None) -> None:
            """Write ``pos`` + recomputed edge geometry (and cell) into ``b`` in place.

            The encoder reads ``edge_diff`` / ``edge_dist`` from the batch rather
            than recomputing from ``pos``, so they must be refreshed for the
            gradient to flow pos → geometry → energy.
            """
            edge_diff = pos[dst] - pos[src]
            b[("atoms", "pos")] = pos
            b[("edges", "edge_diff")] = edge_diff
            b[("edges", "edge_dist")] = edge_diff.norm(dim=-1)
            if cell is not None:
                b[("graphs", "cell")] = cell

        # -- eager pass: materialises lazy params and produces the outputs, using
        #    the batch's cached edge geometry (the original non-grad path). --
        _energy, out = _run_pipeline(batch.clone(), pos0, cell0)

        # -- forces via ForceDerivation (functorch): F = -∂E/∂pos. At ε = 0
        #    the strain is the identity, so differentiating w.r.t. positions
        #    on the original geometry suffices whether or not stress is also
        #    requested. --
        if compute_forces:

            def energy_of_pos(p: torch.Tensor) -> torch.Tensor:
                b = batch.clone()
                _prepare(b, p, cell0)
                return _run_pipeline(b, p, cell0)[0].sum()

            out["forces"] = self.force_derivation(energy_of_pos, pos0)

        # -- stress via functorch: σ = (1/V) ∂E/∂ε through a differentiable
        #    strain perturbation of positions + cell. --
        if compute_stress:
            assert cell0 is not None

            def energy_of_strain(strain: torch.Tensor) -> torch.Tensor:
                pos, cell = _strained(pos0, strain)
                b = batch.clone()
                _prepare(b, pos, cell)
                return _run_pipeline(b, pos, cell)[0].sum()

            strain0 = torch.zeros_like(cell0)
            stress_eps = torch.func.grad(energy_of_strain)(strain0)
            cell_orig = cell0
            # ``stress_eps`` is ``∂E/∂strain[a, b] = ∂E/∂sym_eps[a, b]``
            # (chain rule through ``sym_eps = (strain + strainᵀ)/2``).
            # The standard physics convention treats ε as a *symmetric*
            # tensor with ``ε_αβ = ε_βα`` as a single variable, so
            # ``σ_αβ = (1/V) ∂E/∂ε_αβ`` counts the (α, β) and (β, α)
            # components together for off-diagonal: ``σ_αβ = (1/V) ·
            # 2 · ∂E/∂sym_eps[a, b]`` for ``α ≠ β`` and unchanged for
            # ``α = β``. Since ``stress_eps`` is symmetric (it inherits
            # the ``sym_eps`` symmetry through the chain rule),
            # ``stress_eps + stress_epsᵀ - diag(stress_eps)`` realizes
            # this convention exactly: doubles off-diagonal, leaves
            # diagonal unchanged. This matches the FD strain-perturbation
            # convention ``ε[a, b] = ε[b, a] = dh``.
            stress_eps_sym = (
                stress_eps
                + stress_eps.transpose(-1, -2)
                - torch.diag_embed(torch.diagonal(stress_eps, dim1=-2, dim2=-1))
            )
            volume = torch.linalg.det(cell_orig).abs()
            out["stress"] = stress_eps_sym / volume.unsqueeze(-1).unsqueeze(-1)

        # Surface the double-counting hazard for any subsequent
        # :class:`molpot.potentials.Polarization` call in this session
        # (the boundary contract documented in the class docstring).
        _set_les_electrostatics_precomputed(True)

        return out


# ---------------------------------------------------------------------------
# build_sonata factory
# ---------------------------------------------------------------------------


def build_sonata(
    encoder: nn.Module,
    *,
    sigma: float = 1.0,
    dl: float = 2.0,
    prefactor: float = 90.4756,
    charge: bool = True,
    dipole: bool = True,
    quadrupole: bool = True,
    constrain_total_charge: bool = True,
    short_range_head: nn.Module | list[nn.Module] | None = None,
    total_charge_key: str = "total_charge",
    hidden_dim: int = 128,
    avg_num_neighbors: float | None = None,
) -> Sonata:
    """Build a wired :class:`Sonata` from an encoder and loose hyperparameters.

    Args:
        encoder: Allegro-style encoder. Must satisfy
            ``encoder.expose_tensor_track is True`` and (when ``dipole``
            or ``quadrupole`` is on) ``encoder.l_max >= 2``.
        sigma: σ-Gaussian charge-smearing length in Å. Default ``1.0``.
        dl: Reciprocal-space grid resolution in Å. Default ``2.0``.
        prefactor: Electrostatic prefactor ``1/(2 ε₀)``. Default
            ``90.4756`` (eV·Å·e⁻²).
        charge: Predict atomic charges. Default ``True``.
        dipole: Predict atomic dipoles. Default ``True``. Requires
            ``encoder.l_max >= 2``.
        quadrupole: Predict atomic quadrupoles. Default ``True``.
            Requires ``encoder.l_max >= 2``.
        constrain_total_charge: Project per-graph charge sums onto
            ``total_charge_key``. Default ``True``.
        short_range_head: Optional ``nn.Module`` (or ``list``) writing
            ``"energy_short"`` ``(B,)``.
        total_charge_key: Per-graph total-charge key under ``graphs``.
            Required when ``constrain_total_charge=True``.
        hidden_dim: Hidden width of the multipole head's scalar MLPs.
        avg_num_neighbors: Dataset-wide ⟨|N(i)|⟩ for the edge→atom
            pool normalisation in the multipole head.

    Returns:
        A fully wired :class:`Sonata`.

    Raises:
        ValueError: If ``encoder.expose_tensor_track`` is not ``True``,
            or if ``dipole`` / ``quadrupole`` is requested but
            ``encoder.l_max < 2``.
    """
    if not getattr(encoder, "expose_tensor_track", False):
        raise ValueError(
            "Sonata requires an encoder built with `expose_tensor_track=True` "
            "to expose the equivariant tensor-track features the multipole "
            "head consumes. Reconstruct your encoder with this flag set."
        )
    if (dipole or quadrupole) and getattr(encoder, "l_max", 0) < 2:
        raise ValueError(
            "Sonata requires `encoder.l_max >= 2` when `dipole=True` or "
            f"`quadrupole=True`; got `l_max={getattr(encoder, 'l_max', None)}`."
        )

    head = PermMultipoleHead(
        input_dim=encoder.output_dim,
        avg_num_neighbors=avg_num_neighbors,
        charge=charge,
        dipole=dipole,
        quadrupole=quadrupole,
        constrain_total_charge=constrain_total_charge,
        total_charge_key=total_charge_key,
        hidden_dim=hidden_dim,
        tensor_irreps=encoder.tensor_track_irreps,
    )
    ewald = EwaldMultipoleEnergy(
        sigma=sigma,
        dl=dl,
        prefactor=prefactor,
        remove_self_interaction=True,
        use_epsilon_r_scaling=False,
    )
    return Sonata(
        encoder=encoder,
        perm_multipole_head=head,
        ewald=ewald,
        short_range_head=short_range_head,
    )
