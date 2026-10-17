"""GTO electrostatic *field* features (Polar field-update path).

Isolated molecules: real-space finite-difference port of ACEsuit
``RealSpaceFiniteDifferenceElectrostaticFeatures``. The pair kernel is
``erf(r / (2 σ_comb)) / r`` with
``σ_comb = sqrt((σ_density² + σ_feature²) / 2)``.

Periodic systems: pass ``cell`` for the official k-space potential and
analytic ``∇φ`` (same GTO Fourier convention as Hartree).

Output is native ``cue.ir_mul`` (the cuEquivariance irreducible-representation-
major layout: the ``l = 0`` block first, then the ``l = 1`` block as
``(y, z, x)`` per radial width).

Units: charges e, positions Å. Both feature blocks come out in **eV/e**, the
unit of an electrostatic potential here: the ``l = 0`` block is ``φ`` and the
``l = 1`` block is the finite-difference potential difference
``≈ offset · ∇φ`` (``∇φ`` in eV·e⁻¹·Å⁻¹ times ``offset`` in Å). Both are then
scaled by the dimensionless ``l0_factors`` / ``l1_factors`` normalisations, so
the unit is the same on either side.

Sign convention: the ``l = 1`` block carries ``+∇φ`` — the finite-difference
``φ(r + δ) − φ(r)`` of the isolated path, and its analytic counterpart in
k-space — **not** the physical electric field ``E = −∇φ``. Both paths agree on
this sign; do not flip one of them.

Reference:
    Batatia et al. "MACE-POLAR-1" arXiv:2602.19411, §II.2.6b and SI V.6.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from molix.F.scatter import scatter_sum_compile_safe
from molpot.potentials.elec.prefactors import eV_A
from molpot.potentials.gto.hartree import _KSpaceGridHolder, cached_complete_directed_pairs
from molpot.potentials.gto.kspace import (
    KSpaceGrid,
    cartesian_dipole,
    kspace_cutoff,
    kspace_phi_field,
    reciprocal_basis,
)

#: Local on purpose: ``kspace`` owns its own constants and is free to rename
#: them. A shared ``4π`` is not worth an underscore-crossing sibling import.
_FOURPI = 4.0 * math.pi


def _cl_sigma(ell: int, sigma: float, normalize: str) -> float:
    """GTO radial normalisation ``C_ℓ(σ)`` (official ``get_Cl_sigma``)."""
    if normalize == "multipoles":
        l_dep = math.sqrt(4.0 * math.pi / (2 * ell + 1)) * 2 ** ((2 * ell + 1) / 2)
        l_dep *= math.gamma((2 * ell + 3) / 2)
        return 1.0 / (l_dep * sigma ** (2 * ell + 3))
    if normalize == "receiver":
        l_dep = 2 ** ((ell + 1) / 2) * math.gamma((ell + 3) / 2)
        return 1.0 / (l_dep * sigma ** (ell + 3))
    raise ValueError(f"normalize must be 'multipoles' or 'receiver', got {normalize!r}")


class GTOFieldFeatures(_KSpaceGridHolder, nn.Module):
    """Per-atom electrostatic potential / field features from GTO multipoles.

    **Periodic lifecycle.** ``construct → .to(device) → bind_cell(cell)``
    (eager: this is where :class:`~molpot.potentials.gto.kspace.KSpaceGrid`
    sizes its integer box, once) ``→ compile / export → forward`` each step,
    passing ``cell`` and an explicit ``num_graphs``. Eager ``forward`` binds
    for you and therefore re-checks capacity every step; inside
    ``torch.compile`` that branch folds away, so a grid that was never bound
    fails at trace time with the grid's own named message. The grid is a plain
    attribute, not a submodule — it stays out of the ``state_dict``, as do the
    five ctor-derived buffers below (they are pure functions of the arguments
    in this signature; a checkpoint must not be able to contradict them).

    **Trading head-room for k-points.** ``kspace_capacity_factor`` is the
    supported knob for that trade (see Args). A grid never resizes after its
    first bind, so a different cutoff — or a different head-room — means a new
    module, not a re-bound one.

    Args:
        density_sigma: Source GTO width in Å (Polar-1-M dump: 1.5) — the width
            of the Gaussians carrying the *sources*, i.e. the multipoles read
            from ``source_feats``.
        density_lmax: Source multipole order, 0 (charges) or 1 (charges and
            dipoles); dump: 1.
        feature_lmax: Feature multipole order, 0 (potential only) or 1
            (potential and field); dump: 1.
        feature_widths: Projection GTO widths in Å, one per radial channel of
            the output (dump: ``[1.5, 3.0]``) — the width of the Gaussian each
            atom uses to *sample* the potential.
        offset: Finite-difference displacement in Å (official features: 0.1).
        kspace_cutoff_factor: Polar ``κ`` for the PBC path. Training default
            ``1.0``.
        k_cutoff: Explicit reciprocal cutoff in Å⁻¹, overriding the heuristic
            factor. Use the serialized runtime value when loading official weights.
        kspace_capacity_factor: Dimensionless head-room on the per-axis
            reciprocal index the k-grid sizes itself to,
            ``N_c = ceil(f · ‖a_i‖ k_cutoff / 2π)``, where ``a_i`` is lattice
            vector ``i`` in Å. ``1.2`` leaves room for a cell that expands
            under a barostat (constant-pressure NPT dynamics); a caller at
            fixed volume (NVT) may pass ``1.0`` and pay for fewer k-points.
            Head-room is an allocation decision, never a physics one: the extra
            rows fall outside the cutoff, carry weight ``0`` and contribute
            exactly ``+0.0``, so the set of k-points that reaches the sum — and
            hence the features up to floating-point round-off, not bit-identity
            (padding changes the length of the reduction, and float addition is
            not associative) — does not depend on this factor. Values below
            ``1`` are rejected by
            :class:`~molpot.potentials.gto.kspace.KSpaceGrid`.
        include_self_interaction: Whether an atom sees its own multipole in
            its own features (Polar dump ``field_si``: False). The real-space
            pair sum never pairs an atom with itself, so the flag is inert
            there. The reciprocal sum does include ``i=j``, so when False the
            ``r → 0`` limits :attr:`self_overlap_l0` / :attr:`self_overlap_l1`
            are subtracted on the PBC (periodic-boundary-condition) path to
            match (SI V.6).

    Raises:
        NotImplementedError: If ``density_lmax`` or ``feature_lmax`` is neither
            ``0`` nor ``1``.
        ValueError: If ``feature_widths`` is empty, if ``kspace_cutoff_factor``
            is not strictly positive, or if ``kspace_capacity_factor`` is below
            ``1`` (the last two are rejected while the k-grid is built).
    """

    def __init__(
        self,
        *,
        density_sigma: float,
        density_lmax: int,
        feature_lmax: int,
        feature_widths: list[float],
        offset: float = 0.1,
        kspace_cutoff_factor: float = 1.0,
        k_cutoff: float | None = None,
        kspace_capacity_factor: float = 1.2,
        include_self_interaction: bool = False,
    ) -> None:
        super().__init__()
        if density_lmax not in (0, 1) or feature_lmax not in (0, 1):
            raise NotImplementedError("GTOFieldFeatures supports l_max=0 or 1")
        if not feature_widths:
            raise ValueError("feature_widths must be non-empty")
        self.density_lmax = density_lmax
        self.feature_lmax = feature_lmax
        self.offset = float(offset)
        self.num_radial = len(feature_widths)
        comb = [math.sqrt((density_sigma**2 + float(w) ** 2) / 2.0) for w in feature_widths]
        self.register_buffer(
            "combined_sigma", torch.tensor(comb, dtype=torch.float64), persistent=False
        )
        l0 = [
            _cl_sigma(0, float(w), "receiver") / _cl_sigma(0, float(w), "multipoles")
            for w in feature_widths
        ]
        self.register_buffer("l0_factors", torch.tensor(l0, dtype=torch.float64), persistent=False)
        l1 = [
            math.sqrt(3.0)
            * float(w) ** 2
            * (_cl_sigma(1, float(w), "receiver") / _cl_sigma(0, float(w), "multipoles"))
            / self.offset
            for w in feature_widths
        ]
        self.register_buffer("l1_factors", torch.tensor(l1, dtype=torch.float64), persistent=False)
        # ``r → 0`` limits of the two real-space kernels, per feature width:
        # ``erf(r / 2σ_c) / r → 1 / (σ_c √π)`` for the potential, and the
        # Gaussian dipole's own field at its centre, ``+∇φ = eV_A μ /
        # (6 √π σ_c³)``, for the field. Subtracted on the k-space path when
        # ``include_self_interaction`` is off.
        self_l0 = [
            factor * eV_A / (sigma * math.sqrt(math.pi))
            for factor, sigma in zip(l0, comb, strict=True)
        ]
        self.register_buffer(
            "self_overlap_l0", torch.tensor(self_l0, dtype=torch.float64), persistent=False
        )
        self_l1 = [
            factor * self.offset * eV_A / (6.0 * math.sqrt(math.pi) * sigma**3)
            for factor, sigma in zip(l1, comb, strict=True)
        ]
        self.register_buffer(
            "self_overlap_l1", torch.tensor(self_l1, dtype=torch.float64), persistent=False
        )
        self.density_sigma = float(density_sigma)
        self.kspace_cutoff_factor = float(kspace_cutoff_factor)
        self.include_self_interaction = include_self_interaction
        # One home for the cutoff: ``σ_min`` is a Python float of the ctor
        # arguments (``min(comb)`` is the same number ``combined_sigma`` holds),
        # read once here and frozen on the grid — never per forward.
        self.kgrid = KSpaceGrid(
            k_cutoff=k_cutoff
            if k_cutoff is not None
            else kspace_cutoff(
                min(self.density_sigma, min(comb)),
                max(density_lmax, feature_lmax),
                self.kspace_cutoff_factor,
            ),
            capacity_factor=kspace_capacity_factor,
        )

    @property
    def out_dim(self) -> int:
        """Flattened ``ir_mul`` width ``n_radial * (feature_lmax + 1)**2``."""
        return self.num_radial * (self.feature_lmax + 1) ** 2

    @property
    def k_cutoff(self) -> float:
        """Reciprocal cutoff in Å⁻¹, as frozen on the grid at construction.

        Computed once by :func:`~molpot.potentials.gto.kspace.kspace_cutoff`
        from the smallest width in play (``min(density_sigma, min σ_comb)``),
        the larger of the two multipole orders and ``kspace_cutoff_factor``.
        That formula is the reference implementation's convention, **not** a
        paper equation — do not cite an equation number for it.
        """
        return self.kgrid.k_cutoff

    def _pair_features(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        orig: torch.Tensor | None,
        n_out: int,
    ) -> torch.Tensor:
        src, dst = cached_complete_directed_pairs(self, batch, orig)
        out = charges.new_zeros(n_out, self.num_radial)
        if src.numel() == 0:
            return out
        r = (positions[src] - positions[dst]).norm(dim=-1, keepdim=True)
        sigma = self.combined_sigma.to(dtype=charges.dtype)
        kernel = torch.erf(r / (2.0 * sigma)) / r.clamp(min=1e-6)
        contrib = charges[src].unsqueeze(-1) * kernel * eV_A
        return scatter_sum_compile_safe(contrib, dst, n_out)

    def forward(
        self,
        source_feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor | None = None,
        *,
        num_graphs: int,
    ) -> torch.Tensor:
        """Evaluate field features.

        Args:
            source_feats: Multipoles ``(N, (density_lmax+1)**2)``: monopoles in
                e, dipole components in e·Å.
            positions: Cartesian positions ``(N, 3)`` in Å.
            batch: Graph id ``(N,)``.
            cell: Optional lattice vectors ``(3, 3)`` or ``(1, 3, 3)`` in Å,
                one per row. When set, the official k-space potential / field
                is used and the batch must hold exactly one system (``B = 1``).
            num_graphs: Number of graphs ``B``, dimensionless —
                ``graphs.batch_size[0]`` of the batch. Only the periodic
                branch reduces per graph; the isolated pair sum is per atom
                and never reads it.

        Returns:
            ``(N, out_dim)`` in eV/e, laid out as ``ir_mul`` of
            ``n_radial x Y_{lmax}``.

        Raises:
            ValueError: If ``cell`` is given together with more than one graph
                (the reciprocal sum knows one cell), or if ``cell`` is neither
                ``(3, 3)`` nor ``(1, 3, 3)``.
            RuntimeError: On the eager periodic path, when the cell has
                outgrown the k-grid capacity; under ``torch.compile``, when the
                grid was never bound with :meth:`bind_cell` (an over-capacity
                cell then poisons the features with NaN instead of raising, so
                the k-sum is never silently truncated).
        """
        feats = source_feats.reshape(source_feats.shape[0], -1)
        n = positions.shape[0]
        if cell is not None:
            # First use of ``torch.compiler.is_compiling()`` under ``src/``:
            # it is a trace-time constant, so dynamo folds this branch away
            # (no host sync in the graph) while eager callers keep sizing the
            # grid on the first step and capacity-checking it on every later
            # one. Bind before compiling; the grid itself never resizes.
            if not torch.compiler.is_compiling():
                self.bind_cell(cell)
            return self._kspace_features(feats, positions, batch, cell, num_graphs)
        need_fd = self.density_lmax == 1 or self.feature_lmax == 1
        if not need_fd:
            scalars = self._pair_features(feats[:, 0], positions, batch, None, n)
            return scalars * self.l0_factors.to(dtype=feats.dtype)

        offset = self.offset
        ext_pos = positions.repeat_interleave(4, dim=0)
        ext_pos[1::4, 0] = ext_pos[1::4, 0] + offset
        ext_pos[2::4, 1] = ext_pos[2::4, 1] + offset
        ext_pos[3::4, 2] = ext_pos[3::4, 2] + offset
        charges = feats.new_zeros(n * 4)
        if self.density_lmax == 1 and feats.shape[-1] >= 4:
            charges[1::4] = feats[:, 3] / offset
            charges[2::4] = feats[:, 1] / offset
            charges[3::4] = feats[:, 2] / offset
            charges[0::4] = feats[:, 0] - (charges[1::4] + charges[2::4] + charges[3::4])
        else:
            charges[0::4] = feats[:, 0]
        ext_batch = batch.repeat_interleave(4)
        orig = torch.arange(n, device=batch.device).repeat_interleave(4)
        scalar = self._pair_features(charges, ext_pos, ext_batch, orig, n * 4)
        l0 = self.l0_factors.to(dtype=feats.dtype)
        if self.feature_lmax == 0:
            return scalar[0::4] * l0
        l1 = self.l1_factors.to(dtype=feats.dtype)
        # mul_ir  n x 0e + n x 1o (y, z, x per radial) → ir_mul
        scalars = scalar[0::4] * l0
        vec_y = l1 * (scalar[2::4] - scalar[0::4])
        vec_z = l1 * (scalar[3::4] - scalar[0::4])
        vec_x = l1 * (scalar[1::4] - scalar[0::4])
        vec_ir_mul = torch.stack([vec_y, vec_z, vec_x], dim=1).reshape(n, -1)
        return torch.cat([scalars, vec_ir_mul], dim=-1)

    def _kspace_potential(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        dipoles: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K-space potential and Cartesian gradient at each atom.

        ``σ_comb`` is a vector of feature widths: the source structure factor
        is shared, the Gaussian form factor is per width. Runs over the grid's
        fixed half box, so out-of-cutoff rows carry weight ``0`` and contribute
        exactly ``+0.0``, and the guarded denominator keeps ``0 · inf`` out.
        Traceable throughout — the caller owns binding (``forward`` does it
        eagerly), so an over-capacity cell reaching here gives NaN rather than
        a truncated sum.

        Args:
            charges: Monopoles ``(N,)`` e.
            positions: ``(N, 3)`` Å.
            cell: Lattice ``(3, 3)`` or ``(1, 3, 3)`` Å.
            dipoles: Optional Cartesian dipoles ``(N, 3)`` e·Å.

        Returns:
            ``phi (N, R)`` eV/e, ``field (N, 3, R)`` eV/(e·Å) as ``+∇φ``.
        """
        # ``vectors`` validates the cell shape first, so a bad cell raises the
        # grid's own named ValueError rather than a determinant of the wrong
        # thing; ``det`` then covers ``(3, 3)`` and ``(1, 3, 3)`` statically.
        _, k_sq_guarded, weights = self.kgrid.vectors(cell)
        volume = torch.linalg.det(cell).abs().reshape(())
        sigma = self.combined_sigma.to(dtype=charges.dtype)
        pre = eV_A * _FOURPI / volume
        return kspace_phi_field(
            charges,
            positions,
            self.kgrid.n,
            reciprocal_basis(cell),
            self.kgrid.capacity,
            k_sq_guarded,
            weights,
            sigma,
            pre,
            dipoles,
        )

    def _kspace_features(
        self,
        feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Periodic field features ``(N, out_dim)``, single system.

        Args:
            feats: Flat multipoles ``(N, (density_lmax+1)**2)`` in e / e·Å.
            positions: ``(N, 3)`` Å.
            batch: Graph id ``(N,)``, unused on this single-system path.
            cell: Lattice ``(3, 3)`` or ``(1, 3, 3)`` Å.
            num_graphs: Number of graphs; must be ``1``.

        Returns:
            ``(N, out_dim)`` in eV/e, laid out as ``ir_mul`` of
            ``n_radial x Y_{lmax}``.

        Raises:
            ValueError: If ``num_graphs`` is greater than ``1``.
        """
        if num_graphs > 1:
            raise ValueError(
                "GTO k-space field features are single-system (one cell); "
                f"got {num_graphs} graphs. Drop cell for isolated molecules."
            )
        dipoles = (
            cartesian_dipole(feats) if self.density_lmax == 1 and feats.shape[-1] >= 4 else None
        )
        phi, grad = self._kspace_potential(feats[:, 0], positions, cell, dipoles)
        l0 = self.l0_factors.to(dtype=feats.dtype)
        scalars = phi * l0
        if not self.include_self_interaction:
            # ``φ`` above is the k-space sum over *all* sources, this atom
            # included; the real-space path never pairs an atom with itself.
            self_l0 = self.self_overlap_l0.to(dtype=feats.dtype)
            scalars = scalars - self_l0 * feats[:, 0:1]
        if self.feature_lmax == 0:
            return scalars
        # Real-space l1 multiplies the FD difference ``φ(r+δ)-φ(r) ≈ δ ∂φ``.
        l1 = self.l1_factors.to(dtype=feats.dtype) * self.offset
        vec_y = l1 * grad[:, 1, :]
        vec_z = l1 * grad[:, 2, :]
        vec_x = l1 * grad[:, 0, :]
        if not self.include_self_interaction and dipoles is not None:
            # Source multipoles are ``(q, μ_y, μ_z, μ_x)``; a monopole's own
            # field vanishes at its centre, so only the dipole is removed.
            self_l1 = self.self_overlap_l1.to(dtype=feats.dtype)
            vec_y = vec_y - self_l1 * feats[:, 1:2]
            vec_z = vec_z - self_l1 * feats[:, 2:3]
            vec_x = vec_x - self_l1 * feats[:, 3:4]
        vec_ir_mul = torch.stack([vec_y, vec_z, vec_x], dim=1).reshape(positions.shape[0], -1)
        return torch.cat([scalars, vec_ir_mul], dim=-1)
