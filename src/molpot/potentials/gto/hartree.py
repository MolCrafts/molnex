"""GTO Hartree energy: isolated real-space pair sum or Polar k-space.

The *Hartree energy* is the classical electrostatic energy of a charge density
interacting with itself — here the density built from per-atom Gaussian-type
orbitals (GTO), i.e. charges and dipoles smeared over Gaussians of width ``σ``
instead of sitting at points.

Isolated molecules use a directed complete graph excluding self-pairs; each
directed edge carries ``½ eV_A q_i q_j erf(r/(2σ))/r``, with ``q_i`` the charge
of atom ``i`` in e and ``r`` the pair separation in Å. Periodic systems pass
``cell`` and switch to the official reciprocal sum
(:mod:`molpot.potentials.gto.kspace`) — not PME (particle-mesh Ewald).

Units: charges e, positions Å, energy eV. Prefactor ``eV_A``, the Coulomb
constant ``1/(4πε₀)`` in eV·Å/e²:
:data:`molpot.potentials.elec.prefactors.eV_A`.

Reference:
    Batatia et al. "MACE-POLAR-1" arXiv:2602.19411, §II.2.6b and SI V.6.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Self

import torch
import torch.nn as nn

from molix.F.scatter import scatter_sum_compile_safe
from molpot.potentials.elec.prefactors import eV_A
from molpot.potentials.gto.kernel import GTOCoulombKernel
from molpot.potentials.gto.kspace import (
    KSpaceGrid,
    cartesian_dipole,
    kspace_cutoff,
    reciprocal_basis,
    structure_factor,
)

#: Local on purpose: ``kspace`` owns its own copy and is free to rename it.
#: A shared ``2π`` is not worth an underscore-crossing import between siblings.
_TWOPI = 2.0 * math.pi


def _cl_multipoles(ell: int, sigma: float) -> float:
    """GTO radial normalisation ``C_ℓ(σ)`` with ``normalize='multipoles'``."""
    l_dep = math.sqrt(4.0 * math.pi / (2 * ell + 1)) * 2 ** ((2 * ell + 1) / 2)
    l_dep *= math.gamma((2 * ell + 3) / 2)
    return 1.0 / (l_dep * sigma ** (2 * ell + 3))


def gto_self_overlap(sigma: float, l_max: int) -> torch.Tensor:
    """On-site GTO Coulomb overlap (official ``GTOSelfInteractionBlock``).

    Layout is one copy of ``Y_{l_max}``: monopole then dipole ``(y, z, x)``.
    Isolated Hartree adds ``½ Σ_i Σ_b ω_b q_{i b}²``. The dump at ``σ=1.5 Å``,
    ``l_max=1`` is ``ω = [eV_A/(σ√π), 0.40119, 0.40119, 0.40119]``.

    Args:
        sigma: GTO width in Å.
        l_max: Multipole order, 0 or 1.

    Returns:
        ``((l_max+1)²,)`` overlap in eV per (e or e·Å)².
    """
    n = (l_max + 1) ** 2
    out = torch.zeros(n, dtype=torch.float64)
    field_constant = eV_A * 4.0 * math.pi
    grid = torch.linspace(1.0e-4, 10.0 * float(sigma), 10000, dtype=torch.float64)
    r = grid
    sigma_f = float(sigma)
    for ell in range(l_max + 1):
        a = (2 * ell + 3) / 2
        p = torch.special.gammainc(torch.tensor(a, dtype=torch.float64), 0.5 * r * r / sigma_f**2)
        f1 = (2 ** (ell + 0.5)) * (sigma_f ** (2 * ell + 3)) * p * math.gamma(a) / r.pow(ell + 1)
        f2 = (sigma_f**2) * r.pow(ell) * torch.exp(-0.5 * r * r / sigma_f**2)
        integrand = r.pow(ell + 2) * torch.exp(-0.5 * r * r / sigma_f**2) * (f1 + f2)
        value = float(torch.trapezoid(integrand, r))
        cl = _cl_multipoles(ell, sigma_f)
        coeff = cl * cl * field_constant / (2 * ell + 1) * value
        start = ell**2
        out[start : start + 2 * ell + 1] = coeff
    return out


def _skip_diagonal(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """All directed pairs ``(i, j)`` with ``i != j``, no ``N×N`` mask."""
    if n < 2:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    idx = torch.arange(n * (n - 1), device=device)
    src = torch.div(idx, n - 1, rounding_mode="floor")
    dst = idx.remainder(n - 1)
    return src, dst + (dst >= src).to(dtype=dst.dtype)


def _skip_equal_orig_regular(
    n_atoms: int, copies: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed pairs of ``n_atoms * copies`` sites whose original atom differs.

    Layout is ``orig = arange(n_atoms).repeat_interleave(copies)``.
    """
    atom_src, atom_dst = _skip_diagonal(n_atoms, device)
    copy = torch.arange(copies, device=device)
    src, dst = torch.broadcast_tensors(
        atom_src[:, None, None] * copies + copy[None, :, None],
        atom_dst[:, None, None] * copies + copy[None, None, :],
    )
    return src.reshape(-1), dst.reshape(-1)


def _complete_directed_pairs(
    batch: torch.Tensor,
    orig: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed pairs that share a graph and come from different original atoms.

    ``orig`` is the pre-duplication atom id (identity for ``l_max=0``; repeated
    4× for the dipole finite-difference copies). Copies of the same atom never
    interact — that is the optional self-interaction term.
    """
    n = int(batch.shape[0])
    device = batch.device
    empty = torch.empty(0, dtype=torch.long, device=device)
    if n == 0:
        return empty, empty
    single = bool((batch == batch[0]).all().item())
    if orig is None:
        src, dst = _skip_diagonal(n, device)
        if single:
            return src, dst
        keep = batch[src] == batch[dst]
        return src[keep], dst[keep]

    if single:
        first = orig[0]
        copies = int((orig == first).sum().item())
        n_atoms = n // copies if copies else 0
        if copies > 1 and n_atoms * copies == n:
            expected = torch.arange(n_atoms, device=device, dtype=orig.dtype).repeat_interleave(
                copies
            )
            if torch.equal(orig, expected):
                return _skip_equal_orig_regular(n_atoms, copies, device)
        src, dst = _skip_diagonal(n, device)
        keep = orig[src] != orig[dst]
        return src[keep], dst[keep]

    src, dst = _skip_diagonal(n, device)
    keep = (batch[src] == batch[dst]) & (orig[src] != orig[dst])
    return src[keep], dst[keep]


def cached_complete_directed_pairs(
    owner: object,
    batch: torch.Tensor,
    orig: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse pair indices while ``N``, graph membership and orig layout stay put."""
    osum = 0 if orig is None else int(orig.sum().item()) + 1
    key = (int(batch.shape[0]), int(batch.sum().item()), osum, batch.device)
    cached = getattr(owner, "_pair_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    src, dst = _complete_directed_pairs(batch, orig)
    owner._pair_cache = (key, src, dst)
    return src, dst


class _KSpaceGridHolder(nn.Module):
    """Device glue for a module that owns a :class:`KSpaceGrid`.

    The grid is a plain attribute rather than a submodule — it is runtime
    geometry, not model state, so it must stay out of the ``state_dict`` — and
    therefore nothing in the module tree carries it along when the holder
    moves. Both periodic GTO blocks need exactly this and nothing else, which
    is why it is a mixin here rather than a second copy in ``features.py``.
    """

    #: Annotated, never assigned at class level: the holder's ``__init__``
    #: builds it, and ``nn.Module.__setattr__`` keeps it a plain attribute.
    kgrid: KSpaceGrid

    def bind_cell(self, cell: torch.Tensor) -> None:
        """Size (first call) or capacity-check (later calls) the k-space grid.

        Eager only — see the holder's lifecycle note. Call it once after the
        module is on its final device and before compiling or exporting.

        Args:
            cell: Lattice vectors ``(3, 3)`` or ``(1, 3, 3)`` in Å.

        Raises:
            ValueError: If ``cell`` is neither ``(3, 3)`` nor ``(1, 3, 3)``.
            RuntimeError: On the first call when the mesh would exceed the
                grid's ``max_axis_index``; on a later call when the cell has
                outgrown the capacity fixed by the first one (raise
                ``kspace_capacity_factor`` or build a new module).
        """
        self.kgrid.bind(cell)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        """Carry the k-space grid along with ``.to()`` / ``.cuda()`` / casts.

        The grid follows the holder's first buffer (its integer box is index
        metadata and is never cast to a floating dtype). A holder that has no
        buffer at all leaves the grid where it is.
        """
        super()._apply(fn, recurse)
        buffer = next(self.buffers(), None)
        if buffer is not None:
            self.kgrid.to(buffer.device)
        return self


class GTOHartreeEnergy(_KSpaceGridHolder, nn.Module):
    """GTO Hartree: isolated pair sum, or Polar SI k-space when ``cell`` is set.

    Isolated ``l_max=1`` expands each dipole as three displaced scalar charges
    (official offset 0.02 Å). Periodic ``l_max=1`` uses the analytic dipole
    structure factor (SI V.6), not the finite-difference copies.

    **Periodic lifecycle.** ``construct → .to(device) → bind_cell(cell)``
    (eager: this is where :class:`~molpot.potentials.gto.kspace.KSpaceGrid`
    sizes its integer box, once) ``→ compile / export → forward`` each step,
    passing ``cell`` and an explicit ``num_graphs``. Eager ``forward`` calls
    :meth:`bind_cell` for you and therefore re-checks the capacity every step;
    inside ``torch.compile`` that branch folds away, so a grid that was never
    bound fails at trace time with the grid's own named message. The grid is a
    plain attribute, not a submodule — it stays out of the ``state_dict``.

    **Trading head-room for k-points.** ``kspace_capacity_factor`` is the
    supported knob for that trade (see Args). A grid never resizes after its
    first bind, so a different cutoff — or a different head-room — means a new
    module, not a re-bound one.

    Args:
        sigma: GTO width in Å (Polar-1-M dump: 1.5).
        l_max: Multipole order, 0 (charges only) or 1 (charges and dipoles).
        include_self_interaction: Whether the on-site GTO overlap
            ``½ Σ_i Σ_b ω_b q_{ib}²`` belongs to the energy — ``ω_b`` is the
            overlap of basis component ``b`` with itself
            (:func:`gto_self_overlap`) and ``q_{ib}`` that component's value on
            atom ``i``. This constructor defaults to ``False``; the released
            Polar checkpoint turns it on (its
            ``electrostatic_self_interaction`` is ``True``). The two paths
            start from opposite ends, so the flag acts differently on each
            (SI V.6): the isolated pair sum skips ``i=i``, so the term is
            **added** when True and nothing happens when False; the reciprocal
            sum already contains the ``i=j`` term, so nothing is added when
            True and the term is **subtracted** when False. Adding it on the
            periodic path would double-count it.
        offset: Finite-difference displacement in Å for isolated ``l_max=1``.
        kspace_cutoff_factor: Polar ``κ``. Training default ``1.0``.
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
            hence the energy up to floating-point round-off (the unit test pins
            ``atol=1e-12`` eV, not bit-identity: padding changes the length of
            the reduction, and float addition is not associative) — does not
            depend on this factor. Values below ``1`` are rejected by
            :class:`~molpot.potentials.gto.kspace.KSpaceGrid`.

    Raises:
        NotImplementedError: If ``l_max`` is neither ``0`` nor ``1``.
        ValueError: If ``kspace_cutoff_factor`` is not strictly positive, or if
            ``kspace_capacity_factor`` is below ``1`` — both are rejected while
            the k-grid is built.
    """

    def __init__(
        self,
        *,
        sigma: float,
        l_max: int = 0,
        include_self_interaction: bool = False,
        offset: float = 0.02,
        kspace_cutoff_factor: float = 1.0,
        k_cutoff: float | None = None,
        kspace_capacity_factor: float = 1.2,
    ) -> None:
        super().__init__()
        if l_max not in (0, 1):
            raise NotImplementedError("GTOHartreeEnergy supports l_max=0 or 1")
        self.l_max = l_max
        self.include_self_interaction = include_self_interaction
        self.offset = float(offset)
        self.sigma = float(sigma)
        self.kspace_cutoff_factor = float(kspace_cutoff_factor)
        self.kernel = GTOCoulombKernel(sigma=sigma)
        self.register_buffer("self_overlap", gto_self_overlap(sigma, l_max), persistent=True)
        # One home for the cutoff: a Python float of the ctor arguments, read
        # once here and frozen on the grid, never re-derived per forward.
        self.kgrid = KSpaceGrid(
            k_cutoff=k_cutoff
            if k_cutoff is not None
            else kspace_cutoff(self.sigma, self.l_max, self.kspace_cutoff_factor),
            capacity_factor=kspace_capacity_factor,
        )

    @property
    def k_cutoff(self) -> float:
        """Reciprocal cutoff in Å⁻¹, as frozen on the grid at construction.

        Computed once by :func:`~molpot.potentials.gto.kspace.kspace_cutoff`
        from ``sigma``, ``l_max`` and ``kspace_cutoff_factor``. That formula is
        the reference implementation's convention, **not** a paper equation —
        do not cite an equation number for it.
        """
        return self.kgrid.k_cutoff

    def _pair_energy(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        orig: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src, dst = cached_complete_directed_pairs(self, batch, orig)
        if src.numel() == 0:
            return charges.new_zeros(num_graphs)
        r = (positions[dst] - positions[src]).norm(dim=-1)
        edge = 0.5 * eV_A * self.kernel(r) * charges[src] * charges[dst]
        node = scatter_sum_compile_safe(edge, dst, charges.shape[0])
        return scatter_sum_compile_safe(node, batch, num_graphs)

    def _self_energy(
        self, feats: torch.Tensor, batch: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        overlap = self.self_overlap.to(dtype=feats.dtype)
        width = min(overlap.shape[0], feats.shape[-1])
        node = 0.5 * (feats[:, :width] * overlap[:width] * feats[:, :width]).sum(dim=-1)
        return scatter_sum_compile_safe(node, batch, num_graphs)

    def forward(
        self,
        source_feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Hartree energy per graph.

        Args:
            source_feats: Multipoles ``(N, (l_max+1)**2)`` in e / e·Å.
            positions: Cartesian positions ``(N, 3)`` in Å.
            batch: Graph id ``(N,)``.
            cell: Optional lattice vectors ``(3, 3)`` or ``(1, 3, 3)`` in Å,
                one per row. When set, the official k-space sum runs and the
                batch must hold exactly one system (``B = 1``). When omitted,
                the isolated real-space pair sum runs.
            num_graphs: Number of graphs ``B``, dimensionless. Leaving it
                ``None`` derives it from ``batch`` with a host sync and is
                therefore **eager-only**: that fallback lives here and nowhere
                deeper, so the periodic call chain below stays compile-clean.

        Returns:
            Per-graph energy ``(B,)`` in eV.

        Raises:
            ValueError: If ``cell`` is given together with more than one graph
                (the reciprocal sum knows one cell), or if ``cell`` is neither
                ``(3, 3)`` nor ``(1, 3, 3)``.
            RuntimeError: On the eager periodic path, when the cell has
                outgrown the k-grid capacity; under ``torch.compile``, when the
                grid was never bound with :meth:`bind_cell` (an over-capacity
                cell then poisons the weights with NaN instead of raising, so
                the k-sum is never silently truncated).
        """
        feats = source_feats.reshape(source_feats.shape[0], -1)
        if num_graphs is None:
            n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        else:
            n_graphs = num_graphs
        if cell is not None:
            # First use of ``torch.compiler.is_compiling()`` under ``src/``:
            # it is a trace-time constant, so dynamo folds this branch away
            # (no host sync in the graph) while eager callers keep sizing the
            # grid on the first step and capacity-checking it on every later
            # one. Bind before compiling; the grid itself never resizes.
            if not torch.compiler.is_compiling():
                self.bind_cell(cell)
            energy = self._kspace_energy(feats, positions, batch, cell, n_graphs)
            # The reciprocal sum runs over all i, j — the on-site term is
            # already in there. Adding it again is a double count.
            if not self.include_self_interaction:
                energy = energy - self._self_energy(feats, batch, n_graphs)
            return energy
        if self.l_max == 0:
            energy = self._pair_energy(feats[:, 0], positions, batch, n_graphs)
        else:
            offset = self.offset
            n = positions.shape[0]
            extended_pos = positions.repeat_interleave(4, dim=0)
            extended_pos[1::4, 0] = extended_pos[1::4, 0] + offset
            extended_pos[2::4, 1] = extended_pos[2::4, 1] + offset
            extended_pos[3::4, 2] = extended_pos[3::4, 2] + offset
            charges = feats.new_zeros(n * 4)
            charges[1::4] = feats[:, 3] / offset
            charges[2::4] = feats[:, 1] / offset
            charges[3::4] = feats[:, 2] / offset
            charges[0::4] = feats[:, 0] - (charges[1::4] + charges[2::4] + charges[3::4])
            extended_batch = batch.repeat_interleave(4)
            orig = torch.arange(n, device=batch.device).repeat_interleave(4)
            energy = self._pair_energy(charges, extended_pos, extended_batch, n_graphs, orig)
        if self.include_self_interaction:
            energy = energy + self._self_energy(feats, batch, n_graphs)
        return energy

    def _kspace_energy(
        self,
        feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Polar SI V.6 reciprocal sum for one cell, ``(1,)`` in eV.

        ``E = eV_A · (2π/Ω) Σ_n w(n) exp(-σ² k²) / k² |S(k)|²`` over the
        grid's fixed half box: ``w`` is ``0`` outside the cutoff, so padding
        rows contribute exactly ``+0.0``, and the guarded denominator keeps
        ``0 · inf`` out of the sum. Traceable throughout — the caller owns
        binding (``forward`` does it eagerly), so an over-capacity cell
        reaching here returns NaN rather than a truncated sum.

        Args:
            feats: Flat multipoles ``(N, (l_max+1)**2)`` in e / e·Å.
            positions: ``(N, 3)`` Å.
            batch: Graph id ``(N,)``, unused on this single-system path.
            cell: Lattice ``(3, 3)`` or ``(1, 3, 3)`` Å.
            num_graphs: Number of graphs; must be ``1``.

        Returns:
            Energy ``(1,)`` in eV.

        Raises:
            ValueError: If ``num_graphs`` is greater than ``1``.
        """
        if num_graphs > 1:
            raise ValueError(
                "GTO k-space Hartree is single-system (one cell); "
                f"got {num_graphs} graphs. Drop cell for isolated molecules."
            )
        dipoles = cartesian_dipole(feats) if self.l_max == 1 and feats.shape[-1] >= 4 else None
        # ``vectors`` validates the cell shape first, so a bad cell raises the
        # grid's own named ValueError rather than a determinant of the wrong
        # thing; ``det`` then covers ``(3, 3)`` and ``(1, 3, 3)`` statically.
        _, k_sq_guarded, weights = self.kgrid.vectors(cell)
        volume = torch.linalg.det(cell).abs().reshape(())
        s_real, s_imag = structure_factor(
            positions,
            feats[:, 0],
            self.kgrid.n,
            reciprocal_basis(cell),
            self.kgrid.capacity,
            dipoles,
        )
        s_sq = s_real * s_real + s_imag * s_imag
        kfac = torch.exp(-(self.sigma * self.sigma) * k_sq_guarded) / k_sq_guarded
        energy = eV_A * _TWOPI / volume * (weights * kfac * s_sq).sum()
        return energy.reshape(1)
