"""GTO Hartree energy: isolated real-space pair sum or Polar k-space.

The *Hartree energy* is the classical electrostatic energy of a charge density
interacting with itself — here the density built from per-atom Gaussian-type
orbitals (GTO), i.e. charges and dipoles smeared over Gaussians of width ``σ``
instead of sitting at points.

Isolated molecules use a directed complete graph excluding self-pairs,
enumerated per graph at a bound capacity (:func:`graph_pairs`); each directed
edge carries ``½ eV_A q_i q_j erf(r/(2σ))/r``, with ``q_i`` the charge of atom
``i`` in e and ``r`` the pair separation in Å. Periodic systems pass
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


def complete_directed_pairs(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """All ``n (n - 1)`` directed pairs ``(i, j)`` with ``i != j``.

    A pure function of the row count: ``src`` is ``i`` repeated ``n - 1``
    times and ``dst`` runs over the other rows, skipping ``i`` by one
    arithmetic step — no ``n×n`` mask, nothing read back from a tensor, no
    data-dependent shape. :func:`graph_pairs` lays this local grid over every
    graph of a batch.

    Args:
        n: Rows of one graph's capacity (atoms, or atom copies on the
            ``l_max=1`` layout).
        device: Device of the returned indices.

    Returns:
        ``src, dst``, each ``(n (n - 1),)`` of ``torch.long``.
    """
    idx = torch.arange(n * (n - 1), device=device)
    src = torch.div(idx, n - 1, rounding_mode="floor")
    dst = idx.remainder(n - 1)
    return src, dst + (dst >= src).to(dtype=dst.dtype)


def graph_pairs(
    batch: torch.Tensor, num_graphs: int, capacity: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Directed same-graph pairs of a contiguous batch, ``capacity`` rows per graph.

    **Contract:** ``batch`` is non-decreasing — collate lays atoms out graph
    by graph, so every graph is one contiguous block of rows, and the MD
    driver's single system is one block trivially. On the finite-difference
    copy layout ``batch.repeat_interleave(copies)`` keeps the blocks
    contiguous at ``copies · n_g`` rows, so the same enumeration applies with
    ``capacity = copies · max_atoms_per_graph``.

    Graph ``g`` holds ``n_g`` rows from ``offset_g = Σ_{h<g} n_h``. The local
    off-diagonal grid ``(i, j)`` of :func:`complete_directed_pairs` over
    ``capacity`` rows is laid at every offset, ``src = offset_g + i`` and
    ``dst = offset_g + j`` (clamped into the batch, so a padding row of a
    small graph indexes a real row harmlessly), and ``keep = (i < n_g) & (j <
    n_g)`` marks the pairs that exist. The cost is ``B · K (K − 1)`` rows with
    ``K = capacity``, a function of ``B`` and ``K`` alone: no data-dependent
    shape, nothing read back to the host — the molecular path traces as one
    graph and records into a CUDA graph. The counts come from ``index_add_``
    into a ``(B,)`` buffer (``bincount`` would shape its output off the data).

    Both preconditions are checked with ``torch._assert_async``, which raises
    on the eager path and stays a device-side check on the traced one.

    Args:
        batch: Graph id per row ``(N,)``, non-decreasing.
        num_graphs: Number of graphs ``B``.
        capacity: Rows any one graph may hold, ``K``.

    Returns:
        ``src, dst`` of ``torch.long`` and ``keep`` of ``torch.bool``, each
        ``(B · K (K − 1),)``.

    Raises:
        RuntimeError: If ``batch`` is not non-decreasing, or if a graph holds
            more than ``capacity`` rows.
    """
    counts = batch.new_zeros(num_graphs).index_add_(0, batch, torch.ones_like(batch))
    torch._assert_async(
        (batch[1:] >= batch[:-1]).all(),
        "graph_pairs: batch must be non-decreasing (one contiguous block of rows per graph)",
    )
    torch._assert_async(
        (counts <= capacity).all(),
        "graph_pairs: a graph holds more rows than the bound capacity; "
        "raise bind_capacity(max_atoms_per_graph)",
    )
    offset = (torch.cumsum(counts, dim=0) - counts).unsqueeze(1)
    i, j = complete_directed_pairs(capacity, batch.device)
    last = batch.shape[0] - 1
    src = (offset + i).clamp(max=last).reshape(-1)
    dst = (offset + j).clamp(max=last).reshape(-1)
    counts = counts.unsqueeze(1)
    keep = ((i < counts) & (j < counts)).reshape(-1)
    return src, dst, keep


class _GTOBindings(nn.Module):
    """The two runtime bindings of a GTO block, and the grid's device glue.

    Each block binds two things after construction and before compiling,
    neither of them model state (both stay out of the ``state_dict``): a
    :class:`KSpaceGrid` for the periodic path (:meth:`bind_cell`) and a
    rows-per-graph capacity for the isolated pair list
    (:meth:`bind_capacity`). The grid is a plain attribute rather than a
    submodule, so nothing in the module tree carries it along when the holder
    moves; the capacity is a plain ``int`` — a shape, like the grid's. Both GTO
    blocks need exactly this and nothing else, which is why it is a mixin
    here rather than a second copy in ``features.py``.
    """

    #: Annotated, never assigned at class level: the holder's ``__init__``
    #: builds it, and ``nn.Module.__setattr__`` keeps it a plain attribute.
    kgrid: KSpaceGrid

    #: Atoms any one graph may hold on the isolated path; ``None`` until
    #: :meth:`bind_capacity`. Read through :attr:`max_atoms_per_graph`.
    _max_atoms_per_graph: int | None

    def __init__(self) -> None:
        super().__init__()
        self._max_atoms_per_graph = None

    @property
    def max_atoms_per_graph(self) -> int:
        """Atoms any one graph may hold, as fixed by :meth:`bind_capacity`.

        Raises:
            RuntimeError: Before :meth:`bind_capacity` — at trace time under
                ``torch.compile``, on the first isolated forward in eager.
        """
        if self._max_atoms_per_graph is None:
            raise RuntimeError(
                f"{type(self).__name__} has no pair capacity: call "
                "bind_capacity(max_atoms_per_graph) before the isolated forward or compiling"
            )
        return self._max_atoms_per_graph

    def bind_capacity(self, max_atoms_per_graph: int) -> None:
        """Bound the atoms of any one graph for the isolated pair list.

        The pair list is enumerated per graph at this bound
        (:func:`graph_pairs`), so a batch costs ``B · K (K − 1)`` rows with
        ``K`` the bound (times the copies of the ``l_max=1`` layout) — a
        static shape, which is what lets the isolated path trace as one graph.
        A graph that holds more rows than the bound raises on the forward,
        never truncates. Unlike the k-grid, nothing is allocated here: a
        later call with another value is simply a new static shape (one
        recompile).

        Args:
            max_atoms_per_graph: Largest atom count of any one graph in any
                batch this module will see — ghost atoms a padder assigns to
                a graph slot included, as they are rows of that graph.

        Raises:
            ValueError: If ``max_atoms_per_graph`` is below ``1``.
        """
        if max_atoms_per_graph < 1:
            raise ValueError(f"max_atoms_per_graph must be >= 1, got {max_atoms_per_graph}")
        self._max_atoms_per_graph = int(max_atoms_per_graph)

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


class GTOHartreeEnergy(_GTOBindings, nn.Module):
    """GTO Hartree: isolated pair sum, or Polar SI k-space when ``cell`` is set.

    Isolated ``l_max=1`` expands each dipole as three displaced scalar charges
    (official offset 0.02 Å). Periodic ``l_max=1`` uses the analytic dipole
    structure factor (SI V.6), not the finite-difference copies.

    **Isolated lifecycle.** ``construct → bind_capacity(max_atoms_per_graph)
    → compile / export → forward`` each step. The pair list is enumerated per
    graph at that bound (:func:`graph_pairs`): ``B · K (K − 1)`` rows with
    ``K`` the bound (``4 K`` on the ``l_max=1`` copy layout), so a batch of
    small molecules pays for its own pairs rather than the batch's square.
    Unbound, the forward raises naming :meth:`bind_capacity`.

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
        capacity: int,
        orig: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked per-graph pair sum ``½ eV_A Σ_{i≠j} q_i q_j erf(r_ij/(2σ))/r_ij``.

        The ``B · K (K − 1)`` directed pairs of :func:`graph_pairs` are
        evaluated and the ones that do not exist — padding rows of a graph
        below the capacity, and on the copy layout the copies of one atom —
        are zeroed by the mask. A masked pair may sit at zero distance (a
        padding row clamped onto a real one; coincident copies), so the
        distance is replaced by ``1`` before the kernel and the kernel by
        ``0`` after it — no ``0 · inf`` in the forward, no NaN in the
        backward. The rows come out graph-major, so the per-graph sum is one
        reshape.

        Args:
            charges: Point charges ``(N,)`` in e.
            positions: ``(N, 3)`` in Å.
            batch: Graph id ``(N,)``, non-decreasing.
            num_graphs: Number of graphs ``B``.
            capacity: Rows any one graph may hold, ``K`` (``4 ·
                max_atoms_per_graph`` on the copy layout).
            orig: Original atom id ``(N,)`` on the copy layout, else ``None``.

        Returns:
            Per-graph energy ``(B,)`` in eV.
        """
        src, dst, keep = graph_pairs(batch, num_graphs, capacity)
        if orig is not None:
            keep = keep & (orig[src] != orig[dst])
        r = (positions[dst] - positions[src]).norm(dim=-1)
        kernel = torch.where(keep, self.kernel(torch.where(keep, r, 1.0)), 0.0)
        edge = 0.5 * eV_A * kernel * charges[src] * charges[dst]
        return edge.reshape(num_graphs, -1).sum(dim=1)

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
        *,
        num_graphs: int,
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
            num_graphs: Number of graphs ``B``, dimensionless —
                ``graphs.batch_size[0]`` of the batch.

        Returns:
            Per-graph energy ``(B,)`` in eV.

        Raises:
            ValueError: If ``cell`` is given together with more than one graph
                (the reciprocal sum knows one cell), or if ``cell`` is neither
                ``(3, 3)`` nor ``(1, 3, 3)``.
            RuntimeError: On the isolated path, when the pair capacity was
                never bound with :meth:`bind_capacity`, when ``batch`` is not
                non-decreasing, or when a graph holds more atoms than the
                bound (:func:`graph_pairs`). On the eager periodic path, when
                the cell has outgrown the k-grid capacity; under
                ``torch.compile``, when the grid was never bound with
                :meth:`bind_cell` (an over-capacity cell then poisons the
                weights with NaN instead of raising, so the k-sum is never
                silently truncated).
        """
        feats = source_feats.reshape(source_feats.shape[0], -1)
        if cell is not None:
            # First use of ``torch.compiler.is_compiling()`` under ``src/``:
            # it is a trace-time constant, so dynamo folds this branch away
            # (no host sync in the graph) while eager callers keep sizing the
            # grid on the first step and capacity-checking it on every later
            # one. Bind before compiling; the grid itself never resizes.
            if not torch.compiler.is_compiling():
                self.bind_cell(cell)
            energy = self._kspace_energy(feats, positions, batch, cell, num_graphs)
            # The reciprocal sum runs over all i, j — the on-site term is
            # already in there. Adding it again is a double count.
            if not self.include_self_interaction:
                energy = energy - self._self_energy(feats, batch, num_graphs)
            return energy
        capacity = self.max_atoms_per_graph
        if self.l_max == 0:
            energy = self._pair_energy(feats[:, 0], positions, batch, num_graphs, capacity)
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
            energy = self._pair_energy(
                charges, extended_pos, extended_batch, num_graphs, 4 * capacity, orig
            )
        if self.include_self_interaction:
            energy = energy + self._self_energy(feats, batch, num_graphs)
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
