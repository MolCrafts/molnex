"""Pair Potential terms owned by :class:`Potentials` assembly.

A *pair term* is an interaction between two atoms that is a function of their
separation alone — Lennard-Jones and Coulomb here — evaluated over a neighbour
list rather than over a fixed bonded topology.

Cutoff and kernel parameters live **on the Potential**, not on Potentials.

Energies and lengths follow the owning ForceField's unit set; with the default
``units="real"`` that is kcal/mol for energies, Å for distances, elementary
charges ``e`` for charges, and kcal mol⁻¹ Å⁻¹ for forces. The spec labels
``S3`` (the Lennard-Jones repulsion/dispersion split) and the rest of the
contract vocabulary are defined in ``.claude/notes/composable-potentials.md``
and :mod:`molpot.contract`.

Neighbour lists reaching these terms are **fixed-capacity**: they carry dead
padding rows past their live prefix. Those rows, and live rows beyond the
cutoff, are removed by multiplying their contribution by zero, never by
``tensor[mask]`` indexing — the output shape of a boolean index depends on the
data, which forces a dynamo graph break on the force path (a point where
``torch.compile`` must fall back to the eager interpreter;
``.claude/notes/notes.md`` topic ``compile-no-graph-breaks``). Every term here
therefore takes an ``edge_mask``: :class:`LJCutPotential` and
:class:`CoulombOpenPotential` turn it into a multiplicative weight,
:class:`CoulombPeriodicPotential` hands it to its calculator as a
``pair_mask``.
"""

from __future__ import annotations

from typing import Literal

import torch

from molpot.potentials.elec.calculators import EwaldCalculator, P3MCalculator, PMECalculator
from molpot.potentials.elec.calculators.calculator import Calculator
from molpot.potentials.elec.lib.kvectors import (
    generate_kvectors_for_ewald,
    get_ns_ewald,
    get_ns_mesh,
)
from molpot.potentials.elec.potentials import CoulombPotential
from molpot.potentials.elec.prefactors import kcalmol_A
from molpot.potentials.vdw.lj126 import lj126_pair_energy_split, lorentz_berthelot

__all__ = [
    "LJCutPotential",
    "CoulombOpenPotential",
    "CoulombPeriodicPotential",
]


def _live_pairs(
    distances: torch.Tensor,
    cutoff: float | None,
    edge_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Liveness of a fixed-capacity pair list, as a mask and as a weight.

    Belongs to no pair Potential in particular — every term over a neighbour
    list needs the same two rows-that-count tensors, and the rule (dead row or
    beyond the cutoff contributes exactly zero) is the list's, not the
    kernel's.

    Args:
        distances: Pair separations ``r`` ``(E,)`` in Å, padding rows included.
        cutoff: Real-space cutoff ``r_c`` in Å; ``None`` keeps every pair the
            list produced.
        edge_mask: Live-row mask ``(E,)`` of a fixed-capacity list; ``None``
            treats every row as live.

    Returns:
        ``(live, weight)`` — the boolean mask ``(E,)`` and the same mask cast
        to ``distances.dtype``, i.e. ``1`` on a contributing pair and ``0`` on
        a dead or out-of-range one. Multiplying by ``weight`` is what replaces
        ``distances[live]``: every intermediate keeps its full ``(E,)`` shape,
        so the trace has no data-dependent shape and no graph break.
    """
    if cutoff is None:
        live = edge_mask if edge_mask is not None else torch.ones_like(distances, dtype=torch.bool)
    else:
        live = distances < cutoff
        if edge_mask is not None:
            live = live & edge_mask
    return live, live.to(distances.dtype)


class LJCutPotential:
    """Lennard-Jones 12-6 with explicit real-space cutoff.

    The Lennard-Jones 12-6 pair energy ``4 ε [(σ/r)¹² − (σ/r)⁶]`` models two
    neutral atoms at separation ``r``: a steep repulsive wall where their
    electron clouds overlap, and a weak attractive dispersion tail. ``ε``
    (kcal/mol) is the depth of the resulting energy minimum and ``σ`` (Å) the
    separation at which the energy crosses zero.

    Wraps :func:`lj126_pair_energy_split` + Lorentz–Berthelot mixing (per-atom
    ``ε`` / ``σ`` combined into per-pair values, see
    :func:`~molpot.potentials.vdw.lj126.lorentz_berthelot`). Pairs with
    ``r >= cutoff`` are zeroed (the caller may also pre-filter the list).

    There are two entry points. :meth:`energy` prices a pair list; the
    **analytic-force branch** :meth:`energy_forces` prices the same list and
    additionally returns ``F = -∂E/∂pos`` from the closed-form radial
    derivative, so a caller (:meth:`Potentials.calc_energy_forces
    <molpot.forcefield.potentials.Potentials.calc_energy_forces>`) gets forces
    with no ``requires_grad`` position leaf and no autograd backward. It takes
    the pair **vectors** as well as the distances, because a direction and a
    length that do not come from the same vector are not each other's
    gradient under periodic boundaries — see :meth:`energy_forces`.

    Both entry points return the repulsive and attractive halves separately,
    so :class:`~molpot.forcefield.potentials.Potentials` can label one
    ``local`` and the other ``disp`` (S3) without evaluating a second energy
    path.

    With ``switch_start`` set, the energy is tapered to zero over
    ``[r_s, r_c]`` by the quintic switch ``S`` of :meth:`_switch` instead of
    being truncated at ``r_c``. Truncation leaves a step in the force at the
    cutoff — a neighbour crossing it changes the force discontinuously, which
    an MD integrator turns into a steady energy drift. ``S(r_c) = 0`` **and**
    ``S'(r_c) = 0`` remove both halves of that step.

    Args:
        cutoff: Real-space cutoff radius ``r_c`` in Å. Pairs at or beyond it
            contribute nothing.
        switch_start: Start ``r_s`` of the switching window in Å, with
            ``0 < r_s < r_c``. ``None`` (the default) means no window: ``S ≡ 1``
            up to ``r_c`` and a hard truncation there.

    Attributes:
        cutoff: The cutoff radius in Å, as a Python ``float``.
        switch_start: The switching-window start in Å, or ``None``.

    Raises:
        ValueError: ``cutoff`` is not positive, or ``switch_start`` is given
            and does not satisfy ``0 < switch_start < cutoff``.
    """

    def __init__(self, cutoff: float = 10.0, switch_start: float | None = None) -> None:
        if cutoff <= 0:
            raise ValueError(f"LJ cutoff must be positive, got {cutoff}")
        if switch_start is not None and not 0.0 < switch_start < cutoff:
            raise ValueError(
                f"LJ switch_start must satisfy 0 < switch_start < cutoff={cutoff}, "
                f"got {switch_start}"
            )
        self.cutoff = float(cutoff)
        self.switch_start = None if switch_start is None else float(switch_start)

    def _switch(self, distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quintic switching factor ``S(r)`` and its radial derivative.

        ``S(x) = 1 − x³(10 − 15x + 6x²)`` on
        ``x = clamp((r − r_s)/(r_c − r_s), 0, 1)`` — OpenMM's
        ``useSwitchingFunction`` taper, algebraically the LAMMPS ``lj/mdf``
        form ``(1 − x)³(1 + 3x + 6x²)``. It is the lowest-order polynomial
        with ``S(r_s) = 1``, ``S(r_c) = 0`` and ``S'(r_s) = S'(r_c) = 0``: the
        energy enters the window unchanged, reaches zero at the cutoff, and
        gets there with zero slope — energy *and* force are continuous at
        ``r_c``.
        Differentiating gives ``dS/dr = −30 x²(1 − x)² / (r_c − r_s)`` (Å⁻¹),
        which vanishes at both ends with the clamped ``x``.

        Args:
            distances: Pair separations ``r`` ``(E,)`` in Å.

        Returns:
            ``(S, dS/dr)`` — dimensionless ``(E,)`` and ``(E,)`` in Å⁻¹, or the
            broadcastable scalars ``1`` and ``0`` when ``switch_start`` is
            ``None``. Both come from one call so the energy and the analytic
            force can never taper by different amounts; :meth:`energy` uses
            only the first.
        """
        if self.switch_start is None:
            return (
                torch.ones((), dtype=distances.dtype, device=distances.device),
                torch.zeros((), dtype=distances.dtype, device=distances.device),
            )
        width = self.cutoff - self.switch_start
        x = ((distances - self.switch_start) / width).clamp(0.0, 1.0)
        switch = 1.0 - x.pow(3) * (10.0 - 15.0 * x + 6.0 * x.pow(2))
        return switch, -30.0 * x.pow(2) * (1.0 - x).pow(2) / width

    def energy(
        self,
        *,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        scales: torch.Tensor,
        edge_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Repulsive and dispersive LJ energy for a half neighbor list.

        A *half* neighbour list holds each unordered pair once (``i`` with
        ``j``, not also ``j`` with ``i``), so summing over its entries counts
        every interaction exactly once — no factor of ``½`` is applied here.

        Args:
            edge_index: Half-list pairs ``(E, 2)``, atom indices into the ``N``
                atoms.
            distances: Pair separations ``r`` ``(E,)`` in Å.
            epsilon: Per-atom well depth ``ε`` ``(N,)`` in kcal/mol.
            sigma: Per-atom zero-crossing distance ``σ`` ``(N,)`` in Å.
            scales: Per-pair scale factors ``(E,)``, dimensionless. Bonded
                neighbours are attenuated or removed by the force field's
                exclusion policy — ``0`` drops the pair entirely, ``1`` keeps it
                in full, and an intermediate value is the usual 1-4 scaling
                applied to atoms three bonds apart.
            edge_batch: Index of the molecule (graph) each edge belongs to
                ``(E,)``. ``None`` sums to a scalar. Padding rows must still
                carry an in-range graph index; their contribution is zero.
            num_graphs: Number of molecules ``B``; required when ``edge_batch``
                is given.
            edge_mask: Live-row mask ``(E,)`` of a fixed-capacity neighbour
                list. Dead rows contribute **exactly** zero to every returned
                entry — they are multiplied out, not sliced out, and their
                distance is replaced by ``cutoff`` first so a coincident
                padding pair (``r = 0``) cannot produce ``0 · inf``. ``None``
                treats every row as live.

        Returns:
            ``(repulsion, dispersion)`` — ``Σ scale · S(r) · 4 ε (σ/r)¹²`` and
            ``−Σ scale · S(r) · 4 ε (σ/r)⁶`` accumulated over the pairs within
            the cutoff, with ``ε`` / ``σ`` the per-pair values Lorentz–Berthelot
            mixing makes of the per-atom arguments, ``S`` the switching factor
            of :meth:`_switch` (identically ``1`` without a ``switch_start``)
            and ``scale`` the row's entry of ``scales``. Each is ``(B,)`` in
            kcal/mol when ``edge_batch`` is given, scalar ``()`` otherwise.
            Repulsion is non-negative, dispersion non-positive, and their sum
            is the (scaled, switched) Lennard-Jones energy.
        """
        live, weight = _live_pairs(distances, self.cutoff, edge_mask)
        dist = torch.where(live, distances, distances.new_full((), self.cutoff))
        mixed = lorentz_berthelot({"epsilon": epsilon, "sigma": sigma}, edge_index)
        rep_pair, disp_pair = lj126_pair_energy_split(dist, mixed["epsilon_ij"], mixed["sigma_ij"])
        switch, _ = self._switch(dist)
        sw = switch * (scales * weight)
        rep_pair = rep_pair * sw
        disp_pair = disp_pair * sw
        if edge_batch is None:
            return rep_pair.sum(), disp_pair.sum()
        index = edge_batch.long()
        graphs = int(num_graphs)
        repulsion = rep_pair.new_zeros(graphs)
        dispersion = disp_pair.new_zeros(graphs)
        repulsion.index_add_(0, index, rep_pair)
        dispersion.index_add_(0, index, disp_pair)
        return repulsion, dispersion

    def energy_forces(
        self,
        *,
        n_atoms: int,
        edge_index: torch.Tensor,
        pair_vectors: torch.Tensor,
        distances: torch.Tensor,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        scales: torch.Tensor,
        edge_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """LJ energy halves and analytic forces ``F = -∂E/∂pos`` for a half list.

        Differentiating ``4 ε [(σ/r)¹² − (σ/r)⁶]`` gives the radial derivative
        ``dE/dr = 24 ε (σ/r)⁶ (1 - 2 (σ/r)⁶) / r`` (kcal mol⁻¹ Å⁻¹), where
        ``ε`` and ``σ`` are the **per-pair** values Lorentz–Berthelot mixing
        makes of the per-atom ``epsilon`` / ``sigma`` arguments. Write
        ``g = scale · live · dE/dr`` for that derivative after the pair's scale
        factor and its liveness weight (``1`` on a contributing row, ``0`` on a
        dead or out-of-range one),
        and ``r̂ = pair_vectors / r`` for the unit vector from the first atom of
        the pair to the second. Because ``∂r/∂pos_i = -r̂``, the chain rule
        gives ``F_i = -∂E/∂pos_i = +g r̂`` and, by Newton's third law,
        ``F_j = -g r̂``. Evaluating this closed form needs no ``requires_grad``
        position leaf and no autograd backward. The forces derive from the
        **total** ``repulsion + dispersion``, so splitting the reported energy
        into halves leaves them unchanged.

        Direction and distance are read from the **same** vector, which is why
        this takes ``pair_vectors`` rather than the positions. Under periodic
        boundaries a pair's separation is its *minimum image*
        ``pos_j - pos_i + shift``, and ``distances`` is the norm of exactly
        that. Rebuilding the direction here as the unwrapped
        ``pos_j - pos_i`` would pair a wrapped length with an unwrapped
        heading: every pair whose minimum image crosses the cell would get an
        ``r̂`` that is neither a unit vector nor aimed at the partner actually
        being priced, so ``F`` would stop being ``-∇E`` of the energy this
        same call returns and an MD run would pump energy. One vector in, both
        quantities out of it, is the whole guard — the caller
        (:meth:`~molpot.forcefield.Potentials._placed_pairs`) forms it once and
        hands over the norm beside it. This is measured, not hypothetical: it
        is finding **F27** in ``.claude/notes/composable-potentials.md``, where
        the earlier raw-difference direction disagreed with a finite difference
        of the (correct) energy by more than an order of magnitude on a
        256-atom argon box and made an NVE run gain 9 % of its energy in ten
        steps.

        Inside a switching window the energy is ``S(r) · E_LJ(r)``, so the
        product rule puts *two* terms in the radial derivative,
        ``dE/dr = S · dE_LJ/dr + E_LJ · dS/dr``. Both are here. Keeping only
        the first would taper the energy while leaving the force untapered —
        a non-conservative model that heats an MD run instead of merely
        truncating it.

        Args:
            n_atoms: Number of atoms ``N`` the returned force buffer spans.
                Explicit because nothing here is entitled to infer it: the
                per-atom ``epsilon`` / ``sigma`` are *parameter tables*, whose
                length matching the geometry is a convention of the assembler,
                not an invariant of this call — a table broadcast from one
                type, or one padded to a capacity, would size the buffer wrong
                and scatter the forces of a system into the shape of a
                parameter set. The caller holds ``pos`` and therefore knows
                ``N``, and the sibling force path
                ``Potentials._coulomb_energy_forces`` already takes it as an
                argument for the same reason.
            edge_index: Half-list pairs ``(E, 2)``, atom indices into the ``N``
                atoms; column 0 is the ``i`` above, column 1 the ``j``.
            pair_vectors: Separation vectors ``(E, 3)`` in Å, pointing from
                each row's ``i`` to its ``j`` — ``pos_j - pos_i`` under open
                boundaries, ``pos_j - pos_i + shift`` (the minimum image)
                under a cell. Must be the vector ``distances`` is the norm of.
            distances: Pair separations ``r`` ``(E,)`` in Å. Must be exactly
                ``|pair_vectors|`` row for row — this is the argument
                :meth:`Potentials._placed_pairs
                <molpot.forcefield.potentials.Potentials._placed_pairs>` takes
                as the norm of the vector it returns beside it, never a
                separately stored ``edges.edge_dist``.
            epsilon: Per-atom well depth ``ε`` ``(N,)`` in kcal/mol.
            sigma: Per-atom zero-crossing distance ``σ`` ``(N,)`` in Å.
            scales: Per-pair scale factors ``(E,)``, dimensionless; see
                :meth:`energy`.
            edge_batch: Index of the molecule (graph) each edge belongs to
                ``(E,)``. ``None`` sums to a scalar. Padding rows must still
                carry an in-range graph index.
            num_graphs: Number of molecules ``B``; required when ``edge_batch``
                is given.
            edge_mask: Live-row mask ``(E,)`` of a fixed-capacity neighbour
                list; see :meth:`energy`. A dead row adds exactly zero to both
                energy halves and zero force to both of its atoms, without any
                shape ever depending on how many rows are live.

        Returns:
            ``((repulsion, dispersion), forces)`` — the energy halves of
            :meth:`energy` (each ``(B,)`` or scalar ``()``, kcal/mol) and forces
            ``(N, 3)`` in kcal mol⁻¹ Å⁻¹, with ``N`` the ``n_atoms`` passed in.
        """
        live, weight = _live_pairs(distances, self.cutoff, edge_mask)
        # A dead row is evaluated at ``r_c`` rather than at its stored distance:
        # padding often stores ``r = 0`` between an atom and itself, whose
        # energy overflows to ``inf`` (and whose ``r̂`` is ``0/0``) before the
        # weight can zero it — ``0 · inf`` is ``nan``, and a ``nan`` force
        # poisons the whole scatter.
        dist = torch.where(live, distances, distances.new_full((), self.cutoff))
        mixed = lorentz_berthelot({"epsilon": epsilon, "sigma": sigma}, edge_index)
        eps = mixed["epsilon_ij"]
        sig = mixed["sigma_ij"]

        rep_lj, disp_lj = lj126_pair_energy_split(dist, eps, sig)
        switch, dswitch = self._switch(dist)
        w = scales * weight
        sw = switch * w
        rep_pair = rep_lj * sw
        disp_pair = disp_lj * sw
        if edge_batch is None:
            repulsion, dispersion = rep_pair.sum(), disp_pair.sum()
        else:
            index = edge_batch.long()
            graphs = int(num_graphs)
            repulsion = rep_pair.new_zeros(graphs)
            dispersion = disp_pair.new_zeros(graphs)
            repulsion.index_add_(0, index, rep_pair)
            dispersion.index_add_(0, index, disp_pair)

        sr6 = (sig / dist.clamp(min=1e-6)).pow(6)
        dedr = 24.0 * eps * sr6 * (1.0 - 2.0 * sr6) / dist
        g = (switch * dedr + (rep_lj + disp_lj) * dswitch) * w
        source, target = edge_index[:, 0], edge_index[:, 1]
        # Same ``dist`` as the energy above, so a dead row divides its stored
        # vector by ``r_c`` instead of by its own length: a padding self-loop
        # (zero vector) gives ``0``, a row killed by the half-list or cutoff
        # rule gives some non-unit vector. Both are finite, which is all that
        # is asked of them — ``g`` is zero on every dead row regardless.
        rhat = pair_vectors / dist.unsqueeze(-1)
        f_i = g.unsqueeze(-1) * rhat
        forces = pair_vectors.new_zeros((n_atoms, 3))
        forces.index_add_(0, source, f_i)
        forces.index_add_(0, target, -f_i)
        return (repulsion, dispersion), forces


class CoulombOpenPotential:
    """Open-boundary Coulomb using :class:`CoulombPotential`.

    "Open boundary" means no periodic cell: the molecule sits in vacuum, so the
    electrostatic energy is the plain pair sum ``Σ k_e q_i q_j / r_ij`` over the
    neighbour list, with no reciprocal-space correction.

    Args:
        cutoff: Optional real-space cutoff radius in Å. ``None`` keeps every
            pair the neighbour list produced.
        prefactor: Coulomb constant ``k_e`` converting ``e²/Å`` into the energy
            unit. Defaults to ``kcalmol_A`` ``≈ 332.0637`` kcal Å mol⁻¹ e⁻²,
            the Class-I ``real`` unit value.

    Raises:
        ValueError: ``cutoff`` is given and is not positive.
    """

    def __init__(
        self,
        *,
        cutoff: float | None = None,
        prefactor: float = kcalmol_A,
    ) -> None:
        if cutoff is not None and cutoff <= 0:
            raise ValueError(f"Coulomb cutoff must be positive, got {cutoff}")
        self.cutoff = cutoff
        self.potential = CoulombPotential(prefactor=prefactor)

    def energy(
        self,
        *,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        charges: torch.Tensor,
        scales: torch.Tensor,
        edge_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Coulomb energy of a half neighbor list under open boundaries.

        Args:
            edge_index: Half-list pairs ``(E, 2)`` — each unordered pair once.
            distances: Pair separations ``(E,)`` in Å.
            charges: Per-atom partial charges ``(N,)`` in elementary charges
                ``e``.
            scales: Per-pair scale factors ``(E,)``, dimensionless; ``0`` drops
                an excluded bonded pair, ``1`` keeps a pair in full.
            edge_batch: Index of the molecule (graph) each edge belongs to
                ``(E,)``. ``None`` sums to a scalar. Padding rows must still
                carry an in-range graph index.
            num_graphs: Number of molecules ``B``; required when ``edge_batch``
                is given.
            edge_mask: Live-row mask ``(E,)`` of a fixed-capacity neighbour
                list. Dead rows contribute **exactly** zero: they are
                multiplied out rather than sliced out, at a stand-in distance
                of 1 Å so a coincident padding pair cannot turn the ``1/r``
                kernel's ``inf`` into a ``nan``. ``None`` treats every row as
                live.

        Returns:
            ``Σ scale_ij k_e q_i q_j / r_ij`` — ``(B,)`` when ``edge_batch`` is
            given, scalar ``()`` otherwise, in kcal/mol for the default
            prefactor.
        """
        live, weight = _live_pairs(distances, self.cutoff, edge_mask)
        dist = torch.where(live, distances, distances.new_full((), 1.0))
        v = self.potential.from_dist(dist)
        qi = charges[edge_index[:, 0]]
        qj = charges[edge_index[:, 1]]
        pair_energy = scales * qi * qj * v * weight
        if edge_batch is None:
            return pair_energy.sum()
        energy = pair_energy.new_zeros(int(num_graphs))
        energy.index_add_(0, edge_batch.long(), pair_energy)
        return energy


class CoulombPeriodicPotential:
    """Periodic Coulomb: :class:`CoulombPotential` + Ewald/PME/P3M calculator.

    Under periodic boundary conditions the ``1/r`` sum over all periodic images
    converges too slowly to evaluate directly. Ewald summation splits it in two
    with a Gaussian of width ``smearing``: a short-ranged remainder summed over
    a neighbour list, and a smooth part summed in reciprocal space. PME
    (particle-mesh Ewald) and P3M evaluate that reciprocal part on a grid
    instead of over explicit ``k`` vectors, trading exactness for speed.

    The pair potential owns smearing / mesh parameters; the calculator is
    selected by algorithm name (not a product brand).

    **The reciprocal shape is frozen, the vectors are not.** How many
    k-vectors (or mesh points) the sum runs over is fixed once by
    :meth:`bind_cell` and never moves again; the vectors themselves are
    rebuilt from the live lattice on every :meth:`energy` call, so a barostat
    step sums the box it is actually in. A shape sized off the live cell would
    read three lattice norms back to the host each step — a graph break on the
    force path — and would make the sum's own strain derivative discontinuous
    wherever the grid gained a point.

    Args:
        smearing: Gaussian splitting width in Å. Larger values move work from
            the real-space sum into the reciprocal one.
        calculator: Which reciprocal-space algorithm runs — ``"ewald"`` (every
            k-vector summed explicitly, the small-box reference), ``"pme"`` or
            ``"p3m"`` (both on an FFT mesh). Constructed directly, this term
            defaults to ``"ewald"``; assembled from a ``pair:coulomb`` style it
            gets ``"pme"`` instead, which is the default
            :class:`~molpot.forcefield.potentials.Potentials` applies under a
            cell.
        lr_wavelength: Shortest reciprocal-space wavelength resolved by the
            ``"ewald"`` sum, in Å. Ignored by ``"pme"`` / ``"p3m"``.
        mesh_spacing: Grid spacing of the ``"pme"`` / ``"p3m"`` mesh, in Å.
            Ignored by ``"ewald"``.
        prefactor: Coulomb constant ``k_e``; defaults to ``kcalmol_A``
            ``≈ 332.0637`` kcal Å mol⁻¹ e⁻².
        real_space_cutoff: Cutoff of the real-space part in Å. Defaults to
            ``3 × smearing``, beyond which the short-ranged remainder is
            negligible.
        capacity_factor: Dimensionless head-room on the frozen shape: the grid
            is sized for a cell expanded isotropically by this factor, so an
            NPT run that swells by up to it never has to rebuild anything.
            The default ``1.0`` sizes for the bound box exactly — the NVT
            contract, and the smallest sum that resolves it. It is **not**
            free head-room: a larger factor freezes a larger shape, which adds
            k-vectors to the ``"ewald"`` sum and refines the ``"pme"`` /
            ``"p3m"`` mesh, so the *same* cell is summed more finely and its
            energy moves — an accuracy knob as much as a capacity one, and a
            number to hold fixed between two runs that are compared. Nothing
            bounds it per axis yet (the ``max_axis_index`` ceiling
            :class:`~molpot.potentials.gto.kspace.KSpaceGrid` carries — finding
            F19, which routes the ceiling to the k-space unification), so an
            oversized factor is discovered only as the memory it takes.

    Raises:
        ValueError: ``smearing`` is not positive, ``calculator`` is not one
            of ``ewald`` / ``pme`` / ``p3m``, or ``capacity_factor`` is below
            ``1`` (a grid coarser than the bound cell asks for would drop
            k-vectors that lie inside the cutoff).
    """

    def __init__(
        self,
        *,
        smearing: float = 5.0,
        calculator: Literal["ewald", "pme", "p3m"] = "ewald",
        lr_wavelength: float = 1.0,
        mesh_spacing: float = 1.0,
        prefactor: float = kcalmol_A,
        real_space_cutoff: float | None = None,
        capacity_factor: float = 1.0,
    ) -> None:
        if smearing <= 0:
            raise ValueError("smearing must be positive")
        if capacity_factor < 1.0:
            raise ValueError(f"capacity_factor must be >= 1, got {capacity_factor}")
        self.smearing = float(smearing)
        self.calculator_name = calculator
        self.lr_wavelength = float(lr_wavelength)
        self.mesh_spacing = float(mesh_spacing)
        self.capacity_factor = float(capacity_factor)
        # real-space neighbor cutoff; default ~3*smearing when unset
        self.real_space_cutoff = (
            float(real_space_cutoff) if real_space_cutoff is not None else 3.0 * self.smearing
        )
        self.potential = CoulombPotential(smearing=self.smearing, prefactor=prefactor)
        self._calc: Calculator = self._build_calculator()
        self._bound_ns: tuple[int, int, int] | None = None

    def _build_calculator(self) -> Calculator:
        name = self.calculator_name
        if name == "ewald":
            return EwaldCalculator(self.potential, lr_wavelength=self.lr_wavelength)
        if name == "pme":
            return PMECalculator(self.potential, mesh_spacing=self.mesh_spacing)
        if name == "p3m":
            return P3MCalculator(self.potential, mesh_spacing=self.mesh_spacing)
        raise ValueError(f"unknown electrostatics calculator {name!r}; expected ewald|pme|p3m")

    def _reciprocal_shape(self, cell: torch.Tensor) -> torch.Tensor:
        """Per-axis grid counts ``(3,)`` the lattice ``cell`` asks for.

        The rule itself lives with the k-vectors it sizes —
        :func:`~molpot.potentials.elec.lib.kvectors.get_ns_ewald` and
        :func:`~molpot.potentials.elec.lib.kvectors.get_ns_mesh`, which are
        also what an *unbound* calculator sizes itself at, so a grid frozen at
        ``capacity_factor = 1`` is exactly the grid that was there before any
        binding. This method only picks the one the named algorithm uses, and
        it stays a tensor expression so the eager half (:meth:`bind_cell`,
        which reads the result back to fix the frozen shape) and the in-graph
        half (:meth:`energy`, which compares against that shape without ever
        touching the host) can never disagree about which cells a grid serves.

        Args:
            cell: Lattice vectors ``(3, 3)`` in Å, rows being the vectors.

        Returns:
            The three counts ``(3,)`` as a long tensor on ``cell``'s device.
            :meth:`bind_cell` is the one site that reads them back as Python
            ints; :meth:`energy` compares them as tensors.
        """
        if self.calculator_name == "ewald":
            return get_ns_ewald(cell, self.lr_wavelength)
        return get_ns_mesh(cell, self.mesh_spacing)

    def bind_cell(self, cell: torch.Tensor) -> None:
        """Freeze the reciprocal shape on the first call; capacity-check on every later one.

        Implements :class:`molix.schema.CellBound`, whose four clauses — the
        idempotence, the host-work budget, the never-retain rule and the
        never-truncate rule — are stated there and have their only home there.

        What is specific to this term: the first call reads the three lattice
        norms back to the host (the one-time sizing sync clause 2 allows) and
        sizes the grid for a cell expanded by ``capacity_factor``. Every later
        call only re-checks that the lattice it is handed still fits, and pays
        exactly **one** scalar host read to do it — the per-axis comparison
        stays the tensor expression :meth:`energy` uses, and only the single
        ``bool`` of its reduction crosses to the host, which is the whole
        budget clause 2 allows a later bind. The shape is never resized,
        neither up (that is what ``capacity_factor`` was for) nor down (a
        contracted cell is served by a finer grid than it asked for, at no cost
        but time). Only the integer counts are kept; :meth:`energy` re-reads
        the live ``(3, 3)`` tensor it is handed and re-checks the same
        predicate in-graph, with no host read at all.

        Args:
            cell: Lattice vectors ``(3, 3)`` in Å, rows being the vectors. A
                batched ``(1, 3, 3)`` cell is sliced by the caller.

        Raises:
            ValueError: ``cell`` is not ``(3, 3)``.
            RuntimeError: The cell needs a finer grid than the frozen shape
                provides — raise ``capacity_factor`` (or build a new term) and
                bind again.
        """
        if cell.shape != (3, 3):
            raise ValueError(f"cell must be (3, 3), got {tuple(cell.shape)}")
        # ``detach``: a virial pass binds with a strain-tracking cell, and the
        # grid shape is metadata about the lattice, never part of the graph.
        lattice = cell.detach()
        frozen = self._bound_ns
        if frozen is None:
            expanded = self._reciprocal_shape(lattice * self.capacity_factor)
            self._bound_ns = (int(expanded[0]), int(expanded[1]), int(expanded[2]))
            if isinstance(self._calc, PMECalculator):
                self._calc.bind(self._bound_ns)
            return
        needed = self._reciprocal_shape(lattice)
        # One scalar read, not three: the per-axis comparison is the same
        # tensor expression the in-graph guard at the end of ``energy`` uses,
        # and only the single ``bool`` of its reduction crosses to the host.
        # A later bind fires once per barostat update, so three ``int()``
        # syncs there were three more than the budget.
        overflow = (needed[0] > frozen[0]) | (needed[1] > frozen[1]) | (needed[2] > frozen[2])
        if bool(overflow):
            # Already raising: the detail may sync, in one read rather than
            # per axis, to name the shape asked for and the axes over it.
            asked = tuple(needed.tolist())
            over = [axis for axis in range(3) if asked[axis] > frozen[axis]]
            raise RuntimeError(
                f"periodic Coulomb k-space overflow: this cell asks for {asked} grid points "
                f"but the shape frozen at bind is {frozen} (axes {over}). Raise "
                f"capacity_factor (currently {self.capacity_factor}) and bind again, or "
                f"build a new CoulombPeriodicPotential."
            )

    def energy(
        self,
        *,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        edge_mask: torch.Tensor,
        excluded_pairs: torch.Tensor,
        excluded_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Total electrostatic energy — real + reciprocal, less the exclusion correction.

        **Symbols.** ``r`` is a pair separation in Å, ``q_i`` a partial charge
        in elementary charges ``e``, ``k_e`` the Coulomb constant
        (``prefactor``, kcal Å mol⁻¹ e⁻²) and ``σ`` the Gaussian splitting
        width ``smearing`` in Å — *not* the Lennard-Jones diameter that
        :class:`LJCutPotential` calls ``σ``. ``erf`` is the error function and
        ``erfc = 1 − erf`` its complement; ``erfc`` decays to zero over a few
        ``σ``, which is what makes the first half of the split below
        short-ranged. ``r_c`` is ``real_space_cutoff``, and the subscript ``p``
        indexes the rows of ``excluded_pairs`` — one scaled pair each, with
        endpoints ``i``, ``j``, separation ``r_p`` and scale ``s_p``.

        **The identity.** The Ewald split ``1/r = erfc(r/σ√2)/r + erf(r/σ√2)/r``
        sends the second half to the reciprocal sum, which runs over *every*
        pair of the cell: an atom's Gaussian charge cloud is felt by its bonded
        partner exactly as by any other neighbour. Removing a bonded pair — or
        attenuating it to its declared scale ``s`` — therefore costs the
        **whole** Coulomb term, not only the real-space ``erfc`` share::

            E(excluded) = E(unexcluded) − Σ_p (1 − s_p) k_e q_i q_j / r_p

        for a pair inside the real-space cutoff, and only its reciprocal share
        ``(1 − s_p) k_e q_i q_j erf(r_p/σ√2) / r_p`` for one beyond it: past
        ``r_c`` the real-space sum never added the ``erfc`` half, so subtracting
        it would over-correct. That is exactly LAMMPS ``special_bonds`` under
        ``coul/long`` — which subtracts the full ``1/r`` of an excluded pair
        from the k-space-inclusive total — and OpenMM's ``NonbondedForce``
        exceptions.

        The correction runs over the **topology's** pair table, never over the
        neighbour list. An excluded pair that has drifted past ``r_c`` is absent
        from the list yet still fully present in the reciprocal sum, so a
        list-driven correction would silently stop correcting it; and the pair
        count is a property of the molecule, so it is fixed for the whole run.

        **Boundary — one image per excluded pair.** Each row of
        ``excluded_pairs`` is corrected at its *minimum image*: the single
        nearest periodic copy of the partner, taken from the live lattice
        below. LAMMPS ``special_bonds`` under ``coul/long`` instead excludes
        **every** image of the pair, keyed by atom tag. The two agree as long
        as the shortest box dimension stays at or above ``2 r_c``, which is
        where the real-space sum can hold at most one image of a given pair.
        In a smaller box the same excluded pair enters the real-space sum at
        more than one image and only the nearest is corrected here — that
        regime is outside this term's contract (finding F15 in
        ``.claude/notes/composable-potentials.md``), not a case it detects.

        Every tensor keeps the shape it arrived with. Dead padding rows and rows
        beyond ``r_c`` are removed by the boolean ``pair_mask`` handed to the
        calculator, never by ``tensor[mask]`` — a boolean selection's output
        shape depends on the data, which breaks a compiled graph. Pairs at
        ``s = 0`` stay in the real-space sum; the correction below is what takes
        them out again.

        **The reciprocal sum.** Its k-vectors are regenerated here from the
        ``cell`` that arrived, at the shape :meth:`bind_cell` froze — so the
        lattice may move under a barostat while the number of terms, and hence
        the traced graph, does not. An unbound term binds itself on this first
        call, which is a host read and therefore an eager convenience: bind
        before entering a compiled region. A cell needing a finer grid than the
        frozen one returns ``NaN`` instead of a truncated sum.

        Args:
            positions: Atom positions ``(N, 3)`` in Å.
            charges: Per-atom partial charges ``(N,)`` in elementary charges
                ``e``.
            cell: Lattice vectors ``(3, 3)`` in Å, rows being the vectors.
            edge_index: Half-list pairs ``(E, 2)`` at the capacity the list was
                written at — dead padding rows included.
            distances: Pair separations ``(E,)`` in Å, padding rows included.
            edge_mask: Live-row mask ``(E,)`` of that capacity-shaped list.
                Combined with ``r < r_c`` into the calculator's ``pair_mask``.
            excluded_pairs: Endpoints ``(P, 2)`` of every scaled pair of the
                topology — :attr:`ExclusionTable.pairs
                <molpot.forcefield.exclusions.ExclusionTable.pairs>`. A topology
                that excludes nothing passes ``(0, 2)``, which sums to zero: a
                static no-op, not a Python branch.
            excluded_scales: The matching factors ``s_p`` ``(P,)``,
                dimensionless — ``0`` for a fully excluded 1-2 / 1-3 pair
                (two atoms sharing a bond, or the two ends of a bond angle),
                and for a 1-4 pair (the ends of a torsion) the force field's
                own 1-4 electrostatic scale: ``5/6`` with the AMBER default of
                :class:`~molpot.ir.NonbondedScaling`, ``1/2`` under
                :class:`~molpot.forcefield.container.OPLSAAForceField`.

        Returns:
            Scalar total electrostatic energy ``()``, in kcal/mol for the
            default prefactor — or ``NaN`` when ``cell`` has outgrown the
            frozen reciprocal shape. One cell, one number: the reciprocal sum
            takes no ``edge_batch``, and
            :meth:`~molpot.forcefield.potentials.Potentials.calc_energy`
            refuses a batch of more than one graph rather than charging each of
            them this box's energy.

        Raises:
            ValueError: On the lazy first bind only, when ``cell`` is not
                ``(3, 3)``.
        """
        pair_mask, _ = _live_pairs(distances, self.real_space_cutoff, edge_mask)
        cell = cell.to(dtype=positions.dtype, device=positions.device)
        frozen = self._bound_ns
        if frozen is None:
            # Eager convenience for a caller that never bound one: the shape
            # has to be frozen off *some* lattice and the first one seen is the
            # only candidate. It reads the host, so bind before compiling.
            self.bind_cell(cell)
            frozen = self._bound_ns
            assert frozen is not None  # bind_cell either sets it or raises
        charge_column = charges.unsqueeze(-1)
        phi = self._calc(
            charges=charge_column,
            cell=cell,
            positions=positions,
            neighbor_indices=edge_index,
            neighbor_distances=distances,
            periodic=torch.ones(3, dtype=torch.bool, device=positions.device),
            pair_mask=pair_mask,
            kvectors=(
                generate_kvectors_for_ewald(cell=cell, ns=frozen)
                if self.calculator_name == "ewald"
                else None
            ),
        )
        total = (charge_column * phi).sum()

        # Minimum image of every scaled pair, from the *live* cell. The pairs
        # are intramolecular, so the nearest image is the one the neighbour list
        # would have enumerated. Taken by fractional round-trip rather than from
        # a stored shift: ``round`` has zero derivative, so ``r_p`` reduces to
        # the plain displacement under ``∂/∂pos`` and still carries the lattice
        # under ``∂/∂ε`` — which is what keeps this correction inside both the
        # force and the virial.
        source, target = excluded_pairs[:, 0], excluded_pairs[:, 1]
        # ``inv_ex`` on CUDA — the same convention as
        # :func:`~molpot.potentials.elec.lib.kvectors._generate_kvectors`:
        # ``inv`` checks its info tensor on the host, which is a sync on the
        # force path; ``inv_ex`` leaves the check to the caller (a singular
        # cell is not a lattice, and the NaN it produces poisons the energy).
        inverse_cell = torch.linalg.inv_ex(cell)[0] if cell.is_cuda else torch.linalg.inv(cell)
        fractional = (positions[target] - positions[source]) @ inverse_cell
        imaged = (fractional - fractional.round()) @ cell
        # Clamped before the sqrt: ``norm``'s gradient at a zero vector is NaN,
        # and no later multiply can undo one.
        separation = (imaged * imaged).sum(-1).clamp_min(1e-24).sqrt()
        # A weight, not a selection — how many pairs are inside the cutoff is
        # data, and indexing by it would be a data-dependent shape.
        short_range = torch.where(
            separation < self.real_space_cutoff,
            self.potential.sr_from_dist(separation),
            separation.new_zeros(()),
        )
        removed = (
            (1.0 - excluded_scales)
            * charges[source]
            * charges[target]
            * (self.potential.lr_from_dist(separation) + short_range)
        )
        energy = total - removed.sum()
        # In-graph half of the capacity guard, on the same predicate
        # ``bind_cell`` raises on. Three scalar comparisons against Python
        # ints: no host read, no branch, nothing to trace around — and an
        # over-capacity cell poisons energy, forces and virial together rather
        # than reporting the truncated sum of a box this grid cannot resolve.
        needed = self._reciprocal_shape(cell)
        overflow = (needed[0] > frozen[0]) | (needed[1] > frozen[1]) | (needed[2] > frozen[2])
        return torch.where(overflow, energy.new_full((), float("nan")), energy)
