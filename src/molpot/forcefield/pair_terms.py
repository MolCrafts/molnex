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
padding rows past their live prefix. In :class:`LJCutPotential` and
:class:`CoulombOpenPotential` those rows, and live rows beyond the cutoff, are
removed by multiplying their contribution by zero, never by ``tensor[mask]``
indexing — the output shape of a boolean index depends on the data, which
forces a dynamo graph break on the force path (a point where ``torch.compile``
must fall back to the eager interpreter; ``.claude/notes/notes.md`` topic
``compile-no-graph-breaks``). Those two therefore take an ``edge_mask``.

:class:`CoulombPeriodicPotential` is the one **documented exception**: it
still selects its real-space rows with ``tensor[mask]``, so it is eager-only
and takes no ``edge_mask`` — its caller folds liveness into the ``scales`` it
passes. Making the periodic term capacity-shaped belongs to the periodic
Coulomb step (``composable-a0-03``).
"""

from __future__ import annotations

from typing import Literal

import torch

from molpot.potentials.elec.calculators import EwaldCalculator, P3MCalculator, PMECalculator
from molpot.potentials.elec.calculators.calculator import Calculator
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
    ``r >= cutoff`` are zeroed (the caller may also pre-filter the list). Both
    entry points return the repulsive and attractive halves separately, so
    :class:`~molpot.forcefield.potentials.Potentials` can label one ``local``
    and the other ``disp`` (S3) without evaluating a second energy path.

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
            the cutoff, where ``S`` is the switching factor of :meth:`_switch`
            (identically ``1`` without a ``switch_start``) and ``scale`` the
            row's entry of ``scales``. Each is ``(B,)`` in kcal/mol when
            ``edge_batch`` is given and scalar ``()`` otherwise. Repulsion is
            non-negative, dispersion non-positive, and their sum is the
            (scaled, switched) Lennard-Jones energy.
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
        positions: torch.Tensor,
        edge_index: torch.Tensor,
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
        ``dE/dr = 24 ε (σ/r)⁶ (1 - 2 (σ/r)⁶) / r`` (kcal mol⁻¹ Å⁻¹). Write
        ``g = scale · live · dE/dr`` for that derivative after the pair's scale
        factor and its liveness weight (``1`` on a contributing row, ``0`` on a
        dead or out-of-range one),
        and ``r̂ = (pos_j - pos_i) / r`` for the unit vector from the first atom
        of the pair to the second. Because ``∂r/∂pos_i = -r̂``, the chain rule
        gives ``F_i = -∂E/∂pos_i = +g r̂`` and, by Newton's third law,
        ``F_j = -g r̂``. Evaluating this closed form needs no ``requires_grad``
        position leaf and no autograd backward. The forces derive from the
        **total** ``repulsion + dispersion``, so splitting the reported energy
        into halves leaves them unchanged.

        Inside a switching window the energy is ``S(r) · E_LJ(r)``, so the
        product rule puts *two* terms in the radial derivative,
        ``dE/dr = S · dE_LJ/dr + E_LJ · dS/dr``. Both are here. Keeping only
        the first would taper the energy while leaving the force untapered —
        a non-conservative model that heats an MD run instead of merely
        truncating it.

        Args:
            positions: Atom positions ``(N, 3)`` in Å.
            edge_index: Half-list pairs ``(E, 2)``, atom indices into the ``N``
                atoms; column 0 is the ``i`` above, column 1 the ``j``.
            distances: Pair separations ``r`` ``(E,)`` in Å.
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
            ``(N, 3)`` in kcal mol⁻¹ Å⁻¹.
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
        rhat = (positions[target] - positions[source]) / dist.unsqueeze(-1)
        f_i = g.unsqueeze(-1) * rhat
        forces = positions.new_zeros((positions.shape[0], 3))
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

    Args:
        smearing: Gaussian splitting width in Å. Larger values move work from
            the real-space sum into the reciprocal one.
        calculator: Which reciprocal-space algorithm runs — ``"ewald"``,
            ``"pme"`` or ``"p3m"``.
        lr_wavelength: Shortest reciprocal-space wavelength resolved by the
            ``"ewald"`` sum, in Å.
        mesh_spacing: Grid spacing of the ``"pme"`` / ``"p3m"`` mesh, in Å.
        prefactor: Coulomb constant ``k_e``; defaults to ``kcalmol_A``
            ``≈ 332.0637`` kcal Å mol⁻¹ e⁻².
        real_space_cutoff: Cutoff of the real-space part in Å. Defaults to
            ``3 × smearing``, beyond which the short-ranged remainder is
            negligible.

    Raises:
        ValueError: ``smearing`` is not positive, or ``calculator`` is not one
            of ``ewald`` / ``pme`` / ``p3m``.
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
    ) -> None:
        if smearing <= 0:
            raise ValueError("smearing must be positive")
        self.smearing = float(smearing)
        self.calculator_name = calculator
        self.lr_wavelength = float(lr_wavelength)
        self.mesh_spacing = float(mesh_spacing)
        # real-space neighbor cutoff; default ~3*smearing when unset
        self.real_space_cutoff = (
            float(real_space_cutoff) if real_space_cutoff is not None else 3.0 * self.smearing
        )
        self.potential = CoulombPotential(smearing=self.smearing, prefactor=prefactor)
        self._calc: Calculator = self._build_calculator()

    def _build_calculator(self) -> Calculator:
        name = self.calculator_name
        if name == "ewald":
            return EwaldCalculator(self.potential, lr_wavelength=self.lr_wavelength)
        if name == "pme":
            return PMECalculator(self.potential, mesh_spacing=self.mesh_spacing)
        if name == "p3m":
            return P3MCalculator(self.potential, mesh_spacing=self.mesh_spacing)
        raise ValueError(f"unknown electrostatics calculator {name!r}; expected ewald|pme|p3m")

    def energy(
        self,
        *,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Total electrostatic energy (real + reciprocal).

        Fully excluded pairs (``scale == 0``) are dropped from the real-space
        list before the sum, and the remaining scale factors are applied as a
        correction after it: the calculator's unscaled short-range pair
        energies are subtracted and the scaled ones added back. The
        reciprocal-space part stays unscaled.

        Unlike :meth:`LJCutPotential.energy` and
        :meth:`CoulombOpenPotential.energy`, this method takes no ``edge_mask``
        and selects rows with ``tensor[mask]``, so it is **eager-only**: a
        boolean selection has a data-dependent output shape and breaks a
        compiled graph. A caller handing it a fixed-capacity list must
        therefore zero the ``scales`` of the dead rows first (multiply by the
        liveness mask), which is how the ``scale > 0`` selection below drops
        them. Making this term capacity-shaped belongs to
        ``composable-a0-03``.

        Args:
            positions: Atom positions ``(N, 3)`` in Å.
            charges: Per-atom partial charges ``(N,)`` in elementary charges
                ``e``.
            cell: Lattice vectors ``(3, 3)`` in Å, rows being the vectors.
            edge_index: Half-list pairs ``(E, 2)`` within the real-space cutoff.
            distances: Pair separations ``(E,)`` in Å.
            scales: Per-pair scale factors ``(E,)``, dimensionless, with the
                liveness of a fixed-capacity list already folded in — a dead
                row must arrive at ``0`` so the ``scale > 0`` selection drops
                it.

        Returns:
            Scalar total electrostatic energy ``()``, in kcal/mol for the
            default prefactor.
        """
        # Drop fully excluded pairs from real-space list
        keep = scales > 0
        edge_index = edge_index[keep]
        distances = distances[keep]
        scales = scales[keep]

        # Restrict to real-space cutoff on the Potential
        within = distances < self.real_space_cutoff
        edge_index = edge_index[within]
        distances = distances[within]
        scales = scales[within]

        q = charges
        charges_c = q.unsqueeze(-1)
        cell = cell.to(dtype=positions.dtype, device=positions.device)
        periodic = torch.tensor([True, True, True], device=positions.device)

        if edge_index.numel() == 0:
            # k-space only path: pass empty neighbors
            empty_idx = torch.zeros(0, 2, dtype=torch.long, device=positions.device)
            empty_d = torch.zeros(0, dtype=positions.dtype, device=positions.device)
            phi = self._calc(
                charges=charges_c,
                cell=cell,
                positions=positions,
                neighbor_indices=empty_idx,
                neighbor_distances=empty_d,
                periodic=periodic,
            )
            return (charges_c * phi).sum()

        phi = self._calc(
            charges=charges_c,
            cell=cell,
            positions=positions,
            neighbor_indices=edge_index,
            neighbor_distances=distances,
            periodic=periodic,
        )
        e_full = (charges_c * phi).sum()
        if torch.allclose(scales, torch.ones_like(scales)):
            return e_full
        # Adjust SR for 1–4 scales (k-space remains unscaled)
        v_sr = self.potential.sr_from_dist(distances)
        e_sr_calc = (q[edge_index[:, 0]] * q[edge_index[:, 1]] * v_sr).sum()
        e_sr_scaled = (scales * q[edge_index[:, 0]] * q[edge_index[:, 1]] * v_sr).sum()
        return e_full - e_sr_calc + e_sr_scaled
