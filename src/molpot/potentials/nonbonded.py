"""Non-bonded pairwise potentials: repulsion, dispersion, charge transfer.

A *non-bonded* term prices the interaction between two atoms that are near each
other in space rather than joined by a chemical bond. The three here are the
pieces a physics-aware model splits the short-range interaction into: the steep
**repulsion** felt once two electron clouds overlap, the weak attractive
**dispersion** (London) tail from fluctuating dipoles, and a **charge-transfer**
term for the small amount of electron density that leaks from one atom to its
neighbour at close range.

Each potential takes *per-pair* parameters and per-pair distances and returns
energy. Turning a model's *per-atom* parameters into per-pair ones is a
**combining rule** (equivalently a mixing rule), carried by every potential here
as its ``mixing_fn`` and applied by the caller —
:class:`~molpot.composition.PotentialComposer` — one step before ``forward``.
The three read **different keys**, so their rules are not interchangeable:
:func:`repulsion_mixing`, :func:`dispersion_mixing` and :func:`ct_mixing` are
the matching defaults. They are ordinary callables and deliberately *not*
members of :class:`~molpot.potentials.vdw.lj126.MixingRule`: that enum is a
closed vocabulary because a Lennard-Jones model's rule is recorded as part of
its identity, and it reads ``"epsilon"`` / ``"sigma"``, which are not keys any
potential in this module consumes. Pairing one with a term here raises rather
than silently mixing the wrong table — a ``KeyError`` when the head emits no
``"epsilon"``, and a ``TypeError`` on the unexpected ``epsilon_ij`` keyword when
it does.

Units
-----
Unlike the Class-I Lennard-Jones path, which is pinned to kcal/mol and Å, these
parameters are **learned**: they come from
:class:`~molpot.composition.RepulsionParameterHead`,
:class:`~molpot.composition.ChargeTransferParameterHead` and
:class:`~molpot.composition.TSScalingHead`, whose values inherit whatever unit
system the training targets and the caller-registered free-atom reference
buffers (``c6_free``, ``r_star_free``) are expressed in. This module therefore
fixes no absolute units, and a docstring here that named kcal/mol would be
inventing one.

What it does fix is the *relative* units, which is the part worth checking
before trusting a fitted model:

* every length — ``distance``, ``r_star``, ``s_rep`` — is in one and the same
  length unit;
* ``lam_rep`` and ``lam_ct`` are dimensionless, because each sits inside an
  exponential, whose argument must be;
* ``eps_rep`` is an energy, ``c6`` is energy × length⁶, and ``eps_ct`` is
  energy × length⁴ — each is whatever makes its own term come out an energy;
* ``s_disp`` is dimensionless, while ``s_rep`` is a length. The wall term built
  from ``s_rep`` is the one dimensional loose end in this module — it is a pure
  ratio, so it is an energy only by convention; see :class:`RepulsionExp6`.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from molpot.potentials.mixing import geometric_arithmetic_mixing

# ---------------------------------------------------------------------------
# Default mixing functions
# ---------------------------------------------------------------------------


def repulsion_mixing(
    atom_params: dict[str, torch.Tensor],
    pair_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Combining rule for :class:`RepulsionExp6`: per-atom → per-pair.

    The well depth ``eps_rep`` and the exponent ``lam_rep`` are combined as
    geometric means ``sqrt(p_i p_j)``, the radius ``r_star`` as the arithmetic
    mean ``½(p_i + p_j)`` — energy-like quantities multiply, size-like ones add,
    which is the same split Lorentz–Berthelot makes for Lennard-Jones.

    Args:
        atom_params: Per-atom parameters; must contain ``"eps_rep"`` ``(N,)``
            (an energy), ``"lam_rep"`` ``(N,)`` (dimensionless) and
            ``"r_star"`` ``(N,)`` (a length). Units are the model's own — see
            the module docstring. Extra keys are ignored, so one head may serve
            several potentials.
        pair_index: Edge indices ``(E, 2)``. Both means are symmetric in the two
            columns, so edge direction does not matter.

    Returns:
        Dict with ``"eps_rep_ij"``, ``"lam_rep_ij"`` and ``"r_star_ij"``, each
        ``(E,)`` in the unit of the per-atom entry it came from — exactly the
        keyword arguments :meth:`RepulsionExp6.forward` expects.

    Raises:
        KeyError: A required key is missing from ``atom_params`` — typically a
            rule paired with the wrong potential or head.
    """
    return geometric_arithmetic_mixing(
        atom_params,
        pair_index,
        geometric_keys=["eps_rep", "lam_rep"],
        arithmetic_keys=["r_star"],
    )


def dispersion_mixing(
    atom_params: dict[str, torch.Tensor],
    pair_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Combining rule for :class:`DispersionC6`: per-atom → per-pair.

    The dispersion coefficient ``c6`` is combined as the geometric mean
    ``sqrt(C6_i C6_j)`` and the radius ``r_star`` as the arithmetic mean
    ``½(r*_i + r*_j)``. The geometric mean for ``C6`` is the standard
    approximation to the London formula, which makes the pair coefficient
    proportional to the product of the two atoms' polarizabilities.

    Args:
        atom_params: Per-atom parameters; must contain ``"c6"`` ``(N,)``
            (energy × length⁶) and ``"r_star"`` ``(N,)`` (a length), in the
            model's own units — see the module docstring. A
            :class:`~molpot.composition.TSScalingHead` also emits ``"alpha"``,
            which is ignored here.
        pair_index: Edge indices ``(E, 2)``; both means are symmetric in the two
            columns.

    Returns:
        Dict with ``"c6_ij"`` and ``"r_star_ij"``, each ``(E,)`` in the unit of
        the per-atom entry it came from — the keyword arguments
        :meth:`DispersionC6.forward` expects.

    Raises:
        KeyError: A required key is missing from ``atom_params``.
    """
    return geometric_arithmetic_mixing(
        atom_params,
        pair_index,
        geometric_keys=["c6"],
        arithmetic_keys=["r_star"],
    )


def ct_mixing(
    atom_params: dict[str, torch.Tensor],
    pair_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Combining rule for :class:`ChargeTransfer`: per-atom → per-pair.

    The strength ``eps_ct`` and the exponent ``lam_ct`` are combined as
    geometric means, the radius ``r_star`` as the arithmetic mean — the same
    energy-multiplies / size-adds split as :func:`repulsion_mixing`, over the
    charge-transfer key set.

    Args:
        atom_params: Per-atom parameters; must contain ``"eps_ct"`` ``(N,)``
            (energy × length⁴, the ``r⁻⁴`` prefactor of
            :meth:`ChargeTransfer.forward`), ``"lam_ct"`` ``(N,)``
            (dimensionless) and ``"r_star"`` ``(N,)`` (a length), in the model's
            own units — see the module docstring.
        pair_index: Edge indices ``(E, 2)``; both means are symmetric in the two
            columns.

    Returns:
        Dict with ``"eps_ct_ij"``, ``"lam_ct_ij"`` and ``"r_star_ij"``, each
        ``(E,)`` in the unit of the per-atom entry it came from — the keyword
        arguments :meth:`ChargeTransfer.forward` expects.

    Raises:
        KeyError: A required key is missing from ``atom_params``.
    """
    return geometric_arithmetic_mixing(
        atom_params,
        pair_index,
        geometric_keys=["eps_ct", "lam_ct"],
        arithmetic_keys=["r_star"],
    )


# ---------------------------------------------------------------------------
# Helper: reduce pair energies to per-graph energies
# ---------------------------------------------------------------------------


def _reduce_pair_energy(
    pair_energy: torch.Tensor,
    bidirectional: bool,
    pair_batch: torch.Tensor | None,
    num_graphs: int | None,
) -> torch.Tensor:
    """Sum pair energies into per-graph totals (or a scalar)."""
    if bidirectional:
        pair_energy = 0.5 * pair_energy

    if pair_batch is None:
        return pair_energy.sum()

    if num_graphs is None:
        num_graphs = int(pair_batch.max().item()) + 1

    energy = torch.zeros(num_graphs, dtype=pair_energy.dtype, device=pair_energy.device)
    energy.index_add_(0, pair_batch, pair_energy)
    return energy


# ---------------------------------------------------------------------------
# RepulsionExp6
# ---------------------------------------------------------------------------


class RepulsionExp6(nn.Module):
    """Buckingham-style exponential repulsion with a short-range r^-12 wall.

    ``U = eps * exp(lam * (1 - r / r*)) + (s_rep / r)^12``, where ``r`` is the
    pair separation, ``eps`` the repulsion strength at contact, ``lam`` the
    (dimensionless) decay exponent and ``r*`` the reference radius at which the
    exponential equals ``eps``. In :meth:`forward` these are the per-pair
    ``eps_rep_ij``, ``lam_rep_ij`` and ``r_star_ij``.

    An exponential is the physically motivated shape for overlap repulsion, but
    it *saturates* at ``r → 0`` (it tends to ``eps·e^lam``, a finite number), so
    nothing stops two atoms from collapsing onto each other during optimisation.
    The added ``(s_rep / r)^12`` wall diverges and restores that barrier.

    **Dimensional caveat.** That wall term is a ratio of two lengths, hence
    dimensionless, while the exponential term is an energy — so the sum is only
    an energy once one energy unit is fixed and the wall is read as carrying a
    coefficient of exactly ``1`` in it. The term is therefore not invariant
    under a change of energy unit the way the rest of the module is. This is a
    property of the implemented form, not of the caller's parameters; it is
    recorded here rather than smoothed over, and it is why ``s_rep`` is
    described below as a radius rather than as a coefficient.

    Args:
        mixing_fn: Combining rule making per-pair parameters from per-atom ones,
            called as ``mixing_fn(atom_params, pair_index)`` with ``atom_params``
            holding ``"eps_rep"`` / ``"lam_rep"`` / ``"r_star"`` ``(N,)`` and
            ``pair_index`` ``(E, 2)``, and returning ``"eps_rep_ij"`` /
            ``"lam_rep_ij"`` / ``"r_star_ij"`` ``(E,)``. Applied by
            :class:`~molpot.composition.PotentialComposer`, not by
            :meth:`forward`. Defaults to :func:`repulsion_mixing`; a rule
            written for another potential's key set — including
            :class:`~molpot.potentials.vdw.lj126.MixingRule` — does not fit and
            raises.
        bidirectional: Set when the edge list holds both ``(i, j)`` and
            ``(j, i)``, so pair energies are halved to avoid double counting.
            Leave ``False`` for a half list.
        energy_scale: Dimensionless multiplicative factor applied to every pair
            energy.
        s_rep: Radius of the short-range ``r⁻¹²`` wall, in the model's length
            unit: the separation at which the wall term reaches ``1`` energy
            unit. Below it the wall dominates, above it the wall dies away as
            the twelfth power and the exponential carries the term.
    """

    def __init__(
        self,
        mixing_fn: Callable[
            [dict[str, torch.Tensor], torch.Tensor],
            dict[str, torch.Tensor],
        ] = repulsion_mixing,
        bidirectional: bool = True,
        energy_scale: float = 1.0,
        s_rep: float = 1.5,
    ):
        super().__init__()
        self.mixing_fn = mixing_fn
        self.bidirectional = bidirectional
        self.energy_scale = energy_scale
        self.s_rep = s_rep

    def forward(
        self,
        *,
        distance: torch.Tensor,
        eps_rep_ij: torch.Tensor,
        lam_rep_ij: torch.Tensor,
        r_star_ij: torch.Tensor,
        pair_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Compute repulsion energy.

        Units are the model's own throughout (module docstring): one length unit
        shared by ``distance`` and ``r_star_ij``, one energy unit for the result.

        Args:
            distance: Pair separations ``r`` ``(E,)``, a length. Clamped below at
                ``1e-6`` before the reciprocal so coincident atoms divide by that
                floor rather than by zero.
            eps_rep_ij: Per-pair repulsion strength at ``r = r*`` ``(E,)``, an
                energy.
            lam_rep_ij: Per-pair decay exponent ``(E,)``, dimensionless.
            r_star_ij: Per-pair reference radius ``(E,)``, a length. Also clamped
                at ``1e-6``.
            pair_batch: Index of the graph each edge belongs to ``(E,)``. ``None``
                sums to a scalar.
            num_graphs: Number of graphs ``B``; inferred from ``pair_batch`` when
                ``None``, which costs a host sync and so breaks a compiled graph.

        Returns:
            Per-graph energy ``(B,)``, or a scalar ``()`` when ``pair_batch`` is
            ``None``. Non-negative: both terms of the form are repulsive.
        """
        r = distance.clamp(min=1e-6)
        r_ratio = r / r_star_ij.clamp(min=1e-6)
        exp_term = eps_rep_ij * torch.exp(lam_rep_ij * (1.0 - r_ratio))
        wall_term = (self.s_rep / r).pow(12)
        pair_energy = (exp_term + wall_term) * self.energy_scale
        return _reduce_pair_energy(pair_energy, self.bidirectional, pair_batch, num_graphs)


# ---------------------------------------------------------------------------
# DispersionC6
# ---------------------------------------------------------------------------


class DispersionC6(nn.Module):
    """Tang-Toennies-style C6 dispersion.

    ``U = -C6 / (s_disp * r*^6 + r^6)``, where ``r`` is the pair separation,
    ``C6`` the dispersion coefficient and ``r*`` the reference radius; in
    :meth:`forward` these are the per-pair ``c6_ij`` and ``r_star_ij``.

    Dispersion is the weak attraction between two neutral atoms whose electron
    clouds fluctuate in step, and at long range it is exactly ``-C6/r⁶``. Taken
    literally that diverges as ``r → 0``, which is unphysical — at short range
    the clouds overlap and the multipole expansion behind ``r⁻⁶`` stops holding.
    Adding ``s_disp · r*⁶`` to the denominator *damps* the term: it is
    negligible against ``r⁶`` at long range, so the ``-C6/r⁶`` limit is
    preserved, and it caps the well at ``-C6/(s_disp · r*⁶)`` as ``r → 0``.
    Every term is an energy, with ``C6`` in energy × length⁶.

    Args:
        mixing_fn: Combining rule making per-pair parameters from per-atom ones,
            called as ``mixing_fn(atom_params, pair_index)`` with ``atom_params``
            holding ``"c6"`` / ``"r_star"`` ``(N,)`` and ``pair_index``
            ``(E, 2)``, and returning ``"c6_ij"`` / ``"r_star_ij"`` ``(E,)``.
            Applied by :class:`~molpot.composition.PotentialComposer`, not by
            :meth:`forward`. Defaults to :func:`dispersion_mixing`; a rule
            written for another potential's key set — including
            :class:`~molpot.potentials.vdw.lj126.MixingRule` — does not fit and
            raises.
        bidirectional: Set when the edge list holds both ``(i, j)`` and
            ``(j, i)``, so pair energies are halved to avoid double counting.
            Leave ``False`` for a half list.
        energy_scale: Dimensionless multiplicative factor applied to every pair
            energy.
        s_disp: Dimensionless damping strength. Larger values push the onset of
            damping outwards and shallow the capped well; the default ``120``
            keeps the damping well inside the region where ``r⁻⁶`` was already
            invalid.
    """

    def __init__(
        self,
        mixing_fn: Callable[
            [dict[str, torch.Tensor], torch.Tensor],
            dict[str, torch.Tensor],
        ] = dispersion_mixing,
        bidirectional: bool = True,
        energy_scale: float = 1.0,
        s_disp: float = 120.0,
    ):
        super().__init__()
        self.mixing_fn = mixing_fn
        self.bidirectional = bidirectional
        self.energy_scale = energy_scale
        self.s_disp = s_disp

    def forward(
        self,
        *,
        distance: torch.Tensor,
        c6_ij: torch.Tensor,
        r_star_ij: torch.Tensor,
        pair_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Compute dispersion energy.

        Units are the model's own throughout (module docstring): one length unit
        shared by ``distance`` and ``r_star_ij``, one energy unit for the result.

        Args:
            distance: Pair separations ``r`` ``(E,)``, a length. Clamped below at
                ``1e-6``.
            c6_ij: Per-pair dispersion coefficient ``C6`` ``(E,)``, in
                energy × length⁶.
            r_star_ij: Per-pair reference radius ``(E,)``, a length; sets where
                the short-range damping takes over. Also clamped at ``1e-6``.
            pair_batch: Index of the graph each edge belongs to ``(E,)``. ``None``
                sums to a scalar.
            num_graphs: Number of graphs ``B``; inferred from ``pair_batch`` when
                ``None``, which costs a host sync and so breaks a compiled graph.

        Returns:
            Per-graph energy ``(B,)``, or a scalar ``()`` when ``pair_batch`` is
            ``None``. Non-positive: dispersion is purely attractive.
        """
        r = distance.clamp(min=1e-6)
        r6 = r.pow(6)
        r_star6 = r_star_ij.clamp(min=1e-6).pow(6)
        pair_energy = -c6_ij / (self.s_disp * r_star6 + r6)
        pair_energy = pair_energy * self.energy_scale
        return _reduce_pair_energy(pair_energy, self.bidirectional, pair_batch, num_graphs)


# ---------------------------------------------------------------------------
# ChargeTransfer
# ---------------------------------------------------------------------------


class ChargeTransfer(nn.Module):
    """Charge-transfer potential.

    ``U = (eps / r^4) * exp(-(lam * r* / r)^3)``, where ``r`` is the pair
    separation, ``eps`` the transfer strength, ``lam`` a dimensionless range
    parameter and ``r*`` the reference radius; in :meth:`forward` these are the
    per-pair ``eps_ct_ij``, ``lam_ct_ij`` and ``r_star_ij``.

    *Charge transfer* is the small flow of electron density from one atom onto
    a close neighbour — a genuinely quantum effect that neither electrostatics
    nor dispersion accounts for. It is short-ranged, and the form above encodes
    that twice over: the ``r⁻⁴`` prefactor supplies the decay, and the
    exponential of ``-(lam r*/r)³`` switches the term **off** as ``r`` shrinks,
    since the argument grows without bound there. The product therefore peaks at
    an intermediate separation and vanishes at both extremes, which is why no
    short-range wall is needed here. For ``U`` to be an energy, ``eps`` carries
    units of energy × length⁴.

    Args:
        mixing_fn: Combining rule making per-pair parameters from per-atom ones,
            called as ``mixing_fn(atom_params, pair_index)`` with ``atom_params``
            holding ``"eps_ct"`` / ``"lam_ct"`` / ``"r_star"`` ``(N,)`` and
            ``pair_index`` ``(E, 2)``, and returning ``"eps_ct_ij"`` /
            ``"lam_ct_ij"`` / ``"r_star_ij"`` ``(E,)``. Applied by
            :class:`~molpot.composition.PotentialComposer`, not by
            :meth:`forward`. Defaults to :func:`ct_mixing`; a rule written for
            another potential's key set — including
            :class:`~molpot.potentials.vdw.lj126.MixingRule` — does not fit and
            raises.
        bidirectional: Set when the edge list holds both ``(i, j)`` and
            ``(j, i)``, so pair energies are halved to avoid double counting.
            Leave ``False`` for a half list.
        energy_scale: Dimensionless multiplicative factor applied to every pair
            energy.
    """

    def __init__(
        self,
        mixing_fn: Callable[
            [dict[str, torch.Tensor], torch.Tensor],
            dict[str, torch.Tensor],
        ] = ct_mixing,
        bidirectional: bool = True,
        energy_scale: float = 1.0,
    ):
        super().__init__()
        self.mixing_fn = mixing_fn
        self.bidirectional = bidirectional
        self.energy_scale = energy_scale

    def forward(
        self,
        *,
        distance: torch.Tensor,
        eps_ct_ij: torch.Tensor,
        lam_ct_ij: torch.Tensor,
        r_star_ij: torch.Tensor,
        pair_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Compute charge-transfer energy.

        Units are the model's own throughout (module docstring): one length unit
        shared by ``distance`` and ``r_star_ij``, one energy unit for the result.

        Args:
            distance: Pair separations ``r`` ``(E,)``, a length. Clamped below at
                ``1e-6``.
            eps_ct_ij: Per-pair charge-transfer strength ``(E,)``, in
                energy × length⁴ — it multiplies ``r⁻⁴``.
            lam_ct_ij: Per-pair range parameter ``(E,)``, dimensionless. Larger
                values switch the term off further out.
            r_star_ij: Per-pair reference radius ``(E,)``, a length. Also clamped
                at ``1e-6``.
            pair_batch: Index of the graph each edge belongs to ``(E,)``. ``None``
                sums to a scalar.
            num_graphs: Number of graphs ``B``; inferred from ``pair_batch`` when
                ``None``, which costs a host sync and so breaks a compiled graph.

        Returns:
            Per-graph energy ``(B,)``, or a scalar ``()`` when ``pair_batch`` is
            ``None``. Its sign follows ``eps_ct_ij``, which the
            :class:`~molpot.composition.ChargeTransferParameterHead` floors above
            zero.
        """
        r = distance.clamp(min=1e-6)
        ratio = (lam_ct_ij * r_star_ij.clamp(min=1e-6)) / r
        pair_energy = (eps_ct_ij / r.pow(4)) * torch.exp(-ratio.pow(3))
        pair_energy = pair_energy * self.energy_scale
        return _reduce_pair_energy(pair_energy, self.bidirectional, pair_batch, num_graphs)
