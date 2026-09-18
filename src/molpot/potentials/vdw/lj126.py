"""Lennard-Jones 12-6 potential.

Two neutral atoms a distance ``r`` apart attract each other weakly at long
range (London dispersion, the fluctuating-dipole interaction that falls off as
``r⁻⁶``) and repel each other steeply once their electron clouds overlap. The
Lennard-Jones 12-6 form models both with two parameters per pair,

    ``E(r) = 4 ε [(σ/r)¹² − (σ/r)⁶]``,

where ``ε`` is the depth of the energy minimum and ``σ`` the separation at
which ``E`` crosses zero. The minimum sits at ``r = 2^(1/6) σ`` with
``E = −ε``. The ``r⁻¹²`` exponent is a computational convenience — the square
of the physically motivated ``r⁻⁶`` term — not a derived law.

A force field stores one ``ε`` and one ``σ`` per atom, so a pair needs a
*combining rule* to make one of each per pair. Which rule a model uses is part
of its identity, not a detail of the kernel: Lorentz–Berthelot (``σ``
arithmetic, ``ε`` geometric) and the fully geometric rule of OPLS-AA give
different cross-term energies from the same parameter table. The two are
:class:`MixingRule` here, chosen once when a potential is constructed and
recorded by
:class:`~molpot.ir.nonbonded.NonbondedMethodRecord` so a manifest, an exported
deck and the evaluated model all name the same rule.

Units are the caller's: with the ``real`` unit system used by the classical
force-field layer, ``ε`` is in kcal/mol, ``σ`` and ``r`` in Å, and the returned
energies in kcal/mol.

References:
    Lorentz, H. A. "Ueber die Anwendung des Satzes vom Virial in der kinetischen
    Theorie der Gase" Ann. Phys. 248, 127 (1881) — the arithmetic ``σ`` rule.
    https://doi.org/10.1002/andp.18812480110
    Berthelot, D. "Sur le mélange des gaz" C. R. Acad. Sci. 126, 1703 (1898) —
    the geometric ``ε`` rule.
    Jorgensen, W. L.; Maxwell, D. S.; Tirado-Rives, J. "Development and Testing
    of the OPLS All-Atom Force Field on Conformational Energetics and Properties
    of Organic Liquids" J. Am. Chem. Soc. 118, 11225 (1996) — OPLS-AA combines
    **both** ``ε`` and ``σ`` geometrically.
    https://doi.org/10.1021/ja9621760
"""

from __future__ import annotations

from enum import StrEnum
from typing import Callable

import torch
import torch.nn as nn


def lj126_pair_energy_split(
    distance: torch.Tensor,
    epsilon_ij: torch.Tensor,
    sigma_ij: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repulsive and attractive halves of the LJ 12-6 pair energy.

    ``4 eps [(sigma/r)^12 - (sigma/r)^6]`` splits exactly into the repulsive
    wall ``4 eps (sigma/r)^12`` and the attractive tail ``-4 eps (sigma/r)^6``.
    The assembled potential attributes the first to ``E_local`` and the second
    to ``E_disp`` (S3), so the S-decomposition stays a *grouping* of this one
    energy path rather than a second one: the halves always sum back to
    :func:`lj126_pair_energy`.

    Args:
        distance: Pairwise distances ``r`` ``(E,)`` in Å over the ``E`` pairs.
            Clamped below at ``1e-6`` before the reciprocal, so two coincident
            atoms divide by ``1e-6`` rather than by zero; the resulting energy
            is still enormous (and can overflow to ``inf`` in float32), the
            clamp only keeps the division defined.
        epsilon_ij: Per-pair well depth ``ε_ij`` ``(E,)`` in kcal/mol.
        sigma_ij: Per-pair zero-crossing distance ``σ_ij`` ``(E,)`` in Å.

    Returns:
        ``(repulsion, dispersion)``, each ``(E,)`` in kcal/mol — non-negative
        and non-positive respectively. At the minimum ``r = 2^(1/6) σ_ij``,
        where ``(σ/r)⁶ = 1/2``, they are ``+ε_ij`` and ``-2 ε_ij``, summing to
        the well depth ``-ε_ij``.
    """
    sr6 = (sigma_ij / distance.clamp(min=1e-6)).pow(6)
    four_epsilon = 4.0 * epsilon_ij
    return four_epsilon * sr6.pow(2), -four_epsilon * sr6


def lj126_pair_energy(
    distance: torch.Tensor,
    epsilon_ij: torch.Tensor,
    sigma_ij: torch.Tensor,
) -> torch.Tensor:
    """LJ 12-6 pair energy kernel.

    ``E_ij = 4 * epsilon_ij * [(sigma_ij / r)^12 - (sigma_ij / r)^6]``, computed
    as the sum of :func:`lj126_pair_energy_split` so the split and the total
    can never drift apart.

    Args:
        distance: Pairwise distances ``r`` ``(E,)`` in Å.
        epsilon_ij: Per-pair well depth ``ε_ij`` ``(E,)`` in kcal/mol.
        sigma_ij: Per-pair zero-crossing distance ``σ_ij`` ``(E,)`` in Å.

    Returns:
        Pair energies ``(E,)`` in kcal/mol.
    """
    repulsion, dispersion = lj126_pair_energy_split(distance, epsilon_ij, sigma_ij)
    return repulsion + dispersion


def geometric_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sqrt(a b)`` elementwise, with the zero product masked.

    The one combining primitive every geometric mixing rule in molpot is built
    from — :func:`lorentz_berthelot`'s ``ε``, both halves of
    :func:`geometric_mixing`, and the geometric keys of
    :func:`~molpot.potentials.mixing.geometric_arithmetic_mixing`.

    The **exact** geometric mean: nothing is added under the root. An additive
    ``1e-12`` there (removed from :func:`lorentz_berthelot` on 2026-09-07 and
    from ``geometric_arithmetic_mixing`` on 2026-09-08, ledger debt ``F10``)
    biased every mixed value — ``p = 0.2`` came out ``1.25e-11`` relative too
    large, which is visible at the LJ minimum, and a learnable ``p = 1e-6`` came
    out 41 % too large. The zero case that guard was protecting is instead
    handled by masking: a zero product maps to ``0`` with a finite backward
    pass, where the unmasked ``d sqrt(x)/dx`` diverges. A *negative* product
    still yields ``NaN`` — that is a broken parameterization (a well depth or a
    dispersion coefficient is non-negative by construction), not something to
    swallow.

    Args:
        a: First operand ``(E,)``, in the parameter's own unit — one per pair,
            already gathered from the per-atom table.
        b: Second operand ``(E,)``, in the same unit as ``a``.

    Returns:
        The elementwise geometric mean ``(E,)``, in the unit of ``a``.
    """
    product = a * b
    nonzero = product != 0
    safe = torch.where(nonzero, product, torch.ones_like(product))
    return torch.where(nonzero, torch.sqrt(safe), torch.zeros_like(product))


def lorentz_berthelot(
    atom_params: dict[str, torch.Tensor],
    pair_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Lorentz-Berthelot mixing rules: per-atom → per-pair.

    A force field stores one ``ε`` and one ``σ`` per atom; a pair interaction
    needs one of each per *pair*. The Lorentz–Berthelot rules are the standard
    empirical combination: the Lorentz rule takes the arithmetic mean of the
    sizes, ``σ_ij = ½(σ_i + σ_j)`` (Å), and the Berthelot rule the geometric
    mean of the well depths, ``ε_ij = sqrt(ε_i ε_j)`` (kcal/mol).

    ``epsilon_ij`` is the exact masked geometric mean of
    :func:`geometric_mean`; ``sigma_ij`` is the plain arithmetic mean, which
    needs no guard.

    This is the rule of AMBER and CHARMM, and molnex's default. OPLS-AA is
    geometric in ``σ`` as well — :func:`geometric_mixing` — and which of the two
    a model uses is declared once, as :class:`MixingRule`.

    Args:
        atom_params: Must contain ``"epsilon"`` ``(N,)`` in kcal/mol and
            ``"sigma"`` ``(N,)`` in Å, one entry per atom.
        pair_index: Edge indices ``(E, 2)`` — column 0 the first atom of each
            pair, column 1 the second. Both rules are symmetric in the two
            columns, so the edge direction does not matter here.

    Returns:
        Dict with ``"epsilon_ij"`` ``(E,)`` in kcal/mol and ``"sigma_ij"``
        ``(E,)`` in Å.

    References:
        Lorentz, H. A. "Ueber die Anwendung des Satzes vom Virial in der
        kinetischen Theorie der Gase" Ann. Phys. 248, 127 (1881) — the
        arithmetic ``σ`` rule. https://doi.org/10.1002/andp.18812480110
        Berthelot, D. "Sur le mélange des gaz" C. R. Acad. Sci. 126, 1703
        (1898) — the geometric ``ε`` rule.
    """
    src, dst = pair_index[:, 0], pair_index[:, 1]
    epsilon, sigma = atom_params["epsilon"], atom_params["sigma"]
    return {
        "epsilon_ij": geometric_mean(epsilon[src], epsilon[dst]),
        "sigma_ij": 0.5 * (sigma[src] + sigma[dst]),
    }


def geometric_mixing(
    atom_params: dict[str, torch.Tensor],
    pair_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Geometric mixing rules (OPLS-AA): per-atom → per-pair.

    Both parameters are combined as geometric means,
    ``ε_ij = sqrt(ε_i ε_j)`` (kcal/mol) and ``σ_ij = sqrt(σ_i σ_j)`` (Å). This
    is the rule OPLS-AA was parameterized with: its ``σ`` differs from the
    Lorentz–Berthelot arithmetic mean for every pair of unequal sizes (3.5 Å
    with 2.5 Å gives 2.958 Å here against 3.0 Å there), so a model fitted with
    one rule and evaluated with the other is a different force field.

    Both means use the masked square root of :func:`geometric_mean` — no
    additive ``1e-12``, for the reasons stated there. A zero ``σ`` is a real
    parameterization (a "no Lennard-Jones site" atom, whose ``ε`` is zero too),
    and it maps to ``σ_ij = 0`` with a finite backward exactly like ``ε``; the
    pair energy is then zero through its own ``ε_ij``.

    Args:
        atom_params: Must contain ``"epsilon"`` ``(N,)`` in kcal/mol and
            ``"sigma"`` ``(N,)`` in Å, one entry per atom.
        pair_index: Edge indices ``(E, 2)`` — column 0 the first atom of each
            pair, column 1 the second. Both means are symmetric in the two
            columns, so the edge direction does not matter here.

    Returns:
        Dict with ``"epsilon_ij"`` ``(E,)`` in kcal/mol and ``"sigma_ij"``
        ``(E,)`` in Å.

    References:
        Jorgensen, W. L.; Maxwell, D. S.; Tirado-Rives, J. "Development and
        Testing of the OPLS All-Atom Force Field on Conformational Energetics
        and Properties of Organic Liquids" J. Am. Chem. Soc. 118, 11225 (1996).
        https://doi.org/10.1021/ja9621760
    """
    src, dst = pair_index[:, 0], pair_index[:, 1]
    epsilon, sigma = atom_params["epsilon"], atom_params["sigma"]
    return {
        "epsilon_ij": geometric_mean(epsilon[src], epsilon[dst]),
        "sigma_ij": geometric_mean(sigma[src], sigma[dst]),
    }


class MixingRule(StrEnum):
    """Which combining rule turns per-atom ``ε`` / ``σ`` into per-pair values.

    A closed vocabulary, because the rule is model identity: it is declared as
    the ``mixing`` option of the ``pair:lj/cut`` style, carried by
    :class:`~molpot.ir.nonbonded.NonbondedMethodRecord` into the IR and the
    model manifest, and read back by both export adapters. A string outside this
    enum is refused where it is *declared*
    (:meth:`~molpot.forcefield.ForceField.def_pairstyle`), never at evaluation.

    Being a :class:`~enum.StrEnum`, a member *is* its recorded value
    (``MixingRule.GEOMETRIC == "geometric"``), so a record round-trips through
    JSON without a codec.

    Attributes:
        LORENTZ_BERTHELOT: ``σ`` arithmetic, ``ε`` geometric — AMBER / CHARMM,
            and molnex's default (:func:`lorentz_berthelot`).
        GEOMETRIC: both geometric — OPLS-AA (:func:`geometric_mixing`).
    """

    LORENTZ_BERTHELOT = "lorentz-berthelot"
    GEOMETRIC = "geometric"

    def mix(
        self,
        atom_params: dict[str, torch.Tensor],
        pair_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply this rule to a per-atom parameter table.

        Dispatch is a lookup in :data:`_MIXING_FUNCTIONS`, the one place a
        member is paired with the function that implements it — so a member
        added without an implementation raises here instead of falling through
        to whichever rule an ``else`` branch happened to name, which would be a
        different force field computed silently.

        The lookup key is a Python enum member held as a *static* attribute of
        the calling potential, fixed when that potential was constructed, and
        the table is a module constant. ``torch.compile`` therefore resolves
        both while tracing — one specialised graph per rule, no data-dependent
        control flow and no graph break on the force path.

        Args:
            atom_params: Must contain ``"epsilon"`` ``(N,)`` in kcal/mol and
                ``"sigma"`` ``(N,)`` in Å.
            pair_index: Edge indices ``(E, 2)``.

        Returns:
            Dict with ``"epsilon_ij"`` ``(E,)`` in kcal/mol and ``"sigma_ij"``
            ``(E,)`` in Å.

        Raises:
            ValueError: This member has no entry in :data:`_MIXING_FUNCTIONS`.
        """
        rule = _MIXING_FUNCTIONS.get(self)
        if rule is None:
            raise ValueError(f"no mixing function is registered for {self!r}")
        return rule(atom_params, pair_index)


#: Which function each :class:`MixingRule` member evaluates. Defined beside the
#: enum rather than inside it because an enum class body would take a plain
#: function for a member; kept exhaustive by :meth:`MixingRule.mix`, which
#: raises on a member that is missing here.
_MIXING_FUNCTIONS: dict[
    MixingRule,
    Callable[[dict[str, torch.Tensor], torch.Tensor], dict[str, torch.Tensor]],
] = {
    MixingRule.LORENTZ_BERTHELOT: lorentz_berthelot,
    MixingRule.GEOMETRIC: geometric_mixing,
}


class LJ126(nn.Module):
    """Lennard-Jones 12-6 potential over a neighbour list.

    Pure physics: :meth:`forward` takes distances and *per-pair* ``ε`` / ``σ``
    and returns energy. It mixes nothing itself. The combining rule that turns a
    force field's per-atom tables into those per-pair values is held here as
    :attr:`mixing_fn` and applied by the caller one step earlier —
    :class:`~molpot.composition.PotentialComposer` is that caller in tree, which
    evaluates ``potential.mixing_fn(atom_params, pair_index)`` and passes the
    result in as ``epsilon_ij`` / ``sigma_ij``. Reading the rule off the
    potential is what keeps the parameters and the rule that combined them
    together, rather than leaving the composer to pick one.

    Which rule that is belongs to the model, not to this kernel: see
    :class:`MixingRule`, whose bound :meth:`MixingRule.mix` is itself a valid
    ``mixing_fn``. The assembled Class-I path does not run through this class —
    :class:`~molpot.forcefield.pair_terms.LJCutPotential` holds a
    :class:`MixingRule` directly and applies it inside its own kernel.

    Args:
        mixing_fn: Combining rule, called as
            ``mixing_fn(atom_params, pair_index)`` where ``atom_params`` holds
            ``"epsilon"`` ``(N,)`` in kcal/mol and ``"sigma"`` ``(N,)`` in Å and
            ``pair_index`` is ``(E, 2)``, returning ``"epsilon_ij"`` ``(E,)`` in
            kcal/mol and ``"sigma_ij"`` ``(E,)`` in Å. Defaults to
            :func:`lorentz_berthelot`, the AMBER / CHARMM rule;
            :func:`geometric_mixing` (equivalently
            ``MixingRule.GEOMETRIC.mix``) is the OPLS-AA one.
        bidirectional: Set when the edge list is bidirectional — it holds both
            ``(i, j)`` and ``(j, i)`` — so pair energies are halved to avoid
            double counting. Leave ``False`` for a half list.
        energy_scale: Dimensionless multiplicative factor applied to every pair
            energy.

    Attributes:
        mixing_fn: The combining rule as passed in. Read by the composer before
            it calls :meth:`forward`, never by :meth:`forward` itself.
        bidirectional: Whether the edge list holds each pair twice.
        energy_scale: The dimensionless per-pair scale factor.
    """

    def __init__(
        self,
        mixing_fn: Callable[
            [dict[str, torch.Tensor], torch.Tensor],
            dict[str, torch.Tensor],
        ] = lorentz_berthelot,
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
        epsilon_ij: torch.Tensor,
        sigma_ij: torch.Tensor,
        pair_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Compute LJ 12-6 energy.

        Args:
            distance: Pairwise distances ``r`` ``(E,)`` in Å.
            epsilon_ij: Per-pair well depth ``ε_ij`` ``(E,)`` in kcal/mol.
            sigma_ij: Per-pair zero-crossing distance ``σ_ij`` ``(E,)`` in Å.
            pair_batch: Index of the molecule (graph) each edge belongs to
                ``(E,)``. If None, returns summed scalar.
            num_graphs: Number of molecules ``B`` (inferred from ``pair_batch``
                if None).

        Returns:
            Per-graph energy ``(B,)`` in kcal/mol, or a scalar ``()`` if no
            batching.
        """
        pair_energy = lj126_pair_energy(distance, epsilon_ij, sigma_ij)
        pair_energy = pair_energy * self.energy_scale
        if self.bidirectional:
            pair_energy = 0.5 * pair_energy

        if pair_batch is None:
            return pair_energy.sum()

        if num_graphs is None:
            num_graphs = int(pair_batch.max().item()) + 1

        energy = torch.zeros(num_graphs, dtype=pair_energy.dtype, device=pair_energy.device)
        energy.index_add_(0, pair_batch, pair_energy)
        return energy
