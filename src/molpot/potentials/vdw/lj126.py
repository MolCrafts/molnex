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

Units are the caller's: with the ``real`` unit system used by the classical
force-field layer, ``ε`` is in kcal/mol, ``σ`` and ``r`` in Å, and the returned
energies in kcal/mol.
"""

from __future__ import annotations

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


def lorentz_berthelot(
    atom_params: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Lorentz-Berthelot mixing rules: per-atom → per-pair.

    A force field stores one ``ε`` and one ``σ`` per atom; a pair interaction
    needs one of each per *pair*. The Lorentz–Berthelot rules are the standard
    empirical combination: the Lorentz rule takes the arithmetic mean of the
    sizes, ``σ_ij = ½(σ_i + σ_j)`` (Å), and the Berthelot rule the geometric
    mean of the well depths, ``ε_ij = sqrt(ε_i ε_j)`` (kcal/mol).

    ``epsilon_ij`` is the **exact** geometric mean ``sqrt(eps_i eps_j)``: nothing
    is added under the root. An additive ``1e-12`` there (removed 2026-09-07)
    biased every well depth — ``eps = 0.2`` came out ``1.25e-11`` relative too
    deep, which is visible at the LJ minimum, and a learnable ``eps = 1e-6`` came
    out 41 % too deep. The zero case that guard was protecting is instead handled
    by masking: ``eps_i eps_j == 0`` maps to ``0`` with a finite backward pass,
    where the unmasked ``d sqrt(x)/dx`` would diverge. A *negative* product still
    yields ``NaN`` — that is a broken parameterization, not something to swallow.

    Args:
        atom_params: Must contain ``"epsilon"`` ``(N,)`` in kcal/mol and
            ``"sigma"`` ``(N,)`` in Å, one entry per atom.
        edge_index: Edge indices ``(E, 2)`` — column 0 the first atom of each
            pair, column 1 the second. Both rules are symmetric in the two
            columns, so the edge direction does not matter here.

    Returns:
        Dict with ``"epsilon_ij"`` ``(E,)`` in kcal/mol and ``"sigma_ij"``
        ``(E,)`` in Å.
    """
    src, dst = edge_index[:, 0], edge_index[:, 1]
    epsilon = atom_params["epsilon"]
    sigma = atom_params["sigma"]
    product = epsilon[src] * epsilon[dst]
    nonzero = product != 0
    safe = torch.where(nonzero, product, torch.ones_like(product))
    return {
        "epsilon_ij": torch.where(nonzero, torch.sqrt(safe), torch.zeros_like(product)),
        "sigma_ij": 0.5 * (sigma[src] + sigma[dst]),
    }


class LJ126(nn.Module):
    """Lennard-Jones 12-6 potential.

    Pure physics: takes per-pair parameters and distances, returns energy.

    Args:
        mixing_fn: Converts per-atom params to per-pair params.
            Defaults to ``lorentz_berthelot``.
        bidirectional: Set when the edge list is bidirectional — it holds both
            ``(i, j)`` and ``(j, i)`` — so pair energies are halved to avoid
            double counting. Leave ``False`` for a half list.
        energy_scale: Dimensionless multiplicative factor applied to every pair
            energy.
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
        edge_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Compute LJ 12-6 energy.

        Args:
            distance: Pairwise distances ``r`` ``(E,)`` in Å.
            epsilon_ij: Per-pair well depth ``ε_ij`` ``(E,)`` in kcal/mol.
            sigma_ij: Per-pair zero-crossing distance ``σ_ij`` ``(E,)`` in Å.
            edge_batch: Index of the molecule (graph) each edge belongs to
                ``(E,)``. If None, returns summed scalar.
            num_graphs: Number of molecules ``B`` (inferred from ``edge_batch``
                if None).

        Returns:
            Per-graph energy ``(B,)`` in kcal/mol, or a scalar ``()`` if no
            batching.
        """
        pair_energy = lj126_pair_energy(distance, epsilon_ij, sigma_ij)
        pair_energy = pair_energy * self.energy_scale
        if self.bidirectional:
            pair_energy = 0.5 * pair_energy

        if edge_batch is None:
            return pair_energy.sum()

        if num_graphs is None:
            num_graphs = int(edge_batch.max().item()) + 1

        energy = torch.zeros(num_graphs, dtype=pair_energy.dtype, device=pair_energy.device)
        energy.index_add_(0, edge_batch, pair_energy)
        return energy
