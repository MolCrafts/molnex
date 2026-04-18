"""Non-bonded pairwise potentials: repulsion, dispersion, charge transfer."""

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
    edge_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Mixing rules for RepulsionExp6 parameters."""
    return geometric_arithmetic_mixing(
        atom_params,
        edge_index,
        geometric_keys=["eps_rep", "lam_rep"],
        arithmetic_keys=["r_star"],
    )


def dispersion_mixing(
    atom_params: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Mixing rules for DispersionC6 parameters."""
    return geometric_arithmetic_mixing(
        atom_params,
        edge_index,
        geometric_keys=["c6"],
        arithmetic_keys=["r_star"],
    )


def ct_mixing(
    atom_params: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Mixing rules for ChargeTransfer parameters."""
    return geometric_arithmetic_mixing(
        atom_params,
        edge_index,
        geometric_keys=["eps_ct", "lam_ct"],
        arithmetic_keys=["r_star"],
    )


# ---------------------------------------------------------------------------
# Helper: reduce pair energies to per-graph energies
# ---------------------------------------------------------------------------


def _reduce_pair_energy(
    pair_energy: torch.Tensor,
    bidirectional: bool,
    edge_batch: torch.Tensor | None,
    num_graphs: int | None,
) -> torch.Tensor:
    """Sum pair energies into per-graph totals (or a scalar)."""
    if bidirectional:
        pair_energy = 0.5 * pair_energy

    if edge_batch is None:
        return pair_energy.sum()

    if num_graphs is None:
        num_graphs = int(edge_batch.max().item()) + 1

    energy = torch.zeros(num_graphs, dtype=pair_energy.dtype, device=pair_energy.device)
    energy.index_add_(0, edge_batch, pair_energy)
    return energy


# ---------------------------------------------------------------------------
# RepulsionExp6
# ---------------------------------------------------------------------------


class RepulsionExp6(nn.Module):
    """Buckingham-style exponential repulsion with a short-range r^-12 wall.

    ``U = eps * exp(lam * (1 - r / r*)) + (s_rep / r)^12``

    Args:
        mixing_fn: Converts per-atom params to per-pair params.
        bidirectional: Halve pair energies to avoid double-counting.
        energy_scale: Multiplicative energy scaling factor.
        s_rep: Short-range wall coefficient.
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
        edge_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Compute repulsion energy.

        Args:
            distance: Pairwise distances ``(E,)``.
            eps_rep_ij: Per-pair well depth ``(E,)``.
            lam_rep_ij: Per-pair exponent ``(E,)``.
            r_star_ij: Per-pair equilibrium distance ``(E,)``.
            edge_batch: Graph index per edge ``(E,)``.
            num_graphs: Number of graphs.

        Returns:
            Per-graph energy ``(B,)`` or scalar.
        """
        r = distance.clamp(min=1e-6)
        r_ratio = r / r_star_ij.clamp(min=1e-6)
        exp_term = eps_rep_ij * torch.exp(lam_rep_ij * (1.0 - r_ratio))
        wall_term = (self.s_rep / r).pow(12)
        pair_energy = (exp_term + wall_term) * self.energy_scale
        return _reduce_pair_energy(pair_energy, self.bidirectional, edge_batch, num_graphs)


# ---------------------------------------------------------------------------
# DispersionC6
# ---------------------------------------------------------------------------


class DispersionC6(nn.Module):
    """Tang-Toennies-style C6 dispersion.

    ``U = -C6 / (s_disp * r*^6 + r^6)``

    Args:
        mixing_fn: Converts per-atom params to per-pair params.
        bidirectional: Halve pair energies to avoid double-counting.
        energy_scale: Multiplicative energy scaling factor.
        s_disp: Damping coefficient for short-range regularization.
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
        edge_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Compute dispersion energy.

        Args:
            distance: Pairwise distances ``(E,)``.
            c6_ij: Per-pair C6 coefficient ``(E,)``.
            r_star_ij: Per-pair equilibrium distance ``(E,)``.
            edge_batch: Graph index per edge ``(E,)``.
            num_graphs: Number of graphs.

        Returns:
            Per-graph energy ``(B,)`` or scalar.
        """
        r = distance.clamp(min=1e-6)
        r6 = r.pow(6)
        r_star6 = r_star_ij.clamp(min=1e-6).pow(6)
        pair_energy = -c6_ij / (self.s_disp * r_star6 + r6)
        pair_energy = pair_energy * self.energy_scale
        return _reduce_pair_energy(pair_energy, self.bidirectional, edge_batch, num_graphs)


# ---------------------------------------------------------------------------
# ChargeTransfer
# ---------------------------------------------------------------------------


class ChargeTransfer(nn.Module):
    """Charge-transfer potential.

    ``U = (eps / r^4) * exp(-(lam * r* / r)^3)``

    Args:
        mixing_fn: Converts per-atom params to per-pair params.
        bidirectional: Halve pair energies to avoid double-counting.
        energy_scale: Multiplicative energy scaling factor.
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
        edge_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Compute charge-transfer energy.

        Args:
            distance: Pairwise distances ``(E,)``.
            eps_ct_ij: Per-pair CT energy scale ``(E,)``.
            lam_ct_ij: Per-pair CT exponent ``(E,)``.
            r_star_ij: Per-pair equilibrium distance ``(E,)``.
            edge_batch: Graph index per edge ``(E,)``.
            num_graphs: Number of graphs.

        Returns:
            Per-graph energy ``(B,)`` or scalar.
        """
        r = distance.clamp(min=1e-6)
        ratio = (lam_ct_ij * r_star_ij.clamp(min=1e-6)) / r
        pair_energy = (eps_ct_ij / r.pow(4)) * torch.exp(-ratio.pow(3))
        pair_energy = pair_energy * self.energy_scale
        return _reduce_pair_energy(pair_energy, self.bidirectional, edge_batch, num_graphs)
