"""Antisymmetric bond-charge head — charge-conserving per-atom charge readout.

The head produces per-atom partial charges ``q_i`` via an edge-based
antisymmetric construction

.. math::

    q_{ij} = f(h_i, h_j, r_{ij}) - f(h_j, h_i, r_{ji}),
    \\qquad
    q_i = \\sum_{j \\in \\mathcal{N}(i)} q_{ij}

where ``f`` is a small MLP applied identically in both directions. By
construction ``q_{ij} = -q_{ji}`` and therefore ``\\sum_i q_i = 0`` for any
neutral system — charge conservation is built into the architecture, not
imposed by a per-graph correction step.

An optional learned per-graph bias projects the atomic-charge sum to a
caller-supplied total charge ``Q_\\text{net}`` for explicitly charged
systems (e.g. ions, electrochemical cells). The projection follows the
same uniform-shift recipe used by :class:`molpot.heads.DipoleHead`'s
``charge_neutrality`` branch.

The head emits an atom-level ``q_i`` that plugs directly into
:class:`molpot.potentials.elec.ewald_multipole.EwaldMultipoleEnergy`'s
``q`` input. For an atomic dipole ``\\mu_i``, the LES Ewald already
covers all q-q, q-μ and μ-μ cross terms (Aguado & Madden 2003 split,
Cheng 2025 LES screening) — no separate q-μ kernel is needed here.

References:
    * Cheng B., *Latent Ewald summation for machine-learning potentials*,
      npj Comput. Mater. **11**, 80 (2025).
      https://doi.org/10.1038/s41524-025-01577-7 — supervision on
      energy/force only; partial charges treated as latent variables
      because partitioning-scheme choice (Hirshfeld / Mulliken / CM5 /
      RESP / DDEC6) is not a physical observable.
    * Ko T. W., Finkler J. A., Goedecker S., Behler J.,
      *General-purpose machine learning potentials capturing nonlocal
      charge transfer*, Nat. Commun. **12**, 398 (2021).
      https://doi.org/10.1038/s41467-020-20427-2 — closest 4G-HDNNP
      precedent for enforcing global charge conservation in a learned
      MLIP (their route is the QEq Lagrange multiplier; ours is an
      architectural antisymmetric construction).
    * Aguado A. & Madden P. A., *Ewald summation of electrostatic
      multipole interactions up to the quadrupolar level*,
      J. Chem. Phys. **119**, 7471 (2003).
      https://doi.org/10.1063/1.1605941 — canonical q-μ / μ-μ Ewald
      split that the downstream :class:`EwaldMultipoleEnergy` consumes
      these ``q_i`` (and per-atom ``\\mu_i``) through.

Note:
    The exact ``q_{ij} = f(h_i,h_j,r) - f(h_j,h_i,r)`` antisymmetric
    pair construction does not have (to our knowledge) a published
    MLIP precedent; the derivation that ``\\sum_i q_i = 0`` is
    straightforward and is documented in the test
    ``test_charge_bond.py::test_neutral_sum``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molix import config
from molix.F.scatter import scatter_sum

__all__ = ["BondChargeHead"]


def _graph_counts(batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    ones = torch.ones(batch.shape[0], dtype=config.ftype, device=batch.device)
    return scatter_sum(ones, batch, dim=0, dim_size=num_graphs).clamp(min=1.0)


class BondChargeHead(nn.Module):
    """Antisymmetric bond-charge head emitting charge-conserving ``q_i``.

    Args:
        node_dim: Feature dim of the invariant (scalar) node embedding
            ``h_i`` produced by the encoder.
        edge_dim: Optional feature dim of an invariant edge embedding
            (e.g. radial-basis projected bond distance, MACE-style
            ``R_{ij}`` weights). When ``None`` the head consumes only
            the bond distance scalar.
        hidden_dim: Hidden dimension of the bond-charge MLP.
        full_neighbor_list: When ``True`` (default, matches
            :class:`molix.nn.locality.NeighborList`'s ``symmetry=True``
            default), the edge list contains both ``(i, j)`` and
            ``(j, i)`` and ``q_{ij}`` is scattered only to the source
            atom — antisymmetry of the bidirectional pair guarantees
            ``\\sum_i q_i = 0``. When ``False``, each unordered pair
            appears once and ``q_{ij}`` is scattered as ``+q`` to the
            source AND ``-q`` to the target, which is ~2× cheaper in
            MLP cost but requires the caller to guarantee a half list.
        charge_projection: When ``True``, apply a uniform per-graph
            shift so that ``\\sum_i q_i`` exactly equals the per-graph
            ``total_charge`` value passed to :meth:`forward` (or zero
            when no ``total_charge`` is supplied). The architectural
            antisymmetry already guarantees neutrality to numerical
            precision on a clean bidirectional graph; the projection
            is a belt-and-braces step for non-neutral systems and for
            half-list use where round-off can leak a small residual.

    Forward inputs (keyword-only):
        node_features: ``(N, node_dim)`` invariant per-atom features.
        edge_index: ``(E, 2)`` directed edges; ``[:, 0]`` source,
            ``[:, 1]`` target (matches CLAUDE.md edge convention).
        bond_dist: ``(E,)`` per-edge distances ``\\|r_{ij}\\|``.
        atom_batch: ``(N,)`` int graph membership of each atom.
        num_graphs: number of graphs in the batch.
        edge_features: optional ``(E, edge_dim)`` invariant edge
            features.
        total_charge: optional ``(B,)`` or ``(B, 1)`` per-graph net
            charge target. Required when ``charge_projection=True``
            and the system is non-neutral.

    Returns:
        Dict with keys

        * ``"atomic_charges"`` — ``(N,)`` per-atom charge ``q_i``.
        * ``"bond_charges"`` — ``(E,)`` per-edge ``q_{ij}`` before
          aggregation; useful for diagnostics and the ``L_BO``
          auxiliary loss in downstream Sonata training.
        * ``"charge_sum_pre_proj"`` — ``(B,)`` per-graph charge sum
          before any projection (the architectural sum; should be
          numerically ``\\approx 0`` for neutral targets even without
          projection).
        * ``"charge_sum_post_proj"`` — ``(B,)`` per-graph charge sum
          after projection (matches ``total_charge`` exactly when
          projection is enabled).
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int | None = None,
        hidden_dim: int = 64,
        full_neighbor_list: bool = True,
        charge_projection: bool = True,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim or 0
        self.full_neighbor_list = full_neighbor_list
        self.charge_projection = charge_projection

        in_dim = 2 * node_dim + 1 + self.edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _bond_charges(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        bond_dist: torch.Tensor,
        edge_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute per-edge ``q_{ij}`` via one batched antisymmetric MLP call.

        Forward and reverse-direction inputs are stacked into one
        ``(2E, in_dim)`` tensor and the MLP is invoked once; the
        antisymmetric subtraction is then a single ``forward - reverse``
        on the resulting ``(2E, 1)`` output. This avoids two separate
        smaller GEMM calls and keeps the kernel-launch count constant
        regardless of edge-list size.
        """
        src = edge_index[:, 0]
        tgt = edge_index[:, 1]
        h_src = node_features[src]
        h_tgt = node_features[tgt]
        dist = bond_dist.unsqueeze(-1)

        fwd_parts = [h_src, h_tgt, dist]
        rev_parts = [h_tgt, h_src, dist]
        if edge_features is not None:
            fwd_parts.append(edge_features)
            rev_parts.append(edge_features)

        stacked = torch.cat([torch.cat(fwd_parts, dim=-1), torch.cat(rev_parts, dim=-1)], dim=0)
        scores = self.mlp(stacked).squeeze(-1)
        n_edges = edge_index.shape[0]
        return scores[:n_edges] - scores[n_edges:]

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        bond_dist: torch.Tensor,
        atom_batch: torch.Tensor,
        num_graphs: int,
        edge_features: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute charge-conserving per-atom partial charges.

        Builds antisymmetric bond charges ``q_ij`` and scatters them onto
        atoms so each neutral graph sums to zero by construction; an optional
        projection shifts the per-graph sum to ``total_charge``.

        Args:
            node_features: Per-atom features ``(N, F)``.
            edge_index: Source/target atom pairs ``(E, 2)``.
            bond_dist: Edge distances ``(E,)``.
            atom_batch: Graph membership per atom ``(N,)``.
            num_graphs: Number of graphs in the batch.
            edge_features: Optional per-edge features ``(E, F_e)``.
            total_charge: Optional per-graph net charge ``(num_graphs,)``;
                defaults to zero (neutral) when the projection is enabled.

        Returns:
            Dict with ``atomic_charges`` ``(N,)``, ``bond_charges`` ``(E,)``,
            and the per-graph charge sums before / after projection
            (``charge_sum_pre_proj`` / ``charge_sum_post_proj``, ``(num_graphs,)``).
        """
        n_atoms = node_features.shape[0]
        q_ij = self._bond_charges(node_features, edge_index, bond_dist, edge_features)

        src = edge_index[:, 0]
        charges = torch.zeros(n_atoms, dtype=q_ij.dtype, device=q_ij.device)
        charges.index_add_(0, src, q_ij)
        if not self.full_neighbor_list:
            tgt = edge_index[:, 1]
            charges.index_add_(0, tgt, -q_ij)

        pre = scatter_sum(charges, atom_batch, dim=0, dim_size=num_graphs)

        if self.charge_projection:
            if total_charge is None:
                target = torch.zeros_like(pre)
            else:
                target = total_charge.view_as(pre).to(dtype=charges.dtype)
            correction = (pre - target) / _graph_counts(atom_batch, num_graphs)
            charges = charges - correction[atom_batch]
            post = scatter_sum(charges, atom_batch, dim=0, dim_size=num_graphs)
        else:
            post = pre

        return {
            "atomic_charges": charges,
            "bond_charges": q_ij,
            "charge_sum_pre_proj": pre,
            "charge_sum_post_proj": post,
        }
