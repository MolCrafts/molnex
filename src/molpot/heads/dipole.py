"""Generic molecular-dipole head.

Composes scalar atomic charges (AC), vector atomic dipoles (AD), edge
bond-charge dipoles (BC), and oxidation-state charges (OS) into a graph
dipole. Encoder-agnostic: the caller is responsible for pooling encoder
features to the expected shapes and passing them in as kwargs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molix import config
from molix.F.scatter import scatter_sum
from molpot.heads._common import graph_counts as _graph_counts


def _charge_neutralize(
    charges: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    *,
    total_charge: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = scatter_sum(charges, batch, dim_size=num_graphs)
    if total_charge is None:
        total_charge = torch.zeros_like(pre)
    correction = (pre - total_charge.view_as(pre)) / _graph_counts(batch, num_graphs)
    charges = charges - correction[batch]
    post = scatter_sum(charges, batch, dim_size=num_graphs)
    return charges, pre, post


class DipoleHead(nn.Module):
    """Multi-variant molecular dipole head.

    Variants are composed from atomic-charge / atomic-dipole / bond-charge
    / oxidation-state terms:

    * ``"ac"``       — atomic-charge dipole ``Σ_i q_i r_i``.
    * ``"ad"``       — atomic dipole ``Σ_i μ_i`` from vector node features.
    * ``"bc"``       — bond-charge *dipole only* ``Σ_{ij} q_{ij} r_{ij}``;
      does **not** expose per-atom ``q_i`` and is **not** antisymmetric.
      Use :class:`molpot.heads.BondChargeHead` for charge-conserving
      per-atom ``q_i`` consumable by long-range electrostatics.
    * ``"ad_os"``    — atomic dipole + oxidation-state charge term.
    * Combinations: ``"ac_ad"``, ``"ac_bc"``, ``"ad_bc"``.

    Args:
        node_scalar_dim: Dim of pooled node scalars ``(N, D)`` (used by AC).
        node_vector_dim: Dim of pooled node 3-vectors ``(N, 3, D)`` (AD/OS).
        edge_scalar_dim: Dim of pooled edge scalars ``(E, D)`` (BC).
        edge_vector_dim: Dim of pooled edge 3-vectors ``(E, 3, D)`` (BC).
        hidden_dim: Hidden dim of AC / BC scalar MLPs.
        variant: One of ``"ac"``, ``"ad"``, ``"bc"``, ``"ad_os"``,
            ``"ac_ad"``, ``"ac_bc"``, ``"ad_bc"``.
        vector_dipole: Emit ``"dipole"`` as a vector ``(B, 3)``; otherwise
            emit its norm ``(B,)``.
        charge_neutrality: Apply per-graph charge neutralisation to the AC
            charges before assembling the dipole.
        regularization: Emit a scalar ``"bond_charge_l2"`` for BC variants.
    """

    def __init__(
        self,
        *,
        node_scalar_dim: int,
        node_vector_dim: int | None = None,
        edge_scalar_dim: int | None = None,
        edge_vector_dim: int | None = None,
        hidden_dim: int = 64,
        variant: str = "ac_ad",
        vector_dipole: bool = True,
        charge_neutrality: bool = True,
        regularization: bool = True,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        self.variant = variant
        self.vector_dipole = vector_dipole
        self.charge_neutrality = charge_neutrality
        self.regularization = regularization

        self.uses_ac = "ac" in variant
        self.uses_ad = "ad" in variant
        self.uses_bc = "bc" in variant
        self.uses_os = variant == "ad_os"
        if not (self.uses_ac or self.uses_ad or self.uses_bc or self.uses_os):
            raise ValueError(f"Unsupported dipole variant {variant!r}.")

        if self.uses_ac:
            self.charge_mlp = nn.Sequential(
                nn.Linear(node_scalar_dim, hidden_dim, dtype=config.ftype),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, dtype=config.ftype),
            )
        if self.uses_ad or self.uses_os:
            if node_vector_dim is None:
                raise ValueError("node_vector_dim required for AD / OS variants.")
            self.atomic_dipole_gate = nn.Linear(
                node_vector_dim,
                1,
                bias=False,
                dtype=config.ftype,
            )
        if self.uses_bc:
            if edge_scalar_dim is None or edge_vector_dim is None:
                raise ValueError("edge_scalar_dim and edge_vector_dim required for BC variant.")
            self.bond_scalar_mlp = nn.Sequential(
                nn.Linear(edge_scalar_dim, hidden_dim, dtype=config.ftype),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, dtype=config.ftype),
            )
            self.bond_vector_mlp = nn.Linear(
                edge_vector_dim,
                1,
                bias=False,
                dtype=config.ftype,
            )

    def forward(
        self,
        *,
        pos: torch.Tensor,
        atom_batch: torch.Tensor,
        num_graphs: int,
        node_scalars: torch.Tensor | None = None,
        node_vectors: torch.Tensor | None = None,
        edge_scalars: torch.Tensor | None = None,
        edge_vectors: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_diff: torch.Tensor | None = None,
        oxidation: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Assemble the molecular dipole from the configured ``mode`` terms.

        Each enabled term (atomic-charge ``ac``, atomic-dipole ``ad``,
        bond-charge ``bc``, oxidation-state ``ad_os``) contributes to the
        per-graph dipole; the required inputs depend on which terms are active.

        Args:
            pos: Atom positions ``(N, 3)``.
            atom_batch: Graph membership per atom ``(N,)``.
            num_graphs: Number of graphs in the batch.
            node_scalars: Per-atom scalar features ``(N, F_n)`` (charge terms).
            node_vectors: Per-atom vector features ``(N, 3)`` (atomic-dipole term).
            edge_scalars: Per-edge scalar features ``(E, F_e)`` (bond-charge term).
            edge_vectors: Per-edge vector features ``(E, 3)``.
            edge_index: Source/target atom pairs ``(E, 2)`` (bond terms).
            edge_diff: Edge displacement vectors ``(E, 3)`` (bond terms).
            oxidation: Per-atom oxidation states ``(N,)`` (``ad_os`` term).
            total_charge: Optional per-graph net charge ``(num_graphs,)`` for
                the charge-neutrality projection.

        Returns:
            Dict containing at least ``dipole`` and ``molecular_dipole``
            ``(num_graphs, 3)`` plus the intermediate per-term tensors
            (e.g. ``atomic_charges``, ``atomic_dipoles``, ``bond_charges``)
            produced by the enabled terms.
        """
        dipole = torch.zeros(num_graphs, 3, dtype=pos.dtype, device=pos.device)
        out: dict[str, torch.Tensor] = {}

        if self.uses_ac:
            assert node_scalars is not None
            charges = self.charge_mlp(node_scalars).squeeze(-1)
            if self.charge_neutrality:
                charges, pre, post = _charge_neutralize(
                    charges,
                    atom_batch,
                    num_graphs,
                    total_charge=total_charge,
                )
                out["charge_sum_pre_proj"] = pre
                out["charge_sum_post_proj"] = post
            out["atomic_charges"] = charges
            dipole = dipole + scatter_sum(
                charges.unsqueeze(-1) * pos,
                atom_batch,
                dim_size=num_graphs,
            )

        if self.uses_os:
            assert oxidation is not None
            ox = oxidation.to(dtype=pos.dtype)
            out["oxidation_charges"] = ox
            dipole = dipole + scatter_sum(
                ox.unsqueeze(-1) * pos,
                atom_batch,
                dim_size=num_graphs,
            )

        if self.uses_ad or self.uses_os:
            assert node_vectors is not None
            atomic_dipoles = self.atomic_dipole_gate(node_vectors).squeeze(-1)
            out["atomic_dipoles"] = atomic_dipoles
            dipole = dipole + scatter_sum(atomic_dipoles, atom_batch, dim_size=num_graphs)

        if self.uses_bc:
            assert edge_scalars is not None and edge_index is not None
            assert edge_diff is not None
            edge_batch = atom_batch[edge_index[:, 0]]
            bond_charge = self.bond_scalar_mlp(edge_scalars).squeeze(-1)
            if edge_vectors is not None:
                bond_charge = bond_charge + self.bond_vector_mlp(
                    edge_vectors.square().sum(dim=1),
                ).squeeze(-1)
            out["bond_charges"] = bond_charge
            bond_dipoles = bond_charge.unsqueeze(-1) * edge_diff
            out["bond_dipoles"] = bond_dipoles
            dipole = dipole + scatter_sum(bond_dipoles, edge_batch, dim_size=num_graphs)
            if self.regularization:
                out["bond_charge_l2"] = bond_charge.square().mean()

        out["molecular_dipole"] = dipole
        if self.vector_dipole:
            out["dipole"] = dipole
        else:
            out["dipole"] = torch.sqrt(dipole.square().sum(dim=-1) + 1e-6)
        return out
