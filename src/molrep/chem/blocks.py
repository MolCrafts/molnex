"""Covalent-graph message-passing blocks for chemical perception.

One :class:`ChemGraphBlock` is one hop on the undirected covalent graph.
Blocks never read coordinates; features are dimensionless learned embeddings.

Reference:
    Wang et al., "Espaloma", arXiv:2010.01196 (2022) §5
    https://arxiv.org/abs/2010.01196
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molix import config
from molrep.chem.context import BondContext


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer SiLU MLP baked to project float dtype."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, dtype=config.ftype),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim, dtype=config.ftype),
    )


class ChemGraphBlock(nn.Module):
    """One covalent-graph hop: update ``h_bond`` then residual-update ``h_atom``.

    Bond features stay endpoint-symmetric via a private :class:`BondContext`.
    Each undirected ``h_bond`` is scattered onto both endpoints. With
    ``local_attention=True``, a shared per-bond score is softmax-normalized
    over each atom's incident covalent bonds only (not over all ``E``).

    Args:
        atom_dim: Atom feature dimension ``D_a``.
        bond_dim: Bond feature dimension ``D_b``.
        hidden_dim: Hidden width of the atom residual MLP. Defaults to
            ``atom_dim``. Also forwarded to the private :class:`BondContext`.
        local_attention: If ``True``, neighborhood softmax over covalent
            incident bonds. Default ``False`` (plain ``index_add_`` sum).
        num_bond_types: Bond-type table size; ``0`` disables type
            conditioning.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int | None = None,
        local_attention: bool = False,
        num_bond_types: int = 0,
    ) -> None:
        super().__init__()
        if atom_dim <= 0 or bond_dim <= 0:
            raise ValueError("atom_dim and bond_dim must be positive")
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.local_attention = bool(local_attention)
        hid = int(hidden_dim) if hidden_dim is not None else self.atom_dim
        self.bond_context = BondContext(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            hidden_dim=hidden_dim,
            num_bond_types=num_bond_types,
        )
        self.atom_mlp = _mlp(self.bond_dim, hid, self.atom_dim)
        if self.local_attention:
            self.score = nn.Linear(self.bond_dim, 1, dtype=config.ftype)
        else:
            self.score = None

    def forward(
        self,
        h_atom: torch.Tensor,
        h_bond: torch.Tensor,
        atomi: torch.Tensor,
        atomj: torch.Tensor,
        bond_type: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one covalent hop.

        Args:
            h_atom: Atom features ``(N, atom_dim)``.
            h_bond: Undirected bond features ``(E, bond_dim)``.
            atomi: Endpoint i indices ``(E,)``.
            atomj: Endpoint j indices ``(E,)``.
            bond_type: Optional bond types ``(E,)``.
            mask: Optional live-bond mask ``(E,)``. Dummy valence rows
                (``MoleculePadder``) must not scatter onto ghost atoms.

        Returns:
            Updated ``(h_atom, h_bond)`` with the same shapes. Does not
            read positions.
        """
        h_bond = self.bond_context(
            h_atom,
            atomi,
            atomj,
            bond_type=bond_type,
            h_bond=h_bond,
        )
        if mask is not None:
            h_bond = h_bond * mask.to(dtype=h_bond.dtype).unsqueeze(-1)
        src = atomi.long()
        dst = atomj.long()
        n_atoms = h_atom.shape[0]
        if self.score is None:
            agg = h_atom.new_zeros(n_atoms, self.bond_dim)
            agg.index_add_(0, src, h_bond)
            agg.index_add_(0, dst, h_bond)
        else:
            # Shared undirected score; softmax is per-atom over incident bonds.
            logits = self.score(h_bond).squeeze(-1)
            inc_atoms = torch.cat([src, dst], dim=0)
            inc_logits = torch.cat([logits, logits], dim=0)
            inc_feats = torch.cat([h_bond, h_bond], dim=0)
            max_logits = inc_logits.new_zeros(n_atoms)
            max_logits.scatter_reduce_(0, inc_atoms, inc_logits, reduce="amax", include_self=False)
            exp = (inc_logits - max_logits[inc_atoms]).exp()
            sum_exp = inc_logits.new_zeros(n_atoms)
            sum_exp.index_add_(0, inc_atoms, exp)
            weights = exp / sum_exp[inc_atoms].clamp_min(1e-12)
            agg = h_atom.new_zeros(n_atoms, self.bond_dim)
            agg.index_add_(0, inc_atoms, inc_feats * weights.unsqueeze(-1))
        return h_atom + self.atom_mlp(agg), h_bond
