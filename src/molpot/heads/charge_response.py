"""Charge-response / polarizability head (PiNN-style).

Computes a charge-response kernel χ from pooled scalar/edge features and
derives the molecular polarizability ``α = -r·χ·r^T``. Variants:
``localchi`` / ``local`` / ``etainv`` / ``eem`` / ``acks2``. Encoder-agnostic.

Distinct from :class:`molpot.heads.electrostatics.PolarizabilityHead`, which
targets Sonata's LES (Linear Electrostatic Response) branch — that one is an
equivariant l=0/l=2 readout, not a charge-response kernel.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from molix import config

ANG2BOHR = 1.8897259886


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    if src.numel() == 0:
        return out
    expand_index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, expand_index, src)


def _relative_atom_indices(
    batch: torch.Tensor, num_graphs: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    counts = torch.zeros(num_graphs, dtype=torch.long, device=batch.device)
    counts.scatter_add_(0, batch, torch.ones_like(batch))
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    rel = torch.arange(batch.shape[0], device=batch.device) - offsets[batch]
    nmax = int(counts.max().item()) if counts.numel() else 0
    return rel, counts, nmax


def _dense_positions(
    pos: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rel, counts, nmax = _relative_atom_indices(batch, num_graphs)
    dense = torch.zeros(num_graphs, nmax, 3, dtype=pos.dtype, device=pos.device)
    dense[batch, rel] = pos
    return dense, counts


class ChargeResponseHead(nn.Module):
    """Charge-response / polarizability head (PiNN-style).

    Variants:

    * ``"localchi"`` — symmetric local charge-response kernel.
    * ``"local"``    — local polarizability basis with localchi kernel.
    * ``"etainv"``   — direct positive ``eta^{-1}`` construction.
    * ``"eem"``      — electronegativity-equalization kernel.
    * ``"acks2"``    — ACKS2 Dyson update.

    Args:
        node_scalar_dim: Pooled-node scalar dim (used by ``atom_diag_mlp``
            and optional ``iso_mlp``).
        edge_scalar_dim: Pooled-edge scalar dim (used by
            ``edge_scalar_mlp``).
        edge_vector_dim: Pooled-edge 3-vector inner dim (used by
            ``edge_vector_mlp`` on ``i3.square().sum(dim=1)``).
        atom_types: Element list backing the per-element ``sigma`` table.
        sigma: Optional override of the default per-element sigma map.
        hidden_dim: Hidden dim of MLPs.
        variant: One of ``"localchi" / "local" / "etainv" / "eem" / "acks2"``.
        iso: Add an isotropic atomic-polarizability term on top.
        epsilon: Regularisation for the ``etainv`` construction.
    """

    def __init__(
        self,
        *,
        node_scalar_dim: int,
        edge_scalar_dim: int,
        edge_vector_dim: int,
        atom_types: list[int] | None = None,
        variant: str = "localchi",
        iso: bool = False,
        hidden_dim: int = 64,
        epsilon: float = 0.01,
        sigma: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant.endswith("_iso"):
            variant = variant[:-4]
            iso = True
        if variant not in {"localchi", "local", "etainv", "eem", "acks2"}:
            raise ValueError(f"Unsupported polarizability variant {variant!r}.")
        atom_types = atom_types or [1, 6, 7, 8]

        self.variant = variant
        self.iso = iso
        self.epsilon = float(epsilon)

        self.atom_diag_mlp = nn.Sequential(
            nn.Linear(node_scalar_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_scalar_mlp = nn.Sequential(
            nn.Linear(edge_scalar_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_vector_mlp = nn.Linear(edge_vector_dim, 1, dtype=config.ftype)
        if self.iso:
            self.iso_mlp = nn.Sequential(
                nn.Linear(node_scalar_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        _default_sigma = {1: 0.312, 6: 0.730, 7: 0.709, 8: 0.661, 16: 1.048, 17: 1.016}
        sigma = sigma or _default_sigma
        init_sigma = [float(sigma.get(z, 0.7)) for z in atom_types]
        self.register_buffer(
            "atom_types",
            torch.tensor(atom_types, dtype=torch.long),
            persistent=False,
        )
        self.atom_types: torch.Tensor
        self.sigma_raw = nn.Parameter(torch.tensor(init_sigma, dtype=config.ftype))

    # -- public forward -------------------------------------------------------

    def forward(
        self,
        *,
        pos: torch.Tensor,
        Z: torch.Tensor,
        atom_batch: torch.Tensor,
        num_graphs: int,
        edge_index: torch.Tensor,
        bond_diff: torch.Tensor,
        node_scalars: torch.Tensor,
        edge_scalars: torch.Tensor,
        edge_vectors: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict the charge-response parameters consumed by the electrostatics solve.

        Args:
            pos: Atom positions ``(N, 3)``.
            Z: Atomic numbers ``(N,)``.
            atom_batch: Graph membership per atom ``(N,)``.
            num_graphs: Number of graphs in the batch.
            edge_index: Source/target atom pairs ``(E, 2)``.
            bond_diff: Edge displacement vectors ``(E, 3)``.
            node_scalars: Per-atom scalar features ``(N, F_n)``.
            edge_scalars: Per-edge scalar features ``(E, F_e)``.
            edge_vectors: Optional per-edge vector features ``(E, 3)``.

        Returns:
            Dict of response parameters whose keys depend on ``variant`` —
            e.g. ``alpha`` (hardness), ``chi`` (electronegativity),
            ``atom_mask``, ``atom_diag``, and ``edge_response``.
        """
        atom_diag = self.atom_diag_mlp(node_scalars).squeeze(-1)
        edge_response = self.edge_scalar_mlp(edge_scalars).squeeze(-1)
        if edge_vectors is not None:
            edge_response = edge_response + self.edge_vector_mlp(
                edge_vectors.square().sum(dim=1),
            ).squeeze(-1)

        rel, counts, nmax = _relative_atom_indices(atom_batch, num_graphs)
        atom_mask = torch.arange(nmax, device=pos.device).unsqueeze(0) < counts.unsqueeze(1)
        sigma = self._sigma_for_atoms(Z)

        extra: dict[str, torch.Tensor] = {}

        if self.variant == "local":
            chi = self._make_local_chi(edge_index, edge_response, atom_batch, rel, counts, nmax)
            alpha = self._local_alpha(edge_index, bond_diff, edge_response, atom_batch, num_graphs)
        elif self.variant == "localchi":
            chi = self._make_local_chi(edge_index, edge_response, atom_batch, rel, counts, nmax)
            alpha = self._chi_to_alpha(pos, atom_batch, num_graphs, chi)
        elif self.variant == "etainv":
            chi, eta_inv = self._make_etainv(
                atom_diag,
                edge_index,
                edge_response,
                atom_batch,
                rel,
                counts,
                nmax,
            )
            alpha = self._chi_to_alpha(pos, atom_batch, num_graphs, chi)
            extra["eta_inv"] = eta_inv
        elif self.variant == "eem":
            chi, eta = self._make_eem(atom_diag, sigma, pos, atom_batch, num_graphs, counts, nmax)
            alpha = self._chi_to_alpha(pos, atom_batch, num_graphs, chi)
            extra["eta"] = eta
        else:  # acks2
            chi_s = self._make_local_chi(edge_index, edge_response, atom_batch, rel, counts, nmax)
            eta = self._make_eta(atom_diag, sigma, pos, atom_batch, num_graphs, counts, nmax)
            eye = torch.eye(nmax, dtype=eta.dtype, device=eta.device).expand_as(eta)
            system = eye - eta @ chi_s
            chi = torch.linalg.solve(system.transpose(-1, -2), chi_s)
            alpha = self._chi_to_alpha(pos, atom_batch, num_graphs, chi)
            extra["eta_e"] = eta
            extra["chi_s"] = chi_s

        if self.iso:
            alpha_iso_atom = F.softplus(self.iso_mlp(node_scalars).squeeze(-1))
            alpha_iso = _scatter_sum(alpha_iso_atom, atom_batch, num_graphs)
            eye3 = torch.eye(3, dtype=alpha.dtype, device=alpha.device)
            alpha_iso_tensor = alpha_iso[:, None, None] * eye3[None]
            alpha = alpha + alpha_iso_tensor
            extra["alpha_iso"] = alpha_iso_tensor

        return {
            "alpha": alpha,
            "chi": chi,
            "atom_mask": atom_mask,
            "atom_diag": atom_diag,
            "edge_response": edge_response,
            **extra,
        }

    # -- internal helpers -----------------------------------------------------

    def _sigma_for_atoms(self, Z: torch.Tensor) -> torch.Tensor:
        matches = Z.long().unsqueeze(-1) == self.atom_types.unsqueeze(0)
        sigma_table = self.sigma_raw.abs().clamp(min=1e-8)
        sigma = matches.to(dtype=sigma_table.dtype) @ sigma_table
        fallback = sigma_table.mean()
        return torch.where(matches.any(dim=1), sigma, fallback.expand_as(sigma))

    @staticmethod
    def _make_local_chi(edge_index, edge_response, atom_batch, rel, counts, nmax):
        num_graphs = counts.shape[0]
        chi = torch.zeros(
            num_graphs,
            nmax,
            nmax,
            dtype=edge_response.dtype,
            device=edge_response.device,
        )
        if edge_index.numel() == 0:
            return chi
        src, dst = edge_index[:, 0], edge_index[:, 1]
        b = atom_batch[src]
        ri, rj = rel[src], rel[dst]
        y = edge_response.abs()
        chi.index_put_((b, ri, rj), y, accumulate=True)
        chi.index_put_((b, rj, ri), y, accumulate=True)
        diag = -chi.sum(dim=1)
        idx = torch.arange(nmax, device=chi.device)
        chi[:, idx, idx] = diag
        return chi

    def _make_etainv(self, atom_diag, edge_index, edge_response, atom_batch, rel, counts, nmax):
        num = counts.shape[0]
        m = torch.zeros(num, nmax, nmax, dtype=atom_diag.dtype, device=atom_diag.device)
        idx = torch.arange(nmax, device=atom_diag.device)
        diag_dense = torch.zeros(num, nmax, dtype=atom_diag.dtype, device=atom_diag.device)
        diag_dense[atom_batch, rel] = atom_diag
        m[:, idx, idx] = diag_dense
        if edge_index.numel() > 0:
            src, dst = edge_index[:, 0], edge_index[:, 1]
            b = atom_batch[src]
            m.index_put_((b, rel[src], rel[dst]), edge_response, accumulate=True)
        eta_inv = m.transpose(-1, -2) @ m
        active_eye = torch.diag_embed(
            (idx.unsqueeze(0) < counts.unsqueeze(1)).to(dtype=atom_diag.dtype)
        )
        eta_inv = eta_inv + self.epsilon * active_eye
        chi = self._make_lrf(eta_inv)
        return chi, eta_inv

    def _make_eem(self, atom_diag, sigma, pos, atom_batch, num_graphs, counts, nmax):
        eta = self._make_eta(atom_diag, sigma, pos, atom_batch, num_graphs, counts, nmax)
        chi_blocks = []
        for b_idx, n in enumerate(counts.tolist()):
            block = eta[b_idx, :n, :n]
            inv = torch.linalg.inv(block)
            chi_blocks.append(self._make_lrf(inv.unsqueeze(0))[0])
        padded_chi = torch.zeros_like(eta)
        for b_idx, chi_b in enumerate(chi_blocks):
            n = chi_b.shape[0]
            padded_chi[b_idx, :n, :n] = chi_b
        return padded_chi, eta

    def _make_eta(self, atom_diag, sigma, pos, atom_batch, num_graphs, counts, nmax):
        dense_pos, _ = _dense_positions(pos, atom_batch, num_graphs)
        dense_sigma = torch.zeros(num_graphs, nmax, dtype=sigma.dtype, device=sigma.device)
        rel, _, _ = _relative_atom_indices(atom_batch, num_graphs)
        dense_sigma[atom_batch, rel] = sigma
        r_ij = dense_pos[:, None, :, :] - dense_pos[:, :, None, :]
        r = r_ij.norm(dim=-1)
        gamma = torch.sqrt(dense_sigma[:, None, :].square() + dense_sigma[:, :, None].square())
        eta = torch.special.erf(r / gamma.clamp(min=1e-8) / math.sqrt(2.0)) / r.clamp(min=1e-8)
        eta = eta / ANG2BOHR

        idx = torch.arange(nmax, device=pos.device)
        diag = torch.zeros(num_graphs, nmax, dtype=pos.dtype, device=pos.device)
        diag[atom_batch, rel] = atom_diag.abs()
        sigma_diag = 1.0 / (math.sqrt(math.pi) * dense_sigma.clamp(min=1e-8)) / ANG2BOHR
        diag = diag + sigma_diag
        eta[:, idx, idx] = diag
        active = idx.unsqueeze(0) < counts.unsqueeze(1)
        eta = eta * (active[:, :, None] & active[:, None, :]).to(dtype=eta.dtype)
        return eta

    @staticmethod
    def _make_lrf(ainv: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
        rows = ainv.sum(dim=-1)
        outer = rows.unsqueeze(-1) * rows.unsqueeze(-2)
        denom = rows.sum(dim=-1).clamp(min=eps)
        return -ainv + outer / denom[:, None, None]

    def _chi_to_alpha(self, pos, atom_batch, num_graphs, chi):
        dense_pos, _ = _dense_positions(pos, atom_batch, num_graphs)
        dense_pos = dense_pos * ANG2BOHR
        return -torch.einsum("bix,bij,bjy->bxy", dense_pos, chi, dense_pos)

    def _local_alpha(self, edge_index, bond_diff, edge_response, atom_batch, num_graphs):
        pos_dtype = bond_diff.dtype
        if edge_index.numel() == 0:
            return torch.zeros(num_graphs, 3, 3, dtype=pos_dtype, device=bond_diff.device)
        src = edge_index[:, 0]
        edge_vec = bond_diff * ANG2BOHR
        weighted = edge_response.unsqueeze(-1).unsqueeze(-1).abs() * (
            edge_vec.unsqueeze(-1) * edge_vec.unsqueeze(-2)
        )
        edge_batch = atom_batch[src]
        return _scatter_sum(weighted, edge_batch, num_graphs)
