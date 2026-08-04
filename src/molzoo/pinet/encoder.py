"""PiNet feature encoder (representation only — no energy / force head)."""

from __future__ import annotations

from typing import Literal

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from molrep.embedding.cutoff import CosineCutoff, HalfCosineCutoff, TanhCutoff
from molrep.embedding.radial import GaussianBasis, PolynomialBasis
from molrep.interaction.pinet import GCBlock, ResUpdate

from .geometry import compute_d5, edge_bond_diff
from .spec import PiNetSpec


class PiNet(TensorDictModuleBase):
    """PiNet feature encoder.

    Inputs from the nested ``TensorDict`` schema:

    * ``("atoms", "Z")``: atomic numbers ``(N,)``.
    * ``("atoms", "pos")``: positions ``(N, 3)``.
    * ``("edges", "edge_index")``: source-target pairs ``(E, 2)``.

    Edge geometry is derived from ``pos`` + ``edge_index`` inside ``forward`` so
    force derivation can flow through positions.

    Writes in place:

    * ``("atoms", "node_features")``: scalar P1 states ``(N, depth, D)``.
    * ``("atoms", "p1_block_outputs")``: raw block outputs ``(N, depth, D)``.
    * ``("atoms", "p3_features")`` / ``("atoms", "p5_features")`` when enabled.
    * ``("edges", "i1_features")`` (+ ``i3`` / ``i5`` when enabled).
    """

    in_keys = [
        ("atoms", "Z"),
        ("atoms", "pos"),
        ("edges", "edge_index"),
    ]
    out_keys = [("atoms", "node_features")]

    def __init__(
        self,
        *,
        atom_types: list[int] | None = None,
        r_max: float = 4.0,
        cutoff_type: Literal["f1", "f2", "hip"] = "f1",
        basis_type: Literal["polynomial", "gaussian"] = "polynomial",
        n_basis: int = 4,
        gamma: float | list[float] = 3.0,
        center: float | list[float] | None = None,
        pp_nodes: list[int] | None = None,
        pi_nodes: list[int] | None = None,
        ii_nodes: list[int] | None = None,
        depth: int = 4,
        activation: str = "tanh",
        weighted: bool = False,
        rank: Literal[1, 3, 5] = 3,
    ) -> None:
        super().__init__()
        self.config = PiNetSpec(
            atom_types=atom_types or [1, 6, 7, 8],
            r_max=r_max,
            cutoff_type=cutoff_type,
            basis_type=basis_type,
            n_basis=n_basis,
            gamma=gamma,
            center=center,
            pp_nodes=pp_nodes or [16, 16],
            pi_nodes=pi_nodes or [16, 16],
            ii_nodes=ii_nodes or [16, 16],
            depth=depth,
            activation=activation,
            weighted=weighted,
            rank=rank,
        )
        cfg = self.config
        if cfg.pp_nodes[-1] != cfg.ii_nodes[-1]:
            raise ValueError("PiNet requires pp_nodes[-1] == ii_nodes[-1].")

        self.rank = int(cfg.rank)
        self.depth = int(cfg.depth)
        self.feature_dim = int(cfg.ii_nodes[-1])
        self.n_props = int(self.rank // 2) + 1
        self.output_dim = self.feature_dim
        self.edge_output_dim = self.feature_dim * self.n_props

        self.n_elem = len(cfg.atom_types)
        self.register_buffer(
            "_atom_types",
            torch.tensor(cfg.atom_types, dtype=torch.long),
            persistent=False,
        )
        self._atom_types: torch.Tensor

        z_to_idx = torch.zeros(128, dtype=torch.long)
        for i, z in enumerate(cfg.atom_types):
            z_to_idx[int(z)] = i
        self.register_buffer("_z_to_idx", z_to_idx, persistent=False)
        self._z_to_idx: torch.Tensor

        _cutoff_cls = {"f1": CosineCutoff, "f2": TanhCutoff, "hip": HalfCosineCutoff}
        self.cutoff = _cutoff_cls[cfg.cutoff_type](r_cut=cfg.r_max)

        if cfg.basis_type == "polynomial":
            self.basis_fn = PolynomialBasis(cfg.n_basis)
        else:
            self.basis_fn = GaussianBasis(
                center=cfg.center,
                gamma=cfg.gamma,
                r_cut=cfg.r_max,
                n_basis=cfg.n_basis,
            )

        # Channel schedules: p1 starts as one-hot (n_elem); p3/p5 start width 1.
        p1_dims = [self.n_elem] + [self.feature_dim] * self.depth
        p3_dims = [1] + [self.feature_dim] * self.depth
        p5_dims = [1] + [self.feature_dim] * self.depth

        self.gc_blocks = torch.nn.ModuleList(
            [
                GCBlock(
                    rank=self.rank,
                    weighted=cfg.weighted,
                    pp_nodes=cfg.pp_nodes,
                    pi_nodes=cfg.pi_nodes,
                    ii_nodes=cfg.ii_nodes,
                    n_basis=cfg.n_basis,
                    p1_in_dim=p1_dims[i],
                    p3_in_dim=p3_dims[i],
                    p5_in_dim=p5_dims[i],
                    activation=cfg.activation,
                )
                for i in range(self.depth)
            ]
        )

        self.res_update1 = torch.nn.ModuleList(
            [ResUpdate(in_dim=p1_dims[i], out_dim=p1_dims[i + 1]) for i in range(self.depth)]
        )
        if self.rank >= 3:
            self.res_update3 = torch.nn.ModuleList(
                [ResUpdate(in_dim=p3_dims[i], out_dim=p3_dims[i + 1]) for i in range(self.depth)]
            )
        if self.rank >= 5:
            self.res_update5 = torch.nn.ModuleList(
                [ResUpdate(in_dim=p5_dims[i], out_dim=p5_dims[i + 1]) for i in range(self.depth)]
            )

    @classmethod
    def from_spec(cls, spec: PiNetSpec) -> "PiNet":
        """Build a :class:`PiNet` from a validated :class:`PiNetSpec` config."""
        return cls(**spec.model_dump())

    def forward(self, td: TensorDict) -> TensorDict:
        """Encode a batch, writing per-layer node features back into ``td``."""
        Z = td["atoms", "Z"]
        pos = td["atoms", "pos"]
        # Contiguous layout for AOTInductor stride specialisation (pair_style).
        edge_index = td["edges", "edge_index"].contiguous()

        edge_diff = edge_bond_diff(td["edges"], pos, edge_index)
        edge_dist = edge_diff.norm(dim=-1).clamp(min=1e-8)

        idx = self._z_to_idx[Z]
        p1 = torch.nn.functional.one_hot(idx, num_classes=self.n_elem).to(pos.dtype)
        tensors: dict[str, torch.Tensor] = {"edge_index": edge_index, "p1": p1}

        d3 = edge_diff / edge_dist.unsqueeze(-1)
        tensors["d3"] = d3
        if self.rank >= 3:
            tensors["p3"] = torch.zeros(
                Z.shape[0],
                3,
                1,
                dtype=edge_diff.dtype,
                device=edge_diff.device,
            )
        if self.rank >= 5:
            tensors["p5"] = torch.zeros(
                Z.shape[0],
                5,
                1,
                dtype=edge_diff.dtype,
                device=edge_diff.device,
            )
            tensors["d5"] = compute_d5(d3)

        fc = self.cutoff(edge_dist)
        basis = self.basis_fn(edge_dist, fc=fc)

        p1_states: list[torch.Tensor] = []
        p1_block_outputs: list[torch.Tensor] = []
        p3_states: list[torch.Tensor] = []
        p5_states: list[torch.Tensor] = []
        i1_states: list[torch.Tensor] = []
        i3_states: list[torch.Tensor] = []
        i5_states: list[torch.Tensor] = []

        for i, block in enumerate(self.gc_blocks):
            new = block(tensors, basis)
            p1_block_outputs.append(new["p1"])
            tensors["p1"] = self.res_update1[i](tensors["p1"], new["p1"])
            p1_states.append(tensors["p1"])
            i1_states.append(new["i1"])

            if self.rank >= 3:
                tensors["p3"] = self.res_update3[i](tensors["p3"], new["p3"])
                p3_states.append(tensors["p3"])
                i3_states.append(new["i3"])

            if self.rank >= 5:
                tensors["p5"] = self.res_update5[i](tensors["p5"], new["p5"])
                p5_states.append(tensors["p5"])
                i5_states.append(new["i5"])

        td["atoms", "node_features"] = torch.stack(p1_states, dim=1)
        td["atoms", "p1_block_outputs"] = torch.stack(p1_block_outputs, dim=1)
        td["edges", "i1_features"] = torch.stack(i1_states, dim=1)
        if self.rank >= 3:
            td["atoms", "p3_features"] = torch.stack(p3_states, dim=1)
            td["edges", "i3_features"] = torch.stack(i3_states, dim=1)
        if self.rank >= 5:
            td["atoms", "p5_features"] = torch.stack(p5_states, dim=1)
            td["edges", "i5_features"] = torch.stack(i5_states, dim=1)
        return td
