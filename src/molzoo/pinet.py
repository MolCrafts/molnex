"""PiNet encoder + potential head.

PiNet is a multi-rank representation architecture (P1 scalar, P3 vector, P5
rank-5). The encoder produces ``(N, layers, features)`` node features; the
PiNet-specific head pools across layers, predicts per-atom energies, and
derives forces via autograd. The head lives alongside the encoder (not under
``molpot``) because the pool-then-readout pattern only makes sense for PiNet's
multi-layer output shape — there is no MACE/Allegro reuse to extract.

Dipole and polarizability readouts live in :mod:`molpot` (under
``pinet_dipole`` / ``pinet_polarizability``) because they pair the encoder
with task-specific tensor heads that don't compose with the energy head.

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems", JCTC 2025.
    https://doi.org/10.1021/acs.jctc.4c01570

    Reference implementation:
    https://github.com/Teoroo-CMC/PiNN/blob/master/pinn/networks/pinet2.py
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from molpot.derivation import EnergyAggregation, ForceDerivation
from molpot.heads import ChargeResponseHead, DipoleHead
from molrep.embedding.cutoff import CosineCutoff, HalfCosineCutoff, TanhCutoff
from molrep.embedding.radial import GaussianBasis, PolynomialBasis
from molrep.interaction.pinet import GCBlock, OutLayer, ResUpdate

__all__ = [
    "PiNet",
    "PiNetSpec",
    "PiNetPotential",
    "PiNetDipole",
    "PiNetPolarizability",
]


class PiNetSpec(BaseModel):
    """Configuration snapshot for :class:`PiNet`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    atom_types: list[int] = Field(default_factory=lambda: [1, 6, 7, 8], min_length=1)
    r_max: float = Field(default=4.0, gt=0.0)
    cutoff_type: Literal["f1", "f2", "hip"] = "f1"
    basis_type: Literal["polynomial", "gaussian"] = "polynomial"
    n_basis: int = Field(default=4, gt=0)
    gamma: float | list[float] = 3.0
    center: float | list[float] | None = None
    pp_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    pi_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    ii_nodes: list[int] = Field(default_factory=lambda: [16, 16], min_length=1)
    depth: int = Field(default=4, gt=0)
    activation: str = "tanh"
    weighted: bool = False
    rank: Literal[1, 3, 5] = 3


def _compute_d5(d3: torch.Tensor) -> torch.Tensor:
    """Five-component symmetric-traceless rank-5 direction basis."""
    x, y, z = d3[:, 0], d3[:, 1], d3[:, 2]
    x2, y2, z2 = x.square(), y.square(), z.square()
    return torch.stack(
        [
            (2.0 / 3.0) * x2 - (1.0 / 3.0) * y2 - (1.0 / 3.0) * z2,
            (2.0 / 3.0) * y2 - (1.0 / 3.0) * x2 - (1.0 / 3.0) * z2,
            x * y,
            x * z,
            y * z,
        ],
        dim=1,
    )


class PiNet(TensorDictModuleBase):
    """PiNet feature encoder.

    Inputs from the nested ``TensorDict`` schema:

    * ``("atoms", "Z")``: atomic numbers ``(N,)``.
    * ``("atoms", "pos")``: positions ``(N, 3)``.
    * ``("edges", "edge_index")``: source-target pairs ``(E, 2)``.

    Edge geometry (``bond_diff`` = ``pos[target] - pos[source]`` and
    ``bond_dist`` = ``‖bond_diff‖``) is derived from ``pos`` + ``edge_index``
    inside ``forward``. Caching them in the pipeline buys nothing (one
    gather + diff + norm per step) and breaks autograd-through-pos for
    force training.

    Writes in place:

    * ``("atoms", "node_features")``: scalar P1 *states* ``(N, depth, D)`` —
      residually-updated running states (used by feature-pooling heads).
    * ``("atoms", "p1_block_outputs")``: scalar P1 *raw block outputs*
      ``(N, depth, D)`` — each block's output before the ResUpdate, fed to the
      PiNet2 per-block OutLayer for residual energy accumulation.
    * ``("atoms", "p3_features")``: vector P3 states ``(N, depth, 3, D)``.
    * ``("atoms", "p5_features")``: rank-5 P5 states ``(N, depth, 5, D)``.
    * ``("edges", "i1_features")``: scalar pair interactions ``(E, depth, D * n_props)``.
    * ``("edges", "i3_features")`` / ``("edges", "i5_features")`` when enabled.
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

        # One-hot atom embedding (PiNN PiNet2: ``atom_types`` is the one-hot
        # basis; the scalar P1 track starts as a one-hot of dimension n_elem,
        # NOT a learnable embedding). The first GCBlock's PILayer/ResUpdate then
        # lift it to ``feature_dim``.
        self.n_elem = len(cfg.atom_types)
        self.register_buffer(
            "_atom_types",
            torch.tensor(cfg.atom_types, dtype=torch.long),
            persistent=False,
        )
        self._atom_types: torch.Tensor

        # Z -> embedding-row lookup table. Sized to cover every stable element
        # so a single gather replaces the per-forward Python loop. Unknown Z
        # falls back to row 0, matching the prior masked_fill_ default.
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

        self.gc_blocks = torch.nn.ModuleList(
            [
                GCBlock(
                    rank=self.rank,
                    weighted=cfg.weighted,
                    pp_nodes=cfg.pp_nodes,
                    pi_nodes=cfg.pi_nodes,
                    ii_nodes=cfg.ii_nodes,
                    n_basis=cfg.n_basis,
                    activation=cfg.activation,
                )
                for _ in range(self.depth)
            ]
        )

        # p1 starts as a one-hot of width ``n_elem``; every block thereafter
        # outputs ``feature_dim``. The first ResUpdate therefore projects
        # n_elem -> feature_dim (biaslessly), matching PiNN.
        p1_dims = [self.n_elem] + [self.feature_dim] * self.depth
        self.res_update1 = torch.nn.ModuleList(
            [ResUpdate(in_dim=p1_dims[i], out_dim=p1_dims[i + 1]) for i in range(self.depth)]
        )
        if self.rank >= 3:
            p3_dims = [1] + [self.feature_dim] * self.depth
            self.res_update3 = torch.nn.ModuleList(
                [ResUpdate(in_dim=p3_dims[i], out_dim=p3_dims[i + 1]) for i in range(self.depth)]
            )
        if self.rank >= 5:
            p5_dims = [1] + [self.feature_dim] * self.depth
            self.res_update5 = torch.nn.ModuleList(
                [ResUpdate(in_dim=p5_dims[i], out_dim=p5_dims[i + 1]) for i in range(self.depth)]
            )

    @classmethod
    def from_spec(cls, spec: PiNetSpec) -> "PiNet":
        """Build a :class:`PiNet` from a validated :class:`PiNetSpec` config."""
        return cls(**spec.model_dump())

    def forward(self, td: TensorDict) -> TensorDict:
        """Encode a batch, writing per-layer node features back into ``td``.

        Reads ``atoms.Z`` / ``atoms.pos`` / ``edges.edge_index``, derives edge
        geometry on the fly (so forces flow through ``pos``), runs the PiNet
        message-passing stack, and writes per-layer features
        ``(N, layers, features)`` under ``atoms.node_features``.

        Args:
            td: Post-collate :class:`~tensordict.TensorDict` batch.

        Returns:
            The same batch with ``atoms.node_features`` populated.
        """
        Z = td["atoms", "Z"]
        pos = td["atoms", "pos"]
        edge_index = td["edges", "edge_index"]

        # Edge geometry derived from pos + edge_index per forward.
        # One gather + diff + norm — cheaper than caching, and gradients flow
        # back to pos naturally so force training needs no special path.
        bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
        bond_dist = bond_diff.norm(dim=-1).clamp(min=1e-8)

        idx = self._z_to_idx[Z]
        p1 = torch.nn.functional.one_hot(idx, num_classes=self.n_elem).to(pos.dtype)
        tensors: dict[str, torch.Tensor] = {"edge_index": edge_index, "p1": p1}

        d3 = bond_diff / bond_dist.unsqueeze(-1)
        tensors["d3"] = d3
        if self.rank >= 3:
            tensors["p3"] = torch.zeros(
                Z.shape[0],
                3,
                1,
                dtype=bond_diff.dtype,
                device=bond_diff.device,
            )
        if self.rank >= 5:
            tensors["p5"] = torch.zeros(
                Z.shape[0],
                5,
                1,
                dtype=bond_diff.dtype,
                device=bond_diff.device,
            )
            tensors["d5"] = _compute_d5(d3)

        fc = self.cutoff(bond_dist)
        basis = self.basis_fn(bond_dist, fc=fc)

        p1_states: list[torch.Tensor] = []
        p1_block_outputs: list[torch.Tensor] = []
        p3_states: list[torch.Tensor] = []
        p5_states: list[torch.Tensor] = []
        i1_states: list[torch.Tensor] = []
        i3_states: list[torch.Tensor] = []
        i5_states: list[torch.Tensor] = []

        for i, block in enumerate(self.gc_blocks):
            new = block(tensors, basis)
            # Raw per-block scalar output (PiNN feeds THIS, not the res-updated
            # state, to its per-block OutLayer for residual energy accumulation).
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


# ---------------------------------------------------------------------------
# PiNet energy + force potential
# ---------------------------------------------------------------------------


def _pool_layer(features: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return features.mean(dim=1)
    if reduction == "sum":
        return features.sum(dim=1)
    if reduction == "last":
        return features[:, -1]
    raise ValueError(f"Unknown reduction {reduction!r}.")


class PiNetPotential(nn.Module):
    """PiNet energy + force prediction model.

    Pools the encoder's ``(N, layers, features)`` output across the layer
    axis, predicts per-atom energies via a 2-layer MLP, aggregates them to
    graph energies, and derives forces via ``torch.autograd.grad`` when
    ``compute_forces`` (or ``compute_forces_default``) is set.

    Per-element baseline subtraction (atomic dress) is handled at cache
    time by :class:`molix.data.AtomicDress` — labels are already dressed
    when they reach this model, so no runtime dress add is needed.

    Args:
        encoder: :class:`PiNet` (or any module that writes ``("atoms",
            "node_features")`` into a ``TensorDict``).
        hidden_dim: Hidden dimension of the per-atom energy MLP.
        layer_reduction: How to pool across GC-block layers
            (``"mean"`` / ``"sum"`` / ``"last"``).
        compute_forces: Default value for forward's ``compute_forces``
            kwarg. Set to ``True`` for force training so the Trainer's plain
            ``model(batch)`` call returns ``{"energy", "forces"}``.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        hidden_dim: int = 64,
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        compute_forces: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        # ``layer_reduction`` is accepted for API compatibility but unused by the
        # energy head: PiNet2 does NOT pool layers — it accumulates a per-block
        # OutLayer residually (see below), so there is no layer axis to reduce.
        self.layer_reduction = layer_reduction
        self.compute_forces_default = compute_forces
        # Set by :meth:`compile_energy`; when present, the energy forward runs
        # through the compiled callable instead of the eager method.
        self._compiled_energy_forward = None

        # Per-block output heads, residually accumulated (PiNN PiNet2 OutLayer):
        # per-atom energy = Σ_b OutLayer_b(p1_block_output_b), each OutLayer a
        # tanh FFLayer([hidden_dim]) -> biasless Linear(1). The absolute energy
        # zero-point is owned by the atomic dress, so the projection has no bias.
        depth: int = int(getattr(encoder, "depth", 1))
        self.out_layers = nn.ModuleList(
            [OutLayer([hidden_dim], out_units=1, activation="tanh") for _ in range(depth)]
        )
        self.energy_aggregation = EnergyAggregation(pooling="sum")
        self.force_derivation = ForceDerivation()

    def forward(
        self, batch: TensorDict, *, compute_forces: bool | None = None
    ) -> dict[str, torch.Tensor]:
        if compute_forces is None:
            compute_forces = self.compute_forces_default
        grad_ctx = torch.enable_grad() if compute_forces else nullcontext()
        energy_forward = self._compiled_energy_forward or self._energy_forward
        with grad_ctx:
            if compute_forces and not batch["atoms", "pos"].requires_grad:
                batch["atoms", "pos"] = batch["atoms", "pos"].clone().requires_grad_(True)
            out = energy_forward(batch)
            if compute_forces:
                out["forces"] = self.force_derivation(out["energy"], batch["atoms", "pos"])
        return out

    def _energy_forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Compilable energy computation: encoder -> per-block OutLayer sum -> aggregate.

        Extracted so ``torch.compile`` can wrap it without hitting the
        double-backward limitation that ``ForceDerivation``'s internal
        ``torch.autograd.grad`` triggers.

        Per-atom energy is the residual sum of per-block OutLayers applied to
        each GC block's raw scalar output ``p1_block_outputs`` ``(N, depth, D)``
        — the PiNet2 output mechanism (no layer pooling).
        """
        batch = self.encoder(batch)

        block_outputs = batch["atoms", "p1_block_outputs"]  # (N, depth, D)
        output = block_outputs.new_zeros(block_outputs.shape[0], 1)
        for i, out_layer in enumerate(self.out_layers):
            output = out_layer(block_outputs[:, i, :], output)
        atom_energy = output.squeeze(-1)

        atom_batch = batch["atoms", "batch"]
        num_graphs = batch["graphs"].batch_size[0]
        energy = self.energy_aggregation(atom_energy, atom_batch, num_graphs=num_graphs)

        return {
            "atomic_energy": atom_energy,
            "energy": energy,
        }

    def compile_energy(self, *, backend: str = "inductor", **kwargs) -> None:
        """Compile the whole energy forward (encoder → per-block OutLayer sum →
        aggregation) as one region. **Energy-only training / inference.**

        This is the launch-bound remedy for energy models: the energy forward
        fires hundreds of tiny kernels per step (the GC-block message-passing
        loop plus the per-block ``OutLayer`` residual loop), and
        ``torch.compile`` (inductor) fuses them, collapsing the CPU dispatch
        that leaves the GPU idle. Measured on a GH200 it cuts QM9 (energy-only)
        op-calls/step ~7.1k → ~2.8k and lifts throughput ~1.5× at bs32/bs128.

        **Does NOT work for force training.** When ``compute_forces=True`` the
        loss backprops through forces, and ``ForceDerivation`` builds its force
        with ``create_graph=True`` (a *double* backward).

        Two distinct failure modes, both fatal for *force training*:

        * ``backend="inductor"`` / ``mode="reduce-overhead"`` — backpropagating
          a second time through an ``aot_autograd``-compiled region is
          unsupported (``RuntimeError: ... does not currently support double
          backward``), so it raises loudly at the first training step.
        * ``backend="cudagraphs"`` — does **not** raise (it runs ~5.7× faster),
          but the double backward through the replayed graph produces
          **silently wrong parameter gradients**: GH200-verified, the forward
          energy/forces match eager to machine precision, yet the grads diverge
          ~4e-4 *relative* — and this gap persists in fp64 (10^10× the
          eager-vs-eager scatter_add noise floor), so it is a real correctness
          bug, not precision noise. Never use ``cudagraphs`` for force training:
          a model trained on it learns from corrupt gradients with no error.

        Force training must stay **eager**. The launch-bound remedy is a larger
        physical batch (bs64 ~tripled revMD17 EF throughput, dropping launch-
        bound 65 % → 42 %) or ``Trainer(accumulate_grad_batches=…)``.
        Force *inference* (``create_graph=False``, no double backward) compiles
        fine and is correct — reduce-overhead reached ~8.8× there.

        Materialise lazy parameters with one warm-up forward before calling
        this, so the compiled graph captures concrete shapes.

        Use the **default** inductor mode. ``mode="reduce-overhead"`` (CUDA
        graphs) is *counterproductive* here: molecular batches have variable
        atom/edge counts, and CUDA graphs require static shapes — measured on a
        GH200 it ran slower than eager (the per-step capture/guard overhead
        dominates, op-calls/step stay ~7.2k instead of dropping to ~2.8k).

        Args:
            backend: ``torch.compile`` backend (default ``"inductor"``).
            **kwargs: Forwarded to :func:`torch.compile`. Avoid
                ``mode="reduce-overhead"`` for ragged molecular batches.
        """
        self._compiled_energy_forward = torch.compile(
            self._energy_forward, backend=backend, **kwargs
        )


# ---------------------------------------------------------------------------
# PiNet + DipoleHead composed model
# ---------------------------------------------------------------------------


class PiNetDipole(nn.Module):
    """PiNet encoder paired with :class:`molpot.heads.DipoleHead`.

    The encoder writes scalar/vector tracks ``(N, layers, ...)`` into the
    batch; this wrapper pools across the layer axis and forwards the
    pre-pooled tensors to a generic dipole head. See
    :class:`molpot.heads.DipoleHead` for variant semantics.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        hidden_dim: int = 64,
        variant: str = "ac_ad",
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        vector_dipole: bool = True,
        charge_neutrality: bool = True,
        regularization: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layer_reduction = layer_reduction
        input_dim: int = getattr(encoder, "output_dim", 16)
        edge_dim: int = getattr(encoder, "edge_output_dim", input_dim)
        self.head = DipoleHead(
            node_scalar_dim=input_dim,
            node_vector_dim=input_dim,
            edge_scalar_dim=edge_dim,
            edge_vector_dim=input_dim,
            hidden_dim=hidden_dim,
            variant=variant,
            vector_dipole=vector_dipole,
            charge_neutrality=charge_neutrality,
            regularization=regularization,
        )

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        batch = self.encoder(batch)

        atom_batch = batch["atoms", "batch"]
        num_graphs = batch["graphs"].batch_size[0]
        node_scalars = _pool_layer(
            batch["atoms", "node_features"],
            self.layer_reduction,
        )
        node_vectors = None
        if "p3_features" in batch["atoms"].keys():
            node_vectors = _pool_layer(
                batch["atoms", "p3_features"],
                self.layer_reduction,
            )
        edge_scalars = None
        edge_index = None
        bond_diff = None
        if self.head.uses_bc and "i1_features" in batch["edges"].keys():
            edge_scalars = _pool_layer(
                batch["edges", "i1_features"],
                self.layer_reduction,
            )
            edge_index = batch["edges", "edge_index"]
            pos = batch["atoms", "pos"]
            bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
        edge_vectors = None
        if self.head.uses_bc and "i3_features" in batch["edges"].keys():
            edge_vectors = _pool_layer(
                batch["edges", "i3_features"],
                self.layer_reduction,
            )
        oxidation = None
        if self.head.uses_os and "oxidation" in batch["atoms"].keys():
            oxidation = batch["atoms", "oxidation"]
        total_charge = None
        if self.head.uses_ac and self.head.charge_neutrality:
            try:
                total_charge = batch["graphs", "total_charge"]
            except KeyError:
                pass
        return self.head(
            pos=batch["atoms", "pos"],
            atom_batch=atom_batch,
            num_graphs=num_graphs,
            node_scalars=node_scalars,
            node_vectors=node_vectors,
            edge_scalars=edge_scalars,
            edge_vectors=edge_vectors,
            edge_index=edge_index,
            bond_diff=bond_diff,
            oxidation=oxidation,
            total_charge=total_charge,
        )


# ---------------------------------------------------------------------------
# PiNet + ChargeResponseHead composed model
# ---------------------------------------------------------------------------


class PiNetPolarizability(nn.Module):
    """PiNet encoder paired with :class:`molpot.heads.ChargeResponseHead`.

    See :class:`molpot.heads.ChargeResponseHead` for variant semantics
    (``localchi`` / ``local`` / ``etainv`` / ``eem`` / ``acks2``).
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        atom_types: list[int] | None = None,
        variant: str = "localchi",
        iso: bool = False,
        hidden_dim: int = 64,
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        epsilon: float = 0.01,
        sigma: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layer_reduction = layer_reduction
        input_dim: int = getattr(encoder, "output_dim", 16)
        edge_dim: int = getattr(encoder, "edge_output_dim", input_dim)
        self.head = ChargeResponseHead(
            node_scalar_dim=input_dim,
            edge_scalar_dim=edge_dim,
            edge_vector_dim=input_dim,
            atom_types=atom_types,
            variant=variant,
            iso=iso,
            hidden_dim=hidden_dim,
            epsilon=epsilon,
            sigma=sigma,
        )

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        batch = self.encoder(batch)

        atom_batch = batch["atoms", "batch"]
        num_graphs = batch["graphs"].batch_size[0]
        node_scalars = _pool_layer(
            batch["atoms", "node_features"],
            self.layer_reduction,
        )
        edge_scalars = _pool_layer(
            batch["edges", "i1_features"],
            self.layer_reduction,
        )
        edge_vectors = None
        if "i3_features" in batch["edges"].keys():
            edge_vectors = _pool_layer(
                batch["edges", "i3_features"],
                self.layer_reduction,
            )
        pos = batch["atoms", "pos"]
        edge_index = batch["edges", "edge_index"]
        bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
        return self.head(
            pos=pos,
            Z=batch["atoms", "Z"],
            atom_batch=atom_batch,
            num_graphs=num_graphs,
            edge_index=edge_index,
            bond_diff=bond_diff,
            node_scalars=node_scalars,
            edge_scalars=edge_scalars,
            edge_vectors=edge_vectors,
        )
