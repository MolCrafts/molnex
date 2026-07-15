"""PiNet encoder + potential head.

PiNet is a multi-rank representation architecture (P1 scalar, P3 vector, P5
rank-5). The encoder produces ``(N, layers, features)`` node features; the
PiNet-specific head pools across layers, predicts per-atom energies, and
derives forces via functorch (``torch.func.grad``). The head lives alongside the encoder (not under
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


def _edge_bond_diff(edges, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Source→target edge displacement, periodic-boundary-correct and differentiable.

    Open systems: recompute ``pos[target] - pos[source]`` so forces flow to ``pos``.
    Periodic systems: the neighbour list supplies the *minimum-image* ``edge_diff``
    (the molix analogue of PiNN replicating periodic images). We take that imaged
    value but route the gradient through the raw displacement via a straight-through
    term — exact, because ∂(imaged diff)/∂pos = ∂raw/∂pos = identity (the per-edge
    cell-shift is constant). A no-op for open systems, where the supplied
    ``edge_diff`` already equals ``raw``.

    Args:
        edges: The batch's ``edges`` sub-TensorDict (may carry ``edge_diff``).
        pos: Atom positions ``(N, 3)``.
        edge_index: Source/target pairs ``(E, 2)``; ``[:,0]``=source, ``[:,1]``=target.

    Returns:
        Edge displacement ``(E, 3)`` with periodic-correct value and exact gradient.
    """
    raw = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    if "edge_diff" in edges.keys():
        # ``.detach()`` on the supplied value makes this robust whether the
        # neighbour list's edge_diff is a detached cache leaf (training) or a
        # live in-graph tensor (an MD force_fn that recomputes it): the gradient
        # always flows solely through ``raw`` (the correct, cell-shift-invariant
        # ∂/∂pos), never double-counting a live supplied tensor.
        return edges["edge_diff"].detach() + (raw - raw.detach())
    return raw


class PiNet(TensorDictModuleBase):
    """PiNet feature encoder.

    Inputs from the nested ``TensorDict`` schema:

    * ``("atoms", "Z")``: atomic numbers ``(N,)``.
    * ``("atoms", "pos")``: positions ``(N, 3)``.
    * ``("edges", "edge_index")``: source-target pairs ``(E, 2)``.

    Edge geometry (``edge_diff`` = ``pos[target] - pos[source]`` and
    ``edge_dist`` = ``‖edge_diff‖``) is derived from ``pos`` + ``edge_index``
    inside ``forward``. Caching them in the pipeline buys nothing (one
    gather + diff + norm per step) and breaks the gradient flow pos → geometry
    that force derivation needs.

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
        # Force a standard row-major layout. Eager is stride-invariant, but
        # AOTInductor specializes the compiled kernel on the traced edge_index
        # stride: without this, a contiguous edge_index from a C++ caller (e.g.
        # pair_style molnex) is indexed with the trace-time stride and silently
        # scrambles source/target. ``.contiguous()`` traces into the graph and
        # normalizes any input layout at runtime.
        edge_index = td["edges", "edge_index"].contiguous()

        # Edge geometry: PBC-correct (uses the neighbour list's minimum-image
        # edge_diff under periodic boundaries) and differentiable. See _edge_bond_diff.
        edge_diff = _edge_bond_diff(td["edges"], pos, edge_index)
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
            tensors["d5"] = _compute_d5(d3)

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
    """PiNet energy + force prediction model — ready to use from hyperparameters.

    Pass PiNet hyperparameters directly; the encoder is built internally, e.g.::

        model = PiNetPotential(atom_types=[1, 6, 7, 8], r_max=4.5, depth=5,
                               hidden_dim=64, compute_forces=True)

    Predicts per-atom energies via per-block OutLayers, aggregates them to graph
    energies, and derives forces as ``F = -∂E/∂pos`` with ``torch.func.grad``
    (functorch) when ``compute_forces`` is set. The functorch path traces the
    force into the forward graph, so ``energy → force → loss`` is a single
    backward that ``torch.compile(fullgraph=True)`` can capture — no
    double-backward barrier.

    Per-element baseline subtraction (atomic dress) is handled at cache time by
    :class:`molix.data.AtomicDress` — labels are already dressed when they reach
    this model, so no runtime dress add is needed.

    Args:
        hidden_dim: Hidden dimension of the per-atom energy MLP.
        layer_reduction: How to pool across GC-block layers
            (``"mean"`` / ``"sum"`` / ``"last"``).
        compute_forces: Default value for forward's ``compute_forces``
            kwarg. Set to ``True`` for force training so the Trainer's plain
            ``model(batch)`` call returns ``{"energy", "forces"}``.
        **pinet_kwargs: PiNet hyperparameters (``atom_types``, ``r_max``,
            ``depth``, ``n_basis``, …) used to build the encoder.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        compute_forces: bool = False,
        **pinet_kwargs: object,
    ) -> None:
        super().__init__()
        self.encoder = PiNet(**pinet_kwargs)
        # ``layer_reduction`` is accepted for API compatibility but unused by the
        # energy head: PiNet2 does NOT pool layers — it accumulates a per-block
        # OutLayer residually (see below), so there is no layer axis to reduce.
        self.layer_reduction = layer_reduction
        self.compute_forces_default = compute_forces
        # Set by :meth:`compile_energy`; when present, the energy forward runs
        # through the compiled callable instead of the eager method.
        self._compiled_energy_forward = None
        # Lazy params must be materialized outside any functorch transform; the
        # first forward flips this (see _forward_functorch).
        self._materialized = False

        # Per-block output heads, residually accumulated (PiNN PiNet2 OutLayer):
        # per-atom energy = Σ_b OutLayer_b(p1_block_output_b), each OutLayer a
        # tanh FFLayer([hidden_dim]) -> biasless Linear(1). The absolute energy
        # zero-point is owned by the atomic dress, so the projection has no bias.
        depth: int = int(getattr(self.encoder, "depth", 1))
        self.out_layers = nn.ModuleList(
            [OutLayer([hidden_dim], out_units=1, activation="tanh") for _ in range(depth)]
        )
        self.energy_aggregation = EnergyAggregation(pooling="sum")
        # PiNet is pure-PyTorch → functorch (single backward, torch.compile-friendly).
        self.force_derivation = ForceDerivation(method="functorch")

    def forward(
        self, batch: TensorDict, *, compute_forces: bool | None = None
    ) -> dict[str, torch.Tensor]:
        if compute_forces is None:
            compute_forces = self.compute_forces_default
        if compute_forces:
            return self._forward_functorch(batch)
        energy_forward = self._compiled_energy_forward or self._energy_forward
        return energy_forward(batch)

    def _forward_functorch(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Force forward via ``torch.func.grad`` (no double-backward barrier).

        ``energy_fn`` recomputes the energy from a clone of the batch with the
        candidate positions swapped in; PiNet derives all edge geometry from
        positions internally, so the gradient flows pos → geometry → energy.

        **Training** (``self.training``): an eager energy pass runs FIRST — it
        produces energy outputs connected to the parameters (with ``pos``
        detached) so an energy+force loss trains via ordinary autograd — then
        ``torch.func.grad`` reruns the energy graph for forces. Cost: 2 forwards
        + 1 backward per step.

        **Inference** (``eval()``): the energy outputs need no autograd graph,
        so a single ``torch.func.grad(..., has_aux=True)`` pass returns forces
        AND energies from one forward + one backward — the eager energy pass is
        skipped entirely (~25-40% faster per MD step; the win that motivated
        this split is the pinet-quant water NVE workhorse, which cannot
        ``torch.compile`` its fake-quantized forward).

        Lazy parameters must be materialized OUTSIDE the functorch transform
        (materializing inside corrupts its tensor wrappers and segfaults) —
        training does so via its eager pass; inference runs a one-time throwaway
        eager pass on the first call.
        """
        pos = batch["atoms", "pos"].detach()
        base = batch.clone()
        base["atoms", "pos"] = pos

        def energy_fn(p: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            b = base.clone()
            b["atoms", "pos"] = p
            out = self._energy_forward(b)
            return out["energy"].sum(), out

        if not self.training:
            if not self._materialized:
                self._energy_forward(base.clone())  # materialize lazy params eagerly
                self._materialized = True
            grad, out = torch.func.grad(energy_fn, has_aux=True)(pos)
            out["forces"] = -grad
            return out

        out = self._energy_forward(base.clone())
        self._materialized = True
        out["forces"] = self.force_derivation(lambda p: energy_fn(p)[0], pos)
        return out

    def _energy_forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Compilable energy computation: encoder -> per-block OutLayer sum -> aggregate.

        Extracted as a single region so ``torch.compile`` can wrap the
        energy-only path (see :meth:`compile_energy`), and so the functorch force
        closure in :meth:`_forward_functorch` can call it as a pure function of
        positions.

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
        # Fixed-length padding (CUDA-graph path): a boolean ("atoms","mask") marks
        # real atoms. Zeroing ghost-atom energy here makes padded atoms contribute
        # 0 to per-graph energy — and therefore 0 to forces (the total energy no
        # longer depends on their positions). Real atoms are untouched because no
        # padding edge connects to them, so energy/forces match the unpadded batch.
        # No-op for unpadded batches (no "mask" key). See data.tasks.PadMolecularBatch.
        if "mask" in batch["atoms"].keys():
            atom_energy = atom_energy * batch["atoms", "mask"].to(atom_energy.dtype)
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

        **Energy-only — does not accelerate force training.** The compiled
        callable is invoked only on the ``compute_forces=False`` path; with
        ``compute_forces=True`` the model routes through
        :meth:`_forward_functorch`, which calls the *eager* ``_energy_forward``
        inside its ``torch.func.grad`` closure. So compiling here leaves force
        training unchanged. (The force forward is separately fullgraph-compilable
        via functorch — wrap ``model(b, compute_forces=True)`` directly — because
        ``torch.func.grad`` traces the force into the forward graph and needs no
        double backward; that path is unrelated to this method.)

        Materialise lazy parameters with one warm-up forward before calling
        this, so the compiled graph captures concrete shapes.

        Use the **default** inductor mode *for this ragged energy-only path*.
        ``mode="reduce-overhead"`` (CUDA graphs) is *counterproductive* here:
        without padding, molecular batches have variable atom/edge counts, and
        CUDA graphs require static shapes — measured on a GH200 it ran slower
        than eager (the per-step capture/guard overhead dominates, op-calls/step
        stay ~7.2k instead of dropping to ~2.8k).

        For **force training**, do NOT use this method: compile the whole model
        instead (``trainer.compile(cuda_graphs=True)`` /
        ``molix.Compiler.CUDA_GRAPH_PRESET``) on *padded* fixed shapes
        (``PadMolecularBatch`` + ``drop_last=True``). There ``reduce-overhead``
        is the **fastest** config (~10x eager), the opposite of the ragged case
        — see ``docs/molix/explanation/throughput-and-compilation.md``.

        Args:
            backend: ``torch.compile`` backend (default ``"inductor"``).
            **kwargs: Forwarded to :func:`torch.compile`. Avoid
                ``mode="reduce-overhead"`` for ragged (unpadded) batches.
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
        edge_diff = None
        if self.head.uses_bc and "i1_features" in batch["edges"].keys():
            edge_scalars = _pool_layer(
                batch["edges", "i1_features"],
                self.layer_reduction,
            )
            edge_index = batch["edges", "edge_index"]
            pos = batch["atoms", "pos"]
            edge_diff = _edge_bond_diff(batch["edges"], pos, edge_index)
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
            edge_diff=edge_diff,
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
        edge_diff = _edge_bond_diff(batch["edges"], pos, edge_index)
        return self.head(
            pos=pos,
            Z=batch["atoms", "Z"],
            atom_batch=atom_batch,
            num_graphs=num_graphs,
            edge_index=edge_index,
            edge_diff=edge_diff,
            node_scalars=node_scalars,
            edge_scalars=edge_scalars,
            edge_vectors=edge_vectors,
        )
