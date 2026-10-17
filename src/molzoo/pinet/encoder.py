"""Feature body shared by the PiNet2 products.

:class:`PiNetEncoder` registers the element table, the cutoff and radial
basis, the ``GCBlock`` stack and the residual updates. It registers no energy
readout, reference energy or property head — those are heads, and the product
that subclasses this body registers them on itself, so submodule names stay
unprefixed.

The class exposes primitives, not a pipeline: ``node_attrs`` /
``radial_features`` / ``angular_features`` / ``layer_features`` (and the
:class:`~molzoo.encoder.ElementTable` gate). :meth:`PiNetEncoder.forward`
composes them into the encoder contract (per-layer features written into the
batch); a product composes them into an energy or a property instead.

Hot-path discipline: every branch is frozen into a plain attribute (or the
presence of an optional submodule) by ``__init__``. The spec survives only as
``self._spec``, for provenance.

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems" J. Chem. Theory Comput. 21, 1382 (2025).
    https://doi.org/10.1021/acs.jctc.4c01570
    Shao et al. "PiNN: A Python Library for Building Atomic Neural Networks
    of Molecules and Materials" J. Chem. Inf. Model. 60, 1184 (2020).
    https://doi.org/10.1021/acs.jcim.9b00994
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from molix import config
from molrep.embedding.cutoff import CosineCutoff, HalfCosineCutoff, TanhCutoff
from molrep.embedding.geometry import edge_lengths, edge_vectors
from molrep.embedding.radial import GaussianBasis, PolynomialBasis
from molzoo.encoder import ElementTable
from molzoo.pinet.layers import GCBlock, GCOutput, ResUpdate, compute_d5
from molzoo.pinet.spec import PiNetSpec

_CUTOFFS = {"f1": CosineCutoff, "f2": TanhCutoff, "hip": HalfCosineCutoff}


class PiNetLayer(NamedTuple):
    """One block of :meth:`PiNetEncoder.layer_features`.

    Attributes:
        block: The block's own outputs (``p1_t`` …, ``i1`` …).
        p1: Residual scalar state after the block ``(N, C)``.
        p3: Residual vector state ``(N, 3, C)``, or ``None`` below rank 3.
        p5: Residual rank-2 state ``(N, 5, C)``, or ``None`` below rank 5.
    """

    block: GCOutput
    p1: torch.Tensor
    p3: torch.Tensor | None
    p5: torch.Tensor | None


class PiNetEncoder(ElementTable, TensorDictModuleBase):
    """PiNet2 feature body (``src/molzoo/specs/pinet2.md`` §5).

    Reads ``atoms.Z``, ``atoms.pos``, ``pairs.pair_index`` ``(E, 2)`` and, when
    present, the imaged ``pairs.pair_diff``. :meth:`forward` writes
    ``atoms.node_features`` ``(N, depth, C)`` and ``atoms.p1_block_outputs``
    ``(N, depth, C)``; with ``emit_property_features`` also ``atoms.p3_features``
    / ``atoms.p5_features`` and ``pairs.i1_features`` / ``i3_features`` /
    ``i5_features``.

    Args:
        spec: A validated :class:`~molzoo.pinet.spec.PiNetSpec`. Head fields
            (``out_*``, ``atomic_energies``, ``scale``) are ignored here.
        dtype: Floating dtype every parameter and buffer is built at; ``None``
            is the (transitional) process-wide default.
    """

    in_keys = [("atoms", "Z"), ("atoms", "pos"), ("pairs", "pair_index")]
    out_keys = [("atoms", "node_features"), ("atoms", "p1_block_outputs")]

    def __init__(self, spec: PiNetSpec, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        #: Provenance only. Never read on a hot path.
        self._spec = spec
        #: The build dtype of every float parameter and buffer.
        self.dtype = dtype
        self.register_buffer(
            "z_table", torch.tensor(spec.atomic_numbers, dtype=torch.long), persistent=False
        )

        # ---- frozen switches ----
        self.depth = spec.depth
        self.feature_dim = spec.ii_nodes[-1]
        self.n_props = spec.rank // 2 + 1
        self.has_p3 = spec.rank >= 3
        self.has_p5 = spec.rank >= 5
        self.emit_property_features = spec.emit_property_features

        self.cutoff = _CUTOFFS[spec.cutoff_type](r_cut=spec.r_max, dtype=dtype)
        self.basis_fn = (
            PolynomialBasis(spec.n_basis, dtype=dtype)
            if spec.basis_type == "polynomial"
            else GaussianBasis(
                center=spec.center,
                gamma=spec.gamma,
                r_cut=spec.r_max,
                n_basis=spec.n_basis,
                dtype=dtype,
            )
        )

        # Channel schedules: P1 starts as the one-hot, P3/P5 start one channel wide.
        c = self.feature_dim
        p1_dims = [len(spec.atomic_numbers)] + [c] * spec.depth
        px_dims = [1] + [c] * spec.depth
        self.gc_blocks = nn.ModuleList(
            GCBlock(
                rank=spec.rank,
                weighted=spec.weighted,
                pp_nodes=spec.pp_nodes,
                pi_nodes=spec.pi_nodes,
                ii_nodes=spec.ii_nodes,
                n_basis=spec.n_basis,
                p1_in_dim=p1_dims[t],
                p3_in_dim=px_dims[t],
                p5_in_dim=px_dims[t],
                activation=spec.activation,
                dtype=dtype,
            )
            for t in range(spec.depth)
        )
        self.res_update1 = nn.ModuleList(
            ResUpdate(in_dim=p1_dims[t], out_dim=p1_dims[t + 1], dtype=dtype)
            for t in range(spec.depth)
        )
        self.res_update3 = (
            nn.ModuleList(
                ResUpdate(in_dim=px_dims[t], out_dim=px_dims[t + 1], dtype=dtype)
                for t in range(spec.depth)
            )
            if self.has_p3
            else None
        )
        self.res_update5 = (
            nn.ModuleList(
                ResUpdate(in_dim=px_dims[t], out_dim=px_dims[t + 1], dtype=dtype)
                for t in range(spec.depth)
            )
            if self.has_p5
            else None
        )

    # ------------------------------------------------------------ primitives
    def node_attrs(self, Z: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """One-hot element attributes over the table — PiNN ``AtomicOnehot``.

        Args:
            Z: Atomic numbers ``(N,)``, inside the table.
            dtype: Floating dtype of the result.

        Returns:
            ``(N, n_elements)``.
        """
        index = torch.searchsorted(self.z_table, Z.reshape(-1))
        return torch.nn.functional.one_hot(index, self.z_table.numel()).to(dtype)

    def radial_features(self, lengths: torch.Tensor) -> torch.Tensor:
        """Cutoff-weighted radial basis ``e_ijb = f_c(d) φ_b(d)``.

        Args:
            lengths: Pair distances ``(E,)`` in Å.

        Returns:
            ``(E, n_basis)``.
        """
        return self.basis_fn(lengths, fc=self.cutoff(lengths))

    def angular_features(
        self, vectors: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Unit directions ``d3`` and the rank-2 basis ``d5`` the rank needs.

        Args:
            vectors: Pair displacements ``(E, 3)``.
            lengths: Pair distances ``(E,)``.

        Returns:
            ``(d3 (E, 3) | None, d5 (E, 5) | None)``.
        """
        if not self.has_p3:
            return None, None
        d3 = vectors / lengths.unsqueeze(-1)
        return d3, compute_d5(d3) if self.has_p5 else None

    def layer_features(
        self,
        *,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        basis: torch.Tensor,
        d3: torch.Tensor | None,
        d5: torch.Tensor | None,
    ) -> list[PiNetLayer]:
        """Run the block stack.

        Args:
            node_attrs: One-hot elements ``(N, n_elements)`` (``P1⁰``).
            edge_index: Source/target pairs ``(E, 2)``.
            basis: Radial basis ``(E, n_basis)``.
            d3: Unit directions (rank ≥ 3).
            d5: Rank-2 basis (rank 5).

        Returns:
            One :class:`PiNetLayer` per block.
        """
        n = node_attrs.shape[0]
        p1 = node_attrs
        p3 = node_attrs.new_zeros(n, 3, 1) if self.has_p3 else None
        p5 = node_attrs.new_zeros(n, 5, 1) if self.has_p5 else None
        layers: list[PiNetLayer] = []
        for t, block in enumerate(self.gc_blocks):
            out = block(edge_index, basis, p1, p3=p3, d3=d3, p5=p5, d5=d5)
            p1 = self.res_update1[t](p1, out.p1)
            if self.res_update3 is not None:
                p3 = self.res_update3[t](p3, out.p3)
            if self.res_update5 is not None:
                p5 = self.res_update5[t](p5, out.p5)
            layers.append(PiNetLayer(block=out, p1=p1, p3=p3, p5=p5))
        return layers

    def _pair_geometry(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(edge_index, vectors, lengths)`` from a batch (shared by the products)."""
        # Contiguous for AOTInductor stride specialisation (LAMMPS pair_style).
        edge_index = batch["pairs", "pair_index"].contiguous()
        diff = (
            batch["pairs", "pair_diff"]
            if ("pairs", "pair_diff") in batch.keys(include_nested=True)
            else None
        )
        vectors = edge_vectors(batch["atoms", "pos"], edge_index, edge_diff=diff)
        return edge_index, vectors, edge_lengths(vectors).clamp(min=1e-8)

    def _batch_layers(
        self, batch: TensorDict
    ) -> tuple[list[PiNetLayer], torch.Tensor, torch.Tensor]:
        """Validate once, then ``(layers, edge_index, vectors)`` for a batch."""
        Z = batch["atoms", "Z"]
        self._validate_once(Z)
        edge_index, vectors, lengths = self._pair_geometry(batch)
        d3, d5 = self.angular_features(vectors, lengths)
        layers = self.layer_features(
            node_attrs=self.node_attrs(Z, vectors.dtype),
            edge_index=edge_index,
            basis=self.radial_features(lengths),
            d3=d3,
            d5=d5,
        )
        return layers, edge_index, vectors

    # -------------------------------------------------------------- encoder
    def forward(self, td: TensorDict) -> TensorDict:
        """Encode a batch, writing per-layer features into ``td`` in place."""
        layers, _, _ = self._batch_layers(td)
        td["atoms", "node_features"] = torch.stack([layer.p1 for layer in layers], dim=1)
        td["atoms", "p1_block_outputs"] = torch.stack([layer.block.p1 for layer in layers], dim=1)
        if self.emit_property_features:
            td["pairs", "i1_features"] = torch.stack([layer.block.i1 for layer in layers], dim=1)
            if self.has_p3:
                td["atoms", "p3_features"] = torch.stack([layer.p3 for layer in layers], dim=1)
                td["pairs", "i3_features"] = torch.stack(
                    [layer.block.i3 for layer in layers], dim=1
                )
            if self.has_p5:
                td["atoms", "p5_features"] = torch.stack([layer.p5 for layer in layers], dim=1)
                td["pairs", "i5_features"] = torch.stack(
                    [layer.block.i5 for layer in layers], dim=1
                )
        return td
