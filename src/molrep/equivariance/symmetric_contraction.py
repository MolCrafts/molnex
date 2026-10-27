"""Symmetric contraction of MACE's product block.

The module owns the weights in MACE's own layout — ``weight``
``(num_elements, A, U)`` and the original-MACE ``projection`` ``(A, B)`` that
maps them onto cuEquivariance's orthogonal basis — so an official checkpoint
loads unchanged, and the term table of the standard polynomial read off the
descriptor (:class:`.symmetric_contraction_terms.SymmetricContractionTerms`).
Degrees ``1..contraction_degree`` are all evaluated.

On a CUDA tensor ``molnex_equivariance::contraction`` evaluates the table —
for MACE's family (natural parity, ``l_out ≤ l_in ≤ 3``, degree 2 or 3) with
the kernels compiled for that table, every operand in registers, otherwise
with the generic shared-memory kernel; the weight adjoint is a deterministic
two-pass sum by element either way. On a CPU tensor :mod:`.reference` does.
Supported: degrees 1..3, ``ir_mul``, one multiplicity shared by the input and
the output, float32 or float64. Anything else raises at construction.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import torch
from torch import nn

import molrep.eq as meq

from . import reference
from .backend import NativeBackend
from .symmetric_contraction_terms import SymmetricContractionTerms


class SymmetricContraction(nn.Module):
    """``y = Σ_ν Σ_paths C · (P W)[element] · x^{⊗ν}`` per channel.

    Args:
        irreps_in: Node-feature irreps. One multiplicity, shared with the output.
        irreps_out: Output irreps.
        contraction_degree: Highest body order, 1, 2, or 3.
        num_elements: Weight rows; each node reads the row of its element.
        dtype: ``float32`` or ``float64``, the parameter and coefficient dtype.
        device: Device of the parameters and buffers.

    Attributes:
        weight: ``(num_elements, A, U)`` in MACE's basis.
        projection: ``(A, B)`` onto the contraction basis (a persistent
            buffer, as in a MACE checkpoint).
    """

    def __init__(
        self,
        irreps_in: meq.Irreps,
        irreps_out: meq.Irreps,
        contraction_degree: int,
        num_elements: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("native symmetric contraction requires float32 or float64")
        if int(num_elements) < 1:
            raise ValueError("num_elements must be positive")
        tables = SymmetricContractionTerms(irreps_in, irreps_out, contraction_degree)
        self.irreps_in, self.irreps_out = irreps_in, irreps_out
        self.contraction_degree = int(contraction_degree)
        self.num_elements = int(num_elements)
        self.channels = tables.channels
        self.weight_segments = tables.weight_segments
        self.input_segments = tables.input_segments
        self.output_segments = tables.output_segments
        self.input_dim = self.input_segments * self.channels
        self.output_dim = self.output_segments * self.channels
        self.weight_dim = self.weight_segments * self.channels
        self.register_buffer(
            "projection", torch.tensor(tables.projection, dtype=dtype, device=device)
        )
        self.weight = nn.Parameter(
            torch.randn(
                self.num_elements,
                tables.projection.shape[0],
                self.channels,
                dtype=dtype,
                device=device,
            )
        )
        # Rebuilt from the irreps in ``__init__``; kept out of the state dict so
        # an existing MACE checkpoint still loads into ``weight`` and ``projection``.
        # ``terms`` / ``coefficients`` are the merged table the reference reads
        # (``-1`` for a missing slot); the generic kernel reads the same terms
        # packed (``w << 21 | i << 14 | j << 7 | k``, a missing slot pointing at
        # the sentinel segment ``S``) by output segment with ``out_offsets``;
        # the compiled kernels of MACE's family take the coefficient values as
        # a parameter over their built-in table (``config``).
        self.register_buffer(
            "terms", torch.tensor(tables.terms, dtype=torch.int16, device=device), persistent=False
        )
        self.register_buffer(
            "coefficients",
            torch.tensor(tables.coefficients, dtype=dtype, device=device),
            persistent=False,
        )
        sentinel = self.input_segments
        packed, offsets = self._pack(tables.terms, tables.terms[:, 4], sentinel, tables.terms[:, 0])
        self.register_buffer(
            "packed", torch.tensor(packed, dtype=torch.int32, device=device), persistent=False
        )
        self.register_buffer(
            "out_offsets", torch.tensor(offsets, dtype=torch.int32, device=device), persistent=False
        )
        #: The coefficients as a host tuple: the compiled kernels take them as a parameter.
        self.values: tuple[float, ...] = tuple(float(v) for v in tables.coefficients)
        self._family = self._mace_family(irreps_in, irreps_out, self.contraction_degree)
        #: The compiled table this contraction runs on CUDA, or -1 for the generic kernel.
        self.config = -1
        NativeBackend.load()
        if self._family is not None and torch.cuda.is_available():
            self.config = self._compiled_config(*self._family, tables.terms, sentinel)

    @staticmethod
    def _mace_family(irreps_in, irreps_out, degree: int) -> tuple[int, int, int] | None:
        """``(l_in, l_out, degree)`` when the irreps are MACE's natural-parity family."""
        if degree not in (2, 3):
            return None
        shapes = []
        for irreps in (irreps_in, irreps_out):
            pattern = [(mi.ir.l, mi.ir.p) for mi in irreps]
            l_max = len(pattern) - 1
            if pattern != [(l, (-1) ** l) for l in range(l_max + 1)]:
                return None
            shapes.append(l_max)
        l_in, l_out = shapes
        if not 1 <= l_in <= 3 or l_out > l_in:
            return None
        return l_in, l_out, degree

    @staticmethod
    def _compiled_config(l_in: int, l_out: int, degree: int, terms, sentinel: int) -> int:
        """The compiled table's id when the library's table equals ``terms``, else -1."""
        compiled = torch.ops.molnex_equivariance.contraction_table(l_in, l_out, degree)
        own = torch.tensor(terms, dtype=torch.int16)
        own[own < 0] = sentinel
        if compiled.shape != own.shape or not torch.equal(compiled, own):
            return -1
        return l_in * 100 + l_out * 10 + degree

    @staticmethod
    def _pack(terms, group, sentinel: int, top):
        """Pack ``(w, i, j, k, o)`` rows as ``top << 21 | i << 14 | j << 7 | k`` with group offsets.

        ``terms`` must already be ordered by ``group`` (non-decreasing); a
        missing slot (``-1``) becomes ``sentinel``.
        """
        import numpy as np

        rows = np.asarray(terms, dtype=np.int64)
        group = np.asarray(group, dtype=np.int64)
        top = np.asarray(top, dtype=np.int64)
        if rows.shape[0] and np.any(np.diff(group) < 0):
            raise ValueError("terms are not ordered by their group")
        slots = rows[:, 1:4].copy()
        slots[slots < 0] = sentinel
        if top.size and (top.max() > 1023 or slots.max() > 127):
            raise ValueError("term indices exceed the packed word")
        packed = (top << 21) | (slots[:, 0] << 14) | (slots[:, 1] << 7) | slots[:, 2]
        groups = int(group.max()) + 1 if group.size else 0
        counts = np.bincount(group, minlength=groups) if group.size else np.zeros(0, dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        return packed.astype(np.int32), offsets

    def extra_repr(self) -> str:
        return (
            f"{self.irreps_in} -> {self.irreps_out}, degree={self.contraction_degree}, "
            f"num_elements={self.num_elements}"
        )

    def projected_weight(self) -> torch.Tensor:
        """``(num_elements, B · U)``: the weights in the contraction basis."""
        return torch.einsum("zau,ab->zbu", self.weight, self.projection).flatten(1)

    def forward(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Contract ``features`` with the weights of each node's element.

        Args:
            features: Node features ``(nodes, input_dim)`` in ``ir_mul``.
            indices: Element id of each node, ``(nodes,)``, ``int32`` or ``int64``.

        Returns:
            Contracted features ``(nodes, output_dim)``.
        """
        if features.ndim != 2 or indices.ndim != 1:
            raise ValueError("features is a matrix and indices a vector")
        if features.shape[1] != self.input_dim:
            raise ValueError("feature width differs from the term table")
        if indices.shape[0] != features.shape[0]:
            raise ValueError("indices must have one entry per node")
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be int32 or int64")
        projected = self.projected_weight()
        if features.is_cuda:
            return torch.ops.molnex_equivariance.contraction(
                features.contiguous(),
                projected.contiguous(),
                indices.contiguous(),
                self.packed,
                self.coefficients,
                self.out_offsets,
                self.values,
                self.config,
                self.channels,
                self.output_segments,
            )
        return reference.symmetric_contraction(
            features,
            projected,
            indices,
            self.coefficients,
            self.terms.long(),
            self.channels,
            self.output_segments,
        )
