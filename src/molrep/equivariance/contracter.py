"""Native Allegro tensor product.

The module owns the flattened monomial table and the CUDA op. It does not own
the weights. The caller keeps ``weights`` of shape ``(channels, paths)``, so a
checkpoint state dict is unchanged, and passes that matrix transposed to
``(paths, channels)``, the segment-major layout of the flattened polynomial.

On a CUDA tensor ``molnex_equivariance::contracter`` evaluates the table (the
atom and weight adjoints are deterministic two-pass sums); on a CPU tensor
:mod:`.reference` does. A descriptor outside the uniform-channel family raises
at construction.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nat. Commun. 14, 579 (2023).
    https://arxiv.org/abs/2204.05249
"""

from __future__ import annotations

import torch
from torch import nn

import molrep.eq as meq

from . import reference
from .backend import NativeBackend
from .contracter_terms import ContracterTerms


class Contracter(nn.Module):
    """Evaluate one Allegro contraction on ``(x1, x2, weight, index)``.

    Args:
        irreps_in1: Pair operand, one copy of each irrep.
        irreps_in2: Gathered atom operand, one copy of each irrep.
        irreps_out: Output irreps, one copy of each irrep.
        channels: Shared multiplicity ``u``.
        dtype: ``float32`` or ``float64``. Coefficients are stored in this dtype.
        device: Device of the term-table buffers.

    ``index`` is ``int32`` or ``int64``, one atom id per edge, in
    ``[0, x2.shape[0])``. ``weight`` is ``(weight_segments, channels)`` or the
    same values flattened. ``weight_segments`` equals the unflattened path count.
    """

    def __init__(
        self,
        irreps_in1: meq.Irreps,
        irreps_in2: meq.Irreps,
        irreps_out: meq.Irreps,
        channels: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("native contracter requires float32 or float64")
        tables = ContracterTerms(irreps_in1, irreps_in2, irreps_out, channels)
        self.irreps_in1, self.irreps_in2, self.irreps_out = irreps_in1, irreps_in2, irreps_out
        self.channels = tables.channels
        self.weight_segments = tables.weight_segments
        self.input1_segments = tables.input1_segments
        self.input2_segments = tables.input2_segments
        self.output_segments = tables.output_segments
        self.input1_dim = self.input1_segments * self.channels
        self.input2_dim = self.input2_segments * self.channels
        self.output_dim = self.output_segments * self.channels
        self.weight_dim = self.weight_segments * self.channels
        self.register_buffer(
            "terms", torch.tensor(tables.terms, dtype=torch.int16, device=device), persistent=False
        )
        self.register_buffer(
            "coefficients",
            torch.tensor(tables.coefficients, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "paths", torch.tensor(tables.paths, dtype=torch.int32, device=device), persistent=False
        )
        #: The class coefficient table, a host tuple: the generic kernel's parameter.
        self.cg_values: tuple[float, ...] = tuple(float(v) for v in tables.cg_values)
        #: The flattened term coefficients, the compiled kernels' parameter.
        self.values: tuple[float, ...] = tuple(float(v) for v in tables.coefficients)
        #: The compiled table this product runs on CUDA, or -1 for the class-blocked kernel.
        self.config = -1
        NativeBackend.load()
        if torch.cuda.is_available():
            self.config = self._compiled_config(tables.terms)

    @staticmethod
    def _compiled_config(terms) -> int:
        """The id of the compiled table equal to ``terms``, or -1."""
        own = torch.tensor(terms, dtype=torch.int16)
        table_of = torch.ops.molnex_equivariance.contracter_table
        config = 0
        while True:
            compiled = table_of(config)
            if compiled.shape[0] == 0:
                return -1
            if compiled.shape == own.shape and torch.equal(compiled, own):
                return config
            config += 1

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, weight: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        """Contract the pair operand with the gathered atom operand.

        Args:
            x1: Pair features ``(edges, input1_dim)`` in ``ir_mul``.
            x2: Atom features ``(atoms, input2_dim)`` in ``ir_mul``.
            weight: ``(weight_segments, channels)`` or the flat vector of that size.
            index: Atom row of ``x2`` for each edge, ``(edges,)``.

        Returns:
            ``(edges, output_dim)`` in ``ir_mul``.
        """
        if x1.ndim != 2 or x2.ndim != 2 or index.ndim != 1:
            raise ValueError("features are matrices and index is a vector")
        if x1.shape[1] != self.input1_dim or x2.shape[1] != self.input2_dim:
            raise ValueError("feature width differs from the term table")
        if index.shape[0] != x1.shape[0]:
            raise ValueError("index must have one entry per edge")
        if index.dtype not in (torch.int32, torch.int64):
            raise ValueError("index must be int32 or int64")
        if x1.dtype != self.coefficients.dtype or x2.dtype != self.coefficients.dtype:
            raise ValueError("feature dtype does not match the coefficient dtype")
        if weight.dtype != self.coefficients.dtype:
            raise ValueError("weight dtype does not match the coefficient dtype")
        flat = self._flat_weight(weight)
        if x1.is_cuda:
            return torch.ops.molnex_equivariance.contracter(
                x1.contiguous(),
                x2.contiguous(),
                flat,
                index.contiguous(),
                self.paths,
                self.cg_values,
                self.values,
                self.config,
                self.channels,
                self.output_segments,
            )
        return reference.contracter(
            x1,
            x2,
            flat,
            index,
            self.coefficients,
            self.terms.long(),
            self.channels,
            self.output_segments,
        )

    def _flat_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 2:
            if weight.shape != (self.weight_segments, self.channels):
                raise ValueError("weight must be (paths, channels)")
            return weight.contiguous()
        if weight.ndim == 1 and weight.numel() == self.weight_dim:
            return weight.contiguous()
        raise ValueError("weight must be (paths, channels) or a flat vector of that size")
