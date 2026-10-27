"""Term tables of a symmetric contraction, read off the cuEquivariance descriptor.

The torch module
:class:`molrep.eq.torch.SymmetricContraction` evaluates the standard
polynomial (``poly2``) after the original-MACE projection. This type walks
that polynomial and records one scalar monomial per path. Channels stay
outside the table: every segment has shape ``(U,)``, or ``()`` when ``U``
is 1 and the descriptor has squeezed that axis. The contraction is the same
at each channel.

A term is five ``int16`` indices ``(w, i, j, k, o)``. ``j`` or ``k`` is ``-1``
when the body order does not use that slot. The value is

```
y[o] += C * W[w] * x[i] * x[j]? * x[k]?
```

with the product taken over the slots that are present. A monomial is
symmetric in its slots, so the descriptor's paths that differ only by a
permutation of ``(i, j, k)`` — the same ``w`` and ``o`` — are one term here:
slots sorted ascending, coefficients summed, exact zeros dropped. At MACE's
shapes that is 4–4.6× fewer terms than the descriptor lists (5520 → 1203 for
``128x0e+…+128x3o → 128x0e+128x1o+128x2e`` at degree 3). Repeated indices
stay repeated slots, and each slot contributes once to an adjoint, which is
the product rule (``x²`` contributes ``2 x dx``).

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import numpy as np

import molrep.eq as meq
from molrep.eq.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction,
)

#: Body orders the kernel evaluates. MACE uses 2 or 3.
MAX_DEGREE = 3
#: Input and output component counts. ``l_max = 3`` is 16; 64 covers ``l_max = 7``.
MAX_SEGMENTS = 64


class SymmetricContractionTerms:
    """Monomials of one symmetric contraction.

    Attributes:
        channels: Uniform multiplicity ``U``.
        weight_segments: Segments of the projected weight, ``B``.
        input_segments: Components of the node features.
        output_segments: Components of the output.
        projection: Original-MACE matrix ``(A, B)`` applied as
            ``einsum("zau,ab->zbu")`` before the contraction.
        terms: Merged monomials ``(T, 5)`` ``int16``, columns
            ``w, i, j, k, o`` (slots ascending, ``-1`` last), ordered by
            ``o, i, j, k, w``.
        coefficients: Their scalars ``(T,)`` ``float64``.
        grouped_terms: The same terms, stably grouped by weight segment.
        grouped_coefficients: Coefficients in that order.
        weight_offsets: ``(B + 1,)`` ``int32`` starts of those groups.

    Raises:
        ValueError: The irreps, degree, or descriptor are outside the kernel.
    """

    def __init__(
        self, irreps_in: meq.Irreps, irreps_out: meq.Irreps, contraction_degree: int
    ) -> None:
        if contraction_degree < 1 or contraction_degree > MAX_DEGREE:
            raise ValueError(
                f"native symmetric contraction supports degrees 1..{MAX_DEGREE}, "
                f"got {contraction_degree}"
            )
        if len(irreps_in) == 0 or len(irreps_out) == 0:
            raise ValueError("native symmetric contraction requires non-empty irreps")
        muls = set(irreps_in.muls) | set(irreps_out.muls)
        if len(muls) != 1:
            raise ValueError(
                "native symmetric contraction requires one multiplicity shared by "
                f"every input and output irrep, got {sorted(muls)}"
            )
        channels = muls.pop()
        if irreps_in.dim // channels > MAX_SEGMENTS or irreps_out.dim // channels > MAX_SEGMENTS:
            raise ValueError(
                f"native symmetric contraction supports at most {MAX_SEGMENTS} "
                "input or output components"
            )
        polynomial, projection = symmetric_contraction(
            irreps_in, irreps_out, range(1, contraction_degree + 1)
        )
        poly = polynomial.polynomial
        weight, features = poly.inputs
        (output,) = poly.outputs
        for operand, name in (
            (weight, "weight"),
            (features, "feature"),
            (output, "output"),
        ):
            # Multiplicity 1 is squeezed to a scalar segment. The flat layout
            # is still one value per segment, which is ``U == 1``.
            shape = operand.segment_shape
            scalar = channels == 1 and shape == ()
            if shape != (channels,) and not scalar:
                raise ValueError(
                    f"native symmetric contraction requires every {name} segment to have "
                    f"shape {(channels,)}, got {operand.segments[:4]}"
                )
        if (
            features.num_segments > MAX_SEGMENTS
            or output.num_segments > MAX_SEGMENTS
            or weight.num_segments > 32767
        ):
            raise ValueError(
                f"native symmetric contraction supports at most {MAX_SEGMENTS} "
                "input or output segments and 32767 weight segments"
            )
        if projection.ndim != 2 or projection.shape[1] != weight.num_segments:
            raise ValueError(
                "the symmetric-contraction projection does not match the weight segments "
                f"({projection.shape} vs {weight.num_segments})"
            )
        rows: list[tuple[int, int, int, int, int]] = []
        values: list[float] = []
        seen: set[int] = set()
        for operation, product in poly.operations:
            degree = operation.buffers.count(1)
            expected = (0,) + (1,) * degree + (2,)
            if degree < 1 or degree > contraction_degree or operation.buffers != expected:
                raise ValueError(f"unsupported symmetric-contraction operation {operation.buffers}")
            if degree in seen:
                raise ValueError(f"degree {degree} appears twice in the polynomial")
            seen.add(degree)
            if product.coefficient_subscripts != "":
                raise ValueError(
                    "native symmetric contraction requires scalar path coefficients, "
                    f"got {product.coefficient_subscripts!r}"
                )
            if product.operands[0].segments != weight.segments:
                raise ValueError("a contraction degree does not share the stacked weights")
            for slot in range(1, degree + 1):
                if product.operands[slot].segments != features.segments:
                    raise ValueError("a contraction degree does not share the node features")
            if product.operands[degree + 1].segments != output.segments:
                raise ValueError("a contraction degree does not share the output")
            for path in product.paths:
                if path.coefficients.shape != ():
                    raise ValueError(
                        f"path coefficient has shape {path.coefficients.shape}, expected a scalar"
                    )
                slots = list(path.indices[1:-1])
                if len(slots) != degree:
                    raise ValueError(f"degree {degree} path has indices {path.indices}")
                while len(slots) < 3:
                    slots.append(-1)
                wseg, oseg = path.indices[0], path.indices[-1]
                entry = (wseg, slots[0], slots[1], slots[2], oseg)
                if not (
                    0 <= wseg < weight.num_segments
                    and 0 <= oseg < output.num_segments
                    and all(item == -1 or 0 <= item < features.num_segments for item in slots)
                ):
                    raise ValueError(f"path indices {entry} fall outside the operands")
                rows.append(entry)
                values.append(float(path.coefficients))
        merged: dict[tuple[int, int, int, int, int], float] = {}
        for (wseg, i, j, k, oseg), value in zip(rows, values, strict=True):
            present = sorted(slot for slot in (i, j, k) if slot >= 0)
            key = (oseg, *present, *([-1] * (3 - len(present))), wseg)
            merged[key] = merged.get(key, 0.0) + value
        kept = sorted((key, value) for key, value in merged.items() if value != 0.0)
        terms = np.asarray(
            [(wseg, i, j, k, oseg) for (oseg, i, j, k, wseg), _ in kept], dtype=np.int16
        ).reshape(-1, 5)
        coefficients = np.asarray([value for _, value in kept], dtype=np.float64)
        order = np.argsort(terms[:, 0], kind="stable")
        grouped = np.ascontiguousarray(terms[order])
        counts = np.bincount(terms[:, 0].astype(np.int64), minlength=weight.num_segments)
        offsets = np.zeros(weight.num_segments + 1, dtype=np.int32)
        # ``cumsum`` is int64; a narrowing store must be explicit. A wrapped
        # total would make the grouped ranges disagree with the path list.
        narrowed = np.cumsum(counts, dtype=np.int64).astype(np.int32)
        offsets[1:] = narrowed
        if int(offsets[-1]) != terms.shape[0]:
            raise ValueError("contraction term count does not fit in int32 offsets")
        self.channels = channels
        self.weight_segments = weight.num_segments
        self.input_segments = features.num_segments
        self.output_segments = output.num_segments
        self.projection = np.ascontiguousarray(projection)
        self.terms = np.ascontiguousarray(terms)
        self.coefficients = np.ascontiguousarray(coefficients)
        self.grouped_terms = grouped
        self.grouped_coefficients = np.ascontiguousarray(coefficients[order])
        self.weight_offsets = offsets
