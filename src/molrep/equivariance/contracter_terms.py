"""Term table of the Allegro Contracter, read off its flattened polynomial.

Allegro builds ``u,iu,ju,ku+ijk``, one Clebsch-Gordan matrix per path, then
``eval_last_operand`` and ``flatten_coefficient_modes``. Flattening folds the
irrep-component modes into the segment index and leaves one scalar coefficient
per component path. Several of those paths share one weight segment: the
segment count stays the unflattened path count, which is the width of
``Contracter.weights``.

A term is four ``int16`` indices ``(w, i, j, o)``, the table sorted by
``(i, j, w, o)``. At each channel ``c``,

```
y[o, c] += C * W[w, c] * x1[i, c] * x2[j, c]
```

``x2`` is gathered per edge. The weight is one shared row, not one row per
element. The term table serves the reference evaluation; the CUDA kernel
reads the same product one path at a time — ``paths`` holds each path's
coupling class ``(l1, l2, l3)`` and the component offsets of its three
irreps, and ``cg_values`` the Clebsch–Gordan coefficients of every class in
the flat order of ``csrc/equivariance/cg.hpp`` (zero at a candidate position
the descriptor does not fill). A coefficient outside the selection rule of
that table, or two paths of one class with different coefficients, is
refused at construction rather than computed wrongly.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nat. Commun. 14, 579 (2023).
    https://arxiv.org/abs/2204.05249
"""

from __future__ import annotations

import itertools

import numpy as np

import molrep.eq as meq

from .descriptor import CG_LMAX, cg_class_id, cg_offsets, cg_positions

#: Component counts. ``l = 0..7`` is 64; Allegro's default ``l_max`` is 2.
MAX_SEGMENTS = 64
#: Signed ``int16`` path indices. Segment ``32767`` would not fit.
MAX_WEIGHT_SEGMENTS = 32767


class ContracterTerms:
    """Scalar monomials of one Allegro tensor product.

    Attributes:
        channels: Shared multiplicity ``u``.
        weight_segments: Unflattened Clebsch-Gordan path count.
        input1_segments: Components of the pair operand.
        input2_segments: Components of the gathered atom operand.
        output_segments: Components of the output.
        terms: ``(T, 4)`` ``int16``, columns ``w, i, j, o``.
        coefficients: ``(T,)`` ``float64``.
        paths: ``(P, 4)`` ``int32``: class id and the component offsets of
            the pair, atom and output irreps of each path, in path order.
        cg_values: The flat class coefficient table, ``float64``.

    Raises:
        ValueError: The irreps, channel count, or flattened descriptor are
            outside the kernel.
    """

    def __init__(
        self,
        irreps_in1: meq.Irreps,
        irreps_in2: meq.Irreps,
        irreps_out: meq.Irreps,
        channels: int,
    ) -> None:
        channels = int(channels)
        if channels < 1:
            raise ValueError(f"native contracter requires channels >= 1, got {channels}")
        for name, irreps in (
            ("pair", irreps_in1),
            ("atom", irreps_in2),
            ("output", irreps_out),
        ):
            if len(irreps) == 0 or any(mul != 1 for mul in irreps.muls):
                raise ValueError(
                    "native contracter takes one copy of each irrep; "
                    f"the {name} operand is {irreps}"
                )
            if sum(ir.dim for _, ir in irreps) > MAX_SEGMENTS:
                raise ValueError(
                    f"native contracter supports at most {MAX_SEGMENTS} "
                    f"{name} components, got {irreps}"
                )
        descriptor = meq.SegmentedTensorProduct.from_subscripts("u,iu,ju,ku+ijk")
        for _, ir in irreps_in1:
            descriptor.add_segment(1, (ir.dim, channels))
        for _, ir in irreps_in2:
            descriptor.add_segment(2, (ir.dim, channels))
        for _, ir in irreps_out:
            descriptor.add_segment(3, (ir.dim, channels))
        num_paths = 0
        offsets = cg_offsets()
        values = np.zeros(offsets[("total",)])
        filled: set[tuple[int, int, int]] = set()
        paths: list[tuple[int, int, int, int]] = []
        starts = [
            np.concatenate([[0], np.cumsum([ir.dim for _, ir in irreps])]).astype(int)
            for irreps in (irreps_in1, irreps_in2, irreps_out)
        ]
        for (i3, (_, ir3)), (i1, (_, ir1)), (i2, (_, ir2)) in itertools.product(
            enumerate(irreps_out), enumerate(irreps_in1), enumerate(irreps_in2)
        ):
            if ir3 in ir1 * ir2:
                for cg in meq.clebsch_gordan(ir1, ir2, ir3):
                    descriptor.add_path(None, i1, i2, i3, c=cg)
                    num_paths += 1
                    cls = (ir1.l, ir2.l, ir3.l)
                    if max(cls) > CG_LMAX:
                        raise ValueError(
                            f"native contracter supports l <= {CG_LMAX}, got {ir1} x {ir2} -> {ir3}"
                        )
                    block = np.asarray(cg, dtype=np.float64)
                    positions = cg_positions(*cls)
                    where = {pos: n for n, pos in enumerate(positions)}
                    for m1, m2, m3 in zip(*np.nonzero(block), strict=True):
                        key = (int(m1), int(m2), int(m3))
                        if key not in where:
                            raise ValueError(
                                f"a Clebsch-Gordan coefficient of {ir1} x {ir2} -> {ir3} at "
                                f"{key} lies outside the real-basis selection rule"
                            )
                    table = np.zeros(len(positions))
                    for n, (m1, m2, m3) in enumerate(positions):
                        table[n] = block[m1, m2, m3]
                    start = offsets[cls]
                    if cls in filled:
                        if not np.allclose(values[start : start + len(positions)], table):
                            raise ValueError(
                                f"two paths of class {cls} carry different coefficients"
                            )
                    else:
                        values[start : start + len(positions)] = table
                        filled.add(cls)
                    paths.append(
                        (
                            cg_class_id(*cls),
                            int(starts[0][i1]),
                            int(starts[1][i2]),
                            int(starts[2][i3]),
                        )
                    )
        if num_paths == 0 or num_paths > MAX_WEIGHT_SEGMENTS:
            raise ValueError(
                f"no tensor-product path {irreps_in1} x {irreps_in2} -> {irreps_out}"
                if num_paths == 0
                else f"native contracter supports at most {MAX_WEIGHT_SEGMENTS} weight segments"
            )
        polynomial = meq.SegmentedPolynomial.eval_last_operand(
            descriptor
        ).flatten_coefficient_modes()
        if len(polynomial.operations) != 1:
            raise ValueError(
                f"native contracter expects one operation, got {len(polynomial.operations)}"
            )
        operation, product = polynomial.operations[0]
        if operation.buffers != (0, 1, 2, 3):
            raise ValueError(f"unsupported contracter operation {operation.buffers}")
        if product.coefficient_subscripts != "":
            raise ValueError(
                "native contracter requires scalar path coefficients, "
                f"got {product.coefficient_subscripts!r}"
            )
        weight, features_1, features_2 = polynomial.inputs
        (output,) = polynomial.outputs
        for operand, name in (
            (weight, "weight"),
            (features_1, "pair"),
            (features_2, "atom"),
            (output, "output"),
        ):
            if operand.segment_shape != (channels,):
                raise ValueError(
                    f"native contracter requires every {name} segment to have "
                    f"shape {(channels,)}, got {operand.segment_shape}"
                )
        if weight.num_segments != num_paths:
            raise ValueError(
                "flattened weight segments do not match the unflattened path count "
                f"({weight.num_segments} vs {num_paths})"
            )
        if (
            features_1.num_segments > MAX_SEGMENTS
            or features_2.num_segments > MAX_SEGMENTS
            or output.num_segments > MAX_SEGMENTS
        ):
            raise ValueError(
                f"native contracter supports at most {MAX_SEGMENTS} components per operand"
            )
        rows: list[tuple[int, int, int, int]] = []
        values_flat: list[float] = []
        for path in product.paths:
            if path.coefficients.shape != ():
                raise ValueError(
                    f"path coefficient has shape {path.coefficients.shape}, expected a scalar"
                )
            if len(path.indices) != 4:
                raise ValueError(f"contracter path has indices {path.indices}")
            wseg, i, j, oseg = (int(item) for item in path.indices)
            if not (
                0 <= wseg < weight.num_segments
                and 0 <= i < features_1.num_segments
                and 0 <= j < features_2.num_segments
                and 0 <= oseg < output.num_segments
            ):
                raise ValueError(f"path indices {path.indices} fall outside the operands")
            rows.append((wseg, i, j, oseg))
            values_flat.append(float(path.coefficients))
        # Sorted by (i, j, w, o): the compiled kernels treat the terms of one
        # product x1[i] x2[j] as a run.
        order = sorted(
            range(len(rows)), key=lambda n: (rows[n][1], rows[n][2], rows[n][0], rows[n][3])
        )
        rows = [rows[n] for n in order]
        values_flat = [values_flat[n] for n in order]
        self.channels = channels
        self.weight_segments = weight.num_segments
        self.input1_segments = features_1.num_segments
        self.input2_segments = features_2.num_segments
        self.output_segments = output.num_segments
        self.terms = np.ascontiguousarray(np.asarray(rows, dtype=np.int16).reshape(-1, 4))
        self.coefficients = np.ascontiguousarray(np.asarray(values_flat, dtype=np.float64))
        self.paths = np.ascontiguousarray(np.asarray(paths, dtype=np.int32).reshape(-1, 4))
        self.cg_values = np.ascontiguousarray(values)
