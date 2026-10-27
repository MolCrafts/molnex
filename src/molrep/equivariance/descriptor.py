"""Construction-time lowering of a cuEquivariance channelwise descriptor onto the native term table.

The term table covers MACE's natural-parity O3 family —
``irreps_in1 = U×0e + U×1o + … + U×L1``, ``irreps_in2 = 0e + 1o + … + L2`` and
the output filter ``0e + … + L2`` — with every Clebsch–Gordan *position*
enumerated by the selection rule of the real spherical-harmonic basis (the
CUDA kernels enumerate the same table at compile time,
``csrc/equivariance/channelwise.cuh``). This module builds cuEquivariance's
descriptor for the requested irreps, checks that it has exactly that structure
(path order, segment layout, every non-zero coefficient inside the rule's
candidate set) and reads the coefficient *values* off it in the kernel's term
order. No Clebsch–Gordan algebra is implemented here.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

import molrep.eq as meq

#: What the CUDA kernels are instantiated for (``csrc/equivariance/dispatch.cuh``):
#: channel multiplicities (one thread per channel), node orders and angular /
#: output orders. The term table itself is built for any uniform multiplicity
#: and any ``(l1, l2)``; the CPU reference evaluates all of them, the CUDA
#: operator refuses what it has no kernel for.
CUDA_CHANNELS = (8, 16, 32, 64, 128)  # and any multiple of 128, in tiles of 128
CUDA_CONFIGS = ((0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3))


def natural_irreps(mul: int, lmax: int) -> meq.Irreps:
    """``mul×0e + mul×1o + … + mul×lmax`` with parity ``(-1)^l``."""
    return meq.Irreps(
        "O3", "+".join(f"{mul}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1))
    )


def candidate_m3(m1: int, m2: int, odd: bool = False) -> list[int]:
    """Output components ``m3`` at which a real-basis product ``(m1, m2)`` can be non-zero.

    For an even ``l1 + l2 + l3`` (``odd=False``, the natural-parity couplings)
    factors of the same type — both cosine (``m ≥ 0``) or both sine — combine
    to cosine type at ``|m1|+|m2|`` and ``||m1|−|m2||``, mixed types to sine
    type (``m < 0``), and a zonal factor (``m = 0``) leaves the other's ``m``
    unchanged. For an odd sum (the antisymmetric couplings, ``1 × 1 → 1``) the
    types swap and the zonal pass-through is negated. The same rule is
    ``csrc/equivariance/cg.hpp``; verified against cuEquivariance's O3 tables
    for every ``l ≤ 3`` (34 classes, 611 non-zeros, no violation).
    """
    if m1 == 0:
        return [-m2 if odd else m2]
    if m2 == 0:
        return [-m1 if odd else m1]
    s, d = abs(m1) + abs(m2), abs(abs(m1) - abs(m2))
    if ((m1 > 0) == (m2 > 0)) != odd:
        return [s, d]
    return [-s] + ([-d] if d else [])


#: Highest ``l`` of the compile-time Clebsch–Gordan classes (``cg.hpp``).
CG_LMAX = 3


def cg_classes() -> list[tuple[int, int, int]]:
    """Every coupling ``(l1, l2, l3)`` with ``l ≤ 3`` the triangle rule allows, in table order."""
    return [
        (l1, l2, l3)
        for l1 in range(CG_LMAX + 1)
        for l2 in range(CG_LMAX + 1)
        for l3 in range(CG_LMAX + 1)
        if abs(l1 - l2) <= l3 <= l1 + l2
    ]


def cg_positions(l1: int, l2: int, l3: int) -> list[tuple[int, int, int]]:
    """Candidate ``(m1 + l1, m2 + l2, m3 + l3)`` of one class, in the kernel's order."""
    odd = (l1 + l2 + l3) % 2 == 1
    return [
        (m1 + l1, m2 + l2, m3 + l3)
        for m1 in range(-l1, l1 + 1)
        for m2 in range(-l2, l2 + 1)
        for m3 in candidate_m3(m1, m2, odd)
        if abs(m3) <= l3
    ]


def cg_class_id(l1: int, l2: int, l3: int) -> int:
    """The id a path carries, ``(l1, l2, l3)`` in base 4 (``cg.hpp``)."""
    return (l1 * (CG_LMAX + 1) + l2) * (CG_LMAX + 1) + l3


def cg_offsets() -> dict[tuple[int, int, int], int]:
    """Start of each class in the flat coefficient table."""
    offsets, total = {}, 0
    for cls in cg_classes():
        offsets[cls] = total
        total += len(cg_positions(*cls))
    offsets[("total",)] = total
    return offsets


def paths(l1_max: int, l2_max: int) -> Iterator[tuple[int, int, int, int, int]]:
    """``(l1, l2, l3, segment, first output row)`` per path.

    Paths are enumerated as cuEquivariance does (``l1`` outer, ``l2``, ``l3``
    ascending, natural parity); the segment index is the path's position after
    cuEquivariance's stable sort of the output irreps by ``l3``, which its
    weight segments share.
    """
    enumerated = [
        (l1, l2, l3)
        for l1 in range(l1_max + 1)
        for l2 in range(l2_max + 1)
        for l3 in range(abs(l1 - l2), min(l1 + l2, l2_max) + 1)
        if (l1 + l2 + l3) % 2 == 0
    ]
    order = sorted(range(len(enumerated)), key=lambda q: (enumerated[q][2], q))
    segment = {q: seg for seg, q in enumerate(order)}
    rows = {}
    row = 0
    for q in order:
        rows[q] = row
        row += 2 * enumerated[q][2] + 1
    for q, (l1, l2, l3) in enumerate(enumerated):
        yield l1, l2, l3, segment[q], rows[q]


def terms(l1_max: int, l2_max: int) -> Iterator[tuple[int, int, int, int]]:
    """``(node row, angular component, path, output row)`` in the kernel's term order."""
    for l1, l2, l3, p, row in paths(l1_max, l2_max):
        for i in range(2 * l1 + 1):
            for j in range(2 * l2 + 1):
                for m3 in candidate_m3(i - l1, j - l2):
                    if abs(m3) <= l3:
                        yield l1 * l1 + i, l2 * l2 + j, p, row + m3 + l3


class ChannelwiseDescriptor:
    """The native term table of one channelwise product.

    Attributes:
        l1: Highest order of the node operand.
        l2: Highest order of the angular operand and of the output.
        channels: Uniform channel multiplicity ``U``.
        terms: ``(i, j, p, k)`` per term, the kernel's order.
        coefficients: The Clebsch–Gordan value of each term (a host tuple:
            the kernels take it as a parameter).
        irreps_out: Output irreps, one segment per path.
        weight_numel: Flat per-edge weight width ``paths × U``.
        input_dims: ``(node width, angular width)``.
        output_dim: Output width.

    Raises:
        ValueError: If the irreps are outside the compiled family or the
            cuEquivariance descriptor does not have the enumerated structure.
    """

    def __init__(self, irreps_in1: meq.Irreps, irreps_in2: meq.Irreps, filter_irreps_out) -> None:
        if irreps_in1.irrep_class != meq.O3 or irreps_in2.irrep_class != meq.O3:
            raise ValueError("native channelwise supports O3 only")
        muls = {mi.mul for mi in irreps_in1}
        if len(muls) != 1:
            raise ValueError("native channelwise requires one uniform channel multiplicity")
        channels = muls.pop()
        l1, l2 = len(irreps_in1) - 1, len(irreps_in2) - 1
        if l1 < 0 or irreps_in1 != natural_irreps(channels, l1):
            raise ValueError(
                f"native channelwise requires node irreps {channels}x0e+{channels}x1o+… "
                f"(natural parity), got {irreps_in1}"
            )
        if l2 < 0 or irreps_in2 != natural_irreps(1, l2):
            raise ValueError(
                f"native channelwise requires angular irreps 0e+1o+2e+… (natural parity), "
                f"got {irreps_in2}"
            )
        expression = meq.descriptors.channelwise_tensor_product(
            irreps_in1, irreps_in2, filter_irreps_out
        )
        polynomial = expression.polynomial
        ((_, product),) = polynomial.operations
        if product.subscripts != "uv,iu,jv,kuv+ijk":
            raise ValueError(f"unsupported channelwise descriptor: {product.subscripts}")
        weight, node, angular = polynomial.inputs
        (output,) = polynomial.outputs
        expected = list(paths(l1, l2))
        got = [tuple(path.indices) for path in product.paths]
        if got != [(p, pl1, pl2, p) for (pl1, pl2, _, p, _) in expected] or [
            output.segments[o][0] for (_, _, _, o) in got
        ] != [2 * pl3 + 1 for (_, _, pl3, _, _) in expected]:
            raise ValueError(
                "the cuEquivariance descriptor does not enumerate the natural-parity paths "
                f"of l1={l1}, l2={l2} (filter {filter_irreps_out}); native channelwise needs "
                "the output filter 0e+1o+…+l2"
            )
        if any(seg != (channels, 1) for seg in weight.segments):
            raise ValueError("native channelwise requires v=1 weights (single-copy harmonics)")
        table = list(terms(l1, l2))
        position = {t: n for n, t in enumerate(table)}
        coefficients = np.zeros(len(table))
        for path, (pl1, pl2, pl3, p, row) in zip(product.paths, expected, strict=True):
            cg = np.asarray(path.coefficients)
            if cg.shape != (2 * pl1 + 1, 2 * pl2 + 1, 2 * pl3 + 1):
                raise ValueError(f"unexpected coefficient block {cg.shape} on path {p}")
            for i, j, k in zip(*np.nonzero(cg), strict=True):
                key = (pl1 * pl1 + int(i), pl2 * pl2 + int(j), p, row + int(k))
                if key not in position:
                    raise ValueError(
                        f"coefficient at (l1={pl1}, m1={int(i) - pl1}; l2={pl2}, "
                        f"m2={int(j) - pl2}; l3={pl3}, m3={int(k) - pl3}) lies outside the "
                        "real-basis selection rule"
                    )
                coefficients[position[key]] = float(cg[i, j, k])
        self.l1, self.l2, self.channels = l1, l2, channels
        self.terms: tuple[tuple[int, int, int, int], ...] = tuple(table)
        self.coefficients: tuple[float, ...] = tuple(float(c) for c in coefficients)
        self.irreps_out = expression.operands[-1].irreps
        self.weight_numel = int(weight.size)
        self.input_dims = (int(node.size), int(angular.size))
        self.output_dim = int(output.size)
