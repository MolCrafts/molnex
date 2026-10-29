"""The O(3) coupling convention MACE checkpoints are trained in.

A symmetric contraction's weights are coefficients in a basis of coupled
products, and that basis depends on the phase of every Clebsch-Gordan tensor
it is built from. cuEquivariance's ``meq.O3`` and the convention MACE trains in
agree for every coupling into ``l <= 1`` but not beyond, so a MACE checkpoint
whose product emits ``2e`` (MACE-POLAR-1-L) evaluated with ``meq.O3`` returns a
plausible, wrong ``l = 2`` channel. :class:`MaceO3` is ``meq.O3`` with MACE's
Clebsch-Gordan tensors; the irreps, their order and the parity rule are
unchanged.

MACE's real Clebsch-Gordan tensor is the complex one (Racah's formula) in the
real basis ``Y_{l,m}`` reached by

    Q_l[l+m, l+|m|] = (-1)^m / sqrt(2),  Q_l[l+m, l-|m|] =  i (-1)^m / sqrt(2)   (m > 0)
    Q_l[l, l]       = 1
    Q_l[l+m, l+|m|] = 1 / sqrt(2),       Q_l[l+m, l-|m|] = -i / sqrt(2)          (m < 0)

times ``(-i)^l``, normalised to unit Frobenius norm and signed as in
:func:`real_coupling`, then scaled by ``sqrt(2 l_3 + 1)``.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from functools import cache
from math import factorial, sqrt

import numpy as np

import molrep.eq as meq


def _complex_cg(l1: int, l2: int, l3: int) -> np.ndarray:
    """``<l1 m1 l2 m2 | l3 m3>`` (Racah), indexed ``[l1+m1, l2+m2, l3+m3]``."""
    out = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1))
    prefactor_l = (2 * l3 + 1) * factorial(l3 + l1 - l2) * factorial(l3 - l1 + l2)
    prefactor_l *= factorial(l1 + l2 - l3) / factorial(l1 + l2 + l3 + 1)
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            m3 = m1 + m2
            if abs(m3) > l3:
                continue
            prefactor = prefactor_l * factorial(l3 + m3) * factorial(l3 - m3)
            prefactor *= factorial(l1 - m1) * factorial(l1 + m1)
            prefactor *= factorial(l2 - m2) * factorial(l2 + m2)
            total = 0.0
            for k in range(
                max(0, l2 - l3 - m1, l1 - l3 + m2), min(l1 + l2 - l3, l1 - m1, l2 + m2) + 1
            ):
                total += (-1) ** k / (
                    factorial(k)
                    * factorial(l1 + l2 - l3 - k)
                    * factorial(l1 - m1 - k)
                    * factorial(l2 + m2 - k)
                    * factorial(l3 - l2 + m1 + k)
                    * factorial(l3 - l1 - m2 + k)
                )
            out[l1 + m1, l2 + m2, l3 + m3] = sqrt(prefactor) * total
    return out


def _real_to_complex(l: int) -> np.ndarray:
    """Change of basis ``Q_l`` from MACE's real harmonics to the complex ones."""
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / sqrt(2)
        q[l + m, l - abs(m)] = -1j / sqrt(2)
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / sqrt(2)
        q[l + m, l - abs(m)] = 1j * (-1) ** m / sqrt(2)
    return (-1j) ** l * q


def _canonical(l1: int, l2: int, l3: int) -> np.ndarray:
    """Unit-norm coupling for ``l1 <= l2 <= l3`` with MACE's sign.

    The sign makes the all-``m = 0`` entry positive, or, where that entry
    vanishes, the first non-zero entry in row-major order.
    """
    q1, q2, q3 = _real_to_complex(l1), _real_to_complex(l2), _real_to_complex(l3)
    c = np.einsum("ij,kl,mn,ikn->jlm", q1, q2, np.conj(q3.T), _complex_cg(l1, l2, l3))
    if np.abs(c.imag).max() > 1e-10:
        raise ValueError(f"coupling ({l1}, {l2}, {l3}) is not real in MACE's basis")
    c = c.real / np.linalg.norm(c.real)
    c[np.abs(c) < 1e-14] = 0.0
    pivot = c[l1, l2, l3] if c[l1, l2, l3] != 0 else c.ravel()[np.flatnonzero(c)[0]]
    return -c if pivot < 0 else c


@cache
def real_coupling(l1: int, l2: int, l3: int) -> np.ndarray:
    """MACE's real coupling tensor ``(2l1+1, 2l2+1, 2l3+1)``, unit Frobenius norm.

    Only sorted degrees have a canonical sign; any other order is the sorted
    tensor with its axes permuted, times ``(-1)^(l1+l2+l3)`` for an odd
    permutation. Where degrees tie, the later case below decides, as in MACE.

    Args:
        l1: First input degree.
        l2: Second input degree.
        l3: Output degree, ``|l1 - l2| <= l3 <= l1 + l2``.

    Returns:
        The real tensor.
    """
    odd = (-1) ** (l1 + l2 + l3)
    out = None
    if l1 <= l2 <= l3:
        out = _canonical(l1, l2, l3)
    if l1 <= l3 <= l2:
        out = odd * np.transpose(_canonical(l1, l3, l2), (0, 2, 1))
    if l2 <= l1 <= l3:
        out = odd * np.transpose(_canonical(l2, l1, l3), (1, 0, 2))
    if l3 <= l2 <= l1:
        out = odd * np.transpose(_canonical(l3, l2, l1), (2, 1, 0))
    if l2 <= l3 <= l1:
        out = np.transpose(_canonical(l2, l3, l1), (2, 0, 1))
    if l3 <= l1 <= l2:
        out = np.transpose(_canonical(l3, l1, l2), (1, 2, 0))
    assert out is not None
    return out


class MaceO3(meq.O3):
    """``meq.O3`` with the Clebsch-Gordan phases MACE checkpoints were trained in."""

    def __mul__(rep1: MaceO3, rep2: MaceO3) -> Iterator[MaceO3]:  # noqa: N805
        return [MaceO3(l=ir.l, p=ir.p) for ir in meq.O3.__mul__(rep1, rep2)]

    @classmethod
    def clebsch_gordan(cls, rep1: MaceO3, rep2: MaceO3, rep3: MaceO3) -> np.ndarray:
        """Coupling tensors ``(num_paths, d1, d2, d3)`` for ``rep1 ⊗ rep2 → rep3``."""
        rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)
        if rep1.p * rep2.p == rep3.p and abs(rep1.l - rep2.l) <= rep3.l <= rep1.l + rep2.l:
            return real_coupling(rep1.l, rep2.l, rep3.l)[None] * sqrt(rep3.dim)
        return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    def __lt__(rep1: MaceO3, rep2: MaceO3) -> bool:  # noqa: N805
        rep2 = rep1._from(rep2)
        return (rep1.l, rep1.p) < (rep2.l, rep2.p)

    @classmethod
    def iterator(cls) -> Iterator[MaceO3]:
        """Irreps in MACE's order: ``0e, 0o, 1o, 1e, 2e, 2o, ...``."""
        for l in itertools.count(0):
            yield MaceO3(l=l, p=1 * (-1) ** l)
            yield MaceO3(l=l, p=-1 * (-1) ** l)
