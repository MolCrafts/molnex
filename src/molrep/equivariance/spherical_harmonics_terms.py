"""Term table of real spherical harmonics, read off the cuEquivariance descriptor.

Each path is one monomial in the three Cartesian components. A term is four
``int16`` indices ``(i, j, k, o)``. ``j`` or ``k`` is ``-1`` when the degree
does not use that slot, and all three are ``-1`` for the degree-0 constant.
The value is

```
Y[o] += C * v[i]? * v[j]? * v[k]?
```

Repeated component indices stay repeated slots.

Reference:
    Real spherical harmonics in the cuEquivariance ``SO3`` convention.
"""

from __future__ import annotations

import numpy as np

import molrep.eq as meq

#: ``l = 3`` is the MACE default and the highest degree this kernel evaluates.
MAX_L = 3
#: ``(l + 1) ** 2`` at ``l = 3``.
MAX_COMPONENTS = 16


class SphericalHarmonicsTerms:
    """Monomials of ``Y_lm`` for one sorted list of degrees.

    Attributes:
        ls: The degrees, strictly increasing.
        components: Output width, ``sum(2 l + 1)``.
        terms: ``(T, 4)`` ``int16``, columns ``i, j, k, o``.
        coefficients: ``(T,)`` ``float64``.

    Raises:
        ValueError: A degree is outside ``0..3``, the list is empty or
            unsorted, or the descriptor is not the scalar-segment polynomial
            this kernel evaluates.
    """

    def __init__(self, ls: list[int]) -> None:
        if len(ls) == 0 or any(ell < 0 or ell > MAX_L for ell in ls) or list(ls) != sorted(set(ls)):
            raise ValueError(
                f"native spherical harmonics supports strictly increasing degrees in 0..{MAX_L}, "
                f"got {ls}"
            )
        polynomial = meq.descriptors.spherical_harmonics(meq.SO3(1), list(ls)).polynomial
        (vector,) = polynomial.inputs
        (output,) = polynomial.outputs
        if vector.num_segments != 3 or vector.segment_shape != ():
            raise ValueError("native spherical harmonics reads a 3-vector of scalar components")
        if output.segment_shape != () or output.num_segments > MAX_COMPONENTS:
            raise ValueError(
                f"native spherical harmonics supports at most {MAX_COMPONENTS} components"
            )
        expected = sum(2 * ell + 1 for ell in ls)
        if output.num_segments != expected:
            raise ValueError(
                f"spherical-harmonics output width {output.num_segments} is not {expected}"
            )
        rows: list[tuple[int, int, int, int]] = []
        values: list[float] = []
        for operation, product in polynomial.operations:
            degree = operation.buffers.count(0)
            if operation.buffers != (0,) * degree + (1,):
                raise ValueError(f"unsupported spherical-harmonics operation {operation.buffers}")
            if degree > MAX_L:
                raise ValueError(f"spherical-harmonics degree {degree} exceeds {MAX_L}")
            if product.coefficient_subscripts != "":
                raise ValueError("native spherical harmonics requires scalar path coefficients")
            for path in product.paths:
                if path.coefficients.shape != ():
                    raise ValueError(
                        f"path coefficient has shape {path.coefficients.shape}, expected a scalar"
                    )
                slots = list(path.indices[:-1])
                if len(slots) != degree:
                    raise ValueError(f"degree {degree} path has indices {path.indices}")
                while len(slots) < 3:
                    slots.append(-1)
                oseg = path.indices[-1]
                if (slots[1] >= 0 and slots[0] < 0) or (slots[2] >= 0 and slots[1] < 0):
                    raise ValueError(f"path slots {slots} are not packed")
                entry = (slots[0], slots[1], slots[2], oseg)
                if not (
                    0 <= oseg < output.num_segments
                    and all(item == -1 or 0 <= item < 3 for item in slots)
                ):
                    raise ValueError(f"path indices {entry} fall outside the operands")
                rows.append(entry)
                values.append(float(path.coefficients))
        self.ls = list(ls)
        self.components = output.num_segments
        self.terms = np.ascontiguousarray(np.asarray(rows, dtype=np.int16).reshape(-1, 4))
        self.coefficients = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
