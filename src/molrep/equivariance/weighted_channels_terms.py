"""Term table of Allegro weighted channels, read off its polynomial.

Allegro builds subscripts ``",u,u"``: one scalar basis segment per component,
one weight segment of shape ``(channels,)`` per irrep, and one output segment
of the same shape per component. Each path carries a scalar coefficient and
pairs a component with its irrep. ``eval_last_operand`` makes that last
operand the output.

A term is two ``int16`` indices ``(component, irrep)``. At each channel ``c``,

```
y[component, c] += C * basis[component] * weight[irrep, c]
```

An index sums those rows into the addressed atom. The weight is one row per
edge.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nat. Commun. 14, 579 (2023).
    https://arxiv.org/abs/2204.05249
"""

from __future__ import annotations

import numpy as np

import molrep.eq as meq

#: Allegro's ``l_max`` of 3 is 16 components. ``l = 0..7`` is 64.
MAX_COMPONENTS = 64
#: One weight segment per irrep, stored as ``int16``.
MAX_IRREPS = 64


class WeightedChannelsTerms:
    """Scalar paths of one Allegro channel weighting.

    Attributes:
        channels: Multiplicity ``u``.
        num_irreps: Weight segments, one per irrep.
        components: Basis and output segments.
        terms: ``(T, 2)`` ``int16``, columns ``component, irrep``.
        coefficients: ``(T,)`` ``float64``.

    Raises:
        ValueError: The dimensions, channel count, or descriptor are outside
            the kernel.
    """

    def __init__(self, irrep_dims: list[int], channels: int) -> None:
        channels = int(channels)
        dims = [int(dim) for dim in irrep_dims]
        if channels < 1:
            raise ValueError(f"native weighted channels requires channels >= 1, got {channels}")
        if len(dims) < 1:
            raise ValueError("native weighted channels requires at least one irrep")
        if len(dims) > MAX_IRREPS:
            raise ValueError(f"native weighted channels supports at most {MAX_IRREPS} irreps")
        if any(dim < 1 for dim in dims):
            raise ValueError("irrep dimension must be positive")
        components = sum(dims)
        if components > MAX_COMPONENTS:
            raise ValueError(
                f"native weighted channels supports at most {MAX_COMPONENTS} components, "
                f"got {components}"
            )
        descriptor = meq.SegmentedTensorProduct.from_subscripts(",u,u")
        for _ in dims:
            descriptor.add_segment(1, (channels,))
        for irrep, dim in enumerate(dims):
            for _ in range(dim):
                component = descriptor.add_segment(0, ())
                descriptor.add_segment(2, (channels,))
                descriptor.add_path(component, irrep, component, c=1.0)
        polynomial = meq.SegmentedPolynomial.eval_last_operand(descriptor)
        if len(polynomial.operations) != 1:
            raise ValueError(
                f"native weighted channels expects one operation, got {len(polynomial.operations)}"
            )
        operation, product = polynomial.operations[0]
        if operation.buffers != (0, 1, 2):
            raise ValueError(f"unsupported weighted-channels operation {operation.buffers}")
        if product.coefficient_subscripts != "":
            raise ValueError(
                "native weighted channels requires scalar path coefficients, "
                f"got {product.coefficient_subscripts!r}"
            )
        basis, weight = polynomial.inputs
        (output,) = polynomial.outputs
        if basis.num_segments != components or output.num_segments != components:
            raise ValueError("basis and output segment counts must equal the component count")
        if weight.num_segments != len(dims):
            raise ValueError("weight segment count must equal the irrep count")
        if any(segment != () for segment in basis.segments):
            raise ValueError("basis segments must be scalars")
        expected = (channels,)
        if any(segment != expected for segment in weight.segments):
            raise ValueError(f"weight segments must have shape {expected}")
        if any(segment != expected for segment in output.segments):
            raise ValueError(f"output segments must have shape {expected}")
        rows: list[tuple[int, int]] = []
        coefficients: list[float] = []
        for path in product.paths:
            if len(path.indices) != 3:
                raise ValueError("a weighted-channels path has three segment indices")
            component, irrep, out = (int(index) for index in path.indices)
            if component != out:
                raise ValueError("native weighted channels pairs each component with itself")
            if not 0 <= component < components or not 0 <= irrep < len(dims):
                raise ValueError("path index is outside the descriptor")
            coefficient = np.asarray(path.coefficients, dtype=np.float64)
            if coefficient.shape != ():
                raise ValueError("native weighted channels requires scalar path coefficients")
            rows.append((component, irrep))
            coefficients.append(float(coefficient))
        if len(rows) == 0:
            raise ValueError("native weighted channels produced no path")
        if [component for component, _ in rows] != list(range(components)):
            raise ValueError("native weighted channels expects one path per component, in order")
        self.channels = channels
        self.num_irreps = len(dims)
        self.components = components
        self.terms = np.asarray(rows, dtype=np.int16)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
