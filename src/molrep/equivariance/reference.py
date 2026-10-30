"""Reference evaluation of the native equivariance operators in plain PyTorch.

Every operator of ``libmolnex_equivariance`` is a sum of scalar monomials read
off a term table: ``out[row, o, c] += C_t · Π_f factor_f[row, index_f(t), c]``.
This module evaluates those tables with ordinary tensor operations, so the
same term table runs on a CPU tensor (where the library has no kernel) and
serves as the oracle the CUDA kernels are tested against. Autograd — of any
order — is PyTorch's own here; the CUDA path registers its fused adjoints and
the composed higher orders in :mod:`molrep.equivariance.backend`.

The evaluation is exact in the sense of the term table (every monomial, in
the compute dtype); the summation order differs from the kernels', so the two
agree to rounding, not bitwise. Memory is bounded by ``budget`` elements per
chunk of terms, which keeps a CPU evaluation of a production batch within
reach without materialising the full ``(rows, terms, channels)`` product.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .backend import EdgeSchedule

#: Elements of one ``(rows, chunk, channels)`` product the term loop allows.
BUDGET = 1 << 26


def _chunks(terms: int, rows: int, width: int, budget: int = BUDGET) -> int:
    """Terms per chunk so that one ``(rows, chunk, width)`` product fits ``budget``."""
    return max(1, min(terms, budget // max(1, rows * width)))


def monomials(
    factors: Sequence[tuple[torch.Tensor, torch.Tensor]],
    coefficients: torch.Tensor,
    out_index: torch.Tensor,
    rows: int,
    out_segments: int,
) -> torch.Tensor:
    """``Σ_t C_t · Π_f factor_f[:, index_f[t]]`` scattered into ``out_index[t]``.

    Args:
        factors: ``(tensor, index)`` pairs. A tensor is ``(rows, segments,
            channels)`` or ``(rows, segments)`` (broadcast over the channels);
            a row count of 1 broadcasts over the rows. ``index`` is ``(T,)``.
        coefficients: ``(T,)`` in the output dtype.
        out_index: ``(T,)`` output segment of each term.
        rows: Output rows (the row count of the per-row factors).
        out_segments: Output segment count.

    Returns:
        ``(rows, out_segments, channels)``.
    """
    channels = max((tensor.shape[2] for tensor, _ in factors if tensor.ndim == 3), default=1)
    lead, _ = factors[0]
    out = lead.new_zeros(rows, out_segments, channels)
    terms = int(coefficients.numel())
    if terms == 0:
        return out
    step = _chunks(terms, rows, channels)
    for start in range(0, terms, step):
        stop = min(terms, start + step)
        product = coefficients[start:stop].view(1, -1, 1)
        for tensor, index in factors:
            gathered = tensor[:, index[start:stop]]
            product = product * (gathered if gathered.ndim == 3 else gathered.unsqueeze(-1))
        if product.shape[0] == 1 and rows > 1:
            product = product.expand(rows, -1, -1)
        out = out.index_add(1, out_index[start:stop], product)
    return out


def schedule(
    source: torch.Tensor, target: torch.Tensor, rows_source: int, rows_target: int
) -> EdgeSchedule:
    """The stable CSR views of an edge list, as ``molnex_equivariance::schedule``."""

    def csr(index: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        order = torch.argsort(index, stable=True).to(torch.int32)
        # Counted into ``rows`` slots: ``bincount``'s length follows the data
        # (its largest index), a dynamic shape under compile.
        counts = index.new_zeros(rows).index_add(0, index, torch.ones_like(index))
        rowptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
        return order, rowptr

    order_t, rowptr_t = csr(target, rows_target)
    order_s, rowptr_s = csr(source, rows_source)
    return EdgeSchedule(order_t, rowptr_t, order_s, rowptr_s)


def channelwise(
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    coefficients: torch.Tensor,
    terms: torch.Tensor,
    rows: int,
    channels: int,
    output_segments: int,
) -> torch.Tensor:
    """``out[target, k, c] += C · a[source, i, c] · b[e, j] · w[e, p, c]`` summed over the terms.

    Args:
        a: Node operand ``(rows_a, NA · channels)``.
        b: Angular operand ``(E, NB)``.
        w: Per-edge weights ``(E, NP · channels)``.
        source: ``(E,)`` row of ``a`` per edge.
        target: ``(E,)`` output row per edge.
        coefficients: ``(T,)``.
        terms: ``(T, 4)`` columns ``i, j, p, k``.
        rows: Output row count.
        channels: Channel multiplicity ``U``.
        output_segments: ``NK``.

    Returns:
        ``(rows, NK · channels)``.
    """
    edges = b.shape[0]
    node = a.view(a.shape[0], a.shape[1] // channels, channels)[source]
    weight = w.view(edges, w.shape[1] // channels, channels)
    message = monomials(
        [(node, terms[:, 0]), (b, terms[:, 1]), (weight, terms[:, 2])],
        coefficients,
        terms[:, 3],
        edges,
        output_segments,
    )
    out = a.new_zeros(rows, output_segments, channels).index_add(0, target, message)
    return out.view(rows, output_segments * channels)


def symmetric_contraction(
    x: torch.Tensor,
    w: torch.Tensor,
    indices: torch.Tensor,
    coefficients: torch.Tensor,
    terms: torch.Tensor,
    channels: int,
    output_segments: int,
) -> torch.Tensor:
    """``y[n, o, c] += C · W[index[n], w, c] · x[n, i, c] · x[n, j, c]? · x[n, k, c]?``.

    Args:
        x: Node features ``(N, S · channels)``.
        w: Projected weights ``(num_elements, B · channels)``.
        indices: ``(N,)`` element of each node.
        coefficients: ``(T,)``.
        terms: ``(T, 5)`` columns ``w, i, j, k, o``; ``j`` / ``k`` are ``-1``
            below degree 3.
        channels: ``U``.
        output_segments: ``O``.

    Returns:
        ``(N, O · channels)``.
    """
    nodes = x.shape[0]
    segments = x.shape[1] // channels
    weights = w.view(w.shape[0], w.shape[1] // channels, channels)[indices.long()]
    # A term below degree 3 reads the appended segment of ones in its empty
    # slots, so every term is one three-factor product — one static-shape pass
    # over the whole table, no split by degree (a host sync under compile).
    features = torch.cat([x.view(nodes, segments, channels), x.new_ones(nodes, 1, channels)], dim=1)
    slots = torch.where(terms[:, 1:4] >= 0, terms[:, 1:4], segments)
    factors = [(weights, terms[:, 0])] + [(features, slots[:, slot]) for slot in range(3)]
    out = monomials(factors, coefficients, terms[:, 4], nodes, output_segments)
    return out.view(nodes, output_segments * channels)


def spherical_harmonics(
    v: torch.Tensor, coefficients: torch.Tensor, terms: torch.Tensor, components: int
) -> torch.Tensor:
    """``Y[n, o] += C · v[n, i]? · v[n, j]? · v[n, k]?`` over the monomial table.

    Args:
        v: ``(N, 3)`` Cartesian components.
        coefficients: ``(T,)``.
        terms: ``(T, 4)`` columns ``i, j, k, o``; a missing slot is ``-1``.
        components: Output width.

    Returns:
        ``(N, components)``.
    """
    rows = v.shape[0]
    # Empty slots read an appended component of ones: every term, the constant
    # included, is one three-factor product over the whole table.
    padded = torch.cat([v, v.new_ones(rows, 1)], dim=1)
    slots = torch.where(terms[:, :3] >= 0, terms[:, :3], 3)
    factors = [(padded, slots[:, slot]) for slot in range(3)]
    out = monomials(factors, coefficients, terms[:, 3], rows, components)
    return out.view(rows, components)


def contracter(
    x1: torch.Tensor,
    x2: torch.Tensor,
    w: torch.Tensor,
    index: torch.Tensor,
    coefficients: torch.Tensor,
    terms: torch.Tensor,
    channels: int,
    output_segments: int,
) -> torch.Tensor:
    """``y[e, o, c] += C · W[w, c] · x1[e, i, c] · x2[index[e], j, c]``.

    Args:
        x1: Pair features ``(E, S1 · channels)``.
        x2: Atom features ``(N, S2 · channels)``.
        w: Shared weights ``(B, channels)`` or that vector flattened.
        index: ``(E,)`` atom of each edge.
        coefficients: ``(T,)``.
        terms: ``(T, 4)`` columns ``w, i, j, o``.
        channels: ``U``.
        output_segments: ``O``.

    Returns:
        ``(E, O · channels)``.
    """
    edges = x1.shape[0]
    pair = x1.view(edges, x1.shape[1] // channels, channels)
    atom = x2.view(x2.shape[0], x2.shape[1] // channels, channels)[index.long()]
    weight = w.reshape(1, w.numel() // channels, channels)
    out = monomials(
        [(weight, terms[:, 0]), (pair, terms[:, 1]), (atom, terms[:, 2])],
        coefficients,
        terms[:, 3],
        edges,
        output_segments,
    )
    return out.view(edges, output_segments * channels)


def weighted_channels(
    basis: torch.Tensor,
    weight: torch.Tensor,
    coefficients: torch.Tensor,
    terms: torch.Tensor,
    channels: int,
    num_irreps: int,
    index: torch.Tensor | None,
    num_rows: int | None,
) -> torch.Tensor:
    """``y[e, d, c] += C · basis[e, d] · weight[e, r, c]``, summed into ``index`` when given.

    Args:
        basis: ``(E, D)``.
        weight: ``(E, R, channels)`` irrep-major, or that matrix flattened.
        coefficients: ``(T,)``.
        terms: ``(T, 2)`` columns ``d, r``.
        channels: ``U``.
        num_irreps: ``R``.
        index: ``(E,)`` output row of each edge, or ``None`` for one row per edge.
        num_rows: Output rows when ``index`` is given.

    Returns:
        ``(E, D · channels)`` or ``(num_rows, D · channels)``.
    """
    edges, components = basis.shape
    rows = weight.reshape(edges, num_irreps, channels)
    out = monomials(
        [(basis, terms[:, 0]), (rows, terms[:, 1])], coefficients, terms[:, 0], edges, components
    )
    if index is None:
        return out.view(edges, components * channels)
    summed = basis.new_zeros(int(num_rows), components, channels).index_add(0, index.long(), out)
    return summed.view(int(num_rows), components * channels)
