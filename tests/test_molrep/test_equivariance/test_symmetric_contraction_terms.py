"""Term table of the native symmetric contraction against the polynomial.

The reference is :func:`molrep.eq.segmented_polynomials.SegmentedPolynomial.__call__`,
one node at a time, plus a central difference of that value. The adjoint
below is the product rule written out for body orders 1..3. It does not share
code with the CUDA kernels. Duplicate element ids are accumulated with
``np.add.at``; a slice ``+=`` would drop them.

Tolerances are float64. ``eps = 1e-6`` leaves an ``O(eps**2)`` truncation on
a degree-3 monomial, about ``1e-12`` before roundoff, so ``1e-8`` is loose
for a match and tight for a wrong factor of two.
"""

from __future__ import annotations

import numpy as np
import pytest

import molrep.eq as meq
from molrep.eq.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction,
)
from molrep.equivariance.symmetric_contraction_terms import (
    MAX_DEGREE,
    SymmetricContractionTerms,
)

_EPS = 1e-6
_ATOL = 1e-8
_RTOL = 1e-7


def _polynomial(irreps_in, irreps_out, degree):
    polynomial, _projection = symmetric_contraction(irreps_in, irreps_out, range(1, degree + 1))
    return polynomial.polynomial


def _evaluate(polynomial, features, weight, index):
    """Polynomial value per node. ``weight[index[n]]`` is that node's weight."""
    output = np.zeros((features.shape[0], polynomial.outputs[0].size), dtype=np.float64)
    for node in range(features.shape[0]):
        output[node] = polynomial(weight[index[node]], features[node])[0]
    return output


def _reshape(features, weight, tables):
    nodes = features.shape[0]
    elements = weight.shape[0]
    x = features.reshape(nodes, tables.input_segments, tables.channels)
    w = weight.reshape(elements, tables.weight_segments, tables.channels)
    return x, w


def _contract(tables, features, weight, index):
    """Term-table replay. Independent of the polynomial evaluator."""
    x, w = _reshape(features, weight, tables)
    nodes, _segments, channels = x.shape
    y = np.zeros((nodes, tables.output_segments, channels), dtype=np.float64)
    for node in range(nodes):
        element = int(index[node])
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            wseg, i, j, k, oseg = (int(v) for v in term)
            prod = w[element, wseg] * x[node, i]
            if j >= 0:
                prod = prod * x[node, j]
            if k >= 0:
                prod = prod * x[node, k]
            y[node, oseg] += coefficient * prod
    return y.reshape(nodes, -1)


def _first_adjoint(tables, features, weight, index, cotangent):
    """``(gx, gw)`` of ``sum(y * cotangent)`` by the product rule."""
    x, w = _reshape(features, weight, tables)
    g = cotangent.reshape(x.shape[0], tables.output_segments, tables.channels)
    gx = np.zeros_like(x)
    gw = np.zeros_like(w)
    for node in range(x.shape[0]):
        element = int(index[node])
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            wseg, i, j, k, oseg = (int(v) for v in term)
            scale = coefficient * g[node, oseg]
            wv = w[element, wseg]
            x0 = x[node, i]
            if j < 0:
                gx[node, i] += scale * wv
                np.add.at(gw, (element, wseg), scale * x0)
            elif k < 0:
                x1 = x[node, j]
                gx[node, i] += scale * wv * x1
                gx[node, j] += scale * wv * x0
                np.add.at(gw, (element, wseg), scale * x0 * x1)
            else:
                x1 = x[node, j]
                x2 = x[node, k]
                gx[node, i] += scale * wv * x1 * x2
                gx[node, j] += scale * wv * x0 * x2
                gx[node, k] += scale * wv * x0 * x1
                np.add.at(gw, (element, wseg), scale * x0 * x1 * x2)
    return gx.reshape(features.shape), gw.reshape(weight.shape)


def _second_adjoint(tables, features, weight, index, cotangent, dgx, dgw):
    """``(dx, dw, dg)`` of ``<gx, dgx> + <gw, dgw>``. ``dw`` does not read ``dgw``."""
    x, w = _reshape(features, weight, tables)
    g = cotangent.reshape(x.shape[0], tables.output_segments, tables.channels)
    dx_in = dgx.reshape(x.shape)
    dw_in = dgw.reshape(w.shape)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    dg = np.zeros_like(g)
    for node in range(x.shape[0]):
        element = int(index[node])
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            wseg, i, j, k, oseg = (int(v) for v in term)
            base = coefficient * g[node, oseg]
            wv = w[element, wseg]
            dwv = dw_in[element, wseg]
            x0 = x[node, i]
            d0 = dx_in[node, i]
            if j < 0:
                dx[node, i] += dwv * base
                np.add.at(dw, (element, wseg), d0 * base)
                dg[node, oseg] += coefficient * (d0 * wv + dwv * x0)
            elif k < 0:
                x1 = x[node, j]
                d1 = dx_in[node, j]
                scaled = base * wv
                dx[node, i] += d1 * scaled + dwv * base * x1
                dx[node, j] += d0 * scaled + dwv * base * x0
                np.add.at(dw, (element, wseg), d0 * base * x1 + d1 * base * x0)
                dg[node, oseg] += coefficient * (wv * (d0 * x1 + d1 * x0) + dwv * x0 * x1)
            else:
                x1 = x[node, j]
                x2 = x[node, k]
                d1 = dx_in[node, j]
                d2 = dx_in[node, k]
                scaled = base * wv
                dx[node, i] += scaled * (d1 * x2 + d2 * x1) + dwv * base * x1 * x2
                dx[node, j] += scaled * (d0 * x2 + d2 * x0) + dwv * base * x0 * x2
                dx[node, k] += scaled * (d0 * x1 + d1 * x0) + dwv * base * x0 * x1
                np.add.at(
                    dw,
                    (element, wseg),
                    base * (d0 * x1 * x2 + d1 * x0 * x2 + d2 * x0 * x1),
                )
                dg[node, oseg] += coefficient * (
                    wv * (d0 * x1 * x2 + d1 * x0 * x2 + d2 * x0 * x1) + dwv * x0 * x1 * x2
                )
    return dx.reshape(features.shape), dw.reshape(weight.shape), dg.reshape(cotangent.shape)


def _fd_scalar(fn, value, eps):
    gradient = np.zeros_like(value)
    for flat in range(value.size):
        saved = value.reshape(-1)[flat]
        value.reshape(-1)[flat] = saved + eps
        plus = fn()
        value.reshape(-1)[flat] = saved - eps
        minus = fn()
        value.reshape(-1)[flat] = saved
        gradient.reshape(-1)[flat] = (plus - minus) / (2 * eps)
    return gradient


def _draw(tables, nodes, elements, rng):
    features = rng.standard_normal((nodes, tables.input_segments * tables.channels))
    weight = rng.standard_normal((elements, tables.weight_segments * tables.channels))
    # Element 0 is shared, the last element is unused when elements > 1.
    index = np.zeros(nodes, dtype=np.int64)
    if nodes > 1 and elements > 1:
        index[1:] = rng.integers(0, elements - 1, size=nodes - 1)
    cotangent = rng.standard_normal((nodes, tables.output_segments * tables.channels))
    return features, weight, index, cotangent


class TestSymmetricContractionTerms:
    def test_degree_two_scalar_vector_matches_the_probed_paths(self):
        tables = SymmetricContractionTerms(
            meq.Irreps("O3", "4x0e+4x1o"), meq.Irreps("O3", "4x0e"), 2
        )
        assert tables.channels == 4
        assert tables.input_segments == 4
        assert tables.output_segments == 1
        assert tables.weight_segments == 3
        assert tables.projection.shape == (3, 3)
        assert tables.terms.shape == (5, 5)
        assert tables.terms.dtype == np.int16
        assert tables.weight_offsets.dtype == np.int32
        expected = {
            (0, 0, -1, -1, 0): 1.0,
            (1, 0, 0, -1, 0): 0.7071067811865475,
            (2, 1, 1, -1, 0): 0.40824829046386296,
            (2, 2, 2, -1, 0): 0.40824829046386296,
        }
        found = {
            tuple(int(v) for v in term): float(coefficient)
            for term, coefficient in zip(tables.terms, tables.coefficients, strict=True)
        }
        for key, coefficient in expected.items():
            assert found[key] == pytest.approx(coefficient)

    def test_grouped_offsets_are_a_stable_partition(self):
        tables = SymmetricContractionTerms(
            meq.Irreps("O3", "2x0e+2x1o+2x2e"), meq.Irreps("O3", "2x0e+2x1o"), 3
        )
        assert tables.weight_segments == 17
        assert tables.projection.shape[1] == 17
        assert tables.input_segments == 9
        assert tables.output_segments == 4
        assert int(tables.weight_offsets[0]) == 0
        assert int(tables.weight_offsets[-1]) == tables.terms.shape[0]
        assert np.all(np.diff(tables.weight_offsets.astype(np.int64)) >= 0)
        order = np.argsort(tables.terms[:, 0], kind="stable")
        assert np.array_equal(tables.grouped_terms, tables.terms[order])
        assert np.array_equal(tables.grouped_coefficients, tables.coefficients[order])
        for segment in range(tables.weight_segments):
            start = int(tables.weight_offsets[segment])
            stop = int(tables.weight_offsets[segment + 1])
            if start == stop:
                continue
            assert np.all(tables.grouped_terms[start:stop, 0] == segment)
        # A repeated input slot is a real path, not a table-building accident.
        repeated = tables.terms[:, 2] == tables.terms[:, 3]
        assert np.any(repeated & (tables.terms[:, 3] >= 0))

    def test_replay_matches_the_polynomial(self):
        rng = np.random.default_rng(0)
        cases = [
            ("O3", "2x0e+2x1o", "2x0e", 2, 7, 3),
            ("O3", "1x0e+1x1o", "1x0e", 3, 5, 1),
            ("O3", "2x0e+2x1o+2x2e", "2x0e+2x1o", 3, 4, 3),
            ("SO3", "2x0+2x1+2x2", "2x0+2x1", 2, 3, 2),
        ]
        for group, irreps_in, irreps_out, degree, nodes, elements in cases:
            left = meq.Irreps(group, irreps_in)
            right = meq.Irreps(group, irreps_out)
            tables = SymmetricContractionTerms(left, right, degree)
            polynomial = _polynomial(left, right, degree)
            features, weight, index, _cotangent = _draw(tables, nodes, elements, rng)
            actual = _contract(tables, features, weight, index)
            expected = _evaluate(polynomial, features, weight, index)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            zeros = _contract(tables, np.zeros_like(features), np.zeros_like(weight), index)
            assert np.count_nonzero(zeros) == 0

    def test_first_adjoint_matches_a_finite_difference_of_the_polynomial(self):
        rng = np.random.default_rng(1)
        cases = [
            ("2x0e+2x1o", "2x0e", 2),
            ("1x0e+1x1o", "1x0e", 3),
        ]
        for irreps_in, irreps_out, degree in cases:
            left = meq.Irreps("O3", irreps_in)
            right = meq.Irreps("O3", irreps_out)
            tables = SymmetricContractionTerms(left, right, degree)
            polynomial = _polynomial(left, right, degree)
            features, weight, index, cotangent = _draw(tables, 4, 3, rng)

            def loss():
                return float(np.sum(_evaluate(polynomial, features, weight, index) * cotangent))

            gx, gw = _first_adjoint(tables, features, weight, index, cotangent)
            np.testing.assert_allclose(gx, _fd_scalar(loss, features, _EPS), atol=_ATOL, rtol=_RTOL)
            np.testing.assert_allclose(gw, _fd_scalar(loss, weight, _EPS), atol=_ATOL, rtol=_RTOL)
            # The unused element receives no contribution.
            unused = gw.reshape(3, tables.weight_segments, tables.channels)[2]
            assert np.count_nonzero(unused) == 0

    def test_second_adjoint_matches_a_finite_difference_of_the_first(self):
        rng = np.random.default_rng(2)
        left = meq.Irreps("O3", "1x0e+1x1o")
        right = meq.Irreps("O3", "1x0e")
        tables = SymmetricContractionTerms(left, right, 3)
        features, weight, index, cotangent = _draw(tables, 3, 3, rng)
        dgx = rng.standard_normal(features.shape)
        dgw = rng.standard_normal(weight.shape)

        def directional():
            gx, gw = _first_adjoint(tables, features, weight, index, cotangent)
            return float(np.sum(gx * dgx) + np.sum(gw * dgw))

        dx, dw, dg = _second_adjoint(tables, features, weight, index, cotangent, dgx, dgw)
        np.testing.assert_allclose(
            dx, _fd_scalar(directional, features, _EPS), atol=_ATOL, rtol=_RTOL
        )
        np.testing.assert_allclose(
            dw, _fd_scalar(directional, weight, _EPS), atol=_ATOL, rtol=_RTOL
        )
        np.testing.assert_allclose(
            dg, _fd_scalar(directional, cotangent, _EPS), atol=_ATOL, rtol=_RTOL
        )
        # gw does not depend on w, so the mixed partial through dgw alone is zero.
        dw_only = _second_adjoint(
            tables, features, weight, index, cotangent, np.zeros_like(dgx), dgw
        )[1]
        assert np.max(np.abs(dw_only)) < 1e-12

    @pytest.mark.parametrize(
        ("irreps_in", "irreps_out", "degree", "match"),
        [
            ("2x0e", "2x0e", 0, "degrees 1..3"),
            ("2x0e", "2x0e", 4, "degrees 1..3"),
            ("2x0e+1x1o", "2x0e", 2, "one multiplicity"),
            ("2x0e", "1x0e", 1, "one multiplicity"),
        ],
    )
    def test_rejects_degrees_and_mixed_multiplicities(self, irreps_in, irreps_out, degree, match):
        with pytest.raises(ValueError, match=match):
            SymmetricContractionTerms(
                meq.Irreps("O3", irreps_in), meq.Irreps("O3", irreps_out), degree
            )

    def test_degree_above_the_cap_does_not_build_clebsch_gordan(self):
        assert MAX_DEGREE == 3
        with pytest.raises(ValueError, match="degrees 1..3"):
            SymmetricContractionTerms(
                meq.Irreps("O3", "1x0e+1x1o+1x2e+1x3o"), meq.Irreps("O3", "1x0e"), 4
            )
