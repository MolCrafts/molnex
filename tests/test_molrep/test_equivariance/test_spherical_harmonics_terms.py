"""Spherical-harmonics term table against the polynomial and a finite difference.

The reference is the descriptor polynomial, one vector at a time. The adjoint
is the product rule on the monomials. It does not share code with the kernel.
``l = 1`` is ``sqrt(3)`` times the Cartesian components in this convention.
"""

from __future__ import annotations

import numpy as np
import pytest

import molrep.eq as meq
from molrep.equivariance.spherical_harmonics_terms import SphericalHarmonicsTerms

_EPS = 1e-6
_ATOL = 1e-8
_RTOL = 1e-7


def _polynomial(ls):
    return meq.descriptors.spherical_harmonics(meq.SO3(1), list(ls)).polynomial


def _evaluate(polynomial, vectors):
    output = np.zeros((vectors.shape[0], polynomial.outputs[0].size), dtype=np.float64)
    for row in range(vectors.shape[0]):
        output[row] = polynomial(vectors[row])[0]
    return output


def _contract(tables, vectors):
    y = np.zeros((vectors.shape[0], tables.components), dtype=np.float64)
    for row in range(vectors.shape[0]):
        v = vectors[row]
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            i, j, k, oseg = (int(item) for item in term)
            prod = coefficient
            if i >= 0:
                prod = prod * v[i]
            if j >= 0:
                prod = prod * v[j]
            if k >= 0:
                prod = prod * v[k]
            y[row, oseg] += prod
    return y


def _first_adjoint(tables, vectors, cotangent):
    gx = np.zeros_like(vectors)
    for row in range(vectors.shape[0]):
        v = vectors[row]
        g = cotangent[row]
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            i, j, k, oseg = (int(item) for item in term)
            scale = coefficient * g[oseg]
            if i < 0:
                continue
            if j < 0:
                gx[row, i] += scale
            elif k < 0:
                gx[row, i] += scale * v[j]
                gx[row, j] += scale * v[i]
            else:
                gx[row, i] += scale * v[j] * v[k]
                gx[row, j] += scale * v[i] * v[k]
                gx[row, k] += scale * v[i] * v[j]
    return gx


def _second_adjoint(tables, vectors, cotangent, dgx):
    dx = np.zeros_like(vectors)
    dg = np.zeros_like(cotangent)
    for row in range(vectors.shape[0]):
        v = vectors[row]
        g = cotangent[row]
        d = dgx[row]
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            i, j, k, oseg = (int(item) for item in term)
            if i < 0:
                continue
            if j < 0:
                dg[row, oseg] += coefficient * d[i]
            elif k < 0:
                base = coefficient * g[oseg]
                dx[row, i] += d[j] * base
                dx[row, j] += d[i] * base
                dg[row, oseg] += coefficient * (d[i] * v[j] + d[j] * v[i])
            else:
                base = coefficient * g[oseg]
                dx[row, i] += base * (d[j] * v[k] + d[k] * v[j])
                dx[row, j] += base * (d[i] * v[k] + d[k] * v[i])
                dx[row, k] += base * (d[i] * v[j] + d[j] * v[i])
                dg[row, oseg] += coefficient * (
                    d[i] * v[j] * v[k] + d[j] * v[i] * v[k] + d[k] * v[i] * v[j]
                )
    return dx, dg


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


class TestSphericalHarmonicsTerms:
    def test_degree_one_is_sqrt_three_times_the_vector(self):
        tables = SphericalHarmonicsTerms([0, 1])
        assert tables.components == 4
        assert tables.terms.shape == (4, 4)
        found = {
            tuple(int(v) for v in term): float(coefficient)
            for term, coefficient in zip(tables.terms, tables.coefficients, strict=True)
        }
        assert found[(-1, -1, -1, 0)] == pytest.approx(1.0)
        root = 3**0.5
        assert found[(0, -1, -1, 1)] == pytest.approx(root)
        assert found[(1, -1, -1, 2)] == pytest.approx(root)
        assert found[(2, -1, -1, 3)] == pytest.approx(root)

    def test_replay_matches_the_polynomial(self):
        rng = np.random.default_rng(0)
        for ls in ([0], [0, 1, 2], [1, 3], [0, 1, 2, 3]):
            tables = SphericalHarmonicsTerms(ls)
            polynomial = _polynomial(ls)
            vectors = rng.standard_normal((7, 3))
            actual = _contract(tables, vectors)
            expected = _evaluate(polynomial, vectors)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        full = SphericalHarmonicsTerms([0, 1, 2, 3])
        assert full.components == 16
        assert full.terms.shape[0] == 1 + 3 + 11 + 41
        assert np.any((full.terms[:, 0] == full.terms[:, 1]) & (full.terms[:, 1] >= 0))

    def test_adjoints_match_a_finite_difference(self):
        rng = np.random.default_rng(1)
        tables = SphericalHarmonicsTerms([0, 1, 2, 3])
        polynomial = _polynomial([0, 1, 2, 3])
        vectors = rng.standard_normal((3, 3))
        cotangent = rng.standard_normal((3, tables.components))

        def loss():
            return float(np.sum(_evaluate(polynomial, vectors) * cotangent))

        gx = _first_adjoint(tables, vectors, cotangent)
        np.testing.assert_allclose(gx, _fd_scalar(loss, vectors, _EPS), atol=_ATOL, rtol=_RTOL)

        dgx = rng.standard_normal(vectors.shape)

        def directional():
            return float(np.sum(_first_adjoint(tables, vectors, cotangent) * dgx))

        dx, dg = _second_adjoint(tables, vectors, cotangent, dgx)
        np.testing.assert_allclose(
            dx, _fd_scalar(directional, vectors, _EPS), atol=_ATOL, rtol=_RTOL
        )
        np.testing.assert_allclose(
            dg, _fd_scalar(directional, cotangent, _EPS), atol=_ATOL, rtol=_RTOL
        )

    @pytest.mark.parametrize("ls", [[], [4], [2, 1], [1, 1], [-1]])
    def test_rejects_degrees_outside_the_family(self, ls):
        with pytest.raises(ValueError, match="0..3"):
            SphericalHarmonicsTerms(ls)
