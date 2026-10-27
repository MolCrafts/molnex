"""Contracter term table against the flattened polynomial and a finite difference.

The reference builds the same ``u,iu,ju,ku+ijk`` descriptor Allegro does and
evaluates it one edge at a time. The adjoint is the product rule on
``C * W * x1 * x2``. It does not share code with the kernel. Duplicate source
indices accumulate into the atom gradient with ``np.add.at``.

``0e x 0e -> 0e`` has coefficient 1. Each ``1o`` diagonal into the scalar has
coefficient ``1 / sqrt(3)``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import molrep.eq as meq
from molrep.equivariance.contracter_terms import ContracterTerms

_EPS = 1e-6
_ATOL = 1e-8
_RTOL = 1e-7


def _polynomial(irreps_in1, irreps_in2, irreps_out, channels):
    descriptor = meq.SegmentedTensorProduct.from_subscripts("u,iu,ju,ku+ijk")
    for _, ir in irreps_in1:
        descriptor.add_segment(1, (ir.dim, channels))
    for _, ir in irreps_in2:
        descriptor.add_segment(2, (ir.dim, channels))
    for _, ir in irreps_out:
        descriptor.add_segment(3, (ir.dim, channels))
    num_paths = 0
    for (i3, (_, ir3)), (i1, (_, ir1)), (i2, (_, ir2)) in itertools.product(
        enumerate(irreps_out), enumerate(irreps_in1), enumerate(irreps_in2)
    ):
        if ir3 in ir1 * ir2:
            for cg in meq.clebsch_gordan(ir1, ir2, ir3):
                descriptor.add_path(None, i1, i2, i3, c=cg)
                num_paths += 1
    polynomial = meq.SegmentedPolynomial.eval_last_operand(descriptor).flatten_coefficient_modes()
    return polynomial, num_paths


def _reshape(x1, x2, weight, tables):
    edges = x1.shape[0]
    atoms = x2.shape[0]
    return (
        x1.reshape(edges, tables.input1_segments, tables.channels),
        x2.reshape(atoms, tables.input2_segments, tables.channels),
        weight.reshape(tables.weight_segments, tables.channels),
    )


def _evaluate(polynomial, x1, x2, weight, index):
    output = np.zeros((x1.shape[0], polynomial.outputs[0].size), dtype=np.float64)
    flat_weight = np.ascontiguousarray(weight).reshape(-1)
    for edge in range(x1.shape[0]):
        output[edge] = polynomial(flat_weight, x1[edge], x2[int(index[edge])])[0]
    return output


def _contract(tables, x1, x2, weight, index):
    left, right, shared = _reshape(x1, x2, weight, tables)
    edges, _, channels = left.shape
    y = np.zeros((edges, tables.output_segments, channels), dtype=np.float64)
    for edge in range(edges):
        atom = int(index[edge])
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            wseg, i, j, oseg = (int(item) for item in term)
            y[edge, oseg] += coefficient * shared[wseg] * left[edge, i] * right[atom, j]
    return y.reshape(edges, -1)


def _first_adjoint(tables, x1, x2, weight, index, cotangent):
    left, right, shared = _reshape(x1, x2, weight, tables)
    g = cotangent.reshape(left.shape[0], tables.output_segments, tables.channels)
    gx1 = np.zeros_like(left)
    gx2 = np.zeros_like(right)
    gw = np.zeros_like(shared)
    for edge in range(left.shape[0]):
        atom = int(index[edge])
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            wseg, i, j, oseg = (int(item) for item in term)
            scale = coefficient * g[edge, oseg]
            a = left[edge, i]
            b = right[atom, j]
            wv = shared[wseg]
            gx1[edge, i] += scale * wv * b
            np.add.at(gx2, (atom, j), scale * wv * a)
            gw[wseg] += scale * a * b
    return gx1.reshape(x1.shape), gx2.reshape(x2.shape), gw.reshape(weight.shape)


def _second_adjoint(tables, x1, x2, weight, index, cotangent, dgx1, dgx2, dgw):
    left, right, shared = _reshape(x1, x2, weight, tables)
    g = cotangent.reshape(left.shape[0], tables.output_segments, tables.channels)
    d_left = dgx1.reshape(left.shape)
    d_right = dgx2.reshape(right.shape)
    d_weight = dgw.reshape(shared.shape)
    dx1 = np.zeros_like(left)
    dx2 = np.zeros_like(right)
    dw = np.zeros_like(shared)
    dg = np.zeros_like(g)
    for edge in range(left.shape[0]):
        atom = int(index[edge])
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            wseg, i, j, oseg = (int(item) for item in term)
            a = left[edge, i]
            b = right[atom, j]
            wv = shared[wseg]
            da = d_left[edge, i]
            db = d_right[atom, j]
            dwv = d_weight[wseg]
            base = coefficient * g[edge, oseg]
            dx1[edge, i] += db * wv * base + dwv * b * base
            np.add.at(dx2, (atom, j), da * wv * base + dwv * a * base)
            dw[wseg] += da * base * b + db * base * a
            dg[edge, oseg] += coefficient * (da * wv * b + db * wv * a + dwv * a * b)
    return (
        dx1.reshape(x1.shape),
        dx2.reshape(x2.shape),
        dw.reshape(weight.shape),
        dg.reshape(cotangent.shape),
    )


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


def _draw(tables, edges, atoms, rng):
    x1 = rng.standard_normal((edges, tables.input1_segments * tables.channels))
    x2 = rng.standard_normal((atoms, tables.input2_segments * tables.channels))
    weight = rng.standard_normal((tables.weight_segments, tables.channels))
    # Edge 0 and edge 1 share atom 0. The last atom is unused when atoms > 1.
    index = np.zeros(edges, dtype=np.int64)
    if edges > 2 and atoms > 1:
        index[2:] = rng.integers(0, atoms - 1, size=edges - 2)
    cotangent = rng.standard_normal((edges, tables.output_segments * tables.channels))
    return x1, x2, weight, index, cotangent


class TestContracterTerms:
    def test_scalar_vector_paths_match_the_probed_coefficients(self):
        irreps = meq.Irreps("O3", "0e+1o")
        tables = ContracterTerms(irreps, irreps, irreps, 4)
        assert tables.channels == 4
        assert tables.weight_segments == 4
        assert tables.input1_segments == 4
        assert tables.input2_segments == 4
        assert tables.output_segments == 4
        assert tables.terms.shape == (10, 4)
        assert tables.terms.dtype == np.int16
        assert tables.coefficients.dtype == np.float64
        found = {
            tuple(int(v) for v in term): float(coefficient)
            for term, coefficient in zip(tables.terms, tables.coefficients, strict=True)
        }
        root = 3**-0.5
        assert found[(0, 0, 0, 0)] == pytest.approx(1.0)
        assert found[(1, 1, 1, 0)] == pytest.approx(root)
        assert found[(1, 2, 2, 0)] == pytest.approx(root)
        assert found[(1, 3, 3, 0)] == pytest.approx(root)
        # One Clebsch-Gordan matrix flattens into several component paths.
        assert sum(key[0] == 1 for key in found) == 3

    def test_weight_segments_stay_the_unflattened_path_count(self):
        irreps = meq.Irreps("O3", "0e+1o+2e")
        tables = ContracterTerms(irreps, irreps, irreps, 1)
        polynomial, num_paths = _polynomial(irreps, irreps, irreps, 1)
        assert tables.weight_segments == num_paths == 11
        assert tables.terms.shape[0] == 83
        assert polynomial.inputs[0].segment_shape == (1,)
        assert tables.input1_segments == 9

    def test_replay_matches_the_polynomial(self):
        rng = np.random.default_rng(0)
        cases = [
            ("O3", "0e+1o", "0e+1o", "0e+1o", 4, 5, 3),
            ("O3", "0e+1o+2e", "0e+1o+2e", "0e+1o+2e", 2, 3, 2),
            ("O3", "0e+1o", "1o+2e", "1o+2e+3o", 2, 4, 3),
            ("SO3", "0+1", "0+1", "0+1", 3, 4, 2),
        ]
        for group, left, right, out, channels, edges, atoms in cases:
            irreps_in1 = meq.Irreps(group, left)
            irreps_in2 = meq.Irreps(group, right)
            irreps_out = meq.Irreps(group, out)
            tables = ContracterTerms(irreps_in1, irreps_in2, irreps_out, channels)
            polynomial, num_paths = _polynomial(irreps_in1, irreps_in2, irreps_out, channels)
            assert tables.weight_segments == num_paths
            x1, x2, weight, index, _cotangent = _draw(tables, edges, atoms, rng)
            actual = _contract(tables, x1, x2, weight, index)
            expected = _evaluate(polynomial, x1, x2, weight, index)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            zeros = _contract(
                tables, np.zeros_like(x1), np.zeros_like(x2), np.zeros_like(weight), index
            )
            assert np.count_nonzero(zeros) == 0

    def test_adjoints_match_a_finite_difference(self):
        rng = np.random.default_rng(1)
        irreps = meq.Irreps("O3", "0e+1o")
        tables = ContracterTerms(irreps, irreps, irreps, 2)
        polynomial, _num_paths = _polynomial(irreps, irreps, irreps, 2)
        x1, x2, weight, index, cotangent = _draw(tables, 4, 3, rng)
        assert index.tolist()[:2] == [0, 0]
        assert 2 not in index

        def loss():
            return float(np.sum(_evaluate(polynomial, x1, x2, weight, index) * cotangent))

        gx1, gx2, gw = _first_adjoint(tables, x1, x2, weight, index, cotangent)
        np.testing.assert_allclose(gx1, _fd_scalar(loss, x1, _EPS), atol=_ATOL, rtol=_RTOL)
        np.testing.assert_allclose(gx2, _fd_scalar(loss, x2, _EPS), atol=_ATOL, rtol=_RTOL)
        np.testing.assert_allclose(gw, _fd_scalar(loss, weight, _EPS), atol=_ATOL, rtol=_RTOL)
        assert np.count_nonzero(gx2.reshape(3, tables.input2_segments, tables.channels)[2]) == 0

        dgx1 = rng.standard_normal(x1.shape)
        dgx2 = rng.standard_normal(x2.shape)
        dgw = rng.standard_normal(weight.shape)

        def directional():
            g1, g2, gw_value = _first_adjoint(tables, x1, x2, weight, index, cotangent)
            return float(np.sum(g1 * dgx1) + np.sum(g2 * dgx2) + np.sum(gw_value * dgw))

        dx1, dx2, dw, dg = _second_adjoint(
            tables, x1, x2, weight, index, cotangent, dgx1, dgx2, dgw
        )
        np.testing.assert_allclose(dx1, _fd_scalar(directional, x1, _EPS), atol=_ATOL, rtol=_RTOL)
        np.testing.assert_allclose(dx2, _fd_scalar(directional, x2, _EPS), atol=_ATOL, rtol=_RTOL)
        np.testing.assert_allclose(
            dw, _fd_scalar(directional, weight, _EPS), atol=_ATOL, rtol=_RTOL
        )
        np.testing.assert_allclose(
            dg, _fd_scalar(directional, cotangent, _EPS), atol=_ATOL, rtol=_RTOL
        )

        quiet_x1 = np.zeros_like(x1)
        quiet_x2 = np.zeros_like(x2)
        _dx1, _dx2, dw_only, _dg = _second_adjoint(
            tables, x1, x2, weight, index, cotangent, quiet_x1, quiet_x2, dgw
        )
        assert np.max(np.abs(dw_only)) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("in1", "in2", "out", "channels", "match"),
        [
            ("0e", "1o", "2e", 4, "no tensor-product path"),
            ("2x0e", "0e", "0e", 4, "one copy"),
            ("0e", "0e", "0e", 0, "channels >= 1"),
            ("+".join(f"{ell}e" for ell in range(9)), "0e", "0e", 1, "64"),
        ],
    )
    def test_rejects_inputs_outside_the_family(self, in1, in2, out, channels, match):
        with pytest.raises(ValueError, match=match):
            ContracterTerms(
                meq.Irreps("O3", in1),
                meq.Irreps("O3", in2),
                meq.Irreps("O3", out),
                channels,
            )
