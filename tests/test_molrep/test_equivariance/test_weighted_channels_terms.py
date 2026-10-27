"""Weighted-channels term table against the polynomial and a finite difference.

The reference builds the same ``",u,u"`` descriptor Allegro does and evaluates
it one edge at a time. An index sums those rows. The adjoint is the product
rule on ``C * basis * weight``. It does not share code with the kernel.
``u = 1`` keeps weight segments shaped ``(1,)``.
"""

from __future__ import annotations

import numpy as np
import pytest

import molrep.eq as meq
from molrep.equivariance.weighted_channels_terms import WeightedChannelsTerms

_EPS = 1e-6
_ATOL = 1e-8
_RTOL = 1e-7


def _polynomial(irrep_dims, channels):
    descriptor = meq.SegmentedTensorProduct.from_subscripts(",u,u")
    for _ in irrep_dims:
        descriptor.add_segment(1, (channels,))
    for irrep, dim in enumerate(irrep_dims):
        for _ in range(dim):
            component = descriptor.add_segment(0, ())
            descriptor.add_segment(2, (channels,))
            descriptor.add_path(component, irrep, component, c=1.0)
    return meq.SegmentedPolynomial.eval_last_operand(descriptor)


def _locate(index, edge):
    return edge if index is None else int(index[edge])


def _contract(tables, basis, weight, index, num_rows):
    edges, components = basis.shape
    channels = tables.channels
    rows = edges if index is None else num_rows
    output = np.zeros((rows, components, channels))
    for edge in range(edges):
        row = _locate(index, edge)
        if row < 0 or row >= rows:
            continue
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            component, irrep = (int(item) for item in term)
            output[row, component] += coefficient * basis[edge, component] * weight[edge, irrep]
    return output.reshape(rows, -1)


def _evaluate(polynomial, basis, weight, index, num_rows):
    per_edge = [
        polynomial(basis[edge], np.ascontiguousarray(weight[edge]).reshape(-1))[0]
        for edge in range(basis.shape[0])
    ]
    stacked = np.stack(per_edge)
    if index is None:
        return stacked
    output = np.zeros((num_rows, stacked.shape[1]))
    for edge in range(stacked.shape[0]):
        row = int(index[edge])
        if 0 <= row < num_rows:
            output[row] += stacked[edge]
    return output


def _first(tables, basis, weight, cotangent, index):
    channels = tables.channels
    g = cotangent.reshape(cotangent.shape[0], tables.components, channels)
    grad_basis = np.zeros_like(basis)
    grad_weight = np.zeros_like(weight)
    rows = g.shape[0]
    for edge in range(basis.shape[0]):
        row = _locate(index, edge)
        if row < 0 or row >= rows:
            continue
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            component, irrep = (int(item) for item in term)
            grad_basis[edge, component] += coefficient * np.dot(
                weight[edge, irrep], g[row, component]
            )
            grad_weight[edge, irrep] += coefficient * basis[edge, component] * g[row, component]
    return grad_basis, grad_weight


def _second(tables, basis, weight, cotangent, index, dga, dgw):
    channels = tables.channels
    g = cotangent.reshape(cotangent.shape[0], tables.components, channels)
    grad_basis = np.zeros_like(basis)
    grad_weight = np.zeros_like(weight)
    grad_g = np.zeros_like(g)
    rows = g.shape[0]
    for edge in range(basis.shape[0]):
        row = _locate(index, edge)
        if row < 0 or row >= rows:
            continue
        for term, coefficient in zip(tables.terms, tables.coefficients, strict=True):
            component, irrep = (int(item) for item in term)
            grad_basis[edge, component] += coefficient * np.dot(g[row, component], dgw[edge, irrep])
            grad_weight[edge, irrep] += coefficient * dga[edge, component] * g[row, component]
            grad_g[row, component] += coefficient * (
                weight[edge, irrep] * dga[edge, component]
                + basis[edge, component] * dgw[edge, irrep]
            )
    return grad_basis, grad_weight, grad_g.reshape(cotangent.shape)


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


class TestWeightedChannelsTerms:
    def test_each_component_uses_its_irrep_with_coefficient_one(self):
        tables = WeightedChannelsTerms([1, 3], 2)
        assert tables.channels == 2
        assert tables.num_irreps == 2
        assert tables.components == 4
        assert tables.terms.dtype == np.int16
        assert tables.coefficients.dtype == np.float64
        found = {
            tuple(int(v) for v in term): float(coefficient)
            for term, coefficient in zip(tables.terms, tables.coefficients, strict=True)
        }
        assert found == {
            (0, 0): pytest.approx(1.0),
            (1, 1): pytest.approx(1.0),
            (2, 1): pytest.approx(1.0),
            (3, 1): pytest.approx(1.0),
        }

    def test_unit_channels_keep_a_length_one_segment(self):
        tables = WeightedChannelsTerms([1, 3, 5], 1)
        polynomial = _polynomial([1, 3, 5], 1)
        assert tables.components == 9
        assert tables.terms.shape == (9, 2)
        assert polynomial.inputs[1].segments[0] == (1,)
        assert polynomial.inputs[0].segments[0] == ()

    def test_replay_matches_the_polynomial(self):
        rng = np.random.default_rng(0)
        tables = WeightedChannelsTerms([1, 3, 5], 4)
        polynomial = _polynomial([1, 3, 5], 4)
        basis = rng.standard_normal((5, tables.components))
        weight = rng.standard_normal((5, tables.num_irreps, tables.channels))
        direct = _contract(tables, basis, weight, None, None)
        expected = _evaluate(polynomial, basis, weight, None, None)
        np.testing.assert_allclose(direct, expected, atol=1e-12, rtol=1e-12)
        index = np.array([0, 0, 1, 2, -1])
        scattered = _contract(tables, basis, weight, index, 4)
        oracle = _evaluate(polynomial, basis, weight, index, 4)
        np.testing.assert_allclose(scattered, oracle, atol=1e-12, rtol=1e-12)
        assert np.count_nonzero(scattered.reshape(4, tables.components, tables.channels)[3]) == 0

    def test_adjoints_match_a_finite_difference(self):
        rng = np.random.default_rng(1)
        tables = WeightedChannelsTerms([1, 3], 2)
        polynomial = _polynomial([1, 3], 2)
        basis = rng.standard_normal((4, tables.components))
        weight = rng.standard_normal((4, tables.num_irreps, tables.channels))
        index = np.array([0, 0, 1, -1])
        num_rows = 4
        cotangent = rng.standard_normal((num_rows, tables.components * tables.channels))
        assert 3 not in index

        def loss():
            return float(np.sum(_evaluate(polynomial, basis, weight, index, num_rows) * cotangent))

        grad_basis, grad_weight = _first(tables, basis, weight, cotangent, index)
        np.testing.assert_allclose(
            grad_basis, _fd_scalar(loss, basis, _EPS), atol=_ATOL, rtol=_RTOL
        )
        np.testing.assert_allclose(
            grad_weight, _fd_scalar(loss, weight, _EPS), atol=_ATOL, rtol=_RTOL
        )
        assert np.max(np.abs(grad_basis[3])) == pytest.approx(0.0)
        assert np.max(np.abs(grad_weight[3])) == pytest.approx(0.0)
        output = _contract(tables, basis, weight, index, num_rows)
        scattered = output.reshape(num_rows, tables.components, tables.channels)
        assert np.count_nonzero(scattered[2]) == 0
        assert np.count_nonzero(scattered[3]) == 0

        dga = rng.standard_normal(basis.shape)
        dgw = rng.standard_normal(weight.shape)

        def directional():
            ga, gw = _first(tables, basis, weight, cotangent, index)
            return float(np.sum(ga * dga) + np.sum(gw * dgw))

        da, dw, dg = _second(tables, basis, weight, cotangent, index, dga, dgw)
        np.testing.assert_allclose(da, _fd_scalar(directional, basis, _EPS), atol=_ATOL, rtol=_RTOL)
        np.testing.assert_allclose(
            dw, _fd_scalar(directional, weight, _EPS), atol=_ATOL, rtol=_RTOL
        )
        np.testing.assert_allclose(
            dg, _fd_scalar(directional, cotangent, _EPS), atol=_ATOL, rtol=_RTOL
        )
        quiet = np.zeros_like(basis)
        _da, dw_only, _dg = _second(tables, basis, weight, cotangent, index, quiet, dgw)
        assert np.max(np.abs(dw_only)) == pytest.approx(0.0)
        _da, dw_quiet, _dg = _second(
            tables, basis, weight, cotangent, index, dga, np.zeros_like(weight)
        )
        assert np.max(np.abs(_da)) == pytest.approx(0.0)
        assert dw_quiet.shape == weight.shape

    @pytest.mark.parametrize(
        ("dims", "channels", "match"),
        [
            ([], 1, "at least one"),
            ([1, 0], 1, "positive"),
            ([1], 0, "channels >= 1"),
            ([65], 1, "components"),
            ([1] * 65, 1, "irreps"),
        ],
    )
    def test_rejects_inputs_outside_the_family(self, dims, channels, match):
        with pytest.raises(ValueError, match=match):
            WeightedChannelsTerms(dims, channels)
