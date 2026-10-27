"""The term-table evaluator and the CSR schedule of the reference path."""

import torch

from molrep.equivariance import reference


class TestMonomials:
    def test_broadcasts_a_shared_row_and_scalar_factors(self):
        """A ``(1, S, U)`` factor broadcasts over the rows, a ``(R, S)`` one over the channels."""
        torch.manual_seed(0)
        shared = torch.randn(1, 3, 4, dtype=torch.float64)
        per_row = torch.randn(5, 2, dtype=torch.float64)
        channels = torch.randn(5, 2, 4, dtype=torch.float64)
        terms = torch.tensor([[0, 1, 0, 1], [2, 0, 1, 1], [1, 1, 1, 0]])
        coefficients = torch.tensor([0.5, -2.0, 3.0], dtype=torch.float64)
        out = reference.monomials(
            [(shared, terms[:, 0]), (per_row, terms[:, 1]), (channels, terms[:, 2])],
            coefficients,
            terms[:, 3],
            5,
            2,
        )
        expected = torch.zeros(5, 2, 4, dtype=torch.float64)
        for (a, b, c, o), coefficient in zip(terms.tolist(), coefficients.tolist(), strict=True):
            expected[:, o] += coefficient * shared[0, a] * per_row[:, b, None] * channels[:, c]
        torch.testing.assert_close(out, expected)

    def test_chunks_over_the_terms(self, monkeypatch):
        """A chunk budget of one term per pass gives the same sum."""
        torch.manual_seed(1)
        x = torch.randn(3, 4, 2, dtype=torch.float64)
        terms = torch.tensor([[0, 0], [1, 0], [2, 1], [3, 1], [1, 1]])
        coefficients = torch.rand(5, dtype=torch.float64)
        whole = reference.monomials([(x, terms[:, 0])], coefficients, terms[:, 1], 3, 2)
        monkeypatch.setattr(reference, "BUDGET", 1)
        chunked = reference.monomials([(x, terms[:, 0])], coefficients, terms[:, 1], 3, 2)
        torch.testing.assert_close(whole, chunked)


class TestSchedule:
    def test_is_the_stable_csr(self):
        torch.manual_seed(3)
        source = torch.randint(11, (500,))
        target = torch.randint(7, (500,))
        target[target == 4] = 5  # one row without edges
        schedule = reference.schedule(source, target, 11, 7)
        for order, rowptr, index, rows in [
            (schedule.order_t, schedule.rowptr_t, target, 7),
            (schedule.order_s, schedule.rowptr_s, source, 11),
        ]:
            assert order.dtype == rowptr.dtype == torch.int32
            torch.testing.assert_close(order, torch.argsort(index, stable=True).to(torch.int32))
            counts = torch.bincount(index, minlength=rows)
            expected = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
            torch.testing.assert_close(rowptr, expected)
