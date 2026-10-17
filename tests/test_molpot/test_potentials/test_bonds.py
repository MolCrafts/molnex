"""BondHarmonic canonical bond_index [2, N] contract + edge≠bond enforcement.

See spec graph-connectivity-alignment-03-bonds: covalent connectivity is a
canonical COO ``bond_index`` ``[2, num_bonds]`` (distinct from the geometric
``pair_index`` ``[E, 2]``). A geometric cutoff edge must never be silently
consumed as a bond list — the potential raises instead.
"""

import math

import pytest
import torch

from molpot.potentials.bonds import BondHarmonic


def _potential():
    # one bond type: k=2.0, r0=1.0
    return BondHarmonic(k=torch.tensor([2.0]), r0=torch.tensor([1.0]))


def test_canonical_bond_index_energy_matches_analytic():
    pot = _potential()
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    bond_index = torch.tensor([[0], [1]], dtype=torch.long)  # [2, 1] COO
    bond_types = torch.tensor([0], dtype=torch.long)
    e = pot(pos=pos, bond_index=bond_index, bond_types=bond_types, num_graphs=1)
    # 0.5 * 2.0 * (1.5 - 1.0)^2 = 0.25
    assert math.isclose(float(e), 0.25, rel_tol=1e-6)


def test_empty_bonds_returns_zero():
    pot = _potential()
    pos = torch.tensor([[0.0, 0.0, 0.0]])
    bond_index = torch.zeros(2, 0, dtype=torch.long)  # [2, 0]
    bond_types = torch.zeros(0, dtype=torch.long)
    e = pot(pos=pos, bond_index=bond_index, bond_types=bond_types, num_graphs=1)
    assert float(e) == 0.0


def test_rejects_edge_index_as_bond_index():
    # A geometric pair_index [E, 2] (here E=3) must be rejected, not silently
    # mis-indexed into [2, N] logic (the scientist-confirmed latent bug).
    pot = _potential()
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pair_index = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)  # [E, 2]
    bond_types = torch.tensor([0, 0, 0], dtype=torch.long)
    with pytest.raises(ValueError, match="bond_index"):
        pot(pos=pos, bond_index=pair_index, bond_types=bond_types, num_graphs=1)


def test_omitted_bond_types_prices_the_table_row_per_bond():
    """No ``bond_types`` means "the table already has a row per bond".

    It used to raise. A learned field predicts one row per interaction, so the
    type index would be ``arange(n)`` and the gather a copy of the table whose
    backward is an ``index_put_(accumulate=True)`` with atomics — measured at
    77 % of device time on a production-shaped step. Omitting the index is how
    a caller says the gather is unnecessary, and the energy must be identical
    to passing the identity.
    """
    pot = _potential()
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    bond_index = torch.tensor([[0], [1]], dtype=torch.long)
    identity = pot(
        pos=pos, bond_index=bond_index, bond_types=torch.zeros(1, dtype=torch.long), num_graphs=1
    )
    omitted = pot(pos=pos, bond_index=bond_index, num_graphs=1)
    assert math.isclose(float(omitted), float(identity), rel_tol=1e-12)
    assert math.isclose(float(omitted), 0.25, rel_tol=1e-6)


class TestPerCallTables:
    """A bonded term prices the caller's table when given one, else its buffer.

    The two cases are the two kinds of force field: an exported, fixed field
    looks its ``k`` up once and keeps it (the buffer), while a **learned** field
    predicts its tables per batch. The learned tables have to be computed inside
    the compiled region — a table produced outside carries a ``grad_fn`` that
    ``torch.autograd.grad`` will not trace through — so the term has to accept
    them per call rather than own them.
    """

    @staticmethod
    def _two_bonds() -> dict[str, torch.Tensor]:
        return {
            "pos": torch.tensor(
                [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0], [3.4, 0.0, 0.0]], dtype=torch.float64
            ),
            "bond_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "bond_types": torch.tensor([0, 0], dtype=torch.long),
            "num_graphs": 1,
        }

    def test_an_override_replaces_the_registered_table_for_that_call_only(self) -> None:
        term = BondHarmonic(
            k=torch.tensor([300.0], dtype=torch.float64),
            r0=torch.tensor([1.5], dtype=torch.float64),
        )
        batch = self._two_bonds()

        from_buffer = term(**batch)
        from_call = term(**batch, k=torch.tensor([600.0], dtype=torch.float64))

        assert torch.allclose(from_call, 2.0 * from_buffer)
        assert torch.allclose(term(**batch), from_buffer)

    def test_the_override_carries_gradient_to_whatever_predicted_it(self) -> None:
        predicted = torch.tensor([300.0], dtype=torch.float64, requires_grad=True)
        term = BondHarmonic(k=predicted.detach(), r0=torch.tensor([1.5], dtype=torch.float64))

        term(**self._two_bonds(), k=predicted).backward()

        assert predicted.grad is not None
        assert torch.isfinite(predicted.grad).all()

    def test_a_table_that_is_neither_passed_nor_registered_names_itself(self) -> None:
        term = BondHarmonic(k=torch.ones(1), r0=torch.ones(1))
        with pytest.raises(AttributeError, match="theta0"):
            term.table("theta0", {})
