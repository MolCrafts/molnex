"""Unit tests for the pure-math helpers in molix.autograd.variables.

Mirrors ``src/molix/autograd/variables.py``: the leaf a derivative is taken
against and the strain-gradient convention every virial goes through. The
per-key handlers are private and are covered through ``Derivative``
(``test_derivative.py``).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molix.autograd import leaf, symmetrize_strain_grad
from molix.autograd.variables import CELL_KEY, get_column
from molix.schema import ENERGY_KEY


class TestLeaf:
    """``leaf`` — a detached copy that requires grad, on the source's dtype and device."""

    def test_is_a_fresh_requires_grad_tensor_with_the_same_values(self) -> None:
        source = torch.arange(6, dtype=torch.float64).reshape(2, 3)
        variable = leaf(source)
        assert variable is not source
        assert variable.requires_grad is True
        assert variable.is_leaf
        assert variable.dtype == source.dtype
        torch.testing.assert_close(variable, source)

    def test_drops_the_history_of_an_attached_source(self) -> None:
        """The derivative is of this evaluation alone, not of how ``pos`` was made."""
        upstream = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
        variable = leaf(upstream * 2.0)
        assert variable.grad_fn is None
        (through,) = torch.autograd.grad(variable.sum(), upstream, allow_unused=True)
        assert through is None


def test_get_column_does_not_depend_on_graphs_key_order() -> None:
    """Collate inserts ``graphs`` columns in data-dependent order.

    Dynamo used to guard ``graphs.keys()[1] == 'energy'`` (GH200 2366877).
    """

    def frame(names: list[str]) -> TensorDict:
        graphs = TensorDict(batch_size=[1])
        for name in names:
            graphs[name] = (
                torch.zeros(1) if name != "num_atoms" else torch.ones(1, dtype=torch.long)
            )
        return TensorDict(graphs=graphs, batch_size=[])

    first = frame(["num_atoms", "energy", "total_charge"])
    second = frame(["total_charge", "num_atoms", "energy"])
    assert list(first["graphs"].keys()) != list(second["graphs"].keys())
    assert get_column(first, ENERGY_KEY) is not None
    assert get_column(second, ENERGY_KEY) is not None
    assert get_column(first, CELL_KEY) is None
    assert get_column(second, CELL_KEY) is None


def test_symmetrize_strain_grad_batches_over_leading_dims() -> None:
    """The convention transposes the last two axes only and keeps the trace."""
    grad = torch.randn(4, 3, 3, dtype=torch.float64)
    out = symmetrize_strain_grad(grad)
    torch.testing.assert_close(out, out.transpose(-1, -2))
    torch.testing.assert_close(out.sum(), grad.sum())


def test_symmetrize_strain_grad_is_the_cauchy_value_not_the_engineering_double() -> None:
    """A harmonic bond has ``W_αβ = k d_α d_β`` — no factor 2 on a shear."""
    k = 3.0
    d0 = torch.tensor([0.7, -1.3, 0.4], dtype=torch.float64)
    strain = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    d = d0 @ (torch.eye(3, dtype=torch.float64) + strain).T
    (grad,) = torch.autograd.grad(0.5 * k * d.dot(d), strain)
    torch.testing.assert_close(symmetrize_strain_grad(grad), k * torch.outer(d0, d0))
