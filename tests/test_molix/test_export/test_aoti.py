"""Tests for :mod:`molix.export.aoti`."""

from __future__ import annotations

import pytest
import torch

from molix.export import aoti


@torch.library.custom_op("molnex_export_test::twice", mutates_args=())
def _twice(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x


@_twice.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class _Plain(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x) + x


class _Custom(torch.nn.Module):
    def forward(self, x):
        return torch.ops.molnex_export_test.twice(x) + 1.0


class _Forces(torch.nn.Module):
    """A parameterised energy whose forward returns ``-dE/dx`` via ``autograd.grad``."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            energy = self.linear(x).pow(2).sum()
            (grad,) = torch.autograd.grad(energy, x)
        return energy.detach(), -grad


class _SplitWeight(torch.nn.Module):
    """Forces through a weight with a dimension equal to the example's row count."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 3))

    def forward(self, x):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            head, tail = self.weight.split([3, 5])  # a guard on the weight's 8
            energy = (x @ head.t()).pow(2).sum() + (x @ tail.t()).sum()
            (grad,) = torch.autograd.grad(energy, x)
        return energy.detach(), -grad


class TestCustomOpNamespaces:
    def test_an_aten_only_program_has_none(self):
        exported = torch.export.export(_Plain(), (torch.randn(4),))
        assert aoti.custom_op_namespaces(exported) == frozenset()

    def test_a_custom_operator_namespace_is_reported(self):
        exported = torch.export.export(_Custom(), (torch.randn(4),))
        assert aoti.custom_op_namespaces(exported) == frozenset({"molnex_export_test"})


@pytest.mark.slow
class TestPackage:
    def test_packages_loads_and_reports_namespaces(self, tmp_path):
        x = torch.randn(4)
        written = aoti.package(_Plain(), (x,), tmp_path / "plain.pt2")
        assert written.path.is_file() and written.op_namespaces == frozenset()
        torch.testing.assert_close(aoti.load(written.path)(x), _Plain()(x))

    def test_a_differentiating_forward_keeps_its_gradient_and_weight_names(self, tmp_path):
        torch.manual_seed(0)
        model = _Forces()
        rows = torch.export.Dim("rows", min=2, max=1024)
        written = aoti.package(
            model,
            (torch.randn(8, 3),),
            tmp_path / "forces.pt2",
            dynamic_shapes=({0: rows},),
            inductor_configs={"aot_inductor.use_runtime_constant_folding": True},
        )
        runner = aoti.load(written.path)
        x = torch.randn(37, 3)
        for got, want in zip(runner(x), model(x)):
            torch.testing.assert_close(got, want)
        assert {"linear.weight", "linear.bias"} <= set(runner.get_constant_fqns())

    def test_a_weight_size_equal_to_the_example_does_not_pin_the_rows(self, tmp_path):
        torch.manual_seed(0)
        model = _SplitWeight()
        rows = torch.export.Dim("rows", min=2, max=1024)
        written = aoti.package(
            model, (torch.randn(8, 3),), tmp_path / "split.pt2", dynamic_shapes=({0: rows},)
        )
        x = torch.randn(37, 3)
        for got, want in zip(aoti.load(written.path)(x), model(x)):
            torch.testing.assert_close(got, want)
