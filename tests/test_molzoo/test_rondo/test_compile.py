"""The compile seam: forward, energy and partial Hessian each trace as one graph.

Rondo does no host work per batch: the pair relations arrive from the loader,
the encoder and the tables are tensor algebra on static shapes, and the
force field's potentials are plain torch. Each primitive is different traced
code — the force is one backward, the partial Hessian a double backward.
"""

from __future__ import annotations

import pytest
import torch

from molix.compile import Compiler
from molzoo.rondo import Rondo
from tests.test_molzoo.test_rondo.conftest import DTYPE, TINY, with_hessian_pairs


class TestRondoCompile:
    @pytest.mark.slow
    def test_forward_is_one_graph(self, forcefield, butane_batch):
        """Positions are marked first, as the training step's eager prologue does."""
        torch.manual_seed(0)
        model = Rondo(forcefield, dtype=DTYPE, **TINY)
        batch = butane_batch.clone()
        model.derivative.mark_leaves(batch)
        explanation = Compiler.explain(model, batch)
        # Both fields: upstream defines graph_break_count = graph_count - 1, so
        # a target that never traced reports -1, which reads as better than clean.
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites

    @pytest.mark.slow
    def test_energy_is_one_graph(self, forcefield, butane_batch):
        torch.manual_seed(0)
        model = Rondo(forcefield, dtype=DTYPE, **TINY)
        explanation = Compiler.explain(model.energy, butane_batch.clone())
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites

    @pytest.mark.slow
    def test_the_energy_of_named_styles_is_one_graph(self, forcefield, butane_batch):
        """The style loop is a tuple of names fixed at the call: straight-line code."""
        torch.manual_seed(0)
        model = Rondo(forcefield, dtype=DTYPE, **TINY)

        def bonded(frame):
            return model.energy(frame, styles=("bond:harmonic", "angle:harmonic"))

        explanation = Compiler.explain(bonded, butane_batch.clone())
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites

    @pytest.mark.slow
    def test_partial_hessian_is_one_graph(self, forcefield, butane_batch):
        torch.manual_seed(0)
        model = Rondo(forcefield, dtype=DTYPE, **TINY)
        batch = butane_batch.clone()
        with_hessian_pairs(batch)
        model.derivative.mark_leaves(batch)

        def blocks(frame):
            # What the joint step traces: the tables once, the Hessian on them.
            return model.partial_hessian(frame, model.predict(frame))

        explanation = Compiler.explain(blocks, batch)
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites

    @pytest.mark.slow
    def test_the_eval_forward_is_one_graph(self, forcefield, butane_batch):
        """Eval runs with grad off at entry: ``autograd.grad`` frees the graph, so
        no column may keep a ``grad_fn`` from it (GH200 3304497)."""
        torch.manual_seed(0)
        model = Rondo(forcefield, dtype=DTYPE, **TINY)
        batch = butane_batch.clone()
        with torch.no_grad():
            model.derivative.mark_leaves(batch)
            explanation = Compiler.explain(model, batch)
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites

    @pytest.mark.slow
    def test_inductor_force_gradient_and_update(self, forcefield, butane_batch):
        """Compiled energy and force match eager, including one optimizer step."""
        import copy

        torch.manual_seed(3)
        eager = Rondo(forcefield, dtype=DTYPE, **TINY)
        compiled_model = copy.deepcopy(eager)

        def objective(model, frame):
            out = model.derivative(frame)
            energy = out["graphs", "energy"]
            force = out["atoms", "forces_pred"]
            loss = energy.square().mean() * 1e-4 + force.square().mean() * 1e-3
            return loss, energy, force

        def target(frame):
            return objective(compiled_model, frame)

        probe = butane_batch.clone()
        compiled_model.derivative.mark_leaves(probe)
        explanation = Compiler.explain(target, probe)
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0, explanation.break_sites
        optimized = Compiler(backend="inductor", fullgraph=True, dynamic=False)(target)
        a, b = butane_batch.clone(), butane_batch.clone()
        eager.derivative.mark_leaves(a)
        compiled_model.derivative.mark_leaves(b)
        expected = objective(eager, a)
        actual = optimized(b)
        for x, y in zip(expected, actual, strict=True):
            torch.testing.assert_close(x, y, rtol=1e-8, atol=1e-8)
        expected[0].backward()
        actual[0].backward()
        for x, y in zip(eager.parameters(), compiled_model.parameters(), strict=True):
            assert (x.grad is None) == (y.grad is None)
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, rtol=1e-7, atol=1e-8)
        for model in (eager, compiled_model):
            torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        for x, y in zip(eager.parameters(), compiled_model.parameters(), strict=True):
            torch.testing.assert_close(x, y, rtol=1e-7, atol=1e-8)
