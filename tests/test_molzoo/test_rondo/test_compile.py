"""The compile seam: forward, energy and partial Hessian each trace as one graph.

Rondo does no host work per batch: the pair relations arrive from the loader,
the hybridization and the tables are tensor algebra on static shapes, and the
force field's potentials are plain torch. Each primitive is different traced
code — the force is one backward, the partial Hessian a double backward.
"""

from __future__ import annotations

import pytest
import torch

from molix.compile import Compiler
from molzoo.rondo import Rondo
from tests.test_molzoo.test_rondo.conftest import SPEC, with_hessian_pairs


class TestRondoCompile:
    @pytest.mark.slow
    def test_forward_is_one_graph(self, forcefield, butane_batch):
        """Positions are marked first, as the training step's eager prologue does."""
        torch.manual_seed(0)
        model = Rondo(forcefield, SPEC)
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
        model = Rondo(forcefield, SPEC)
        explanation = Compiler.explain(model.energy, butane_batch.clone())
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites

    @pytest.mark.slow
    def test_the_energy_of_named_styles_is_one_graph(self, forcefield, butane_batch):
        """The style loop is a tuple of names fixed at the call: straight-line code."""
        torch.manual_seed(0)
        model = Rondo(forcefield, SPEC)

        def bonded(frame):
            return model.energy(frame, styles=("bond:harmonic", "angle:harmonic"))

        explanation = Compiler.explain(bonded, butane_batch.clone())
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites

    @pytest.mark.slow
    def test_partial_hessian_is_one_graph(self, forcefield, butane_batch):
        torch.manual_seed(0)
        model = Rondo(forcefield, SPEC)
        batch = butane_batch.clone()
        with_hessian_pairs(batch)
        model.derivative.mark_leaves(batch)
        explanation = Compiler.explain(model.partial_hessian, batch)
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites

    @pytest.mark.slow
    def test_the_eval_forward_is_one_graph(self, forcefield, butane_batch):
        """Eval runs with grad off at entry: ``autograd.grad`` frees the graph, so
        no column may keep a ``grad_fn`` from it (GH200 3304497)."""
        torch.manual_seed(0)
        model = Rondo(forcefield, SPEC)
        batch = butane_batch.clone()
        with torch.no_grad():
            model.derivative.mark_leaves(batch)
            explanation = Compiler.explain(model, batch)
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites
