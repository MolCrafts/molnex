"""The compile seam: what the traced half of a Rondo step is.

A Class-I step splits in two. The host half fills the field, assembles the
container and binds this batch's topology — molpy work, none of it physics,
and worth 22 graph breaks if it is dragged inside a compiled region. The
tensor half is the derivative over that container's energy, and it is the part
that must trace clean (spec section 3.7).
"""

from __future__ import annotations

import pytest
import torch

from molix.compile import Compiler
from molzoo.rondo import Rondo, RondoPotential, RondoSpec


class TestRondoCompile:
    @pytest.mark.slow
    @pytest.mark.parametrize("trainable", [None, ("bonds",)])
    def test_the_traced_half_is_one_graph(self, butane_batch, trainable):
        """``energy`` — encoder, heads and the Class-I kernels — traces clean.

        One eager call first: that is what assembles the container and binds
        the topology, which the traced half requires and must not contain.

        Both ``trainable`` settings, because they are different traced code:
        naming a subset puts a per-namespace rebuild of the table dict inside
        the compiled region, and that is the form the Hessian stage runs in.

        On an empty ``break_sites`` with a non-zero count, re-run the target
        under ``TORCH_LOGS=graph_breaks`` — upstream left the reasons empty for
        real breaks on torch 2.15.0.dev while reporting them for a control
        (2026-09-18), so the sites here are a convenience, not the diagnosis.
        """
        torch.manual_seed(0)
        model = RondoPotential(
            Rondo(RondoSpec(hidden=8, n_blocks=1, n_torsion_terms=2, eem_max_atoms=32)),
            trainable=trainable,
        )
        with torch.no_grad():
            model(butane_batch.clone())

        explanation = Compiler.explain(model.energy, butane_batch.clone())
        # Both fields, because upstream defines
        # graph_break_count = graph_count - 1: a target that never traced at
        # all reports -1, which would read as better than clean.
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites
