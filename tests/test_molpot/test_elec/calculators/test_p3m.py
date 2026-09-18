"""Tests for the P3M calculator."""

import pytest
import torch

from molpot.potentials.elec import CoulombPotential, P3MCalculator
from molpot.potentials.elec.lib import get_ns_mesh

DTYPE = torch.float64
#: Gaussian split width (Å) and target mesh spacing (Å) of the fixture — the
#: values the bound shape below is sized for.
SMEARING = 1.0
MESH_SPACING = 1.0
#: Cube edge (Å) of the fixture box: four ions, neutral overall.
EDGE = 4.0
POSITIONS = (
    (0.5, 0.5, 0.5),
    (2.5, 0.5, 0.5),
    (0.5, 2.5, 0.5),
    (0.5, 0.5, 2.5),
)
CHARGES = (1.0, -1.0, 1.0, -1.0)


class TestP3MCalculator:
    @pytest.mark.slow
    def test_the_bound_kspace_derivatives_trace_to_one_graph_in_inference(self) -> None:
        """The iron law on the P3M reciprocal route: one graph, zero breaks, in inference.

        Traced the way MD traces it — ``no_grad`` outside, the leaves and the
        ``autograd.grad`` inside (:mod:`molix.autograd`), differentiating the
        k-space energy against both positions and cell. That is the one
        setting in which per-call tensors kept on a module break the graph:
        derived from a leaf they require grad, and dynamo reports each store
        as a leaked intermediate. The mesh stencil and the k-space filter are
        therefore values passed along, never state — and ``bind`` freezes the
        shape so nothing is sized off the cell inside the trace.

        PME's copy of this rule is asserted on the assembled force field
        (``tests/test_molpot/test_forcefield``); P3M has no assembly-level
        fixture, so its guard sits here, on the calculator that calls the
        stencil and the filter.
        """
        from molix.autograd.variables import leaf
        from molix.compile import Compiler

        calculator = P3MCalculator(CoulombPotential(smearing=SMEARING), mesh_spacing=MESH_SPACING)
        cell = torch.eye(3, dtype=DTYPE) * EDGE
        calculator.bind(tuple(int(n) for n in get_ns_mesh(cell, MESH_SPACING)))
        positions = torch.tensor(POSITIONS, dtype=DTYPE)
        charges = torch.tensor(CHARGES, dtype=DTYPE).unsqueeze(-1)
        periodic = torch.ones(3, dtype=torch.bool)

        def kspace_derivatives(
            positions: torch.Tensor, cell: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            pos, lattice = leaf(positions), leaf(cell)
            with torch.enable_grad():
                phi = calculator._compute_kspace(charges, lattice, pos, periodic)
                return torch.autograd.grad((charges * phi).sum(), (pos, lattice))

        with torch.no_grad():
            summary = Compiler.explain(kspace_derivatives, positions, cell)
        assert (summary.graph_count, summary.graph_break_count) == (1, 0), (
            f"the bound P3M k-space route must trace as one graph; "
            f"break sites: {summary.break_sites}"
        )
