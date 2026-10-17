"""Tests for molix.compile.Compiler."""

import inspect

import mollog
import pytest
import torch
import torch.nn as nn

from molix.compile import Compiler, mark_dynamic_batch, unwrap
from molix.compile import logger as compile_logger


class TestCompiler:
    """Tests for Compiler configuration + application."""

    @pytest.mark.slow
    def test_compiled_module_produces_same_output(self):
        module = nn.Linear(10, 5)
        x = torch.randn(3, 10)
        with torch.no_grad():
            expected = module(x)
        compiled = Compiler(backend="eager")(module)
        with torch.no_grad():
            actual = compiled(x)
        assert torch.allclose(expected, actual)

    def test_cuda_graph_preset_values(self):
        """The named preset is exactly the benchmarked winning force-training config."""
        assert Compiler.CUDA_GRAPH_PRESET == {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "mode": "reduce-overhead",
        }

    def test_cuda_graphs_applies_preset(self):
        c = Compiler(cuda_graphs=True)
        assert (c.backend, c.fullgraph, c.dynamic, c.mode) == (
            "inductor",
            True,
            False,
            "reduce-overhead",
        )

    def test_cuda_graphs_overrides_explicit_args(self):
        c = Compiler(cuda_graphs=True, dynamic=True, mode="max-autotune")
        assert c.dynamic is False
        assert c.mode == "reduce-overhead"

    def test_cuda_graphs_property_reports_the_graph_capturing_modes(self):
        """``Compiler.cuda_graphs`` is what the MD driver reads to apply the CUDA-graph
        step contract (``cudagraph_mark_step_begin`` + cloned outputs): true for the
        preset and for either inductor mode that wraps the graph in CUDA graphs,
        false for the default and for a non-inductor backend."""
        assert Compiler(cuda_graphs=True).cuda_graphs is True
        assert Compiler(mode="reduce-overhead").cuda_graphs is True
        assert Compiler(mode="max-autotune").cuda_graphs is True
        assert Compiler().cuda_graphs is False
        assert Compiler(fullgraph=True, dynamic=True).cuda_graphs is False
        assert Compiler(backend="aot_eager", mode="reduce-overhead").cuda_graphs is False

    @pytest.mark.slow
    def test_compiles_a_bound_method_and_a_plain_function(self):
        """The compile seam is a flat callable — a bound ``energy``, not a module."""

        class _Affine:
            def __init__(self, w: torch.Tensor) -> None:
                self.w = w

            def apply(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.w + 1

        def scale(x: torch.Tensor) -> torch.Tensor:
            return x * 3.0 - 0.5

        obj = _Affine(torch.arange(9, dtype=torch.float64).reshape(3, 3))
        x = torch.arange(12, dtype=torch.float64).reshape(4, 3)
        assert torch.equal(Compiler(backend="aot_eager")(obj.apply)(x), obj.apply(x))
        assert torch.equal(Compiler(backend="aot_eager")(scale)(x), scale(x))

    def test_log_line_names_the_target_qualname(self):
        """A bound method has no ``__class__.__name__`` worth logging — use the qualname.

        The capture is a ``mollog`` handler on the module's own logger, not
        pytest's ``caplog``: ``molix.compile.logger`` comes from
        ``molix.logger.getLogger`` → ``mollog.get_logger``, and a
        ``mollog.Logger`` dispatches to its own handler tree, never through
        stdlib ``logging``. ``caplog`` installs a stdlib root handler and so
        records nothing here.
        """

        class _Capture(mollog.Handler):
            def __init__(self) -> None:
                super().__init__()
                self.messages: list[str] = []

            def emit(self, record: mollog.LogRecord) -> None:
                self.messages.append(record.message)

        class _Affine:
            def __init__(self, w: torch.Tensor) -> None:
                self.w = w

            def apply(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.w + 1

        obj = _Affine(torch.arange(9, dtype=torch.float64).reshape(3, 3))
        capture = _Capture()
        capture.set_level(mollog.INFO)
        compile_logger.add_handler(capture)
        try:
            Compiler(backend="aot_eager")(obj.apply)
        finally:
            compile_logger.remove_handler(capture)
        assert any(obj.apply.__qualname__ in message for message in capture.messages)

    @pytest.mark.slow
    def test_call_traces_autograd_grad_in_graph(self):
        """A leaf + ``torch.autograd.grad`` inside the target is one graph after ``__call__``.

        dynamo's default (``trace_autograd_ops=False``) graph-breaks at the
        ``autograd.grad`` call, so ``fullgraph=True`` would raise. ``__call__``
        turns the flag on process-wide before the first trace; the flag is
        forced off here first so the test does not depend on test order.
        """
        x = torch.arange(12, dtype=torch.float64).reshape(4, 3)

        def energy_forces(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            leaf = pos.detach().requires_grad_(True)
            with torch.enable_grad():
                energy = (leaf * leaf).sum()
                (grad,) = torch.autograd.grad(energy, leaf)
            return energy.detach(), -grad

        torch._dynamo.config.trace_autograd_ops = False
        torch._dynamo.reset()
        compiled = Compiler(fullgraph=True, backend="aot_eager")(energy_forces)
        with torch.no_grad():
            energy, forces = compiled(x)
            ref_energy, ref_forces = energy_forces(x)
        assert torch.equal(energy, ref_energy)
        assert torch.equal(forces, ref_forces)
        assert not forces.requires_grad


class TestInductorFallbacks:
    """Tests for the ATen ``index_add`` fallback registration (``register_inductor_fallbacks``)."""

    @pytest.mark.slow
    def test_index_add_is_an_extern_kernel_under_inductor(self):
        """A pointwise producer fused into ``index_add_`` must not leak ``f(0)`` per padded lane.

        Inductor lowers ``index_add_`` → ``index_put_(accumulate=True)`` →
        ``ir.Scatter(atomic_add)`` and inlines the pointwise producer into the
        scatter loader. The load is masked (padded lanes read 0) but the fused
        epilogue ``f(0)`` is still atomically added for those lanes, so the
        scatter target gains ``(roundup(N, 32) - N) * f(0)``. Here
        ``f(v) = 0.7 * v + 1.3`` and ``N = 4995`` → ``(5024 - 4995) * 1.3 =
        37.7`` of pure fiction. Removing the ``aten.index_add`` decomposition
        and registering the ATen fallback makes inductor emit an extern
        ``index_add`` kernel, which has no fused epilogue and therefore no leak.

        Evidence: GH200 job 1974355 measured PEO-box ``dE = 0.0`` with the
        fallback registered, versus ``dE = 14.24 eV`` without it (29 padded
        lanes x the Polar readout shift).

        Both halves are asserted here because they are one fact: the numbers
        agree *because* the op is extern, and an extern op that silently
        regained a decomposition would put the leak back.
        """
        import torch._inductor.decomposition
        import torch._inductor.lowering

        def f(v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, dtype=v.dtype).index_add_(0, b, v * 0.7 + 1.3)

        v = torch.randn(4995, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        v = v * 500.0
        b = torch.zeros(4995, dtype=torch.long)

        compiled = Compiler(backend="inductor")(f)
        assert torch.allclose(compiled(v, b), f(v, b), atol=1e-9, rtol=0.0)

        assert torch.ops.aten.index_add.default in torch._inductor.lowering.fallbacks
        assert torch.ops.aten.index_add.default not in torch._inductor.decomposition.decompositions


class TestExplainSummary:
    """Tests for Compiler.explain() and the molnex-owned ExplainSummary."""

    @pytest.mark.slow
    def test_explain_traces_autograd_grad_in_graph(self):
        """``explain`` measures with ``trace_autograd_ops`` on, like ``__call__`` compiles.

        Forced off around the call: ``explain`` must patch it on itself, or the
        ``autograd.grad`` below reports as a break that a ``Compiler`` would
        never see.
        """
        x = torch.arange(12, dtype=torch.float64).reshape(4, 3)

        def energy_forces(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            leaf = pos.detach().requires_grad_(True)
            with torch.enable_grad():
                energy = (leaf * leaf).sum()
                (grad,) = torch.autograd.grad(energy, leaf)
            return energy.detach(), -grad

        with torch._dynamo.config.patch(trace_autograd_ops=False), torch.no_grad():
            summary = Compiler.explain(energy_forces, x)
        assert summary.graph_count == 1
        assert summary.graph_break_count == 0
        assert summary.break_reasons == ()

    @pytest.mark.slow
    def test_explain_reports_graph_count_breaks_reasons_and_sites(self):
        """One trace answers "how many graphs", "how many breaks" and "broken where".

        ``graph_break_count`` alone cannot tell 0 breaks from "never traced"
        (upstream defines it as ``graph_count - 1``), hence the paired
        ``graph_count == 1`` on the clean function.

        ``print`` is the break marker because on this torch (2.12) dynamo still
        refuses to trace the builtin. Were that to change, the marker line would
        become ``torch._dynamo.graph_break()``; nothing dynamo-version specific
        is hard-coded here — the expected line number is read back out of the
        function's own source.
        """
        x = torch.arange(12, dtype=torch.float64).reshape(4, 3)

        def clean(t: torch.Tensor) -> torch.Tensor:
            return (t * 2.0).sum()

        def breaks(t: torch.Tensor) -> torch.Tensor:
            y = t + 1.0
            print("graph break marker")
            return y * 2.0

        clean_summary = Compiler.explain(clean, x)
        assert isinstance(clean_summary, tuple)
        assert clean_summary.graph_count == 1
        assert clean_summary.graph_break_count == 0
        assert clean_summary.break_reasons == ()
        assert clean_summary.break_sites == ()

        lines, first_lineno = inspect.getsourcelines(breaks)
        offsets = [i for i, line in enumerate(lines) if line.strip().startswith("print(")]
        assert len(offsets) == 1
        expected_site = f"{__file__}:{first_lineno + offsets[0]}"

        summary = Compiler.explain(breaks, x)
        assert summary.graph_break_count >= 1
        assert len(summary.break_reasons) == summary.graph_break_count
        assert len(summary.break_sites) == summary.graph_break_count
        assert all(isinstance(reason, str) for reason in summary.break_reasons)
        assert all(isinstance(site, str) for site in summary.break_sites)
        assert expected_site in summary.break_sites


class TestMarkDynamicBatch:
    """TensorDict count-axis marks for the ragged-train compile path."""

    def test_marks_atom_dim0_and_bond_index_dim1(self):
        from tensordict import TensorDict

        z = torch.arange(4)
        pos = torch.zeros(4, 3)
        pair_index = torch.zeros(5, 2, dtype=torch.long)
        bond_index = torch.zeros(2, 3, dtype=torch.long)
        batch = TensorDict(
            {
                "atoms": TensorDict({"Z": z, "pos": pos}, batch_size=[4]),
                "pairs": TensorDict({"pair_index": pair_index}, batch_size=[5]),
                "bonds": TensorDict({"bond_index": bond_index}, batch_size=[]),
            },
            batch_size=[],
        )
        mark_dynamic_batch(batch)
        assert 0 in z._dynamo_dynamic_indices
        assert 0 in pos._dynamo_dynamic_indices
        assert 1 not in getattr(pos, "_dynamo_dynamic_indices", set())
        assert 0 in pair_index._dynamo_dynamic_indices
        assert 1 in bond_index._dynamo_dynamic_indices
        assert 0 not in getattr(bond_index, "_dynamo_dynamic_indices", set())

    def test_ignores_plain_tensors_and_dicts(self):
        mark_dynamic_batch(torch.arange(3))
        mark_dynamic_batch({"Z": torch.arange(3)})

    @pytest.mark.slow
    def test_dynamic_compile_does_not_recompile_on_ragged_atom_count(self):
        """``dynamic=True`` still guards TensorDict leaf sizes without marks.

        Packed Class-I batches change N every step; GH200 profile 2362410
        hit ``recompile_limit`` on ``atoms.Z`` 172 vs 190. The Compiler
        wrapper marks count axes before the compiled region.
        """
        from tensordict import TensorDict

        def loss(batch: object) -> torch.Tensor:
            td = batch
            z = td["atoms", "Z"].float()
            eidx = td["pairs", "pair_index"].float()
            return z.sum() + eidx.sum()

        torch._dynamo.reset()
        compiled = Compiler(backend="eager", fullgraph=True, dynamic=True)(loss)

        def make_batch(n_atoms: int, n_pairs: int) -> TensorDict:
            return TensorDict(
                {
                    "atoms": TensorDict(
                        {"Z": torch.arange(n_atoms)},
                        batch_size=[n_atoms],
                    ),
                    "pairs": TensorDict(
                        {
                            "pair_index": torch.stack(
                                (
                                    torch.zeros(n_pairs, dtype=torch.long),
                                    torch.ones(n_pairs, dtype=torch.long),
                                ),
                                dim=1,
                            )
                        },
                        batch_size=[n_pairs],
                    ),
                },
                batch_size=[],
            )

        ns = (3, 5, 8, 4, 11, 7, 9, 6, 12, 2)
        for n in ns:
            compiled(make_batch(n, n + 1))


class TestUnwrap:
    """``unwrap`` reads through a ``torch.compile`` wrapper; a plain module is itself."""

    def test_compiled_module_unwraps_to_its_module_and_a_plain_one_is_itself(self):
        from torch._dynamo.eval_frame import OptimizedModule

        inner = nn.Linear(2, 2)
        compiled = torch.compile(inner)

        assert isinstance(compiled, OptimizedModule)
        assert unwrap(compiled) is inner
        assert unwrap(inner) is inner
