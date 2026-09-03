"""Tests for molix.compile.Compiler."""

import inspect

import mollog
import pytest
import torch
import torch.nn as nn

from molix.compile import Compiler, ExplainSummary
from molix.compile import logger as compile_logger


class TestCompiler:
    """Tests for Compiler configuration + application."""

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

    @pytest.mark.slow
    def test_compiles_a_bound_method_and_a_plain_function(self):
        """The compile seam is a flat callable — a bound ``energy_core``, not a module."""

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


class TestExplainSummary:
    """Tests for Compiler.explain() and the molnex-owned ExplainSummary."""

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

    def test_explain_summary_is_the_only_explain_surface(self):
        """ac-002: the return type is exported beside ``Compiler``; the bare
        count helper it replaces is gone."""
        import molix

        assert molix.ExplainSummary is ExplainSummary
        assert "ExplainSummary" in molix.__all__
        assert not hasattr(Compiler, "count_graph_breaks")
