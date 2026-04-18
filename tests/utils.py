"""Test utilities for torch.compile compatibility assertions.

Provides a standard harness for validating that nn.Module subclasses are
compatible with modern ``torch.compile`` (PyTorch >=2.6). Two modes are
supported:

- **strict**: ``backend="inductor"``, ``fullgraph=True`` — asserts zero graph
  breaks and numerical equivalence. Use for modules that must compile cleanly
  into a single graph for inference/export.
- **relaxed**: ``backend="eager"``, ``fullgraph=False`` — only verifies that
  compilation does not raise and outputs match. Use for modules with known
  graph breaks (e.g. cuEquivariance, scatter, autograd.grad) where full-graph
  compilation is not yet supported upstream.
"""

from __future__ import annotations

import torch
import torch.export
import torch.nn as nn


def assert_compile_compatible(
    module: nn.Module,
    *args,
    strict: bool = False,
    check_graph_breaks: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    **kwargs,
):
    """Assert a module is compatible with ``torch.compile``.

    Args:
        module: Module under test.
        *args: Positional forward arguments.
        strict: If True, use ``backend="inductor"`` and ``fullgraph=True``.
            Otherwise use ``backend="eager"`` and ``fullgraph=False``.
        check_graph_breaks: If True, assert zero graph breaks via
            ``torch._dynamo.explain``. Implied when ``strict=True``.
        rtol: Numerical tolerance for output comparison.
        atol: Numerical tolerance for output comparison.
        **kwargs: Forward keyword arguments.

    Returns:
        tuple ``(uncompiled_output, compiled_output)``.
    """
    torch._dynamo.reset()
    module.eval()

    with torch.no_grad():
        output_uncompiled = module(*args, **kwargs)

    if check_graph_breaks or strict:
        explanation = torch._dynamo.explain(module)(*args, **kwargs)
        assert explanation.graph_break_count == 0, (
            f"{type(module).__name__} has "
            f"{explanation.graph_break_count} graph break(s): "
            f"{explanation.break_reasons}"
        )

    backend = "inductor" if strict else "eager"
    fullgraph = bool(strict)
    compiled_module = torch.compile(module, backend=backend, fullgraph=fullgraph)

    with torch.no_grad():
        output_compiled = compiled_module(*args, **kwargs)

    assert_outputs_close(output_uncompiled, output_compiled, rtol=rtol, atol=atol)
    return output_uncompiled, output_compiled


def assert_module_compiles(module, *args, **kwargs):
    """Backwards-compatible relaxed compile check.

    Equivalent to ``assert_compile_compatible(module, *args, strict=False)``
    but without numerical comparison (callers do that themselves).
    """
    with torch.no_grad():
        output_uncompiled = module(*args, **kwargs)

    compiled_module = torch.compile(module, backend="eager", fullgraph=False)

    with torch.no_grad():
        output_compiled = compiled_module(*args, **kwargs)

    return output_uncompiled, output_compiled


def assert_module_exports(module, args_tuple, kwargs_dict=None):
    """Assert that a module can be exported with torch.export."""
    kwargs_dict = kwargs_dict or {}
    module.eval()

    exported_program = torch.export.export(
        module,
        args=args_tuple,
        kwargs=kwargs_dict,
        strict=False,
    )

    with torch.no_grad():
        output_original = module(*args_tuple, **kwargs_dict)
        output_exported = exported_program.module()(*args_tuple, **kwargs_dict)

    return exported_program, output_original, output_exported


def assert_outputs_close(output1, output2, rtol=1e-4, atol=1e-4):
    """Assert that two outputs are close, recursing into tuples/lists/dicts."""
    if isinstance(output1, torch.Tensor):
        assert torch.allclose(output1, output2, rtol=rtol, atol=atol)
    elif isinstance(output1, (tuple, list)):
        assert len(output1) == len(output2)
        for o1, o2 in zip(output1, output2):
            assert_outputs_close(o1, o2, rtol=rtol, atol=atol)
    elif isinstance(output1, dict):
        assert set(output1.keys()) == set(output2.keys())
        for key in output1.keys():
            assert_outputs_close(output1[key], output2[key], rtol=rtol, atol=atol)
    else:
        assert output1 == output2
