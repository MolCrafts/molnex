"""Test utilities for compile and export assertions."""

import torch
import torch.export


def assert_module_compiles(module, *args, **kwargs):
    """Assert that a module can be compiled with torch.compile.

    Args:
        module: The PyTorch module to test
        *args: Positional arguments to pass to the module
        **kwargs: Keyword arguments to pass to the module

    Returns:
        tuple: (uncompiled_output, compiled_output)
    """
    # Test uncompiled forward pass
    with torch.no_grad():
        output_uncompiled = module(*args, **kwargs)

    # Compile the module
    compiled_module = torch.compile(module, backend="eager", fullgraph=False)

    # Test compiled forward pass
    with torch.no_grad():
        output_compiled = compiled_module(*args, **kwargs)

    return output_uncompiled, output_compiled


def assert_module_exports(module, args_tuple, kwargs_dict=None):
    """Assert that a module can be exported with torch.export.

    Args:
        module: The PyTorch module to test
        args_tuple: Tuple of example inputs for export
        kwargs_dict: Optional dict of keyword arguments

    Returns:
        torch.export.ExportedProgram: The exported program
    """
    kwargs_dict = kwargs_dict or {}

    # Set module to eval mode for export
    module.eval()

    # Export the module
    exported_program = torch.export.export(
        module,
        args=args_tuple,
        kwargs=kwargs_dict,
        strict=False,  # Allow some dynamic behavior
    )

    # Test that exported program can run
    with torch.no_grad():
        output_original = module(*args_tuple, **kwargs_dict)
        output_exported = exported_program.module()(*args_tuple, **kwargs_dict)

    return exported_program, output_original, output_exported


def assert_outputs_close(output1, output2, rtol=1e-4, atol=1e-4):
    """Assert that two outputs are close.

    Handles different output types (tensors, tuples, dicts).
    """
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
        # For other types, use equality
        assert output1 == output2
