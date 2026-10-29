"""Unit tests for molzoo.checkpoint (shared official-checkpoint machinery)."""

from __future__ import annotations

import io
import pickle
import sys

import pytest
import torch

from molzoo.checkpoint import (
    CheckpointRemap,
    ModuleStub,
    OfficialArchive,
    StubUnpickler,
)


class TestCheckpointRemap:
    def test_longest_prefix_wins_and_none_drops(self):
        remap = CheckpointRemap({"a.": "x.", "a.b": "y", "a.c": None})
        renamed = remap.rename({"a.w": torch.ones(1), "a.b": torch.ones(1), "a.c": torch.ones(1)})
        assert sorted(renamed) == ["x.w", "y"]

    def test_rejects_alias_collision(self):
        remap = CheckpointRemap({"old.": "new."})
        with pytest.raises(RuntimeError, match="collision"):
            remap.rename({"old.weight": torch.ones(1), "new.weight": torch.zeros(1)})

    def test_preflight_rejection_does_not_write_parameters(self):
        target = torch.nn.Linear(2, 3)
        before = {k: v.clone() for k, v in target.state_dict().items()}
        with pytest.raises(RuntimeError, match="not covered"):
            CheckpointRemap({}).load(target, {"weight": torch.zeros(3, 2)})
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_equal_numel_does_not_prove_layout(self):
        target = torch.nn.Linear(2, 3, bias=False)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            CheckpointRemap({}).load(target, {"weight": torch.zeros(2, 3)})

    def test_a_declared_in_out_layout_is_transposed_by_type_not_shape(self):
        """A module declaring ``upstream_linear_layout = "in_out"`` gets upstream's
        ``(in, out)`` ``.weight`` transposed onto ``.linear.weight`` — even when the
        matrix is square and the shapes would have matched untransposed."""

        class Declared(torch.nn.Module):
            upstream_linear_layout = "in_out"

            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3, bias=False)

        target = torch.nn.Module()
        target.m = Declared()
        upstream = torch.arange(9.0).reshape(3, 3)  # (in, out); not symmetric
        assert CheckpointRemap({}).load(target, {"m.weight": upstream}) == ([], [])
        torch.testing.assert_close(target.m.linear.weight, upstream.t(), atol=0, rtol=0)

    def test_a_registered_conversion_is_applied_by_type_not_shape(self):
        """A module of a type in ``conversions`` has that function applied to
        upstream's ``.weight`` — even when the shapes already agree and no
        reshape rule would have fired."""

        def transposed(module: torch.nn.Module, weight: torch.Tensor) -> torch.Tensor:
            return weight.t()

        class Declared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(3, 3))

        target = torch.nn.Module()
        target.m = Declared()
        upstream = torch.arange(9.0).reshape(3, 3)  # not symmetric
        remap = CheckpointRemap({}, conversions={Declared: transposed})
        assert remap.load(target, {"m.weight": upstream}) == ([], [])
        torch.testing.assert_close(target.m.weight, upstream.t(), atol=0, rtol=0)

    def test_unknown_policy_is_refused(self):
        with pytest.raises(ValueError, match="on_unexpected"):
            CheckpointRemap({}, on_unexpected="ignore")

    def test_skipped_substrings_are_dropped_before_the_unexpected_check(self):
        """A key naming a ``skip`` substring is neither renamed nor reported."""
        remap = CheckpointRemap({}, skip=(".graph.c", "output_mask"))
        renamed = remap.rename(
            {
                "a.f.m.graphs.0.graph.c0": torch.ones(1),
                "b.output_mask": torch.ones(1),
                "w": torch.ones(1),
            }
        )
        assert list(renamed) == ["w"]
        target = torch.nn.Linear(2, 3)
        state = {
            "weight": torch.zeros(3, 2),
            "bias": torch.zeros(3),
            "conv.output_mask": torch.ones(3),
        }
        assert remap.load(target, state) == ([], [])


class _Upstream(torch.nn.Module):
    """Stands in for an upstream class the reader must not import."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.depth = 3


class TestStubUnpickler:
    def test_stubs_classes_outside_torch_and_stdlib(self):
        unpickler = StubUnpickler(io.BytesIO(b""))
        assert unpickler.find_class("collections", "OrderedDict").__name__ == "OrderedDict"
        assert issubclass(unpickler.find_class("allegro.nn", "Allegro_Module"), ModuleStub)


class TestOfficialArchive:
    def test_torch_load_rebuilds_the_tree_without_the_upstream_class(self, tmp_path, monkeypatch):
        path = tmp_path / "model.pt"
        torch.save(_Upstream(), path)
        # Hide the class: the reader must not need it.
        monkeypatch.delattr(sys.modules[__name__], "_Upstream")
        tree = OfficialArchive.torch_load(path)
        assert isinstance(tree, ModuleStub) and type(tree).__name__ == "_Upstream"
        assert tree.depth == 3
        assert set(tree.state_dict()) == {"linear.weight", "linear.bias"}

    def test_unpickle_reads_raw_bytes(self):
        assert OfficialArchive.unpickle(pickle.dumps({"a": 1})) == {"a": 1}

    @pytest.mark.parametrize("state", [{}, {"w": 1}, [torch.ones(1)]])
    def test_tensor_state_rejects_a_malformed_state(self, state):
        with pytest.raises(TypeError):
            OfficialArchive.tensor_state(state)
