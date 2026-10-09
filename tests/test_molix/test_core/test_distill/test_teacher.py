"""Teacher stamps nested columns from a TensorDict forward."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.core.distill import Teacher


def _batch() -> TensorDict:
    """Two atoms, one graph, with an energy label and positions."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    return TensorDict(
        {
            "atoms": TensorDict({"pos": pos}, batch_size=[2]),
            "graphs": TensorDict({"energy": torch.tensor([1.5])}, batch_size=[1]),
        },
        batch_size=[],
    )


class _InPlace(nn.Module):
    """Writes energy and features onto the batch and returns it."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(4.0))

    def forward(self, batch: TensorDict) -> TensorDict:
        batch["graphs", "energy"] = batch["graphs", "energy"] + self.bias
        summed = batch["atoms", "pos"].sum(dim=-1, keepdim=True)
        batch["atoms", "node_features"] = summed + self.bias
        return batch


class _DictReturn(nn.Module):
    """The refused contract: predictions in a plain dict, labels restored."""

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        label = batch["graphs", "energy"]
        batch["graphs", "energy"] = label * 2
        batch["graphs", "energy"] = label
        return {"energy": label * 2}


class TestTeacher:
    """One forward: TensorDict columns in, detached destination columns out."""

    def test_stamps_columns_and_leaves_the_student_batch_untouched(self) -> None:
        batch = _batch()
        label = batch["graphs", "energy"].clone()
        pos = batch["atoms", "pos"].clone()
        teacher = Teacher(
            _InPlace(),
            columns=((("atoms", "node_features"), ("atoms", "node_features_teacher")),),
        )

        teacher.write(batch)

        expected = pos.sum(dim=-1, keepdim=True) + 4
        stamped = batch["atoms", "node_features_teacher"]
        assert torch.allclose(stamped, expected)
        assert stamped.grad_fn is None
        assert not stamped.requires_grad
        assert torch.equal(batch["graphs", "energy"], label)
        assert torch.equal(batch["atoms", "pos"], pos)
        assert "node_features" not in batch["atoms"].keys()

    def test_student_backward_does_not_reach_the_teacher(self) -> None:
        batch = _batch()
        module = _InPlace()
        teacher = Teacher(
            module,
            columns=((("graphs", "energy"), ("graphs", "energy_teacher")),),
        )
        teacher.write(batch)
        student = nn.Linear(3, 1)
        predicted = student(batch["atoms", "pos"]).sum()
        loss = (predicted - batch["graphs", "energy_teacher"].sum()).square()

        loss.backward()

        assert module.bias.grad is None
        assert student.weight.grad is not None

    def test_dict_return_is_refused(self) -> None:
        teacher = Teacher(
            _DictReturn(),
            columns=((("graphs", "energy"), ("graphs", "energy_teacher")),),
        )

        with pytest.raises(TypeError, match="TensorDict"):
            teacher.write(_batch())

    def test_missing_source_raises(self) -> None:
        teacher = Teacher(
            _InPlace(),
            columns=((("atoms", "forces"), ("atoms", "forces_teacher")),),
        )

        with pytest.raises(KeyError, match="forces"):
            teacher.write(_batch())

    def test_missing_destination_namespace_raises(self) -> None:
        teacher = Teacher(
            _InPlace(),
            columns=((("graphs", "energy"), ("missing", "energy_teacher")),),
        )

        with pytest.raises(KeyError, match="missing"):
            teacher.write(_batch())

    def test_device_mismatch_raises_before_the_forward(self) -> None:
        module = _InPlace()
        module.to("meta")
        teacher = Teacher(
            module,
            columns=((("graphs", "energy"), ("graphs", "energy_teacher")),),
        )

        with pytest.raises(ValueError, match="meta"):
            teacher.write(_batch())

    def test_write_inside_a_compiled_region_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        teacher = Teacher(
            _InPlace(),
            columns=((("graphs", "energy"), ("graphs", "energy_teacher")),),
        )

        with pytest.raises(RuntimeError, match="eager prologue"):
            teacher.write(_batch())

    def test_overlapping_destinations_raise_at_init(self) -> None:
        columns = (
            (("atoms", "node_features"), ("atoms", "node_features_teacher")),
            (("graphs", "energy"), ("atoms", "node_features_teacher")),
        )

        with pytest.raises(ValueError, match="overlap"):
            Teacher(_InPlace(), columns=columns)
