"""Distill steps stamp teacher columns before the student runs."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.compile import Compiler
from molix.core.distill import DistillEvalStep, DistillTrainStep, Teacher
from molix.core.state import TrainState
from molix.core.trainer import Trainer

_ENERGY = (("graphs", "energy"), ("graphs", "energy_teacher"))


def _batch() -> TensorDict:
    """Two atoms, one graph."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    return TensorDict(
        {
            "atoms": TensorDict({"pos": pos}, batch_size=[2]),
            "graphs": TensorDict({"energy": torch.tensor([1.5])}, batch_size=[1]),
        },
        batch_size=[],
    )


class _EnergyModel(nn.Module):
    """Writes a constant energy onto the batch and returns it."""

    def __init__(self, value: float, log: list[str], name: str) -> None:
        super().__init__()
        self.value = value
        self.log = log
        self.name = name
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: TensorDict) -> TensorDict:
        self.log.append(self.name)
        batch["graphs", "energy"] = torch.full_like(batch["graphs", "energy"], self.value)
        batch["graphs", "energy"] = batch["graphs", "energy"] * self.scale
        return batch


class _Student(nn.Module):
    """Reads the teacher column the prologue just stamped."""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log
        self.linear = nn.Linear(3, 1)

    def forward(self, batch: TensorDict) -> TensorDict:
        seen = batch["graphs", "energy_teacher"]
        self.log.append(f"student:{float(seen[0])}")
        total = self.linear(batch["atoms", "pos"]).sum()
        batch["graphs", "student_energy"] = total.reshape(1)
        return batch


def _loss(preds: TensorDict, batch: TensorDict) -> torch.Tensor:
    return (preds["graphs", "student_energy"] - batch["graphs", "energy_teacher"]).square().mean()


def _trainer(model: nn.Module, step: object) -> Trainer:
    return Trainer(
        model=model,
        loss_fn=_loss,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        train_step=step if isinstance(step, DistillTrainStep) else None,
        eval_step=step if isinstance(step, DistillEvalStep) else None,
    )


class TestDistillSteps:
    """Prologue order, disjoint destinations, and the eval twin."""

    def test_train_step_stamps_before_the_student(self) -> None:
        log: list[str] = []
        teacher = Teacher(_EnergyModel(4.0, log, "teacher"), columns=(_ENERGY,))
        step = DistillTrainStep((teacher,))
        trainer = _trainer(_Student(log), step)

        loss, _ = step.forward_loss(trainer, TrainState(), _batch())

        assert log == ["teacher", "student:4.0"]
        assert torch.isfinite(loss)

    def test_eval_step_stamps_before_the_student(self) -> None:
        log: list[str] = []
        teacher = Teacher(_EnergyModel(7.0, log, "teacher"), columns=(_ENERGY,))
        step = DistillEvalStep((teacher,))
        trainer = _trainer(_Student(log), step)

        step.forward_loss(trainer, TrainState(), _batch())

        assert log == ["teacher", "student:7.0"]

    def test_overlapping_destinations_raise(self) -> None:
        log: list[str] = []
        first = Teacher(_EnergyModel(1.0, log, "a"), columns=(_ENERGY,))
        second = Teacher(_EnergyModel(2.0, log, "b"), columns=(_ENERGY,))

        with pytest.raises(ValueError, match="overlap"):
            DistillTrainStep((first, second))

    def test_bare_teacher_is_not_a_sequence(self) -> None:
        teacher = Teacher(_EnergyModel(1.0, [], "a"), columns=(_ENERGY,))

        with pytest.raises(TypeError, match="tuple"):
            DistillTrainStep(teacher)  # type: ignore[arg-type]

    def test_empty_teachers_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            DistillTrainStep(())

    def test_two_teachers_write_disjoint_columns(self) -> None:
        log: list[str] = []
        energy = Teacher(_EnergyModel(3.0, log, "energy"), columns=(_ENERGY,))

        class _Feature(nn.Module):
            def forward(self, batch: TensorDict) -> TensorDict:
                log.append("feature")
                batch["atoms", "node_features"] = batch["atoms", "pos"]
                return batch

        feature = Teacher(
            _Feature(),
            columns=((("atoms", "node_features"), ("atoms", "node_features_teacher")),),
        )
        step = DistillTrainStep((energy, feature))
        batch = _batch()

        step.forward_loss(_trainer(_Student(log), step), TrainState(), batch)

        assert log[:2] == ["energy", "feature"]
        assert torch.equal(batch["graphs", "energy_teacher"], torch.tensor([3.0]))
        assert torch.equal(batch["atoms", "node_features_teacher"], batch["atoms", "pos"])


class _HostSyncTeacher(nn.Module):
    """``.item()`` would break a graph that traced this forward."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: TensorDict) -> TensorDict:
        self.calls += 1
        _ = batch["atoms", "pos"].detach().sum().item()
        ones = torch.ones_like(batch["graphs", "energy"])
        batch["graphs", "energy"] = ones * self.scale
        return batch


class _CleanStudent(nn.Module):
    """In-place scalar energy. Static keys, no host sync."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, batch: TensorDict) -> TensorDict:
        total = self.linear(batch["atoms", "pos"]).sum()
        batch["graphs", "student_energy"] = total.reshape(1)
        return batch


class TestDistillTrainStepTrace:
    """The compiled region is the student. The teacher's host sync stays outside it."""

    @pytest.mark.slow
    def test_traced_body_is_one_graph_when_the_teacher_syncs(self) -> None:
        module = _HostSyncTeacher()
        teacher = Teacher(module, columns=(_ENERGY,))
        step = DistillTrainStep((teacher,))
        trainer = Trainer(
            model=_CleanStudent(),
            loss_fn=_loss,
            optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
            train_step=step,
        )
        batch = _batch()

        step.forward_loss(trainer, trainer.state, batch)
        summary = Compiler.explain(step._traced_forward_loss, trainer, trainer.state, batch)

        assert module.calls == 1
        assert (summary.graph_count, summary.graph_break_count) == (1, 0), summary.break_sites
