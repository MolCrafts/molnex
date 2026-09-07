"""Tests for :mod:`molix.core.steps.multitask` (TaskSpec / TaskRouter / steps).

Every test targets a single method or invariant of the multi-task routing
surface. Losses and the model are local stubs, so nothing here imports
``molpot`` / ``molrep`` / ``molzoo`` — the step is task-agnostic code.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.config import config
from molix.core.state import TrainState
from molix.core.steps import multitask as multitask_module
from molix.core.steps.eval import DefaultEvalStep
from molix.core.steps.multitask import (
    MultiTaskEvalStep,
    MultiTaskTrainStep,
    TaskRouter,
    TaskSpec,
)
from molix.core.steps.train import DefaultTrainStep
from molix.core.trainer import Trainer
from molix.hooks import GradClipHook

Batch = Mapping[str, Mapping[str, torch.Tensor]]
Predictions = dict[str, torch.Tensor]
LossFn = Callable[[Predictions, Batch], torch.Tensor]


@pytest.fixture(autouse=True)
def _reset_precision() -> None:
    """Precision is a global singleton — restore fp32 around every test."""
    config.set_precision("fp32")
    yield
    config.set_precision("fp32")


# ---------------------------------------------------------------------------
# Stubs (deterministic; no RNG, no wall clock, no filesystem)
# ---------------------------------------------------------------------------

_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (1.0, 0.5, -0.5),
    (0.25, -0.75, 0.5),
    (-1.0, 0.25, 0.75),
)


class _StubModel(nn.Module):
    """Positions-dependent stub encoder: ``pos`` -> ``{"energy": (N,)}``."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1, dtype=dtype)

    def forward(self, batch: Batch) -> Predictions:
        pos = batch["atoms"]["pos"]
        return {"energy": self.linear(pos).squeeze(-1)}


class _RecordingLoss:
    """Records its call arguments and returns a parameterized scalar."""

    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor
        self.calls: list[tuple[Predictions, Batch]] = []

    def __call__(self, predictions: Predictions, batch: Batch) -> torch.Tensor:
        self.calls.append((predictions, batch))
        return self.factor * predictions["energy"].square().mean()


class _AfterBackwardCounter:
    """Counts ``on_after_backward`` fires (accumulation-boundary probe)."""

    def __init__(self) -> None:
        self.calls = 0

    def on_after_backward(self, trainer: Trainer, state: TrainState) -> None:
        self.calls += 1


def _positions(dtype: torch.dtype, shift: float = 0.0) -> torch.Tensor:
    return torch.tensor(_POSITIONS, dtype=dtype) + shift


def _plain_batch(
    code: int,
    *,
    dtype: torch.dtype = torch.float32,
    n_graphs: int = 2,
    shift: float = 0.0,
) -> dict[str, dict[str, torch.Tensor]]:
    """Hand-built single-task batch as a plain nested ``dict``."""
    return {
        "atoms": {"pos": _positions(dtype, shift)},
        "graphs": {"task": torch.full((n_graphs,), code, dtype=torch.int64)},
    }


def _tensordict_batch(
    code: int,
    *,
    dtype: torch.dtype = torch.float32,
    n_graphs: int = 2,
    shift: float = 0.0,
) -> TensorDict:
    """Same batch as :func:`_plain_batch`, as a nested ``TensorDict``."""
    pos = _positions(dtype, shift)
    return TensorDict(
        {
            "atoms": TensorDict({"pos": pos}, batch_size=[pos.shape[0]]),
            "graphs": TensorDict(
                {"task": torch.full((n_graphs,), code, dtype=torch.int64)},
                batch_size=[n_graphs],
            ),
        },
        batch_size=[],
    )


def _four_tasks() -> tuple[TaskSpec, ...]:
    """The v0 task mixture: energy+forces / hessian / torsion / charge."""
    return (
        TaskSpec(name="energy_force", code=0, loss=_RecordingLoss(1.0)),
        TaskSpec(name="hessian", code=1, loss=_RecordingLoss(2.0)),
        TaskSpec(name="torsion", code=2, loss=_RecordingLoss(3.0)),
        TaskSpec(name="charge", code=3, loss=_RecordingLoss(4.0)),
    )


def _make_trainer(
    *,
    loss_fn: LossFn | None = None,
    model: nn.Module | None = None,
    train_step: object | None = None,
    eval_step: object | None = None,
    hooks: list[object] | None = None,
    accumulate_grad_batches: int = 1,
    lr: float = 0.1,
    dtype: torch.dtype = torch.float32,
) -> Trainer:
    torch.manual_seed(0)
    return Trainer(
        model=model if model is not None else _StubModel(dtype=dtype),
        loss_fn=loss_fn,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=lr),
        train_step=train_step,
        eval_step=eval_step,
        hooks=hooks,
        accumulate_grad_batches=accumulate_grad_batches,
    )


def _run_window(
    trainer: Trainer,
    step: object,
    batches: list[TensorDict],
) -> tuple[list[bool], list[torch.Tensor]]:
    """Drive one accumulation window, advancing the Trainer's micro-step counter."""
    state = TrainState()
    applied: list[bool] = []
    losses: list[torch.Tensor] = []
    for batch in batches:
        outputs = step.on_train_batch(trainer, state, batch)
        applied.append(bool(outputs["optimizer_applied"]))
        losses.append(state["train"]["loss"].clone())
        trainer._micro_step += 1
    return applied, losses


def _slash_keys(node: Mapping[str, object], prefix: str = "") -> list[str]:
    """Every key containing ``/`` anywhere in a nested mapping."""
    found: list[str] = []
    for key, value in node.items():
        if isinstance(key, str) and "/" in key:
            found.append(f"{prefix}{key}")
        if isinstance(value, Mapping):
            found.extend(_slash_keys(value, prefix=f"{prefix}{key}."))
    return found


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------


class TestTaskSpec:
    """Basics + immutability of the three-field task declaration."""

    def test_fields_are_exactly_name_code_loss(self) -> None:
        """No ``weight`` / ``scale`` / ``reference_method`` field (spec §2/§3)."""
        names = tuple(field.name for field in dataclasses.fields(TaskSpec))
        assert names == ("name", "code", "loss")

    def test_holds_its_declaration(self) -> None:
        loss = _RecordingLoss()
        spec = TaskSpec(name="charge", code=3, loss=loss)
        assert (spec.name, spec.code) == ("charge", 3)
        assert spec.loss is loss

    def test_is_frozen(self) -> None:
        spec = TaskSpec(name="charge", code=3, loss=_RecordingLoss())
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.code = 4  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# TaskRouter
# ---------------------------------------------------------------------------


class TestTaskRouter:
    """Label resolution and eager, actionable validation."""

    @pytest.mark.parametrize("code", [0, 1, 2, 3])
    def test_route_dispatches_each_code_to_its_own_task(self, code: int) -> None:
        tasks = _four_tasks()
        router = TaskRouter(tasks)

        spec = router.route(_tensordict_batch(code))

        assert spec is tasks[code]

    @pytest.mark.parametrize("code", [0, 1, 2, 3])
    def test_route_returns_the_task_whose_loss_records_the_call(self, code: int) -> None:
        tasks = _four_tasks()
        router = TaskRouter(tasks)
        batch = _tensordict_batch(code)
        predictions: Predictions = {"energy": torch.zeros(4)}

        router.route(batch).loss(predictions, batch)

        called = [spec.name for spec in tasks if spec.loss.calls]
        assert called == [tasks[code].name]

    def test_route_treats_plain_dict_like_tensordict(self) -> None:
        tasks = _four_tasks()
        router = TaskRouter(tasks)

        assert router.route(_plain_batch(2)) is router.route(_tensordict_batch(2))

    def test_route_reads_a_custom_task_key(self) -> None:
        tasks = _four_tasks()
        router = TaskRouter(tasks, task_key=("meta", "kind"))
        batch = {"meta": {"kind": torch.tensor([1, 1])}}

        assert router.route(batch) is tasks[1]

    @pytest.mark.parametrize(
        "batch",
        [
            {"atoms": {"pos": torch.zeros(4, 3)}},
            {"graphs": {"energy": torch.zeros(2)}},
        ],
        ids=["no-graphs-namespace", "no-task-entry"],
    )
    def test_route_rejects_a_missing_label(self, batch: Batch) -> None:
        router = TaskRouter(_four_tasks())

        with pytest.raises(ValueError, match="task") as excinfo:
            router.route(batch)

        message = str(excinfo.value)
        assert "graphs" in message
        assert "task" in message

    def test_route_rejects_an_empty_label(self) -> None:
        router = TaskRouter(_four_tasks())
        batch = {"graphs": {"task": torch.zeros(0, dtype=torch.int64)}}

        with pytest.raises(ValueError, match="task"):
            router.route(batch)

    def test_route_rejects_a_floating_label(self) -> None:
        router = TaskRouter(_four_tasks())
        batch = {"graphs": {"task": torch.zeros(2, dtype=torch.float32)}}

        with pytest.raises(ValueError, match="task") as excinfo:
            router.route(batch)

        assert "float32" in str(excinfo.value)

    def test_route_rejects_an_undeclared_code(self) -> None:
        router = TaskRouter(_four_tasks())

        with pytest.raises(ValueError, match="7") as excinfo:
            router.route(_plain_batch(7))

        message = str(excinfo.value)
        for name in ("energy_force", "hessian", "torsion", "charge"):
            assert name in message

    def test_route_rejects_a_mixed_task_batch(self) -> None:
        router = TaskRouter(_four_tasks())
        batch = {"graphs": {"task": torch.tensor([0, 1], dtype=torch.int64)}}

        with pytest.raises(ValueError) as excinfo:
            router.route(batch)

        message = str(excinfo.value)
        assert "0" in message
        assert "1" in message

    def test_rejects_an_empty_task_list(self) -> None:
        with pytest.raises(ValueError):
            TaskRouter(())

    def test_rejects_duplicate_task_names(self) -> None:
        loss = _RecordingLoss()
        tasks = (
            TaskSpec(name="charge", code=0, loss=loss),
            TaskSpec(name="charge", code=1, loss=loss),
        )

        with pytest.raises(ValueError, match="charge"):
            TaskRouter(tasks)

    def test_rejects_duplicate_task_codes(self) -> None:
        loss = _RecordingLoss()
        tasks = (
            TaskSpec(name="charge", code=1, loss=loss),
            TaskSpec(name="hessian", code=1, loss=loss),
        )

        with pytest.raises(ValueError, match="1"):
            TaskRouter(tasks)

    def test_exposes_its_tasks_in_declaration_order(self) -> None:
        tasks = _four_tasks()

        assert tuple(TaskRouter(tasks).tasks) == tasks


# ---------------------------------------------------------------------------
# MultiTaskTrainStep
# ---------------------------------------------------------------------------


class TestMultiTaskTrainStep:
    """Cadence reuse (no fork), state contract, hot path, parity."""

    def test_does_not_override_on_train_batch(self) -> None:
        assert MultiTaskTrainStep.on_train_batch is DefaultTrainStep.on_train_batch

    def test_on_eval_batch_is_not_implemented(self) -> None:
        step = MultiTaskTrainStep(TaskRouter(_four_tasks()))

        with pytest.raises(NotImplementedError):
            step.on_eval_batch(None, None, None)

    def test_advertises_one_scalar_key_per_task(self) -> None:
        router = TaskRouter(_four_tasks())

        assert MultiTaskTrainStep(router).scalar_keys == (
            ("train", "loss_energy_force"),
            ("train", "loss_hessian"),
            ("train", "loss_torsion"),
            ("train", "loss_charge"),
        )

    def test_writes_the_routed_task_scalar_detached(self) -> None:
        tasks = _four_tasks()
        step = MultiTaskTrainStep(TaskRouter(tasks))
        trainer = _make_trainer(train_step=step)
        state = TrainState()

        step.on_train_batch(trainer, state, _tensordict_batch(1))

        scalar = state["train"]["loss_hessian"]
        assert isinstance(scalar, torch.Tensor)
        assert not isinstance(scalar, float)
        assert scalar.requires_grad is False
        assert scalar.grad_fn is None

    def test_writes_only_the_routed_task_scalar(self) -> None:
        tasks = _four_tasks()
        step = MultiTaskTrainStep(TaskRouter(tasks))
        trainer = _make_trainer(train_step=step)
        state = TrainState()

        step.on_train_batch(trainer, state, _tensordict_batch(1))

        assert "loss_hessian" in state["train"]
        assert "loss_charge" not in state["train"]

    def test_still_writes_the_inherited_loss_scalar(self) -> None:
        step = MultiTaskTrainStep(TaskRouter(_four_tasks()))
        trainer = _make_trainer(train_step=step)
        state = TrainState()

        step.on_train_batch(trainer, state, _tensordict_batch(0))

        assert torch.equal(state["train"]["loss"], state["train"]["loss_energy_force"])

    def test_writes_no_slash_key_into_state(self) -> None:
        step = MultiTaskTrainStep(TaskRouter(_four_tasks()))
        trainer = _make_trainer(train_step=step)
        state = TrainState()

        step.on_train_batch(trainer, state, _tensordict_batch(3))

        assert _slash_keys(state) == []

    def test_runs_without_a_trainer_loss_fn(self) -> None:
        """The step owns its losses — ``Trainer(loss_fn=None)`` must work."""
        step = MultiTaskTrainStep(TaskRouter(_four_tasks()))
        trainer = _make_trainer(loss_fn=None, train_step=step)

        outputs = step.on_train_batch(trainer, TrainState(), _tensordict_batch(0))

        assert outputs["optimizer_applied"] is True

    def test_makes_no_host_sync_and_no_clipping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No ``.item()`` (launch-bound hot path) and no in-step gradient clipping."""

        def _no_item(self: torch.Tensor) -> float:
            raise AssertionError("Tensor.item() forces a host sync in the hot path")

        def _no_clip(*args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("clipping is owned by GradClipHook, not the step")

        monkeypatch.setattr(torch.Tensor, "item", _no_item)
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _no_clip)

        step = MultiTaskTrainStep(TaskRouter(_four_tasks()))
        trainer = _make_trainer(train_step=step)

        outputs = step.on_train_batch(trainer, TrainState(), _tensordict_batch(2))

        assert outputs["optimizer_applied"] is True

    def test_grad_clip_hook_still_records_the_grad_norm(self) -> None:
        step = MultiTaskTrainStep(TaskRouter(_four_tasks()))
        trainer = _make_trainer(train_step=step, hooks=[GradClipHook(max_norm=1.0)])
        state = TrainState()

        step.on_train_batch(trainer, state, _tensordict_batch(0))

        assert isinstance(state["train"]["grad_norm"], torch.Tensor)

    def test_matches_the_default_step_on_a_single_task_stream(self) -> None:
        """fp64 + fixed seed + ``accumulate_grad_batches=2``: byte-for-byte cadence."""
        config.set_precision("fp64")
        torch.manual_seed(20260814)
        reference_model = _StubModel(dtype=torch.float64)
        multitask_model = copy.deepcopy(reference_model)
        loss = _RecordingLoss()
        batches = [
            _tensordict_batch(0, dtype=torch.float64, shift=0.0),
            _tensordict_batch(0, dtype=torch.float64, shift=0.5),
        ]

        reference_counter = _AfterBackwardCounter()
        reference_trainer = _make_trainer(
            loss_fn=loss,
            model=reference_model,
            train_step=DefaultTrainStep(),
            hooks=[reference_counter],
            accumulate_grad_batches=2,
        )
        multitask_counter = _AfterBackwardCounter()
        multitask_step = MultiTaskTrainStep(
            TaskRouter((TaskSpec(name="energy_force", code=0, loss=loss),))
        )
        multitask_trainer = _make_trainer(
            loss_fn=None,
            model=multitask_model,
            train_step=multitask_step,
            hooks=[multitask_counter],
            accumulate_grad_batches=2,
        )

        reference_applied, reference_losses = _run_window(
            reference_trainer, reference_trainer.train_step, batches
        )
        multitask_applied, multitask_losses = _run_window(
            multitask_trainer, multitask_step, batches
        )

        assert reference_applied == [False, True]
        assert multitask_applied == reference_applied
        assert reference_counter.calls == 1
        assert multitask_counter.calls == reference_counter.calls
        for expected, actual in zip(reference_losses, multitask_losses):
            assert torch.equal(expected, actual)
        for expected_p, actual_p in zip(reference_model.parameters(), multitask_model.parameters()):
            assert torch.equal(expected_p, actual_p)


# ---------------------------------------------------------------------------
# MultiTaskEvalStep
# ---------------------------------------------------------------------------


class TestMultiTaskEvalStep:
    """Eval-side routing keeps the default grad-context semantics."""

    def test_does_not_override_on_eval_batch(self) -> None:
        assert MultiTaskEvalStep.on_eval_batch is DefaultEvalStep.on_eval_batch

    def test_advertises_eval_namespace_scalar_keys(self) -> None:
        router = TaskRouter(_four_tasks())

        assert MultiTaskEvalStep(router).scalar_keys == (
            ("eval", "loss_energy_force"),
            ("eval", "loss_hessian"),
            ("eval", "loss_torsion"),
            ("eval", "loss_charge"),
        )

    def test_writes_the_routed_task_scalar_detached(self) -> None:
        step = MultiTaskEvalStep(TaskRouter(_four_tasks()))
        trainer = _make_trainer(eval_step=step)
        state = TrainState()

        step.on_eval_batch(trainer, state, _tensordict_batch(2))

        scalar = state["eval"]["loss_torsion"]
        assert isinstance(scalar, torch.Tensor)
        assert scalar.requires_grad is False
        assert scalar.grad_fn is None

    def test_keeps_the_graph_alive_by_default(self) -> None:
        """Default ``no_grad=False`` — force / Hessian losses need a live graph."""
        step = MultiTaskEvalStep(TaskRouter(_four_tasks()))
        trainer = _make_trainer(eval_step=step)

        outputs = step.on_eval_batch(trainer, TrainState(), _tensordict_batch(0))

        assert outputs["loss"].requires_grad is True

    def test_forwards_no_grad_to_the_default_step(self) -> None:
        step = MultiTaskEvalStep(TaskRouter(_four_tasks()), no_grad=True)
        trainer = _make_trainer(eval_step=step)

        outputs = step.on_eval_batch(trainer, TrainState(), _tensordict_batch(0))

        assert outputs["loss"].requires_grad is False


# ---------------------------------------------------------------------------
# Static guards on the module source
# ---------------------------------------------------------------------------


def _multitask_ast() -> ast.Module:
    source = Path(multitask_module.__file__).read_text(encoding="utf-8")
    return ast.parse(source)


def test_multitask_module_imports_no_science_package() -> None:
    """``molix`` must gain no ``molpot`` / ``molrep`` / ``molzoo`` dependency."""
    imported: set[str] = set()
    for node in ast.walk(_multitask_ast()):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    assert not [name for name in imported if name.split(".")[0] in {"molpot", "molrep", "molzoo"}]


def test_multitask_module_never_calls_item() -> None:
    """Per-task scalars stay detached device tensors — one host sync per batch."""
    called = {
        node.func.attr
        for node in ast.walk(_multitask_ast())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "item" not in called


def test_multitask_module_exposes_no_weighting_surface() -> None:
    """No weight / scale / normalization / reference_method parameter (spec §2/§3)."""
    forbidden = {"weight", "scale", "normalization", "reference_method"}
    parameters = {arg.arg for arg in ast.walk(_multitask_ast()) if isinstance(arg, ast.arg)}
    fields = {
        node.target.id
        for node in ast.walk(_multitask_ast())
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert not forbidden & (parameters | fields)
