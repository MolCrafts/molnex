"""Tests for the seams of :class:`molix.core.steps.train.DefaultTrainStep`.

Cadence (AMP / accumulation / ``optimizer_applied`` / ``on_after_backward``)
is covered by ``tests/test_molix/test_core/test_step_protocol.py``; this
module targets the two seams: ``compute_loss`` (change *what* is computed)
and ``forward_loss`` (the traced region ``Trainer.compile`` swaps).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.autograd import Derivative, Potential
from molix.compile import Compiler
from molix.config import config
from molix.core.losses import energy_force_mse, energy_mse
from molix.core.state import TrainState
from molix.core.steps.train import DefaultTrainStep
from molix.core.trainer import Trainer
from tests.conftest import make_graph_batch
from tests.test_molzoo.test_mace.conftest import (
    ATOMIC_ENERGIES,
    ATOMIC_NUMBERS,
    CLUSTER_POS,
    CLUSTER_Z,
    SINGLE_GRAPH,
    TINY_MATPES_KWARGS,
    full_edge_index,
)

Batch = Mapping[str, Mapping[str, torch.Tensor]]
Predictions = dict[str, torch.Tensor]
LossFn = Callable[[Predictions, Batch], torch.Tensor]


@pytest.fixture(autouse=True)
def _reset_precision() -> None:
    """Precision is a global singleton — restore fp32 around every test."""
    config.set_precision("fp32")
    yield
    config.set_precision("fp32")


class _StubModel(nn.Module):
    """Positions-dependent stub encoder: ``pos`` -> ``{"energy": (N,)}``."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, batch: Batch) -> Predictions:
        return {"energy": self.linear(batch["atoms"]["pos"]).squeeze(-1)}


class _RecordingLoss:
    """Records its call arguments and returns a parameterized scalar."""

    def __init__(self) -> None:
        self.calls: list[tuple[Predictions, Batch]] = []

    def __call__(self, predictions: Predictions, batch: Batch) -> torch.Tensor:
        self.calls.append((predictions, batch))
        return predictions["energy"].square().mean()


class _SpyTrainStep(DefaultTrainStep):
    """Records every ``compute_loss`` call and the autocast state it saw."""

    def __init__(self) -> None:
        self.calls: list[tuple[Batch, Predictions]] = []
        self.autocast_flags: list[bool] = []

    def compute_loss(
        self,
        trainer: Trainer,
        state: TrainState,
        batch: Batch,
        predictions: Predictions,
    ) -> torch.Tensor:
        self.calls.append((batch, predictions))
        self.autocast_flags.append(torch.is_autocast_enabled("cpu"))
        return super().compute_loss(trainer, state, batch, predictions)


def _batch() -> dict[str, dict[str, torch.Tensor]]:
    pos = torch.tensor(
        [(0.0, 0.0, 0.0), (1.0, 0.5, -0.5), (0.25, -0.75, 0.5)],
        dtype=torch.float32,
    )
    return {"atoms": {"pos": pos}}


def _make_trainer(
    *,
    loss_fn: LossFn | None,
    train_step: object | None = None,
    autocast: torch.dtype | None = None,
) -> Trainer:
    torch.manual_seed(0)
    return Trainer(
        model=_StubModel(),
        loss_fn=loss_fn,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        train_step=train_step,
        autocast=autocast,
    )


class _RecordingForwardLoss:
    """Records the calls through ``forward_loss`` — the attribute the cadence runs."""

    def __init__(self, inner: Callable[..., tuple[torch.Tensor, Any]]) -> None:
        self.inner = inner
        self.calls: list[tuple[Any, ...]] = []
        self.results: list[tuple[torch.Tensor, Any]] = []

    def __call__(self, *args: Any) -> tuple[torch.Tensor, Any]:
        self.calls.append(args)
        result = self.inner(*args)
        self.results.append(result)
        return result


class _Energy(nn.Module):
    """``graphs.energy = Σ_i linear(pos_i)`` written onto a frame — an energy for the op."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, batch: TensorDict) -> TensorDict:
        batch["graphs", "energy"] = self.linear(batch["atoms", "pos"]).sum().reshape(1)
        return batch


class _Potential(Potential):
    """A potential in the molnex shape: ``forward`` runs the op it exposes as ``derivative``."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self._derivative = Derivative(self.energy, wrt=("atoms.pos",))

    def energy(self, batch: TensorDict) -> TensorDict:
        batch["graphs", "energy"] = self.linear(batch["atoms", "pos"]).sum().reshape(1)
        return batch


class _RecordingTracedBody:
    """Stands in for the traced region: records ``atoms.pos`` as the region receives it."""

    def __init__(self, inner: Callable[..., tuple[torch.Tensor, Any]]) -> None:
        self.inner = inner
        self.received: list[torch.Tensor] = []

    def __call__(self, trainer: Trainer, state: TrainState, batch: TensorDict) -> Any:
        self.received.append(batch["atoms", "pos"])
        return self.inner(trainer, state, batch)


def _frame() -> TensorDict:
    """A three-atom, one-graph frame for the op-shaped models above."""
    return TensorDict(
        atoms=TensorDict(pos=_batch()["atoms"]["pos"], batch_size=[3]),
        graphs=TensorDict(batch_size=[1]),
        batch_size=[],
    )


def _force_square_loss(predictions: TensorDict, batch: TensorDict) -> torch.Tensor:
    return predictions["atoms", "forces"].square().mean()


def _energy_square_loss(predictions: TensorDict, batch: TensorDict) -> torch.Tensor:
    return predictions["graphs", "energy"].square().mean()


def _frame_trainer(model: nn.Module, *, loss_fn: LossFn, train_step: DefaultTrainStep) -> Trainer:
    torch.manual_seed(0)
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        train_step=train_step,
    )


def _mace_trainer(*, wrt: tuple[str, ...], loss_fn: LossFn) -> tuple[Trainer, TensorDict]:
    """A tiny fp64 MACE-MatPES force (or energy-only) trainer on the five-atom cluster.

    The targets ride under ``graphs.energy_ref`` / ``atoms.forces_ref`` because
    the model writes its own ``graphs.energy`` / ``atoms.forces`` onto the
    batch. One eager forward runs first: the element gate is the documented
    eager precondition of the compile boundary.
    """
    from molzoo.mace.potential import MaceMatpes
    from molzoo.mace.spec import MACEMatpesSpec

    config.set_precision("fp64")
    torch.manual_seed(0)
    model = MaceMatpes(
        MACEMatpesSpec(
            atomic_numbers=list(ATOMIC_NUMBERS),
            atomic_energies=list(ATOMIC_ENERGIES),
            **TINY_MATPES_KWARGS,
        ),
        wrt=wrt,
    )
    n_atoms = len(CLUSTER_Z)
    batch = make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        pair_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(n_atoms, dtype=torch.long),
        graphs={"energy_ref": torch.tensor([-3100.0], dtype=torch.float64)},
    )
    batch["atoms", "forces_ref"] = torch.zeros(n_atoms, 3, dtype=torch.float64)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.01),
    )
    model(batch)
    return trainer, batch


def _force_loss() -> LossFn:
    return energy_force_mse(
        energy_target_key="energy_ref",
        force_target_key="forces_ref",
        energy_pred_key=("graphs", "energy"),  # type: ignore[arg-type]
        force_pred_key=("atoms", "forces"),  # type: ignore[arg-type]
    )


class TestDefaultTrainStep:
    """The ``compute_loss`` and ``forward_loss`` seams: bodies and call sites."""

    def test_forward_loss_returns_the_seam_loss_and_the_predictions(self) -> None:
        loss_fn = _RecordingLoss()
        trainer = _make_trainer(loss_fn=loss_fn)
        batch = _batch()

        loss, predictions = DefaultTrainStep().forward_loss(trainer, TrainState(), batch)

        assert len(loss_fn.calls) == 1
        assert loss_fn.calls[0][0] is predictions
        assert torch.equal(loss, predictions["energy"].square().mean())
        assert torch.equal(predictions["energy"], trainer.model(batch)["energy"])

    def test_on_train_batch_runs_the_instance_forward_loss(self) -> None:
        """The cadence calls ``self.forward_loss`` — the attribute ``compile`` swaps."""
        step = DefaultTrainStep()
        trainer = _make_trainer(loss_fn=_RecordingLoss(), train_step=step)
        recording = _RecordingForwardLoss(step.forward_loss)
        step.forward_loss = recording  # type: ignore[method-assign]
        state = TrainState()
        batch = _batch()

        outputs = step.on_train_batch(trainer, state, batch)

        assert len(recording.calls) == 1
        assert all(got is want for got, want in zip(recording.calls[0], (trainer, state, batch)))
        assert outputs["loss"] is recording.results[0][0]
        assert outputs["predictions"] is recording.results[0][1]

    def test_forward_loss_marks_the_leaves_before_the_traced_body(self) -> None:
        """The traced region receives ``atoms.pos`` as a fresh live leaf — the caller-owned
        leaf rule that makes a force-supervised step one graph on stock torch."""
        step = DefaultTrainStep()
        model = Derivative(_Energy(), wrt=("atoms.pos",))
        trainer = _frame_trainer(model, loss_fn=_force_square_loss, train_step=step)
        recording = _RecordingTracedBody(step._traced_forward_loss)
        step._traced_forward_loss = recording  # type: ignore[method-assign]
        batch = _frame()
        pos = batch["atoms", "pos"]

        step.forward_loss(trainer, TrainState(), batch)

        (received,) = recording.received
        assert received is not pos
        assert received.is_leaf and received.requires_grad
        assert torch.equal(received, pos)

    def test_forward_loss_finds_the_op_behind_a_potential(self) -> None:
        """A :class:`Potential` — every molnex potential — is marked through its ``derivative``."""
        step = DefaultTrainStep()
        trainer = _frame_trainer(_Potential(), loss_fn=_force_square_loss, train_step=step)
        recording = _RecordingTracedBody(step._traced_forward_loss)
        step._traced_forward_loss = recording  # type: ignore[method-assign]

        step.forward_loss(trainer, TrainState(), _frame())

        (received,) = recording.received
        assert received.is_leaf and received.requires_grad

    def test_forward_loss_marks_nothing_without_variables(self) -> None:
        """An energy-only op (``wrt=()``) — or a model with no op — needs no prologue."""
        step = DefaultTrainStep()
        trainer = _frame_trainer(
            Derivative(_Energy()), loss_fn=_energy_square_loss, train_step=step
        )
        batch = _frame()
        pos = batch["atoms", "pos"]

        step.forward_loss(trainer, TrainState(), batch)

        assert batch["atoms", "pos"] is pos
        assert not pos.requires_grad

    def test_compute_loss_delegates_to_trainer_loss_fn(self) -> None:
        loss_fn = _RecordingLoss()
        trainer = _make_trainer(loss_fn=loss_fn)
        batch = _batch()
        predictions = trainer.model(batch)

        loss = DefaultTrainStep().compute_loss(trainer, TrainState(), batch, predictions)

        assert len(loss_fn.calls) == 1
        assert loss_fn.calls[0][0] is predictions
        assert loss_fn.calls[0][1] is batch
        assert torch.equal(loss, predictions["energy"].square().mean())

    def test_compute_loss_asserts_without_a_loss_fn(self) -> None:
        trainer = _make_trainer(loss_fn=None)
        batch = _batch()
        predictions = trainer.model(batch)

        with pytest.raises(AssertionError):
            DefaultTrainStep().compute_loss(trainer, TrainState(), batch, predictions)

    def test_on_train_batch_computes_the_loss_through_the_seam(self) -> None:
        step = _SpyTrainStep()
        trainer = _make_trainer(loss_fn=_RecordingLoss(), train_step=step)
        batch = _batch()

        outputs = step.on_train_batch(trainer, TrainState(), batch)

        assert len(step.calls) == 1
        assert step.calls[0][0] is batch
        assert step.calls[0][1] is outputs["predictions"]

    def test_seam_runs_outside_autocast_without_amp(self) -> None:
        step = _SpyTrainStep()
        trainer = _make_trainer(loss_fn=_RecordingLoss(), train_step=step)

        step.on_train_batch(trainer, TrainState(), _batch())

        assert step.autocast_flags == [False]

    def test_seam_runs_inside_autocast_with_amp(self) -> None:
        step = _SpyTrainStep()
        trainer = _make_trainer(loss_fn=_RecordingLoss(), train_step=step, autocast=torch.bfloat16)

        step.on_train_batch(trainer, TrainState(), _batch())

        assert step.autocast_flags == [True]


class TestDefaultTrainStepTrace:
    """``_traced_forward_loss`` under dynamo: the one region, after the eager leaf prologue."""

    @pytest.mark.slow
    def test_force_step_traces_to_one_graph(self) -> None:
        """Forward, ``autograd.grad`` and the E+F loss are one graph with zero breaks on
        stock torch: the position leaf is marked before the region, so it is a graph
        input and the attached forces leak nothing."""
        trainer, batch = _mace_trainer(wrt=("atoms.pos",), loss_fn=_force_loss())
        trainer.model.derivative.mark_leaves(batch)

        summary = Compiler.explain(
            trainer.train_step._traced_forward_loss, trainer, trainer.state, batch
        )

        assert (summary.graph_count, summary.graph_break_count) == (1, 0), summary.break_sites

    @pytest.mark.slow
    def test_energy_only_step_traces_to_one_graph(self) -> None:
        loss_fn = energy_mse("energy_ref", pred_key=("graphs", "energy"))  # type: ignore[arg-type]
        trainer, batch = _mace_trainer(wrt=(), loss_fn=loss_fn)

        summary = Compiler.explain(
            trainer.train_step._traced_forward_loss, trainer, trainer.state, batch
        )

        assert (summary.graph_count, summary.graph_break_count) == (1, 0), summary.break_sites

    @pytest.mark.slow
    def test_compiled_step_matches_eager_parameter_gradients(self) -> None:
        """Backward through the ``fullgraph`` region (AOTAutograd) equals eager to 1e-12 —
        through the installed route: ``forward_loss`` marks, the compiled body traces."""
        trainer, batch = _mace_trainer(wrt=("atoms.pos",), loss_fn=_force_loss())
        params = [p for p in trainer.model.parameters() if p.requires_grad]

        loss, _ = trainer.train_step.forward_loss(trainer, trainer.state, batch)
        loss.backward()
        expected = [p.grad.detach().clone() for p in params]
        for p in params:
            p.grad = None

        trainer.compile(backend="aot_eager", fullgraph=True)
        loss, _ = trainer.train_step.forward_loss(trainer, trainer.state, batch)
        loss.backward()

        for p, grad in zip(params, expected, strict=True):
            torch.testing.assert_close(p.grad, grad, atol=1e-12, rtol=0.0)
