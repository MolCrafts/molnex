"""Tests for the checkpoint infrastructure."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from molix.config import set_precision
from molix.core.checkpoint import (
    TorchSaveBackend,
    Checkpoint,
    capture_rng_states,
    restore_rng_states,
)
from molix.core.hooks import CheckpointHook
from molix.core.state import TrainState
from molix.core.trainer import Trainer


@pytest.fixture(autouse=True)
def _reset_precision():
    set_precision("fp32")
    yield
    set_precision("fp32")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x, **_kwargs):
        return self.linear(x)


def _simple_loss(predictions, batch):
    return ((predictions - batch["targets"]["y_energy"]) ** 2).mean()


def _make_batch():
    return {"x": torch.randn(4, 10), "targets": {"y_energy": torch.randn(4, 1)}}


class _MockDataModule:
    def __init__(self, batches_per_epoch: int = 3):
        self._n = batches_per_epoch

    def setup(self, stage: str = "fit") -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def train_dataloader(self):
        for _ in range(self._n):
            yield _make_batch()

    def val_dataloader(self):
        yield _make_batch()


def _make_trainer(**kwargs):
    model = _SimpleModel()
    return Trainer(
        model=model,
        loss_fn=_simple_loss,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.01),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# RNG state capture/restore
# ---------------------------------------------------------------------------


class TestRNGStates:
    def test_capture_returns_torch_and_python(self):
        states = capture_rng_states()
        assert "torch" in states
        assert "python" in states

    def test_roundtrip_torch_rng(self):
        states = capture_rng_states()
        # Generate some random numbers to change state
        torch.randn(100)
        restore_rng_states(states)
        # After restore, next random should match
        a = torch.randn(5)
        restore_rng_states(states)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_roundtrip_python_rng(self):
        states = capture_rng_states()
        random.random()  # advance state
        restore_rng_states(states)
        a = random.random()
        restore_rng_states(states)
        b = random.random()
        assert a == b


# ---------------------------------------------------------------------------
# TorchSaveBackend
# ---------------------------------------------------------------------------


class TestTorchSaveBackend:
    def test_save_load_roundtrip(self, tmp_path):
        backend = TorchSaveBackend()
        data = {"key": torch.tensor([1, 2, 3]), "epoch": 5}
        filepath = tmp_path / "test.pt"
        backend.save(data, filepath)
        loaded = backend.load(filepath)
        assert loaded["epoch"] == 5
        assert torch.equal(loaded["key"], data["key"])

    def test_atomic_write_creates_file(self, tmp_path):
        backend = TorchSaveBackend()
        filepath = tmp_path / "subdir" / "ckpt.pt"
        backend.save({"a": 1}, filepath)
        assert filepath.exists()

    def test_no_tmp_file_left_on_success(self, tmp_path):
        backend = TorchSaveBackend()
        filepath = tmp_path / "ckpt.pt"
        backend.save({"a": 1}, filepath)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


# ---------------------------------------------------------------------------
# Checkpoint state_dict / load_state_dict
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def _make_checkpoint(self, **kwargs):
        """Helper: create Checkpoint with required model+optimizer."""
        if "model" not in kwargs:
            kwargs["model"] = _SimpleModel()
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = torch.optim.SGD(kwargs["model"].parameters(), lr=0.01)
        return Checkpoint(**kwargs)

    def test_state_dict_contains_scalars(self):
        ts = self._make_checkpoint(epoch=3, global_step=100, best_metric=0.5)
        sd = ts.state_dict()
        assert sd["epoch"] == 3
        assert sd["global_step"] == 100
        assert sd["best_metric"] == 0.5
        assert "rng_states" in sd

    def test_state_dict_contains_model(self):
        model = _SimpleModel()
        ts = self._make_checkpoint(model=model)
        sd = ts.state_dict()
        assert "model_state_dict" in sd
        assert "linear.weight" in sd["model_state_dict"]

    def test_state_dict_contains_optimizer(self):
        model = _SimpleModel()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        ts = Checkpoint(model=model, optimizer=opt)
        sd = ts.state_dict()
        assert "optimizer_state_dict" in sd

    def test_state_dict_contains_lr_scheduler(self):
        model = _SimpleModel()
        opt = torch.optim.Adam(model.parameters())
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
        ts = Checkpoint(model=model, optimizer=opt, lr_scheduler=sched)
        sd = ts.state_dict()
        assert "lr_scheduler_state_dict" in sd

    def test_state_dict_contains_scaler(self):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler()
        ts = Checkpoint(model=model, optimizer=opt, scaler=scaler)
        sd = ts.state_dict()
        assert "scaler_state_dict" in sd

    def test_load_state_dict_restores_model(self):
        model_a = _SimpleModel()
        model_b = _SimpleModel()
        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)

        with torch.no_grad():
            model_a.linear.weight.fill_(1.0)
            model_b.linear.weight.fill_(0.0)

        ts_a = Checkpoint(model=model_a, optimizer=opt_a, epoch=5, global_step=50)
        sd = ts_a.state_dict()

        ts_b = Checkpoint(model=model_b, optimizer=opt_b)
        ts_b.load_state_dict(sd)

        assert ts_b.epoch == 5
        assert ts_b.global_step == 50
        assert torch.equal(model_b.linear.weight, model_a.linear.weight)

    def test_load_state_dict_restores_optimizer(self):
        model = _SimpleModel()
        # Adam always populates state (step count, exp_avg, exp_avg_sq)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        # Run a step to populate optimizer state
        loss = model(torch.randn(2, 10)).sum()
        loss.backward()
        opt.step()

        ts = Checkpoint(model=model, optimizer=opt)
        sd = ts.state_dict()

        # Fresh optimizer
        model2 = _SimpleModel()
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        ts2 = Checkpoint(model=model2, optimizer=opt2)
        ts2.load_state_dict(sd)

        # Optimizer state should be restored
        assert len(opt2.state) > 0

    def test_load_state_dict_restores_lr_scheduler(self):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

        # Step scheduler to change its state
        sched.step()
        sched.step()

        ts = Checkpoint(
            model=model, optimizer=opt, lr_scheduler=sched
        )
        sd = ts.state_dict()

        model2 = _SimpleModel()
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.5)

        ts2 = Checkpoint(
            model=model2, optimizer=opt2, lr_scheduler=sched2
        )
        ts2.load_state_dict(sd)

        assert sched2.last_epoch == sched.last_epoch

    def test_ddp_unwrap(self):
        """state_dict unwraps model.module if present (simulated)."""
        model = _SimpleModel()

        class FakeDDP(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

        wrapped = FakeDDP(model)
        opt = torch.optim.SGD(wrapped.parameters(), lr=0.01)
        ts = Checkpoint(model=wrapped, optimizer=opt)
        sd = ts.state_dict()

        assert "linear.weight" in sd["model_state_dict"]

    def test_roundtrip_via_backend(self, tmp_path):
        """Full save→load roundtrip through TorchSaveBackend."""
        backend = TorchSaveBackend()
        model = _SimpleModel()
        opt = torch.optim.Adam(model.parameters())

        ts = Checkpoint(
            model=model,
            optimizer=opt,
            epoch=7,
            global_step=210,
            best_metric=0.123,
        )
        sd = ts.state_dict()
        backend.save(sd, tmp_path / "ckpt.pt")

        # Load into fresh objects
        model2 = _SimpleModel()
        opt2 = torch.optim.Adam(model2.parameters())
        ts2 = Checkpoint(model=model2, optimizer=opt2)
        loaded_sd = backend.load(tmp_path / "ckpt.pt")
        ts2.load_state_dict(loaded_sd)

        assert ts2.epoch == 7
        assert ts2.global_step == 210
        assert ts2.best_metric == pytest.approx(0.123)
        assert torch.equal(
            model2.linear.weight, model.linear.weight
        )


# ---------------------------------------------------------------------------
# TrainState best_metric property
# ---------------------------------------------------------------------------


class TestTrainStateBestMetric:
    def test_best_metric_default_none(self):
        state = TrainState()
        assert state.best_metric is None

    def test_best_metric_setter(self):
        state = TrainState()
        state.best_metric = 0.42
        assert state.best_metric == pytest.approx(0.42)
        assert state["best_metric"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


class TestTrainerNewParams:
    def test_default_no_scheduler_no_scaler(self):
        trainer = _make_trainer()
        assert trainer.lr_scheduler is None
        assert trainer.scaler is None
        assert trainer._checkpoint.model is trainer.model
        assert trainer._checkpoint.optimizer is trainer.optimizer

    def test_lr_scheduler_factory(self):
        trainer = _make_trainer(
            lr_scheduler_factory=lambda opt: torch.optim.lr_scheduler.StepLR(
                opt, step_size=10
            ),
        )
        assert trainer.lr_scheduler is not None
        assert trainer._checkpoint.lr_scheduler is trainer.lr_scheduler

    def test_use_amp(self):
        trainer = _make_trainer()
        trainer.set_precision("fp16-mixed")
        assert trainer.scaler is not None
        assert trainer._checkpoint.scaler is trainer.scaler

    def test_resume_path_stored(self):
        trainer = _make_trainer(resume_from_checkpoint="/tmp/test.pt")
        assert trainer._resume_from_checkpoint == "/tmp/test.pt"

    def test_resolve_checkpoint_path_missing(self):
        result = Trainer._resolve_checkpoint_path("/nonexistent/path.pt")
        assert result is None

    def test_resolve_checkpoint_path_exists(self, tmp_path):
        ckpt = tmp_path / "test.pt"
        ckpt.touch()
        result = Trainer._resolve_checkpoint_path(str(ckpt))
        assert result == ckpt

    def test_resolve_auto_no_env(self, monkeypatch):
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
        result = Trainer._resolve_checkpoint_path("auto")
        assert result is None


# ---------------------------------------------------------------------------
# Trainer resume integration
# ---------------------------------------------------------------------------


class TestTrainerResume:
    def test_resume_continues_from_checkpoint(self, tmp_path):
        """Train 2 epochs, save, resume, train 2 more."""
        dm = _MockDataModule(batches_per_epoch=3)

        # First training run
        trainer1 = _make_trainer()
        state1 = trainer1.train(dm, max_epochs=2)
        assert state1.epoch == 2
        assert state1.global_step == 6

        # Save checkpoint
        ckpt_path = tmp_path / "ckpt.pt"
        backend = TorchSaveBackend()
        trainer1._checkpoint.epoch = state1.epoch
        trainer1._checkpoint.global_step = state1.global_step
        backend.save(trainer1._checkpoint.state_dict(), ckpt_path)

        # Resume training
        trainer2 = _make_trainer(resume_from_checkpoint=str(ckpt_path))
        state2 = trainer2.train(dm, max_epochs=4)  # total 4 epochs

        assert state2.epoch == 4
        assert state2.global_step == 12  # 6 from before + 6 new

    def test_resume_from_nonexistent_starts_fresh(self, tmp_path):
        """Resume from missing file starts from epoch 0."""
        dm = _MockDataModule(batches_per_epoch=2)
        trainer = _make_trainer(
            resume_from_checkpoint=str(tmp_path / "missing.pt")
        )
        state = trainer.train(dm, max_epochs=1)
        assert state.epoch == 1
        assert state.global_step == 2


# ---------------------------------------------------------------------------
# CheckpointHook integration
# ---------------------------------------------------------------------------


class TestCheckpointHookIntegration:
    def test_saves_complete_state_dict(self, tmp_path):
        """Checkpoint contains all state components."""
        ckpt_dir = str(tmp_path / "ckpts")
        hook = CheckpointHook(
            checkpoint_dir=ckpt_dir,
            save_every_n_epochs=1,
            save_last=True,
        )
        trainer = _make_trainer(hooks=[hook])
        trainer.train(_MockDataModule(batches_per_epoch=2), max_epochs=1)

        # Check last.pt
        backend = TorchSaveBackend()
        sd = backend.load(Path(ckpt_dir) / "last.pt")
        assert "model_state_dict" in sd
        assert "optimizer_state_dict" in sd
        assert "rng_states" in sd
        assert sd["epoch"] == 1
        assert sd["global_step"] == 2

    def test_save_best(self, tmp_path):
        """Best checkpoint saved when metric improves."""
        ckpt_dir = str(tmp_path / "ckpts")
        hook = CheckpointHook(
            checkpoint_dir=ckpt_dir,
            save_every_n_epochs=100,  # disable periodic saves
            save_last=False,
            save_best=True,
            best_metric_name="eval/loss",
            best_metric_mode="min",
        )

        trainer = _make_trainer(hooks=[hook])

        # Manually set eval/loss to simulate metric tracking
        # (MetricsHook or eval step normally does this)
        class FakeMetricHook:
            """Writes a decreasing eval/loss each epoch."""

            def __init__(self):
                self._epoch = 0

            def on_epoch_end(self, trainer, state):
                state["eval/loss"] = 1.0 - self._epoch * 0.3
                self._epoch += 1

        trainer.hooks.insert(0, FakeMetricHook())  # run before checkpoint hook
        trainer.train(_MockDataModule(batches_per_epoch=2), max_epochs=3)

        best_path = Path(ckpt_dir) / "best.pt"
        assert best_path.exists()

        sd = TorchSaveBackend().load(best_path)
        # Best should be the lowest loss (last epoch: 1.0 - 0.6 = 0.4)
        assert sd["best_metric"] == pytest.approx(0.4)

    def test_lr_scheduler_in_checkpoint(self, tmp_path):
        """Checkpoint includes lr_scheduler state."""
        ckpt_dir = str(tmp_path / "ckpts")
        hook = CheckpointHook(
            checkpoint_dir=ckpt_dir,
            save_last=True,
        )
        trainer = _make_trainer(
            lr_scheduler_factory=lambda opt: torch.optim.lr_scheduler.StepLR(
                opt, step_size=1
            ),
            hooks=[hook],
        )
        trainer.train(_MockDataModule(batches_per_epoch=2), max_epochs=1)

        sd = TorchSaveBackend().load(Path(ckpt_dir) / "last.pt")
        assert "lr_scheduler_state_dict" in sd


# ---------------------------------------------------------------------------
# DefaultTrainStep with lr_scheduler
# ---------------------------------------------------------------------------


class TestDefaultTrainStepScheduler:
    def test_lr_scheduler_steps_per_batch(self):
        """lr_scheduler.step() is called once per training batch."""
        trainer = _make_trainer(
            lr_scheduler_factory=lambda opt: torch.optim.lr_scheduler.StepLR(
                opt, step_size=1, gamma=0.5
            ),
        )
        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        dm = _MockDataModule(batches_per_epoch=3)
        trainer.train(dm, max_epochs=1)

        # After 3 batches with StepLR(step_size=1, gamma=0.5):
        # lr = 0.01 * 0.5^3 = 0.00125
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr
        assert final_lr == pytest.approx(initial_lr * 0.5**3)
