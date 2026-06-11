"""Tests for :class:`molix.profiler.TrainerProfiler` and :class:`MockModel`."""

from __future__ import annotations

import torch

from molix.profiler import MockBatch, MockModel, TrainerProfiler, mock_node_feature_loss


class TestMockModel:
    """MockModel honours the encoder protocol with ~zero compute."""

    def test_writes_node_features_and_returns_batch(self):
        batch = MockBatch(n_atoms=16, n_edges=32, n_graphs=2, device="cpu")()
        out = MockModel(n_features=4)(batch)
        assert out is batch  # encoder mutates and returns the same td
        nf = out["atoms", "node_features"]
        assert nf.shape == (16, 1, 4)

    def test_loss_is_differentiable(self):
        batch = MockBatch(n_atoms=8, n_edges=16, n_graphs=2, device="cpu")()
        model = MockModel()
        loss = mock_node_feature_loss(model(batch), batch)
        loss.backward()
        assert model.w.grad is not None


class TestTrainerProfiler:
    """The profiler drives a real Trainer over the mock model and attributes time."""

    def test_run_reports_throughput_and_hotspots(self):
        result = TrainerProfiler(device="cpu").run(n_steps=200, n_warmup=10, top=8)
        assert result.n_steps == 200
        assert result.device == "cpu"
        assert result.steps_per_sec > 0
        assert result.wall_ms_per_step > 0
        assert 0 < len(result.hotspots) <= 8
        # hotspot rows carry the per-step attribution columns
        row = result.hotspots[0]
        assert {"func", "self_us", "cum_us", "calls"} <= set(row)

    def test_hotspots_include_trainer_machinery(self):
        result = TrainerProfiler(device="cpu").run(n_steps=200, n_warmup=10, top=20)
        funcs = " ".join(r["func"] for r in result.hotspots)
        # the loop itself and the Step protocol must show up
        assert "_train" in funcs or "on_train_batch" in funcs

    def test_hooks_count_recorded(self):
        from molix.core.hook import BaseHook

        class _Noop(BaseHook):
            def on_train_batch_end(self, trainer, state, batch, outputs):
                pass

        result = TrainerProfiler(device="cpu", hooks=[_Noop(), _Noop()]).run(
            n_steps=100, n_warmup=5, top=5
        )
        assert result.n_hooks == 2

    def test_custom_batch_size_respected(self):
        batch = MockBatch(n_atoms=64, n_edges=256, n_graphs=4, device="cpu")()
        result = TrainerProfiler(device="cpu").run(n_steps=100, n_warmup=5, batch=batch, top=5)
        assert result.steps_per_sec > 0
        assert not torch.cuda.is_available() or result.device == "cpu"
