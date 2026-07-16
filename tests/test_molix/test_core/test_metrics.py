import pytest
import torch

from molix.core.metrics import MAE, MSE, RMSE, Accuracy, MetricCollection, R2Score


class TestMAE:
    def test_zero_error(self):
        metric = MAE()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        metric.update(preds, targets)
        assert float(metric.compute()) == pytest.approx(0.0)

    def test_compute_returns_zero_dim_tensor(self):
        """compute() returns a 0-d tensor (torchmetrics contract), not a float —
        so the training hot path never .item()-syncs."""
        metric = MAE()
        metric.update(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 2.5]))
        out = metric.compute()
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0
        assert float(out) == pytest.approx(0.5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compute_stays_on_device(self):
        """compute() keeps the result on the inputs' device (no implicit copy)."""
        metric = MAE()
        metric.update(
            torch.tensor([1.0, 2.0], device="cuda"),
            torch.tensor([1.5, 2.5], device="cuda"),
        )
        assert metric.compute().is_cuda

    def test_reset(self):
        metric = MAE()
        metric.update(torch.tensor([1.0]), torch.tensor([2.0]))
        metric.reset()
        assert metric.preds == []
        assert metric.targets == []


class TestRMSE:
    def test_known_error(self):
        metric = RMSE()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 3.0, 5.0])
        metric.update(preds, targets)
        assert float(metric.compute()) == pytest.approx((5.0 / 3.0) ** 0.5, abs=1e-3)


class TestMSE:
    def test_known_error(self):
        metric = MSE()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 3.0, 5.0])
        metric.update(preds, targets)
        assert float(metric.compute()) == pytest.approx(5.0 / 3.0, abs=1e-3)


class TestR2Score:
    def test_perfect_predictions(self):
        metric = R2Score()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        metric.update(preds, targets)
        assert float(metric.compute()) == pytest.approx(1.0)


class TestAccuracy:
    def test_partial_accuracy(self):
        metric = Accuracy()
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 1, 1])
        metric.update(preds, targets)
        assert float(metric.compute()) == pytest.approx(0.5)


class TestMetricCollection:
    def test_collection_compute(self):
        metrics = MetricCollection([MAE(), RMSE()])
        preds = torch.tensor([1.0, 2.0])
        targets = torch.tensor([1.5, 2.5])
        metrics.update(preds, targets)
        results = metrics.compute()
        assert set(results.keys()) == {"MAE", "RMSE"}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_collection_gpu_inputs(self):
        metrics = MetricCollection([MAE()])
        preds = torch.tensor([1.0, 2.0], device="cuda")
        targets = torch.tensor([1.5, 2.5], device="cuda")
        metrics.update(preds, targets)
        results = metrics.compute()
        assert float(results["MAE"]) == pytest.approx(0.5)
