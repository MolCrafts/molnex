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


class TestMoleculeCenteredRMSE:
    def test_self_zero(self):
        from molix.core.metrics import MoleculeCenteredRMSE

        m = MoleculeCenteredRMSE()
        # values {0,0.25,1.0} same for pred/true → centered RMSE 0
        e = torch.tensor([0.0, 0.25, 1.0])
        g = torch.tensor([0, 0, 0])
        m.update(e, e, g)
        assert float(m.compute()) == pytest.approx(0.0)


class TestMoleculeCenteredMAE:
    def test_offsets(self):
        from molix.core.metrics import MoleculeCenteredMAE

        m = MoleculeCenteredMAE()
        pred = torch.tensor([1.0, 2.0, 5.0, 7.0])
        true = torch.tensor([10.0, 11.0, 0.0, 2.0])
        g = torch.tensor([0, 0, 1, 1])
        m.update(pred, true, g)
        assert float(m.compute()) == pytest.approx(0.0)


class TestPotentialReport:
    """Force, energy-per-atom, virial and outlier sums, ignoring a loss weight."""

    def test_per_element_force_and_energy_per_atom(self):
        from molix.core.metrics import PotentialReport

        report = PotentialReport(elements=(1, 8), outlier_force=1.0)
        pred = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        ref = torch.zeros(3, 3)
        report.update_forces(pred, ref, z=torch.tensor([1, 1, 8]))
        report.update_outliers(pred, ref)
        values = report.scalars()
        assert float(values["force_mae"]) == pytest.approx(5.0 / 9.0)
        assert float(values["force_mae_Z1"]) == pytest.approx(2.0 / 6.0)
        assert float(values["force_mae_Z8"]) == pytest.approx(1.0)
        assert float(values["force_outlier_count"]) == pytest.approx(1.0)
        assert float(values["force_outlier_fraction"]) == pytest.approx(1.0 / 3.0)

        report.update_energy(
            torch.tensor([4.0, 9.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([2.0, 3.0]),
            mask=torch.tensor([True, False]),
        )
        assert float(report.scalars()["energy_per_atom_mae"]) == pytest.approx(2.0)

    def test_padding_is_dropped_and_a_loss_weight_is_not_an_input(self):
        from molix.core.metrics import PotentialReport

        report = PotentialReport(outlier_force=1.0)
        pred = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        ref = torch.zeros(2, 3)
        report.update_forces(pred, ref, mask=torch.tensor([True, False]))
        report.update_outliers(pred, ref, mask=torch.tensor([True, False]))
        values = report.scalars()
        assert float(values["force_mae"]) == pytest.approx(0.0)
        assert float(values["force_outlier_count"]) == pytest.approx(0.0)

    def test_virial_mae_uses_live_graphs(self):
        from molix.core.metrics import PotentialReport

        report = PotentialReport()
        pred = torch.ones(2, 3, 3)
        ref = torch.zeros(2, 3, 3)
        report.update_virial(pred, ref, mask=torch.tensor([True, False]))
        assert float(report.scalars()["virial_mae"]) == pytest.approx(1.0)

    def test_hook_reads_predictions_and_not_the_loss_weight(self):
        from tensordict import TensorDict

        from molix.core.state import TrainState
        from molix.hooks import PotentialReportHook

        atoms = TensorDict(
            {
                "Z": torch.tensor([1, 1]),
                "forces": torch.zeros(2, 3),
                "mask": torch.tensor([True, True]),
                "force_weight": torch.zeros(2),
            },
            batch_size=[2],
        )
        graphs = TensorDict(
            {"energy": torch.tensor([0.0]), "num_atoms": torch.tensor([2.0])},
            batch_size=[1],
        )
        batch = TensorDict({"atoms": atoms, "graphs": graphs}, batch_size=[])
        predictions = {
            "forces": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "energy": torch.tensor([4.0]),
        }
        state = TrainState()
        PotentialReportHook(elements=(1,), outlier_force=1.0).on_train_batch_end(
            None, state, batch, {"predictions": predictions}
        )
        assert float(state["train"]["force_mae"]) == pytest.approx(1.0 / 6.0)
        assert float(state["train"]["energy_per_atom_mae"]) == pytest.approx(2.0)
        assert float(state["train"]["force_outlier_count"]) == pytest.approx(0.0)
