"""Unit tests for :mod:`molix.core.audit`.

Mirrors ``src/molix/core/audit.py``. Two units under test:
:class:`TaskGradientProbe` (per-task, per-group gradient-norm measurement)
and :class:`GradientBalanceReport` (the dimensionless ratio table).

Analytic fixture (float64, closed form — no third-party oracle)
--------------------------------------------------------------
Three ``nn.Linear(1, 1, bias=False)`` children share one trunk::

    w_t = 2.0 (trunk), w_a = 1.0 (head_a), w_b = 1.0 (head_b), x = [[1.0]]
    h = w_t * x = 2.0        a = w_a * h = 2.0        b = w_b * h = 2.0
    L_a = a**2 = 4.0         L_b = 100 * b**2 = 400.0

with ``g_{t,G} = ||dL_t/dtheta_G||_2``:

    dL_a/dw_t = 2*a*w_a*x   =   4.0     dL_b/dw_t = 200*b*w_b*x = 400.0
    dL_a/dw_a = 2*a*h       =   8.0     dL_b/dw_a = 0.0
    dL_a/dw_b = 0.0                     dL_b/dw_b = 200*b*h     = 800.0

so ``I_trunk = 400/4 = 100.0`` (shared) while both heads are task-specific
(``min_t g = 0`` -> ``None``). All goldens are hand-derived and asserted at
``atol=1e-9``; the same table appears in the spec
``.claude/specs/universal-class-i-13-grad-audit.md`` §"Testing strategy".
"""

from __future__ import annotations

import ast
import inspect
import json
from typing import Mapping

import pytest
import torch
import torch.nn as nn

import molix.core.audit as audit_module
from molix.core.audit import GradientBalanceReport, TaskGradientProbe, TaskGradientSnapshot
from molix.core.checkpoint.rng import capture_rng_states

#: Absolute tolerance for the analytic goldens (exact closed-form values).
TOL = 1e-9


# ---------------------------------------------------------------------------
# Analytic fixture
# ---------------------------------------------------------------------------


class _TwoTaskModel(nn.Module):
    """Shared trunk feeding two single-weight task heads (float64)."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.head_a = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.head_b = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.trunk.weight.fill_(2.0)
            self.head_a.weight.fill_(1.0)
            self.head_b.weight.fill_(1.0)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return ``{"a": head_a(trunk(x)), "b": head_b(trunk(x))}``."""
        hidden = self.trunk(batch["x"])
        return {"a": self.head_a(hidden), "b": self.head_b(hidden)}


class _RngConsumingTwoTaskModel(_TwoTaskModel):
    """Same fixture, but its forward advances the global torch RNG.

    The jitter is multiplied by zero so predictions stay exactly analytic;
    only the RNG stream moves, which is what the restoration test needs.
    """

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the analytic predictions after drawing one random number."""
        jitter = torch.randn((), dtype=torch.float64) * 0.0
        return {name: value + jitter for name, value in super().forward(batch).items()}


def _loss_a(
    predictions: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """``L_a = a**2`` — golden 4.0 at ``a = 2.0``."""
    del batch
    return (predictions["a"] ** 2).sum()


def _loss_b(
    predictions: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """``L_b = 100 * b**2`` — golden 400.0 at ``b = 2.0``."""
    del batch
    return 100.0 * (predictions["b"] ** 2).sum()


def _loss_nan(
    predictions: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """A loss that is non-finite while still attached to the graph."""
    del batch
    return (predictions["a"] * float("nan")).sum()


@pytest.fixture
def model() -> _TwoTaskModel:
    """The analytic shared-trunk two-task model."""
    return _TwoTaskModel()


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    """The single-element float64 input batch ``x = [[1.0]]``."""
    return {"x": torch.tensor([[1.0]], dtype=torch.float64)}


@pytest.fixture
def probe(model: _TwoTaskModel) -> TaskGradientProbe:
    """Probe over both analytic task losses with default groups and units."""
    return TaskGradientProbe(model, {"a": _loss_a, "b": _loss_b})


@pytest.fixture
def snapshot(probe: TaskGradientProbe, batch: dict[str, torch.Tensor]) -> TaskGradientSnapshot:
    """One probe point with exactly one batch per task."""
    return probe.run([("a", batch), ("b", batch)], label="step_0", at_step=0)


@pytest.fixture
def report(snapshot: TaskGradientSnapshot) -> GradientBalanceReport:
    """Report carrying the single analytic snapshot."""
    built = GradientBalanceReport()
    built.add(snapshot)
    return built


def _module_tree() -> ast.Module:
    """Parse the audit module source for the boundary source scans."""
    return ast.parse(inspect.getsource(audit_module))


def _imported_modules(tree: ast.Module) -> set[str]:
    """Collect every module name imported by ``tree``."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


class TestTaskGradientProbe:
    """Per-task, per-group gradient-norm measurement."""

    def test_run_records_analytic_task_losses(self, snapshot: TaskGradientSnapshot) -> None:
        """Mean per-task loss matches the closed form (4.0 / 400.0)."""
        assert snapshot.loss["a"] == pytest.approx(4.0, abs=TOL)
        assert snapshot.loss["b"] == pytest.approx(400.0, abs=TOL)

    def test_run_records_analytic_per_group_grad_norms(
        self, snapshot: TaskGradientSnapshot
    ) -> None:
        """``g_{t,G}`` matches the hand-derived table for all six cells."""
        assert snapshot.grad_norm["a"]["trunk"] == pytest.approx(4.0, abs=TOL)
        assert snapshot.grad_norm["a"]["head_a"] == pytest.approx(8.0, abs=TOL)
        assert snapshot.grad_norm["a"]["head_b"] == pytest.approx(0.0, abs=TOL)
        assert snapshot.grad_norm["b"]["trunk"] == pytest.approx(400.0, abs=TOL)
        assert snapshot.grad_norm["b"]["head_a"] == pytest.approx(0.0, abs=TOL)
        assert snapshot.grad_norm["b"]["head_b"] == pytest.approx(800.0, abs=TOL)

    def test_run_records_batch_counts_per_task(
        self, probe: TaskGradientProbe, batch: dict[str, torch.Tensor]
    ) -> None:
        """``n_batches`` counts the pairs consumed for each task."""
        taken = probe.run([("a", batch), ("a", batch), ("b", batch)], label="step_0", at_step=0)
        assert taken.n_batches == {"a": 2, "b": 1}

    def test_run_averages_loss_over_repeated_batches(
        self, probe: TaskGradientProbe, batch: dict[str, torch.Tensor]
    ) -> None:
        """Repeating an identical batch leaves the mean loss unchanged."""
        taken = probe.run([("a", batch), ("a", batch), ("b", batch)], label="step_0", at_step=0)
        assert taken.loss["a"] == pytest.approx(4.0, abs=TOL)

    def test_run_defaults_unit_labels_to_dimensionless(
        self, snapshot: TaskGradientSnapshot
    ) -> None:
        """Without an explicit mapping every task is labelled ``"1"``."""
        assert snapshot.units == {"a": "1", "b": "1"}

    def test_run_labels_the_probe_point(
        self, probe: TaskGradientProbe, batch: dict[str, torch.Tensor]
    ) -> None:
        """``label`` / ``at_step`` are carried onto the snapshot verbatim."""
        taken = probe.run([("a", batch), ("b", batch)], label="step_100", at_step=100)
        assert (taken.label, taken.at_step) == ("step_100", 100)

    def test_run_restores_parameter_state_dict(
        self,
        model: _TwoTaskModel,
        probe: TaskGradientProbe,
        batch: dict[str, torch.Tensor],
    ) -> None:
        """No parameter byte moves: the probe never steps an optimizer."""
        before = {name: value.clone() for name, value in model.state_dict().items()}
        probe.run([("a", batch), ("b", batch)], label="step_0", at_step=0)
        after = model.state_dict()
        assert set(after) == set(before)
        assert all(torch.equal(after[name], value) for name, value in before.items())

    def test_run_restores_pre_seeded_gradient_buffer(
        self,
        model: _TwoTaskModel,
        probe: TaskGradientProbe,
        batch: dict[str, torch.Tensor],
    ) -> None:
        """A ``.grad`` present on entry comes back byte-identical."""
        model.trunk.weight.grad = torch.full_like(model.trunk.weight, 3.0)
        expected = model.trunk.weight.grad.clone()
        probe.run([("a", batch), ("b", batch)], label="step_0", at_step=0)
        restored = model.trunk.weight.grad
        assert restored is not None
        assert torch.equal(restored, expected)

    def test_run_leaves_absent_gradient_buffer_none(
        self,
        model: _TwoTaskModel,
        probe: TaskGradientProbe,
        batch: dict[str, torch.Tensor],
    ) -> None:
        """A ``.grad`` that was ``None`` on entry is ``None`` on exit."""
        assert model.head_b.weight.grad is None
        probe.run([("a", batch), ("b", batch)], label="step_0", at_step=0)
        assert model.head_b.weight.grad is None

    def test_run_restores_training_mode(
        self,
        model: _TwoTaskModel,
        probe: TaskGradientProbe,
        batch: dict[str, torch.Tensor],
    ) -> None:
        """``model.training`` is the caller's, not the probe's."""
        model.train()
        probe.run([("a", batch), ("b", batch)], label="step_0", at_step=0)
        assert model.training is True

    def test_run_restores_torch_rng_state(self, batch: dict[str, torch.Tensor]) -> None:
        """RNG consumed inside the forward is rewound on exit."""
        rng_model = _RngConsumingTwoTaskModel()
        rng_probe = TaskGradientProbe(rng_model, {"a": _loss_a, "b": _loss_b})
        torch.manual_seed(1234)
        before = capture_rng_states()["torch"]
        rng_probe.run([("a", batch), ("b", batch)], label="step_0", at_step=0)
        after = capture_rng_states()["torch"]
        assert torch.equal(after, before)

    def test_run_rejects_task_with_no_batch(
        self, model: _TwoTaskModel, batch: dict[str, torch.Tensor]
    ) -> None:
        """A never-sampled task would be a fabricated zero row -> refuse."""
        starved = TaskGradientProbe(model, {"a": _loss_a, "unsampled": _loss_b})
        with pytest.raises(ValueError, match="unsampled"):
            starved.run([("a", batch)], label="step_0", at_step=0)

    def test_run_rejects_unknown_task_label(
        self, probe: TaskGradientProbe, batch: dict[str, torch.Tensor]
    ) -> None:
        """A stream label with no loss callable is a caller bug -> refuse."""
        with pytest.raises(ValueError, match="ghost_task"):
            probe.run(
                [("a", batch), ("b", batch), ("ghost_task", batch)],
                label="step_0",
                at_step=0,
            )

    def test_run_rejects_non_finite_loss(
        self, model: _TwoTaskModel, batch: dict[str, torch.Tensor]
    ) -> None:
        """A NaN loss names the offending task instead of being recorded."""
        broken = TaskGradientProbe(model, {"nan_task": _loss_nan})
        with pytest.raises(ValueError, match="nan_task"):
            broken.run([("nan_task", batch)], label="step_0", at_step=0)

    def test_snapshot_to_dict_is_plain_json(self, snapshot: TaskGradientSnapshot) -> None:
        """``TaskGradientSnapshot.to_dict`` survives a JSON round-trip."""
        payload = snapshot.to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_module_never_constructs_or_steps_an_optimizer(self) -> None:
        """Source scan: no optimizer identifier and no ``.step()`` call."""
        tree = _module_tree()
        identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        identifiers |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        identifiers |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
        assert not [name for name in identifiers if "optimizer" in name.lower()]
        stepped = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
        ]
        assert not stepped


class TestGradientBalanceReport:
    """Dimensionless ratio table over one or more probe points."""

    def test_ratios_normalize_to_per_group_maximum(self, report: GradientBalanceReport) -> None:
        """``r_{t,trunk}`` is ``{a: 0.01, b: 1.0}`` for the shared group."""
        trunk = report.ratios()["step_0"]["trunk"]
        assert trunk["a"] == pytest.approx(0.01, abs=TOL)
        assert trunk["b"] == pytest.approx(1.0, abs=TOL)

    def test_ratios_put_each_task_specific_head_at_unity(
        self, report: GradientBalanceReport
    ) -> None:
        """The owning task of a task-specific head normalizes to 1.0."""
        ratios = report.ratios()["step_0"]
        assert ratios["head_a"]["a"] == pytest.approx(1.0, abs=TOL)
        assert ratios["head_b"]["b"] == pytest.approx(1.0, abs=TOL)

    def test_imbalance_reports_shared_group_ratio(self, report: GradientBalanceReport) -> None:
        """``I_trunk = max_t g / min_t g = 400/4 = 100.0``."""
        assert report.imbalance()["step_0"]["trunk"] == pytest.approx(100.0, abs=TOL)

    def test_imbalance_is_none_for_task_specific_groups(
        self, report: GradientBalanceReport
    ) -> None:
        """A head only one task touches is not an imbalance -> ``None``."""
        imbalance = report.imbalance()["step_0"]
        assert imbalance["head_a"] is None
        assert imbalance["head_b"] is None

    def test_balanced_is_false_below_the_measured_imbalance(
        self, report: GradientBalanceReport
    ) -> None:
        """One decade is not enough for ``I_trunk = 100``."""
        assert report.balanced(max_ratio=10.0) is False

    def test_balanced_is_true_above_the_measured_imbalance(
        self, report: GradientBalanceReport
    ) -> None:
        """Three decades admit ``I_trunk = 100``."""
        assert report.balanced(max_ratio=1000.0) is True

    def test_balanced_rejects_single_task_report(self) -> None:
        """One task admits no comparison — refusing beats laundering a GO."""
        lonely = GradientBalanceReport()
        lonely.add(
            TaskGradientSnapshot(
                label="step_0",
                at_step=0,
                loss={"a": 4.0},
                grad_norm={"a": {"trunk": 4.0}},
                n_batches={"a": 1},
                units={"a": "1"},
            )
        )
        with pytest.raises(ValueError):
            lonely.balanced(max_ratio=10.0)

    def test_balanced_rejects_report_without_shared_group(self) -> None:
        """All-task-specific groups leave nothing to compare."""
        disjoint = GradientBalanceReport()
        disjoint.add(
            TaskGradientSnapshot(
                label="step_0",
                at_step=0,
                loss={"a": 4.0, "b": 400.0},
                grad_norm={
                    "a": {"head_a": 8.0, "head_b": 0.0},
                    "b": {"head_a": 0.0, "head_b": 800.0},
                },
                n_batches={"a": 1, "b": 1},
                units={"a": "1", "b": "1"},
            )
        )
        with pytest.raises(ValueError):
            disjoint.balanced(max_ratio=10.0)

    def test_balanced_rejects_mixed_unit_labels(self) -> None:
        """``g_{t,G}`` carries ``[L_t]/[theta_G]``: unlike units are not comparable."""
        mixed = GradientBalanceReport()
        mixed.add(
            TaskGradientSnapshot(
                label="step_0",
                at_step=0,
                loss={"a": 4.0, "b": 400.0},
                grad_norm={"a": {"trunk": 4.0}, "b": {"trunk": 400.0}},
                n_batches={"a": 1, "b": 1},
                units={"a": "1", "b": "kcal**2/mol**2"},
            )
        )
        with pytest.raises(ValueError):
            mixed.balanced(max_ratio=10.0)

    def test_add_rejects_duplicate_label(
        self, report: GradientBalanceReport, snapshot: TaskGradientSnapshot
    ) -> None:
        """Two snapshots may not share a probe-point label."""
        with pytest.raises(ValueError, match="step_0"):
            report.add(snapshot)

    def test_snapshots_preserve_insertion_order(
        self, report: GradientBalanceReport, snapshot: TaskGradientSnapshot
    ) -> None:
        """``snapshots`` exposes the probe points in the order added."""
        report.add(
            TaskGradientSnapshot(
                label="step_100",
                at_step=100,
                loss=dict(snapshot.loss),
                grad_norm={task: dict(groups) for task, groups in snapshot.grad_norm.items()},
                n_batches=dict(snapshot.n_batches),
                units=dict(snapshot.units),
            )
        )
        assert [entry.label for entry in report.snapshots] == ["step_0", "step_100"]

    def test_to_dict_carries_schema_and_sections(self, report: GradientBalanceReport) -> None:
        """Evidence dict is self-describing: schema + snapshots + tables."""
        payload = report.to_dict()
        assert payload["schema"] == "molix.grad_balance/1"
        assert set(payload) >= {"schema", "snapshots", "ratios", "imbalance"}
        first = payload["snapshots"][0]
        assert set(first) >= {"label", "at_step", "loss", "grad_norm", "n_batches", "units"}

    def test_to_dict_round_trips_through_json(
        self, report: GradientBalanceReport, snapshot: TaskGradientSnapshot
    ) -> None:
        """A two-snapshot report is plain JSON, ``None`` imbalances included."""
        report.add(
            TaskGradientSnapshot(
                label="step_100",
                at_step=100,
                loss=dict(snapshot.loss),
                grad_norm={task: dict(groups) for task, groups in snapshot.grad_norm.items()},
                n_batches=dict(snapshot.n_batches),
                units=dict(snapshot.units),
            )
        )
        payload = report.to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_module_imports_no_io_or_experiment_tracking(self) -> None:
        """Source scan: the primitive never writes evidence to disk itself."""
        tree = _module_tree()
        modules = _imported_modules(tree)
        forbidden = ("molexp", "molix.io", "pathlib")
        assert not [
            name
            for name in modules
            if any(name == root or name.startswith(f"{root}.") for root in forbidden)
        ]
        writes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"dump", "save", "write_text", "write_bytes"}
                )
                or (isinstance(node.func, ast.Name) and node.func.id == "open")
            )
        ]
        assert not writes
