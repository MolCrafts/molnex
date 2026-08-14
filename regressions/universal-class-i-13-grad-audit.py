"""Public-API regression for the per-task gradient-balance audit primitive.

Spec: ``universal-class-i-13-grad-audit`` (acceptance ac-012).

One minimal scenario, public API only: an analytic two-task model with a shared
trunk is probed once by :class:`~molix.core.audit.TaskGradientProbe`, the single
probe point is folded into a :class:`~molix.core.audit.GradientBalanceReport`,
and the ratio table, the shared-group imbalance, the unit-weight verdict at two
thresholds and the JSON evidence dict are checked against hard-coded goldens.
Everything runs on CPU in float64. No third-party oracle is imported,
subprocessed, or consulted — here or at capture time: every golden below is
closed-form differentiation of the literals in this file. Only ``torch`` and
``molix`` are imported.

Domain basis — Takaba et al., "Machine-learned molecular mechanics force fields
from large-scale quantum chemical data", arXiv:2307.07085 (espaloma-0.3 sets all
task weights to 1, SI §C.2.3), with the dimensional precondition from Qiu et
al., *J. Chem. Theory Comput.* **17**, 6262-6280 (2021),
https://doi.org/10.1021/acs.jctc.1c00571 §2.1.7 (ForceBalance normalization
makes every task loss dimensionless, so ``g_{t,G}`` is comparable across tasks).
For task ``t`` and parameter group ``G``::

    g_{t,G} = ||d L_t / d theta_G||_2
    r_{t,G} = g_{t,G} / max_{t'} g_{t',G}
    I_G     = max_t g_{t,G} / min_t g_{t,G}   (None when min_t g_{t,G} = 0)

    fixed unit weights are justified  <=>  I_G <= max_ratio for every *shared* G

**Analytic fixture — three ``nn.Linear(1, 1, bias=False)`` children, one shared
trunk** (the same fixture as ``tests/test_molix/test_core/test_audit.py``,
re-typed locally so this file stands alone)::

    w_t = 2.0 (trunk),  w_a = 1.0 (head_a),  w_b = 1.0 (head_b),  x = [[1.0]]

    h = w_t . x   = 2.0
    a = w_a . h   = 2.0            b = w_b . h   = 2.0
    L_a = a^2     = 4.0            L_b = 100 b^2 = 400.0

    d L_a / d w_t = 2 a w_a x     = 4.0
    d L_a / d w_a = 2 a h         = 8.0
    d L_a / d w_b = 0.0            (head_b is not on task a's graph)
    d L_b / d w_t = 200 b w_b x   = 400.0
    d L_b / d w_a = 0.0            (head_a is not on task b's graph)
    d L_b / d w_b = 200 b h       = 800.0

Each group holds exactly one scalar weight, so ``g_{t,G} = |d L_t / d w_G|``:

    group    g_{a,G}   g_{b,G}   r_{a,G}   r_{b,G}   I_G
    trunk        4.0     400.0      0.01       1.0   400/4 = 100.0  (shared)
    head_a       8.0       0.0       1.0       0.0   None (task-specific)
    head_b       0.0     800.0       0.0       1.0   None (task-specific)

so ``balanced(max_ratio=10.0)`` is **False** (one decade cannot admit
``I_trunk = 100``) while ``balanced(max_ratio=1000.0)`` is **True**. Both heads
are task-specific — a head receiving gradient from one task only is correct, not
an imbalance — so they are excluded from the verdict.

    capture: none — closed-form; every golden above is analytic differentiation
             of the literals in this file. Nothing was captured from a run and no
             oracle software of any kind is involved.
    verify:  PYTHONPATH=src python regressions/universal-class-i-13-grad-audit.py
    commit:  f8a3759 (+ working tree implementing universal-class-i-13-grad-audit)
    torch:   2.12.1+cpu
    device:  cpu, float64 (atol 1e-9 on every loss and gradient norm; the
             task-specific ``None`` imbalances and the JSON round-trip are exact)
    date:    2026-08-14
    note:    prose uses em dashes (—) like the sibling regression headers, while
             every formula stays ASCII (``d L / d w``, ``||.||_2``, ``->``) so the
             derivation can be pasted into a shell or a review comment unchanged.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping

import torch
from torch import nn

from molix.config import config
from molix.core.audit import GradientBalanceReport, TaskGradientProbe

#: One batch as the probe stream and the model exchange it.
Batch = Mapping[str, torch.Tensor]

# ---------------------------------------------------------------------------
# Fixture literals (see module docstring); all dimensionless
# ---------------------------------------------------------------------------

#: Shared-trunk weight ``w_t``.
_W_TRUNK = 2.0
#: Task-head weights ``w_a`` / ``w_b``.
_W_HEAD_A = 1.0
_W_HEAD_B = 1.0
#: The single-element input ``x = [[1.0]]``.
_X: list[list[float]] = [[1.0]]
#: Loss scale of task ``b``: ``L_b = 100 b^2`` against ``L_a = a^2``.
_LOSS_B_SCALE = 100.0
#: Probe-point label and step for the one snapshot taken here.
_LABEL = "step_0"
_AT_STEP = 0

# ---------------------------------------------------------------------------
# Hard-coded goldens (closed-form, derived in the module docstring)
# ---------------------------------------------------------------------------

#: Per-task losses ``L_a`` / ``L_b`` at the initial weights (hard-coded golden).
_GOLDEN_LOSS: dict[str, float] = {"a": 4.0, "b": 400.0}
#: Per-task, per-group gradient norms ``g_{t,G}`` (hard-coded golden).
_GOLDEN_GRAD_NORM: dict[str, dict[str, float]] = {
    "a": {"trunk": 4.0, "head_a": 8.0, "head_b": 0.0},
    "b": {"trunk": 400.0, "head_a": 0.0, "head_b": 800.0},
}
#: Shared-group imbalance ``I_trunk = 400/4`` (hard-coded golden).
_GOLDEN_IMBALANCE_TRUNK = 100.0
#: Task-specific groups report no imbalance (hard-coded golden).
_TASK_SPECIFIC_GROUPS: tuple[str, ...] = ("head_a", "head_b")
#: Verdict thresholds bracketing ``I_trunk``: one decade rejects, three admit.
_MAX_RATIO_REJECT = 10.0
_MAX_RATIO_ACCEPT = 1000.0
#: Evidence-dict schema tag (hard-coded golden).
_GOLDEN_SCHEMA = "molix.grad_balance/1"
#: Absolute tolerance for every float64 scalar.
_ATOL = 1e-9


class _TwoTaskModel(nn.Module):
    """Shared trunk feeding two single-weight task heads (float64)."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.head_a = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.head_b = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.trunk.weight.fill_(_W_TRUNK)
            self.head_a.weight.fill_(_W_HEAD_A)
            self.head_b.weight.fill_(_W_HEAD_B)

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Return ``{"a": head_a(trunk(x)), "b": head_b(trunk(x))}``."""
        hidden = self.trunk(batch["x"])
        return {"a": self.head_a(hidden), "b": self.head_b(hidden)}


def _loss_a(predictions: Mapping[str, torch.Tensor], batch: Batch) -> torch.Tensor:
    """``L_a = a^2`` — golden 4.0 at ``a = 2.0``."""
    del batch
    return (predictions["a"] ** 2).sum()


def _loss_b(predictions: Mapping[str, torch.Tensor], batch: Batch) -> torch.Tensor:
    """``L_b = 100 b^2`` — golden 400.0 at ``b = 2.0``."""
    del batch
    return _LOSS_B_SCALE * (predictions["b"] ** 2).sum()


def _check(name: str, got: float, golden: float) -> None:
    """Assertion helper: *got* is a float within ``_ATOL`` of *golden*."""
    delta = abs(got - golden)
    if delta > _ATOL:
        raise AssertionError(f"{name} drift: {got!r} vs {golden!r} (|d| = {delta})")


def main() -> int:
    config.set_precision("fp64")

    # 1. One probe point over the analytic fixture: one batch per task, so the
    #    recorded means are the closed-form values themselves.
    model = _TwoTaskModel()
    batch: dict[str, torch.Tensor] = {"x": torch.tensor(_X, dtype=torch.float64)}
    probe = TaskGradientProbe(model, {"a": _loss_a, "b": _loss_b})
    snapshot = probe.run([("a", batch), ("b", batch)], label=_LABEL, at_step=_AT_STEP)

    # 2. Per-task losses L_a = 4, L_b = 400.
    for task, golden in _GOLDEN_LOSS.items():
        _check(f"loss[{task}]", snapshot.loss[task], golden)

    # 3. Per-task, per-group gradient norms — the full six-cell table, including
    #    the two exact zeros that make each head task-specific.
    for task, groups in _GOLDEN_GRAD_NORM.items():
        for group, golden in groups.items():
            _check(f"grad_norm[{task}][{group}]", snapshot.grad_norm[task][group], golden)

    # 4. Ratio table, imbalance and the unit-weight verdict at two thresholds.
    report = GradientBalanceReport()
    report.add(snapshot)
    imbalance = report.imbalance()[_LABEL]
    trunk = imbalance["trunk"]
    if trunk is None:
        raise AssertionError("imbalance[trunk] is None — the shared trunk lost a task's gradient")
    _check("imbalance[trunk]", trunk, _GOLDEN_IMBALANCE_TRUNK)
    for group in _TASK_SPECIFIC_GROUPS:
        if imbalance[group] is not None:
            raise AssertionError(
                f"imbalance[{group}] is {imbalance[group]!r}, expected None — a head only one "
                "task touches is task-specific, not an imbalance"
            )
    rejected = report.balanced(max_ratio=_MAX_RATIO_REJECT)
    accepted = report.balanced(max_ratio=_MAX_RATIO_ACCEPT)
    if rejected is not False:
        raise AssertionError(
            f"balanced(max_ratio={_MAX_RATIO_REJECT}) returned {rejected!r}, expected False — "
            f"I_trunk = {_GOLDEN_IMBALANCE_TRUNK} exceeds one decade"
        )
    if accepted is not True:
        raise AssertionError(
            f"balanced(max_ratio={_MAX_RATIO_ACCEPT}) returned {accepted!r}, expected True — "
            f"I_trunk = {_GOLDEN_IMBALANCE_TRUNK} is inside three decades"
        )

    # 5. Evidence hand-off: the dict the caller gives molexp is plain JSON.
    payload = report.to_dict()
    if payload["schema"] != _GOLDEN_SCHEMA:
        raise AssertionError(f"schema drift: {payload['schema']!r} vs {_GOLDEN_SCHEMA!r}")
    if json.loads(json.dumps(payload)) != payload:
        raise AssertionError("report.to_dict() did not survive a JSON round-trip")

    print(
        "universal-class-i-13-grad-audit: per-task gradient balance (cpu, fp64) -> "
        "loss a=4.0 b=400.0, grad norms trunk 4.0/400.0, head_a 8.0/0.0, head_b 0.0/800.0, "
        "I_trunk=100.0 with both heads task-specific (None), balanced(10.0) False, "
        f"balanced(1000.0) True, to_dict() JSON round-trips under schema {_GOLDEN_SCHEMA} OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
