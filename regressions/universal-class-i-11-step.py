"""Public-API regression for MultiTaskTrainStep (per-batch task routing).

Spec: ``universal-class-i-11-step`` (acceptance ac-013).

One minimal scenario, public API only: a fixed-weight tiny model is trained for
**exactly one optimizer window** (``accumulate_grad_batches=2``, two hand-built
single-task batches) through :meth:`molix.core.Trainer.train` with
``loss_fn=None`` — the step owns its losses. Batch 1 carries the
``energy_force`` label (code 0), batch 2 the ``charge`` label (code 1);
:class:`~molix.core.steps.TaskRouter` dispatches each to its own
:class:`~molix.core.losses.ScaleNormalizedLoss`, and the routed scalars land in
``state["train"]["loss_energy_force"]`` / ``state["train"]["loss_charge"]``.
:class:`~molix.hooks.GradClipHook` owns the clipping and publishes the pre-clip
gradient norm; :class:`~molix.hooks.Log` is registered with the per-task keys, so
this run also exercises the step-advertised ``scalar_keys`` acceptance in
``Log._validate_keys``. Everything runs in float64. No third-party oracle is
imported, subprocessed, or consulted — here or at capture time: every golden
below is closed-form arithmetic on the literals in this file.

Model (a stub, not a physical potential — chosen so both tasks are exactly
hand-computable while sharing **one** parameter set)::

    s_i = w . r_i + b          per-atom site scalar,  w = (1, 2, 3), b = 0.5
    E_g = sum_{i in g} s_i     graph energy      (so F_i = -dE/dr_i = -w exactly)
    F_i = -w                   per-atom force
    q_i = s_i                  per-atom charge

**Batch 1 — ``energy_force``, code 0** (2 graphs x 2 atoms)::

    atom  r_i          s_i = w.r_i + b     graph
    0     (1, 0, 0)    1.5                 0
    1     (0, 1, 0)    2.5                 0
    2     (0, 0, 1)    3.5                 1
    3     (1, 1, 1)    6.5                 1

    E = (4.0, 10.0), reference (3.0, 12.0)  -> residuals (1.0, -2.0)
    MSE_E = (1 + 4) / 2 = 2.5

    F_pred = (-1, -2, -3) for every atom; the reference forces differ in one
    component per atom (see ``_FORCE_REF``), giving residuals
    (0,0,0), (-1,0,0), (0,-1,0), (0,0,-2)
    MSE_F = (0 + 1 + 1 + 4) / 12 = 0.5

    energy_force_mse(lambda_F=1) = 2.5 + 0.5 = 3.0
    ScaleNormalizedLoss(scale = 2.0 kcal/mol) -> 3.0 / 2.0**2 = **0.75**

**Batch 2 — ``charge``, code 1** (2 graphs x 2 atoms)::

    atom  r_i          q_i = w.r_i + b   q_ref    residual
    0     (0, 0, 0)    0.5               0.0       0.5
    1     (1, 0, 0)    1.5               2.0      -0.5
    2     (0, 1, 0)    2.5               1.5       1.0
    3     (0, 0, 1)    3.5               3.5       0.0

    ChargeMSELoss = (0.25 + 0.25 + 1.0 + 0.0) / 4 = 0.375
    ScaleNormalizedLoss(scale = 0.5 e) -> 0.375 / 0.5**2 = **1.5**

``state["train"]["loss"]`` is rewritten by every micro-batch, so after the
window it holds the **second** micro-batch's routed loss, ``1.5`` — the routed
loss really is the one that was back-propagated.

**Pre-clip gradient norm** (``GradClipHook``, fired once at the window end, on
the gradient accumulated over both micro-batches, each scaled by ``1/2``)::

    d(MSE_E)/dw = 1.0 * (r0 + r1) + (-2.0) * (r2 + r3) = (-1, -1, -4)
    d(MSE_E)/db = 1.0 * 2 + (-2.0) * 2                 = -2
    d(MSE_F)/dw = -(1/6) * sum_i residual_i = -(1/6) * (-1, -1, -2)
                = (1/6, 1/6, 1/3)          d(MSE_F)/db = 0
    dL_ef/dw = ((-1,-1,-4) + (1/6,1/6,1/3)) / 4 = (-5/24, -5/24, -11/12)
    dL_ef/db = -2 / 4 = -1/2

    dL_q/dw = 4 * (1/2) * sum_i c_i r_i = 4 * (-1/4, 1/2, 0) = (-1, 2, 0)
    dL_q/db = 4 * (1/2) * sum_i c_i     = 4 * 1/2            = 2

    grad_w = (dL_ef/dw + dL_q/dw) / 2 = (-29/48, 43/48, -11/24)
    grad_b = (-1/2 + 2) / 2           = 3/4
    ||grad|| = sqrt(841/2304 + 1849/2304 + 484/2304 + 1296/2304)
             = sqrt(745/384) = 1.392876220870565

``max_norm=1.0`` is below that norm, so the clip is genuinely active — the
recorded scalar is the **pre-clip** norm by contract.

    capture: none — every golden above is closed-form arithmetic on the
             literals in this file; nothing was captured from a run.
    verify:  PYTHONPATH=src python regressions/universal-class-i-11-step.py
    commit:  27e1839 (+ working tree implementing universal-class-i-11-step)
    torch:   2.12.1+cpu
    device:  cpu, float64 (atol 1e-12 on every scalar)
    date:    2026-08-14
"""

from __future__ import annotations

import io
import sys
from collections.abc import Callable, Iterable, Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict

from molix import logging as molix_logging
from molix.config import config
from molix.core.losses import ChargeMSELoss, ScaleNormalizedLoss, energy_force_mse
from molix.core.steps import MultiTaskTrainStep, TaskRouter, TaskSpec
from molix.core.trainer import Trainer
from molix.hooks import GradClipHook, Log

Predictions = dict[str, torch.Tensor]
TaskLoss = Callable[[Predictions, TensorDict], torch.Tensor]

# ---------------------------------------------------------------------------
# Fixture tables (see module docstring)
# ---------------------------------------------------------------------------

#: Site-head weight ``w`` — hard-set, never randomly initialized.
_WEIGHT: list[list[float]] = [[1.0, 2.0, 3.0]]
#: Site-head bias ``b`` — hard-set.
_BIAS: list[float] = [0.5]

#: Batch 1 positions, Å; graphs {0: atoms 0-1, 1: atoms 2-3}.
_EF_POSITIONS: list[list[float]] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
]
#: Batch 1 reference energies, kcal/mol (model gives 4.0 / 10.0).
_ENERGY_REF: list[float] = [3.0, 12.0]
#: Batch 1 reference forces, kcal mol⁻¹ Å⁻¹ (model gives ``-w`` for every atom).
_FORCE_REF: list[list[float]] = [
    [-1.0, -2.0, -3.0],
    [0.0, -2.0, -3.0],
    [-1.0, -1.0, -3.0],
    [-1.0, -2.0, -1.0],
]
#: Batch 2 positions, Å; graphs {0: atoms 0-1, 1: atoms 2-3}.
_CHARGE_POSITIONS: list[list[float]] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]
#: Batch 2 reference MBIS-style charges, e (model gives 0.5 / 1.5 / 2.5 / 3.5).
_CHARGE_REF: list[float] = [0.0, 2.0, 1.5, 3.5]
#: Per-atom graph membership shared by both batches.
_ATOM_GRAPH: list[int] = [0, 0, 1, 1]

#: Task codes as they ride at ``batch["graphs", "task"]``.
_CODE_ENERGY_FORCE = 0
_CODE_CHARGE = 1
#: Fixed physical denominators (ForceBalance discipline; never dataset-scanned).
_SCALE_ENERGY_FORCE = 2.0  # kcal/mol
_SCALE_CHARGE = 0.5  # e
#: Optimizer window: two micro-batches, one optimizer step.
_ACCUMULATE_GRAD_BATCHES = 2
#: Gradient-clip ceiling — below the analytic norm, so the clip really fires.
_MAX_GRAD_NORM = 1.0
#: SGD learning rate (the parameter update itself is not a golden here).
_LEARNING_RATE = 0.1

# ---------------------------------------------------------------------------
# Hard-coded goldens (closed-form, derived in the module docstring)
# ---------------------------------------------------------------------------

#: ``ScaleNormalizedLoss(energy_force_mse) = (2.5 + 0.5) / 2.0**2`` (hard-coded golden).
_GOLDEN_LOSS_ENERGY_FORCE = 0.75
#: ``ScaleNormalizedLoss(ChargeMSELoss) = 0.375 / 0.5**2`` (hard-coded golden).
_GOLDEN_LOSS_CHARGE = 1.5
#: ``state["train"]["loss"]`` after the window = the *last* micro-batch's routed
#: loss, i.e. the charge task (hard-coded golden).
_GOLDEN_LOSS = 1.5
#: Pre-clip ``||grad||`` over the accumulated window, ``sqrt(745/384)``
#: (hard-coded golden).
_GOLDEN_GRAD_NORM = 1.392876220870565
#: One optimizer step for two micro-batches (hard-coded golden).
_GOLDEN_GLOBAL_STEP = 1
#: Absolute tolerance for every float64 scalar above.
_ATOL = 1e-12


class TinyClassIModel(nn.Module):
    """Fixed-weight stub: one site head feeding energy, forces and charges.

    ``s_i = w · r_i + b``; the graph energy is the segment sum of ``s_i`` so the
    analytic force is exactly ``F_i = -∂E/∂r_i = -w``, and the charge head reuses
    the same site scalar. One shared parameter set therefore receives gradient
    from **both** tasks of the mixture, which is the property this regression
    exercises. The weights are hard-set (never randomly initialized), so the run
    is RNG-free.
    """

    def __init__(self) -> None:
        super().__init__()
        self.site = nn.Linear(3, 1, dtype=torch.float64)
        with torch.no_grad():
            self.site.weight.copy_(torch.tensor(_WEIGHT, dtype=torch.float64))
            self.site.bias.copy_(torch.tensor(_BIAS, dtype=torch.float64))

    def forward(self, batch: TensorDict) -> Predictions:
        """Predict ``energy`` ``(B,)``, ``forces`` ``(N, 3)`` and ``q`` ``(N,)``.

        Args:
            batch: Collated batch with ``("atoms", "pos")``, ``("atoms", "batch")``
                and ``("graphs", "num_atoms")``.

        Returns:
            Prediction dict handed verbatim to the routed task loss.
        """
        pos = batch["atoms", "pos"]
        graph_of_atom = batch["atoms", "batch"]
        n_graphs = batch["graphs", "num_atoms"].shape[0]
        site = self.site(pos).squeeze(-1)
        energy = torch.zeros(n_graphs, dtype=pos.dtype).index_add(0, graph_of_atom, site)
        forces = -self.site.weight.expand(pos.shape[0], 3)
        return {"energy": energy, "forces": forces, "q": site}


class TwoBatchDataModule:
    """Duck-typed datamodule: two hand-built single-task train batches.

    The validation loader is deliberately empty — this scenario asserts one
    training window, and the Trainer's epoch-end eval phase must survive an
    empty stream.
    """

    batch_size = 2

    def __init__(self, batches: list[TensorDict]) -> None:
        self.batches = batches

    def train_dataloader(self) -> Iterable[TensorDict]:
        """Yield the two micro-batches of the single optimizer window."""
        return list(self.batches)

    def val_dataloader(self) -> Iterable[TensorDict]:
        """No validation batches in this scenario."""
        return []


def _energy_force_batch() -> TensorDict:
    """Batch 1: the ``energy_force`` task (code 0), 2 graphs x 2 atoms."""
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": torch.tensor(_EF_POSITIONS, dtype=torch.float64),
                    "forces": torch.tensor(_FORCE_REF, dtype=torch.float64),
                    "batch": torch.tensor(_ATOM_GRAPH, dtype=torch.int64),
                },
                batch_size=[len(_EF_POSITIONS)],
            ),
            "graphs": TensorDict(
                {
                    "task": torch.full((2,), _CODE_ENERGY_FORCE, dtype=torch.int64),
                    "energy": torch.tensor(_ENERGY_REF, dtype=torch.float64),
                    "num_atoms": torch.tensor([2, 2], dtype=torch.int64),
                },
                batch_size=[2],
            ),
        },
        batch_size=[],
    )


def _charge_batch() -> TensorDict:
    """Batch 2: the ``charge`` task (code 1), 2 graphs x 2 atoms."""
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": torch.tensor(_CHARGE_POSITIONS, dtype=torch.float64),
                    "q": torch.tensor(_CHARGE_REF, dtype=torch.float64),
                    "batch": torch.tensor(_ATOM_GRAPH, dtype=torch.int64),
                },
                batch_size=[len(_CHARGE_POSITIONS)],
            ),
            "graphs": TensorDict(
                {
                    "task": torch.full((2,), _CODE_CHARGE, dtype=torch.int64),
                    "num_atoms": torch.tensor([2, 2], dtype=torch.int64),
                },
                batch_size=[2],
            ),
        },
        batch_size=[],
    )


def _charge_task_loss() -> TaskLoss:
    """``ChargeMSELoss`` on ``e``, normalized by its **own** fixed denominator.

    Per-task key adaptation is the caller's job (the router hands predictions and
    batch to the loss verbatim), so the atom-level reference charges are lifted
    out of the batch here. The charge task keeps a separate reference method
    (PBE0/def2-TZVPD MBIS) and a separate denominator from the energetics tasks.
    """
    normalized = ScaleNormalizedLoss(ChargeMSELoss(), scale=_SCALE_CHARGE)

    def _fn(predictions: Predictions, batch: TensorDict) -> torch.Tensor:
        return normalized(predictions, {"q": batch["atoms", "q"]})

    return _fn


def _build_trainer(step: MultiTaskTrainStep) -> Trainer:
    """Wire the fixed-weight model, ``loss_fn=None`` and the two hooks."""
    return Trainer(
        model=TinyClassIModel(),
        loss_fn=None,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LEARNING_RATE),
        train_step=step,
        hooks=[
            GradClipHook(max_norm=_MAX_GRAD_NORM),
            Log(
                every_n_steps=1,
                keys=[
                    ("train", "loss"),
                    ("train", "loss_energy_force"),
                    ("train", "loss_charge"),
                    ("train", "grad_norm"),
                ],
            ),
        ],
        accumulate_grad_batches=_ACCUMULATE_GRAD_BATCHES,
    )


def _check_scalar(state: Mapping[str, object], path: tuple[str, str], golden: float) -> None:
    """Assertion helper: ``state[ns][name]`` is a detached tensor equal to *golden*."""
    namespace = state[path[0]]
    if not isinstance(namespace, Mapping) or path[1] not in namespace:
        raise AssertionError(f"state{list(path)} was never written; got {namespace!r}")
    value = namespace[path[1]]
    if not isinstance(value, torch.Tensor):
        raise AssertionError(f"state{list(path)} must stay a tensor, got {type(value).__name__}")
    if value.requires_grad or value.grad_fn is not None:
        raise AssertionError(f"state{list(path)} must be detached; got {value!r}")
    delta = abs(float(value) - golden)
    if delta > _ATOL:
        raise AssertionError(
            f"state{list(path)} drift: {float(value)!r} vs {golden!r} (|d| = {delta})"
        )


def _check_log_columns(captured: str) -> None:
    """Assertion: ``Log`` accepted and rendered the step-advertised per-task keys."""
    for column in ("loss_energy_force", "loss_charge", "grad_norm"):
        if column not in captured:
            raise AssertionError(
                f"Log never rendered the {column!r} column; captured: {captured!r}"
            )


def main() -> int:
    config.set_precision("fp64")

    step = MultiTaskTrainStep(
        TaskRouter(
            (
                TaskSpec(
                    name="energy_force",
                    code=_CODE_ENERGY_FORCE,
                    loss=ScaleNormalizedLoss(energy_force_mse(), scale=_SCALE_ENERGY_FORCE),
                ),
                TaskSpec(name="charge", code=_CODE_CHARGE, loss=_charge_task_loss()),
            )
        )
    )
    trainer = _build_trainer(step)
    datamodule = TwoBatchDataModule([_energy_force_batch(), _charge_batch()])

    # Route the banner + thermo table into a buffer so the run stays quiet and
    # the Log columns can be asserted on.
    captured = io.StringIO()
    molix_logging.basicConfig(level="INFO", stream=captured)
    try:
        state = trainer.train(datamodule, max_steps=1)
    finally:
        molix_logging.shutdown()

    if state["global_step"] != _GOLDEN_GLOBAL_STEP:
        raise AssertionError(
            f"global_step drift: {state['global_step']} vs {_GOLDEN_GLOBAL_STEP} "
            f"(two micro-batches must close exactly one optimizer window)"
        )
    if "loss_hessian" in state["train"] or "loss_torsion" in state["train"]:
        raise AssertionError(f"an undeclared task scalar was published: {sorted(state['train'])}")
    _check_scalar(state, ("train", "loss_energy_force"), _GOLDEN_LOSS_ENERGY_FORCE)
    _check_scalar(state, ("train", "loss_charge"), _GOLDEN_LOSS_CHARGE)
    _check_scalar(state, ("train", "loss"), _GOLDEN_LOSS)
    _check_scalar(state, ("train", "grad_norm"), _GOLDEN_GRAD_NORM)
    _check_log_columns(captured.getvalue())

    print(
        "universal-class-i-11-step: one optimizer window (accum=2, fp64) -> "
        "train/loss_energy_force 0.75, train/loss_charge 1.5, train/loss 1.5 "
        "(last routed micro-batch), pre-clip train/grad_norm sqrt(745/384)="
        "1.392876220870565, global_step 1, Log accepted the step-advertised "
        "per-task keys OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
