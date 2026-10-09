"""Multi-task train/eval steps with per-batch task routing.

One shared model is trained against several heterogeneous QM tasks whose
losses are not comparable term-by-term and whose reference levels of theory
differ. Each collated batch carries **exactly one** task label as a
graph-level target (``batch["graphs", "task"]``); :class:`TaskRouter` reads
that label and dispatches the batch to the one :class:`TaskSpec` that declares
it. :class:`MultiTaskTrainStep` / :class:`MultiTaskEvalStep` inherit
:class:`~molix.core.steps.train.DefaultTrainStep` /
:class:`~molix.core.steps.eval.DefaultEvalStep` and override *only* their
``compute_loss`` seam, so AMP autocast, gradient accumulation,
``optimizer_applied`` scheduler-desync detection and eval grad-context
semantics are reused rather than forked.

Losses arrive **pre-constructed** from the caller, so this module stays
task-agnostic and ``molix`` gains no ``molpot`` dependency. Gradient clipping
remains owned by :class:`~molix.hooks.training.GradClipHook`, per-task metrics
by ``MetricsHook(name_prefix=…)``.

Units:
    Per-task state scalars (``state[<namespace>]["loss_<name>"]``) are
    **dimensionless**: every task's loss is normalized by the caller's fixed
    physical denominator before it reaches this module. The underlying
    residual units are kcal mol⁻¹ and kcal mol⁻¹ Å⁻¹ (energy + forces),
    kcal mol⁻¹ Å⁻² / kcal mol⁻¹ rad⁻² (projected internal Hessian),
    kcal mol⁻¹ (torsion profile) and e (MBIS charges). This module performs
    no unit conversion and imports nothing from :mod:`molix.units`, so no
    conversion float is restated here.

v0 science constraints (recorded here because this module is the single place
where the task mixture becomes operational):

1. **LJ is frozen.** σ/ε are excluded from optimization in v0: gas-phase
   intramolecular energies, forces, Hessians and torsion scans do not
   determine them. espaloma-0.3 inherits openff-2.0.0 vdW and ByteFF pins
   GAFF-2.2 vdW for the same reason. No task routed here supplies a vdW
   gradient signal.
2. **The charge task keeps its own reference method and its own denominator.**
   PBE0/def2-TZVPD MBIS charges and B3LYP-D3(BJ)/DZVP energetics must never
   share one metadata field or one normalization denominator. :class:`TaskSpec`
   therefore has exactly three fields and this module exposes no shared
   scale / normalization / reference-method surface: each task owns a fully
   constructed loss whose denominator was supplied by the caller.
3. **Normalization is fixed physical denominators, not learned or
   dataset-scanned weights** (ForceBalance discipline, Qiu et al. §2.1.7).
   After normalization espaloma-0.3 uses unit weights across tasks (SI
   §C.2.3). Because a batch is single-task, any residual per-task weight is
   expressible as a denominator rescale, so no weight vector is exposed and
   two competing weighting surfaces cannot drift apart.
4. **Energy centering is per group (scan / conformer set), exactly.** It is
   applied inside the caller's loss; the router never re-partitions a batch,
   so groups are never mixed.
5. **Declared v0 biases.** (a) MM energies / forces / Hessians are evaluated
   at the QM constrained-torsion geometries **without MM reoptimization**, so
   torsion parameters absorb relaxation error; (b) EEM charges remain
   topology-only (no geometry dependence). Both are recorded, not corrected.

Reference:
    Takaba et al. "Machine-learned molecular mechanics force fields from
    large-scale quantum chemical data" (espaloma-0.3), SI §C.2.3.
    https://arxiv.org/abs/2307.07085

    Zheng et al. "Data-Driven Parametrization of Molecular Mechanics Force
    Fields for Expansive Chemical Space Coverage" (ByteFF).
    https://arxiv.org/abs/2408.12817

    Qiu et al. J. Chem. Theory Comput. 17, 6262 (2021), §2.1.7 —
    ForceBalance objective normalization.
    https://doi.org/10.1021/acs.jctc.1c00571

    Verstraelen et al. J. Chem. Theory Comput. 12, 3894 (2016) — MBIS.
    https://doi.org/10.1021/acs.jctc.6b00456
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from molix.core.steps.base import _mark_leaves
from molix.core.steps.eval import DefaultEvalStep
from molix.core.steps.train import DefaultTrainStep

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


@dataclass(frozen=True)
class TaskSpec:
    """One task of the mixture: its name, its integer label, and its loss.

    Exactly three fields. There is deliberately no ``weight`` / ``scale`` /
    ``reference_method`` field: normalization is a fixed physical denominator
    baked into ``loss`` by the caller, and each task's QM reference level
    travels with the data that produced it (module docstring §2/§3).

    Args:
        name: State-scalar suffix — the routed loss lands in
            ``state[<namespace>]["loss_<name>"]``. Must be non-empty.
        code: Integer label as it rides in ``batch[task_key]``.
        loss: Fully constructed, dimensionless-output loss called as
            ``loss(predictions, batch)``.
    """

    name: str
    code: int
    loss: Callable[[Mapping[str, torch.Tensor], Any], torch.Tensor]


class TaskRouter:
    """Resolves the single task label of a batch to its :class:`TaskSpec`.

    Validation is eager: the declaration is checked at construction and the
    batch label at every :meth:`route`, both with actionable ``ValueError``
    messages, because a silently mis-routed batch trains the wrong objective.

    Args:
        tasks: The declared tasks, in the order they should be advertised.
            Must be non-empty with unique, non-empty ``name``s and unique
            ``code``s.
        task_key: Graph-namespace path the label rides on. Named ``task``
            (never ``task_id``) so it can never be confused with the pipeline
            cache-identity ``task_id``.

    Raises:
        ValueError: Empty ``tasks``, an empty task name, a duplicate name or a
            duplicate code.
    """

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        *,
        task_key: tuple[str, str] = ("graphs", "task"),
    ) -> None:
        declared = tuple(tasks)
        if not declared:
            raise ValueError("TaskRouter needs at least one TaskSpec; got an empty task sequence.")
        by_name: dict[str, TaskSpec] = {}
        by_code: dict[int, TaskSpec] = {}
        for spec in declared:
            if not spec.name:
                raise ValueError(
                    f"TaskSpec.name must be non-empty (code {spec.code} declares an empty name)."
                )
            if spec.name in by_name:
                raise ValueError(
                    f"duplicate TaskSpec name {spec.name!r}; task names must be unique "
                    "(they become the state-scalar suffix loss_<name>)."
                )
            if spec.code in by_code:
                raise ValueError(
                    f"duplicate TaskSpec code {spec.code!r} for tasks "
                    f"{by_code[spec.code].name!r} and {spec.name!r}; codes must be unique."
                )
            by_name[spec.name] = spec
            by_code[spec.code] = spec
        self.tasks = declared
        self.task_key = task_key
        self._by_code = by_code

    def route(self, batch: Any) -> TaskSpec:
        """Return the task this batch belongs to.

        Reads nothing but the label: ``predictions`` and ``batch`` are later
        handed to the routed loss verbatim, so per-task key adaptation stays in
        the caller's closure. The label is resolved with two ``Mapping.get``
        steps so a nested ``dict`` and a ``TensorDict`` behave identically and
        a missing label never surfaces as a bare ``KeyError``.

        ``torch.unique(labels).tolist()`` is the **one and only host sync** this
        step performs per batch. It stays in this method, which the steps call
        from the eager prologue of ``forward_loss``, outside the region
        ``Trainer.compile`` traces. The traced body calls the already chosen
        loss and does not read the label.

        Args:
            batch: The collated batch (nested ``TensorDict`` or plain nested
                mapping) carrying the label at ``task_key``.

        Returns:
            The declared :class:`TaskSpec` whose ``code`` the batch carries.

        Raises:
            ValueError: The label is missing, empty, of floating dtype, carries
                more than one code (mixed-task batch), or carries an undeclared
                code.
        """
        namespace_key, label_key = self.task_key
        namespace = batch.get(namespace_key, None)
        labels = None if namespace is None else namespace.get(label_key, None)
        if labels is None:
            raise ValueError(
                f"batch is missing its task label at {self.task_key!r} "
                f"(batch[{namespace_key!r}][{label_key!r}]); declare it as a graph-level "
                "target in the frame's meta so the collate routes it into the graphs namespace."
            )
        if labels.numel() == 0:
            raise ValueError(
                f"task label at {self.task_key!r} is empty; every batch carries exactly "
                "one task code, one entry per graph."
            )
        if labels.is_floating_point():
            raise ValueError(
                f"task label at {self.task_key!r} must be an integer tensor, got dtype "
                f"{labels.dtype}; a floating label would truncate silently and route the "
                "batch to the wrong task."
            )
        codes = torch.unique(labels).tolist()
        if len(codes) != 1:
            declared = {code: spec.name for code, spec in self._by_code.items()}
            raise ValueError(
                f"mixed-task batch at {self.task_key!r}: found codes {codes} "
                f"(declared tasks {declared}). One batch carries exactly one task — "
                "the sampler must emit single-task batches."
            )
        code = int(codes[0])
        spec = self._by_code.get(code)
        if spec is None:
            declared = {c: s.name for c, s in self._by_code.items()}
            raise ValueError(
                f"undeclared task code {code} at {self.task_key!r}; declared tasks are "
                f"{declared}. Add a TaskSpec for it or fix the label."
            )
        return spec


class MultiTaskTrainStep(DefaultTrainStep):
    """Training step that routes each batch to one task's loss.

    Inherits ``on_train_batch`` unchanged — AMP, the accumulation window,
    ``optimizer_applied`` and the ``on_after_backward`` fire point are the
    default step's, byte for byte — and overrides only
    :meth:`compute_loss`. Gradient clipping is never performed here;
    :class:`~molix.hooks.training.GradClipHook` keeps sole ownership.

    A ``WeightedLoss`` evaluates *every* term, so it may only appear **inside**
    one :attr:`TaskSpec.loss`; cross-task selection is this router's job.

    The label read (:meth:`TaskRouter.route`) is the named graph break and
    stays in the eager prologue. ``_traced_forward_loss`` is the compiled
    region: the model and the already chosen loss. Dynamo specializes that
    region once per task. A loss that itself does not trace
    (``TorsionProfileLoss``, ``center_by_group``) still will not.

    Args:
        router: The :class:`TaskRouter` holding the declared task mixture.
    """

    namespace = "train"

    def __init__(self, router: TaskRouter) -> None:
        self.router = router
        self._routed: TaskSpec | None = None

    @property
    def scalar_keys(self) -> tuple[tuple[str, str], ...]:
        """State paths this step writes — ``("train", "loss_<name>")`` per task.

        Mirrors the ``ScalarHook.scalar_keys`` convention so ``Log``,
        ``JournalHook`` and ``TensorBoardHook`` accept per-task columns with no
        new hook type.
        """
        return tuple((self.namespace, f"loss_{spec.name}") for spec in self.router.tasks)

    def compute_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
        predictions: Any,
    ) -> torch.Tensor:
        """Route the batch and evaluate that task's loss.

        The label was resolved in :meth:`forward_loss` before this runs, so
        the traced body does not host-sync. The per-task state write also
        stays in that prologue.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`
                (``loss_fn`` may be ``None`` — this step owns its losses).
            state: The :class:`~molix.core.state.TrainState` to record into.
                Unused here; the prologue publishes the scalar.
            batch: The collated batch carrying the task label.
            predictions: The model output for ``batch``, handed to the routed
                loss verbatim.

        Returns:
            The dimensionless scalar loss of the routed task.
        """
        del trainer, state
        spec = self._routed if self._routed is not None else self.router.route(batch)
        return spec.loss(predictions, batch)

    def forward_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Resolve the task, then run the traced body.

        :meth:`TaskRouter.route` is the host sync and sits outside the
        region :meth:`~molix.core.trainer.Trainer.compile` replaces.

        Args:
            trainer: The owning trainer.
            state: Training state. The routed loss is written here, detached.
            batch: The collated batch carrying the task label.

        Returns:
            ``(loss, predictions)`` as :meth:`DefaultTrainStep.forward_loss`.
        """
        self._routed = self.router.route(batch)
        _mark_leaves(trainer.model, batch)
        loss, predictions = self._traced_forward_loss(trainer, state, batch)
        state[self.namespace][f"loss_{self._routed.name}"] = loss.detach()
        return loss, predictions

    def _traced_forward_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Model plus the already chosen loss. No label read, no state write."""
        predictions = trainer.model(batch)
        return self.compute_loss(trainer, state, batch, predictions), predictions


class MultiTaskEvalStep(DefaultEvalStep):
    """Evaluation step that routes each batch to one task's loss.

    Inherits ``on_eval_batch`` unchanged. The label read is the eager
    prologue, as on :class:`MultiTaskTrainStep`, so ``compile(step=True)``
    traces the model and the chosen loss.

    Args:
        router: The :class:`TaskRouter` holding the declared task mixture.
        no_grad: Forwarded to :class:`~molix.core.steps.eval.DefaultEvalStep`.
            Must stay ``False`` (the default) for force / Hessian tasks — their
            losses need a live graph.
    """

    namespace = "eval"

    def __init__(self, router: TaskRouter, *, no_grad: bool = False) -> None:
        super().__init__(no_grad=no_grad)
        self.router = router
        self._routed: TaskSpec | None = None

    @property
    def scalar_keys(self) -> tuple[tuple[str, str], ...]:
        """State paths this step writes — ``("eval", "loss_<name>")`` per task."""
        return tuple((self.namespace, f"loss_{spec.name}") for spec in self.router.tasks)

    def compute_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
        predictions: Any,
    ) -> torch.Tensor:
        """Route the batch and evaluate that task's loss.

        The label was resolved in :meth:`forward_loss`. The returned loss
        keeps its graph so a force / Hessian task can still differentiate.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`
                (``loss_fn`` may be ``None`` — this step owns its losses).
            state: Unused; the prologue publishes the scalar.
            batch: The collated batch carrying the task label.
            predictions: The model output for ``batch``, handed to the routed
                loss verbatim.

        Returns:
            The dimensionless scalar loss of the routed task.
        """
        del trainer, state
        spec = self._routed if self._routed is not None else self.router.route(batch)
        return spec.loss(predictions, batch)

    def forward_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Resolve the task, then run the traced eval body."""
        self._routed = self.router.route(batch)
        _mark_leaves(trainer.model, batch)
        loss, predictions = self._traced_forward_loss(trainer, state, batch)
        state[self.namespace][f"loss_{self._routed.name}"] = loss.detach()
        return loss, predictions

    def _traced_forward_loss(
        self,
        trainer: "Trainer",
        state: "TrainState",
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Model plus the already chosen loss. No label read, no state write."""
        predictions = trainer.model(batch)
        return self.compute_loss(trainer, state, batch, predictions), predictions
