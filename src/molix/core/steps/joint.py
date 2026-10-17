"""Joint train / eval steps: one padded batch from every source per optimizer step.

ByteFF trains its force field on several datasets at once — one batch of each
per step, one loss per dataset, the weighted sum back-propagated once
(``trainer.py``, ``train_epoch``), and each loss moves only the parameter
tables it names (``set_trainable_terms``). :class:`JointObjective` is one such
dataset's declaration; :class:`JointTrainStep` / :class:`JointEvalStep` run the
schedule over a nested batch keyed by objective name — what
:class:`~molix.data.JointDataModule` yields.

**The model is a protocol, not a class.** molix imports nothing from molzoo,
so the step asks the model for three primitives and one attribute::

    tables = model.tables(batch)                 # {namespace: {parameter: tensor}}
    frame  = model.energy(batch, tables=None)    # writes graphs.energy (B,), returns batch
    frame  = model.energy(batch, tables, styles=("bond:harmonic",))   # these styles only
    blocks = model.partial_hessian(batch, tables=None)   # (P, 9) on hessian_pairs
    model.derivative.wrt                         # the variables, e.g. ("atoms.pos",)

``styles=`` is asked for only by an objective that names some
(:attr:`JointObjective.styles`, :attr:`JointObjective.nonbonded`); a model that
prices every style on every call need not accept it.

**One forward per source.** ByteFF draws one batch per dataset and scores
every loss of that dataset on it (``trainer.py``, ``train_epoch``). Objectives
that name the same :attr:`JointObjective.source` read the same sub-batch, and
the model's ``tables`` — the encoder and every head — run **once** for it;
each objective then keeps its own trainable subset (:meth:`JointTrainStep.keep`,
detached views, no second forward) and prices its own primitive. Four Hessian
objectives on one source cost one encoder pass instead of four, and the
``nonbonded`` force reuses the same tables detached instead of a fifth; the
small tier's step used to hold five encoder graphs for one batch of data and
the large tier could not hold one step at all.

**Where "which loss trains which table" lives.** Between the heads and the
energy: :meth:`JointTrainStep.keep` detaches every predicted table the
objective does not name, and the objective hands the kept tables to
``energy`` / ``partial_hessian`` (``Derivative`` passes them through to the
energy as keyword arguments). It cannot sit in the loss — by then the tables
have become an energy — and the model does not know an objective exists. A
loss sees predictions and labels on the same sub-batch, which is how
:class:`~molix.core.losses.BoltzmannProfileLoss` and friends already read
``pred_key`` columns. An objective that supervises a predicted table itself
(the MBIS charges) declares ``output="tables"``: the kept tables are written
onto the batch and no energy is priced.

**Two uses of a partial energy**, both ByteFF's. An objective with
:attr:`~JointObjective.styles` prices only those — ``Bond_Energy`` /
``Angle_Energy`` (``loss.py:302-314``) are the bond and angle energy on a QM
minimum, the strain ``r0`` / ``θ0`` remove, which the equilibrium force alone
cannot fix: a common shift of every ``θ0`` at a planar or tetrahedral centre
moves no atom (on the force loss alone every sp3 ``θ0`` walked down to 95°).
An objective with :attr:`~JointObjective.nonbonded` also writes
those styles' own force to ``atoms.forces_nonbonded`` — detached, before the
full forward — so a force loss can leave out a molecule whose non-bonded force
says its geometry clashes (``Force_MSE``'s ``skip_large_nb``, ``loss.py:373-380``).

**The compile seam** is :attr:`traced_bodies`: one callable per **source**,
``body(trainer, state, sub_batch) -> (losses, sub_batch)`` with one loss per
objective of that source in declaration order, each body a region of its
own. :meth:`~molix.core.trainer.Trainer.compile` (``step=True``) compiles each
entry in place — K explicit compiled objects, so dynamo never has to
specialise one region on K shapes and the recompile limit is never near.
``forward_loss`` is the eager prologue around them: it reads the sub-batches,
marks the position leaf before a body that differentiates
(:meth:`~molix.autograd.Derivative.mark_leaves`), builds each objective's
derivative op once, and sums the weighted losses — a static Python schedule
with no tensor-valued control flow. ``on_train_batch`` / ``on_eval_batch`` are
inherited unchanged: AMP, the accumulation window and the one ``backward()``
are the default steps'.

Reference:
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics
    force fields for expansive chemical space coverage." *Chem. Sci.* (2025).
    https://arxiv.org/abs/2408.12817
"""

from __future__ import annotations

import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

import torch

from molix.autograd import Derivative
from molix.compile import unwrap
from molix.core.steps.eval import DefaultEvalStep
from molix.core.steps.train import DefaultTrainStep
from molix.frame import beside_labels
from molix.schema import (
    ENERGY_KEY,
    ENERGY_PRED_KEY,
    FORCES_PRED_KEY,
    HESSIAN_KEY,
    NONBONDED_FORCES_KEY,
    POS_KEY,
)

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer

__all__ = ["NONBONDED_FORCES_KEY", "JointEvalStep", "JointObjective", "JointTrainStep"]

#: Namespace → parameter name → tensor, as ``model.tables`` returns it.
Tables = dict[str, dict[str, torch.Tensor]]
Output = Literal["forward", "energy", "partial_hessian", "tables"]
#: A source's traced region: ``(losses in objective order, sub_batch)``.
Body = Callable[["Trainer", "TrainState", Any], tuple[tuple[torch.Tensor, ...], Any]]

_OUTPUTS: tuple[str, ...] = get_args(Output)
#: The outputs whose body differentiates against the model's variables.
_DIFFERENTIATING: tuple[str, ...] = ("forward", "partial_hessian")


def _own_code(body: Body, source: str) -> Body:
    """*body* under a code object of its own, so dynamo caches it alone.

    ``torch.compile`` keys its cache on the function's **code object**, and
    every ``source_body`` closure :meth:`_JointBodies._source_body` makes
    shares one. K bodies across the train and eval steps therefore looked to
    dynamo like K recompiles of a single function — guards on the closure and
    on the sub-batch's key count — and the ninth was refused (the default
    ``recompile_limit`` is 8; five objectives over train and eval are ten).
    A copy of the function over a renamed code object is its own
    cache key; the closure, globals and defaults are the same objects.

    Args:
        body: A body closure from :meth:`_JointBodies._source_body`.
        source: The source's name, appended to the code name so a dynamo log
            reads ``source_body[hessian]``.

    Returns:
        The same callable under its own code object.
    """
    code = body.__code__.replace(
        co_name=f"{body.__code__.co_name}[{source}]",
        co_qualname=f"{body.__code__.co_qualname}[{source}]",
    )
    clone = types.FunctionType(
        code, body.__globals__, body.__name__, body.__defaults__, body.__closure__
    )
    clone.__kwdefaults__ = body.__kwdefaults__
    clone.__doc__ = body.__doc__
    return clone


@dataclass(frozen=True)
class JointObjective:
    """One dataset of the joint schedule: its loss, weight, and the tables it trains.

    Args:
        name: The state-scalar suffix (``state[<ns>]["loss_<name>"]``) and,
            without *source*, the key of this objective's sub-batch in the
            nested batch. Non-empty.
        loss: Called as ``loss(sub_batch, sub_batch)`` — predictions and labels
            are columns of the same sub-batch.
        weight: Multiplier on this loss in the summed objective. ``> 0``.
        trainable: The parameter tables this loss may move — ``"<namespace>.<name>"``
            (``"bonds.k"``) or a bare namespace (``"bonds"``, every parameter of
            that table). Every other predicted table is detached before the
            energy. ``None`` keeps every table's gradient.
        output: Which model primitive the body runs — ``"forward"`` (energy and
            its derivatives through :class:`~molix.autograd.Derivative`),
            ``"energy"`` (the energy alone), ``"partial_hessian"``, or
            ``"tables"`` (no energy: the kept tables are written onto the
            batch for a loss that reads a predicted table itself).
        styles: The styles the energy sums, as the model's force field names
            them (``("bond:harmonic", "angle:harmonic")``); ``None`` prices
            every style. For ``"forward"`` and ``"energy"`` — the partial
            Hessian is over every bonded term.
        nonbonded: Styles whose own force the ``"forward"`` body writes to
            :data:`NONBONDED_FORCES_KEY` beside the full force, every table
            detached and no graph kept, for a force loss that skips molecules
            by it. ``None`` writes nothing.
        source: Key of this objective's sub-batch in the nested batch, when it
            is not the objective's own name. Objectives with one source share
            that sub-batch and one ``tables`` prediction per step (ByteFF: every
            loss of a dataset on the one batch drawn from it). ``None`` reads
            ``batch[name]``.

    Raises:
        ValueError: An empty name or source, a non-positive weight, an unknown
            output, an empty / malformed ``trainable``, ``styles`` or
            ``nonbonded`` entry, ``styles`` on a ``"partial_hessian"`` or
            ``"tables"`` objective, or ``nonbonded`` on one that is not
            ``"forward"``.
        TypeError: ``loss`` is not callable.
    """

    name: str
    loss: Callable[[Any, Any], torch.Tensor]
    weight: float = 1.0
    trainable: tuple[str, ...] | None = None
    output: Output = "forward"
    styles: tuple[str, ...] | None = None
    nonbonded: tuple[str, ...] | None = None
    source: str | None = None

    @property
    def batch_key(self) -> str:
        """The nested-batch key this objective reads: *source*, else *name*."""
        return self.name if self.source is None else self.source

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"JointObjective.name must be a non-empty string, got {self.name!r}")
        if self.source is not None and (not isinstance(self.source, str) or not self.source):
            raise ValueError(
                f"JointObjective {self.name!r}: source names a sub-batch (a non-empty string) "
                f"or is None, got {self.source!r}"
            )
        if not callable(self.loss):
            raise TypeError(
                f"JointObjective {self.name!r}: loss must be callable as loss(pred, batch), "
                f"got {type(self.loss).__name__}"
            )
        if not self.weight > 0.0:
            raise ValueError(f"JointObjective {self.name!r}: weight must be > 0, got {self.weight}")
        if self.output not in _OUTPUTS:
            raise ValueError(
                f"JointObjective {self.name!r}: output must be one of {_OUTPUTS}, "
                f"got {self.output!r}"
            )
        if self.trainable is not None:
            entries = tuple(self.trainable)
            if not entries:
                raise ValueError(
                    f"JointObjective {self.name!r}: trainable names at least one table or "
                    "parameter; None trains every table"
                )
            for entry in entries:
                if not isinstance(entry, str) or not entry or entry.count(".") > 1:
                    raise ValueError(
                        f"JointObjective {self.name!r}: a trainable entry names a table "
                        f"('bonds') or a parameter ('bonds.k'), got {entry!r}"
                    )
            object.__setattr__(self, "trainable", entries)
        for field in ("styles", "nonbonded"):
            value = getattr(self, field)
            if value is None:
                continue
            entries = tuple(value)
            if not entries or not all(isinstance(entry, str) and entry for entry in entries):
                raise ValueError(
                    f"JointObjective {self.name!r}: {field} is a non-empty tuple of style names "
                    f"('bond:harmonic'), got {value!r}"
                )
            object.__setattr__(self, field, entries)
        if self.styles is not None and self.output in ("partial_hessian", "tables"):
            raise ValueError(
                f"JointObjective {self.name!r}: styles select the energy's terms; the partial "
                "Hessian is over every bonded term and 'tables' prices no energy"
            )
        if self.nonbonded is not None and self.output != "forward":
            raise ValueError(
                f"JointObjective {self.name!r}: nonbonded forces are written by the 'forward' "
                f"body only, output is {self.output!r}"
            )


class _JointBodies:
    """What the train and eval steps share: the bodies, the schedule, the detach.

    Mixed in ahead of the default step so :meth:`forward_loss` and
    :meth:`compute_loss` override the default's while ``on_train_batch`` /
    ``on_eval_batch`` are inherited.

    Args:
        objectives: The schedule, in the order the bodies run. Names unique;
            the sources run in the order they are first named, each source's
            objectives in declaration order.

    Raises:
        ValueError: No objective, or two with the same name.
    """

    #: The state namespace this step writes — ``"train"`` or ``"eval"``.
    namespace: str
    #: The traced bodies host-sync nowhere; the schedule traces.
    untraceable: str | None = None

    def __init__(self, objectives: Sequence[JointObjective]) -> None:
        declared = tuple(objectives)
        if not declared:
            raise ValueError("a joint step needs at least one JointObjective")
        names = [objective.name for objective in declared]
        if len(set(names)) != len(names):
            raise ValueError(f"JointObjective names must be unique, got {names}")
        self.objectives = declared
        grouped: dict[str, list[JointObjective]] = {}
        for objective in declared:
            grouped.setdefault(objective.batch_key, []).append(objective)
        #: Source → its objectives, in declaration order.
        self.sources: dict[str, tuple[JointObjective, ...]] = {
            source: tuple(group) for source, group in grouped.items()
        }
        self.traced_bodies: dict[str, Body] = {
            source: self._source_body(source, group) for source, group in self.sources.items()
        }
        # One Derivative op per "forward" objective, built in the eager prologue
        # the first time it is needed and reused; keyed by name, holding the
        # model it closes over so a swapped model can never be served a stale op.
        self._derivatives: dict[str, tuple[object, Derivative]] = {}
        # The same for the ``nonbonded`` force op of a "forward" objective.
        self._nonbonded: dict[str, tuple[object, Derivative]] = {}

    # ------------------------------------------------------------------
    # the detach between heads and energy
    # ------------------------------------------------------------------

    @staticmethod
    def keep(tables: Tables, trainable: Sequence[str] | None) -> Tables:
        """The tables with every parameter not in *trainable* detached.

        A pure function of its arguments; straight-line Python over string
        keys, so it traces inside a body.

        Args:
            tables: Namespace → parameter → tensor, as ``model.tables`` returns.
            trainable: ``"<ns>.<name>"`` / ``"<ns>"`` entries, or ``None``.

        Returns:
            The same nesting; a tensor is the original where kept, its
            ``.detach()`` otherwise. ``None`` returns *tables* itself.

        Raises:
            KeyError: An entry names a table or parameter *tables* does not
                hold — an objective that would otherwise train nothing fails
                on its first step instead of silently.
        """
        if trainable is None:
            return tables
        for entry in trainable:
            namespace, _, name = entry.partition(".")
            if namespace not in tables or (name and name not in tables[namespace]):
                raise KeyError(
                    f"trainable entry {entry!r} names no predicted table: the model predicts "
                    f"{ {ns: sorted(table) for ns, table in tables.items()} }"
                )
        kept: Tables = {}
        for namespace, table in tables.items():
            whole = namespace in trainable
            kept[namespace] = {
                name: tensor if whole or f"{namespace}.{name}" in trainable else tensor.detach()
                for name, tensor in table.items()
            }
        return kept

    # ------------------------------------------------------------------
    # bodies
    # ------------------------------------------------------------------

    @staticmethod
    def _model(trainer: Trainer) -> Any:
        """``trainer.model`` as the protocol object, under any compile / DDP wrapper."""
        return unwrap(trainer.model)

    def _derivative(self, objective: JointObjective, model: Any) -> Derivative:
        """The energy-and-derivatives op of a ``"forward"`` objective over *model*.

        ``Derivative`` over ``model.energy(frame, tables=tables)`` with the
        model's own variables — the tables arrive as the op's keyword argument,
        predicted once per source by the body and kept per objective. Built
        once per objective and reused — a parameterless module, so building it
        lazily costs nothing. Called in the eager prologue (construction) and
        in the body (a cache hit).
        """
        cached = self._derivatives.get(objective.name)
        if cached is not None and cached[0] is model:
            return cached[1]
        styles = objective.styles

        def energy(frame: Any, *, tables: Tables) -> Any:
            if styles is None:
                return model.energy(frame, tables=tables)
            return model.energy(frame, tables=tables, styles=styles)

        # The step puts the force beside its label, whatever the model's own op does.
        wrt = model.derivative.wrt
        outputs = tuple(FORCES_PRED_KEY if key == POS_KEY else None for key in wrt)
        op = Derivative(energy, wrt=wrt, outputs=outputs)
        self._derivatives[objective.name] = (model, op)
        return op

    def _nonbonded_derivative(self, objective: JointObjective, model: Any) -> Derivative:
        """The force of ``objective.nonbonded`` alone, written to :data:`NONBONDED_FORCES_KEY`.

        The body hands it the source's tables with every one detached
        (``keep(..., ())``): the geometry is the only variable, and the op runs
        under ``no_grad`` so the force comes out detached and the energy graph
        is not kept. Positions only — a clash is a question about
        ``atoms.pos``, whatever else the model's own op differentiates against.
        """
        cached = self._nonbonded.get(objective.name)
        if cached is not None and cached[0] is model:
            return cached[1]
        styles = objective.nonbonded

        def energy(frame: Any, *, tables: Tables) -> Any:
            return model.energy(frame, tables=tables, styles=styles)

        op = Derivative(energy, wrt=(POS_KEY,), outputs=(NONBONDED_FORCES_KEY,))
        self._nonbonded[objective.name] = (model, op)
        return op

    def _source_body(self, source: str, objectives: tuple[JointObjective, ...]) -> Body:
        """The traced region of one source: ``tables`` once, then each objective.

        Resolves the model from ``trainer`` at call time (the step is built
        before the trainer), predicts the tables once for the sub-batch, and
        scores every objective of the source on them in declaration order —
        a static Python loop over a tuple fixed at construction. Returns
        ``(losses, sub_batch)`` with the predictions written beside the labels.
        """

        def source_body(
            trainer: Trainer, state: TrainState, sub: Any
        ) -> tuple[tuple[torch.Tensor, ...], Any]:
            del state
            model = self._model(trainer)
            tables = model.tables(sub)
            losses: list[torch.Tensor] = []
            for objective in objectives:
                losses.append(self._score(objective, model, sub, tables))
            return tuple(losses), sub

        return _own_code(source_body, source)

    def _score(self, objective: JointObjective, model: Any, sub: Any, tables: Tables) -> Any:
        """One objective on its source's tables: keep, primitive, predictions, loss.

        Straight-line code on constants of the objective (its output kind, its
        styles), so it inlines into the source's region. Every write lands on
        *sub* beside the labels, which stay the labels.
        """
        kept = self.keep(tables, objective.trainable)
        if objective.output == "forward":
            op = self._derivative(objective, model)
            aside = (
                None
                if objective.nonbonded is None
                else self._nonbonded_derivative(objective, model)
            )

            def priced() -> Any:
                if aside is not None:
                    # Grad off at entry: the op's one ``autograd.grad`` consumes
                    # the graph and the force lands detached; the full forward
                    # below overwrites the energy it wrote. The tables are the
                    # source's, every one detached.
                    with torch.no_grad():
                        aside(sub, tables=self.keep(tables, ()))
                return op(sub, tables=kept)

            # The force lands beside its label (the op's output key); the
            # energy is moved beside its own once both forwards have run.
            beside_labels(sub, {ENERGY_KEY: ENERGY_PRED_KEY}, priced)
            return objective.loss(sub, sub)
        if objective.output == "energy":
            styles = objective.styles

            def priced() -> Any:
                if styles is None:
                    return model.energy(sub, tables=kept)
                return model.energy(sub, tables=kept, styles=styles)

            beside_labels(sub, {ENERGY_KEY: ENERGY_PRED_KEY}, priced)
            return objective.loss(sub, sub)
        if objective.output == "tables":
            # No energy: the kept tables become columns of the namespaces the
            # batch carries, as ``energy`` would have written them, and the
            # loss reads the one it supervises.
            for namespace, table in kept.items():
                if namespace in sub.keys():
                    for name, value in table.items():
                        sub[namespace, name] = value
            return objective.loss(sub, sub)
        sub[HESSIAN_KEY] = model.partial_hessian(sub, tables=kept)
        return objective.loss(sub, sub)

    # ------------------------------------------------------------------
    # the schedule
    # ------------------------------------------------------------------

    def forward_loss(
        self,
        trainer: Trainer,
        state: TrainState,
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Run every source's body on its sub-batch; return the weighted sum.

        The eager prologue of each body: the sub-batch is read by source, the
        ``"forward"`` ops are built on first use, and the position leaf is
        marked once when any objective of the source differentiates
        (``"forward"``, ``"partial_hessian"``). Then the body — the compiled
        region, one ``tables`` and every objective of the source — and the
        detached per-objective scalars into ``state[namespace]["loss_<name>"]``.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`.
            state: The :class:`~molix.core.state.TrainState` to record into.
            batch: Nested ``TensorDict(batch_size=[])`` keyed by source
                (:attr:`JointObjective.batch_key`), each value that source's
                padded sub-batch.

        Returns:
            ``(total, predictions)`` — ``sum(weight * loss)`` over the
            objectives, and ``{source: sub_batch}`` with the predictions written.
        """
        total: torch.Tensor | None = None
        predictions: dict[str, Any] = {}
        for source, objectives in self.sources.items():
            sub = batch[source]
            losses = self.source_losses(trainer, state, sub, source)
            for objective, loss in zip(objectives, losses, strict=True):
                state[self.namespace][f"loss_{objective.name}"] = loss.detach()
                term = objective.weight * loss
                total = term if total is None else total + term
            predictions[source] = sub
        assert total is not None
        return total, predictions

    def source_losses(
        self, trainer: Trainer, state: TrainState, sub: Any, source: str
    ) -> tuple[torch.Tensor, ...]:
        """One source's objective losses, in declaration order, on its sub-batch.

        The eager prologue (the ``"forward"`` ops built on first use, the
        position leaf marked when any objective differentiates) and then the
        source's traced body. The predictions are written onto *sub*.

        Args:
            trainer: The owning :class:`~molix.core.trainer.Trainer`.
            state: The :class:`~molix.core.state.TrainState` the body may read.
            sub: The source's padded sub-batch.
            source: Its name (:attr:`JointObjective.batch_key`).

        Returns:
            The unweighted loss of every objective of *source*.
        """
        model = self._model(trainer)
        objectives = self.sources[source]
        differentiates = False
        for objective in objectives:
            if objective.output == "forward":
                self._derivative(objective, model)
                if objective.nonbonded is not None:
                    self._nonbonded_derivative(objective, model)
            differentiates = differentiates or objective.output in _DIFFERENTIATING
        if differentiates:
            model.derivative.mark_leaves(sub)
        losses, _ = self.traced_bodies[source](trainer, state, sub)
        return losses

    def _traced_forward_loss(
        self,
        trainer: Trainer,
        state: TrainState,
        batch: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Not the seam here — the compiled regions are :attr:`traced_bodies`."""
        raise NotImplementedError(
            f"{type(self).__name__} has one traced body per source (traced_bodies); "
            "there is no single traced region"
        )

    def compute_loss(
        self,
        trainer: Trainer,
        state: TrainState,
        batch: Any,
        predictions: Any,
    ) -> torch.Tensor:
        """Not the seam here — each objective's loss runs inside its body."""
        raise NotImplementedError(
            f"{type(self).__name__} evaluates each objective's loss inside its traced body; "
            "declare the loss on the JointObjective"
        )

    def _loss_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple((self.namespace, f"loss_{o.name}") for o in self.objectives)


class JointTrainStep(_JointBodies, DefaultTrainStep):
    """Training step over a nested batch: one body per source, one backward.

    Inherits ``on_train_batch`` — AMP, the accumulation window,
    ``optimizer_applied`` and the ``on_after_backward`` fire point are the
    default step's — and replaces ``forward_loss`` with the joint schedule.
    Runs with ``Trainer(loss_fn=None)``: the objectives own their losses.

    Args:
        objectives: The schedule, in the order the bodies run.
    """

    namespace = "train"

    def __init__(self, objectives: Sequence[JointObjective]) -> None:
        _JointBodies.__init__(self, objectives)

    @property
    def scalar_keys(self) -> tuple[tuple[str, str], ...]:
        """``("train", "loss_<name>")`` per objective, for the scalar hooks."""
        return self._loss_keys()


class JointEvalStep(_JointBodies, DefaultEvalStep):
    """Evaluation step over a nested batch, same bodies, no parameter update.

    Inherits ``on_eval_batch`` — the grad context and AMP are the default
    step's — and replaces ``forward_loss`` with the joint schedule.

    Args:
        objectives: The schedule, in the order the bodies run.
        no_grad: Forwarded to :class:`~molix.core.steps.eval.DefaultEvalStep`.
            Must stay ``False`` (the default) while a ``"forward"`` or
            ``"partial_hessian"`` objective is present — their bodies
            differentiate.
    """

    namespace = "eval"

    def __init__(self, objectives: Sequence[JointObjective], *, no_grad: bool = False) -> None:
        DefaultEvalStep.__init__(self, no_grad=no_grad)
        _JointBodies.__init__(self, objectives)

    @property
    def scalar_keys(self) -> tuple[tuple[str, str], ...]:
        """``("eval", "loss")`` then ``("eval", "loss_<name>")`` per objective.

        ``("eval", "loss")`` is listed so
        :meth:`~molix.core.trainer.Trainer._run_eval_phase` publishes its
        mean over the phase rather than the last batch's value.
        """
        return (("eval", "loss"), *self._loss_keys())
