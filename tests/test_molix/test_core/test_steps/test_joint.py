"""Tests for :mod:`molix.core.steps.joint` — the schedule, its bodies, the detach.

The model is a local stub implementing only what the step reads —
``tables`` / ``energy`` / ``partial_hessian`` / ``derivative`` — so nothing
here imports ``molpot`` / ``molrep`` / ``molzoo``.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.autograd import Derivative
from molix.core.state import TrainState
from molix.core.steps.eval import DefaultEvalStep
from molix.core.steps.joint import JointEvalStep, JointObjective, JointTrainStep
from molix.core.steps.train import DefaultTrainStep
from molix.core.trainer import Trainer

ENERGY_PRED = ("graphs", "energy_pred")
FORCES_PRED = ("atoms", "forces_pred")
HESSIAN = ("hessian_pairs", "partial_hessian")


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubModel(nn.Module):
    """The protocol and nothing else: one shared encoder feeding two one-parameter tables.

    Two "styles": ``quad`` prices ``x |pos|²`` per atom, ``lin`` prices
    ``y Σ pos``; ``energy(styles=)`` sums the ones named.
    """

    STYLES = ("quad", "lin")

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.encoder = nn.Linear(3, 4)
        self.head_a = nn.Linear(4, 1)  # table "a", parameter "x"
        self.head_b = nn.Linear(4, 1)  # table "b", parameter "y"
        self._derivative = Derivative(self.energy, wrt=("atoms.pos",))

    @property
    def derivative(self) -> Derivative:
        return self._derivative

    def _heads(
        self, frame: Any, *, trainable: tuple[str, ...] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.encoder(frame["atoms", "pos"]))
        x = self.head_a(hidden).squeeze(-1)
        y = self.head_b(hidden).squeeze(-1)
        if trainable is not None:
            named = set(trainable)
            if "a" not in named and "a.x" not in named:
                x = x.detach()
            if "b" not in named and "b.y" not in named:
                y = y.detach()
        if "a" in frame.keys():
            frame["a", "x"] = x
        if "b" in frame.keys():
            frame["b", "y"] = y
        return x, y

    def energy(
        self,
        frame: Any,
        *,
        styles: tuple[str, ...] | None = None,
        trainable: tuple[str, ...] | None = None,
    ) -> Any:
        chosen = self.STYLES if styles is None else styles
        unknown = [style for style in chosen if style not in self.STYLES]
        if unknown:
            raise ValueError(f"unknown styles {unknown}")
        x, y = self._heads(frame, trainable=trainable)
        pos = frame["atoms", "pos"]
        per_atom = pos.new_zeros(pos.shape[0])
        if "quad" in chosen:
            per_atom = per_atom + x * pos.pow(2).sum(-1)
        if "lin" in chosen:
            per_atom = per_atom + y * pos.sum(-1)
        n_graphs = frame["graphs", "num_atoms"].shape[0]
        frame["graphs", "energy"] = per_atom.new_zeros(n_graphs).index_add(
            0, frame["atoms", "batch"], per_atom
        )
        return frame

    def tables(self, frame: Any, *, trainable: tuple[str, ...] | None = None) -> Any:
        self._heads(frame, trainable=trainable)
        return frame

    def partial_hessian(
        self, frame: Any, *, trainable: tuple[str, ...] | None = None
    ) -> torch.Tensor:
        x, y = self._heads(frame, trainable=trainable)
        pos = frame["atoms", "pos"]
        i, j = frame["hessian_pairs", "i"], frame["hessian_pairs", "j"]
        diff = pos[j] - pos[i]
        stiffness = x[i] + y[j]
        return stiffness.unsqueeze(-1) * (diff.unsqueeze(-1) * diff.unsqueeze(-2)).reshape(-1, 9)


class _Recording:
    """Squared mean of one prediction column; remembers what it was handed."""

    def __init__(self, key: tuple[str, str]) -> None:
        self.key = key
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, pred: Any, batch: Any) -> torch.Tensor:
        self.calls.append((pred, batch))
        return pred[self.key].pow(2).mean()


def _sub(seed: int) -> TensorDict:
    """One padded sub-batch: 4 atoms in 2 graphs, labels, and two Hessian pairs."""
    gen = torch.Generator().manual_seed(seed)
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": torch.randn(4, 3, generator=gen),
                    "batch": torch.tensor([0, 0, 1, 1]),
                    "mask": torch.ones(4),
                    "forces": torch.randn(4, 3, generator=gen),
                },
                batch_size=[4],
            ),
            "graphs": TensorDict(
                {"num_atoms": torch.tensor([2, 2]), "energy": torch.randn(2, generator=gen)},
                batch_size=[2],
            ),
            "hessian_pairs": TensorDict(
                {
                    "i": torch.tensor([0, 2]),
                    "j": torch.tensor([1, 3]),
                    "block": torch.randn(2, 9, generator=gen),
                    "mask": torch.ones(2),
                },
                batch_size=[2],
            ),
        },
        batch_size=[],
    )


def _nested(*names: str) -> TensorDict:
    return TensorDict({name: _sub(seed) for seed, name in enumerate(names)}, batch_size=[])


def _two_objectives(
    trainable: tuple[str, ...] | None = None,
) -> tuple[JointObjective, JointObjective]:
    return (
        JointObjective("a", _Recording(FORCES_PRED), weight=2.0, trainable=trainable),
        JointObjective(
            "b", _Recording(ENERGY_PRED), weight=0.5, trainable=trainable, output="energy"
        ),
    )


def _trainer(train_step: Any = None, eval_step: Any = None) -> Trainer:
    return Trainer(
        model=_StubModel(),
        loss_fn=None,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        train_step=train_step,
        eval_step=eval_step,
    )


# ---------------------------------------------------------------------------
# JointObjective
# ---------------------------------------------------------------------------


class TestJointObjective:
    @pytest.mark.parametrize(
        "override",
        [
            {"name": ""},
            {"weight": 0.0},
            {"output": "hessian"},
            {"trainable": ("a.x.y",)},
            {"trainable": ()},
        ],
        ids=[
            "empty-name",
            "zero-weight",
            "unknown-output",
            "malformed-trainable",
            "empty-trainable",
        ],
    )
    def test_rejects_a_malformed_declaration(self, override: dict[str, Any]) -> None:
        declaration: dict[str, Any] = {"name": "a", "loss": _Recording(ENERGY_PRED), **override}
        with pytest.raises(ValueError):
            JointObjective(**declaration)

    # ---------------------------------------------------------------------------
    # keep — the detach between heads and energy
    # ---------------------------------------------------------------------------

    def test_styles_do_not_apply_to_the_partial_hessian(self) -> None:
        with pytest.raises(ValueError, match="partial Hessian"):
            JointObjective("h", _Recording(HESSIAN), output="partial_hessian", styles=("quad",))

    def test_source_is_a_name_or_none(self) -> None:
        loss = _Recording(ENERGY_PRED)
        assert JointObjective("a", loss).batch_key == "a"
        assert JointObjective("a", loss, source="s").batch_key == "s"
        with pytest.raises(ValueError, match="source"):
            JointObjective("a", loss, source="")

    def test_styles_do_not_apply_to_the_tables_output(self) -> None:
        with pytest.raises(ValueError, match="styles"):
            JointObjective("a", _Recording(ENERGY_PRED), output="tables", styles=("quad",))

    def test_nonbonded_needs_the_forward_body(self) -> None:
        with pytest.raises(ValueError, match="forward"):
            JointObjective("e", _Recording(ENERGY_PRED), output="energy", nonbonded=("lin",))

    @pytest.mark.parametrize("field", ["styles", "nonbonded"])
    def test_an_empty_style_list_is_refused(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            JointObjective("a", _Recording(FORCES_PRED), **{field: ()})


class TestKeep:
    @staticmethod
    def _tables() -> dict[str, dict[str, torch.Tensor]]:
        source = torch.ones(2, requires_grad=True)
        return {"a": {"x": source * 2.0, "z": source * 3.0}, "b": {"y": source * 4.0}}

    def test_detaches_every_parameter_not_named(self) -> None:
        kept = JointTrainStep.keep(self._tables(), ("a.x",))
        assert kept["a"]["x"].requires_grad
        assert not kept["a"]["z"].requires_grad
        assert not kept["b"]["y"].requires_grad

    def test_a_bare_namespace_keeps_its_whole_table(self) -> None:
        kept = JointTrainStep.keep(self._tables(), ("a",))
        assert kept["a"]["x"].requires_grad and kept["a"]["z"].requires_grad
        assert not kept["b"]["y"].requires_grad

    def test_none_returns_the_tables_themselves(self) -> None:
        tables = self._tables()
        assert JointTrainStep.keep(tables, None) is tables

    @pytest.mark.parametrize("entry", ["c", "a.w"], ids=["unknown-table", "unknown-parameter"])
    def test_an_unknown_name_is_refused(self, entry: str) -> None:
        with pytest.raises(KeyError, match=entry):
            JointTrainStep.keep(self._tables(), (entry,))


# ---------------------------------------------------------------------------
# JointTrainStep
# ---------------------------------------------------------------------------


class TestJointTrainStep:
    def test_does_not_override_on_train_batch(self) -> None:
        assert JointTrainStep.on_train_batch is DefaultTrainStep.on_train_batch

    def test_rejects_duplicate_objective_names(self) -> None:
        loss = _Recording(ENERGY_PRED)
        with pytest.raises(ValueError, match="unique"):
            JointTrainStep([JointObjective("a", loss), JointObjective("a", loss)])

    def test_each_body_owns_its_code_object(self) -> None:
        # dynamo caches per code object: three energy bodies sharing one would be
        # three "recompiles" of a single function and hit recompile_limit.
        loss = _Recording(ENERGY_PRED)
        step = JointTrainStep(
            [
                JointObjective("a", loss, output="energy"),
                JointObjective("b", loss, output="energy"),
                JointObjective("c", loss, output="forward"),
            ]
        )
        codes = {name: body.__code__ for name, body in step.traced_bodies.items()}
        assert codes["a"] is not codes["b"]
        assert codes["a"].co_name == "source_body[a]"
        assert codes["b"].co_name == "source_body[b]"
        assert codes["c"].co_name == "source_body[c]"

    def test_total_is_the_weighted_sum_of_the_objective_losses(self) -> None:
        step = JointTrainStep(_two_objectives())
        trainer = _trainer(train_step=step)
        state = TrainState()

        total, _ = step.forward_loss(trainer, state, _nested("a", "b"))

        assert torch.equal(total, 2.0 * state["train"]["loss_a"] + 0.5 * state["train"]["loss_b"])

    def test_source_losses_are_what_forward_loss_publishes(self) -> None:
        step = JointTrainStep(_two_objectives())
        trainer = _trainer(train_step=step)
        batch = _nested("a", "b")
        state = TrainState()

        losses = step.source_losses(trainer, state, batch["a"].clone(), "a")
        step.forward_loss(trainer, state, batch)

        assert len(losses) == len(step.sources["a"])
        assert torch.equal(losses[0], state["train"]["loss_a"])

    def test_publishes_each_loss_detached(self) -> None:
        step = JointTrainStep(_two_objectives())
        trainer = _trainer(train_step=step)
        state = TrainState()

        step.on_train_batch(trainer, state, _nested("a", "b"))

        for name in ("loss_a", "loss_b"):
            scalar = state["train"][name]
            assert isinstance(scalar, torch.Tensor)
            assert scalar.requires_grad is False and scalar.grad_fn is None

    def test_each_body_sees_only_its_own_sub_batch(self) -> None:
        objectives = _two_objectives()
        step = JointTrainStep(objectives)
        trainer = _trainer(train_step=step)
        batch = _nested("a", "b")

        step.forward_loss(trainer, TrainState(), batch)

        for objective in objectives:
            ((pred, seen),) = objective.loss.calls  # ty: ignore[unresolved-attribute]
            assert pred is seen is batch[objective.name]

    def test_trainable_cuts_the_other_tables_gradient_but_not_the_encoders(self) -> None:
        step = JointTrainStep(_two_objectives(trainable=("a.x",)))
        trainer = _trainer(train_step=step)
        model = trainer.model

        step.on_train_batch(trainer, TrainState(), _nested("a", "b"))

        assert model.head_b.weight.grad is None
        assert model.head_a.weight.grad is not None
        assert model.encoder.weight.grad is not None

    def test_forward_output_writes_predictions_and_keeps_the_labels(self) -> None:
        step = JointTrainStep([JointObjective("a", _Recording(FORCES_PRED))])
        trainer = _trainer(train_step=step)
        batch = _nested("a")
        sub = batch["a"]
        energy_label, forces_label = sub["graphs", "energy"].clone(), sub["atoms", "forces"].clone()

        step.on_train_batch(trainer, TrainState(), batch)

        pos = sub["atoms", "pos"]
        assert pos.is_leaf and pos.requires_grad and pos.grad is not None
        assert sub["atoms", "forces_pred"].shape == (4, 3)
        assert sub["graphs", "energy_pred"].shape == (2,)
        assert torch.equal(sub["atoms", "forces"], forces_label)
        assert torch.equal(sub["graphs", "energy"], energy_label)

    def test_energy_output_writes_energy_pred_and_keeps_the_label(self) -> None:
        step = JointTrainStep([JointObjective("b", _Recording(ENERGY_PRED), output="energy")])
        trainer = _trainer(train_step=step)
        batch = _nested("b")
        sub = batch["b"]
        energy_label = sub["graphs", "energy"].clone()

        step.forward_loss(trainer, TrainState(), batch)

        assert sub["graphs", "energy_pred"].requires_grad
        assert torch.equal(sub["graphs", "energy"], energy_label)
        assert not sub["atoms", "pos"].requires_grad

    def test_styles_restrict_the_energy_the_body_prices(self) -> None:
        """ByteFF ``Bond_Energy``: the objective's styles alone become ``energy_pred``."""
        step = JointTrainStep(
            [JointObjective("b", _Recording(ENERGY_PRED), output="energy", styles=("quad",))]
        )
        trainer = _trainer(train_step=step)
        batch = _nested("b")
        sub = batch["b"]

        step.forward_loss(trainer, TrainState(), batch)

        pos = sub["atoms", "pos"]
        hidden = torch.tanh(trainer.model.encoder(pos))
        quad = trainer.model.head_a(hidden).squeeze(-1) * pos.pow(2).sum(-1)
        expected = quad.new_zeros(2).index_add(0, sub["atoms", "batch"], quad)
        torch.testing.assert_close(sub["graphs", "energy_pred"], expected)

    def test_nonbonded_styles_write_their_force_aside_detached(self) -> None:
        """ByteFF ``skip_large_nb`` reads the pair styles' own force: every table
        detached, no graph kept, the full force untouched beside it."""
        step = JointTrainStep([JointObjective("a", _Recording(FORCES_PRED), nonbonded=("lin",))])
        trainer = _trainer(train_step=step)
        batch = _nested("a")
        sub = batch["a"]
        energy_label, forces_label = sub["graphs", "energy"].clone(), sub["atoms", "forces"].clone()

        step.forward_loss(trainer, TrainState(), batch)

        aside = sub["atoms", "forces_nonbonded"]
        assert not aside.requires_grad
        # ``lin`` is y Σ_k pos_k per atom with y detached: the force is -y on every component.
        hidden = torch.tanh(trainer.model.encoder(sub["atoms", "pos"]))
        y = trainer.model.head_b(hidden).squeeze(-1).detach()
        torch.testing.assert_close(aside, -y.unsqueeze(-1).expand(-1, 3))
        full = sub["atoms", "forces_pred"]
        assert full.requires_grad
        pos = sub["atoms", "pos"].detach().clone().requires_grad_(True)
        probe = sub.clone()
        probe["atoms", "pos"] = pos
        (grad,) = torch.autograd.grad(trainer.model.energy(probe)["graphs", "energy"].sum(), pos)
        torch.testing.assert_close(full.detach(), -grad)
        assert torch.equal(sub["graphs", "energy"], energy_label)
        assert torch.equal(sub["atoms", "forces"], forces_label)

    def test_partial_hessian_output_writes_the_blocks(self) -> None:
        step = JointTrainStep([JointObjective("h", _Recording(HESSIAN), output="partial_hessian")])
        trainer = _trainer(train_step=step)
        batch = _nested("h")

        step.forward_loss(trainer, TrainState(), batch)

        assert batch["h", "hessian_pairs", "partial_hessian"].shape == (2, 9)

    def test_one_traced_body_per_source(self) -> None:
        step = JointTrainStep(_two_objectives())
        assert tuple(step.traced_bodies) == ("a", "b")
        loss = _Recording(ENERGY_PRED)
        shared = JointTrainStep(
            [
                JointObjective("k", loss, output="energy", source="s"),
                JointObjective("g", _Recording(FORCES_PRED), source="s"),
                JointObjective("m", loss, output="energy"),
            ]
        )
        assert tuple(shared.traced_bodies) == ("s", "m")
        assert shared.sources["s"] == shared.objectives[:2]
        assert shared.sources["m"] == shared.objectives[2:]

    def test_objectives_on_one_source_share_the_batch(self) -> None:
        """ByteFF: every loss of a dataset reads the one batch drawn from it."""
        k = JointObjective(
            "k", _Recording(HESSIAN), trainable=("a.x",), output="partial_hessian", source="s"
        )
        g = JointObjective(
            "g", _Recording(FORCES_PRED), trainable=("b.y",), source="s", nonbonded=("lin",)
        )
        e = JointObjective(
            "e",
            _Recording(ENERGY_PRED),
            trainable=("a.x",),
            output="energy",
            styles=("quad",),
            source="s",
        )
        step = JointTrainStep([k, g, e])
        trainer = _trainer(train_step=step)
        model = trainer.model
        batch = TensorDict({"s": _sub(0)}, batch_size=[])
        state = TrainState()

        total, predictions = step.forward_loss(trainer, state, batch)
        for objective in (k, g, e):
            ((pred, got),) = objective.loss.calls  # ty: ignore[unresolved-attribute]
            assert pred is got is batch["s"]
        assert tuple(predictions) == ("s",)
        assert torch.equal(
            total, state["train"]["loss_k"] + state["train"]["loss_g"] + state["train"]["loss_e"]
        )
        total.backward()
        assert model.encoder.weight.grad is not None
        assert model.head_a.weight.grad is not None
        assert model.head_b.weight.grad is not None

    def test_keep_is_per_objective_on_the_shared_tables(self) -> None:
        """Two objectives on one source, both naming ``a.x``: ``b`` gets no gradient."""
        step = JointTrainStep(
            [
                JointObjective(
                    "k",
                    _Recording(HESSIAN),
                    trainable=("a.x",),
                    output="partial_hessian",
                    source="s",
                ),
                JointObjective("g", _Recording(FORCES_PRED), trainable=("a.x",), source="s"),
            ]
        )
        trainer = _trainer(train_step=step)
        model = trainer.model

        step.on_train_batch(trainer, TrainState(), TensorDict({"s": _sub(0)}, batch_size=[]))

        assert model.head_b.weight.grad is None
        assert model.head_a.weight.grad is not None
        assert model.encoder.weight.grad is not None

    def test_tables_output_writes_the_kept_tables_and_prices_no_energy(self) -> None:
        loss = _Recording(("a", "x"))
        step = JointTrainStep([JointObjective("t", loss, trainable=("a.x",), output="tables")])
        trainer = _trainer(train_step=step)
        batch = _nested("t")
        sub = batch["t"]
        sub["a"] = TensorDict({}, batch_size=[4])  # the namespaces the tables land on
        sub["b"] = TensorDict({}, batch_size=[4])

        step.forward_loss(trainer, TrainState(), batch)

        assert sub["a", "x"].requires_grad
        assert not sub["b", "y"].requires_grad
        assert ("graphs", "energy_pred") not in sub.keys(include_nested=True)
        assert not sub["atoms", "pos"].requires_grad
        ((pred, got),) = loss.calls
        assert pred is got is sub

    def test_tables_output_runs_on_a_source_without_an_energy_label(self) -> None:
        # MBIS records carry charges and no energy.
        loss = _Recording(("a", "x"))
        step = JointTrainStep([JointObjective("t", loss, trainable=("a.x",), output="tables")])
        trainer = _trainer(train_step=step)
        batch = _nested("t")
        sub = batch["t"]
        sub["a"] = TensorDict({}, batch_size=[4])
        sub["b"] = TensorDict({}, batch_size=[4])
        del sub["graphs", "energy"]
        columns = sorted(sub["graphs"].keys())

        step.forward_loss(trainer, TrainState(), batch)

        assert sub["a", "x"].requires_grad
        assert len(loss.calls) == 1
        # Scored twice (a resident validation batch), it must arrive the same.
        assert sorted(sub["graphs"].keys()) == columns

    def test_trainer_compile_replaces_every_body_and_leaves_the_model_alone(self) -> None:
        train_step = JointTrainStep(_two_objectives())
        eval_step = JointEvalStep(_two_objectives())
        trainer = _trainer(train_step=train_step, eval_step=eval_step)
        model = trainer.model
        compiled: dict[int, object] = {}

        def fake_compile(target: Any, **_: Any) -> object:
            compiled[id(target)] = sentinel = mock.Mock(name="compiled")
            return sentinel

        with mock.patch("molix.compile.torch.compile", side_effect=fake_compile):
            trainer.compile()

        for step in (train_step, eval_step):
            assert all(isinstance(body, mock.Mock) for body in step.traced_bodies.values())
        assert len(compiled) == 4
        assert trainer.model is model

    def test_compute_loss_is_not_the_seam(self) -> None:
        step = JointTrainStep(_two_objectives())
        with pytest.raises(NotImplementedError, match="traced body"):
            step.compute_loss(None, None, None, None)

    def test_wrap_distributed_refuses_the_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch.distributed as dist

        trainer = _trainer(train_step=JointTrainStep(_two_objectives()))
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        with pytest.raises(ValueError, match="JointTrainStep"):
            trainer._wrap_distributed()


# ---------------------------------------------------------------------------
# JointEvalStep
# ---------------------------------------------------------------------------


class TestJointEvalStep:
    def test_does_not_override_on_eval_batch(self) -> None:
        assert JointEvalStep.on_eval_batch is DefaultEvalStep.on_eval_batch

    def test_scalar_keys_lead_with_the_phase_mean_loss(self) -> None:
        assert JointEvalStep(_two_objectives()).scalar_keys == (
            ("eval", "loss"),
            ("eval", "loss_a"),
            ("eval", "loss_b"),
        )


# ---------------------------------------------------------------------------
# Compile seam
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    ("output", "key", "extra"),
    [
        ("forward", FORCES_PRED, {}),
        ("energy", ENERGY_PRED, {}),
        ("partial_hessian", HESSIAN, {}),
        ("forward", FORCES_PRED, {"nonbonded": ("lin",)}),
        ("energy", ENERGY_PRED, {"styles": ("quad",)}),
    ],
)
def test_each_body_explains_as_one_graph(
    output: str, key: tuple[str, str], extra: dict[str, Any]
) -> None:
    """A body is the model primitive, the detach and the loss — no prologue inside."""
    from molix.compile import Compiler

    step = JointTrainStep(
        [JointObjective("a", _Recording(key), trainable=("a.x",), output=output, **extra)]
    )
    trainer = _trainer(train_step=step)
    # The eager prologue once: builds the derivative op and marks the leaf.
    step.forward_loss(trainer, TrainState(), _nested("a"))
    fresh = _nested("a")["a"]
    if output != "energy":
        trainer.model.derivative.mark_leaves(fresh)

    summary = Compiler.explain(step.traced_bodies["a"], trainer, TrainState(), fresh)

    assert (summary.graph_count, summary.graph_break_count) == (1, 0), summary.break_sites


@pytest.mark.slow
def test_a_source_region_with_every_output_explains_as_one_graph() -> None:
    """One source, four objectives of every output kind: one region, no break."""
    from molix.compile import Compiler

    step = JointTrainStep(
        [
            JointObjective(
                "k", _Recording(HESSIAN), trainable=("a.x",), output="partial_hessian", source="s"
            ),
            JointObjective(
                "g", _Recording(FORCES_PRED), trainable=("b.y",), source="s", nonbonded=("lin",)
            ),
            JointObjective(
                "e",
                _Recording(ENERGY_PRED),
                trainable=("a.x",),
                output="energy",
                styles=("quad",),
                source="s",
            ),
            JointObjective(
                "t", _Recording(("a", "x")), trainable=("a.x",), output="tables", source="s"
            ),
        ]
    )
    trainer = _trainer(train_step=step)

    def nested() -> TensorDict:
        sub = _sub(0)
        sub["a"] = TensorDict({}, batch_size=[4])
        sub["b"] = TensorDict({}, batch_size=[4])
        return TensorDict({"s": sub}, batch_size=[])

    # The eager prologue once: builds the derivative ops and marks the leaf.
    step.forward_loss(trainer, TrainState(), nested())
    fresh = nested()["s"]
    trainer.model.derivative.mark_leaves(fresh)

    summary = Compiler.explain(step.traced_bodies["s"], trainer, TrainState(), fresh)

    assert (summary.graph_count, summary.graph_break_count) == (1, 0), summary.break_sites
