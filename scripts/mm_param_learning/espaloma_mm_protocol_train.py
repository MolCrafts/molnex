#!/usr/bin/env python3
"""Espaloma-protocol MM fitting on a small PhAlkEthOH subset (our molexp project).

Matches Espaloma's published MM-small validation protocol as closely as our stack allows:

* molecule-level shuffle/split seed **2666**, ratios **8:1:1** (80/10/10)
* loss = graph-level **MSE(E_pred, E_ref)** (absolute MM energy, not QM-centering)
* optimizer **Adam(lr=1e-4)**
* train loader **batch_size=100** conformers
* report L1 (MAE) in kcal/mol (Espaloma notebook multiplies Hartree metrics by 627.5;
  our labels are already kcal/mol after conversion)

Architecture (Espaloma ↔ ours):

| Espaloma | Ours |
|----------|------|
| SAGEConv ×3, 128, ReLU | ChemEncoder atom/bond/angle/proper dim=128 |
| Janossy → bond/angle log-coeff, torsion k×6 | Bond/Angle/ProperParamHead (n_terms=6) |
| GeometryInGraph + EnergyInGraph | ForceField.to_potentials() Class-I kernels |
| ChargeEquilibrium + NB | ``--energy-scope full``: EEM + LJ + Coulomb (algorithm-named) |

Small system (default): the **12 smallest molecules** (by atom count) in the
normalized PhAlkEthOH tree, **all 100 confs** (or ``--max-confs``).

Runs under the molexp project ``mm-param-learning`` experiment
``espaloma-mm-protocol-mini`` when ``--record-root`` points at a run dir.

Training composition (``universal-class-i-13-grad-audit``)
---------------------------------------------------------

There is **no** hand-rolled epoch loop and **no** flat conformer pool here.
The script composes molix primitives — ``ConformerDataset`` +
``EnergyTaskSamplerSource`` (:class:`molix.data.TaskMixtureBatchSampler`)
behind a :class:`molix.data.DataModule`, a
:class:`molix.core.steps.MultiTaskTrainStep` routing the single ``energy``
task, and :class:`molix.core.trainer.Trainer` with
``GradClipHook`` / ``MetricsHook`` / ``CheckpointHook(save_best=True)`` /
``EarlyStop(if_nan=True)`` — so gradient clipping, best-model tracking, the
optimizer step and the epoch cadence are all owned by the library. Composition
stays in this script: no library façade wraps it.

Two deliberate cadence deviations from the retired flat-pool loop (the pinned
protocol numbers — objective, ``lr``, split seed 2666 / 8:1:1, metric
definitions, CLI — are unchanged):

1. **Batches are molecule-grouped, not a shuffled conformer pool.** One batch
   carries whole molecules (every conformer of each), packed under a per-batch
   atom budget of ``batch_size × max(atoms per conformer)`` — so a batch holds
   about ``batch_size`` conformers, but the exact count varies with molecule
   size and a molecule's conformer set never straddles two batches. Required by
   ``.claude/notes/production-training-checklist.md`` §1 (no flat concatenated
   frame pool enters the loader).
2. **``--eval-every`` is realised in optimizer steps**, as
   ``eval_every × steps_per_epoch`` (``Trainer.eval_every_n_steps``), instead of
   an epoch-modulo test; a final eval phase runs after the loop so the last
   epoch always reports. ``train_mse`` is the mean over the batches since the
   previous eval, and ``train_mae_kcal`` / ``train_rmse_kcal`` are
   ``MetricsHook``'s per-batch train values rather than a second full pass over
   the train split (``eval_split`` remains the final test-split reporter).

``--grad-audit-steps N`` (``0`` = off) additionally attaches
:class:`molix.hooks.GradientBalanceHook` at steps ``(0, N)`` and records
``artifacts/grad_balance_audit.json`` — on this single-task run that is the
checklist §2 per-group finite/nonzero-gradient table, not a multi-task verdict.

This module is a library of model / collate helpers. The operator entry is
``molexp run scripts/mm_param_learning/train.py``.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from molhub.dataset import PhalkethohMMDataset
from molpy import Frame
from tensordict import TensorDict

from molix.core.audit import TaskGradientProbe
from molix.core.hook import BaseHook
from molix.core.metrics import MAE, RMSE
from molix.core.state import display, resolve
from molix.core.steps import (
    MultiTaskEvalStep,
    MultiTaskTrainStep,
    TaskRouter,
    TaskSpec,
    batch_to,
)
from molix.core.trainer import Trainer
from molix.data import (
    BaseDataset,
    DataModule,
    MoleculeCollator,
    TaskMixtureBatchSampler,
    TotalCharge,
)
from molix.hooks import (
    CheckpointHook,
    EarlyStop,
    GradClipHook,
    GradientBalanceHook,
    MetricsHook,
)
from molpot.forcefield import ForceField, Parameterizer

# Espaloma notebook protocol pins
ESPALOMA_SPLIT_SEED = 2666
ESPALOMA_SPLIT_RATIOS = (8, 1, 1)  # train:val:test parts
ESPALOMA_ADAM_LR = 1e-4
ESPALOMA_BATCH_SIZE = 100
ESPALOMA_HIDDEN = 128
ESPALOMA_N_TORSION_TERMS = 6
ESPALOMA_GRAD_CLIP_NORM = 1.0  # was the retired loop's inline clip_grad_norm_

# Energy already converted to kcal/mol in the normalized MolHub tree
# (Hartree * 627.5094740631). Espaloma notebook reports L1 * 627.5.

# Single physical task of this protocol run. The name is the task-mixture /
# TaskRouter label; the code is what rides at ``batch["graphs", "task"]``.
TASK_ENERGY = "energy"
TASK_CODE_ENERGY = 0

#: Metric tracked for ``save_best`` / ``--early-stop-patience`` — the
#: ``MetricsHook`` eval-side MAE in kcal/mol (Espaloma's L1 report).
BEST_METRIC = ("eval", "MAE")

#: Batches replayed at every gradient-audit probe point. Fixed (and small) so
#: the two snapshots are comparable and the audit stays bounded.
GRAD_AUDIT_PROBE_BATCHES = 4


def label_frame(fr) -> Frame:
    """Label one MolHub frame with its valence families and task code.

    The ``task`` scalar rides along in ``meta`` as an **integer** so every
    collated batch is routable by :class:`~molix.core.steps.TaskRouter` (one
    task per batch — here always :data:`TASK_CODE_ENERGY`), which refuses a
    floating label rather than truncate one silently.

    Args:
        fr: A MolHub molpy ``Frame`` with ``atoms`` and ``bonds`` blocks and an
            ``mm_energy`` meta scalar.

    Returns:
        A new ``Frame`` carrying the same atoms/bonds plus ``angles`` /
        ``propers`` / ``impropers`` blocks and the ``task`` meta scalar.
    """
    atoms = fr["atoms"]
    bonds = fr["bonds"]
    from scripts.mm_param_learning.class_i_fit import _valence_blocks

    valence = _valence_blocks(fr)

    meta = {k: getattr(v, "value", v) for k, v in (getattr(fr, "meta", None) or {}).items()}
    meta["task"] = TASK_CODE_ENERGY
    # No ``total_charge`` is written here: the neutrality of this dataset is
    # declared once, where the batch tasks are mounted (``TotalCharge()``), not
    # re-stated per frame. A dataset that does publish a charge keeps it — the
    # task fills only an absent column, and the meta scalar of a frame that
    # carries one collates into exactly that column.
    out = Frame(meta=meta)
    out["atoms"] = {column: atoms[column] for column in atoms.keys()}
    out["bonds"] = {column: bonds[column] for column in bonds.keys()}
    for name, columns in valence.items():
        out[name] = {col: np.asarray(values, dtype=np.int64) for col, values in columns.items()}
    return out


_MM_COLLATOR = MoleculeCollator()


def frames_to_batch(
    frames: list, device: torch.device, forcefield: ForceField
) -> tuple[TensorDict, torch.Tensor]:
    """Pack heterogeneous conformers into one graph-aware GPU batch.

    Applies by hand the two tasks a :class:`~molix.data.DataModule` mounts
    (:func:`build_datamodule`), because this path collates without one: the
    field's own pair list — molpot consumes a placed ``pairs`` namespace and
    never enumerates one — and the neutral ``graphs.total_charge`` the charge
    model is fail-closed on.

    Args:
        frames: Conformer frames to pack.
        device: Where the batch is moved.
        forcefield: The field of the model these batches feed
            (``model.imm.template``). Its ``neighbor_list()`` is the graph, so
            the batch reaches exactly the radius that model prices — a second
            field built here would be that radius stated twice.
    """
    batch = _MM_COLLATOR([label_frame(f) for f in frames])
    for task in (forcefield.neighbor_list(), TotalCharge()):
        batch = task.execute(batch)
    batch = batch.to(device)
    return batch, batch["graphs", "mm_energy"]


def molecule_split(mol_ids: list[str], seed: int = ESPALOMA_SPLIT_SEED) -> dict[str, list[str]]:
    """Espaloma notebook: shuffle(seed) then split([8,1,1])."""
    ids = list(mol_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b, c = ESPALOMA_SPLIT_RATIOS
    total = a + b + c
    n_tr = max(1, int(round(n * a / total))) if n >= 3 else max(1, n - 2)
    n_vl = max(1, int(round(n * b / total))) if n >= 3 else 1
    if n_tr + n_vl >= n:
        n_tr = max(1, n - 2)
        n_vl = 1
    n_te = n - n_tr - n_vl
    if n_te < 1:
        n_te = 1
        n_tr = max(1, n - n_vl - n_te)
    return {
        "train": ids[:n_tr],
        "val": ids[n_tr : n_tr + n_vl],
        "test": ids[n_tr + n_vl :],
        "seed": seed,
        "ratios": [a, b, c],
        "scheme": "espaloma-original",
    }


def load_all_frames(
    data_root: Path, *, max_confs: int | None = None
) -> tuple[dict[str, list], dict[str, int]]:
    """Load every molecule from train/val/test into memory (Espaloma full scale).

    Returns:
        frames_by_mol: molecule_id → list of Frames (capped by *max_confs*).
        n_atoms: molecule_id → atom count.
    """
    by: dict[str, list] = defaultdict(list)
    n_atoms: dict[str, int] = {}
    for split in ("train", "val", "test"):
        ds = PhalkethohMMDataset(data_root, download=False, split=split)
        for i in range(len(ds)):
            fr = ds[i]
            raw_mid = fr.meta["molecule_id"]
            mid = str(getattr(raw_mid, "value", raw_mid))
            n_atoms[mid] = int(len(fr["atoms"]["number"]))
            if max_confs is None or len(by[mid]) < max_confs:
                by[mid].append(fr)
    return dict(by), n_atoms


def load_official_split(data_root: Path) -> dict[str, list[str]]:
    """Molecule-level split from molhub-normalized ``splits.json`` (seed 2666)."""
    path = Path(data_root) / "splits.json"
    raw = json.loads(path.read_text())
    return {
        "train": [str(x) for x in raw["train"]],
        "val": [str(x) for x in raw["val"]],
        "test": [str(x) for x in raw["test"]],
        "seed": int(raw.get("seed", ESPALOMA_SPLIT_SEED)),
        "ratios": list(raw.get("ratios", list(ESPALOMA_SPLIT_RATIOS))),
        "scheme": str(raw.get("scheme", "espaloma-original")),
    }


def pick_molecules(
    data_root: Path,
    n_mols: int = 0,
    max_confs: int | None = None,
    *,
    selection: str = "all",
) -> dict[str, list]:
    """Load corpus and optionally subset molecules.

    Args:
        n_mols: ``0`` or negative → all molecules. Positive → subset size.
        selection: ``all`` keeps arbitrary order; ``smallest`` ranks by atom count
            (legacy mini protocol).
    """
    by, n_atoms = load_all_frames(data_root, max_confs=max_confs)
    mids = list(by.keys())
    if n_mols is not None and n_mols > 0 and n_mols < len(mids):
        if selection == "smallest":
            ranked = sorted(n_atoms.items(), key=lambda kv: (kv[1], kv[0]))
            mids = [mid for mid, _ in ranked[:n_mols]]
        else:
            mids = sorted(mids)[:n_mols]
    return {mid: by[mid] for mid in mids}


class MMEnergyModel(nn.Module):
    """Parameterizer + OPLS-AA energy as one ``forward``.

    :class:`~molpot.forcefield.Parameterizer` returns a filled
    :class:`~molpot.forcefield.ForceField`; the molix
    ``loss_fn(predictions, batch)`` / ``MetricsHook`` / probe convention wants
    a prediction mapping. This two-line composition is the seam — it adds no
    physics and owns no state beyond the wrapped module, so the published
    checkpoint stays a bare ``Parameterizer`` state dict.

    Args:
        imm: The parameterizing module built by :func:`build_espaloma_like_model`.
    """

    def __init__(self, imm: Parameterizer) -> None:
        super().__init__()
        self.imm = imm

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Predict the graph energies of *batch*.

        Args:
            batch: Post-collate batch with ``atoms`` / ``bonds`` / valence
                namespaces and the ``("graphs", "task")`` label.

        Returns:
            ``{"energy": (B,) kcal/mol}``.
        """
        return {"energy": self.imm(batch).to_potentials().calc_energy(batch).reshape(-1)}

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Trainable parameters per Parameterizer component, for the gradient audit.

        Returns:
            Component name (``encoder`` / ``bond_head`` / … ) → its trainable
            parameters; components without any are omitted.
        """
        groups = {
            name: tuple(p for p in child.parameters() if p.requires_grad)
            for name, child in self.imm.named_children()
        }
        return {name: params for name, params in groups.items() if params}


class ConformerDataset(BaseDataset):
    """Molecule-grouped conformer view over in-memory MolHub frames.

    Flat *per conformer* for indexing, but every conformer carries its
    ``molecule_id`` as a sampling-group label, so
    :class:`~molix.data.TaskMixtureBatchSampler` can keep a molecule's
    conformer set whole inside one batch. Packing cost is
    ``frame["atoms"].nrows`` per conformer. This protocol builds no cutoff
    graph (Class-I reads covalent topology only).

    Args:
        frames_by_mol: Molecule id → its conformer frames.
        mol_ids: Molecules of this split, in the order they are laid out.
    """

    def __init__(self, frames_by_mol: dict[str, list], mol_ids: Sequence[str]) -> None:
        self._frames = [frame for mid in mol_ids for frame in frames_by_mol[mid]]
        self.group_ids: tuple[str, ...] = tuple(mid for mid in mol_ids for _ in frames_by_mol[mid])
        self.task_labels: tuple[str, ...] = (TASK_ENERGY,) * len(self._frames)

    def __len__(self) -> int:
        """Number of conformers in this split."""
        return len(self._frames)

    def __getitem__(self, idx: int) -> Frame:
        """The ``idx``-th conformer as a labelled molpy ``Frame``."""
        return label_frame(self._frames[idx])

    def atom_budget(self, batch_size: int) -> int:
        """Per-batch atom budget standing in for a conformer ``batch_size``.

        ``batch_size × max(atoms per conformer)`` keeps a batch at about
        ``batch_size`` conformers while leaving room for the largest molecule's
        whole conformer set (see the module docstring, deviation 1). That room
        is guaranteed only while ``--max-confs <= --batch-size`` (the pinned
        protocol has both at 100); above that, a molecule whose conformer set
        outgrows the budget becomes a warned singleton batch — the sampler
        never splits a unit and never drops one.

        Args:
            batch_size: The protocol's conformer batch size.

        Returns:
            Total atom cap per batch.
        """
        n_atoms = max((int(frame["atoms"].nrows) for frame in self._frames), default=0)
        return batch_size * n_atoms


class EnergyTaskSamplerSource:
    """Per-epoch molecule-grouped batch sampler for the single-task protocol.

    Implements the :class:`~molix.data.BatchSamplerSource` protocol:
    :class:`~molix.data.DataModule` calls
    ``batch_sampler(train_dataset, seed=seed + epoch)`` once per epoch, so each
    epoch reshuffles the molecules while epoch *k* stays re-derivable. The
    mixture is a single task at probability 1.0 — this run fits energies only —
    and every batch is molecule-integral, which is what makes the retired flat
    conformer pool structurally unreachable.

    Args:
        max_atoms: Per-batch atom budget (see
            :meth:`ConformerDataset.atom_budget`).
    """

    def __init__(self, *, max_atoms: int) -> None:
        self.max_atoms = max_atoms

    def batch_sampler(self, dataset: BaseDataset, *, seed: int) -> Iterable[list[int]]:
        """Build this epoch's molecule-grouped plan over *dataset*.

        Args:
            dataset: The train :class:`ConformerDataset`.
            seed: Per-epoch seed supplied by the DataModule.

        Returns:
            A sized, iterable plan of conformer-index batches.

        Raises:
            TypeError: *dataset* carries no per-conformer task / group labels.
        """
        if not isinstance(dataset, ConformerDataset):
            raise TypeError(
                "EnergyTaskSamplerSource needs a ConformerDataset (it supplies the "
                f"per-conformer task_labels / group_ids); got {type(dataset).__name__}."
            )
        return TaskMixtureBatchSampler(
            dataset,
            indices=range(len(dataset)),
            task_labels=dataset.task_labels,
            group_ids=dataset.group_ids,
            mixture={TASK_ENERGY: 1.0},
            max_atoms=self.max_atoms,
            seed=seed,
        )


def mse_energy(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Espaloma GraphMetric MSE on graph energies."""
    return torch.mean((pred - ref) ** 2)


def mse_energy_task(predictions: Mapping[str, torch.Tensor], batch: TensorDict) -> torch.Tensor:
    """The pinned objective in molix ``loss_fn(predictions, batch)`` form.

    Absolute (not QM-centered) graph-level MSE in kcal/mol — unchanged from the
    retired loop, only re-shaped for :class:`~molix.core.steps.TaskSpec`.

    Args:
        predictions: Model output carrying ``"energy"`` ``(B,)``.
        batch: Collated batch carrying ``("graphs", "mm_energy")`` ``(B,)``.

    Returns:
        Scalar MSE in ``kcal²·mol⁻²``.
    """
    return mse_energy(predictions["energy"].reshape(-1), batch["graphs", "mm_energy"].reshape(-1))


def mae_energy(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - ref)).detach())


@torch.no_grad()
def eval_split(
    model: MMEnergyModel,
    frames_by_mol: dict[str, list],
    mol_ids: list[str],
    device: torch.device,
    batch_size: int = ESPALOMA_BATCH_SIZE,
) -> dict[str, float]:
    """Topology once per molecule; evaluate all confs; report MAE/RMSE kcal/mol."""
    preds: list[torch.Tensor] = []
    refs: list[torch.Tensor] = []
    frames = [frame for mid in mol_ids for frame in frames_by_mol[mid]]
    # The graph is priced by the field this model was built on, not by a second
    # one assembled here (F21).
    forcefield = model.imm.template
    for start in range(0, len(frames), batch_size):
        batch, ref = frames_to_batch(frames[start : start + batch_size], device, forcefield)
        preds.append(model(batch)["energy"].reshape(-1))
        refs.append(ref.reshape(-1))
    pred = torch.cat(preds)
    ref = torch.cat(refs)
    return {
        "mae_kcal": mae_energy(pred, ref),
        "rmse_kcal": float(torch.sqrt(torch.mean((pred - ref) ** 2)).detach()),
        "n": int(pred.numel()),
    }


def _log_metrics(metrics_writer: object | None, row: dict) -> None:
    """Append scalars in molexp MetricsWriter format (t/k/s/v), if available."""
    if metrics_writer is None:
        return
    step = row.get("epoch")
    for key in (
        "train_mse",
        "train_mae_kcal",
        "train_rmse_kcal",
        "val_mae_kcal",
        "val_rmse_kcal",
        "elapsed_s",
        "lr",
    ):
        if key not in row:
            continue
        val = row[key]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            metrics_writer.scalar(key, float(val), step=step)


class _ProtocolJournal(BaseHook):
    """Emit one pinned-protocol history row per eval phase.

    The row schema (``epoch`` / ``train_mse`` / ``*_mae_kcal`` / ``elapsed_s``
    / ``lr`` / ``batch_size``) is this run's molexp artifact contract —
    ``artifacts/history.json``, the ``MetricsWriter`` scalars and the stdout
    JSON lines the sbatch tails — so it stays script-local glue on top of
    :class:`~molix.hooks.MetricsHook` rather than a library hook.

    ``epoch`` is the in-progress epoch index for evals fired by the step
    cadence, and ``epochs`` (all epochs completed) for the post-loop final eval;
    ``global_step`` disambiguates. ``train_mse`` covers the batches since the
    previous eval.

    Args:
        history: List the rows are appended to (returned by :func:`train`).
        lr: Adam learning rate, echoed into every row.
        batch_size: Conformer batch size, echoed into every row.
        record_root: Run directory for the legacy ``metrics/metrics.jsonl``
            dump, or ``None``.
        metrics_writer: molexp ``MetricsWriter`` (``ctx.metrics``) or ``None``.
    """

    def __init__(
        self,
        history: list[dict],
        *,
        lr: float,
        batch_size: int,
        record_root: Path | None,
        metrics_writer: object | None,
    ) -> None:
        self._history = history
        self._lr = lr
        self._batch_size = batch_size
        self._record_root = record_root
        self._metrics_writer = metrics_writer
        self._loss_sum = 0.0
        self._n_batches = 0
        self._t0 = time.time()

    def on_train_batch_end(self, trainer, state, batch, outputs) -> None:
        """Accumulate the train MSE since the previous eval (was the epoch mean)."""
        self._loss_sum += float(state["train"]["loss"])
        self._n_batches += 1

    def on_eval_step_complete(self, trainer, state) -> None:
        """Append + emit the row for the eval phase that just finished."""
        row = {
            "epoch": int(state.epoch),
            "global_step": int(state.global_step),
            "train_mse": self._loss_sum / max(self._n_batches, 1),
            "train_mae_kcal": _as_float(state["train"].get("MAE")),
            "train_rmse_kcal": _as_float(state["train"].get("RMSE")),
            "val_mae_kcal": _as_float(state["eval"].get("MAE")),
            "val_rmse_kcal": _as_float(state["eval"].get("RMSE")),
            "elapsed_s": time.time() - self._t0,
            "lr": self._lr,
            "batch_size": self._batch_size,
        }
        self._loss_sum = 0.0
        self._n_batches = 0
        self._t0 = time.time()

        self._history.append(row)
        print(json.dumps(row), flush=True)
        _log_metrics(self._metrics_writer, row)

        # Legacy free-form row dump (debug). Prefer MetricsWriter when present.
        if self._record_root is not None and self._metrics_writer is None:
            metrics_dir = self._record_root / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            with (metrics_dir / "metrics.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")


class _PatienceStopped(RuntimeError):
    """Raised by :class:`_ValPatienceStop` to unwind the Trainer loop."""


class _ValPatienceStop(BaseHook):
    """Stop training after *patience* non-improving eval phases.

    molix ships only ``EarlyStop(if_nan=True)``; promoting a general
    patience-based early stop into :mod:`molix.hooks` is its own spec (named in
    ``universal-class-i-13-grad-audit`` §"Out of scope"). Until that lands, the
    protocol's ``--early-stop-patience`` lives here and unwinds by raising,
    exactly as :class:`~molix.hooks.EarlyStop` does.

    Args:
        metric: State path of the monitored metric (lower is better).
        patience: Consecutive non-improving eval phases tolerated.
    """

    def __init__(self, metric: tuple[str, str], *, patience: int) -> None:
        self._metric = metric
        self._patience = patience
        self._best = float("inf")
        self._stale = 0

    def on_eval_step_complete(self, trainer, state) -> None:
        """Track the best value and raise once *patience* is exhausted."""
        raw = resolve(state, self._metric)
        if raw is None:
            return
        value = float(raw)
        if value < self._best:
            self._best = value
            self._stale = 0
            return
        self._stale += 1
        if self._stale >= self._patience:
            raise _PatienceStopped(
                f"{display(self._metric)} did not improve for {self._stale} eval(s); "
                f"best {self._best} kcal/mol"
            )


def _as_float(value: torch.Tensor | float | None) -> float | None:
    """Materialize a state scalar (0-d device tensor or number) as ``float``."""
    if value is None:
        return None
    return float(value)


def _probe_batches(
    datamodule: DataModule, device: torch.device, n_batches: int
) -> list[tuple[str, TensorDict]]:
    """Materialize the fixed ``(task, batch)`` stream for the gradient audit.

    Args:
        datamodule: Supplies the real train batches (same loader the run trains
            on, so the audit measures the production input path).
        device: Device the batches are moved to (the model's).
        n_batches: Upper bound on the number of batches held.

    Returns:
        ``[(task_name, batch), ...]`` — replayed identically at every probe
        point so the two snapshots are comparable.
    """
    stream: list[tuple[str, TensorDict]] = []
    for batch in datamodule.train_dataloader():
        stream.append((TASK_ENERGY, batch_to(batch, device=device)))
        if len(stream) >= n_batches:
            break
    return stream


def train(
    model: MMEnergyModel,
    datamodule: DataModule,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    record_root: Path | None,
    eval_every: int = 5,
    metrics_writer: object | None = None,
    early_stop_patience: int = 0,
    grad_audit_steps: int = 0,
) -> tuple[list[dict], dict | None]:
    """Espaloma-style Adam + MSE energy training, composed from molix primitives.

    The loop, the optimizer step, gradient clipping and best-model tracking are
    the :class:`~molix.core.trainer.Trainer`'s and its hooks'; this function
    only wires them and owns the protocol's reporting. See the module docstring
    for the two cadence deviations from the retired flat-pool loop.

    Args:
        model: The energy model under training.
        datamodule: Molecule-grouped train batches + fixed-size val batches.
        epochs: Epoch budget (``Trainer(max_epochs=...)``).
        batch_size: Conformer batch size, echoed into the history rows.
        lr: Adam learning rate (Espaloma pin ``1e-4``).
        device: Device the model and batches live on.
        record_root: molexp run directory, or ``None``.
        eval_every: Eval cadence in *epochs*, realised in optimizer steps.
        metrics_writer: molexp ``MetricsWriter`` or ``None``.
        early_stop_patience: Non-improving eval phases tolerated before
            stopping. ``0`` disables (Espaloma notebook trains to a fixed
            budget and inspects offline; we offer patience for formal runs).
        grad_audit_steps: When ``> 0``, probe per-task/per-group gradient norms
            at step ``0`` and at this step, and return the report dict.

    Returns:
        ``(history, grad_balance_audit)`` — the protocol rows, and the
        :class:`~molix.core.audit.GradientBalanceReport` dict when
        ``grad_audit_steps > 0`` (else ``None``).
    """
    steps_per_epoch = max(1, len(datamodule.train_dataloader()))
    history: list[dict] = []
    journal = _ProtocolJournal(
        history,
        lr=lr,
        batch_size=batch_size,
        record_root=record_root,
        metrics_writer=metrics_writer,
    )
    checkpoint_dir = (
        (record_root / "checkpoints") if record_root is not None else Path("checkpoints")
    )

    # One task, one batch label: MultiTask{Train,Eval}Step route through the
    # single pinned objective, which keeps the sampler's single-task invariant
    # enforced end to end instead of assumed.
    router = TaskRouter((TaskSpec(name=TASK_ENERGY, code=TASK_CODE_ENERGY, loss=mse_energy_task),))
    hooks: list[BaseHook] = [
        GradClipHook(max_norm=ESPALOMA_GRAD_CLIP_NORM),
        MetricsHook(
            [MAE(), RMSE()],
            pred_key=("predictions", "energy"),
            target_key=("graphs", "mm_energy"),
        ),
        journal,
        CheckpointHook(
            checkpoint_dir=str(checkpoint_dir),
            save_best=True,
            best_metric_name=BEST_METRIC,
            best_metric_mode="min",
        ),
        EarlyStop(if_nan=True, model=model, out_dir=checkpoint_dir),
    ]
    if early_stop_patience > 0:
        hooks.append(_ValPatienceStop(BEST_METRIC, patience=early_stop_patience))

    audit_hook: GradientBalanceHook | None = None
    if grad_audit_steps > 0:
        audit_hook = GradientBalanceHook(
            TaskGradientProbe(
                model,
                {TASK_ENERGY: mse_energy_task},
                groups=model.parameter_groups(),
                units={TASK_ENERGY: "kcal**2/mol**2"},
            ),
            _probe_batches(datamodule, device, GRAD_AUDIT_PROBE_BATCHES),
            at_steps=(0, grad_audit_steps),
        )
        hooks.append(audit_hook)

    trainer = Trainer(
        model,
        loss_fn=None,  # the routed step owns the objective
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=lr),
        train_step=MultiTaskTrainStep(router),
        eval_step=MultiTaskEvalStep(router, no_grad=True),
        hooks=hooks,
        eval_every_n_steps=max(1, eval_every * steps_per_epoch),
        device=str(device),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        trainer.train(datamodule, max_epochs=epochs)
    except _PatienceStopped as stopped:
        # Only the patience stop is a *successful* termination; a NaN abort from
        # EarlyStop (or any other RuntimeError) propagates and fails the run.
        print(
            json.dumps(
                {
                    "early_stop": True,
                    "epoch": int(trainer.state.epoch),
                    "global_step": int(trainer.state.global_step),
                    "patience": early_stop_patience,
                    "reason": str(stopped),
                }
            ),
            flush=True,
        )
    else:
        # The retired loop forced an eval on the last epoch; the step cadence
        # may not land there, so evaluate once more unless it just did.
        if trainer.state.steps_since_last_eval != 0:
            trainer.evaluate(datamodule)

    _restore_best(model, checkpoint_dir / "best.pt", device)
    audit = audit_hook.report.to_dict() if audit_hook is not None else None
    return history, audit


def _restore_best(model: MMEnergyModel, best_path: Path, device: torch.device) -> None:
    """Load ``CheckpointHook``'s ``best.pt`` weights back into *model*.

    Replaces the retired hand-rolled best-state clone: the hook already wrote
    the best-val checkpoint, so restoring is a read.

    Args:
        model: Model to load into (in place).
        best_path: ``best.pt`` written by :class:`~molix.hooks.CheckpointHook`.
        device: Map location for the payload.
    """
    if not best_path.exists():
        return
    # weights_only=False: the payload also carries optimizer / RNG state
    # (molix.core.checkpoint.Checkpoint.state_dict), and this file is ours.
    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])


def build_datamodule(
    frames_by_mol: dict[str, list],
    split: dict[str, list[str]],
    *,
    batch_size: int,
    seed: int,
    forcefield: ForceField,
) -> DataModule:
    """Molecule-grouped train batches + fixed-size val batches for the Trainer.

    Args:
        frames_by_mol: Molecule id → conformer frames.
        split: Molecule-level split (Espaloma seed 2666, 8:1:1).
        batch_size: Conformer batch size; the train side converts it to an atom
            budget (see :meth:`ConformerDataset.atom_budget`).
        seed: Base seed; the DataModule reseeds the sampler as ``seed + epoch``.
        forcefield: The field of the model these batches feed
            (``model.imm.template``, i.e. ``build_espaloma_like_model``'s
            ``.template``). It owns both post-collate facts: the radius its
            pair styles are priced at, and — with ``TotalCharge`` — the neutral
            net charge its charge model is fail-closed on.

    Returns:
        A :class:`~molix.data.DataModule` satisfying ``DataModuleProtocol``.
    """
    train_ds = ConformerDataset(frames_by_mol, split["train"])
    val_ds = ConformerDataset(frames_by_mol, split["val"])
    return DataModule(
        train_ds,
        val_ds,
        batch_tasks=[forcefield.neighbor_list(), TotalCharge()],
        batch_size=batch_size,
        batch_sampler_source=EnergyTaskSamplerSource(max_atoms=train_ds.atom_budget(batch_size)),
        num_workers=0,  # frames are already in memory; no worker round trip
        pin_memory=False,
        persistent_workers=False,
        seed=seed,
    )


def _resolve_molexp_run(
    *,
    run_dir: Path | None,
    workspace: Path | None,
    project: str,
    experiment: str,
    run_id: str | None,
    record_root: Path | None,
):
    """Open a molexp Run from ``--run`` dir, workspace ids, or record_root.

    Returns ``(run, record_root)`` or ``(None, record_root)`` when molexp is
    unavailable / not requested.
    """
    try:
        from molexp.workspace import Workspace
        from molexp.workspace.run import Run
    except ImportError:
        return None, record_root

    if run_dir is not None:
        try:
            run = Run.load(Path(run_dir).expanduser().resolve())
            return run, Path(run.path())
        except Exception:
            return None, Path(run_dir)

    if workspace is not None and run_id is not None:
        ws = Workspace.load(Path(workspace).expanduser().resolve())
        run = ws.project(project).experiment(experiment).run(run_id)
        return run, Path(run.path())

    if record_root is not None:
        try:
            run = Run.load(Path(record_root).expanduser().resolve())
            return run, Path(run.path())
        except Exception:
            return None, Path(record_root)

    return None, record_root
