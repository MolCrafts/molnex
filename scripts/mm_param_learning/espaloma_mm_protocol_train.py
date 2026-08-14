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

Example::

    export PY=/path/to/python  # torch+molpy env
    export PYTHONPATH=src:../molhub/src
    $PY scripts/mm_param_learning/espaloma_mm_protocol_train.py \\
      --data-root .../phalkethoh-mm-small-normalized \\
      --epochs 200 --device cpu \\
      --record-root .../experiments/espaloma-mm-protocol-mini/runs/run-001
"""

from __future__ import annotations

import argparse
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
from tensordict import TensorDict

from molix.config import config
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
    TargetSchema,
    TaskMixtureBatchSampler,
)
from molix.hooks import (
    CheckpointHook,
    EarlyStop,
    GradClipHook,
    GradientBalanceHook,
    MetricsHook,
)
from molpot.charge import (
    AtomicChargeModel,
    ChiEtaHead,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
)
from molpot.composition.heads import LJParameterHead
from molpot.composition.mm_heads import (
    AngleParamHead,
    BondParamHead,
    ProperTorsionParamHead,
)
from molpot.forcefield import ClassicalIMM, ClassIForceField
from molrep.chem.encoder import ChemEncoder

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


def enumerate_angles(
    atomi: np.ndarray, atomj: np.ndarray, n_atoms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """i-j-k angles from undirected bonds (j central)."""
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for i, j in zip(atomi.tolist(), atomj.tolist(), strict=True):
        adj[i].append(j)
        adj[j].append(i)
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []
    for j in range(n_atoms):
        nbrs = adj[j]
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                ii.append(nbrs[a])
                jj.append(j)
                kk.append(nbrs[b])
    if not ii:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    return (
        np.asarray(ii, dtype=np.int64),
        np.asarray(jj, dtype=np.int64),
        np.asarray(kk, dtype=np.int64),
    )


def enumerate_propers(
    atomi: np.ndarray, atomj: np.ndarray, n_atoms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """i-j-k-l propers with central bond j-k."""
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    edges = list(zip(atomi.tolist(), atomj.tolist(), strict=True))
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []
    ll: list[int] = []
    seen: set[tuple[int, int]] = set()
    for a, b in edges:
        for j, k in ((a, b), (b, a)):
            if (j, k) in seen:
                continue
            seen.add((j, k))
            seen.add((k, j))
            for i in adj[j]:
                if i == k:
                    continue
                for l in adj[k]:
                    if l == j or l == i:
                        continue
                    ii.append(i)
                    jj.append(j)
                    kk.append(k)
                    ll.append(l)
    if not ii:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, z
    return (
        np.asarray(ii, dtype=np.int64),
        np.asarray(jj, dtype=np.int64),
        np.asarray(kk, dtype=np.int64),
        np.asarray(ll, dtype=np.int64),
    )


def frame_to_sample(fr) -> dict:
    """Convert one MolHub frame to the pre-collate molecular schema.

    The ``task`` target rides along as a graph-level integer label so every
    collated batch is routable by :class:`~molix.core.steps.TaskRouter` (one
    task per batch — here always :data:`TASK_CODE_ENERGY`).
    """
    atoms = fr["atoms"]
    bonds = fr["bonds"]
    z = torch.as_tensor(np.asarray(atoms["number"]), dtype=torch.long)
    pos = torch.stack(
        [torch.as_tensor(np.asarray(atoms[c]), dtype=torch.float64) for c in ("x", "y", "z")],
        dim=-1,
    )
    atomi = np.asarray(bonds["atomi"], dtype=np.int64)
    atomj = np.asarray(bonds["atomj"], dtype=np.int64)
    ang_i, ang_j, ang_k = enumerate_angles(atomi, atomj, int(z.shape[0]))
    pr_i, pr_j, pr_k, pr_l = enumerate_propers(atomi, atomj, int(z.shape[0]))
    sample = {
        "Z": z,
        "pos": pos,
        "bond_index": torch.as_tensor(np.stack([atomi, atomj]), dtype=torch.long),
        "targets": {
            "mm_energy": float(fr.meta["mm_energy"]),
            "task": TASK_CODE_ENERGY,
        },
    }
    if ang_i.size:
        sample["angles"] = {
            "atomi": torch.as_tensor(ang_i),
            "atomj": torch.as_tensor(ang_j),
            "atomk": torch.as_tensor(ang_k),
        }
    if pr_i.size:
        sample["propers"] = {
            "atomi": torch.as_tensor(pr_i),
            "atomj": torch.as_tensor(pr_j),
            "atomk": torch.as_tensor(pr_k),
            "atoml": torch.as_tensor(pr_l),
        }
    return sample


_MM_TARGET_SCHEMA = TargetSchema(graph_level=frozenset({"mm_energy", "task"}))
_MM_COLLATOR = MoleculeCollator(_MM_TARGET_SCHEMA)


def frames_to_batch(frames: list, device: torch.device) -> tuple[TensorDict, torch.Tensor]:
    """Pack heterogeneous conformers into one graph-aware GPU batch."""
    batch = _MM_COLLATOR([frame_to_sample(frame) for frame in frames]).to(device)
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
            mid = str(fr.meta["molecule_id"])
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


def build_espaloma_like_model(
    *,
    energy_scope: str = "bonded",
    hidden: int = ESPALOMA_HIDDEN,
    n_blocks: int = 5,
) -> ClassicalIMM:
    """ChemEncoder + Class-I heads.

    Args:
        energy_scope:
            * ``bonded`` — bond/angle/proper only (legacy mini baseline)
            * ``full`` — + EEM charges + LJ + Coulomb (open Class-I NB)
    """
    if energy_scope not in ("bonded", "full"):
        raise ValueError(f"energy_scope must be bonded|full, got {energy_scope!r}")

    encoder = ChemEncoder(
        atom_dim=hidden,
        bond_dim=hidden,
        angle_dim=hidden,
        proper_dim=hidden,
        improper_dim=hidden,
        hidden_dim=hidden,
        n_blocks=n_blocks,
        local_attention=False,
    )
    ff = ClassIForceField(name="class-i-learned")
    if energy_scope == "full":
        # Cutoff lives on the Potential (via style options), not on Potentials.
        ff.def_pairstyle("lj/cut", cutoff=10.0)
        ff.def_pairstyle("coulomb")

    atom_head = None
    charge_model = None
    if energy_scope == "full":
        atom_head = LJParameterHead(feature_dim=hidden, hidden_dim=hidden)
        charge_model = AtomicChargeModel(
            ChiEtaHead(feature_dim=hidden, hidden_dim=hidden),
            ElectronegativityEqualization(DiagonalHardnessKernel()),
        )

    return ClassicalIMM(
        encoder,
        forcefield=ff,
        bond_head=BondParamHead(feature_dim=hidden, hidden_dim=hidden),
        angle_head=AngleParamHead(feature_dim=hidden, hidden_dim=hidden),
        proper_head=ProperTorsionParamHead(
            feature_dim=hidden,
            hidden_dim=hidden,
            n_terms=ESPALOMA_N_TORSION_TERMS,
            periodicity=tuple(range(1, ESPALOMA_N_TORSION_TERMS + 1)),
        ),
        atom_head=atom_head,
        charge_model=charge_model,
    ).double()


class MMEnergyModel(nn.Module):
    """Class-I parameterization + Class-I energy as one ``forward``.

    :class:`~molpot.forcefield.ClassicalIMM` returns a filled ``ForceField``;
    the molix ``loss_fn(predictions, batch)`` / ``MetricsHook`` / probe
    convention wants a prediction mapping. This two-line composition is the
    seam — it adds no physics and owns no state beyond the wrapped module, so
    the published checkpoint stays a bare ``ClassicalIMM`` state dict.

    Args:
        imm: The parameterizing module built by :func:`build_espaloma_like_model`.
    """

    def __init__(self, imm: ClassicalIMM) -> None:
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
        """Trainable parameters per Class-I component, for the gradient audit.

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
    conformer set whole inside one batch. ``atom_counts`` is the per-conformer
    cost vector the token packer needs; ``edge_counts`` is zero because this
    protocol builds no cutoff graph (Class-I reads covalent topology only).

    Args:
        frames_by_mol: Molecule id → its conformer frames.
        mol_ids: Molecules of this split, in the order they are laid out.
    """

    def __init__(self, frames_by_mol: dict[str, list], mol_ids: Sequence[str]) -> None:
        self._frames = [frame for mid in mol_ids for frame in frames_by_mol[mid]]
        self.group_ids: tuple[str, ...] = tuple(mid for mid in mol_ids for _ in frames_by_mol[mid])
        self.task_labels: tuple[str, ...] = (TASK_ENERGY,) * len(self._frames)
        self.atom_counts = torch.tensor(
            [int(len(frame["atoms"]["number"])) for frame in self._frames], dtype=torch.long
        )
        self.edge_counts = torch.zeros_like(self.atom_counts)
        self.target_schema = _MM_TARGET_SCHEMA

    def __len__(self) -> int:
        """Number of conformers in this split."""
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        """The ``idx``-th conformer in the pre-collate molecular schema."""
        return frame_to_sample(self._frames[idx])

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
        return batch_size * int(self.atom_counts.max())


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
    for start in range(0, len(frames), batch_size):
        batch, ref = frames_to_batch(frames[start : start + batch_size], device)
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
) -> DataModule:
    """Molecule-grouped train batches + fixed-size val batches for the Trainer.

    Args:
        frames_by_mol: Molecule id → conformer frames.
        split: Molecule-level split (Espaloma seed 2666, 8:1:1).
        batch_size: Conformer batch size; the train side converts it to an atom
            budget (see :meth:`ConformerDataset.atom_budget`).
        seed: Base seed; the DataModule reseeds the sampler as ``seed + epoch``.

    Returns:
        A :class:`~molix.data.DataModule` satisfying ``DataModuleProtocol``.
    """
    train_ds = ConformerDataset(frames_by_mol, split["train"])
    val_ds = ConformerDataset(frames_by_mol, split["val"])
    return DataModule(
        train_ds,
        val_ds,
        target_schema=_MM_TARGET_SCHEMA,
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


def main(argv: list[str] | None = None) -> int:
    from scripts.mm_param_learning.constants import (
        DATASET_COORDINATE_PRIMARY,
        DEFAULT_NORMALIZED_DATA_ROOT,
        DEFAULT_WORKSPACE_ROOT,
        PROJECT_SLUG,
        PROTOCOL_EXPERIMENT_SLUG,
        PROTOCOL_RUN_ID,
    )

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_NORMALIZED_DATA_ROOT,
        help="Normalized PhAlkEthOH tree (molhub convert output)",
    )
    p.add_argument(
        "--n-mols",
        type=int,
        default=0,
        help="Molecule subset size; 0 = all molecules (Espaloma MM-small full = 173)",
    )
    p.add_argument(
        "--max-confs",
        type=int,
        default=100,
        help="Cap confs/mol (Espaloma MM-small uses 100)",
    )
    p.add_argument(
        "--selection",
        choices=("all", "smallest"),
        default="all",
        help="How to choose --n-mols subset (ignored when n_mols=0)",
    )
    p.add_argument(
        "--use-official-splits",
        action="store_true",
        default=True,
        help="Use data_root/splits.json (Espaloma seed 2666); default on",
    )
    p.add_argument(
        "--no-official-splits",
        action="store_false",
        dest="use_official_splits",
        help="Re-shuffle with --seed instead of splits.json",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Espaloma notebook budget is 10000; formal default 1000",
    )
    p.add_argument("--batch-size", type=int, default=ESPALOMA_BATCH_SIZE)
    p.add_argument("--hidden", type=int, default=ESPALOMA_HIDDEN)
    p.add_argument("--n-blocks", type=int, choices=(4, 5, 6), default=5)
    p.add_argument("--lr", type=float, default=ESPALOMA_ADAM_LR)
    p.add_argument("--seed", type=int, default=ESPALOMA_SPLIT_SEED)
    p.add_argument("--eval-every", type=int, default=5, help="Full train/val MAE every N epochs")
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=50,
        help="Stop after this many non-improving evals (0=off). Espaloma uses offline inspect.",
    )
    p.add_argument(
        "--grad-audit-steps",
        type=int,
        default=0,
        help=(
            "Probe per-group gradient norms at step 0 and step N (0=off); "
            "writes artifacts/grad_balance_audit.json"
        ),
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--run",
        type=Path,
        default=None,
        help="molexp run directory (preferred; enables Run lifecycle + MetricsWriter)",
    )
    p.add_argument(
        "--record-root",
        type=Path,
        default=None,
        help="Alias of --run (run directory)",
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--energy-scope",
        choices=("bonded", "full"),
        default="bonded",
        help="bonded: valence only; full: + EEM charges + LJ + Coulomb",
    )
    p.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="molexp workspace root (with --run-id; prefer --run)",
    )
    p.add_argument("--project", default=PROJECT_SLUG, help="molexp project id")
    p.add_argument(
        "--experiment",
        "--experiment-slug",
        dest="experiment",
        default=PROTOCOL_EXPERIMENT_SLUG,
        help="molexp experiment id",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help=f"molexp run id (default with --workspace: {PROTOCOL_RUN_ID})",
    )
    p.add_argument(
        "--dataset-coordinate",
        default=DATASET_COORDINATE_PRIMARY,
        help="MolHub coordinate (audit; data-root is local materialization)",
    )
    p.add_argument(
        "--no-harvest",
        action="store_true",
        help="Skip post-run harvest_run Finding KnowledgeItem",
    )
    args = p.parse_args(argv)

    if args.run is None and args.record_root is not None:
        args.run = args.record_root
    if args.workspace is not None and args.run_id is None and args.run is None:
        args.run_id = PROTOCOL_RUN_ID
    if args.workspace is None and args.run_id is not None and args.run is None:
        args.workspace = DEFAULT_WORKSPACE_ROOT

    molexp_run, record_root = _resolve_molexp_run(
        run_dir=args.run,
        workspace=args.workspace,
        project=args.project,
        experiment=args.experiment,
        run_id=args.run_id,
        record_root=args.record_root,
    )
    if args.run is not None and molexp_run is None:
        record_root = Path(args.run)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # Class-I kernels and the MM energy labels are float64; this makes the
    # dataloader's collate cast match the model instead of silently down-casting
    # positions to float32 (must precede DataModule construction).
    config.set_precision("fp64")

    frames_by_mol = pick_molecules(
        args.data_root,
        n_mols=args.n_mols,
        max_confs=args.max_confs,
        selection=args.selection,
    )
    mol_ids = list(frames_by_mol.keys())
    if args.use_official_splits:
        official = load_official_split(args.data_root)
        # Intersect with loaded mols (subset runs stay consistent)
        have = set(mol_ids)
        split = {
            "train": [m for m in official["train"] if m in have],
            "val": [m for m in official["val"] if m in have],
            "test": [m for m in official["test"] if m in have],
            "seed": official["seed"],
            "ratios": official["ratios"],
            "scheme": official["scheme"],
        }
        # Drop molecules not in any split (should not happen)
        keep = set(split["train"]) | set(split["val"]) | set(split["test"])
        frames_by_mol = {m: frames_by_mol[m] for m in mol_ids if m in keep}
        mol_ids = list(frames_by_mol.keys())
    else:
        split = molecule_split(mol_ids, seed=args.seed)
    n_atoms = {mid: int(len(frames_by_mol[mid][0]["atoms"]["number"])) for mid in mol_ids}

    if args.energy_scope == "full":
        model_meta = {
            "encoder": f"ChemEncoder dim={args.hidden}, n_blocks={args.n_blocks}",
            "heads": "Bond+Angle+Proper(n_terms=6)+LJParameterHead+ChiEtaHead",
            "charge": "ElectronegativityEqualization(DiagonalHardnessKernel)",
            "energy": "ForceField.to_potentials Class-I (bonded+LJ+Coulomb open)",
            "gap_vs_literature": [
                "EEM not numerically audited vs espaloma ChargeEquilibrium",
                "open Coulomb (no PME default); optional cell→Ewald/PME",
                "ChemEncoder ≠ SAGE×3 backbone",
            ],
        }
    else:
        model_meta = {
            "encoder": f"ChemEncoder dim={args.hidden}, n_blocks={args.n_blocks}",
            "heads": "Bond+Angle+Proper(n_terms=6)",
            "energy": "ForceField.to_potentials Class-I (bonded only)",
            "omitted_vs_espaloma": [
                "Charge equilibration (EEM)",
                "nonbonded LJ/Coulomb",
            ],
        }

    meta = {
        "protocol": "espaloma-mm-small-notebook",
        "references": ["arXiv:2010.01196", "DOI:10.1039/D2SC02739A"],
        "split_seed": args.seed,
        "split_ratios_parts": list(ESPALOMA_SPLIT_RATIOS),
        "adam_lr": args.lr,
        "batch_size": args.batch_size,
        "energy_scope": args.energy_scope,
        "loss": "MSE(E_pred, E_ref) graph-level absolute kcal/mol",
        "metric_report": "MAE/RMSE kcal/mol (labels already kcal/mol)",
        "model": model_meta,
        "molhub": {
            "dataset_coordinate": args.dataset_coordinate,
            "data_root": str(args.data_root),
        },
        "subset": {
            "n_mols": len(mol_ids),
            "mol_ids": mol_ids,
            "n_atoms": n_atoms,
            "confs_per_mol": {mid: len(frames_by_mol[mid]) for mid in mol_ids},
            "selection": (
                "all molecules"
                if args.n_mols <= 0
                else f"{args.n_mols} molecules ({args.selection})"
            ),
            "max_confs": args.max_confs,
            "hidden": args.hidden,
            "n_blocks": args.n_blocks,
            "n_torsion_terms": ESPALOMA_N_TORSION_TERMS,
        },
        "split": {k: v for k, v in split.items() if k != "seed"},
        "batching": (
            "molecule-grouped TaskMixtureBatchSampler "
            f"(mixture {{{TASK_ENERGY!r}: 1.0}}, atom budget = batch_size x "
            "max atoms/conformer); conformer count per batch ~ batch_size"
        ),
        "early_stop_patience": args.early_stop_patience,
        "grad_audit_steps": args.grad_audit_steps,
        "project": args.project,
        "experiment": args.experiment,
        "run_id": getattr(molexp_run, "id", args.run_id),
    }
    print(json.dumps({"setup": meta}, indent=2), flush=True)

    def _prepare_record(root: Path | None) -> None:
        if root is None:
            return
        root.mkdir(parents=True, exist_ok=True)
        (root / "artifacts").mkdir(exist_ok=True)
        (root / "metrics").mkdir(exist_ok=True)
        (root / "artifacts" / "setup.json").write_text(json.dumps(meta, indent=2) + "\n")

    def _finish_artifacts(
        root: Path | None,
        history: list,
        final: dict,
        model: MMEnergyModel,
        audit: dict | None,
    ) -> Path | None:
        if root is None:
            return None
        (root / "artifacts" / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        (root / "artifacts" / "final.json").write_text(json.dumps(final, indent=2) + "\n")
        if audit is not None:
            (root / "artifacts" / "grad_balance_audit.json").write_text(
                json.dumps(audit, indent=2) + "\n"
            )
        ckpt = args.checkpoint or (root / "artifacts" / "model.pt")
        # Publish the bare ClassicalIMM state dict — the script-local energy
        # wrapper is composition, not part of the parameterization.
        torch.save({"model": model.imm.state_dict(), "meta": meta, "final": final}, ckpt)
        return Path(ckpt)

    def _run_body(
        metrics_writer: object | None,
    ) -> tuple[list, dict, MMEnergyModel, dict | None]:
        model = MMEnergyModel(
            build_espaloma_like_model(
                energy_scope=args.energy_scope,
                hidden=args.hidden,
                n_blocks=args.n_blocks,
            )
        ).to(device)
        datamodule = build_datamodule(
            frames_by_mol,
            split,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        history, audit = train(
            model,
            datamodule,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            record_root=record_root,
            eval_every=args.eval_every,
            metrics_writer=metrics_writer,
            early_stop_patience=args.early_stop_patience,
            grad_audit_steps=args.grad_audit_steps,
        )
        te = eval_split(model, frames_by_mol, split["test"], device, batch_size=args.batch_size)
        val_rows = [h for h in history if "val_mae_kcal" in h]
        final = {
            "best_val_mae_kcal": min(h["val_mae_kcal"] for h in val_rows) if val_rows else None,
            "best_val_epoch": min(val_rows, key=lambda h: h["val_mae_kcal"])["epoch"]
            if val_rows
            else None,
            "test_mae_kcal": te["mae_kcal"],
            "test_rmse_kcal": te["rmse_kcal"],
            "test_n": te["n"],
            "epochs": args.epochs,
            "energy_scope": args.energy_scope,
            "dataset_coordinate": args.dataset_coordinate,
        }
        print(json.dumps({"final": final}), flush=True)
        return history, final, model, audit

    if molexp_run is not None:
        _prepare_record(record_root)
        with molexp_run.start() as ctx:
            history, final, model, audit = _run_body(ctx.metrics)
            ckpt = _finish_artifacts(record_root, history, final, model, audit)
            ctx.set_result("final", final)
            ctx.set_result("setup", meta)
            if ckpt is not None:
                ctx.set_result("checkpoint", str(ckpt))
            if audit is not None:
                # E4/E5 gradient-balance evidence; the report dict is plain JSON.
                ctx.set_result("grad_balance_audit", audit)
            ctx.mark_succeeded()
        if not args.no_harvest:
            try:
                from molexp.workspace import harvest_run

                narrative = (
                    f"Class-I MM protocol mini ({args.energy_scope}): "
                    f"best val MAE {final.get('best_val_mae_kcal')} kcal/mol, "
                    f"test MAE {final.get('test_mae_kcal')} kcal/mol on "
                    f"{args.dataset_coordinate}."
                )
                item = harvest_run(
                    molexp_run,
                    kind="Finding",
                    narrative=narrative,
                    created_by="mm-param-learning/espaloma_mm_protocol_train",
                    results={k: v for k, v in final.items() if v is not None},
                )
                print(json.dumps({"harvest": item.name, "status": molexp_run.status}), flush=True)
            except Exception as exc:  # noqa: BLE001 — harvest is best-effort after train
                print(json.dumps({"harvest_error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 0

    # Standalone path (no molexp Run)
    if record_root is not None:
        (record_root / "metrics" / "metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (record_root / "metrics" / "metrics.jsonl").write_text("")
    _prepare_record(record_root)
    try:
        from molexp.workspace.metrics import MetricsWriter

        writer = MetricsWriter(record_root) if record_root is not None else None
    except ImportError:
        writer = None
    history, final, model, audit = _run_body(writer)
    _finish_artifacts(record_root, history, final, model, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
