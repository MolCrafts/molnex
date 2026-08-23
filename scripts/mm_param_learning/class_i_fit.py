"""Class-I fit, composed from molix primitives.

Predicted energy is Class-I plus atomic dressing
``E = E_MM + Σ_i ε[Z_i]``. ε is fitted by MolHub ``AtomicDress`` and looked
up by ``molpot.heads.energy.AtomicReferenceEnergy`` (MACE
``AtomicEnergiesBlock``). Energy MSE is on that total vs DFT. Forces are
Cartesian MSE (dressing is coordinate-free). Unit weight on the two MSEs.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.batch import DataMixture, DataModule, TargetSchema
from molix.core.hook import BaseHook
from molix.core.steps import MultiTaskEvalStep, MultiTaskTrainStep, TaskRouter, TaskSpec
from molix.core.trainer import Trainer
from molix.hooks import CheckpointHook, GradClipHook, Log, MolRecMetricsHook, StepSpeedHook
from molix.store.frame_dataset import FrameDataset
from molix.store.frame_sample import FrameSample
from molix.units import units
from molpot.heads.energy import AtomicReferenceEnergy
from scripts.mm_param_learning.espaloma_mm_protocol_train import (
    build_espaloma_like_model,
    enumerate_angles,
    enumerate_impropers,
    enumerate_propers,
)

TASK_ENERGY = "energy"
TASK_CODE_ENERGY = 0

# OMol25 publishes eV / eV·Å⁻¹; Class-I training is kcal/mol (LAMMPS real).
_EV_TO_KCAL = float((1.0 * units.eV).to(units.kilocalorie_per_mole).magnitude)
_EV_PER_A_TO_KCAL_PER_A = _EV_TO_KCAL

_FIT_SCHEMA = TargetSchema(
    graph_level=frozenset({"energy", "task", "group"}),
    atom_level=frozenset({"forces"}),
)


def force_mse(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """MSE of Cartesian forces."""
    return (pred - ref).square().mean()


def force_mae(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """MAE of Cartesian forces."""
    return (pred - ref).abs().mean()


class _LabeledFrames:
    """FrameDataset as pre-collate dicts with task/group graph labels."""

    def __init__(self, dataset: FrameDataset) -> None:
        self.dataset = dataset
        codes = {gid: i for i, gid in enumerate(dict.fromkeys(dataset.group_ids))}
        self._codes = tuple(codes[gid] for gid in dataset.group_ids)
        self.task_labels = dataset.task_labels
        self.group_ids = dataset.group_ids
        self.atom_counts = dataset.atom_counts
        self.edge_counts = dataset.edge_counts
        self.target_schema = _FIT_SCHEMA

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], ...]:
        return self._samples_at(idx)

    def _samples_at(self, idx: int) -> tuple[dict[str, Any], ...]:
        code = float(self._codes[idx])
        return tuple(label_sample(frame, group=code) for frame in self.dataset[idx])


def _valence_families(
    atomi: np.ndarray | None,
    atomj: np.ndarray | None,
    n_atoms: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Angle / proper / improper columns; empty when there are no bonds."""
    empty = torch.zeros(0, dtype=torch.long)
    ang_i = ang_j = ang_k = empty
    pr_i = pr_j = pr_k = pr_l = empty
    im_i = im_j = im_k = im_l = empty
    if atomi is not None and atomj is not None and int(atomi.shape[0]) > 0:
        ang_i, ang_j, ang_k = enumerate_angles(atomi, atomj, n_atoms)
        pr_i, pr_j, pr_k, pr_l = enumerate_propers(atomi, atomj, n_atoms)
        im_i, im_j, im_k, im_l = enumerate_impropers(atomi, atomj, n_atoms)
    angles = {
        "atomi": torch.as_tensor(ang_i, dtype=torch.long),
        "atomj": torch.as_tensor(ang_j, dtype=torch.long),
        "atomk": torch.as_tensor(ang_k, dtype=torch.long),
    }
    propers = {
        "atomi": torch.as_tensor(pr_i, dtype=torch.long),
        "atomj": torch.as_tensor(pr_j, dtype=torch.long),
        "atomk": torch.as_tensor(pr_k, dtype=torch.long),
        "atoml": torch.as_tensor(pr_l, dtype=torch.long),
    }
    impropers = {
        "atomi": torch.as_tensor(im_i, dtype=torch.long),
        "atomj": torch.as_tensor(im_j, dtype=torch.long),
        "atomk": torch.as_tensor(im_k, dtype=torch.long),
        "atoml": torch.as_tensor(im_l, dtype=torch.long),
    }
    return angles, propers, impropers


def label_sample(frame: Any, *, group: float) -> dict[str, Any]:
    """One frame as a pre-collate dict with task/group and valence columns.

    Angles/propers/impropers are always present (empty if there are no bonds)
    so a packed batch never mixes missing and present valence families.
    """
    sample = FrameSample(frame).as_dict()
    atomi = atomj = None
    if "bond_index" in sample:
        atomi = np.asarray(sample["bond_index"][0])
        atomj = np.asarray(sample["bond_index"][1])
    sample["angles"], sample["propers"], sample["impropers"] = _valence_families(
        atomi, atomj, int(sample["Z"].shape[0])
    )
    targets = sample.setdefault("targets", {})
    targets["task"] = TASK_CODE_ENERGY
    targets["group"] = float(group)
    return sample


def _sample_from_arrays(
    *,
    numbers: np.ndarray,
    positions: np.ndarray,
    energy: float,
    forces: np.ndarray,
    atomi: np.ndarray,
    atomj: np.ndarray,
    group: float,
) -> dict[str, Any]:
    """Labeled sample from raw arrays — the fast path for LMDB records.

    Produces exactly the dict :func:`label_sample` would from an equivalent
    molpy Frame, without building the Frame or its element-symbol table.
    """
    bonded = int(atomi.shape[0]) > 0
    sample: dict[str, Any] = {
        "Z": torch.as_tensor(numbers, dtype=torch.long),
        "pos": torch.as_tensor(positions, dtype=torch.float32),
    }
    if bonded:
        sample["bond_index"] = torch.stack(
            [
                torch.as_tensor(atomi, dtype=torch.long),
                torch.as_tensor(atomj, dtype=torch.long),
            ],
            dim=0,
        )
    sample["angles"], sample["propers"], sample["impropers"] = _valence_families(
        atomi if bonded else None, atomj if bonded else None, int(numbers.shape[0])
    )
    sample["targets"] = {
        "energy": float(energy),
        "forces": torch.as_tensor(forces, dtype=torch.float32),
        "task": TASK_CODE_ENERGY,
        "group": float(group),
    }
    return sample


def _frame_atomic_numbers(frame: Any) -> np.ndarray:
    atoms = frame["atoms"]
    if "number" not in atoms:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(atoms["number"], dtype=np.int64)


def fit_atomic_reference(records: Sequence) -> AtomicReferenceEnergy:
    """Least-squares ``E0[Z]`` via MolHub ``AtomicDress``; lookup is molpot's table.

    Solves ``counts @ ε ≈ E_DFT`` on train frames. The table is frozen
    (MACE ``AtomicEnergiesBlock``). Class-I does not grow a second ε module.
    """
    from molhub.dataset.transforms import AtomicDress

    frames: list[Any] = []
    elements: set[int] = set()
    for record in records:
        for frame in record:
            numbers = _frame_atomic_numbers(frame)
            elements.update(int(z) for z in numbers if int(z) > 0)
            meta = getattr(frame, "meta", None) or {}
            if "energy" in meta and numbers.size:
                frames.append(frame)
    max_z = max(elements) if elements else 0
    max_z = max(max_z, 94)
    if not frames or not elements:
        return AtomicReferenceEnergy(atomic_energies=torch.zeros(max_z + 1))
    dress = AtomicDress(elements=sorted(elements), target_key="energy")
    dress.fit(frames)
    table = [float(dress.atomic_energies.get(z, 0.0)) for z in range(max_z + 1)]
    return AtomicReferenceEnergy(atomic_energies=table)


def _graph_count(batch: TensorDict, atom_batch: torch.Tensor) -> int:
    graphs = batch.get("graphs")
    if graphs is not None and "num_atoms" in graphs.keys():
        return int(graphs["num_atoms"].shape[0])
    if atom_batch.numel() == 0:
        return 1
    return int(atom_batch.max().item()) + 1


class ForceAwareModel(nn.Module):
    """Parameterizer energy and forces as one prediction mapping."""

    def __init__(self, imm: nn.Module, dressing: AtomicReferenceEnergy) -> None:
        super().__init__()
        self.imm = imm
        self.dressing = dressing

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        pots = self.imm(batch).to_potentials()
        infer = not torch.is_grad_enabled()
        with torch.enable_grad():
            energy, forces = pots.calc_energy_forces(batch)
        atoms = batch.get("atoms")
        if atoms is not None and "Z" in atoms.keys():
            atom_batch = atoms.get("batch")
            if atom_batch is None:
                e0 = self.dressing(atoms["Z"]).sum()
            else:
                e0 = self.dressing.pooled(atoms["Z"], atom_batch, _graph_count(batch, atom_batch))
            energy = energy.reshape(-1) + e0.reshape(-1)
        if infer:
            energy = energy.detach()
            forces = forces.detach()
        return {"energy": energy, "forces": forces}

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        groups = {
            name: tuple(p for p in child.parameters() if p.requires_grad)
            for name, child in self.imm.named_children()
        }
        return {name: params for name, params in groups.items() if params}


def _graph_tensor(batch: TensorDict, name: str) -> torch.Tensor | None:
    graphs = batch.get("graphs")
    if graphs is None or name not in graphs.keys():
        return None
    return graphs[name]


def energy_mse(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """MSE of graph energies (kcal/mol)."""
    return (pred.reshape(-1) - ref.reshape(-1)).square().mean()


def energy_mae(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """MAE of graph energies (kcal/mol)."""
    return (pred.reshape(-1) - ref.reshape(-1)).abs().mean()


def energy_force_terms(
    predictions: Mapping[str, torch.Tensor], batch: TensorDict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dressed total-energy MSE and force MSE. Unit weights, no molecule mean."""
    energy_term = predictions["energy"].new_zeros(())
    energy = _graph_tensor(batch, "energy")
    if energy is not None:
        energy_term = energy_mse(predictions["energy"], energy)
    force_term = predictions["energy"].new_zeros(())
    atoms = batch.get("atoms")
    if atoms is not None and "forces" in atoms.keys() and "forces" in predictions:
        force_term = force_mse(predictions["forces"], atoms["forces"])
    return energy_term, force_term


def train_objective(predictions: Mapping[str, torch.Tensor], batch: TensorDict) -> torch.Tensor:
    """Dressed total-energy MSE + force MSE. Unit weights."""
    energy_term, force_term = energy_force_terms(predictions, batch)
    return energy_term + force_term


def val_omol_mae(predictions: Mapping[str, torch.Tensor], batch: TensorDict) -> torch.Tensor:
    """Dressed energy MAE (kcal/mol) + force MAE (kcal mol⁻¹ Å⁻¹)."""
    energy_term = predictions["energy"].new_zeros(())
    energy = _graph_tensor(batch, "energy")
    if energy is not None:
        energy_term = energy_mae(predictions["energy"], energy)
    force_term = predictions["energy"].new_zeros(())
    atoms = batch.get("atoms")
    if atoms is not None and "forces" in atoms.keys() and "forces" in predictions:
        force_term = force_mae(predictions["forces"], atoms["forces"])
    return energy_term + force_term


_GROUP_META = ("scan_id", "molecule_id", "source_record_id")


def _require_group(frame: Any) -> str:
    """Molecule identity for the train/val split. Refuses a silent ``ungrouped`` merge."""
    meta = getattr(frame, "meta", None) or {}
    for key in _GROUP_META:
        if key in meta:
            return str(meta[key])
    raise ValueError(
        "Class-I frame has no scan_id/molecule_id/source_record_id; "
        "refusing to merge distinct molecules into one split group"
    )


class ClassIFit:
    """One Class-I train/eval step over already-loaded FrameDatasets."""

    def __init__(
        self,
        train_records: Sequence,
        val_records: Sequence,
        *,
        train_task: str = TASK_ENERGY,
        val_task: str = TASK_ENERGY,
        hidden: int = 64,
        n_blocks: int = 4,
        n_torsion_terms: int = 8,
        max_atoms: int = 4096,
        epochs: int = 2,
        lr: float = 1e-4,
        seed: int = 2666,
        device: str = "cpu",
        record_root: Path | None = None,
        energy_scope: str = "full",
        eval_every_epochs: int = 1,
        eval_loss: Callable[..., torch.Tensor] | None = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        multiprocessing_context: str | None = "spawn",
        clip_max_norm: float = 1.0e3,
    ) -> None:
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.n_torsion_terms = n_torsion_terms
        self.max_atoms = max_atoms
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.device = torch.device(device)
        self.record_root = record_root
        self.energy_scope = energy_scope
        self.eval_every_epochs = max(1, int(eval_every_epochs))
        self.eval_loss = eval_loss if eval_loss is not None else val_omol_mae
        self.clip_max_norm = float(clip_max_norm)
        self.num_workers = int(num_workers)
        self.prefetch_factor = prefetch_factor
        self.multiprocessing_context = multiprocessing_context
        if _is_labeled_source(train_records) and _is_labeled_source(val_records):
            self.train_ds = train_records
            self.val_ds = val_records
            ready = getattr(train_records, "dressing", None)
            self.dressing = (
                ready
                if ready is not None
                else AtomicReferenceEnergy(atomic_energies=torch.zeros(95))
            )
        else:
            self.train_ds = _LabeledFrames(
                FrameDataset.from_records(train_records, task=train_task, group_of=_require_group)
            )
            self.val_ds = _LabeledFrames(
                FrameDataset.from_records(val_records, task=val_task, group_of=_require_group)
            )
            self.dressing = fit_atomic_reference(train_records)

    def datamodule(self) -> DataModule:
        train = DataMixture(
            {TASK_ENERGY: self.train_ds},
            mixture={TASK_ENERGY: 1.0},
            max_atoms=self.max_atoms,
            target_schema=_FIT_SCHEMA,
        )
        on_cuda = self.device.type == "cuda"
        return DataModule(
            train,
            self.val_ds,
            target_schema=_FIT_SCHEMA,
            batch_size=8,
            num_workers=self.num_workers,
            pin_memory=on_cuda,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            multiprocessing_context=self.multiprocessing_context,
            seed=self.seed,
        )

    def model(self) -> ForceAwareModel:
        imm = build_espaloma_like_model(
            energy_scope=self.energy_scope,
            hidden=self.hidden,
            n_blocks=self.n_blocks,
            n_torsion_terms=self.n_torsion_terms,
            dtype=torch.float32,
        )
        dressing = AtomicReferenceEnergy(
            atomic_energies=self.dressing.atomic_energies.detach().cpu().clone()
        )
        return ForceAwareModel(imm, dressing).to(self.device)

    def run(
        self,
        *,
        resume_from: Path | str | None = None,
        max_steps: int | None = None,
    ) -> dict[str, float]:
        """Train on OMol25 labels; report the configured eval loss."""
        datamodule = self.datamodule()
        model = self.model()
        spec = TaskSpec(name=TASK_ENERGY, code=TASK_CODE_ENERGY, loss=train_objective)
        router = TaskRouter((spec,))
        if self.record_root is not None:
            checkpoint_dir = self.record_root / "checkpoints"
        else:
            checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        hooks: list[BaseHook] = [
            GradClipHook(max_norm=self.clip_max_norm),
            StepSpeedHook(window_size=4),
            Log(
                every_n_steps=5,
                keys=(
                    ("train", "loss_energy"),
                    ("train", "grad_norm"),
                    ("eval", "loss_energy"),
                    ("performance", "step_per_second"),
                ),
            ),
            CheckpointHook(
                checkpoint_dir=str(checkpoint_dir),
                save_best=True,
                best_metric_name=("eval", "loss_energy"),
                best_metric_mode="min",
            ),
        ]
        if self.record_root is not None:
            hooks.append(MolRecMetricsHook(record_root=self.record_root))
        steps_per_epoch = max(1, len(datamodule.train_dataloader()))
        eval_every_n_steps = (
            None if self.eval_every_epochs <= 1 else steps_per_epoch * self.eval_every_epochs
        )
        trainer = Trainer(
            model,
            loss_fn=None,
            optimizer_factory=lambda params: torch.optim.Adam(params, lr=self.lr),
            train_step=MultiTaskTrainStep(router),
            eval_step=MultiTaskEvalStep(
                TaskRouter(
                    (TaskSpec(name=TASK_ENERGY, code=TASK_CODE_ENERGY, loss=self.eval_loss),)
                ),
                no_grad=True,
            ),
            hooks=hooks,
            eval_every_n_steps=eval_every_n_steps,
            resume_from_checkpoint=None if resume_from is None else str(resume_from),
            device=str(self.device),
        )
        trainer.train(datamodule, max_epochs=self.epochs, max_steps=max_steps)
        val_mae = trainer.state.get(("eval", "loss_energy"))
        train_loss = trainer.state.get(("train", "loss_energy"))
        if train_loss is None:
            train_loss = trainer.state.get(("train", "loss"))
        step_per_second = trainer.state.get(("performance", "step_per_second"))
        return {
            "train_loss": float(train_loss) if train_loss is not None else float("nan"),
            "val_mae_kcal": float(val_mae) if val_mae is not None else float("nan"),
            "step_per_second": (
                float(step_per_second) if step_per_second is not None else float("nan")
            ),
            "epochs": float(self.epochs),
            "n_train": float(len(self.train_ds)),
            "n_val": float(len(self.val_ds)),
        }

    def probe_finite(self, *, max_steps: int = 33) -> dict[str, float]:
        """Walk the packed train loader; count batches whose force Hessian is non-finite."""
        datamodule = self.datamodule()
        model = self.model()
        model.train()
        n_bad = 0
        n_steps = 0
        for step, batch in enumerate(datamodule.train_dataloader(), start=1):
            if step > max_steps:
                break
            batch = batch.to(self.device)
            model.zero_grad(set_to_none=True)
            out = model(batch)
            loss = train_objective(out, batch)
            energy_ok = bool(torch.isfinite(out["energy"]).all())
            forces_ok = bool(torch.isfinite(out["forces"]).all())
            loss_ok = bool(torch.isfinite(loss).all())
            if loss_ok:
                loss.backward()
                grads_ok = all(
                    param.grad is None or bool(torch.isfinite(param.grad).all())
                    for param in model.parameters()
                )
            else:
                grads_ok = False
            n_atoms = int(batch["atoms", "Z"].shape[0])
            graphs = batch.get("graphs")
            n_mols = int(graphs["num_atoms"].shape[0]) if graphs is not None else 0
            print(
                f"step={step} n_mols={n_mols} n_atoms={n_atoms} "
                f"E={int(energy_ok)} F={int(forces_ok)} L={int(loss_ok)} G={int(grads_ok)} "
                f"loss={float(loss.detach()) if loss_ok else float('nan')}",
                flush=True,
            )
            if not (energy_ok and forces_ok and loss_ok and grads_ok):
                n_bad += 1
            n_steps += 1
        return {"n_steps": float(n_steps), "n_bad": float(n_bad)}

    def align(self) -> dict[str, float]:
        """Packed vs per-molecule parity on the train set. Not a training run."""
        from scripts.mm_param_learning.align import _flatten, align_suite

        return align_suite(self.model(), _flatten(self))


def split_records(
    records: Sequence,
    *,
    train_frac: float = 0.8,
    seed: int = 2666,
) -> tuple[list, list]:
    """Molecule-level shuffle then ``train_frac`` / remainder split.

    All frames of one ``molecule_id`` stay on the same side.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not records:
        raise ValueError("split_records needs at least one record")
    by: dict[str, list] = {}
    order: list[str] = []
    for record in records:
        mid = _require_group(record[0])
        if mid not in by:
            by[mid] = []
            order.append(mid)
        by[mid].append(record)
    rng = random.Random(seed)
    rng.shuffle(order)
    n_train = int(round(len(order) * train_frac))
    n_train = min(max(n_train, 1), len(order) - 1)
    train_ids = order[:n_train]
    val_ids = order[n_train:]
    train = [item for mid in train_ids for item in by[mid]]
    val = [item for mid in val_ids for item in by[mid]]
    return train, val


def _record_n_atoms(record: Sequence) -> int:
    atoms = record[0]["atoms"]
    if "number" in atoms:
        return int(len(atoms["number"]))
    return int(len(atoms["x"]))


def _record_z(record: Sequence) -> set[int]:
    atoms = record[0]["atoms"]
    if "number" not in atoms:
        return set()
    return {int(z) for z in atoms["number"]}


def split_coverage(train: Sequence, val: Sequence) -> dict[str, Any]:
    """Molecule-level coverage. Refuses an overlapping split."""
    train_ids = [_require_group(rec[0]) for rec in train]
    val_ids = [_require_group(rec[0]) for rec in val]
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"train/val molecule_id overlap: {sorted(overlap)[:8]}")
    train_n = [_record_n_atoms(rec) for rec in train]
    val_n = [_record_n_atoms(rec) for rec in val]
    train_z: set[int] = set()
    val_z: set[int] = set()
    for rec in train:
        train_z |= _record_z(rec)
    for rec in val:
        val_z |= _record_z(rec)
    return {
        "n_train_frames": len(train),
        "n_val_frames": len(val),
        "n_train_molecules": len(set(train_ids)),
        "n_val_molecules": len(set(val_ids)),
        "train_n_atoms_min": min(train_n) if train_n else 0,
        "train_n_atoms_max": max(train_n) if train_n else 0,
        "val_n_atoms_min": min(val_n) if val_n else 0,
        "val_n_atoms_max": max(val_n) if val_n else 0,
        "train_Z": sorted(train_z),
        "val_Z": sorted(val_z),
        "overlap": 0,
    }


def write_split_manifest(
    path: Path, train: Sequence, val: Sequence, *, fingerprint: str | None = None
) -> Path:
    """Write molecule ids for each split. Coverage is included.

    ``fingerprint``, when given, is recorded at the top level so the manifest
    is provably part of one run identity (checklist §0).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "train_ids": [_require_group(rec[0]) for rec in train],
        "val_ids": [_require_group(rec[0]) for rec in val],
        "coverage": split_coverage(train, val),
    }
    if fingerprint is not None:
        payload = {"fingerprint": fingerprint, **payload}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_omol25_records(
    *,
    packed: str | Path | None = None,
    role: str = "val-neutral",
    max_atoms: int = 32,
    n_keep: int = 0,
) -> list:
    """Load OMol25-small records. ``packed`` is the offline test path."""
    from molhub.dataset.omol25 import OMol25, OMol25Dataset, OMol25Small

    if packed:
        source = OMol25Dataset(packed)
    else:
        # Scan extra rows so the small-molecule filter can still fill *n_keep*.
        scan_cap = None if n_keep <= 0 else max(int(n_keep) * 16, 64)
        source = OMol25(role=role, max_rows=scan_cap).load()
    small = OMol25Small(source, max_atoms=max_atoms, limit=n_keep if n_keep > 0 else None)
    small.load()
    records = [small[i] for i in range(len(small))]
    if not records:
        raise ValueError("OMol25-small kept no frames")
    return records


def load_omol25_split(
    *,
    packed: str | Path | None = None,
    role: str = "val-neutral",
    max_atoms: int = 32,
    n_keep: int = 0,
    train_frac: float = 0.8,
    seed: int = 2666,
) -> tuple[list, list]:
    """OMol25-small, molecule-level ``train_frac`` split."""
    raw = load_omol25_records(packed=packed, role=role, max_atoms=max_atoms, n_keep=n_keep)
    return split_records(raw, train_frac=train_frac, seed=seed)


def _is_labeled_source(obj: Any) -> bool:
    return (
        hasattr(obj, "atom_counts")
        and hasattr(obj, "task_labels")
        and hasattr(obj, "target_schema")
    )


class LazyOMol25:
    """Labeled Class-I samples over an OMol25 LMDB, converted on read.

    Does not hold frames. ``atom_counts`` is filled by :func:`load_omol25_official`.
    Source frames stay in published eV; this view converts to kcal/mol on read.
    """

    def __init__(
        self,
        source: Any,
        indices: Sequence[int],
        atom_counts: Sequence[int],
        *,
        dressing: AtomicReferenceEnergy | None = None,
    ) -> None:
        from molhub.dataset.omol25 import OMol25Small

        self._source = source
        self._indices = tuple(int(i) for i in indices)
        self._bond = OMol25Small._bonded
        self.atom_counts = torch.as_tensor(list(atom_counts), dtype=torch.long)
        self.edge_counts = torch.zeros(len(self._indices), dtype=torch.long)
        self.task_labels = (TASK_ENERGY,) * len(self._indices)
        self.group_ids = tuple(str(i) for i in self._indices)
        self.target_schema = _FIT_SCHEMA
        self.dressing = dressing

    def __len__(self) -> int:
        return len(self._indices)

    def subset(self, indices: Sequence[int]) -> "LazyOMol25":
        """A lazy view over the given row indices (same source + dressing)."""
        kept = [int(i) for i in indices]
        counts = self.atom_counts[kept].tolist()
        return LazyOMol25(self._source, kept, counts, dressing=self.dressing)

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], ...]:
        src_idx = self._indices[idx]
        view = _omol_lmdb_view(self._source)
        record_fn = getattr(view, "record", None)
        if callable(record_fn) and getattr(view, "_tables", None) is not None:
            return (self._sample_from_record(record_fn(src_idx), group=float(idx)),)
        # Frame-shaped source (tests / packed NPZ): keep the molpy path.
        frame = self._source[src_idx][0]
        bonded = self._bond(frame, record_id=str(src_idx))
        converted = self._to_kcal(
            bonded if bonded is not None else frame,
            record_id=str(src_idx),
        )
        return (label_sample(converted, group=float(idx)),)

    def _sample_from_record(self, record: dict[str, Any], *, group: float) -> dict[str, Any]:
        """Raw LMDB row → labeled sample, skipping the molpy Frame round trip."""
        from molhub.dataset.omol25 import infer_bonds

        numbers = np.asarray(record["numbers"], dtype=np.int64).reshape(-1)
        positions = np.asarray(record["positions"], dtype=np.float64)
        atomi, atomj = infer_bonds(numbers, positions)
        forces = np.asarray(record["forces"], dtype=np.float64) * _EV_PER_A_TO_KCAL_PER_A
        return _sample_from_arrays(
            numbers=numbers,
            positions=positions,
            energy=float(record["energy"]) * _EV_TO_KCAL,
            forces=forces,
            atomi=atomi,
            atomj=atomj,
            group=group,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Drop the imported bond helper; workers restore it in ``__setstate__``."""
        state = self.__dict__.copy()
        state["_bond"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        from molhub.dataset.omol25 import OMol25Small

        self.__dict__.update(state)
        self._bond = OMol25Small._bonded

    def _to_kcal(self, frame: Any, *, record_id: str) -> Any:
        from molpy import Block, Frame

        atoms = frame["atoms"]
        scale = _EV_PER_A_TO_KCAL_PER_A
        out = Frame()
        atom_block: dict[str, Any] = {
            "element": np.asarray(atoms["element"]),
            "number": np.asarray(atoms["number"]),
            "x": np.asarray(atoms["x"]),
            "y": np.asarray(atoms["y"]),
            "z": np.asarray(atoms["z"]),
        }
        if "fx" in atoms:
            atom_block["fx"] = np.asarray(atoms["fx"], dtype=np.float64) * scale
            atom_block["fy"] = np.asarray(atoms["fy"], dtype=np.float64) * scale
            atom_block["fz"] = np.asarray(atoms["fz"], dtype=np.float64) * scale
        out["atoms"] = Block(atom_block)
        if "bonds" in frame:
            bonds = frame["bonds"]
            out["bonds"] = Block(
                {
                    "atomi": np.asarray(bonds["atomi"]),
                    "atomj": np.asarray(bonds["atomj"]),
                }
            )
        meta = dict(getattr(frame, "meta", None) or {})
        meta["energy"] = float(meta["energy"]) * _EV_TO_KCAL
        meta["source_record_id"] = str(meta.get("source_record_id") or record_id)
        meta["molecule_id"] = str(meta.get("molecule_id") or record_id)
        out.meta = meta
        return out


def _omol_lmdb_view(source: Any) -> Any:
    """Underlying :class:`OMol25Lmdb` when *source* is a loaded :class:`OMol25`."""
    view = getattr(source, "_view", None)
    if view is not None:
        return view
    return source


def _raw_omol_record(source: Any, idx: int) -> dict[str, Any]:
    """LMDB dict when available; otherwise a Frame-shaped fallback for tests."""
    view = _omol_lmdb_view(source)
    tables = getattr(view, "_tables", None)
    index = getattr(view, "_index", None)
    if tables is not None and index is not None:
        table_idx, row_id = index[idx]
        rec = dict(tables[table_idx].get(row_id))
        rec["energy_is_ev"] = True
        return rec
    frame = source[idx][0]
    atoms = frame["atoms"]
    meta = getattr(frame, "meta", None) or {}
    return {
        "numbers": np.asarray(atoms["number"], dtype=np.int64),
        "energy": float(meta.get("energy", 0.0)),
        "energy_is_ev": True,
    }


def _n_atoms_cache_path(source: Any) -> Path:
    home = Path(os.environ.get("MOLHUB_HOME", Path.home() / ".cache" / "molhub"))
    root = str(getattr(source, "root", getattr(source, "source_id", type(source).__name__)))
    digest = hashlib.sha1(f"{root}:{len(source)}".encode()).hexdigest()[:16]
    path = home / "derived" / "omol25-index" / f"{digest}-n_atoms.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _omol_atom_counts(source: Any) -> np.ndarray:
    """Per-row atom counts from LMDB ``numbers``, cached on disk.

    Does not build molpy Frames. Last 4m pass kept every row (max 350 atoms);
    the 8-hour Frame scan was the bug.
    """
    path = _n_atoms_cache_path(source)
    n_src = len(source)
    if path.is_file():
        counts = np.load(path)
        if counts.shape == (n_src,):
            print(f"omol n_atoms cache hit {path} n={n_src}", flush=True)
            return counts.astype(np.int32, copy=False)
    # Official 4m: a prior full pass kept every row, max 350 atoms. Do not
    # decode 4M Frames/LMDB records on a GPU node just to count atoms.
    view = _omol_lmdb_view(source)
    if getattr(view, "_tables", None) is not None:
        hint = 512
        print(
            f"omol n_atoms cache miss n={n_src}; pack with {hint} "
            "(4m max observed 350; skip GPU index scan)",
            flush=True,
        )
        return np.full(n_src, hint, dtype=np.int32)
    counts = np.empty(n_src, dtype=np.int32)
    for i in range(n_src):
        numbers = np.asarray(_raw_omol_record(source, i)["numbers"]).reshape(-1)
        counts[i] = int(numbers.shape[0])
    np.save(path, counts)
    return counts


def _e0_cache_path(source: Any, *, max_atoms: int, n_fit: int, seed: int) -> Path:
    """On-disk cache for ε[Z] keyed by source identity and fit hyper-parameters."""
    home = Path(os.environ.get("MOLHUB_HOME", Path.home() / ".cache" / "molhub"))
    root = str(getattr(source, "root", getattr(source, "source_id", type(source).__name__)))
    digest = hashlib.sha1(f"{root}:{len(source)}:{max_atoms}:{n_fit}:{seed}".encode()).hexdigest()[
        :16
    ]
    path = home / "derived" / "omol25-index" / f"{digest}-e0.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fit_e0_subsample(
    source: Any,
    indices: Sequence[int],
    *,
    n_fit: int = 8192,
    seed: int = 2666,
    max_atoms: int = 0,
) -> np.ndarray:
    """Least-squares ε[Z] on a small random subset, cached on disk.

    8192 samples are ample for a ~10-element least-squares table; the old
    32768-sample pass spent most of ``load_omol25_official`` re-reading
    random LMDB records (≈18 ms/record cold). The result is cached per
    source + ``max_atoms`` so a resubmitted smoke/prod run skips the reads.
    """
    cache_path = _e0_cache_path(source, max_atoms=int(max_atoms), n_fit=n_fit, seed=seed)
    if cache_path.is_file():
        cached = np.load(cache_path)
        if cached.shape == (95,):
            return cached.astype(np.float64, copy=False)

    factor = _EV_TO_KCAL
    pick = list(indices)
    if len(pick) > n_fit:
        rng = random.Random(seed)
        pick = rng.sample(pick, n_fit)
    size = 95
    ata = np.zeros((size, size), dtype=np.float64)
    atb = np.zeros(size, dtype=np.float64)
    for idx in pick:
        rec = _raw_omol_record(source, int(idx))
        numbers = np.asarray(rec["numbers"], dtype=np.int64).reshape(-1)
        energy = float(rec["energy"])
        if rec.get("energy_is_ev", True):
            energy *= factor
        hist = np.bincount(np.clip(numbers, 0, size - 1), minlength=size).astype(np.float64)[:size]
        ata += np.outer(hist, hist)
        atb += hist * energy
    e0 = np.linalg.lstsq(ata, atb, rcond=None)[0]
    np.save(cache_path, e0)
    return e0


def _index_omol_source(
    source: Any, *, max_atoms: int, fit_e0: bool = True
) -> tuple[list[int], list[int], np.ndarray]:
    """Keep rows with ``1 ≤ n_atoms ≤ max_atoms``. Counts come from cache/LMDB."""
    n_atoms = _omol_atom_counts(source)
    keep = np.flatnonzero((n_atoms >= 1) & (n_atoms <= int(max_atoms)))
    if keep.size == 0:
        return [], [], np.zeros(95, dtype=np.float64)
    e0 = (
        _fit_e0_subsample(source, keep.tolist(), max_atoms=int(max_atoms))
        if fit_e0
        else np.zeros(95, dtype=np.float64)
    )
    return keep.tolist(), n_atoms[keep].astype(int).tolist(), e0


def load_omol25_official(
    *,
    train_role: str = "4m",
    val_role: str = "val-neutral",
    max_atoms: int = 16384,
) -> tuple[LazyOMol25, LazyOMol25]:
    """Official OMol25: ``4m`` train, ``val-neutral`` val. Lazy, kcal, bonds on read."""
    from molhub.dataset.omol25 import OMol25

    t0 = perf_counter()
    train_src = OMol25(role=train_role).load()
    print(
        f"omol train_role={train_role} n={len(train_src)} load_s={perf_counter() - t0:.1f}",
        flush=True,
    )
    t0 = perf_counter()
    val_src = OMol25(role=val_role).load()
    print(
        f"omol val_role={val_role} n={len(val_src)} load_s={perf_counter() - t0:.1f}",
        flush=True,
    )
    t0 = perf_counter()
    train_keep, train_n, e0 = _index_omol_source(train_src, max_atoms=max_atoms)
    print(f"omol train index_s={perf_counter() - t0:.1f}", flush=True)
    t0 = perf_counter()
    val_keep, val_n, _ = _index_omol_source(val_src, max_atoms=max_atoms, fit_e0=False)
    print(f"omol val index_s={perf_counter() - t0:.1f}", flush=True)
    if not train_keep or not val_keep:
        raise ValueError("OMol25 official split kept no frames")
    dressing = AtomicReferenceEnergy(atomic_energies=e0.tolist())
    train = LazyOMol25(train_src, train_keep, train_n, dressing=dressing)
    val = LazyOMol25(val_src, val_keep, val_n, dressing=dressing)
    print(f"omol kept train={len(train)} val={len(val)}", flush=True)
    return train, val


def _official_prefix(
    source: Any, n: int, max_atoms: int, scan_cap: int
) -> tuple[list[int], list[int]]:
    """First *n* row indices of *source* with real ``1 ≤ n_atoms ≤ max_atoms``.

    Reads per-row ``numbers`` for a bounded prefix only, so it never runs the
    full 4M-row index scan the 512-hint fallback exists to avoid.
    """
    indices: list[int] = []
    counts: list[int] = []
    limit = min(int(len(source)), int(scan_cap))
    for i in range(limit):
        if len(indices) >= n:
            break
        rec = _raw_omol_record(source, i)
        numbers = np.asarray(rec.get("numbers", np.zeros(0, dtype=np.int64))).reshape(-1)
        n_atoms = int(numbers.shape[0])
        if 1 <= n_atoms <= int(max_atoms):
            indices.append(i)
            counts.append(n_atoms)
    return indices, counts


def load_omol25_official_tiny(
    *,
    n_train: int = 48,
    n_val: int = 12,
    max_atoms: int = 32,
    seed: int = 2666,
    scan_cap: int = 8192,
) -> tuple[LazyOMol25, LazyOMol25]:
    """Official ``4m`` / ``val-neutral`` coordinates on a bounded real subset.

    For the align / inspect / overfit calibration gates: same coordinate
    identity as production, but only the first ``n_train`` / ``n_val`` rows
    with ``1 ≤ n_atoms ≤ max_atoms`` atoms are materialised, with real
    per-row atom counts (no 512 hint). ε[Z] is fitted on that tiny subset.
    The molecule count is an explicitly enumerated calibration-only difference.
    """
    from molhub.dataset.omol25 import OMol25

    train_src = OMol25(role="4m").load()
    val_src = OMol25(role="val-neutral").load()
    train_keep, train_n = _official_prefix(train_src, n_train, max_atoms, scan_cap)
    val_keep, val_n = _official_prefix(val_src, n_val, max_atoms, scan_cap)
    if not train_keep or not val_keep:
        raise ValueError(
            "official tiny subset kept no frames "
            f"(n_train={len(train_keep)}, n_val={len(val_keep)})"
        )
    e0 = _fit_e0_subsample(
        train_src, train_keep, n_fit=len(train_keep), max_atoms=max_atoms, seed=seed
    )
    dressing = AtomicReferenceEnergy(atomic_energies=e0.tolist())
    return (
        LazyOMol25(train_src, train_keep, train_n, dressing=dressing),
        LazyOMol25(val_src, val_keep, val_n, dressing=dressing),
    )


def load_phalkethoh_val(root: str | Path, *, n_keep: int = 0) -> list:
    """Load the published PhAlkEthOH val split. Never used as train."""
    from molhub.dataset.phalkethoh_mm import PhalkethohMMDataset

    ds = PhalkethohMMDataset(
        root, download=False, split="val", max_frames=n_keep if n_keep > 0 else None
    )
    records = [ds[i] for i in range(len(ds))]
    if not records:
        raise ValueError("PhAlkEthOH val split is empty")
    return records
