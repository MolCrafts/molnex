"""Official ``omol25-fairchem-4m`` composition site.

FAIR AllScAIP OMol-only 4M at commit ``41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9``.
This profile does not grant production GO: see
``.claude/notes/production-training-checklist.md``. Historical Class-I
checklist ticks do not transfer; this pin needs a fresh molexp calibration.

Does not call ``energy_force_terms``; residual labels are ``EnergyResidual``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
from molhub.loader import DataModule

from molix.core.hook import BaseHook
from molix.core.trainer import Trainer
from molix.data import EnergyResidual
from molix.hooks import CheckpointHook
from molix.units import units
from molpot.heads.energy import AtomicReferenceEnergy
from scripts.mm_param_learning.class_i_fit import (
    ClassIFit,
    _omol_elem_refs_path,
    class_i_data_module,
    fairchem_4m_objective,
    load_omol25_fairchem_4m,
)
from scripts.mm_param_learning.configs import (
    OMOL25_FAIRCHEM_4M,
    require_protocol_resolved,
)


def load_omol_elem_refs() -> AtomicReferenceEnergy:
    """Published combined HOF+linear table. Fail-closes if the YAML is absent."""
    path = _omol_elem_refs_path()
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("load_omol_elem_refs needs PyYAML to read omol-elem-refs") from exc
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, Mapping) or "omol_elem_refs" not in payload:
        raise ValueError(f"{path} has no omol_elem_refs table")
    energies, supported = _dense_elem_refs(payload["omol_elem_refs"])
    return AtomicReferenceEnergy(
        atomic_energies=energies,
        supported=supported,
        scheme="combined_hof_linear",
        units="eV",
    )


def _dense_elem_refs(raw: object) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(raw, Mapping):
        zs = [int(z) for z in raw]
        size = (max(zs) + 1) if zs else 0
        energies = torch.zeros(size, dtype=torch.float64)
        supported = torch.zeros(size, dtype=torch.bool)
        for z, value in raw.items():
            index = int(z)
            energies[index] = float(value)
            supported[index] = True
        return energies, supported
    values = [float(item) for item in raw]  # type: ignore[union-attr]
    energies = torch.tensor(values, dtype=torch.float64)
    supported = torch.ones(energies.numel(), dtype=torch.bool)
    return energies, supported


def official_class_i_fit(
    profile: Mapping[str, Any],
    train: Any,
    roles: Mapping[str, Any],
    table: AtomicReferenceEnergy,
) -> ClassIFit:
    """Wire the named protocol into ClassIFit knobs. Cosine is run()-only."""
    require_protocol_resolved(profile)
    residual = EnergyResidual(
        coefficients=table.atomic_energies.detach().cpu().clone(),
        support=table.supported.detach().cpu().clone(),
    )
    lr = float(profile["lr"])
    weight_decay = float(profile["weight_decay"])

    def _adamw(params: Any) -> torch.optim.AdamW:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    return ClassIFit(
        train,
        roles["general"],
        loss=fairchem_4m_objective,
        eval_loss=fairchem_4m_objective,
        optimizer_factory=_adamw,
        batch_tasks=(residual,),
        energy_unit=units.eV,
        seed=int(profile["shuffle_seed"]),
        clip_max_norm=float(profile["clip_max_norm"]),
        eval_every_n_steps=int(profile["eval_every_n_steps"]),
        epochs=int(profile["epochs"]),
        max_atoms=int(profile["batch_max_atoms"]),
    )


def official_cosine_factory(
    steps_per_epoch: int,
    profile: Mapping[str, Any] = OMOL25_FAIRCHEM_4M,
) -> Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]:
    """Trainer-arity warmup then cosine. ``T_max`` is the 75-epoch cosine segment."""
    warmup_epochs = int(profile["warmup_epochs"])
    epochs = int(profile["epochs"])
    start_factor = float(profile["lr_start_factor"])
    min_factor = float(profile["lr_min_factor"])
    warmup_steps = warmup_epochs * int(steps_per_epoch)
    cosine_steps = (epochs - warmup_epochs) * int(steps_per_epoch)

    def _factory(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=min_factor * float(optimizer.param_groups[0]["lr"]),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        # torch>=2.15 stores the list as ``_milestones``; tests and callers
        # still address ``milestones``.
        scheduler.milestones = scheduler._milestones
        return scheduler

    return _factory


def attach_omol_elem_refs(model: nn.Module, table: AtomicReferenceEnergy) -> None:
    """Register the published table as checkpoint submodule ``omol_elem_refs``."""
    model.add_module("omol_elem_refs", table)


def official_val_datamodule(val_records: Any, *, train_dm: DataModule) -> DataModule:
    """Eval DataModule that reuses the train collate list and packing knobs."""
    return DataModule(
        train_dm.train_dataset,
        val_records,
        batch_tasks=train_dm.batch_tasks,
        batch_size=train_dm.batch_size,
        num_workers=train_dm.num_workers,
        pin_memory=train_dm.pin_memory,
        prefetch_factor=train_dm.prefetch_factor,
        multiprocessing_context=train_dm.multiprocessing_context,
        seed=train_dm.seed,
    )


def _overrides_eval_complete(hook: object) -> bool:
    for cls in type(hook).__mro__:
        if cls is BaseHook:
            return False
        if "on_eval_step_complete" in cls.__dict__:
            return True
    method = getattr(type(hook), "on_eval_step_complete", None)
    return method is not None and method is not BaseHook.on_eval_step_complete


@contextmanager
def _without_eval_persistence(trainer: Trainer) -> Iterator[None]:
    original = list(trainer.hooks)
    trainer.hooks = [hook for hook in original if not _overrides_eval_complete(hook)]
    try:
        yield
    finally:
        trainer.hooks = original


def eval_extra_roles(
    trainer: Trainer,
    roles: Mapping[str, Any],
    *,
    make_val_datamodule: Callable[[Any], Any],
) -> None:
    """Score non-general eval roles into a one-home ``state["eval"][role]`` map."""
    current = dict(trainer.state["eval"])
    general = {key: value for key, value in current.items() if not isinstance(value, dict)}
    homes: dict[str, dict[str, Any]] = {"general": general}
    trainer.state["eval"] = {"general": dict(general)}
    with _without_eval_persistence(trainer):
        for name, records in roles.items():
            if name == "general":
                continue
            returned = trainer.evaluate(make_val_datamodule(records))
            mapping = {
                key: value for key, value in dict(returned).items() if not isinstance(value, dict)
            }
            homes[name] = mapping
            trainer.state["eval"] = {role: dict(values) for role, values in homes.items()}


def _prune_step_checkpoints(directory: Path, keep: int) -> None:
    ranked: list[tuple[int, Path]] = []
    for path in directory.glob("step_*.pt"):
        suffix = path.stem.removeprefix("step_")
        try:
            ranked.append((int(suffix), path))
        except ValueError:
            continue
    ranked.sort()
    extra = ranked if keep <= 0 else ranked[:-keep]
    for _, path in extra:
        path.unlink(missing_ok=True)


def run() -> None:
    """Complete official composition. Does not call ``ClassIFit.run``."""
    profile = OMOL25_FAIRCHEM_4M
    require_protocol_resolved(profile)
    train, roles = load_omol25_fairchem_4m()
    table = load_omol_elem_refs()
    fit = official_class_i_fit(profile, train, roles, table)
    model = fit.model()
    attach_omol_elem_refs(model, table)
    dm = class_i_data_module(fit)
    fit.lr_scheduler_factory = official_cosine_factory(len(dm.train_dataloader()), profile)
    trainer = fit.trainer(model)
    torch.backends.cuda.matmul.allow_tf32 = bool(profile["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(profile["tf32"])
    checkpoint_dir = Path("checkpoints")
    for hook in trainer.hooks:
        if isinstance(hook, CheckpointHook):
            hook.save_every_n_steps = int(profile["eval_every_n_steps"])
            checkpoint_dir = Path(hook.checkpoint_dir)
    trainer.train(dm, max_epochs=int(profile["epochs"]))
    eval_extra_roles(
        trainer,
        roles,
        make_val_datamodule=lambda ds: official_val_datamodule(ds, train_dm=dm),
    )
    _prune_step_checkpoints(checkpoint_dir, int(profile["checkpoint_keep"]))


if __name__ == "__main__":
    run()
