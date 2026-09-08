"""Packed vs per-molecule Class-I parity. Checklist §2.

Tolerances are declared here before any comparison. Reference path runs
one molecule at a time; packed path is ``collate_molecules`` over the same
dicts. Energy/force residuals, the train objective, parameter-group
gradients, one Adam step, rigid invariants, charge sums, and packed
identity leak must agree.

Size buckets (atoms / molecule, inclusive) are fixed here and reused by
the GH200 calibration job::

    small  1–8
    medium 9–16
    large  17–32

    PYTHONPATH=.:src:../molhub/src:../molpy/src \\
      python scripts/mm_param_learning/align.py
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from molpy import Frame
from tensordict import TensorDict

from molix.batch import TotalCharge
from molix.batch.collate import collate_molecules
from molix.config import config
from molix.core.steps import batch_to
from molpot.forcefield import ForceField
from scripts.mm_param_learning.class_i_fit import (
    ClassIFit,
    ForceAwareModel,
    load_omol25_official_tiny,
    train_objective,
)

# Declared before comparison (checklist §2). float32 Class-I + EEM.
# Energy/loss/force use atol + rtol * scale: dressed totals are ~1e6 kcal/mol,
# so a 0.06 kcal packed residual is float32 noise, not a geometry bug. The
# packed-vs-single force check is relative too: at |F| ~ 2^16 a single float32
# ULP is 2^-7 (~8e-3), so an absolute-only force bound would false-positive.
ATOL_ENERGY = 1e-4
RTOL_ENERGY = 1e-6
ATOL_FORCE = 1e-3
RTOL_FORCE = 1e-3
ATOL_LOSS = 1e-4
RTOL_LOSS = 1e-3
RTOL_GRAD = 1e-3
ATOL_GRAD = 1e-5
ATOL_UPDATE = 5e-4
RTOL_UPDATE = 1e-3
ATOL_INVARIANT = 1e-4
ATOL_CHARGE = 1e-5
ATOL_LEAK = 1e-4


def _within(delta: float, scale: float, *, atol: float, rtol: float) -> bool:
    """True if ``delta <= atol + rtol * |scale|``."""
    return float(delta) <= float(atol) + float(rtol) * max(abs(float(scale)), 1e-12)


# Inclusive atom-count buckets. v0 OMol25-small is capped at 32.
SIZE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("small", 1, 8),
    ("medium", 9, 16),
    ("large", 17, 32),
)

_SHIFT = (1.3, -0.7, 0.4)
_ROT_Z90 = (
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max absolute residual; empty tensors (e.g. unused buffers) → 0.0."""
    delta = (a - b).abs()
    if delta.numel() == 0:
        return 0.0
    return float(delta.max().item())


def _abs_max(t: torch.Tensor) -> float:
    """``|t|.max()`` with empty → 0.0 (PyTorch refuses ``.max()`` on numel 0)."""
    if t.numel() == 0:
        return 0.0
    return float(t.abs().max().item())


def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _field_of(model: ForceAwareModel) -> ForceField:
    """The one force field *model* prices with — its Parameterizer's template.

    Read rather than rebuilt, for the same reason as :func:`_device_of`: a
    probe that assembles its own equal-looking field is enumerating the graph
    at a radius it has to keep in step with the model's by hand, which is
    exactly how these scripts once priced a 15 Å model on a 10 Å list (ledger
    finding **F21**).
    """
    return model.imm.template


def _collate(frames: Sequence[Any], device: torch.device, forcefield: ForceField) -> TensorDict:
    """One packed batch, carrying what a :class:`~molix.batch.DataModule` mounts.

    These probes collate directly, so the two post-collate tasks
    :meth:`~scripts.mm_param_learning.class_i_fit.ClassIFit.datamodule` mounts
    are applied here by hand: the pair list the fitted field is priced against
    (molpot consumes a placed ``edges`` namespace and never enumerates one) and
    the neutral ``graphs.total_charge`` its charge model is fail-closed on.

    The cast is the project dtype policy (``molix.config``'s ``ftype``) — the
    same one the DataModule's ``_CollateFn`` applies — rather than a probe of
    whatever dtype the model happens to carry. The model is built in that dtype
    too (``ClassIFit.model``), so reference and packed path agree by policy
    instead of by inspection.

    Args:
        frames: Frames to collate into one batch.
        device: Where the batch is moved.
        forcefield: The field of the model this batch is fed to — pass
            :func:`_field_of`, so the graph reaches the radius that model
            prices and no other.
    """
    batch = collate_molecules(frames)
    for task in (forcefield.neighbor_list(), TotalCharge()):
        batch = task.execute(batch)
    return batch_to(batch, device=device, dtype=config["ftype"])


def _grad_map(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }


def _rebuild(frame: Any, *, atoms: dict[str, Any] | None = None) -> Frame:
    """Copy *frame*, overriding the named ``atoms`` columns.

    ``Frame.meta`` returns a copy and blocks are replaced wholesale, so a
    perturbed frame is built rather than mutated.

    Args:
        frame: The source molpy ``Frame``.
        atoms: Columns to override in the ``atoms`` block; every other block
            and every other column is carried over unchanged.

    Returns:
        The new ``Frame``.
    """
    meta = {k: getattr(v, "value", v) for k, v in (getattr(frame, "meta", None) or {}).items()}
    out = Frame(meta=meta)
    for name in frame.keys():
        block = frame[name]
        columns = {column: np.asarray(block[column]) for column in block.keys()}
        if name == "atoms" and atoms:
            columns.update(atoms)
        out[name] = columns
    return out


def _positions(frame: Any) -> np.ndarray:
    atoms = frame["atoms"]
    return np.stack([np.asarray(atoms[axis], dtype=np.float64) for axis in ("x", "y", "z")], -1)


def _forces(frame: Any) -> np.ndarray | None:
    atoms = frame["atoms"]
    if "fx" not in atoms.keys():
        return None
    return np.stack([np.asarray(atoms[axis], dtype=np.float64) for axis in ("fx", "fy", "fz")], -1)


def _xyz_columns(pos: np.ndarray, forces: np.ndarray | None) -> dict[str, np.ndarray]:
    columns = {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]}
    if forces is not None:
        columns |= {"fx": forces[:, 0], "fy": forces[:, 1], "fz": forces[:, 2]}
    return columns


def _n_atoms(frame: Any) -> int:
    return int(np.asarray(frame["atoms"]["number"]).shape[0])


def bucket_of(n_atoms: int) -> str | None:
    """Return the declared size-bucket name, or None if out of range."""
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= n_atoms <= hi:
            return name
    return None


def _shift_frame(frame: Any, delta: Sequence[float]) -> Frame:
    """Translate every position by *delta* — energy and forces must not move."""
    pos = _positions(frame) + np.asarray(tuple(delta), dtype=np.float64)
    return _rebuild(frame, atoms=_xyz_columns(pos, _forces(frame)))


def _rotate_frame(frame: Any, rot: Sequence[Sequence[float]]) -> Frame:
    """Rotate positions and forces — energy invariant, forces equivariant."""
    matrix = np.asarray([tuple(row) for row in rot], dtype=np.float64)
    forces = _forces(frame)
    return _rebuild(
        frame,
        atoms=_xyz_columns(
            _positions(frame) @ matrix.T,
            None if forces is None else forces @ matrix.T,
        ),
    )


def _permute_frame(frame: Any, perm: torch.Tensor) -> Frame:
    """Relabel atoms by *perm* — every atom index in every block follows."""
    order = np.asarray(perm.tolist(), dtype=np.int64)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.shape[0], dtype=np.int64)

    meta = {k: getattr(v, "value", v) for k, v in (getattr(frame, "meta", None) or {}).items()}
    out = Frame(meta=meta)
    forces = _forces(frame)
    atoms = frame["atoms"]
    columns = {c: np.asarray(atoms[c])[order] for c in atoms.keys()}
    columns |= _xyz_columns(_positions(frame)[order], None if forces is None else forces[order])
    out["atoms"] = columns
    for name in frame.keys():
        if name == "atoms":
            continue
        block = frame[name]
        out[name] = {
            column: (
                inverse[np.asarray(block[column], dtype=np.int64)]
                if column.startswith("atom")
                else np.asarray(block[column])
            )
            for column in block.keys()
        }
    return out


def _charges(model: ForceAwareModel, batch: TensorDict) -> torch.Tensor:
    filled = model.imm(batch)
    atoms = filled.params("atoms")
    if atoms is None or "q" not in atoms:
        raise RuntimeError("Class-I forward wrote no atoms['q']")
    return atoms["q"]


def align_packed(
    model: ForceAwareModel,
    samples: Sequence[dict[str, Any]],
    *,
    atol_energy: float = ATOL_ENERGY,
    atol_force: float = ATOL_FORCE,
    atol_loss: float = ATOL_LOSS,
    rtol_grad: float = RTOL_GRAD,
    atol_grad: float = ATOL_GRAD,
    atol_update: float = ATOL_UPDATE,
    rtol_update: float = RTOL_UPDATE,
    lr: float = 1e-4,
) -> dict[str, float]:
    """Compare per-molecule reference vs one packed batch.

    Returns:
        Max residuals and 0/1 pass flags. Raises if *samples* is empty.
    """
    if not samples:
        raise ValueError("align_packed needs at least one sample")
    device = _device_of(model)
    field = _field_of(model)
    singles_e: list[torch.Tensor] = []
    singles_mm: list[torch.Tensor] = []
    singles_e0: list[torch.Tensor] = []
    singles_f: list[torch.Tensor] = []
    for sample in samples:
        out = model(_collate([sample], device, field))
        singles_e.append(out["energy"].reshape(-1))
        singles_mm.append(out["energy_mm"].reshape(-1))
        singles_e0.append(out["e0"].reshape(-1))
        singles_f.append(out["forces"])
    energy_ref = torch.cat(singles_e)
    energy_mm_ref = torch.cat(singles_mm)
    e0_ref = torch.cat(singles_e0)
    forces_ref = torch.cat(singles_f, dim=0)

    packed = _collate(samples, device, field)
    out_p = model(packed)
    energy_p = out_p["energy"].reshape(-1)
    energy_max = _max_abs(energy_p, energy_ref)
    energy_scale = float(torch.maximum(energy_p.abs().max(), energy_ref.abs().max()).item())
    force_max = _max_abs(out_p["forces"], forces_ref)
    force_scale = float(torch.maximum(out_p["forces"].abs().max(), forces_ref.abs().max()).item())

    loss_ref = train_objective(
        {
            "energy": energy_ref,
            "energy_mm": energy_mm_ref,
            "e0": e0_ref,
            "forces": forces_ref,
        },
        packed,
    )
    loss_p = train_objective(out_p, packed)
    loss_max = float((loss_p - loss_ref).abs().item())
    loss_scale = max(abs(float(loss_p.detach())), abs(float(loss_ref.detach())))

    model.zero_grad(set_to_none=True)
    loss_p.backward(retain_graph=True)
    grads_p = _grad_map(model)
    model.zero_grad(set_to_none=True)
    loss_ref.backward()
    grads_r = _grad_map(model)

    grad_max = 0.0
    grad_ok = True
    for name, g_p in grads_p.items():
        g_r = grads_r.get(name)
        if g_r is None:
            grad_ok = False
            grad_max = float("inf")
            break
        scale = max(_abs_max(g_p), _abs_max(g_r))
        residual = _max_abs(g_p, g_r)
        grad_max = max(grad_max, residual)
        if residual > atol_grad + rtol_grad * max(scale, 1e-12):
            grad_ok = False

    snap = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.zero_grad(set_to_none=True)
    train_objective(model(packed), packed).backward()
    torch.optim.Adam(model.parameters(), lr=lr).step()
    after_p = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(snap)
    model.zero_grad(set_to_none=True)
    singles_e2: list[torch.Tensor] = []
    singles_mm2: list[torch.Tensor] = []
    singles_e0_2: list[torch.Tensor] = []
    singles_f2: list[torch.Tensor] = []
    for sample in samples:
        out = model(_collate([sample], device, field))
        singles_e2.append(out["energy"].reshape(-1))
        singles_mm2.append(out["energy_mm"].reshape(-1))
        singles_e0_2.append(out["e0"].reshape(-1))
        singles_f2.append(out["forces"])
    train_objective(
        {
            "energy": torch.cat(singles_e2),
            "energy_mm": torch.cat(singles_mm2),
            "e0": torch.cat(singles_e0_2),
            "forces": torch.cat(singles_f2, dim=0),
        },
        packed,
    ).backward()
    torch.optim.Adam(model.parameters(), lr=lr).step()
    update_max = 0.0
    update_ok = True
    for key, value in model.state_dict().items():
        other = value.detach()
        residual = _max_abs(after_p[key], other)
        update_max = max(update_max, residual)
        scale = max(_abs_max(after_p[key]), _abs_max(other))
        if residual > atol_update + rtol_update * max(scale, 1e-12):
            update_ok = False

    return {
        "n_mols": float(len(samples)),
        "energy_max_abs": energy_max,
        "force_max_abs": force_max,
        "loss_max_abs": loss_max,
        "grad_max_abs": grad_max,
        "update_max_abs": update_max,
        "energy_ok": float(_within(energy_max, energy_scale, atol=atol_energy, rtol=RTOL_ENERGY)),
        "force_ok": float(_within(force_max, force_scale, atol=atol_force, rtol=RTOL_FORCE)),
        "loss_ok": float(_within(loss_max, loss_scale, atol=atol_loss, rtol=RTOL_LOSS)),
        "grad_ok": float(grad_ok),
        "update_ok": float(update_ok),
    }


def align_invariants(
    model: ForceAwareModel,
    sample: dict[str, Any],
    *,
    atol_energy: float = ATOL_INVARIANT,
    atol_force: float = ATOL_FORCE,
    rtol_force: float = RTOL_FORCE,
    atol_charge: float = ATOL_CHARGE,
) -> dict[str, float]:
    """Rigid translation/rotation, atom permutation, and charge conservation."""
    device = _device_of(model)
    field = _field_of(model)
    base = model(_collate([sample], device, field))
    energy = base["energy"].reshape(-1)
    forces = base["forces"]

    shifted = model(_collate([_shift_frame(sample, _SHIFT)], device, field))
    energy_scale = float(energy.abs().max().item())
    trans_e = _max_abs(shifted["energy"].reshape(-1), energy)
    trans_f = _max_abs(shifted["forces"], forces)

    rot = torch.tensor(_ROT_Z90, dtype=forces.dtype, device=device)
    rotated = model(_collate([_rotate_frame(sample, _ROT_Z90)], device, field))
    rot_expected_f = forces @ rot.T
    rot_e = _max_abs(rotated["energy"].reshape(-1), energy)
    rot_f = _max_abs(rotated["forces"], rot_expected_f)
    rot_f_scale = float(
        torch.maximum(rotated["forces"].abs().max(), rot_expected_f.abs().max()).item()
    )

    n_atoms = _n_atoms(sample)
    perm = torch.arange(n_atoms - 1, -1, -1)
    permuted = model(_collate([_permute_frame(sample, perm)], device, field))
    perm_e = _max_abs(permuted["energy"].reshape(-1), energy)
    perm_f = _max_abs(permuted["forces"], forces[perm])

    batch = _collate([sample], device, field)
    charges = _charges(model, batch)
    atom_batch = batch["atoms", "batch"].long()
    n_graphs = int(atom_batch.max().item()) + 1
    q_sum = charges.new_zeros(n_graphs).index_add_(0, atom_batch, charges)
    charge_max = float(q_sum.abs().max().item())

    return {
        "trans_energy_max_abs": trans_e,
        "trans_force_max_abs": trans_f,
        "rot_energy_max_abs": rot_e,
        "rot_force_max_abs": rot_f,
        "perm_energy_max_abs": perm_e,
        "perm_force_max_abs": perm_f,
        "charge_sum_max_abs": charge_max,
        "trans_ok": float(
            _within(trans_e, energy_scale, atol=atol_energy, rtol=RTOL_ENERGY)
            and trans_f <= atol_force
        ),
        "rot_ok": float(
            _within(rot_e, energy_scale, atol=atol_energy, rtol=RTOL_ENERGY)
            and rot_f <= atol_force + rtol_force * max(rot_f_scale, 1e-12)
        ),
        "perm_ok": float(
            _within(perm_e, energy_scale, atol=atol_energy, rtol=RTOL_ENERGY)
            and perm_f <= atol_force
        ),
        "charge_ok": float(charge_max <= atol_charge),
    }


# Large enough that Huber even-symmetry cannot cancel a uniform offset on
# typical O(10) kcal/mol residuals (the old +3 shift was a no-op when a
# 2-graph pack had residuals {a, b} with {|a|,|b|} = {|a+3|,|b+3|}).
_ENERGY_LABEL_SHIFT = 50.0


def align_energy_shift(
    model: ForceAwareModel,
    samples: Sequence[dict[str, Any]],
    *,
    atol_loss: float = ATOL_LOSS,
) -> dict[str, float]:
    """A constant offset on the energy labels must change the energy term.

    Applied to the batch target, not the prediction: the residualized Huber
    trains ``E_MM`` against ``E_DFT − e0``, so shifting the dressed prediction
    is not the quantity in the loss. A 50 kcal/mol label shift sits in the
    Huber linear region for any O(10) residual.
    """
    if len(samples) < 2:
        raise ValueError("align_energy_shift needs at least two samples")
    device = _device_of(model)
    field = _field_of(model)
    paired = [_rebuild(samples[0]), _rebuild(samples[1])]
    packed = _collate(paired, device, field)
    out = model(packed)
    loss0 = train_objective(out, packed)
    shifted = packed.clone()
    shifted["graphs", "energy"] = packed["graphs", "energy"] + _ENERGY_LABEL_SHIFT
    loss1 = train_objective(out, shifted)
    residual = float((loss1 - loss0).abs().item())
    return {
        "energy_shift_abs": residual,
        "energy_shift_ok": float(residual > atol_loss),
    }


def align_leak(
    model: ForceAwareModel,
    samples: Sequence[dict[str, Any]],
    *,
    atol_energy: float = ATOL_LEAK,
    atol_force: float = ATOL_FORCE,
) -> dict[str, float]:
    """Perturb molecule 1; molecule 0 energy/forces in the packed batch must hold."""
    if len(samples) < 2:
        raise ValueError("align_leak needs at least two samples")
    device = _device_of(model)
    field = _field_of(model)
    first_n = _n_atoms(samples[0])
    packed = model(_collate(list(samples), device, field))
    energy0 = packed["energy"].reshape(-1)[0].detach()
    forces0 = packed["forces"][:first_n].detach()

    mutated = [_rebuild(samples[0]), _shift_frame(samples[1], (8.0, 0.0, 0.0))]
    mutated.extend(_rebuild(sample) for sample in samples[2:])
    after = model(_collate(mutated, device, field))
    energy_max = float((after["energy"].reshape(-1)[0] - energy0).abs().item())
    force_max = _max_abs(after["forces"][:first_n], forces0)
    return {
        "leak_energy_max_abs": energy_max,
        "leak_force_max_abs": force_max,
        "leak_ok": float(energy_max <= atol_energy and force_max <= atol_force),
    }


def _valence_rows(sample: dict[str, Any], name: str) -> bool:
    block = sample.get(name)
    if not isinstance(block, dict) or "atomi" not in block:
        return False
    atomi = block["atomi"]
    return int(getattr(atomi, "numel", lambda: len(atomi))()) > 0


def _active_groups(samples: Sequence[dict[str, Any]]) -> frozenset[str]:
    names = {"encoder"}
    if any("bond_index" in sample for sample in samples):
        names.add("bond_head")
    if any(_valence_rows(sample, "angles") for sample in samples):
        names.add("angle_head")
    if any(_valence_rows(sample, "propers") for sample in samples):
        names.add("proper_head")
    if any(_valence_rows(sample, "impropers") for sample in samples):
        names.add("improper_head")
    return frozenset(names)


def align_finite(
    model: ForceAwareModel,
    samples: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """NaN/Inf scan plus per-group finite/nonzero gradients.

    Only topology families present in *samples* must carry a nonzero
    gradient. A water (no propers) does not fail ``proper_head``.
    """
    if not samples:
        raise ValueError("align_finite needs at least one sample")
    device = _device_of(model)
    field = _field_of(model)
    packed = _collate(samples, device, field)
    out = model(packed)
    loss = train_objective(out, packed)
    model.zero_grad(set_to_none=True)
    loss.backward()

    finite = bool(torch.isfinite(out["energy"]).all() and torch.isfinite(out["forces"]).all())
    required = _active_groups(samples)
    groups_ok = True
    n_groups = 0
    n_nonzero = 0
    grad_norm_max = 0.0
    for name, params in model.parameter_groups().items():
        n_groups += 1
        pieces = [param.grad.detach().reshape(-1) for param in params if param.grad is not None]
        if not pieces:
            if name in required:
                groups_ok = False
            continue
        flat = torch.cat(pieces)
        if not bool(torch.isfinite(flat).all()):
            groups_ok = False
            finite = False
            continue
        norm = float(flat.norm().item())
        grad_norm_max = max(grad_norm_max, norm)
        if norm > 0.0:
            n_nonzero += 1
        elif name in required:
            groups_ok = False

    return {
        "finite_ok": float(finite),
        "groups_ok": float(groups_ok),
        "n_grad_groups": float(n_groups),
        "n_nonzero_groups": float(n_nonzero),
        "grad_norm_max": grad_norm_max,
        "pred_energy_abs_max": float(out["energy"].detach().abs().max().item()),
        "pred_force_abs_max": float(out["forces"].detach().abs().max().item()),
    }


def align_suite(
    model: ForceAwareModel,
    samples: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Packed + invariants + leak + finite on one sample list."""
    if not samples:
        raise ValueError("align_suite needs at least one sample")
    report = align_packed(model, samples)
    report.update(align_invariants(model, samples[0]))
    report.update(align_finite(model, samples))
    if len(samples) >= 2:
        report.update(align_leak(model, samples))
        report.update(align_energy_shift(model, samples))
    else:
        report["leak_ok"] = 0.0
        report["energy_shift_ok"] = 0.0
    flags = (
        "energy_ok",
        "force_ok",
        "loss_ok",
        "grad_ok",
        "update_ok",
        "trans_ok",
        "rot_ok",
        "perm_ok",
        "charge_ok",
        "leak_ok",
        "energy_shift_ok",
        "finite_ok",
        "groups_ok",
    )
    report["ok"] = float(all(report.get(name, 0.0) for name in flags))
    return report


def _partition_buckets(
    samples: Sequence[dict[str, Any]],
    *,
    n_per_bucket: int,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in SIZE_BUCKETS}
    for sample in samples:
        name = bucket_of(_n_atoms(sample))
        if name is None:
            continue
        if len(buckets[name]) < n_per_bucket:
            buckets[name].append(sample)
    return buckets


def _flatten(fit: ClassIFit) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for idx in range(len(fit.train_ds)):
        samples.extend(fit.train_ds[idx])
    return samples


def run_align(
    *,
    n_keep: int = 48,
    n_per_bucket: int = 4,
    device: str = "cpu",
) -> dict[str, Any]:
    """Bounded official-4m packed-vs-single + invariant suite. Not a training run."""
    from scripts.mm_param_learning.prod_day import _require_nobackup_molhub_home

    _require_nobackup_molhub_home()
    train, val = load_omol25_official_tiny(n_train=n_keep, n_val=max(n_keep // 4, 2), max_atoms=32)
    fit = ClassIFit(
        train,
        val,
        hidden=64,
        n_blocks=4,
        device=device,
        epochs=1,
        energy_scope="full",
    )
    model = fit.model()
    samples = _flatten(fit)
    buckets = _partition_buckets(samples, n_per_bucket=n_per_bucket)
    report: dict[str, Any] = {
        "device": device,
        "n_loaded": float(len(samples)),
        "n_per_bucket": float(n_per_bucket),
        "buckets": {},
    }
    all_ok = True
    for name, _lo, _hi in SIZE_BUCKETS:
        chosen = buckets[name]
        sizes = [_n_atoms(sample) for sample in chosen]
        entry: dict[str, Any] = {
            "n_mols": float(len(chosen)),
            "n_atoms_min": float(min(sizes) if sizes else 0),
            "n_atoms_max": float(max(sizes) if sizes else 0),
        }
        if len(chosen) < 2:
            entry["ok"] = 0.0
            entry["skipped"] = 1.0
            all_ok = False
        else:
            entry.update(align_suite(model, chosen))
            all_ok = all_ok and bool(entry["ok"])
        report["buckets"][name] = entry
    report["ok"] = float(all_ok)
    return report


if __name__ == "__main__":
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps(run_align(device=_device), indent=2, sort_keys=True))
