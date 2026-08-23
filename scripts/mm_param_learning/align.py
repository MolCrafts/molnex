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
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from tensordict import TensorDict

from molix.batch.collate import TargetSchema, collate_molecules
from scripts.mm_param_learning.class_i_fit import (
    _FIT_SCHEMA,
    ClassIFit,
    ForceAwareModel,
    load_omol25_split,
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


def _collate(
    samples: Sequence[dict[str, Any]],
    schema: TargetSchema,
    device: torch.device,
) -> TensorDict:
    return collate_molecules(samples, schema).to(device)


def _grad_map(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }


def _clone_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            out[key] = value.detach().clone()
        elif isinstance(value, Mapping):
            out[key] = _clone_sample(value)
        else:
            out[key] = value
    return out


def _n_atoms(sample: Mapping[str, Any]) -> int:
    return int(sample["Z"].shape[0])


def bucket_of(n_atoms: int) -> str | None:
    """Return the declared size-bucket name, or None if out of range."""
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= n_atoms <= hi:
            return name
    return None


def _shift_sample(sample: Mapping[str, Any], delta: Sequence[float]) -> dict[str, Any]:
    out = _clone_sample(sample)
    shift = torch.tensor(tuple(delta), dtype=out["pos"].dtype)
    out["pos"] = out["pos"] + shift
    return out


def _rotate_sample(sample: Mapping[str, Any], rot: Sequence[Sequence[float]]) -> dict[str, Any]:
    out = _clone_sample(sample)
    matrix = torch.tensor([tuple(row) for row in rot], dtype=out["pos"].dtype)
    out["pos"] = out["pos"] @ matrix.T
    targets = out.get("targets")
    if isinstance(targets, dict) and "forces" in targets:
        targets["forces"] = targets["forces"] @ matrix.T
    return out


def _permute_sample(sample: Mapping[str, Any], perm: torch.Tensor) -> dict[str, Any]:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(int(perm.numel()), dtype=perm.dtype)
    out = _clone_sample(sample)
    out["Z"] = out["Z"][perm]
    out["pos"] = out["pos"][perm]
    if "bond_index" in out:
        out["bond_index"] = inv[out["bond_index"]]
    for family in ("angles", "propers", "impropers"):
        block = out.get(family)
        if not isinstance(block, dict):
            continue
        for col in ("atomi", "atomj", "atomk", "atoml"):
            if col in block:
                block[col] = inv[block[col]]
    targets = out.get("targets")
    if isinstance(targets, dict) and "forces" in targets:
        targets["forces"] = targets["forces"][perm]
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
    target_schema: TargetSchema = _FIT_SCHEMA,
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
    singles_e: list[torch.Tensor] = []
    singles_f: list[torch.Tensor] = []
    for sample in samples:
        out = model(_collate([sample], target_schema, device))
        singles_e.append(out["energy"].reshape(-1))
        singles_f.append(out["forces"])
    energy_ref = torch.cat(singles_e)
    forces_ref = torch.cat(singles_f, dim=0)

    packed = _collate(samples, target_schema, device)
    out_p = model(packed)
    energy_p = out_p["energy"].reshape(-1)
    energy_max = _max_abs(energy_p, energy_ref)
    energy_scale = float(torch.maximum(energy_p.abs().max(), energy_ref.abs().max()).item())
    force_max = _max_abs(out_p["forces"], forces_ref)
    force_scale = float(
        torch.maximum(out_p["forces"].abs().max(), forces_ref.abs().max()).item()
    )

    loss_ref = train_objective({"energy": energy_ref, "forces": forces_ref}, packed)
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
    singles_f2: list[torch.Tensor] = []
    for sample in samples:
        out = model(_collate([sample], target_schema, device))
        singles_e2.append(out["energy"].reshape(-1))
        singles_f2.append(out["forces"])
    train_objective(
        {"energy": torch.cat(singles_e2), "forces": torch.cat(singles_f2, dim=0)},
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
    target_schema: TargetSchema = _FIT_SCHEMA,
    atol_energy: float = ATOL_INVARIANT,
    atol_force: float = ATOL_FORCE,
    rtol_force: float = RTOL_FORCE,
    atol_charge: float = ATOL_CHARGE,
) -> dict[str, float]:
    """Rigid translation/rotation, atom permutation, and charge conservation."""
    device = _device_of(model)
    base = model(_collate([sample], target_schema, device))
    energy = base["energy"].reshape(-1)
    forces = base["forces"]

    shifted = model(_collate([_shift_sample(sample, _SHIFT)], target_schema, device))
    energy_scale = float(energy.abs().max().item())
    trans_e = _max_abs(shifted["energy"].reshape(-1), energy)
    trans_f = _max_abs(shifted["forces"], forces)

    rot = torch.tensor(_ROT_Z90, dtype=forces.dtype, device=device)
    rotated = model(_collate([_rotate_sample(sample, _ROT_Z90)], target_schema, device))
    rot_expected_f = forces @ rot.T
    rot_e = _max_abs(rotated["energy"].reshape(-1), energy)
    rot_f = _max_abs(rotated["forces"], rot_expected_f)
    rot_f_scale = float(
        torch.maximum(rotated["forces"].abs().max(), rot_expected_f.abs().max()).item()
    )

    n_atoms = _n_atoms(sample)
    perm = torch.arange(n_atoms - 1, -1, -1)
    permuted = model(_collate([_permute_sample(sample, perm)], target_schema, device))
    perm_e = _max_abs(permuted["energy"].reshape(-1), energy)
    perm_f = _max_abs(permuted["forces"], forces[perm])

    batch = _collate([sample], target_schema, device)
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


def align_energy_shift(
    model: ForceAwareModel,
    samples: Sequence[dict[str, Any]],
    *,
    target_schema: TargetSchema = _FIT_SCHEMA,
    atol_loss: float = ATOL_LOSS,
) -> dict[str, float]:
    """A constant energy offset must change the dressed total-energy MSE."""
    if len(samples) < 2:
        raise ValueError("align_energy_shift needs at least two samples")
    device = _device_of(model)
    paired = [_clone_sample(samples[0]), _clone_sample(samples[1])]
    packed = _collate(paired, target_schema, device)
    out = model(packed)
    loss0 = train_objective(out, packed)
    shifted = {"energy": out["energy"] + 3.0, "forces": out["forces"]}
    loss1 = train_objective(shifted, packed)
    residual = float((loss1 - loss0).abs().item())
    return {
        "energy_shift_abs": residual,
        "energy_shift_ok": float(residual > atol_loss),
    }


def align_leak(
    model: ForceAwareModel,
    samples: Sequence[dict[str, Any]],
    *,
    target_schema: TargetSchema = _FIT_SCHEMA,
    atol_energy: float = ATOL_LEAK,
    atol_force: float = ATOL_FORCE,
) -> dict[str, float]:
    """Perturb molecule 1; molecule 0 energy/forces in the packed batch must hold."""
    if len(samples) < 2:
        raise ValueError("align_leak needs at least two samples")
    device = _device_of(model)
    first_n = _n_atoms(samples[0])
    packed = model(_collate(list(samples), target_schema, device))
    energy0 = packed["energy"].reshape(-1)[0].detach()
    forces0 = packed["forces"][:first_n].detach()

    mutated = [_clone_sample(samples[0]), _shift_sample(samples[1], (8.0, 0.0, 0.0))]
    mutated.extend(_clone_sample(sample) for sample in samples[2:])
    after = model(_collate(mutated, target_schema, device))
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
    *,
    target_schema: TargetSchema = _FIT_SCHEMA,
) -> dict[str, float]:
    """NaN/Inf scan plus per-group finite/nonzero gradients.

    Only topology families present in *samples* must carry a nonzero
    gradient. A water (no propers) does not fail ``proper_head``.
    """
    if not samples:
        raise ValueError("align_finite needs at least one sample")
    device = _device_of(model)
    packed = _collate(samples, target_schema, device)
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
    *,
    target_schema: TargetSchema = _FIT_SCHEMA,
) -> dict[str, float]:
    """Packed + invariants + leak + finite on one sample list."""
    if not samples:
        raise ValueError("align_suite needs at least one sample")
    report = align_packed(model, samples, target_schema=target_schema)
    report.update(align_invariants(model, samples[0], target_schema=target_schema))
    report.update(align_finite(model, samples, target_schema=target_schema))
    if len(samples) >= 2:
        report.update(align_leak(model, samples, target_schema=target_schema))
        report.update(align_energy_shift(model, samples, target_schema=target_schema))
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
    """Bounded OMol25 packed-vs-single + invariant suite. Not a training run."""
    train, val = load_omol25_split(role="val-neutral", n_keep=n_keep, max_atoms=32)
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
