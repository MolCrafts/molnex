"""Checklist §2 scale / NaN / outlier / clip inspection. Not a training run.

Reports prediction vs target magnitudes, dressed energy vs force term
split, pre-clip gradient norms, and whether the clip hook would fire.
Declared thresholds live here before any comparison.

    PYTHONPATH=.:src:../molhub/src:../molpy/src \\
      python scripts/mm_param_learning/scale_audit.py
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

import torch

from scripts.mm_param_learning.align import (
    _active_groups,
    _collate,
    _device_of,
    _field_of,
)
from scripts.mm_param_learning.class_i_fit import (
    FORCE_OUTLIER_KCAL_A,
    ClassIFit,
    ForceAwareModel,
    _graph_tensor,
    energy_force_terms,
    load_omol25_official_tiny,
    train_objective,
)

# Declared before inspection (checklist §2). Matches ClassIFit GradClipHook.
CLIP_MAX_NORM = 1.0e3
PRED_TARGET_RATIO_MAX = 50.0
ENERGY_TERM_WARN = 1.0e8


def _median_abs(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().abs().median().item())


def _max_abs(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().abs().max().item())


def inspect_scales(
    model: ForceAwareModel,
    samples: Sequence[dict[str, Any]],
    *,
    clip_max_norm: float = CLIP_MAX_NORM,
    force_outlier: float = FORCE_OUTLIER_KCAL_A,
    ratio_max: float = PRED_TARGET_RATIO_MAX,
) -> dict[str, float]:
    """One packed forward+backward. Returns magnitudes and 0/1 flags."""
    if not samples:
        raise ValueError("inspect_scales needs at least one sample")
    device = _device_of(model)
    packed = _collate(samples, device, _field_of(model))
    out = model(packed)
    energy = out["energy"].reshape(-1)
    forces = out["forces"]
    ref_e = _graph_tensor(packed, "energy")
    group = _graph_tensor(packed, "group")
    ref_f = packed["atoms", "forces"] if packed.get("atoms") is not None else None

    loss = train_objective(out, packed)
    energy_term, force_term = energy_force_terms(out, packed)

    model.zero_grad(set_to_none=True)
    loss.backward()
    pieces = [
        param.grad.detach().reshape(-1)
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    grad_norm = float(torch.cat(pieces).norm().item()) if pieces else 0.0

    groups = []
    if group is not None:
        groups = [int(g) for g in group.detach().cpu().reshape(-1).tolist()]
    counts = Counter(groups)
    n_groups = float(len(counts))
    n_singletons = float(sum(1 for n in counts.values() if n < 2))
    n_multi = float(sum(1 for n in counts.values() if n >= 2))

    pred_e_med = _median_abs(energy)
    pred_f_med = _median_abs(forces)
    ref_e_med = _median_abs(ref_e) if ref_e is not None else 0.0
    ref_f_med = _median_abs(ref_f) if ref_f is not None else 0.0
    force_ratio = pred_f_med / max(ref_f_med, 1e-12)
    n_force_outliers = 0.0
    if ref_f is not None:
        n_force_outliers = float((ref_f.detach().norm(dim=-1) > force_outlier).sum().item())

    finite = bool(
        torch.isfinite(energy).all()
        and torch.isfinite(forces).all()
        and torch.isfinite(loss)
        and all(torch.isfinite(chunk).all() for chunk in pieces)
    )
    collapse = bool(len(samples) > 1 and n_groups == 1.0 and ref_e_med > ENERGY_TERM_WARN)
    energy_ok = (
        bool(torch.isfinite(energy_term.detach()))
        and float(energy_term.detach()) < ENERGY_TERM_WARN
    )
    # At init a dressed-energy model starts with ~0 MM forces: the frozen
    # per-element dressing is coordinate-free, so the force signal is learned
    # from zero. A small pred/ref ratio is therefore expected, not a collapse.
    # Only the upper (explosion) bound is an init gate; force collapse is
    # caught by the overfit gate once training starts.
    force_ok = force_ratio <= ratio_max
    required = _active_groups(samples)
    groups_ok = True
    n_nonzero = 0.0
    for name, params in model.parameter_groups().items():
        grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
        if not grads:
            if name in required:
                groups_ok = False
            continue
        norm = float(torch.cat(grads).norm().item())
        if norm > 0.0:
            n_nonzero += 1.0
        elif name in required:
            groups_ok = False

    return {
        "n_mols": float(len(samples)),
        "n_groups": n_groups,
        "n_singletons": n_singletons,
        "n_multi_groups": n_multi,
        "energy_term": float(energy_term.detach()),
        "force_term": float(force_term.detach()),
        "loss": float(loss.detach()),
        "pred_energy_abs_med": pred_e_med,
        "pred_energy_abs_max": _max_abs(energy),
        "ref_energy_abs_med": ref_e_med,
        "ref_energy_abs_max": _max_abs(ref_e) if ref_e is not None else 0.0,
        "pred_force_abs_med": pred_f_med,
        "pred_force_abs_max": _max_abs(forces),
        "ref_force_abs_med": ref_f_med,
        "ref_force_abs_max": _max_abs(ref_f) if ref_f is not None else 0.0,
        "force_pred_over_ref": force_ratio,
        "n_force_outliers": n_force_outliers,
        "grad_norm": grad_norm,
        "clip_max_norm": float(clip_max_norm),
        "clip_would_fire": float(grad_norm > clip_max_norm),
        "n_nonzero_groups": n_nonzero,
        "finite_ok": float(finite),
        "collapse_ok": float(not collapse),
        "energy_scale_ok": float(energy_ok),
        "force_scale_ok": float(force_ok),
        "groups_ok": float(groups_ok),
        "ok": float(finite and not collapse and energy_ok and force_ok and groups_ok),
    }


def _flatten(fit: ClassIFit) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for idx in range(len(fit.train_ds)):
        samples.extend(fit.train_ds[idx])
    return samples


def run_inspect(*, n_keep: int = 16, device: str = "cpu") -> dict[str, Any]:
    """Bounded official-4m scale inspection. Not a training run."""
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
    report = inspect_scales(fit.model(), _flatten(fit))
    report["device"] = device
    return report


if __name__ == "__main__":
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps(run_inspect(device=_device), indent=2, sort_keys=True))
