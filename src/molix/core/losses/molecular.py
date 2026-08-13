"""Molecular-ML loss presets built on top of the generic :mod:`losses`.

The generic :class:`MSELoss` / :class:`MAELoss` are schema-agnostic: they
extract values by plain-string key. Training scripts then wrap them in
a per-script closure that knows *where* to read targets from the
``TensorDict`` — leading to three copies of

.. code-block:: python

    def loss_fn(preds, batch):
        return mse(preds["energy"], batch["graphs", target_key])

These presets absorb that wrapping. They return ``Callable[[preds, batch], Tensor]``
that consumes ``preds`` (plain dict from the model forward) and ``batch``
(a ``TensorDict``), following the convention encoded in
:class:`~molix.data.collate.TargetSchema`:

* graph-level targets (e.g. ``energy``) live at ``batch["graphs", key]``;
* atom-level targets (e.g. ``forces``) live at ``batch["atoms", key]``.

The factories do **not** auto-discover the schema — the caller passes the
exact target key name, so the resulting callable is a pure function with
static shape/key access (keeps ``torch.compile`` graphs stable).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch
import torch.nn as nn

from molix.config import config
from molix.core.losses.energy import MSELoss

__all__ = [
    "ChargeMSELoss",
    "ProjectedInternalHessianLoss",
    "ScaleNormalizedLoss",
    "TorsionProfileLoss",
    "center_by_group",
    "energy_force_mse",
    "energy_mse",
    "molecule_centered_energy_mse",
    "parameter_bag_mse",
]


def center_by_group(
    values: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    """Subtract per-group means from *values* (Espaloma molecule centering).

    Args:
        values: Scalar energies ``(B,)`` or ``(B, 1)``.
        group_ids: Integer molecule / group ids ``(B,)`` aligned with *values*.

    Returns:
        Centered values with the same shape as *values*.
    """
    flat = values.reshape(-1)
    ids = group_ids.reshape(-1).long()
    if flat.numel() != ids.numel():
        raise ValueError(f"values and group_ids length mismatch: {flat.numel()} vs {ids.numel()}")
    out = flat.clone()
    for g in ids.unique():
        mask = ids == g
        out[mask] = flat[mask] - flat[mask].mean()
    return out.view_as(values)


def molecule_centered_energy_mse(
    target_key: str = "mm_energy",
    *,
    pred_key: str = "energy",
    group_key: str = "molecule_id_index",
    reduction: str = "mean",
) -> Callable[[Mapping[str, Any], Any], torch.Tensor]:
    """MSE after independent molecule-mean centering of pred and ref energies.

    Both predictions and targets are centered **per molecule** before the MSE
    (Espaloma relative conformational energy protocol). Units: kcal/mol.

    Args:
        target_key: Graph-level target at ``batch["graphs", target_key]``.
        pred_key: Model energy key in ``preds``.
        group_key: Graph-level integer molecule index at
            ``batch["graphs", group_key]``.
        reduction: Forwarded to :class:`torch.nn.MSELoss`.

    Returns:
        ``loss_fn(preds, batch) -> Tensor``.
    """
    mse = nn.MSELoss(reduction=reduction)

    def _fn(preds: Mapping[str, Any], batch: Any) -> torch.Tensor:
        e_pred = preds[pred_key]
        e_true = batch["graphs", target_key].view_as(e_pred)
        groups = batch["graphs", group_key]
        return mse(center_by_group(e_pred, groups), center_by_group(e_true, groups))

    return _fn


def parameter_bag_mse(
    pred_bags: Mapping[str, torch.Tensor],
    ref_bags: Mapping[str, torch.Tensor],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Optional B1 diagnostic: MSE between predicted and reference parameter tensors.

    Args:
        pred_bags: Mapping name → predicted parameter tensor.
        ref_bags: Mapping name → reference parameter tensor (same keys/shapes).
        reduction: ``mean`` / ``sum`` / ``none`` over concatenated squared errors.

    Returns:
        Scalar (or unreduced) MSE over all bag entries.
    """
    sq: list[torch.Tensor] = []
    for key, pref in pred_bags.items():
        if key not in ref_bags:
            raise KeyError(f"ref_bags missing key {key!r}")
        ref = ref_bags[key]
        if pref.shape != ref.shape:
            raise ValueError(f"bag {key!r} shape {pref.shape} != ref {ref.shape}")
        sq.append((pref - ref).reshape(-1).pow(2))
    if not sq:
        return torch.zeros(())
    cat = torch.cat(sq)
    if reduction == "mean":
        return cat.mean()
    if reduction == "sum":
        return cat.sum()
    if reduction == "none":
        return cat
    raise ValueError(f"Unknown reduction {reduction!r}")


def energy_mse(
    target_key: str = "energy",
    *,
    pred_key: str = "energy",
    reduction: str = "mean",
    per_atom: bool = False,
    num_atoms_key: str = "num_atoms",
) -> Callable[[Mapping[str, Any], Any], torch.Tensor]:
    """MSE between predicted energy and a graph-level target.

    Args:
        target_key: Graph-level target name stored at
            ``batch["graphs", target_key]`` by the collator.
        pred_key: Key under which the model forward writes the energy
            prediction in its output dict.
        reduction: ``'mean'``, ``'sum'``, or ``'none'`` — forwarded to
            :class:`torch.nn.MSELoss`.
        per_atom: When ``True``, divide the energy residual by each graph's
            atom count before squaring, so the loss is in (energy/atom)²
            and large molecules don't dominate variable-size batches. The
            atom count is read from ``batch["graphs", num_atoms_key]``.
            Leave ``False`` to match references that fit total-energy MSE
            (e.g. fixed-composition rMD17).
        num_atoms_key: Graph-level key holding the per-graph atom count,
            used only when ``per_atom`` is set.

    Returns:
        ``loss_fn(preds, batch) -> Tensor`` suitable for the Trainer's
        ``step.loss_fn`` slot.
    """
    mse = nn.MSELoss(reduction=reduction)

    def _fn(preds: Mapping[str, Any], batch: Any) -> torch.Tensor:
        e_pred = preds[pred_key]
        e_true = batch["graphs", target_key].view_as(e_pred)
        if per_atom:
            n_atoms = batch["graphs", num_atoms_key].view_as(e_pred).to(e_pred.dtype)
            scale = n_atoms.clamp(min=1)
            return mse(e_pred / scale, e_true / scale)
        return mse(e_pred, e_true)

    return _fn


def energy_force_mse(
    *,
    energy_target_key: str = "energy",
    force_target_key: str = "forces",
    energy_pred_key: str = "energy",
    force_pred_key: str = "forces",
    lambda_F: float = 1.0,
    reduction: str = "mean",
    per_atom: bool = False,
    num_atoms_key: str = "num_atoms",
) -> Callable[[Mapping[str, Any], Any], torch.Tensor]:
    """Energy MSE + ``lambda_F`` × forces MSE.

    The energy target lives at ``batch["graphs", energy_target_key]``
    (graph-level scalar). The forces target lives at
    ``batch["atoms", force_target_key]`` (atom-level vector, shape
    ``(N_total, 3)``).

    Args:
        energy_target_key: Graph-level energy target key in the batch.
        force_target_key: Atom-level force target key in the batch.
        energy_pred_key: Energy key on the model-forward output dict.
        force_pred_key: Force key on the model-forward output dict.
        lambda_F: Weight on the force-MSE term (the literature ``rho``).
            The force MSE is already per-atom (atom-level vectors), so it is
            directly comparable across system sizes; pair with
            ``per_atom=True`` to put the energy term on the same footing.
        reduction: Forwarded to both :class:`torch.nn.MSELoss` instances.
        per_atom: Normalize the energy residual by atom count — see
            :func:`energy_mse`. The force term is unaffected.
        num_atoms_key: Graph-level per-graph atom-count key for ``per_atom``.

    Returns:
        ``loss_fn(preds, batch) -> Tensor``.
    """
    energy_fn = energy_mse(
        energy_target_key,
        pred_key=energy_pred_key,
        reduction=reduction,
        per_atom=per_atom,
        num_atoms_key=num_atoms_key,
    )
    mse_f = nn.MSELoss(reduction=reduction)

    def _fn(preds: Mapping[str, Any], batch: Any) -> torch.Tensor:
        f_pred = preds[force_pred_key]
        f_true = batch["atoms", force_target_key].view_as(f_pred)
        return energy_fn(preds, batch) + lambda_F * mse_f(f_pred, f_true)

    return _fn


class ProjectedInternalHessianLoss(nn.Module):
    """MSE on a Wilson-projected internal Hessian (Class-I force constants).

    Projects a Cartesian Hessian ``H_x`` onto internal coordinates with the
    Wilson ``B`` matrix, then compares either ``diag(H_q)`` to Class-I force
    constants (``target_kind="diagonal"``) or the full ``H_q`` to a supplied
    reference (``target_kind="hq_ref"``). Diagonal mode does **not** penalize
    off-diagonal (Class-II) couplings toward zero.

    Units: ``H_x`` / ``H_q`` match the internal coordinate of each ``B`` row —
    kcal mol⁻¹ Å⁻² (bonds) or kcal mol⁻¹ rad⁻² (angles). Force-constant
    targets use the same units. Cartesian ``H_x`` is never flattened.

    Reference:
        Wilson, E. B. *J. Chem. Phys.* **9**, 76 (1941).
        https://doi.org/10.1063/1.1750829
    """

    def __init__(
        self,
        *,
        target_kind: Literal["diagonal", "hq_ref"] = "diagonal",
        hessian_key: str = "hessian",
        b_matrix_key: str = "B",
        diagonal_key: str = "diagonal",
        hq_ref_key: str = "hq_ref",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if target_kind not in ("diagonal", "hq_ref"):
            raise ValueError(f"target_kind must be 'diagonal' or 'hq_ref', got {target_kind!r}")
        self.target_kind = target_kind
        self.hessian_key = hessian_key
        self.b_matrix_key = b_matrix_key
        self.diagonal_key = diagonal_key
        self.hq_ref_key = hq_ref_key
        self.mse = nn.MSELoss(reduction=reduction)

    @staticmethod
    def project(cartesian_hessian: torch.Tensor, b_matrix: torch.Tensor) -> torch.Tensor:
        """Project ``H_x`` to internal coordinates: ``H_q = (B⁺)ᵀ H_x B⁺``.

        Args:
            cartesian_hessian: Cartesian Hessian ``H_x`` of shape ``(3N, 3N)``,
                kcal mol⁻¹ Å⁻².
            b_matrix: Wilson ``B = ∂q/∂x`` of shape ``(n_q, 3N)``.

        Returns:
            Internal Hessian ``H_q`` of shape ``(n_q, n_q)``.

        Raises:
            ValueError: If ``H_x`` is not square or ``B`` column count ≠ ``3N``.
        """
        if cartesian_hessian.ndim != 2 or cartesian_hessian.shape[0] != cartesian_hessian.shape[1]:
            raise ValueError(
                f"cartesian_hessian must be square (3N, 3N); got {tuple(cartesian_hessian.shape)}"
            )
        n3 = cartesian_hessian.shape[0]
        if b_matrix.ndim != 2 or b_matrix.shape[1] != n3:
            raise ValueError(f"b_matrix must be (n_q, 3N={n3}); got {tuple(b_matrix.shape)}")
        b_pinv = torch.linalg.pinv(b_matrix)
        return b_pinv.T @ cartesian_hessian @ b_pinv

    def forward(self, pred: Mapping[str, Any], target: Mapping[str, Any]) -> torch.Tensor:
        """Project ``pred[hessian_key]`` and MSE against the chosen target.

        Args:
            pred: Mapping with Cartesian Hessian at ``hessian_key``.
            target: Mapping with Wilson ``B`` at ``b_matrix_key`` and either
                force constants ``(n_q,)`` at ``diagonal_key`` or a full
                ``H_q`` at ``hq_ref_key``.

        Returns:
            Scalar MSE (kcal mol⁻¹ Å⁻² or kcal mol⁻¹ rad⁻², squared).
        """
        h_q = self.project(pred[self.hessian_key], target[self.b_matrix_key])
        if self.target_kind == "diagonal":
            return self.mse(torch.diag(h_q), target[self.diagonal_key])
        return self.mse(h_q, target[self.hq_ref_key])


class TorsionProfileLoss(nn.Module):
    """Molecule-centered MSE on a torsion energy profile, optional barrier term.

    Predictions and references are centered independently with
    :func:`center_by_group` before MSE, so a constant energy shift per
    molecule / scan is invisible. An optional barrier term compares
    ``max E − min E`` per group.

    Units: kcal/mol (profile and barrier heights).
    """

    def __init__(
        self,
        *,
        pred_key: str = "energy",
        target_key: str = "energy",
        group_key: str = "molecule_id_index",
        barrier_weight: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.pred_key = pred_key
        self.target_key = target_key
        self.group_key = group_key
        self.barrier_weight = float(barrier_weight)
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred: Mapping[str, Any], target: Mapping[str, Any]) -> torch.Tensor:
        """Centered profile MSE plus ``barrier_weight`` × barrier MSE.

        Args:
            pred: Mapping with energies at ``pred_key``, kcal/mol.
            target: Mapping with reference energies at ``target_key``
                (kcal/mol) and integer group ids at ``group_key``.

        Returns:
            Scalar loss in (kcal/mol)².
        """
        e_pred = pred[self.pred_key]
        e_true = target[self.target_key]
        groups = target[self.group_key]
        loss = self.mse(center_by_group(e_pred, groups), center_by_group(e_true, groups))
        if self.barrier_weight == 0.0:
            return loss
        return loss + self.barrier_weight * self.mse(
            self._barrier_heights(e_pred, groups),
            self._barrier_heights(e_true, groups),
        )

    def _barrier_heights(self, values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """Per-group ``max − min`` energy; empty and singleton groups are 0."""
        flat = values.reshape(-1)
        ids = group_ids.reshape(-1).long()
        heights: list[torch.Tensor] = []
        for g in ids.unique():
            group_vals = flat[ids == g]
            if group_vals.numel() <= 1:
                heights.append(flat.new_zeros(()))
            else:
                heights.append(group_vals.max() - group_vals.min())
        if not heights:
            return flat.new_zeros((0,))
        return torch.stack(heights)


class ChargeMSELoss(nn.Module):
    """Charge-only MSE against a partitioning reference (MBIS / AM1-BCC).

    Compares ``pred[pred_key]`` to ``target[target_key]``. Does not read
    energy keys, does not penalize net charge, and does not couple ``E(q)``.

    Units: elementary charge ``e``.
    """

    def __init__(
        self,
        pred_key: str = "q",
        target_key: str = "q",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.mse = MSELoss(pred_key=pred_key, target_key=target_key, reduction=reduction)

    def forward(self, pred: Any, target: Any) -> torch.Tensor:
        """MSE between predicted and reference partial charges.

        Args:
            pred: Mapping (or tensor) of predicted charges, ``e``.
            target: Mapping (or tensor) of reference charges, ``e``.

        Returns:
            Scalar MSE in ``e²``.
        """
        return self.mse(pred, target)


class ScaleNormalizedLoss(nn.Module):
    """Wrap a loss as the dimensionless ``L / s²`` used by ``WeightedLoss``.

    ``scale`` is a registered buffer (moves with ``.to()`` / ``state_dict``),
    not an ``nn.Parameter``. The caller supplies the training-set statistic
    ``s``; this module does not scan a dataset.

    Units: ``scale`` has the same physical unit as the inner residual so
    that ``L / scale²`` is dimensionless.
    """

    def __init__(self, loss: nn.Module, scale: float) -> None:
        super().__init__()
        if scale == 0:
            raise ValueError("scale must be non-zero")
        self.loss = loss
        self.register_buffer("scale", torch.tensor(float(scale), dtype=config.ftype))
        self.scale: torch.Tensor

    def forward(self, pred: Any, target: Any) -> torch.Tensor:
        """Return ``loss(pred, target) / scale²``.

        Args:
            pred: Forwarded to the inner loss.
            target: Forwarded to the inner loss.

        Returns:
            Dimensionless scaled loss.
        """
        return self.loss(pred, target) / self.scale.square()
