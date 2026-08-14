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
    "ParsleyEnergyWeight",
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
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Subtract per-group means from *values* (Espaloma molecule centering).

    With ``weights=None`` each group subtracts its plain mean. With *weights*
    supplied each group subtracts its weight-normalized mean ``Σwv / Σw``,
    which is the offset that minimizes the weighted squared residual
    (``argmin_c Σ w (v − c)² = Σwv / Σw``) and degenerates to the plain mean
    when all weights are equal. Zero-weight entries then drop out of the
    offset as well as of the objective.

    Args:
        values: Scalar energies ``(B,)`` or ``(B, 1)``, kcal/mol.
        group_ids: Integer molecule / group ids ``(B,)`` aligned with *values*.
        weights: Optional non-negative per-configuration weights ``(B,)``
            (dimensionless, e.g. from :class:`ParsleyEnergyWeight`). Each
            group's weights must sum to a non-zero value.

    Returns:
        Centered values with the same shape as *values*.

    Raises:
        ValueError: If *group_ids* or *weights* length differs from *values*.
    """
    flat = values.reshape(-1)
    ids = group_ids.reshape(-1).long()
    if flat.numel() != ids.numel():
        raise ValueError(f"values and group_ids length mismatch: {flat.numel()} vs {ids.numel()}")
    out = flat.clone()
    if weights is None:
        for g in ids.unique():
            mask = ids == g
            out[mask] = flat[mask] - flat[mask].mean()
        return out.view_as(values)
    w = weights.reshape(-1).to(flat.dtype)
    if w.numel() != flat.numel():
        raise ValueError(f"values and weights length mismatch: {flat.numel()} vs {w.numel()}")
    for g in ids.unique():
        mask = ids == g
        w_g = w[mask]
        total = w_g.sum()
        if total == 0:
            raise ValueError(
                "center_by_group: weights of one group sum to zero — the "
                "weighted group mean is undefined; supply at least one "
                "non-zero weight per group."
            )
        out[mask] = flat[mask] - (w_g * flat[mask]).sum() / total
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


class ParsleyEnergyWeight(nn.Module):
    """Parsley per-configuration weight from a relative in-scan energy.

    Maps each configuration's relative QM energy ``ΔE = E_QM − min_scan E_QM``
    (kcal/mol) to a dimensionless weight, so chemically irrelevant high-energy
    grid points stop counting as much as the barrier region:

    .. code-block:: text

        w(ΔE) = 1                                        ΔE < denom
        w(ΔE) = denom / sqrt(denom² + (ΔE − denom)²)     denom ≤ ΔE < upper
        w(ΔE) = 0                                        ΔE ≥ upper

    At the default ``denom = 1``, ``upper = 5`` kcal/mol the middle branch is
    ``w(ΔE) = (1 + (ΔE − 1)²)^(−1/2)``: continuous at ``ΔE = denom`` (``w = 1``),
    positive, and monotonically decreasing to the ``upper`` cutoff.

    Interpretation note: the Parsley requirement text writes the middle branch
    as ``1/(1+(E−1)²)−½``; the trailing ``−½`` is read here as the **exponent**.
    The literal "minus one half" reading is discontinuous at ``ΔE = 1``
    (jumps ``1 → 0.5``) and goes **negative** above 2 kcal/mol, which would
    reward increasing the error on those configurations — not a plausible
    objective. The exponent reading also matches the ``energy_denom`` /
    ``energy_upper`` attenuation of the torsion-profile targets that produced
    the published numbers.

    Weights are constants of the objective, never a gradient path: the input is
    detached and the returned tensor has ``requires_grad=False``. It is returned
    in the **input** dtype; ``denom`` / ``upper`` are non-learnable buffers
    registered at :data:`~molix.config.config` precision so they ride ``.to()``
    and ``state_dict``.

    Units: ``denom`` / ``upper`` and the input ``ΔE`` are kcal/mol; the weights
    are dimensionless. The eq.-3 ``1/dE²`` prefactor is **not** applied here —
    compose it with :class:`ScaleNormalizedLoss` (``scale=dE``) instead.

    Reference:
        Qiu et al. "Development and Benchmarking of Open Force Field v1.0.0 —
        the Parsley Small-Molecule Force Field." *J. Chem. Theory Comput.*
        **17**, 6262-6280 (2021), §2.1.7 eqs. 3-5.
        https://doi.org/10.1021/acs.jctc.1c00571

        Boothroyd et al. "Development and Benchmarking of Open Force Field
        2.0.0 — the Sage Small Molecule Force Field." *J. Chem. Theory Comput.*
        **19**, 3251-3275 (2023). https://doi.org/10.1021/acs.jctc.3c00039
    """

    denom: torch.Tensor
    upper: torch.Tensor

    def __init__(self, *, denom: float = 1.0, upper: float = 5.0) -> None:
        super().__init__()
        if denom <= 0:
            raise ValueError(f"denom must be positive (kcal/mol), got {denom!r}")
        if upper < denom:
            raise ValueError(f"upper must be >= denom, got upper={upper!r} denom={denom!r}")
        self.register_buffer("denom", torch.tensor(float(denom), dtype=config.ftype))
        self.register_buffer("upper", torch.tensor(float(upper), dtype=config.ftype))

    def forward(self, relative_energies: torch.Tensor) -> torch.Tensor:
        """Evaluate the three-branch weight on relative in-scan energies.

        Args:
            relative_energies: Non-negative ``ΔE = E_QM − min_scan E_QM``
                of any shape, kcal/mol.

        Returns:
            Weights with the same shape and dtype as *relative_energies*,
            dimensionless, detached (``requires_grad=False``).

        Raises:
            ValueError: If any entry is negative — that means absolute rather
                than relative energies were passed, which would silently weight
                every configuration at 1.
        """
        delta = relative_energies.detach()
        if bool((delta < 0).any()):
            raise ValueError(
                "relative in-scan energies expected (dE = E_QM - min_scan E_QM >= 0); "
                f"got a negative minimum {float(delta.min())!r}"
            )
        denom = self.denom.to(dtype=delta.dtype, device=delta.device)
        upper = self.upper.to(dtype=delta.dtype, device=delta.device)
        decay = denom / torch.sqrt(denom.square() + (delta - denom).square())
        w = torch.where(delta < denom, torch.ones_like(delta), decay)
        return torch.where(delta >= upper, torch.zeros_like(delta), w)


class TorsionProfileLoss(nn.Module):
    """Molecule-centered MSE on a torsion energy profile, optional barrier term.

    Predictions and references are centered independently with
    :func:`center_by_group` before MSE, so a constant energy shift per
    molecule / scan is invisible. An optional barrier term compares
    ``max E − min E`` per group.

    Supplying ``energy_weight`` (e.g. :class:`ParsleyEnergyWeight`) switches
    the profile term to the Parsley per-configuration weighted form: each scan
    contributes ``Σ w r² / Σ w`` — one ``O(1)`` number regardless of its grid
    count — and the centering offset becomes the weighted group mean, so
    zero-weight configurations drop out of the numerator, the denominator, and
    the offset alike. The default ``None`` leaves the plain path untouched.

    Units: kcal/mol (profile and barrier heights).

    Reference:
        Qiu et al. *J. Chem. Theory Comput.* **17**, 6262-6280 (2021),
        §2.1.7 eqs. 3-5. https://doi.org/10.1021/acs.jctc.1c00571
    """

    def __init__(
        self,
        *,
        pred_key: str = "energy",
        target_key: str = "energy",
        group_key: str = "molecule_id_index",
        barrier_weight: float = 0.0,
        reduction: str = "mean",
        energy_weight: nn.Module | None = None,
    ) -> None:
        """Configure the torsion-profile objective.

        Args:
            pred_key: Key of the predicted scan energies in ``pred``
                (``(C,)``, kcal/mol).
            target_key: Key of the reference scan energies in ``target``
                (``(C,)``, kcal/mol).
            group_key: Key of the per-configuration scan/group ids in
                ``target`` (``(C,)``, integer).
            barrier_weight: Weight of the additive, unweighted barrier-height
                MSE term (dimensionless multiplier; 0 disables it).
            reduction: ``"mean"`` / ``"sum"`` / ``"none"``. With
                ``energy_weight`` set the reduction runs over profiles, not
                configurations.
            energy_weight: Optional per-configuration weight policy (e.g.
                :class:`ParsleyEnergyWeight`) mapping relative in-scan
                reference energies (kcal/mol) to dimensionless weights;
                ``None`` keeps the plain mean-centered MSE path.
        """
        super().__init__()
        self.pred_key = pred_key
        self.target_key = target_key
        self.group_key = group_key
        self.barrier_weight = float(barrier_weight)
        self.mse = nn.MSELoss(reduction=reduction)
        self.energy_weight = energy_weight

    def forward(self, pred: Mapping[str, Any], target: Mapping[str, Any]) -> torch.Tensor:
        """Centered profile term plus ``barrier_weight`` × barrier MSE.

        Without ``energy_weight`` the profile term is the plain mean-centered
        MSE reduced over **configurations**. With ``energy_weight`` set it is
        the per-profile ``Σ w r² / Σ w`` vector reduced by the same ``reduction``
        string over **profiles** (``"mean"`` / ``"sum"`` → scalar, ``"none"`` →
        ``(G,)``) — that profile-level reduction is what balances scans of
        different grid counts. The barrier term is unchanged and stays
        **unweighted** (it is already one scalar per group), and remains
        additive exactly as before.

        Args:
            pred: Mapping with energies at ``pred_key``, kcal/mol.
            target: Mapping with reference energies at ``target_key``
                (kcal/mol) and integer group ids at ``group_key``.

        Returns:
            Loss in (kcal/mol)² — scalar, or ``(G,)`` with ``reduction="none"``.
        """
        e_pred = pred[self.pred_key]
        e_true = target[self.target_key]
        groups = target[self.group_key]
        if self.energy_weight is None:
            loss = self.mse(center_by_group(e_pred, groups), center_by_group(e_true, groups))
        else:
            loss = self._weighted_profile(e_pred, e_true, groups)
        if self.barrier_weight == 0.0:
            return loss
        return loss + self.barrier_weight * self.mse(
            self._barrier_heights(e_pred, groups),
            self._barrier_heights(e_true, groups),
        )

    def _weighted_profile(
        self,
        e_pred: torch.Tensor,
        e_true: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-profile ``Σ w r² / Σ w``, reduced by ``self.mse.reduction``."""
        flat_pred = e_pred.reshape(-1)
        flat_true = e_true.reshape(-1)
        ids = group_ids.reshape(-1).long()
        weights = self.energy_weight(self._relative_to_group_min(flat_true.detach(), ids))
        weights = weights.to(flat_pred.dtype)
        resid = center_by_group(flat_pred, ids, weights=weights) - center_by_group(
            flat_true, ids, weights=weights
        )
        terms: list[torch.Tensor] = []
        for g in ids.unique():
            mask = ids == g
            w_g = weights[mask]
            terms.append((w_g * resid[mask].square()).sum() / w_g.sum())
        per_profile = torch.stack(terms) if terms else flat_pred.new_zeros((0,))
        reduction = self.mse.reduction
        if reduction == "mean":
            return per_profile.mean()
        if reduction == "sum":
            return per_profile.sum()
        if reduction == "none":
            return per_profile
        raise ValueError(f"Unknown reduction {reduction!r}")

    def _relative_to_group_min(self, values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """Per-group ``value − min(group)`` — relative in-scan energies, kcal/mol."""
        flat = values.reshape(-1)
        ids = group_ids.reshape(-1).long()
        if flat.numel() != ids.numel():
            raise ValueError(
                f"values and group_ids length mismatch: {flat.numel()} vs {ids.numel()}"
            )
        out = flat.clone()
        for g in ids.unique():
            mask = ids == g
            out[mask] = flat[mask] - flat[mask].min()
        return out.view_as(values)

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
