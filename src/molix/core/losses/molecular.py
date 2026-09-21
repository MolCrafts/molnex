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
the collate's per-atom routing:

* graph-level targets (e.g. ``energy``) live at ``batch["graphs", key]``;
* atom-level targets (e.g. ``forces``) live at ``batch["atoms", key]``.

The factories do **not** auto-discover the schema — the caller passes the
exact target key name, so the resulting callable is a pure function with
static shape/key access (keeps ``torch.compile`` graphs stable).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch
import torch.nn as nn

from molix.config import config
from molix.core.losses.energy import MSELoss

__all__ = [
    "AtomicChargeLoss",
    "BoltzmannProfileLoss",
    "ChargeMSELoss",
    "ParsleyEnergyWeight",
    "PartialHessianLoss",
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


def energy_loss(
    target_key: str = "energy",
    *,
    pred_key: str = "energy",
    elementwise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    reduction: str = "mean",
    per_atom: bool = False,
    num_atoms_key: str = "num_atoms",
    weight_key: str | None = "mask",
) -> Callable[[Mapping[str, Any], Any], torch.Tensor]:
    """Graph-level energy loss under a caller-chosen elementwise measure.

    Three things are varied independently here, and this is the one that says
    *how a residual is measured*. The other two are the per-atom weight (see
    :func:`force_loss`) and how several terms combine
    (:class:`~molix.core.losses.WeightedLoss`).

    ``elementwise`` is any callable that maps ``(pred, target)`` to a tensor of
    **per-element** residuals — it must not reduce, because the reduction is
    this function's (a weight cannot be applied after a mean). Torch's own loss
    modules satisfy it at ``reduction="none"``, which is the point: squared
    error is ``nn.MSELoss(reduction="none")`` and a Huber is
    ``nn.HuberLoss(delta=…, reduction="none")``. molnex adds no loss class of
    its own.

    **When a Huber earns its keep.** Its gradient is bounded past ``delta``,
    so a residual 100× the typical one contributes 1×, not 10⁴×. That matters
    wherever the target has a tail the model's functional form cannot follow —
    a Class-I force field against DFT forces whose mean is 20 kcal mol⁻¹ Å⁻¹
    and whose max is 250, say, where a squared error spends the batch chasing
    what cannot be fitted.

    Args:
        target_key: Graph-level target at ``batch["graphs", target_key]``.
        pred_key: Key the model's forward writes its energy under.
        elementwise: Per-element residual measure; ``None`` is
            ``nn.MSELoss(reduction="none")``.
        reduction: ``'mean'`` / ``'sum'`` / ``'none'``, applied here.
        per_atom: Divide the residual by each graph's atom count first, so the
            loss is per atom and large molecules do not dominate a
            variable-size batch.
        num_atoms_key: Per-graph atom count, read only when ``per_atom``.
        weight_key: Per-graph weight column on ``graphs``, or ``None`` for no
            weighting — the same road :func:`force_loss` takes on ``atoms``.
            A :class:`~molix.data.MoleculePadder` with ``max_graphs`` writes
            ``graphs.mask`` (False on its dead slots), so under ``'mean'``
            the divisor is the live graph count and a padded batch scores
            exactly as the unpadded one. Absent from the batch means
            unweighted.

    Returns:
        ``loss_fn(preds, batch) -> Tensor``.
    """
    measure = nn.MSELoss(reduction="none") if elementwise is None else elementwise

    def _fn(preds: Mapping[str, Any], batch: Any) -> torch.Tensor:
        e_pred = preds[pred_key]
        graphs = batch["graphs"]
        e_true = graphs[target_key].view_as(e_pred)
        if per_atom:
            scale = graphs[num_atoms_key].view_as(e_pred).to(e_pred.dtype).clamp(min=1)
            e_pred, e_true = e_pred / scale, e_true / scale
        weight = None
        if weight_key is not None and weight_key in graphs.keys():
            weight = graphs[weight_key].view_as(e_pred).to(e_pred.dtype)
        return _reduce(measure(e_pred, e_true), weight, reduction)

    return _fn


def force_loss(
    target_key: str = "forces",
    *,
    pred_key: str = "forces",
    elementwise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    reduction: str = "mean",
    weight_key: str | None = "mask",
    outlier_norm: float | None = None,
) -> Callable[[Mapping[str, Any], Any], torch.Tensor]:
    """Atom-level force loss, weighted by a per-atom column.

    The weight is the second of the three axes, and it has exactly one road:
    a per-atom column on ``atoms``, multiplied into the residual before the
    reduction divides by its sum. Padding ghosts and reference outliers are the
    same thing to this function — a row that should not count — so neither
    needs its own code path, and neither is ever a boolean gather (that is a
    data-dependent shape, which ``torch.compile`` cannot trace).

    Args:
        target_key: Atom-level target at ``batch["atoms", target_key]``.
        pred_key: Key the model's forward writes its forces under.
        elementwise: Per-element residual measure; ``None`` is
            ``nn.MSELoss(reduction="none")``. See :func:`energy_loss`.
        reduction: ``'mean'`` / ``'sum'`` / ``'none'``. Under ``'mean'`` the
            divisor is the **weighted** element count, so the value does not
            move with how many rows a batch happens to mask out.
        weight_key: Per-atom weight column on ``atoms``, or ``None`` for no
            weighting. Absent from the batch means unweighted.
        outlier_norm: Drop an atom whose **reference** force norm exceeds this,
            in the reference's own unit, by multiplying its weight by zero.
            ``None`` keeps every atom. A Class-I functional form cannot
            represent a 200-800 kcal mol⁻¹ Å⁻¹ DFT tail, so fitting one moves
            every parameter that can reach it; the bound says so declaratively
            instead of a caller re-deriving the same mask. It is a weight and
            not a gather for the reason above — and because a gather makes the
            mean's divisor the number of survivors, so the loss scale would move
            with how many outliers a batch happened to draw.

    Returns:
        ``loss_fn(preds, batch) -> Tensor``.
    """
    measure = nn.MSELoss(reduction="none") if elementwise is None else elementwise
    bound = None if outlier_norm is None else float(outlier_norm)

    def _fn(preds: Mapping[str, Any], batch: Any) -> torch.Tensor:
        f_pred = preds[pred_key]
        atoms = batch["atoms"]
        f_true = atoms[target_key].view_as(f_pred)
        weight = None
        if weight_key is not None and weight_key in atoms.keys():
            weight = atoms[weight_key].to(f_pred.dtype).unsqueeze(-1)
        if bound is not None:
            inlier = (f_true.detach().norm(dim=-1) <= bound).to(f_pred.dtype).unsqueeze(-1)
            weight = inlier if weight is None else weight * inlier
        return _reduce(measure(f_pred, f_true), weight, reduction)

    return _fn


def _reduce(per_element: torch.Tensor, weight: torch.Tensor | None, reduction: str) -> torch.Tensor:
    """Reduce per-element residuals, dividing a mean by the weight it kept."""
    if weight is not None:
        per_element = per_element * weight
    if reduction == "none":
        return per_element
    total = per_element.sum()
    if reduction == "sum":
        return total
    if weight is None:
        return per_element.mean()
    # Elements per weighted row: 3 for (N, 3) forces, 1 for (B,) energies.
    kept = weight.sum() * math.prod(per_element.shape[1:])
    return total / kept.clamp(min=1.0)


def energy_mse(
    target_key: str = "energy",
    *,
    pred_key: str = "energy",
    reduction: str = "mean",
    per_atom: bool = False,
    num_atoms_key: str = "num_atoms",
    weight_key: str | None = "mask",
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
        weight_key: Per-graph weight column on ``graphs`` (``graphs.mask``
            on a padded batch), or ``None``; see :func:`energy_loss`.

    Returns:
        ``loss_fn(preds, batch) -> Tensor`` suitable for the Trainer's
        ``step.loss_fn`` slot.
    """
    return energy_loss(
        target_key,
        pred_key=pred_key,
        reduction=reduction,
        per_atom=per_atom,
        num_atoms_key=num_atoms_key,
        weight_key=weight_key,
    )


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

    On a padded batch (:class:`molix.data.MoleculePadder`, which writes a
    boolean ``atoms.mask``) the force term runs over the **real** atoms only:
    the squared residual is multiplied by the mask and, under ``"mean"``,
    divided by ``3 · mask.sum()`` — the real atom count — rather than by
    ``3 · max_atoms``, so the value does not depend on how much padding a batch
    happens to carry. Ghost rows are a multiply by zero, never a slice, so the
    loss keeps its static shape under ``torch.compile``.

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
    energy_fn = energy_loss(
        energy_target_key,
        pred_key=energy_pred_key,
        reduction=reduction,
        per_atom=per_atom,
        num_atoms_key=num_atoms_key,
    )
    force_fn = force_loss(force_target_key, pred_key=force_pred_key, reduction=reduction)

    def _fn(preds: Mapping[str, Any], batch: Any) -> torch.Tensor:
        return energy_fn(preds, batch) + lambda_F * force_fn(preds, batch)

    return _fn


class ProjectedInternalHessianLoss(nn.Module):
    """MSE on a Wilson-projected internal Hessian (Class-I force constants).

    Projects a Cartesian Hessian ``H_x`` onto internal coordinates with the
    Wilson ``B`` matrix, then compares either ``diag(H_q)`` to Class-I force
    constants (``target_kind="diagonal"``) or the full ``H_q`` to a supplied
    reference (``target_kind="hq_ref"``). Diagonal mode does **not** penalize
    off-diagonal (Class-II) couplings toward zero.

    Layout: ``H_x`` and every ``B`` row use the flat Cartesian index
    ``i = 3a + c`` for atom ``a`` and component ``c ∈ {x, y, z}`` — i.e.
    ``x0, y0, z0, x1, …`` — which is the layout ``graphs.hessian`` ``(3N, 3N)``
    carries through the batch. This ordering is the contract between the
    Hessian producer (:class:`molpot.derivation.HessianDerivation`) and this
    consumer; a transposed or atom-major layout silently yields wrong force
    constants.

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
        pred_key: str = "charge",
        target_key: str = "charge",
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


class PartialHessianLoss(nn.Module):
    """Trace-relative MAE on the partial Hessian — ByteFF's first stage.

    The target is not the whole ``(3N, 3N)`` Hessian but the ``3x3`` mixed blocks
    ``d2E / dx_a dx_b`` of the atom pairs that share a bonded term, deduplicated
    into the ``hessian_pairs`` namespace by
    :class:`~molpot.forcefield.partial_hessian.PartialHessianPairs`. The quantum
    block is the ``block`` column; the molecular-mechanics block comes from
    :meth:`~molpot.forcefield.potentials.Potentials.partial_hessian`.

    **Why relative, and why relative to the trace.** Curvature spans orders of
    magnitude across a molecule — a C-H stretch against a torsional softness —
    so an absolute residual would fit the stiff pairs and ignore everything else.
    Each block is divided by ``|trace|`` of its own quantum block, which is the
    rotationally invariant size of that block, floored at *trace_floor* so a pair
    with almost no curvature cannot dominate by dividing by nearly zero.

    **Why a blown-up prediction is dropped, not clipped.** Early in a fit an
    unconstrained head can put an absurd constant on one term; clipping would
    keep its gradient and let it steer the batch, and letting it through would
    swamp the mean. ByteFF drops the row and logs the molecule. This drops the
    row by multiplying a mask, with no host sync: the threshold test is a tensor
    reduction, never a ``where`` over indices.

    Padded pair rows are excluded by the namespace's own ``mask``. A dead row's
    label is zero and its prediction is zero, so it adds nothing to the numerator
    — but it would still count in a plain ``mean()``, making the loss shrink in
    proportion to how empty the batch happened to be.

    Units are the Hessian's: kcal mol⁻¹ Å⁻². *trace_floor* is therefore
    dimensionful, and its value is ByteFF's.

    Args:
        key: Prediction key holding the ``(n_pairs, 9)`` MM blocks.
        label: Column on ``hessian_pairs`` holding the quantum blocks.
        trace_floor: Lower clamp on ``|trace|`` of a label block,
            kcal mol⁻¹ Å⁻².
        blowup: A predicted block with any entry larger than this in magnitude
            drops its row, kcal mol⁻¹ Å⁻².

    Reference:
        Zheng, T. et al. "Data-driven parametrization of molecular mechanics
        force fields for expansive chemical space coverage." *Chem. Sci.* (2025).
        https://doi.org/10.48550/arXiv.2408.12817
    """

    def __init__(
        self,
        *,
        key: str = "partial_hessian",
        label: str = "block",
        trace_floor: float = 10.0,
        blowup: float = 1e4,
    ) -> None:
        super().__init__()
        if trace_floor <= 0.0:
            raise ValueError(f"trace_floor must be > 0 kcal/mol/A^2, got {trace_floor}")
        if blowup <= 0.0:
            raise ValueError(f"blowup must be > 0 kcal/mol/A^2, got {blowup}")
        self.key = str(key)
        self.label = str(label)
        self.trace_floor = float(trace_floor)
        self.blowup = float(blowup)

    def forward(self, pred: Mapping[str, Any], batch: Any) -> torch.Tensor:
        """Mean trace-relative absolute residual over the live pair blocks.

        Args:
            pred: Mapping carrying ``(n_pairs, 9)`` MM blocks under
                :attr:`key`.
            batch: Post-collate batch carrying ``hessian_pairs`` with the label
                column and, on a padded batch, its ``mask``.

        Returns:
            Scalar loss, dimensionless.

        Raises:
            KeyError: The prediction is absent.
            ValueError: The batch carries no ``hessian_pairs`` namespace or no
                label column on it, or the shapes disagree.
        """
        if self.key not in pred:
            raise KeyError(
                f"{self.key!r} is not among the predictions {sorted(pred)}; declare it on "
                "ForceFieldPredictions so the partial Hessian is computed"
            )
        blocks = pred[self.key]
        if "hessian_pairs" not in batch.keys():
            raise ValueError(
                "PartialHessianLoss: the batch carries no hessian_pairs namespace — "
                "there is nothing to fit against"
            )
        namespace = batch["hessian_pairs"]
        label = namespace.get(self.label)
        if label is None:
            raise ValueError(
                f"hessian_pairs carries no {self.label!r} column; the quantum blocks are "
                "the label this stage fits"
            )
        if blocks.shape != label.shape:
            raise ValueError(
                f"predicted partial Hessian {tuple(blocks.shape)} does not match the label "
                f"{tuple(label.shape)}; both are (n_pairs, 9) in hessian_pairs order"
            )
        label = label.to(blocks.dtype)
        rows = (blocks.abs() <= self.blowup).all(dim=-1).to(blocks.dtype)
        mask = namespace.get("mask")
        if mask is not None:
            rows = rows * mask.to(blocks.dtype)
        trace = label[:, 0] + label[:, 4] + label[:, 8]
        denominator = trace.abs().clamp(min=self.trace_floor).unsqueeze(-1)
        residual = ((blocks - label).abs() / denominator).sum(dim=-1)
        return (residual * rows).sum() / (rows.sum() * label.shape[1]).clamp(min=1.0)


class AtomicChargeLoss(nn.Module):
    """MSE on per-atom charges against a partitioning reference, on a packed batch.

    :class:`ChargeMSELoss` compares two flat mappings; this reads the reference off
    the batch's ``atoms`` namespace and excludes the padded atoms, which is what a
    capacity-packed batch needs. A ghost atom's label is zero while the charge
    solve still returns *something* for it, so averaging over the ghosts pulls
    every predicted charge toward zero in proportion to how empty the batch was.

    **Why this exists at all.** ByteFF does not train charges — it takes AM1-BCC
    bond-charge corrections as fixed input. Rondo's charges come from a
    charge-equilibration solve with learned electronegativity and hardness
    (spec `rondo.md` departure D1), so they are parameters, and THEMol's MBIS set
    is what determines them. Fitting them against MBIS before the bonded stages
    is what puts Rondo in ByteFF's position of entering the bonded fit with the
    charges already decided.

    **No net-charge penalty.** The KKT solve enforces ``sum q = Q`` exactly as a
    constraint (:class:`~molpot.charge.ElectronegativityEqualization`), so a
    soft penalty here would only fight the constraint's own Lagrange multiplier.

    Units: elementary charge ``e``; the loss is in ``e**2``.

    Args:
        key: Prediction key holding the ``(n_atoms,)`` charges.
        label: Column on ``atoms`` holding the reference charges. THEMol's MBIS
            set publishes them as ``atomic_charge``.
    """

    def __init__(self, *, key: str = "charge", label: str = "atomic_charge") -> None:
        super().__init__()
        self.key = str(key)
        self.label = str(label)

    def forward(self, pred: Mapping[str, Any], batch: Any) -> torch.Tensor:
        """Mean squared charge error over the live atoms.

        Args:
            pred: Mapping carrying ``(n_atoms,)`` charges under :attr:`key`.
            batch: Post-collate batch whose ``atoms`` carries the label column
                and, on a padded batch, its ``mask``.

        Returns:
            Scalar MSE in ``e**2``.

        Raises:
            KeyError: The prediction is absent.
            ValueError: ``atoms`` carries no label column, or the shapes disagree.
        """
        if self.key not in pred:
            raise KeyError(
                f"{self.key!r} is not among the predictions {sorted(pred)}; declare it on "
                "ForceFieldPredictions so the charge solve runs"
            )
        charges = pred[self.key].reshape(-1)
        atoms = batch["atoms"]
        reference = atoms.get(self.label)
        if reference is None:
            raise ValueError(
                f"atoms carries no {self.label!r} column; the partitioned charges are the "
                "label this stage fits"
            )
        reference = reference.reshape(-1).to(charges.dtype)
        if charges.shape != reference.shape:
            raise ValueError(
                f"predicted charges {tuple(charges.shape)} do not match the label "
                f"{tuple(reference.shape)}; both are one value per atom of the batch"
            )
        live = atoms.get("mask")
        weight = torch.ones_like(charges) if live is None else live.reshape(-1).to(charges.dtype)
        return ((charges - reference).pow(2) * weight).sum() / weight.sum().clamp(min=1.0)


class BoltzmannProfileLoss(nn.Module):
    """ByteFF's torsion-stage objective: relative energies, Boltzmann-weighted.

    The energies a QM scan publishes are absolute — a THEMol record sits near
    -503 415 kcal/mol — while the quantity a torsion term has to reproduce is the
    ~10 kcal/mol shape of the profile. Subtracting an atomic reference leaves a
    per-molecule constant far larger than the signal, which is how a validation
    MAE can sit at 231 651 kcal/mol while the training loss moves. This compares
    each conformer to **its own profile's minimum** instead, so the constant never
    enters, and no reference-energy table is needed on this stage at all.

    Three pieces, each doing one job (ByteFF ``TorsionLossType.boltzmann_mse``):

    * **Alignment.** Prediction and label are each shifted by their own profile's
      minimum, so both are relative energies before anything is compared.
    * **The Boltzmann weight.** ``exp(min(0, (clamp - dE) / decay))`` on the
      *lower* of the two aligned energies: conformers within *clamp* of the
      minimum count fully and higher ones decay with scale *decay*. A barrier top
      that only one of the two curves reaches cannot then dominate the fit. The
      weight is detached — it selects what to fit, it is not itself fitted.
    * **The offset.** The weighted mean residual per profile, also detached, is
      added back before squaring. It is the offset that minimises the weighted
      squared residual, so the fit never spends capacity on a constant that a
      torsion term cannot represent anyway.

    The residual is then a plain mean over each profile's live conformers, and a
    plain mean over profiles — so a 24-point scan and a 12-point one weigh the
    same, which per-conformer averaging would not give.

    **Profiles come from ``graphs.profile``**, the batch-local dense index
    :class:`~molix.data.ProfileIndex` writes. Reducing into a buffer of the graph
    axis keeps the shape static and the step free of host syncs; dead graph slots
    carry the last index, which is the padder's ghost sink and never a live
    profile.

    **The saturating variant.** With *soft_max* set, a profile's loss is passed
    through ``tanh(L / soft_max) * soft_max`` before the average — ByteFF's
    ``boltzmann_soft_mse``, which it uses where ``boltzmann_mse`` is not robust
    enough. The transform is applied as a **detached** rescaling, so the reported
    value saturates at *soft_max* and the gradient is attenuated by the same
    factor rather than clipped. It is what off-equilibrium data wants: one frame
    that walked into a ``(sigma/r)**12`` wall then contributes a bounded amount
    instead of the whole batch's gradient. ``None`` (default) is the plain
    quadratic, which is right for a well-behaved torsion scan.

    Units: kcal/mol in, kcal²/mol² out. *clamp*, *decay* and *soft_max* are all
    dimensionful — the first two energies, the last an energy **squared**, since
    it caps a squared residual.

    Args:
        pred_key: Prediction key holding the ``(B,)`` energies.
        target_key: Column on ``graphs`` holding the reference energies.
        profile_key: Column on ``graphs`` holding the batch-local profile index.
        clamp: Energies within this of a profile's minimum keep full weight.
        decay: Decay scale of the weight above *clamp*.
        soft_max: Saturation cap on a profile's loss, kcal² mol⁻². ``None``
            leaves the loss quadratic.

    Reference:
        Zheng, T. et al. "Data-driven parametrization of molecular mechanics
        force fields for expansive chemical space coverage." *Chem. Sci.* (2025).
        https://doi.org/10.48550/arXiv.2408.12817
    """

    def __init__(
        self,
        *,
        pred_key: str = "energy",
        target_key: str = "energy",
        profile_key: str = "profile",
        clamp: float = 2.0,
        decay: float = 2.0,
        soft_max: float | None = None,
    ) -> None:
        super().__init__()
        if decay <= 0.0:
            raise ValueError(f"decay must be > 0 kcal/mol, got {decay}")
        if soft_max is not None and soft_max <= 0.0:
            raise ValueError(f"soft_max must be > 0 (kcal/mol)^2, got {soft_max}")
        self.pred_key = str(pred_key)
        self.target_key = str(target_key)
        self.profile_key = str(profile_key)
        self.clamp = float(clamp)
        self.decay = float(decay)
        self.soft_max = None if soft_max is None else float(soft_max)

    def forward(self, pred: Mapping[str, Any], batch: Any) -> torch.Tensor:
        """Mean over profiles of the weighted, offset-free squared residual.

        Args:
            pred: Mapping carrying ``(B,)`` energies under :attr:`pred_key`.
            batch: Post-collate batch whose ``graphs`` carries the reference
                energies, the profile index and, when padded, ``mask``.

        Returns:
            Scalar loss in kcal² mol⁻².

        Raises:
            KeyError: The prediction is absent.
            ValueError: ``graphs`` carries no reference or no profile column, or
                the shapes disagree.
        """
        if self.pred_key not in pred:
            raise KeyError(f"{self.pred_key!r} is not among the predictions {sorted(pred)}")
        energy = pred[self.pred_key].reshape(-1)
        graphs = batch["graphs"]
        reference = graphs.get(self.target_key)
        if reference is None:
            raise ValueError(f"graphs carries no {self.target_key!r} column to fit against")
        profile = graphs.get(self.profile_key)
        if profile is None:
            raise ValueError(
                f"graphs carries no {self.profile_key!r} column; mount molix.data.ProfileIndex "
                "as a batch task so a profile can be reduced over"
            )
        reference = reference.reshape(-1).to(energy.dtype)
        if energy.shape != reference.shape:
            raise ValueError(
                f"predicted energies {tuple(energy.shape)} do not match the reference "
                f"{tuple(reference.shape)}; both are one value per graph"
            )
        profile = profile.reshape(-1).long()
        n_graphs = int(energy.shape[0])
        mask = graphs.get("mask")
        live = torch.ones_like(energy) if mask is None else mask.reshape(-1).to(energy.dtype)

        # A dead slot must never win a minimum, so it is sent to +inf first.
        far = torch.full_like(energy, float("inf"))
        counts = energy.new_zeros(n_graphs).index_add(0, profile, live)
        safe = counts.clamp(min=1.0)
        reference_min = far.scatter_reduce(
            0, profile, torch.where(live > 0, reference, far), reduce="amin"
        )
        energy_min = far.scatter_reduce(
            0, profile, torch.where(live > 0, energy, far), reduce="amin"
        )
        reference_relative = reference - reference_min[profile]
        energy_relative = energy - energy_min[profile]

        weight = (
            torch.exp(
                (
                    (self.clamp - torch.minimum(reference_relative, energy_relative)) / self.decay
                ).clamp(max=0.0)
            ).detach()
            * live
        )
        weight_total = energy.new_zeros(n_graphs).index_add(0, profile, weight)
        offset = (
            energy.new_zeros(n_graphs).index_add(0, profile, (reference - energy) * weight)
            / weight_total.clamp(min=torch.finfo(energy.dtype).eps)
        ).detach()

        residual = (energy - reference + offset[profile]).pow(2) * weight
        per_profile = energy.new_zeros(n_graphs).index_add(0, profile, residual) / safe
        if self.soft_max is not None:
            # tanh(L/m)*m, written as a detached rescaling of L so the gradient is
            # attenuated by the saturation rather than clipped at it.
            #
            # `clamp(min=eps)` and not `where(L < eps, eps, L)`: a profile's loss is
            # a sum of squares times non-negative weights over a positive count, so
            # it is never negative and the two are the same function here — but the
            # `where` form takes a Python float as its second argument, which
            # becomes a **host** scalar tensor in the launch, and a CUDA kernel
            # handed one dies in `cuPointerGetAttribute` with
            # `CUDA_ERROR_INVALID_VALUE`. It cost stage 3 every cell of its first
            # calibration grid (job 2751984) and is invisible on CPU, where there is
            # no pointer attribute to query. At L = 0 this gives tanh(0)*m/eps = 0,
            # so a profile that already fits contributes nothing, as before.
            eps = torch.finfo(energy.dtype).eps
            saturating = (
                torch.tanh(per_profile / self.soft_max) * self.soft_max / per_profile.clamp(min=eps)
            ).detach()
            per_profile = per_profile * saturating
        # Only the profiles this batch actually holds are averaged over; a slot
        # no graph landed in contributes neither a numerator nor a denominator.
        occupied = (counts > 0).to(energy.dtype)
        return (per_profile * occupied).sum() / occupied.sum().clamp(min=1.0)
