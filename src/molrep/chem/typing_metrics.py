"""Typing recovery metrics for GAFF atom-type evaluation probes.

Eval-only reporting: overall accuracy, per-element accuracy, rare-type
accuracy, confusion matrix, and molecule-level error counts.

This module is **not** part of production :class:`ClassicalMMParameterizer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["TypingRecoveryMetrics", "TypingRecoveryReport"]


@dataclass(frozen=True)
class TypingRecoveryReport:
    """Immutable typing recovery report.

    Attributes:
        overall_accuracy: Fraction of correctly predicted atom types in ``[0, 1]``.
        per_element_accuracy: Map atomic number → accuracy.
        rare_type_accuracy: Accuracy on types with support ≤ ``rare_max_count``.
        confusion: Confusion matrix ``(n_types, n_types)`` as int64 tensor.
        molecule_error_counts: Map molecule_id → number of mis-typed atoms.
        n_atoms: Number of labeled atoms scored.
        n_molecules: Number of molecules with at least one labeled atom.
    """

    overall_accuracy: float
    per_element_accuracy: dict[int, float]
    rare_type_accuracy: float | None
    confusion: torch.Tensor
    molecule_error_counts: dict[str, int]
    n_atoms: int
    n_molecules: int


@dataclass
class TypingRecoveryMetrics:
    """Accumulate type predictions for a :class:`TypingRecoveryReport`.

    Args:
        num_types: Size of the discrete type vocabulary (confusion matrix).
        rare_max_count: Types with ≤ this many labels count as rare.
    """

    num_types: int
    rare_max_count: int = 5
    _pred: list[torch.Tensor] = field(default_factory=list, repr=False)
    _true: list[torch.Tensor] = field(default_factory=list, repr=False)
    _Z: list[torch.Tensor] = field(default_factory=list, repr=False)
    _mol: list[str] = field(default_factory=list, repr=False)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        Z: torch.Tensor | None = None,
        molecule_ids: list[str] | None = None,
    ) -> None:
        """Accumulate one batch of integer type ids.

        Args:
            pred: Predicted type indices ``(N,)``.
            target: Reference type indices ``(N,)``.
            Z: Optional atomic numbers ``(N,)``.
            molecule_ids: Optional per-atom molecule id strings length ``N``.
        """
        pred = pred.detach().long().reshape(-1).cpu()
        target = target.detach().long().reshape(-1).cpu()
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch {pred.shape} vs {target.shape}")
        self._pred.append(pred)
        self._true.append(target)
        if Z is not None:
            self._Z.append(Z.detach().long().reshape(-1).cpu())
        if molecule_ids is not None:
            if len(molecule_ids) != pred.numel():
                raise ValueError("molecule_ids length must match N atoms")
            self._mol.extend(list(molecule_ids))

    def reset(self) -> None:
        self._pred.clear()
        self._true.clear()
        self._Z.clear()
        self._mol.clear()

    def compute(self) -> TypingRecoveryReport:
        if not self._pred:
            empty = torch.zeros(self.num_types, self.num_types, dtype=torch.int64)
            return TypingRecoveryReport(
                overall_accuracy=0.0,
                per_element_accuracy={},
                rare_type_accuracy=None,
                confusion=empty,
                molecule_error_counts={},
                n_atoms=0,
                n_molecules=0,
            )
        pred = torch.cat(self._pred)
        true = torch.cat(self._true)
        n = int(pred.numel())
        correct = pred == true
        overall = float(correct.float().mean().item()) if n else 0.0

        confusion = torch.zeros(self.num_types, self.num_types, dtype=torch.int64)
        for t, p in zip(true.tolist(), pred.tolist(), strict=True):
            if 0 <= t < self.num_types and 0 <= p < self.num_types:
                confusion[t, p] += 1

        per_el: dict[int, float] = {}
        if self._Z:
            Z = torch.cat(self._Z)
            for z in sorted(set(Z.tolist())):
                mask = Z == z
                if mask.any():
                    per_el[int(z)] = float(correct[mask].float().mean().item())

        # Rare types by true-label support
        rare_acc: float | None = None
        type_counts = torch.bincount(true, minlength=self.num_types)
        rare_mask = torch.zeros(n, dtype=torch.bool)
        for t in range(self.num_types):
            if 0 < int(type_counts[t]) <= self.rare_max_count:
                rare_mask |= true == t
        if rare_mask.any():
            rare_acc = float(correct[rare_mask].float().mean().item())

        mol_err: dict[str, int] = {}
        if self._mol:
            for mid, ok in zip(self._mol, correct.tolist(), strict=True):
                if not ok:
                    mol_err[mid] = mol_err.get(mid, 0) + 1
            n_mol = len(set(self._mol))
        else:
            n_mol = 0

        return TypingRecoveryReport(
            overall_accuracy=overall,
            per_element_accuracy=per_el,
            rare_type_accuracy=rare_acc,
            confusion=confusion,
            molecule_error_counts=mol_err,
            n_atoms=n,
            n_molecules=n_mol,
        )
