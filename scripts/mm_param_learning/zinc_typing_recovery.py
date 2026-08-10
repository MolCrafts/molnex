"""Experiment script scaffold for Validation A (zinc-typing recovery).

Loads MolHub ``dataset:espaloma/zinc-typing@1`` when available; otherwise
operates on a caller-provided batch of continuous features + labels.
Does not wire into ClassicalMMParameterizer.
"""

from __future__ import annotations

from typing import Any

import torch

from molrep.chem import AtomTypeReadout, ChemEncoder, TypingRecoveryMetrics


def run_typing_recovery(
    *,
    encoder: ChemEncoder | None = None,
    num_types: int = 2,
    batches: list[tuple[Any, torch.Tensor]] | None = None,
    steps: int = 50,
    lr: float = 0.05,
) -> dict[str, Any]:
    """Train a temporary TypeHead and return a metrics report dict.

    Args:
        encoder: Optional ChemEncoder; default small encoder.
        num_types: Discrete vocabulary size.
        batches: List of ``(batch_tensordict, type_id_tensor)``.
        steps: Optimization steps (eval-only throwaway head).
        lr: Adam learning rate.

    Returns:
        Dict with overall_accuracy and n_atoms from TypingRecoveryMetrics.
    """
    if batches is None:
        raise ValueError("batches required (MolHub path injects offline frames → batch)")
    enc = encoder or ChemEncoder(atom_dim=16, bond_dim=8)
    probe = AtomTypeReadout(enc, num_types=num_types)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        for batch, y in batches:
            opt.zero_grad()
            logits = probe(batch)["logits"]
            loss = torch.nn.functional.cross_entropy(logits, y.long())
            loss.backward()
            opt.step()
    metrics = TypingRecoveryMetrics(num_types=num_types)
    for batch, y in batches:
        pred = probe(batch)["pred_type_id"]
        Z = batch["atoms", "Z"]
        metrics.update(pred, y, Z=Z)
    report = metrics.compute()
    return {
        "overall_accuracy": report.overall_accuracy,
        "n_atoms": report.n_atoms,
        "per_element_accuracy": report.per_element_accuracy,
    }
