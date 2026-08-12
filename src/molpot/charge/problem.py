"""Charge-equilibration problem description (data only)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["ChargeEquilibrationProblem"]


@dataclass
class ChargeEquilibrationProblem:
    """Batch of molecules for a charge-equilibration solve.

    Attributes:
        atom_batch: Graph index per atom ``(N,)``.
        num_graphs: Number of molecules ``B``.
        total_charge: Target total charge per molecule ``(B,)``.
        pos: Optional positions ``(N, 3)`` when the Coulomb kernel needs geometry.
        Z: Optional atomic numbers ``(N,)``.
    """

    atom_batch: torch.Tensor
    num_graphs: int
    total_charge: torch.Tensor
    pos: torch.Tensor | None = None
    Z: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.atom_batch.ndim != 1:
            raise ValueError(f"atom_batch must be (N,), got {tuple(self.atom_batch.shape)}")
        if self.total_charge.shape != (self.num_graphs,):
            raise ValueError(
                f"total_charge must be (num_graphs,)={(self.num_graphs,)}, "
                f"got {tuple(self.total_charge.shape)}"
            )

    @classmethod
    def from_batch(
        cls,
        batch: object,
        *,
        total_charge: float | torch.Tensor = 0.0,
    ) -> ChargeEquilibrationProblem:
        """Build from a TensorDict / Mapping with ``atoms.batch`` and ``atoms.pos``."""
        from collections.abc import Mapping

        from tensordict import TensorDict

        if not isinstance(batch, (Mapping, TensorDict)) or "atoms" not in batch:
            raise ValueError("batch must provide atoms namespace")
        atoms = batch["atoms"]
        if "batch" not in atoms:
            # single molecule
            n = int(atoms["pos"].shape[0]) if "pos" in atoms else int(atoms["Z"].shape[0])
            atom_batch = torch.zeros(n, dtype=torch.long, device=atoms["pos"].device)
        else:
            atom_batch = atoms["batch"].long()
        num_graphs = int(atom_batch.max().item()) + 1 if atom_batch.numel() else 0
        device = atom_batch.device
        dtype = atoms["pos"].dtype if "pos" in atoms else torch.float32
        if isinstance(total_charge, (int, float)):
            qtot = torch.full((num_graphs,), float(total_charge), device=device, dtype=dtype)
        else:
            qtot = total_charge.to(device=device, dtype=dtype)
            if qtot.shape != (num_graphs,):
                raise ValueError("total_charge tensor must match num_graphs")
        pos = atoms["pos"] if "pos" in atoms else None
        Z = atoms["Z"] if "Z" in atoms else None
        return cls(
            atom_batch=atom_batch,
            num_graphs=num_graphs,
            total_charge=qtot,
            pos=pos,
            Z=Z,
        )
