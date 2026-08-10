"""Eval-only GAFF atom-type readout on continuous ChemEncoder embeddings.

Hard isolation rules
--------------------
* Temporary :class:`~molrep.heads.TypeHead` sits **on top of** continuous
  embeddings — atom-type labels never enter the encoder as inputs.
* Production :class:`~molpot.composition.parameterizer.ClassicalMMParameterizer`
  must not import this module.

References:
    Wang et al., GAFF, J. Comput. Chem. 2004.
    Espaloma Chem. Sci. 2022 DOI 10.1039/D2SC02739A.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from molrep.chem.encoder import ChemEncoder
from molrep.heads.type import TypeHead

__all__ = ["AtomTypeReadout"]


class AtomTypeReadout(nn.Module):
    """Compose :class:`ChemEncoder` + :class:`TypeHead` for typing recovery.

    Args:
        encoder: Continuous chemical perception encoder.
        num_types: Discrete type vocabulary size.
        dropout: TypeHead dropout.
    """

    def __init__(
        self,
        encoder: ChemEncoder,
        *,
        num_types: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = TypeHead(
            hidden_dim=int(encoder.atom_dim),
            num_types=num_types,
            dropout=dropout,
        )
        self.num_types = num_types

    @property
    def atom_dim(self) -> int:
        return int(self.head.hidden_dim)

    def encode(self, batch: TensorDict) -> torch.Tensor:
        """Run encoder and return atom chem features ``(N, D)``.

        Labels under ``atoms.atom_type`` / ``atom_type_id`` are **ignored**.
        """
        out = self.encoder(batch)
        feats = out["atoms", "chem_features"]
        return feats

    def forward(self, batch: TensorDict) -> dict[str, Any]:
        """Return logits and argmax type ids from continuous embeddings.

        Args:
            batch: Nested TensorDict with atoms.Z and topology.

        Returns:
            Dict with ``logits`` ``(N, num_types)`` and ``pred_type_id`` ``(N,)``.
        """
        feats = self.encode(batch)
        logits = self.head(feats)
        return {"logits": logits, "pred_type_id": self.head.decode(logits), "features": feats}
