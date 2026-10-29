"""Eval-only GAFF atom-type readout on a message-passing encoder's atom features.

Hard isolation rules
--------------------
* Temporary :class:`~molrep.heads.TypeHead` sits **on top of** continuous
  embeddings — atom-type labels never enter the encoder as inputs.
* Production :class:`~molpot.forcefield.ClassicalIMM`
  must not import this module.

References:
    Wang et al., GAFF, J. Comput. Chem. 2004.
    Espaloma Chem. Sci. 2022 DOI 10.1039/D2SC02739A.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from molrep.encoders.message_passing import MessagePassingEncoder
from molrep.heads.type import TypeHead

__all__ = ["AtomTypeReadout"]


class AtomTypeReadout(nn.Module):
    """Compose :class:`MessagePassingEncoder` + :class:`TypeHead` for typing recovery.

    Args:
        encoder: The encoder whose atom features are probed.
        num_types: Discrete type vocabulary size.
        dropout: TypeHead dropout.
        dtype: Floating dtype the head is built at (the encoder's).
    """

    def __init__(
        self,
        encoder: MessagePassingEncoder,
        *,
        num_types: int,
        dropout: float = 0.0,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = TypeHead(
            hidden_dim=encoder.spec.atom_dim,
            num_types=num_types,
            dropout=dropout,
            dtype=dtype,
        )
        self.num_types = num_types

    @property
    def atom_dim(self) -> int:
        return int(self.head.hidden_dim)

    def encode(self, batch: TensorDict) -> torch.Tensor:
        """Run the encoder and return its atom features ``(N, D)``.

        Labels under ``atoms.atom_type`` / ``atom_type_id`` are **ignored**.
        """
        return self.encoder(batch).atom

    def forward(self, batch: TensorDict) -> TensorDict:
        """Write type logits and argmax ids onto ``atoms`` and return ``batch``.

        Args:
            batch: Nested TensorDict with atoms.Z and topology.

        Returns:
            ``batch``, with ``atoms.logits`` ``(N, num_types)`` and
            ``atoms.pred_type_id`` ``(N,)``.
        """
        feats = self.encode(batch)
        logits = self.head(feats)
        batch["atoms", "logits"] = logits
        batch["atoms", "pred_type_id"] = self.head.decode(logits)
        return batch
