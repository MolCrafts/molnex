"""ChemPerception — thin molzoo recipe over :class:`~molrep.chem.encoder.ChemEncoder`.

Encoder-only: writes continuous chem features into the batch TensorDict.
No energy head, no molpot import.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molrep.chem.encoder import ChemEncoder
from molrep.chem.features import ChemEmbeddings

from .spec import ChemPerceptionSpec


class ChemPerception(ChemEncoder):
    """Continuous chemical-perception recipe (molzoo).

    Thin config-driven wrapper around :class:`~molrep.chem.encoder.ChemEncoder`.
    Writes ``*.chem_features``; does not predict energy or forces.

    Args:
        spec: Optional :class:`ChemPerceptionSpec`. When omitted, keyword
            arguments populate a new spec (same fields as the parent
            encoder).
        dtype: Floating dtype every parameter and buffer is built at.
        **kwargs: Forwarded to :class:`ChemPerceptionSpec` when ``spec`` is
            not provided.

    Example::

        from molzoo.chem import ChemPerception, ChemPerceptionSpec

        model = ChemPerception(atom_dim=16, bond_dim=16, dtype=torch.float32)
        batch = model(batch)  # writes atoms/bonds/… chem_features
    """

    def __init__(
        self,
        spec: ChemPerceptionSpec | None = None,
        *,
        dtype: torch.dtype,
        **kwargs,
    ) -> None:
        if spec is None:
            spec = ChemPerceptionSpec(**kwargs)
        elif kwargs:
            raise TypeError(
                "ChemPerception accepts either a ChemPerceptionSpec or keyword overrides, not both"
            )
        super().__init__(
            atom_dim=spec.atom_dim,
            bond_dim=spec.bond_dim,
            angle_dim=spec.angle_dim,
            proper_dim=spec.proper_dim,
            improper_dim=spec.improper_dim,
            num_elements=spec.num_elements,
            hidden_dim=spec.hidden_dim,
            atom_channels=spec.atom_channels,
            bond_channels=spec.bond_channels,
            n_blocks=spec.n_blocks,
            local_attention=spec.local_attention,
            dtype=dtype,
        )
        self.config = spec

    def forward(self, batch: TensorDict) -> TensorDict:
        """Run chemical perception and write features onto ``batch``.

        Args:
            batch: Nested TensorDict with ``atoms.Z`` and valence topology.

        Returns:
            The same batch with ``*.chem_features`` populated.
        """
        return super().forward(batch)

    def embeddings(self, batch: TensorDict) -> ChemEmbeddings:
        """View written chem features as :class:`~molrep.chem.features.ChemEmbeddings`.

        Args:
            batch: Batch previously processed by :meth:`forward`.

        Returns:
            :class:`ChemEmbeddings` viewing the feature fields.
        """
        return super().embeddings(batch)
