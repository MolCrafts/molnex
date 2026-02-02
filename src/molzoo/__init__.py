"""MolZoo: Molecular Neural Network Architectures.

End-to-end models combining molrep representations and molpot interactions
for molecular property prediction (energy, forces, etc.).

Models:
    MACE: Equivariant message-passing feature extractor
        - Learnable per-atom embeddings from atomic numbers
        - Distance-dependent RBF and angle-dependent SH features
        - Equivariant convolution blocks with tensor products

        Usage:
            >>> mace = MACE(num_species=100, hidden_dim=128, l_max=2)
            >>> features = mace(atom_td)  # [num_atoms, hidden_dim]

    ScaleShiftMACE: Complete model with energy/force prediction
        - MACE encoder + scatter energy pooling + autograd forces
        - Outputs molecular energy and per-atom forces
        - Supports mean or sum pooling for energy aggregation

        Usage:
            >>> model = ScaleShiftMACE(num_species=100, hidden_dim=128, out_dim=1)
            >>> output = model(**batch.to_model_kwargs())
            >>> energy = output['energy']
            >>> forces = output['forces']

Configuration:
    All models use Pydantic specs for reproducible configuration:

    >>> from molzoo import ScaleShiftMACESpec
    >>> spec = ScaleShiftMACESpec(num_species=100, hidden_dim=128, ...)
    >>> model = spec.build()
    >>> config_dict = spec.model_dump()  # Export config

Training Integration:
    Compatible with molix Trainer and custom training loops:

    >>> from molix import Trainer
    >>> trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=opt)
    >>> state = trainer.train_step(batch)

See README.md in this directory for detailed architecture documentation.
"""

from molzoo.mace import (
    MACE,
    MACESpec,
    ScaleShiftMACE,
    ScaleShiftMACESpec,
)

__all__ = [
    # Feature extractors
    "MACE",
    "MACESpec",
    # Complete models
    "ScaleShiftMACE",
    "ScaleShiftMACESpec",
]
