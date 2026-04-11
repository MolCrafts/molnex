# MolRep

MolRep provides molecular representation learning components:

- **Embedding**: Node, radial, angular, and cutoff functions
- **Interaction**: Equivariant linear, tensor product, symmetric contraction, aggregation
- **Readout**: Pooling, ProductHead, ScalarHead

All components use plain PyTorch inputs and outputs.

Common inputs: `Z`, `pos`, `edge_index`, `bond_diff`, `bond_dist`.
