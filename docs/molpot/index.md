# MolPot

MolPot provides potential functions and readout components:

- **Potentials**: LJ126, BondHarmonic, AngleHarmonic, DihedralHarmonic, and more
- **Heads**: AtomicEnergyMLP, EnergyHead, TypeHead
- **Derivation**: EnergyAggregation, ForceDerivation, StressDerivation
- **Pooling**: SumPooling, MeanPooling, MaxPooling, LayerPooling, EdgeToNodePooling
- **Composition**: PotentialComposer, parameter heads, MultiHead

Inputs use plain tensor arguments (`pos`, `edge_index`, `batch`, etc.).
