# MolZoo

MolZoo is the model-family layer of MolNex. It contains reference encoder
assemblies built from lower-level `molrep` modules, while training and
downstream physical outputs stay in `molix` and `molpot`.

The current written documentation is Allegro-focused. MACE remains in the API,
but it is intentionally not expanded in this pass.

## Documentation Layout

- [Tutorial](tutorials/index.md): how MolZoo models are organized and how an
  encoder is connected to data, readout, losses, and training.
- [Allegro User Guide](user-guide/allegro.md): theory, formulas, implementation
  contract, hands-on tutorial, and spec crosswalk for `molzoo.Allegro`.
- Allegro Spec: exact tensor contracts, paper-to-code mapping, adaptation
  ledger, and validation contract. It is included under `Spec` in the MolZoo
  navigation.

There is no separate "Explanation" section for MolZoo. Model theory belongs in
the model's user guide, next to the code path and the source spec.

## Package Boundary

MolZoo owns assembled encoders. It does not own:

- neighbor-list generation
- the training loop
- energy readout and aggregation
- force derivation
- dataset statistics

For Allegro, this means:

```text
molix.data.NeighborList
  -> GraphBatch
  -> molzoo.Allegro
  -> edge_features
  -> molpot.heads.EdgeEnergyHead
  -> energy / force losses
  -> molix.Trainer
```

## References

- [Allegro paper](https://www.nature.com/articles/s41467-023-36329-y)
- [Official Allegro documentation](https://nequip.readthedocs.io/projects/allegro/en/latest/)
- MolNex Allegro spec, included as `MolZoo -> Spec -> Allegro Spec`
