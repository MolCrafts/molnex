# MolZoo

MolZoo is the model-family layer of MolNex. It contains reference encoder
assemblies built from lower-level `molrep` modules. Training stays in `molix`;
physics heads, derivation, and composition stay in `molpot`.

## Families

| Model | Kind | Spec | Notes |
|-------|------|------|--------|
| Allegro | encoder | [`specs/allegro.md`](specs/allegro.md) | Full user guide below |
| MACE | encoder | `src/molzoo/specs/mace.md` | Writes `atoms.node_features` |
| PiNet | encoder + temporary potential | `src/molzoo/specs/pinet2.md` (source tree) | Package `molzoo.pinet/` (`encoder`, `potential`, `properties`); long-term potential home is molpot |
| MACE-OMOL | full energy/force | [`specs/mace_omol.md`](specs/mace_omol.md) | Lazy import; not encoder-only |
| Sonata | composition | lives in **`molpot.composition`**, not molzoo | `build_sonata` |

## Documentation Layout

- [Tutorial](tutorials/index.md): how MolZoo models are organized and how an
  encoder is connected to data, readout, losses, and training.
- [Allegro User Guide](user-guide/allegro.md): theory, formulas, implementation
  contract, hands-on tutorial, and spec crosswalk for `molzoo.Allegro`.
- Specs under `Spec` in the MolZoo navigation (Allegro, MACE-OMOL today).

There is no separate "Explanation" section for MolZoo. Model theory belongs in
the model's user guide, next to the code path and the source spec.

## Package Boundary

MolZoo owns assembled **encoder recipes** (and, temporarily, a few full
energy/force façades such as `PiNetPotential` / `MACEOMol`). It does not own:

- neighbor-list generation
- the training loop
- long-term energy aggregation / force derivation (those live in `molpot`)
- dataset statistics

For Allegro, this means:

```text
molix.data.NeighborList
  -> nested TensorDict batch
  -> molzoo.Allegro
  -> edge / node features
  -> molpot.heads.EdgeEnergyHead
  -> energy / force losses
  -> molix.Trainer
```

## References

- [Allegro paper](https://www.nature.com/articles/s41467-023-36329-y)
- [Official Allegro documentation](https://nequip.readthedocs.io/projects/allegro/en/latest/)
- MolNex Allegro spec, included as `MolZoo -> Spec -> Allegro Spec`
