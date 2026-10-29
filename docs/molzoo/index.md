# MolZoo

MolZoo is the model-family layer of MolNex. Each family is a torch-free spec,
a feature encoder assembled from generic `molrep` blocks and the family's own
`layers.py`, and named products that add the energy (or property) heads and,
where one exists, load the official checkpoint. Training stays in `molix`; reusable physics heads,
derivation and composition stay in `molpot` and `molix.autograd`.

## Families

| Family | Package | Encoder | Products | Spec |
|--------|---------|---------|----------|------|
| Allegro | `molzoo.allegro` | `AllegroEncoder(AllegroSpec(...), dtype=...)` → `pairs.edge_features` | `AllegroPotential` (energy, forces, virial); `AllegroPotential.load("*.nequip.zip")` reads official nequip packages | [`specs/allegro.md`](specs/allegro.md) |
| MACE | `molzoo.mace` | `MACEEncoder(MACESpec(...), dtype=...)` → `atoms.node_features` (primitives exposed) | `MaceMatpes`, `MaceOmol`, `MacePolar`, `MaceOff`, `MaceLes`; `.load(path, config=...)` reads the official cuEq-converted archives | [`specs/mace_matpes.md`](specs/mace_matpes.md), [`specs/mace_omol.md`](specs/mace_omol.md), [`specs/mace_off.md`](specs/mace_off.md) |
| PiNet2 | `molzoo.pinet` | `PiNetEncoder(PiNetSpec(...), dtype=...)` → `atoms.node_features` | `PiNetPotential` (energy, forces), `PiNetDipole`, `PiNetPolarizability` | [`specs/pinet2.md`](specs/pinet2.md) |
| Rondo | `molzoo.rondo` | `ChemEncoder` channels | `Rondo(forcefield, *, dtype=)`. The force field is one `Potentials` (`Potentials.from_forcefield`); `forward` writes `atoms.forces_pred` | [`specs/rondo.md`](specs/rondo.md) |

Every product is a `molix.autograd.Potential` with two public methods and no
layer between them: `energy(frame)` reads its columns straight off the
post-collate batch and writes `atoms.energy` / `graphs.energy` (the `graphs`
namespace is required), and `forward` is `Derivative(self.energy, wrt)` fixed
at construction (`wrt=("atoms.pos",)` → `atoms.forces`; `"graphs.strain"` →
`graphs.virial`). A product is constructed from its typed spec object and
the floating dtype it is built at —
`MaceMatpes(MACEMatpesSpec(...), wrt=..., dtype=...)`,
`AllegroPotential(AllegroSpec(...), wrt=..., dtype=...)`,
`PiNetPotential(PiNetSpec(...), wrt=..., dtype=...)` — and imported from its subpackage
(`from molzoo.mace import MaceMatpes`; `molzoo` itself exports nothing).
`Sonata` (encoder + `molpot` heads) lives in `molpot.composition`, not here.

## Documentation Layout

- [Tutorial](tutorials/index.md): how a MolZoo model is connected to data,
  neighbor lists, losses and training.
- [Allegro User Guide](user-guide/allegro.md): theory, the forward pass
  equation by equation, `AllegroEncoder` / `AllegroPotential`, and loading
  official `.nequip.zip` packages.
- [MACE precision and checkpoints](user-guide/mace.md): MatPES/Polar precision
  controls, official weight loading and current large-system validation limits.
- Specs under `Spec` in the MolZoo navigation (Allegro, MACE-OMOL, PiNet2,
  Rondo).

There is no separate "Explanation" section for MolZoo. Model theory belongs in
the model's user guide, next to the code path and the source spec.

## Package Boundary

MolZoo owns specs, encoders and the named products built on them. It does not
own:

- dataset acquisition and caching (MolHub)
- neighbor-list generation (`molix.neighbor.NeighborList`)
- the derivative op (`molix.autograd.Derivative`)
- the training loop and losses (`molix`)
- dataset statistics

For Allegro, this means:

```text
molix.data.DataModule (+ NeighborList batch task)
  -> nested TensorDict batch
  -> molzoo.allegro.AllegroPotential
       (AllegroEncoder body -> edge readout -> per-type scale/shift -> Derivative)
  -> graphs.energy / atoms.forces
  -> losses
  -> molix.Trainer
```

## References

- [Allegro paper](https://arxiv.org/abs/2204.05249)
- [Official Allegro documentation](https://nequip.readthedocs.io/projects/allegro/en/latest/)
- MolNex Allegro spec, included as `MolZoo -> Spec -> Allegro Spec`
