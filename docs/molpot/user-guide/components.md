# Components

MolPot provides potential terms, parameter heads, pooling layers, derivation
operators, and composition utilities.

For the learnable Class-I path (`OPLSAAForceField` **is** a
`molpy.ForceField`, `to_potentials()` is the torch override), use
[ForceField](forcefield.md) and the
[Class-I tutorial](../tutorials/class-i-forcefield.md). The
`PotentialComposer` section below is the older head→term assembly, not
the Class-I container.

## Potential Terms

Potential terms are PyTorch modules that compute energy from distances and
per-pair parameters.

```python
from molpot.potentials import AngleHarmonic, BondHarmonic, LJ126

lj = LJ126()
bond = BondHarmonic()
angle = AngleHarmonic()
```

`LJ126` expects `distance`, `epsilon_ij`, and `sigma_ij` — all *per-pair*. A
force field stores one `epsilon` (well depth, kcal/mol) and one `sigma`
(zero-crossing distance, Å) per *atom*, so something has to make one of each per
pair first. That something is a **combining rule**, and `LJ126` carries it as
`mixing_fn`; `PotentialComposer` applies it before calling the term.

Which rule a model uses is part of its identity, not a numerical detail: the
same parameter table combined two different ways is two different force fields
on every cross pair. molnex spells the choice as `MixingRule`:

- `MixingRule.LORENTZ_BERTHELOT` — `sigma` arithmetic, `epsilon` geometric. The
  AMBER / CHARMM rule and molnex's default (`lorentz_berthelot`).
- `MixingRule.GEOMETRIC` — both geometric. The OPLS-AA rule
  (`geometric_mixing`), which `OPLSAAForceField` declares.

```python
from molpot.potentials import MixingRule, geometric_mixing

lj_opls = LJ126(mixing_fn=geometric_mixing)      # or MixingRule.GEOMETRIC.mix
```

On the Class-I path you do not pass a function at all — you declare the rule on
the style, `def_pairstyle("lj/cut", mixing="geometric")`, and it is recorded in
the model manifest and carried into the LAMMPS and OpenMM exports. See the
[ff-export guide](../../molix/user-guide/ff-export.md).

## Parameter Heads

Heads predict per-atom parameters from node features.

```python
from molpot.composition import LJParameterHead

head = LJParameterHead(feature_dim=128, hidden_dim=64)
params = head(node_features)
```

## Pooling

Pooling layers reshape encoder output before potential composition.

```python
from molpot.pooling import LayerPooling, SumPooling

layer_pool = LayerPooling("mean")
sum_pool = SumPooling()
```

## Composition

```python
from molpot.composition import PotentialComposer
from molpot.potentials import LJ126

composer = PotentialComposer(
    head=LJParameterHead(feature_dim=64),
    potentials={"lj126": LJ126()},
)
```

The composer returns:

- `energy`
- `term_energies`
- `parameters`
- `forces`, when `compute_forces=True`
