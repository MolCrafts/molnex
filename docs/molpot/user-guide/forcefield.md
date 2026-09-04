# ForceField

`ForceField` is the classical parameter container. It owns styles and
param tables. Concrete forms inherit it — there is no
`ForceField.class_i()` façade.

```python
from molpot.forcefield import OPLSAAForceField, Parameterizer

template = OPLSAAForceField()
model = Parameterizer(
    encoder,
    template,
    bond_head=...,
    angle_head=...,
    proper_head=...,      # form="opls" → k1..k4
    improper_head=...,
    atom_head=...,
    charge_model=...,
)
ff = model(batch)                # filled OPLSAAForceField copy; template untouched
pots = ff.to_potentials()        # overrides molpy native → torch
E = pots.calc_energy(batch)
out = pots(batch)                # writes graphs.energy, atoms.forces
ir = ff.to_ir()                  # Class-I PotentialIR bags
# ff is a molpy.ForceField — pass it to molpy.io after realize()
```

## Styles

Register extra styles, then fill tables with `with_params` (copy-on-write):

```python
ff = (
    OPLSAAForceField()
    .def_pairstyle("lj/cut", cutoff=12.0)
    .with_params(
        bonds={"k": k, "r0": r0},
        angles={"k": ka, "theta0": theta0},
        propers={"k1": k1, "k2": k2, "k3": k3, "k4": k4},
        atoms={"q": q, "epsilon": eps, "sigma": sig},
    )
)
```

Pair cutoffs and electrostatic calculator options belong on the
potential (`def_pairstyle(..., cutoff=…, calculator="pme")`), not on
`to_potentials()`.

## `to_ir`

`to_ir()` materializes Class-I bags from the same continuous /
per-interaction tables that `Potentials` evaluates. OPLS `k1..k4` become
a multi-term cosine bag with `k = K/2` and `γ ∈ {0, π}`. Charges use
the `q` key only.

`ff.realize(system)` fills native molpy type tables so `molpy.io`
can write this object directly. See
[Force-field export](../../molix/user-guide/ff-export.md).

To name those types by SMARTS: `molpy.io.write_smarts` on a molpy
`Atomistic`, merge parameters inside each key (`Condenser`), then
`style.def_type`. See the
[Class-I tutorial](../tutorials/class-i-forcefield.md).
