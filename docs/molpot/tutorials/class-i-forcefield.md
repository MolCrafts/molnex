# Class-I ForceField

Learn continuous Class-I parameters, evaluate them as a torch module, and
write a LAMMPS deck through molpy.

## 1. Template and fill

```python
import torch
from tensordict import TensorDict
from molpot.forcefield import OPLSForceField

ff = (
    OPLSForceField()
    .def_pairstyle("lj/cut", cutoff=12.0)
    .with_params(
        bonds={
            "k": torch.tensor([200.0]),
            "r0": torch.tensor([1.53]),
        },
        propers={
            "k1": torch.tensor([1.0]),
            "k2": torch.tensor([2.0]),
            "k3": torch.tensor([0.0]),
            "k4": torch.tensor([0.0]),
        },
    )
)

batch = TensorDict(
    {
        "atoms": TensorDict(
            {"pos": torch.tensor([[0.0, 0.0, 0.0], [1.53, 0.0, 0.0]])},
            batch_size=[2],
        ),
        "bonds": TensorDict(
            {
                "atomi": torch.tensor([0], dtype=torch.long),
                "atomj": torch.tensor([1], dtype=torch.long),
            },
            batch_size=[1],
        ),
    },
    batch_size=[],
)
```

`OPLSForceField` **is** a `molpy.ForceField` (`ClassIForceField` →
`ForceField`). Training heads (`ClassicalIMM`) return the same type via
`with_params`.

## 2. Evaluate as a torch module

```python
pots = ff.to_potentials()   # overrides molpy native kernels
E = pots.calc_energy(batch)
out = pots(batch)           # writes graphs.energy and atoms.forces
```

## 2b. Condense by SMARTS, then `def_type`

Environment keys are local SMARTS from `molpy.io.write_smarts`. Merge
parameters inside each key, then register molpy types yourself:

```python
from molpy.io import write_smarts
from molrep.condensation import Condenser, InteractionClass

envs = [
    [
        tuple(sorted((write_smarts(mol, b.itom), write_smarts(mol, b.jtom))))
        for b in mol.bonds
    ]
]
result = Condenser().merge(
    [ff.params("bonds")],
    interaction=InteractionClass.BOND,
    environments=envs,
)
astyle = ff.get_style("atom", "full")
bstyle = ff.get_style("bond", "harmonic")
for rec in result.type_system.records():
    si, sj = envs[0][result.assignment.row_indices[rec.support_ids[0]]]
    ai = astyle.get_type_by_name(si) or astyle.def_type(si)
    aj = astyle.get_type_by_name(sj) or astyle.def_type(sj)
    bstyle.def_type(ai, aj, name=rec.label, **rec.prototype)
```

`mol` is a molpy `Atomistic`. If one SMARTS bucket's parameters exceed
the merge budget, the condenser raises — the environment is too coarse.


Bind to in-process MD:

```python
from molix.md import MD, PotentialForceField
from molix.units import units

kcal_to_md = (1.0 * units.kilocalorie_per_mole).to(
    "amu * angstrom ** 2 / femtosecond ** 2"
).magnitude
md = MD(
    PotentialForceField(pots, batch, energy_scale=kcal_to_md),
    mass=torch.tensor([12.011, 12.011]),
    dt=0.5,
)
```

## 3. Write LAMMPS files

```python
from molix.ff_export import LammpsWriter

writer = LammpsWriter()
writer.write_data("system.data", batch)
writer.write_settings("system.ff", ff, batch)
```

The writer calls `ff.clone().realize(batch)` then `molpy.io`. There is no
`to_molpy()`.
