# Class-I ForceField

Learn continuous Class-I parameters, evaluate them as a torch module, and
write a LAMMPS deck through molpy.

## 1. Template and fill

```python
import torch
from tensordict import TensorDict
from molpot.forcefield import OPLSAAForceField

ff = (
    OPLSAAForceField()
    .def_pairstyle("lj/cut", cutoff=12.0)
    .with_params(
        # A table for *every* style OPLSAAForceField declares — bond, angle,
        # dihedral, improper, lj/cut and coulomb. Values are OPLS-like
        # magnitudes for two united-atom CH3 sites joined by a C–C bond.
        bonds={
            "k": torch.tensor([200.0]),           # kcal mol⁻¹ Å⁻²
            "r0": torch.tensor([1.53]),           # Å
        },
        angles={
            # this kernel is E = ½k(θ − θ0)², so k is twice the OPLS/LAMMPS
            # K ≈ 58.35 of a C–C–C bend
            "k": torch.tensor([116.7]),           # kcal mol⁻¹ rad⁻²
            "theta0": torch.tensor([1.967]),      # 112.7° in rad
        },
        propers={
            "k1": torch.tensor([1.0]),            # kcal/mol
            "k2": torch.tensor([2.0]),
            "k3": torch.tensor([0.0]),
            "k4": torch.tensor([0.0]),
        },
        impropers={
            # E = (k / idivf)·[1 + cos(2φ − π)], the OPLS sp² improper
            "k": torch.tensor([[1.1]]),           # [n_types, n_terms], kcal/mol
            "periodicity": torch.tensor([2]),     # [n_terms]
            "phase": torch.tensor([[torch.pi]]),  # [n_types, n_terms], rad
            "idivf": torch.tensor([1.0]),         # [n_types]
        },
        atoms={
            # one row per atom, in the order of atoms.pos below
            "q": torch.tensor([0.0, 0.0]),            # e — neutral, identical sites
            "epsilon": torch.tensor([0.207, 0.207]),  # kcal/mol
            "sigma": torch.tensor([3.775, 3.775]),    # Å
        },
    )
)

batch = TensorDict(
    {
        "atoms": TensorDict(
            {
                "pos": torch.tensor([[0.0, 0.0, 0.0], [1.53, 0.0, 0.0]]),
                # graph membership: both atoms belong to molecule 0
                "batch": torch.zeros(2, dtype=torch.long),
            },
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

`OPLSAAForceField` **is** a `molpy.ForceField` (it subclasses
`molpot.forcefield.ForceField`, which subclasses molpy's). Training heads
return the same type via `with_params`.

Every declared style needs its parameter table: an unparameterised
declaration used to contribute a silent zero, and since a0-01 the first
energy call raises instead (S1 — a style is either parameterised or absent).
An empty *topology* is a different fact and stays a zero, which is why the
angle, dihedral and improper tables above are filled even though this
two-atom fragment has no angle, torsion or improper to index them with. The
two nonbonded terms do evaluate here, and both come out at 0: the single pair
is 1-2 bonded, which OPLS-AA excludes outright (scale 0).

## 2. Evaluate as a torch module

```python
from molix.batch import NeighborList

pots = ff.to_potentials()   # overrides molpy native kernels
batch = NeighborList(cutoff=12.0).execute(batch)  # molix builds the pair list
E = pots.calc_energy(batch)
out = pots(batch)           # writes graphs.energy and atoms.forces
```

`Potentials` consumes the `edges` list molix builds and never enumerates
pairs itself, so the neighbour-list line is required once a pair style is
declared — give it the largest pair cutoff (here the `lj/cut` 12 Å) and keep
the `symmetry=True` default.

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
import torch
from molix.md import MD, VelocityVerletIntegrator

md = (
    MD(torch.float64)
    .set_forcefield(ff)          # the ForceField itself, not a wrapper
    .set_neighbors(nl)
    .set_integrator(VelocityVerletIntegrator(0.5))
)
```

`MD` is a composer: each component is constructed on its own and attached, and
every setter re-runs `ff.to_potentials()`. Masses come from the system
(`atoms.mass`), bound at `run`, and the conversion into integrator units
(amu, Å, fs) is read off `ff.units` by `molix.units.preset_energy_to_md` — a
Class-I ForceField at the default `units="real"` reports kcal/mol, and nothing
in the example spells out a conversion factor.

## 3. Write LAMMPS files

```python
from molix.ff_export import LammpsWriter

writer = LammpsWriter()
writer.write_data("system.data", batch)
writer.write_settings("system.ff", ff, batch)
```

The writer calls `ff.clone().realize(batch)` then `molpy.io`. There is no
`to_molpy()`.
