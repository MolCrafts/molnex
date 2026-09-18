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
    # Re-declaring a style *replaces* its options, so the OPLS-AA combining
    # rule has to be restated alongside the cutoff — dropping it would silently
    # fall back to Lorentz–Berthelot, a different force field on every pair of
    # unequal sizes.
    .def_pairstyle("lj/cut", cutoff=12.0, mixing="geometric")
    # The template declares pair:coulomb bare, which under open boundaries is
    # the untruncated N(N-1)/2 sum. This example is an isolated fragment, so
    # state the truncation it is actually run at (§2 builds its list from it).
    .def_pairstyle("coulomb", cutoff=12.0)
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
            "charge": torch.tensor([0.0, 0.0]),            # e — neutral, identical sites
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
        # One graph. The namespace is sized [B], and that size is where the
        # per-graph columns (energies, and the total charge written in §2) get
        # their length from — a collated batch always has it.
        "graphs": TensorDict({"num_atoms": torch.tensor([2])}, batch_size=[1]),
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
from molix.data import TotalCharge

pots = ff.to_potentials()   # overrides molpy native kernels
for task in (ff.neighbor_list(), TotalCharge()):
    batch = task.execute(batch)   # molix builds the pair list and the charge column
E = pots.calc_energy(batch)                  # () here — (B,) for a batch
out = ff.to_potentials(wrt=("atoms.pos",))(batch)   # graphs.energy, atoms.forces (N, 3) kcal mol⁻¹ Å⁻¹
```

Those two tasks are what a Class-I field needs on every batch, and both are the
field's semantics rather than script glue. In a training run they are mounted
once — `DataModule(..., batch_tasks=[ff.neighbor_list(), TotalCharge()])` — and
run per batch inside the collate.

`Potentials` consumes the `edges` list molix builds and never enumerates pairs
itself, so a list is required once a pair style is declared. `ff.neighbor_list()`
is that list: the field resolves the largest radius among the terms that will
run (here both pair styles were declared at 12 Å) and keeps the `symmetry=True`
default, because molpot prices the `source < target` half of a bidirectional
list and a half list built the other way would make every pair term cost exactly
zero. `pbc=True` switches the Coulomb radius to the Ewald split's real-space
cutoff for a periodic system.

`TotalCharge()` writes `graphs.total_charge` (B,) in e — `0.0`, i.e. *these
graphs are neutral* — which a charge model is fail-closed on and which
`bind_topology` cross-checks against the sum of `atoms.charge` (0 here, by
construction). A dataset carrying its own charge column keeps it.

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
    MD(torch.float64)            # trajectory precision
    .set_forcefield(ff)          # the ForceField itself, not a wrapper
    .set_integrator(VelocityVerletIntegrator(0.5))
)
```

`MD` is a composer: each component is constructed on its own and attached, and
every setter re-runs `ff.to_potentials()`. Masses come from the system
(`atoms.mass`), bound at `run`, and the conversion into integrator units
(amu, Å, fs) is read off `ff.units` by `molix.units.preset_energy_to_md` — a
Class-I ForceField at the default `units="real"` reports kcal/mol, and nothing
in the example spells out a conversion factor.

No neighbour list is mounted here, so `edges` stays whatever was bound into the
batch — acceptable for a two-atom fragment whose single pair never leaves the
cutoff, wrong for anything that diffuses. A real run attaches
`molix.md.NeighborList(cell=…, cutoff=…, positions=…, skin=…)` with
`set_neighbors`, which owns its own rebuild cadence and needs a cell (it is a
minimum-image list). Two further policies are `MD`'s own and re-applied on every
rebuild: `set_compiler(Compiler())` compiles the analytic NVE core and the
ordinary energy primitive used by the independent NPT virial derivation,
and `MD(dtype, potential_dtype=…)` evaluates the potential at a second
precision. Both are in the [MD guide](../../molix/user-guide/md.md).

## 3. Write LAMMPS files

A deck is three files: `system.data` (atoms, box, topology), a bonded include
and a pair include. The first two are molpy's; the nonbonded section is molnex's
own, because a field that declares pair styles has no faithful spelling on the
molrs side — LAMMPS wants `hybrid/overlay` and one explicit `pair_coeff` per
type pair, and molrs would write plain `hybrid`, which hands each type pair to a
single sub-style and so drops Coulomb from every Lennard-Jones pair.

```python
from molix.ff_export import LammpsAdapter, LammpsWriter

writer = LammpsWriter()
types = ff.assign_types(batch)          # one condensation, three files
writer.write_data("system.data", batch, types=types)
writer.write_settings("system.in.settings", ff, batch, types=types, skip_pair_style=True)
writer.write_pair_block(
    "system.in.pair", LammpsAdapter().translate(ff.to_ir()), types=types
)
```

`write_settings` calls `ff.clone().realize(batch, types=types)` then `molpy.io`
— the assignment made above is what registers the molpy types, so a deck is
condensed once and every file numbers its types from that one answer. There is
no `to_molpy()`. `skip_pair_style=True` drops every `pair:*` style from that clone,
so the include is bonded-only and the two files cannot disagree about the
nonbonded model; it also carries no `units` line, since LAMMPS refuses `units`
once `read_data` has made a box. `write_pair_block` then emits the `pair_style`
line, one `pair_coeff i j` for every `i <= j` *type* pair mixed with the
combining rule the field declares (`mixing` on `pair:lj/cut`; geometric for
OPLS-AA), `kspace_style pppm` with `kspace_modify gewald` under a
lattice Coulomb solver, and `special_bonds` from `ff.nonbonded_scaling`.

The box comes from `batch["graphs", "cell"]` when the batch carries one and is a
padded open box otherwise. Types are condensed on parameter equality by
`ForceField.assign_types` — atoms sharing (mass, charge, ε, σ) are one atom type,
interactions between the same atom types with the same constants one bonded type
— and all three calls take that one result, so the data file's labels, the
include's coeff names and the pair block's numbers are the same typing. The
labels are `t1`/`b1`/`a1`/`d1`/`i1`-style, zero-padded to the type count so
LAMMPS accepts them and molrs, which sorts labels as text, numbers the types in
class order. Source the three files in this order:

```
units real
atom_style full
read_data system.data
include system.in.settings
include system.in.pair
```

A bonded-only field needs neither the flag nor the pair block. The field above
is not one: `OPLSAAForceField` declares `pair:coulomb`, whose default solver is
a lattice sum, so its deck belongs in a periodic box.
`interface/tests/parity_classical.py` runs both shapes against a real `lmp` —
`--case butane` (bonded-only, open boundary) agrees with molnex to 2.6e-12 in
relative energy, `--case periodic` (8 waters plus Na⁺ and Cl⁻ in a 12 Å cube,
switched Lennard-Jones and Ewald-split Coulomb) to 9.9e-9.
