<!-- mol:note:topic:learnable-classical-ff -->
# Learnable classical FF — ForceField path (2026-08-11)

## Rule (binding)

### OOP, single responsibility, no glue

| Type | Owns |
|------|------|
| `ForceField` | styles + param tables; `def_*style`, `with_params`, `to_potentials()` |
| `Potentials` | `get_schema`, `calc_energy`, `calc_forces` (bound to one FF) |
| `ClassicalIMM` | encoder + heads → **returns ForceField**; `project` only |

No free-function helpers as public API. No `fill` / parameterizer / composer façade.
No energy method on `ClassicalIMM` — energy is always on `Potentials`.

### User path

```python
ff = ForceField.class_i(dihedral="fourier")
model = ClassicalIMM(encoder, forcefield=ff, bond_head=..., ...)
ff_t = model(batch)                          # ForceField
E = ff_t.to_potentials().calc_energy(batch)
ff_opls = model.project(ForceField.opls())   # general → OPLS form only
```

### `project` contract

- **Does:** map continuous/general param tables into a concrete style set
  (e.g. multi-term Fourier → OPLS `k1..k4` with LAMMPS sign/½ convention).
- **Does not:** train, evaluate energy, read XML.
- **Fails loud:** missing required source params, unsupported style pair,
  non-zero Fourier residual outside n∈{1,2,3,4} or γ∉{0,π}.
- **Parity:** when the source is already on the OPLS manifold (n=1..4,
  γ=0,π,0,π), `E`/`F` after project match to numerical tolerance.

### Charge equilibration (P0)

Algorithm-named solvers under ``molpot.charge`` (no product brands):

```python
from molpot.charge import (
    AtomicChargeModel, ChiEtaHead,
    DiagonalHardnessKernel, ElectronegativityEqualization,
)

charge = AtomicChargeModel(
    ChiEtaHead(feature_dim=…),
    ElectronegativityEqualization(DiagonalHardnessKernel()),
)
model = ClassicalIMM(encoder, forcefield=ff, charge_model=charge)
ff_t = model(batch)   # writes atoms.q via EEM KKT; energy still to_potentials()
```

- **Produces ``q`` only** — Coulomb energy via ``pair:coulomb`` on ``Potentials``.
- Kernels: ``DiagonalHardnessKernel``, ``ShieldedCoulombKernel`` (Ohno).
- Nonbonded: **Potential terms own cutoffs**; ``Potentials`` only assembles them.
  - ``ff.def_pairstyle("lj/cut", cutoff=12.0)`` → ``LJCutPotential``
  - ``ff.def_pairstyle("coulomb", calculator="ewald"|"pme"|"p3m", smearing=…)``
  - open → ``CoulombOpenPotential``; cell present → ``CoulombPeriodicPotential``
  - neighbors: open all-pairs; periodic → molrs ``NeighborQuery``
  - kernels reused: ``lj126_pair_energy``, ``CoulombPotential``, Ewald/PME/P3M calculators
  - scales: ``ForceField.nonbonded_scaling`` (1–2/1–3/1–4)
- Later: ACKS2 as separate solver type.

### Status

General Class-I and OPLS/GAFF are the **same** container type (specificity differs).
IO stays in molpy.io (no `ForceField.from_xml`).

### Experiment ops (molexp × molhub × molmcp)

Binding for the Class-I reproduction track:

| Layer | Owns |
|-------|------|
| **molexp** | `Workspace → Project → Experiment → Run`; metrics `t/k/s/v`; sourced `KnowledgeItem` |
| **molhub** | Coordinates only: `dataset:espaloma/phalkethoh-mm-small@1` (models: run assets until `model:` coords exist) |
| **molmcp molexp plane** | Scaffold + **knowledge** (`list_knowledge` / `write_knowledge` / `harvest_run` / `get_run_metrics`) |
| **molnex scripts** | Science: `setup_class_i_protocol.py` → `espaloma_mm_protocol_train.py --workspace … --run-id …` |

Norms:

1. Register experiment/run via setup (idempotent; **refreshes** parameter_space).
2. Train under `Run.start()` with `MetricsWriter` + `mark_succeeded` + `harvest_run(Finding)`.
3. Knowledge is **sourced** (`SourceRef` required) — no free Notes for protocol claims.
4. Dataset bytes via molhub convert/cache; coordinate is the audit handle, not the blob path.

Operator workspace: `…/workspaces/mm-param-learning`, experiment `class-i-full-energy-mini`.

### Supersedes

Composer / Parameterizer / IR bags as user path; glue free functions; `model.fill`.
