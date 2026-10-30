# A minimal end-to-end case: train → static inference → AOTI

| Field | Value |
|---|---|
| Opened | 2026-10-30 |
| Chain | rondo-enc; architecture discussion of 2026-10-30 (encoder → canonical features → parameters → terms) |
| Status | dropped 2026-10-30 (operator: "我们不做这个方向了"). §2.1–2.2 landed (atom energies, Rondo writes atoms.energy); §2.3 / §3 not done |
| Scope | operator 2026-10-30 ("先用一个最小的case，验证架构的正确性，包括训练、推理、aoti导出") |

## 1. Question

Does the layered design hold end to end on one molecule: the parameters are trained through
the joint step, predicted **once** for inference (no encoder or head on the per-step path),
and the static field exports through the engine-neutral contract (`molix.deploy`) to an
AOTI package that reproduces it?

## 2. What is built (molnex)

1. **Atom energies, one owner** (`molpot.potentials`): a pair style exposes
   `pair_energy(frame, **params) -> (pair_index, per_pair)`; `forward` reduces it per graph
   as before. `Potentials.atom_energy(frame, params, *, styles=None) -> (N,)` splits each
   term's energy equally over its atoms (a bond ½ each, an angle ⅓, a torsion and an
   improper ¼, a live pair ½ each) — LAMMPS `compute pe/atom`'s convention — so the atom
   energies sum to the graph energy.
2. **Rondo writes `atoms.energy`** in `energy_at` (the molzoo product contract requires it;
   Rondo did not), from `Potentials.atom_energy`.
3. **`StaticRondo(model, frame)`** (`molzoo/rondo/static.py`, a `molix.autograd.Potential`):
   the static form of one molecule's field. Construction runs `model.predict(frame)` once,
   under `no_grad`, and keeps as buffers: every style's potential parameters, the bonded
   index tables, and the molecule's 1-2 / 1-3 / 1-4 relation as a dense `(N, N)` table.
   `energy(frame)` reads only `atoms.pos` / `atoms.batch` and the incoming `pairs`, looks up
   each pair's relation, and prices with the stored parameters — no encoder, no head. It
   writes `atoms.energy` and `graphs.energy`; `forward` is the `Derivative`. Topology-level
   features (spec 10 §7) are what make the cache valid for a whole trajectory.

Unit tests (`tests/`): the atom energies sum to the graph energy per style; `StaticRondo`
equals `Rondo` in energy and forces on the molecule it was built from; a `StaticRondo`
call runs no encoder; `StaticRondo.energy` traces as one graph (slow tier).

## 3. The case (lab, `rondo-enc/minimal/`)

One held-out THEMol hessian-relax molecule (its relaxation steps: energies and forces),
a tiny Rondo (the unit-test sizes), float64, CPU:

1. **Train** — `JointTrainStep` with one `forward` objective (energy + force) for a few
   hundred steps; the loss falls.
2. **Infer** — `StaticRondo(trained, molecule)`: energies and forces on every relaxation step
   equal the trained `Rondo`'s (rel ≤ 1e-12); over 100 force calls the encoder runs 0 times
   (forward hook count), and `Rondo` runs it 100.
3. **Export** — `AtomEnergyAdapter(StaticRondo)` → `molix.export.aoti.package` (fixed
   shapes: the molecule's atoms and its full intramolecular pair list) → `load` → energy,
   forces, atom energies and virial equal the eager adapter's (rel ≤ 1e-10) on every
   relaxation step; `molix.deploy.verify` passes.

## 4. Result

Paused before §2.3. Landed: `PairPotential.pair_energy` and `BondedPotential.term_energies`
(the per-term primitives), `atom_energy` on both and on `Potentials` (equal split, unit
tests: per-pair halves / per-term shares and the graph sums); Rondo's `price` and
`energy_at` go through `atom_energy` (one path), and `energy_at` writes `atoms.energy`.
