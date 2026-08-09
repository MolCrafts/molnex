# MACE-OMOL port notes

Goal: load the official **MACE-OMOL** foundation model into MolNex
(`molrep` / `molpot` / `molzoo`, cuEquivariance) and reproduce energy/forces.

## Dependency policy

MolNex package code and in-repo scripts **must not** import ASE, e3nn, or
`mace-torch`. Allowed runtime stack: MolCrafts packages (`molpy`, `mollog`,
`molcfg`, …), PyTorch / TensorDict, numpy, and cuEquivariance.

Upstream MACE / e3nn comparison is offline, out of this tree: run any
bit-for-bit oracle against a separate checkout and paste numbers into the
spec (`src/molzoo/specs/mace_omol.md` §7.4). Do not re-introduce those
imports here.

## In-tree artifacts

- `SPEC.md` — port design notes
- `src/molzoo/mace/variants.py` — `MACEOMol` (thin alias over
  `molzoo.mace.potential.MACEPotential`) + the `load_omol_state_dict`
  back-compat loader
- `src/molzoo/mace/checkpoint.py` — `OMOL_REMAP`, the official-weight key
  remap consumed by `MACEPotential.from_checkpoint`
- `src/molzoo/specs/mace_omol.md` — paper↔code contract and run log
  (mirrored byte-for-byte at `docs/molzoo/specs/mace_omol.md`)
