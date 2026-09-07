# Notes

Passive internal project context — outlives individual features and specs.
See `CLAUDE.md` for the active workflow and invariants.

| File | Purpose |
|------|---------|
| `notes.md` | Evolving architectural decisions, captured by `/mol:note` |
| `architecture.md` | Project blueprint — modules, public surface, layer roles. Populated by `/mol:map`, consumed by `librarian` during `/mol:spec` |
| `open-questions.md` | Uncertainties from bootstrap; fill in over time |
| `learnable-classical-ff.md` | Binding placement rules for the learnable Class-I force-field path (`ForceField` ⊂ `molpy.ForceField`, `Parameterizer`, EEM charges) |
| `mace-polar-10k-review.md` | MACE-POLAR 10k PEO execution review; single-GPU rollout contract; multi-GPU global-reduction rules |
| `composable-potentials.md` | Spec S1–S8 ↔ molnex mapping, shared contract / manifest / variational-polarization designs, `composable-a0…a2` chain roadmap, findings ledger |
| `production-training-checklist.md` | Binding preflight and sign-off record before any training submission |

More specialised directories (`contracts/`, `rubrics/`, `decisions/`, `debt/`, `handoffs/`)
appear when the repo grows real content that justifies them.
