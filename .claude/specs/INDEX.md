# MolNex Specs Index

Only live specifications appear in the active set. Completed, superseded, or
investigation-only specifications are closed explicitly below; closed records
must not be treated as authority for a new training submission.

## Active specifications

- [mace-polar-port-02-blocks](mace-polar-port-02-blocks.md) — NonLinearBiasReadout.out_mul、自旋源、层混合 [code-complete]
- [pinet-quantization-thermal-noise-02-md-driver](pinet-quantization-thermal-noise-02-md-driver.md) — in-process velocity-Verlet and Langevin MD driver; approved with acceptance work pending.
- [cuet-force-doublebackward](cuet-force-doublebackward.md) — cuet equivariant Linear double-backward repair; in progress with acceptance work pending.

## Closed specifications and chains

| Spec or chain | Closed | Disposition |
|---|---|---|
| md-forcefield-molpy-reexport-02-nblist | 2026-09-04 | Completed; remaining ledger flipped after regressions/md-forcefield-molpy-reexport-02-nblist.py exited 0. NeighborList live MIC geometry. |
| md-forcefield-molpy-reexport-03-potentials | 2026-09-04 | Completed; remaining ledger flipped after regressions/md-forcefield-molpy-reexport-03-potentials.py exited 0. pair terms consume placed distances. |
| md-forcefield-molpy-reexport-04-mace | 2026-09-04 | Completed; remaining ledger flipped after regressions/md-forcefield-molpy-reexport-04-mace.py exited 0. MACE consumes edge_diff not shifts. |
| md-forcefield-molpy-reexport-05-wire | 2026-09-04 | Completed; remaining ledger flipped after regressions/md-forcefield-molpy-reexport-05-wire.py exited 0. def_pairstyle + set_forcefield/set_neighbors. |
| oplsaa-parameterizer-01-types | 2026-09-04 | Completed; all acceptance criteria verified. OPLS-AA types + improper schema. |
| oplsaa-parameterizer-02-heads | 2026-09-04 | Completed; all acceptance criteria verified. ProperTorsionParamHead k1..k4. |
| oplsaa-parameterizer-03-parameterizer | 2026-09-04 | Completed; all acceptance criteria verified. Parameterizer + ForceFieldPredictions. |
| oplsaa-parameterizer-04-cut | 2026-09-04 | Completed; all acceptance criteria verified. public-surface cut + chain regression. |
| md-forcefield-molpy-reexport-01-reexport | 2026-09-04 | Completed; all acceptance criteria verified. molix.md ForceField re-exports molpy.ForceField. |
| mace-neighbor-graph-correctness | 2026-09-04 | Completed; all acceptance criteria verified. independent multi-image neighbor oracle. |
| mace-polar-port-01-backbone | 2026-09-04 | Completed; 7/7 criteria verified (ledger flipped 2026-09-04: backbone regression + unit tests). |
| mace-polar-port-03-field | 2026-09-04 | Completed; all criteria verified. SparseUvu / field readout. |
| mace-polar-port-04-gto | 2026-09-04 | Completed; all criteria verified. Native GTO Hartree. |
| mace-polar-port-05-wire | 2026-09-04 | Completed; all criteria verified. MacePolar + POLAR_REMAP isolated forward. |
| polar-compile-clean-03-wire | 2026-09-04 | Completed; all 14 criteria verified. Compiled Polar energy_core installed on the mlip; inductor aten.index_add fallback; GH200 job 1976727 fp32 0.190 s / 25.87 GiB / 0 recompiles. Merge unit with 02-seam. |
| polar-compile-clean-02-seam | 2026-09-03 | Completed; all 9 (code) criteria verified. Generic Compiler + ExplainSummary(break_sites), CellBound / CoreCompilable in molix.schema, rebind_cell single owner, PiNetPotential.compile deleted, raw explain sites migrated. Merge unit with 03-wire (closed 2026-09-04). |
| polar-compile-clean-01-kgrid | 2026-09-03 | Completed; all 16 criteria verified (12 evidence, 4 agent-auto). Fixed-capacity KSpaceGrid + explicit num_graphs; periodic GTO branch and MacePolar.energy_core at 1 graph / 0 breaks (commit eabfc6d7). Chain 02-03 active. |
| lammps-iface-05-pbc | 2026-08-30 | Completed; all 17 criteria verified (11 evidence, 6 agent-auto). Chain debt PAID 2026-08-30: aspirin goldens captured+verified and periodic 1/2-rank/Python three-way all ≤1.6e-15 on a GH200 source-built LAMMPS 22Jul2025 (job 1805410; runbook provenance in interface/lammps/README.md). |
| lammps-iface-04-facade | 2026-08-30 | Completed; 16/16 — the chain-debt aspirin parity run was captured and verified 2026-08-30 (goldens block in scripts/lammps_iface_parity.py, rel 0.0 energy / 8.7e-16 forces). |
| lammps-iface-03-graph | 2026-08-30 | Completed; all 16 acceptance criteria verified (15 evidence, 1 agent-auto). |
| lammps-iface-02-view | 2026-08-30 | Completed; all 14 acceptance criteria verified (12 evidence, 2 agent-auto). |
| lammps-iface-01-meta | 2026-08-30 | Completed; all 14 acceptance criteria verified (12 evidence, 2 agent-auto). Chain 02-05 active. |
| [dynamic-batching-packed-collate-02-collate](dynamic-batching-packed-collate-02-collate.md) | 2026-08-13 | Completed; all acceptance criteria verified. |
| [mace-omol-port-01-native-port](mace-omol-port-01-native-port.md) | 2026-08-13 | Completed; all acceptance criteria verified. |
| [mace-omol-port-02-pipeline-integration](mace-omol-port-02-pipeline-integration.md) | 2026-08-13 | Completed; all acceptance criteria verified. |
| [lammps-pair-molnex-01-aoti-force-export-wall](lammps-pair-molnex-01-aoti-force-export-wall.md) | 2026-08-13 | Investigation closed; implementation requires a separate successor spec. |
| `learnable-classical-ff-01..09` | 2026-08-13 | Completed library chain; removed from the active specification set. |
| `universal-class-i-01..04` | 2026-08-13 | Completed library steps; removed from the active specification set. |
| `mm-param-val-01..08` | 2026-08-13 | Closed and removed from the active specification set; superseded by `universal-class-i-05-dataset-composition`. |
| [molix-data-dissolution-04-etl-exit] | 2026-08-14 | Completed; all 9 acceptance criteria verified. ETL layer removed from MolNex; molhub-side reception is molix-data-dissolution-05 (molhub repo). |
| [molix-data-dissolution-03-transforms] | 2026-08-14 | Completed; all 10 acceptance criteria verified. |
| [molix-data-dissolution-02-batch] | 2026-08-14 | Completed; all 11 acceptance criteria verified. |
| [molix-data-dissolution-01-store] | 2026-08-14 | Completed; all 9 acceptance criteria verified. |
| [universal-class-i-13-grad-audit] | 2026-08-14 | Completed; all 13 acceptance criteria verified. |
| [universal-class-i-12-torsion-weights] | 2026-08-14 | Completed; all 11 acceptance criteria verified. |
| [universal-class-i-11-step] | 2026-08-14 | Completed; all 14 acceptance criteria verified. |
| [universal-class-i-10-sampler] | 2026-08-14 | Completed; all 14 acceptance criteria verified. |
| [universal-class-i-09-packer] | 2026-08-14 | Completed; all 11 acceptance criteria verified. |
| [universal-class-i-08-adapter] | 2026-08-14 | Completed; all 12 acceptance criteria verified. |
| [universal-class-i-07-wilson] | 2026-08-14 | Completed; all 11 acceptance criteria verified. |
| [universal-class-i-06-hessian] | 2026-08-14 | Completed; all 10 acceptance criteria verified. |
| `universal-class-i-05-dataset-composition` | 2026-08-14 | Closed in MolNex because dataset composition is not a MolNex responsibility; moved to MolHub as `universal-class-i-01-dataset-composition`. |
