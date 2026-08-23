# MolNex Specs Index

Only live specifications appear in the active set. Completed, superseded, or
investigation-only specifications are closed explicitly below; closed records
must not be treated as authority for a new training submission.

## Active specifications

- [oplsaa-parameterizer-01-types](oplsaa-parameterizer-01-types.md) — OPLSAAForceField 类型面与 PotentialMeta improper schema [code-complete]
- [oplsaa-parameterizer-02-heads](oplsaa-parameterizer-02-heads.md) — ProperTorsionParamHead OPLS-AA k1..k4 直接输出 [code-complete]
- [oplsaa-parameterizer-03-parameterizer](oplsaa-parameterizer-03-parameterizer.md) — ClassicalIMM→Parameterizer；ClassIPredictions→ForceFieldPredictions [code-complete]
- [oplsaa-parameterizer-04-cut](oplsaa-parameterizer-04-cut.md) — 公开面硬切清零与链回归 regressions/oplsaa-parameterizer.py [code-complete]
- [md-forcefield-molpy-reexport-01-reexport](md-forcefield-molpy-reexport-01-reexport.md) — molix.md ForceField 再导出 molpy.ForceField；删除自制适配器 [code-complete]
- [md-forcefield-molpy-reexport-02-nblist](md-forcefield-molpy-reexport-02-nblist.md) — NeighborList 写入 live MIC 几何；MD 调度 nblist 再求值 [code-complete]
- [md-forcefield-molpy-reexport-03-potentials](md-forcefield-molpy-reexport-03-potentials.md) — molpot pair 项读 batch 位移/距离，禁止二次 PBC [code-complete]
- [md-forcefield-molpy-reexport-04-mace](md-forcefield-molpy-reexport-04-mace.md) — MACE 消费 edge_diff，shifts 不再是 PBC 通道 [code-complete]
- [md-forcefield-molpy-reexport-05-wire](md-forcefield-molpy-reexport-05-wire.md) — 全树调用方改走 def_pairstyle + set_forcefield/set_neighbors [code-complete]
- [pinet-quantization-thermal-noise-02-md-driver](pinet-quantization-thermal-noise-02-md-driver.md) — in-process velocity-Verlet and Langevin MD driver; approved with acceptance work pending.
- [cuet-force-doublebackward](cuet-force-doublebackward.md) — cuet equivariant Linear double-backward repair; in progress with acceptance work pending.
- [mace-neighbor-graph-correctness](mace-neighbor-graph-correctness.md) — independent multi-image neighbor oracle; code complete with one acceptance criterion pending.

## Closed specifications and chains

| Spec or chain | Closed | Disposition |
|---|---|---|
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
