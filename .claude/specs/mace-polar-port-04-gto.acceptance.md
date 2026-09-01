---
slug: mace-polar-port-04-gto
created: 2026-08-30
criteria:
  - id: ac-001
    summary: GTO kernel is official erf(r/(2σ))/r not paper Eq.37
    type: code
    pass_when: |
      GTOCoulombKernel(sigma=1.5) at r=1.0 equals erf(1/3)/1;
      it is not equal to erf(1/√2 / 1.5)/1 (paper Eq.37);
      sigma<=0 raises.
    status: verified
    last_checked: 2026-08-30
    verified_by: tests/test_molpot/test_potentials/test_gto/test_kernel.py
  - id: ac-002
    summary: Fukui uses ASE multiplicity and (Q±(M-1))/2 targets
    type: code
    pass_when: |
      Closed-shell Q=0 M=1 leaves already-equilibrated monopoles unchanged;
      Q=1 M=1 targets n_up=n_down=0.5; Q=0 M=2 (doublet) targets
      n_up=0.5 n_down=-0.5; fukui channels normalize to 1 per graph.
    status: verified
    last_checked: 2026-08-30
    verified_by: tests/test_molpot/test_potentials/test_gto/test_fukui.py
  - id: ac-003
    summary: Isolated l=0 Hartree is k_e q_i q_j erf(r/(2σ))/r
    type: code
    pass_when: |
      Two unit charges 1 Å apart, σ=1.5 Å, energy equals
      eV_A * erf(1/3); self pairs are excluded; two graphs do not mix.
    status: verified
    last_checked: 2026-08-30
    verified_by: tests/test_molpot/test_potentials/test_gto/test_hartree.py
  - id: ac-004
    summary: Regression pins dump σ=1.5, kernel vs Eq.37, and Fukui multiplicity
    type: runtime
    pass_when: |
      regressions/mace-polar-port-04-gto.py exits 0 with hard-coded σ=1.5,
      kernel erf(1/3), Hartree two-charge golden, Fukui M=2S+1; no
      graph_longrange/e3nn/mace import.
    status: verified
    last_checked: 2026-08-30
    verified_by: regressions/mace-polar-port-04-gto.py
out_of_scope:
  - k-space PBC, external field, Polar molzoo wiring, official energy parity (05)
---

# Acceptance — mace-polar-port-04-gto

本步完成 = 孤立分子 GTO 核是官方 `erf(r/(2σ))/r`，Fukui 按多重度约束自旋通道 0e，l=0 Hartree 为 \(k_e q_i q_j\,\mathrm{erf}/r\)。无 PBC、无 molzoo 接线。
