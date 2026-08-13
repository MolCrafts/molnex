---
spec: universal-class-i-04-bundle
slug: universal-class-i-04-bundle
created: 2026-08-13
criteria:
  - id: ac-001
    summary: PesBundle validates topology and stacked pos at construct
    type: code
    pass_when: |
      TestPesBundle in tests/test_molix/test_data/test_pes_bundle.py
      constructs PesBundle(topology, pos, group_id=...) with atoms.Z (N,)
      and pos (C, N, 3); rejects atoms.pos or edges in topology, C==0,
      and N mismatch with ValueError.
    status: pending
  - id: ac-002
    summary: topology_batch contains no coordinates
    type: code
    pass_when: |
      TestPesBundle asserts topology_batch() has no key named pos at the
      top level or under atoms (encoder contract), while atoms.Z and
      covalent namespaces remain.
    status: pending
  - id: ac-003
    summary: Conformation views share bond_index and differ in pos
    type: code
    pass_when: |
      TestPesBundle asserts conformation_batch(0) and conformation_batch(1)
      have torch.equal bonds.bond_index and not torch.equal atoms.pos;
      atoms.pos matches the corresponding pos[i] slice (atol 1e-8 Å).
    status: pending
  - id: ac-004
    summary: from_conformer_list stacks matching graphs only
    type: code
    pass_when: |
      TestPesBundle asserts from_conformer_list on two nested TensorDicts
      with equal Z/valence and distinct pos yields n_conformers==2 and a
      topology_batch without pos; unequal bond_index raises ValueError.
    status: pending
  - id: ac-005
    summary: group_split_indices never leaks conformers of one molecule
    type: code
    pass_when: |
      TestPesBundle builds 10 PesBundle objects × 2 confs, calls
      group_split_indices on flattened conformer_group_ids with
      ratios=(0.8, 0.1, 0.1) and seed=0, and asserts 8/1/1 unique
      group_ids, pairwise-disjoint splits, and both confs of each
      group_id in the same split.
    status: pending
  - id: ac-006
    summary: PesBundle is exported from molix.data
    type: code
    pass_when: |
      src/molix/data/__init__.py imports PesBundle and lists "PesBundle"
      in __all__ (Group-splits segment, before group_split_indices).
    status: pending
  - id: ac-007
    summary: Google docstrings name tensor shapes and units
    type: docs
    pass_when: |
      PesBundle, topology_batch, conformation_batch, from_conformer_list,
      and conformer_group_ids in src/molix/data/pes_bundle.py have
      Google-style docstrings stating Å / e / caller-owned energy units.
    status: pending
  - id: ac-008
    summary: Regression pins two-conf water topology and coordinates
    type: runtime
    pass_when: |
      regressions/universal-class-i-04-bundle.py exits 0 and asserts:
      topology_batch has no pos; conformation_batch(0) and (1) share
      identical bonds.bond_index; atoms.pos equals the script's
      hard-coded water geometries (Å)
      pos0=[[0.0,0.0,0.0],[0.9572,0.0,0.0],[-0.2399,0.9266,0.0]] and
      pos1=[[0.0,0.0,0.0],[1.02,0.0,0.0],[-0.25,0.97,0.0]]
      with atol 1e-8; no third-party import or subprocess.
    status: pending
out_of_scope:
  - encoder / heads / losses implementation
  - training scripts and dataset download
  - MolHub / THEMol / OMol25 I/O
  - forcefield, IR, charge solvers, Condenser, ff_export, PiNet
  - relative-energy centering inside the bundle
  - make_pes_bundle factory or run_everything façade
---

# Acceptance — universal-class-i-04-bundle

Done 表示 `molix.data.PesBundle` 能拥有「一图 + 多构象」、给出无坐标的拓扑视图与带 `pos` 的构象视图，并且用现有 `group_split_indices` 按分子划分时构象不泄漏。不要求跑通 encoder 或势能数值。
