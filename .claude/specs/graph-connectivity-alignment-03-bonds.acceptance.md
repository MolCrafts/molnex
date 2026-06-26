---
slug: graph-connectivity-alignment-03-bonds
criteria:
  - id: ac-001
    summary: harmonic.py raises clear error when [E,2] edge_index is passed as bond_index
    type: code
    pass_when: |
      A regression test in tests/test_molpot/test_potentials/test_bonds.py
      passes an [E,2] edge_index tensor as bond_index to BondHarmonic.forward
      and asserts a ValueError naming the edge≠bond / bond_index-[2,N] contract
      is raised — and asserts no scalar energy is returned.
    status: verified
    last_checked: 2026-06-26
  - id: ac-002
    summary: multi-molecule BondHarmonic energy equals sum of per-molecule energies
    type: code
    pass_when: |
      A test builds a 2+-molecule batch, collates it (bond_index atom-offset via
      the phase-01 registry), and asserts the batched BondHarmonic energy equals
      the sum of the independently computed per-molecule energies within 1e-6.
    status: verified
    last_checked: 2026-06-26
  - id: ac-003
    summary: the 3-way connectivity fallback is removed from harmonic.py
    type: code
    pass_when: |
      grep over src/molpot/potentials/bonds/harmonic.py finds no
      data["edge_index"], data.get("bonds"), or data["bonds"].get(...) fallback
      path; bond_index is sourced only from kwargs/data["bond_index"].
    status: verified
    last_checked: 2026-06-26
  - id: ac-004
    summary: bond_index [2,N] + bond_types round-trips through collate_molecules preserving connectivity
    type: code
    pass_when: |
      A test feeds a synthetic bond_index [2,N] + bond_types (and molpy
      atomi/atomj columns via bond_index_from_columns) through collate_molecules
      and asserts source/target atom pairs and bond_types are preserved after
      atom-offset rebase (batch["bonds","bond_index"] / ["bond_types"]).
      AMENDED (impl): scope narrowed from the original "collate_molecules AND
      collate_packed" — the collate_packed/PackedCache bonds-bucket path is
      DEFERRED as YAGNI: no dataset produces bond_index samples today, so the
      packed-cache bond bucket would be speculative infra. The spec-01 registry
      already reserves the bond_index offset slot; add the packed path when a
      real producer (a bonded-topology dataset) lands. See spec Out of scope.
    status: verified
    last_checked: 2026-06-26
  - id: ac-005
    summary: CLAUDE.md carries the torch_geometric parity table + bond_index/edge_index distinction
    type: docs
    pass_when: |
      CLAUDE.md contains a torch_geometric collate/Batch parity table listing
      __inc__/__cat_dim__ (implemented via the lazy registry), and follow_batch /
      ptr / to_data_list/unbatch / exclude_keys / HeteroData as SKIP with reasons,
      plus the packed-mmap-collate divergence note; and a stated distinction
      between bond_index ([2,N], covalent) and edge_index ([E,2], cutoff/neighbor).
    status: pending
---

# Acceptance criteria

- **ac-001 (code)** — The exact scientist-found bug: an `[E, 2]` neighbor pair handed to the bonded potential must fail loudly, not return a silently wrong energy. The empty guard `size(1) == 0` is removed/replaced because it never triggers for `[E, 2]`.
- **ac-002 (code)** — Proves the offset fix. A topological term summed over a batch must be partition-additive once each molecule's `bond_index` is rebased by its atom offset (consumed from the sub-spec 01 registry). Failure here means `bond_index` is not being offset.
- **ac-003 (code)** — Enforces the category boundary structurally: with the fallback gone, `edge_index` can never reach `BondHarmonic`. A grep gate keeps the deletion permanent.
- **ac-004 (code)** — The canonical `[2, N]` + `bond_types` contract survives both the per-sample and packed-mmap collate paths; connectivity (source/target identity) and bond types are preserved after rebase.
- **ac-005 (docs)** — Captures the parity table and the bond_index/edge_index distinction in CLAUDE.md so the deliberate divergences from PyG (and the deliberate transpose) are documented contract, not tribal knowledge.
