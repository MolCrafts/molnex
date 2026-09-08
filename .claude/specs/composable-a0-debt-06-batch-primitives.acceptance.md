---
slug: composable-a0-debt-06-batch-primitives
status: approved
criteria:
  - id: ac-001
    summary: TotalCharge writes or normalises graphs.total_charge as a float column
    type: code
    pass_when: |
      On a batch without the column TotalCharge(0.0).execute writes graphs.total_charge
      of shape (B,) in atoms.pos's dtype; an int64 column is cast to that dtype;
      a float column is left untouched; task_id names the charge
      (tests/test_molix/test_batch/test_charge.py).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molix/test_batch/test_charge.py (4 passed) - absent column filled (B,) in atoms.pos's dtype (fp64 and fp32 probed), int64 column cast with values kept, float column returned as the same tensor, task_id total_charge:<float>; BatchTask membership asserted"
  - id: ac-002
    summary: the force field declares the neighbour list it prices against
    type: code
    pass_when: |
      OPLSAAForceField().neighbor_list(pbc=True) is a molix.batch.NeighborList
      with cutoff == the larger of its lj_cutoff and Ewald real_space_cutoff
      (15.0 by default), symmetry True, pbc True; pbc / max_num_pairs are
      forwarded; under pbc=False the declared coulomb cutoff is the radius
      (12.0 for cutoff=12.0) and a bare open pair:coulomb (every pair) or a field
      with no pair style raise ValueError
      (tests/test_molpot/test_forcefield/test_container.py).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/test_molpot/test_forcefield/test_container.py - OPLSAAForceField().neighbor_list(pbc=True) is a molix.batch.NeighborList with cutoff 15.0, symmetry True, pbc True; lj/cut 8.0 with pbc=True / max_num_pairs=64 forwarded; coulomb cutoff=12.0 -> 12.0 under pbc=False and 15.0 under pbc=True; the bare open template raises ValueError naming cutoff; a bonded-only field raises naming pair; TypeAssignment.atom_charges carries the keyed charges (writer test: data file rows 0.3 / -0.3 from the field's table)"
  - id: ac-003
    summary: the four scripts use the primitives and carry no placement glue
    type: code
    pass_when: |
      grep for LJ_CUT_CUTOFF, _PLACE_EDGES, _dtype_of and manual total_charge
      emission in scripts/mm_param_learning is empty; each DataModule / collate
      choke point mounts ff.neighbor_list() and TotalCharge(); the four modules
      import under PYTHONPATH=src and ruff is clean.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "grep for LJ_CUT_CUTOFF / _PLACE_EDGES / _dtype_of in scripts/mm_param_learning is empty; 7 mounts of ff.neighbor_list() / TotalCharge() across class_i_fit.datamodule, espaloma build_datamodule + frames_to_batch, align._collate (scale_audit follows); the four modules import under PYTHONPATH=src:.; ruff clean; end-to-end collate -> placed edges -> graphs.total_charge (2,) fp32 -> forward finite (implementer run); the scripts' field declares pair:coulomb cutoff=OPEN_COULOMB_CUTOFF_A 10.0 (the truncation they already ran)"
---
