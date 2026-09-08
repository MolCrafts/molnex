---
slug: composable-a0-debt-07-frame-native
status: approved
criteria:
  - id: ac-001
    summary: the chains' batch readers go through the TensorFrame view
    type: code
    pass_when: |
      One wrap, molix.batch.as_tensor_frame (in molix/batch/frame.py, beside
      the view), is what Potentials, ForceField.assign_types / realize,
      LammpsWriter, TotalCharge and the MD adapter read through - no twin per
      package; a TensorFrame and a plain TensorDict give the same energies /
      assignments / decks / rollouts; a nested Mapping is wrapped; a list raises
      TypeError; TotalCharge refuses a plain dict; the duck-typed _namespace /
      _ns_tensor readers are gone. The storage-sharing fact is asserted once, in
      tests/test_molix/test_batch/test_frame.py; as_tensor_frame's four branches
      once, in tests/test_molix/test_batch/test_frame.py.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "compute job 2184349 (fast tier of test_potentials / test_container / test_lammps_writer / test_driver / test_charge / test_view (since folded into test_frame.py) after the architect review round: 153 passed; job 2179351 before it: 148 passed) - a TensorFrame and the plain batch give bitwise-equal energies / forces / virial / assignments / decks; forward writes into the wrapped batch; a nested Mapping is wrapped; a list raises TypeError (test_frame); TotalCharge refuses a dict (test_charge); _namespace / _ns_tensor deleted (0 uses); the writer's _tensor twin deleted; grep of src/tests/interface/benchmarks finds one spelling of the wrap (molix.batch.as_tensor_frame) and no 'spelled here' twin"
  - id: ac-002
    summary: every system the chains build is a molpy Frame collated by collate_molecules
    type: code
    pass_when: |
      The butane-pair box, the parity tool's butane and water/ion cases, the
      two NVE benchmarks and the driver / charge / container / writer fixtures
      the chains added build molpy Frames (atoms columns, bonds / angles /
      propers blocks, meta total_charge) and go through collate_molecules; the
      periodic ones set graphs.cell from frame.box after collate (F35); parity
      both cases PASS with the recorded numbers; the symmetry / NVE gates keep
      their recorded values. Named residue, by design: _lj_pair_batch,
      _coulomb_batch and _polar_batch in test_potentials.py stay hand-built -
      their neighbour lists are function arguments (edge_index + shifts) so
      the FD strain derivative and the trace gates see a list that is a
      function of the positions, and a Frame carries no list (spec Out of
      scope).
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "tests/conftest.py butane_pair_frame + collate_frames (one Frame per periodic cell, F20; fp64 columns re-read after collate, F36; graphs.cell from the box, F35; atoms.vel written, F38); parity tool butane_frame / water_ion_frame and both benchmarks' _cluster_frame / _lattice_frame go through collate_molecules + ff.neighbor_list().execute + TotalCharge(); after the review round jobs 2184351 / 2184352 parity PASS with the recorded numbers (butane E 0.1309516320367863 force_max 4.921673210243161, periodic E -47.23502966626038 force_max 89.37654718130673, all rel 0.0) and job 2184355 NVE (ljcut) PASS with E_tot(0) -1.143881e-01 identical to job 2182238; the round-1 jobs 2182235 / 2182236 / 2182237 / 2182238 were bit-identical to the HEAD builders; the three hand-built fixtures are named in the spec's Out of scope; the five F35 sites and four F36 sites are listed in the ledger rows"
  - id: ac-003
    summary: the typed view costs no graph break
    type: performance
    pass_when: |
      The slow trace gates (traces_clean, compiled_route, classical_core,
      graph_breaks) report the same break sites as before this step, and the
      classical_core gate traces the container production hands the core -
      as_tensor_frame(batch), the view - not a plain batch the run never
      compiles.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "compute job 2184350 (after the review round; _classical_core and _placed_pairs take the view only, the gate explains captured[-1] on as_tensor_frame(lj_batch) / as_tensor_frame(periodic)) - slow trace gates (traces_clean / compiled_route / classical_core / graph_breaks) 5 passed at the recorded break sites, lj/cut + bonded core 1 graph / 0 breaks, periodic core breaks only at the force.py seam; job 2179352 before the round: 5 passed; job 2179302 probe: wrap-then-read, view passthrough and plain passthrough each trace 1 graph / 0 breaks"
---
