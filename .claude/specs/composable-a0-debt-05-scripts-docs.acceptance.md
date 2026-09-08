---
slug: composable-a0-debt-05-scripts-docs
status: approved
criteria:
  - id: ac-001
    summary: the scripts import and declare the batch tasks they need
    type: code
    pass_when: |
      _probe_nan_term.py is gone; the ledger records F23 as closed by the peer
      session's Frame-native port and F12 / F21 as carried by (or follow-ups on)
      that series with the exact sites; ruff clean on scripts/.
    status: verified
    last_checked: 2026-09-08
    verified_by: impl
    evidence: "scripts/mm_param_learning/_probe_nan_term.py and its run_class_i_nan_probe_gh200.sbatch deleted; ledger F23 closed by the peer session's Frame-native port, F12 / F21 reassigned and closed by composable-a0-debt-06 (ForceField.neighbor_list + molix.batch.TotalCharge); ruff clean on scripts/"
  - id: ac-002
    summary: CLAUDE.md states the tier rule once, consistently
    type: docs
    pass_when: |
      The 'No e2e runs' sentence names it as the fast-tier rule and points at the slow-tier exception; the sentence in the two-tier paragraph is unchanged; the hunk lands without touching the peer's CLAUDE.md hunks.
    status: pending
    last_checked: null
---
