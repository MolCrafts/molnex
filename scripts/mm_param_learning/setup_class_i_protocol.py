#!/usr/bin/env python3
"""Register Class-I full-energy protocol under the molexp workspace.

Follows molexp hierarchy: Workspace → Project → Experiment → Run.
Datasets / coordinates are audited via MolHub; science runs use local
materialized trees referenced in run params.

Knowledge uses molexp's sourced :class:`~molexp.workspace.KnowledgeItem`
(``ProtocolNote`` / ``OpenQuestion``), not free-floating Notes without
``SourceRef``.

Example::

    PYTHONPATH=src:../molhub/src python scripts/mm_param_learning/setup_class_i_protocol.py
    # then train into the registered run (see printed train_cmd):
    python scripts/mm_param_learning/espaloma_mm_protocol_train.py ...
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.mm_param_learning.constants import (
    DATASET_COORDINATE_PRIMARY,
    DEFAULT_NORMALIZED_DATA_ROOT,
    DEFAULT_WORKSPACE_ROOT,
    PROGRAM,
    PROJECT_SLUG,
    PROTOCOL_EXPERIMENT_SLUG,
    PROTOCOL_NOTE_NAME,
    PROTOCOL_RUN_ID,
)
from scripts.mm_param_learning.materialize_workspace import MmParamLearningWorkspace


def _protocol_note_body() -> str:
    return f"""# Class-I MM protocol architecture (molnex × molexp × molhub)

Program: `{PROGRAM}`
Updated: {datetime.now(timezone.utc).strftime("%Y-%m-%d")}

## Hierarchy

```
Workspace  {DEFAULT_WORKSPACE_ROOT.name}
└── Project  {PROJECT_SLUG}
    ├── Experiment  espaloma-mm-protocol-mini     # bonded-only baseline
    └── Experiment  {PROTOCOL_EXPERIMENT_SLUG}   # bonded + EEM + LJ + Coulomb
```

## MolHub

| Role | Coordinate |
|------|------------|
| Energy labels / confs | `{DATASET_COORDINATE_PRIMARY}` |
| Local materialization | `{DEFAULT_NORMALIZED_DATA_ROOT}` |
| Model registry | *pending* — checkpoints as run assets until `model:` coords exist |

Data is fetched/materialized via the **Python molhub SDK** only. This note
never stores blobs; runs record coordinates + local `data_root`.

## Energy path (molnex)

```
ChemEncoder → ClassicalIMM
  bond/angle/proper heads
  LJParameterHead → atoms.epsilon/sigma
  AtomicChargeModel(ChiEtaHead, ElectronegativityEqualization)
       → atoms.q
ForceField.to_potentials()
  bonded + LJCutPotential + CoulombOpenPotential
```

Algorithm names only (no product brands on types): EEM/QEq-family
`ElectronegativityEqualization`, not Espaloma-named modules.

## Literature gap (honest)

| Item | Status |
|------|--------|
| Protocol hyperparams (seed 2666, Adam 1e-4, MSE abs E) | implemented |
| Bonded Class-I | mini runs exist |
| EEM + LJ + Coulomb in train script | `--energy-scope full` |
| Numerical q-parity vs espaloma solver | **open** |
| Full PhAlkEthOH-MM-small scale | **open** |
| Paper table claim | **not yet** |

## How to run a protocol attempt

1. Ensure dataset root (normalized PhAlkEthOH tree from molhub convert).
2. Create/update workspace: `python scripts/mm_param_learning/setup_class_i_protocol.py`
3. Train with the printed `train_cmd` (molexp Run lifecycle + MetricsWriter).
4. Metrics land in molexp `metrics/metrics.jsonl` (t/k/s/v); harvest writes a Finding.
"""


def _open_questions_body() -> str:
    return """# Open questions — Class-I full energy

1. **EEM parity** — Does `ElectronegativityEqualization(DiagonalHardnessKernel)`
   match espaloma `ChargeEquilibrium` q to ~1e-4 e on PhAlkEthOH frames?
2. **NB defaults** — open Coulomb vs PME for gas-phase confs; when is cell required?
3. **Encoder gap** — ChemEncoder vs SAGE×3: is bonded-only MAE comparable before NB?
4. **Scale** — mini (12 mol × 20 conf) → full 173 × 100 protocol budget on GH200.
"""


def setup(
    *,
    root: Path | None = None,
    run_id: str = PROTOCOL_RUN_ID,
    n_mols: int = 12,
    max_confs: int = 20,
    epochs: int = 50,
) -> dict:
    """Register full-energy experiment + run + knowledge (add/set CRUD)."""
    summary = MmParamLearningWorkspace(root=root).materialize()
    ws = summary["workspace"]
    if ws.has_project(PROJECT_SLUG):
        project = ws.project(PROJECT_SLUG)
    else:
        project = ws.add_project(PROJECT_SLUG)
    ws_root = Path(summary["root"])

    params = {
        "dataset_coordinate": DATASET_COORDINATE_PRIMARY,
        "data_root": str(DEFAULT_NORMALIZED_DATA_ROOT),
        "energy_scope": "full",
        "program": PROGRAM,
        "protocol": "espaloma-mm-small-notebook",
        "n_mols": n_mols,
        "max_confs": max_confs,
        "epochs": epochs,
        "lr": 1e-4,
        "batch_size": 100,
        "seed": 2666,
        "train_script": "scripts/mm_param_learning/espaloma_mm_protocol_train.py",
    }
    description = (
        "Class-I full energy mini: bonded + EEM charges + LJ + Coulomb; "
        "Espaloma-protocol split/opt; literature-gap explicit"
    )
    tags = ["mm-param-learning", "protocol", "full-energy", "class-i"]

    if project.has_experiment(PROTOCOL_EXPERIMENT_SLUG):
        exp = project.set_experiment(
            PROTOCOL_EXPERIMENT_SLUG,
            params=params,
            description=description,
            tags=tags,
        )
    else:
        exp = project.add_experiment(
            PROTOCOL_EXPERIMENT_SLUG,
            params=params,
            description=description,
            tags=tags,
        )

    if exp.has_run(run_id):
        run = exp.set_run(run_id, params=params)
    else:
        run = exp.add_run(params=params, id=run_id)

    project.add_knowledge(
        PROTOCOL_NOTE_NAME,
        kind="ProtocolNote",
        body=_protocol_note_body(),
        sources=[exp, DATASET_COORDINATE_PRIMARY, "DOI:10.1039/D2SC02739A"],
        created_by="mm-param-learning/setup_class_i_protocol",
        title="Class-I protocol architecture",
    )
    project.add_knowledge(
        "class-i-open-questions",
        kind="OpenQuestion",
        body=_open_questions_body(),
        sources=[exp, DATASET_COORDINATE_PRIMARY],
        created_by="mm-param-learning/setup_class_i_protocol",
        title="Class-I open questions",
    )

    run_dir = Path(run.path())
    exp_dir = Path(exp.path())
    data_root = DEFAULT_NORMALIZED_DATA_ROOT

    train_cmd = (
        "PYTHONPATH=src:../molhub/src "
        "python scripts/mm_param_learning/espaloma_mm_protocol_train.py "
        f"--data-root {data_root} "
        f"--energy-scope full --n-mols {n_mols} --max-confs {max_confs} "
        f"--epochs {epochs} "
        f"--run {run_dir} "
        f"--dataset-coordinate {DATASET_COORDINATE_PRIMARY}"
    )

    return {
        "workspace_root": str(ws_root),
        "project": PROJECT_SLUG,
        "experiment": PROTOCOL_EXPERIMENT_SLUG,
        "experiment_dir": str(exp_dir),
        "run_id": run.id if hasattr(run, "id") else run_id,
        "run_dir": str(run_dir),
        "dataset_coordinate": DATASET_COORDINATE_PRIMARY,
        "data_root": str(data_root),
        "knowledge_notes": [PROTOCOL_NOTE_NAME, "class-i-open-questions"],
        "train_cmd": train_cmd,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--run-id", default=PROTOCOL_RUN_ID)
    p.add_argument("--n-mols", type=int, default=12)
    p.add_argument("--max-confs", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args(argv)
    out = setup(
        root=args.root,
        run_id=args.run_id,
        n_mols=args.n_mols,
        max_confs=args.max_confs,
        epochs=args.epochs,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
