"""Class-I v0 — OMol25-small molecule-level 8:2 self-split.

Science + workspace declaration only. Drive it with molexp::

    molexp run scripts/mm_param_learning/train.py --profile smoke
    molexp run scripts/mm_param_learning/train.py --profile gh200 \\
        --config scripts/mm_param_learning/molcfg.yaml \\
        -ws /path/to/workspaces/mm-param-learning
    molexp run scripts/mm_param_learning/train.py --profile smoke --override epochs=5

Do not add argparse. Hyperparameters live in ``molcfg.yaml`` and
``--override KEY=VAL``. ``-ws`` selects the workspace root.
"""

from __future__ import annotations

from datetime import datetime, timezone

import molexp as me

from scripts.mm_param_learning.constants import (
    COLLECTION_COORDINATE,
    DEFAULT_WORKSPACE_ROOT,
    PROGRAM,
    PROJECT_SLUG,
    PROTOCOL_NOTE_NAME,
    TRAIN_COORDINATE,
    V0_EXPERIMENT_SLUG,
    VAL_COORDINATE,
)
from scripts.mm_param_learning.workflows import ClassIV0Workflow

_ws = me.Workspace(DEFAULT_WORKSPACE_ROOT, name=PROJECT_SLUG)
_project = _ws.add_project(PROJECT_SLUG)
_experiment = _project.add_experiment(
    V0_EXPERIMENT_SLUG,
    params={
        "program": PROGRAM,
        "collection": COLLECTION_COORDINATE,
        "train_coordinate": TRAIN_COORDINATE,
        "val_coordinate": VAL_COORDINATE,
    },
    description="v0: OMol25-small train/val molecule-level 8:2, THEMol deferred",
    tags=["class-i", "v0", "omol25"],
)
ClassIV0Workflow().attach(_experiment, params={"seed": [2666]})

_project.add_knowledge(
    PROTOCOL_NOTE_NAME,
    kind="ProtocolNote",
    body=(
        f"# Class-I v0\n\nProgram: `{PROGRAM}`\n"
        f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"
        f"Train `{TRAIN_COORDINATE}`. Validate `{VAL_COORDINATE}` "
        "(molecule-level 8:2 self-split). THEMol is not a resolve gate. Drive with "
        "`molexp run scripts/mm_param_learning/train.py --profile smoke`.\n"
    ),
    sources=[_experiment, COLLECTION_COORDINATE, TRAIN_COORDINATE, VAL_COORDINATE],
    created_by="mm_param_learning/train.py",
    title="Class-I v0 architecture",
)
