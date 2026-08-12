"""Constants for the MM parameter-learning validation program (milestone 1).

Coordinates are provisional MolHub handles registered by mm-param-val-02/03.
Workflow stubs shape-check coordinates only — they never call ``molhub.fetch``.
"""

from __future__ import annotations

from pathlib import Path

PROGRAM: str = "mm-param-learning-baseline"
"""Program id written into every seed run's params."""

PROJECT_SLUG: str = "mm-param-learning"
"""Workspace + project slug."""

DEFAULT_WORKSPACE_ROOT: Path = Path(
    "/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/workspaces/mm-param-learning"
)
"""Operator default root. CI / tests must pass an explicit temp ``root=``."""

DEFAULT_NORMALIZED_DATA_ROOT: Path = (
    DEFAULT_WORKSPACE_ROOT / "cache" / "espaloma" / "phalkethoh-mm-small-normalized"
)
"""Local molhub-converted PhAlkEthOH tree (not a blob store — coordinate is canonical)."""

PROTOCOL_EXPERIMENT_SLUG: str = "class-i-full-energy-mini"
"""molexp experiment slug for the Class-I full-energy mini track."""

PROTOCOL_RUN_ID: str = "protocol-full-energy-mini-001"
"""Default run id under :data:`PROTOCOL_EXPERIMENT_SLUG` (setup + train)."""

ESPALOMA_SCALE_EXPERIMENT_SLUG: str = "class-i-full-energy-espaloma-scale"
"""Formal run: full PhAlkEthOH-MM-small (173 mol × 100 conf) + dim=128."""

ESPALOMA_SCALE_RUN_ID: str = "protocol-full-energy-espaloma-scale-001"

# Experiment slug → primary MolHub dataset coordinate (or empty when N/A).
MOLHUB_COORDINATES: dict[str, str] = {
    "potential-parity": "dataset:espaloma/phalkethoh-mm-small@1",
    "zinc-typing-recovery": "dataset:espaloma/zinc-typing@1",
    "phalkethoh-mm-energy": "dataset:espaloma/phalkethoh-mm-small@1",
    "latent-analysis": "dataset:espaloma/zinc-typing@1",
    # Protocol reproduction track (molexp-managed)
    "espaloma-mm-protocol-mini": "dataset:espaloma/phalkethoh-mm-small@1",
    PROTOCOL_EXPERIMENT_SLUG: "dataset:espaloma/phalkethoh-mm-small@1",
}

EXPERIMENT_SLUGS: tuple[str, ...] = tuple(MOLHUB_COORDINATES.keys())

# Validation narrative stages 1–6 (Knowledge Note inventory).
VALIDATION_STAGES: tuple[tuple[str, str], ...] = (
    ("1", "Dataset inventory & MolHub coordinates"),
    ("2", "B0 potential IR / Class-I kernel parity"),
    ("3", "A GAFF typing recovery (zinc-typing)"),
    ("4", "B1/B2 molecule-centered PhAlkEthOH MM energy"),
    ("5", "D latent embedding purity analysis"),
    ("6", "Synthesis & milestone gate"),
    ("7", "Protocol full Class-I energy (bonded+EEM+LJ+Coulomb) mini"),
)

KNOWLEDGE_NOTE_NAME: str = "mm-param-validation-stages"
"""Idempotent Knowledge Note slug under the workspace bundle root."""

PROTOCOL_NOTE_NAME: str = "class-i-protocol-architecture"
"""Architecture + literature-gap note for the Class-I reproduction track."""

DATASET_COORDINATE_PRIMARY: str = "dataset:espaloma/phalkethoh-mm-small@1"
"""Primary MolHub dataset for energy-protocol experiments."""
