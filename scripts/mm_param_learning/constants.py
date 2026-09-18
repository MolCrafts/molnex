"""Constants for the Class-I molexp program.

v0 train/val coordinates are the active roles of
``dataset:molcrafts/class1-universal-2026@1``. Scripts read them from
:class:`~molhub.dataset.Class1Universal2026` at run time; the literals
below are the expected pins for workspace params and tests.
"""

from __future__ import annotations

from pathlib import Path

PROGRAM: str = "class-i-v0"
"""Program id written into every seed run's params."""

PROJECT_SLUG: str = "mm-param-learning"
"""Workspace + project slug."""

DEFAULT_WORKSPACE_ROOT: Path = Path(
    "/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/workspaces/mm-param-learning"
)
"""Operator default root. CI / tests must pass an explicit temp ``root=``."""

COLLECTION_COORDINATE: str = "dataset:molcrafts/class1-universal-2026@1"
"""MolHub collection whose active roles are the v0 resolve set."""

TRAIN_COORDINATE: str = "dataset:molcrafts/omol25-small@1"
"""Active train role — OMol25-small (DFT E/F)."""

VAL_COORDINATE: str = TRAIN_COORDINATE
"""v0 val is a molecule-level 8:2 self-split of the same OMol25-small source."""

PHALKETHOH_COORDINATE: str = "dataset:espaloma/phalkethoh-mm-small@1"
"""Legacy Espaloma-MM coordinate. Not the v0 val pin."""

DEFAULT_VAL_DATA_ROOT: Path = (
    DEFAULT_WORKSPACE_ROOT / "cache" / "espaloma" / "phalkethoh-mm-small-normalized"
)
"""Local PhAlkEthOH tree for legacy Espaloma-MM scripts only."""

V0_EXPERIMENT_SLUG: str = "universal-class-i-v0-calibration-gh200"
"""Authoritative v0 calibration experiment."""

V0_RUN_ID: str = "class-i-v0-001"

V0_PROD_EXPERIMENT_SLUG: str = "universal-class-i-v0-prod-day1-gh200"
"""Bounded one-day / three-size campaign. Not unbounded production."""

V0_PROD_RUN_IDS: dict[str, str] = {
    "small": "f2dd9ed5",
    "medium": "90e71a0d",
    "large": "ef38fb1c",
}
"""Pending molexp run ids for the prod-day1 campaign, one per model size."""

V0_PROD_DAY2_EXPERIMENT_SLUG: str = "universal-class-i-v0-prod-day2-gh200"
"""Continue from prod-day1 ``last.pt``. Same science pins, new molexp experiment."""

V0_PROD_DAY1_HOST_RUNS: dict[str, str] = {
    "small": "prod-day1-small-20260823T193943Z",
    "medium": "prod-day1-medium-20260823T193943Z",
    "large": "prod-day1-large-20260824T013930Z",
}
"""Host training trees under molnex ``runs/`` that hold day-1 ``last.pt``."""

PROTOCOL_EXPERIMENT_SLUG: str = V0_EXPERIMENT_SLUG
"""Alias kept so older setup callers resolve to v0."""

PROTOCOL_RUN_ID: str = V0_RUN_ID

ESPALOMA_SCALE_EXPERIMENT_SLUG: str = "class-i-full-energy-espaloma-scale"
ESPALOMA_SCALE_RUN_ID: str = "protocol-full-energy-espaloma-scale-001"

# Experiment slug → primary MolHub dataset coordinate (legacy tracks keep theirs).
MOLHUB_COORDINATES: dict[str, str] = {
    "potential-parity": PHALKETHOH_COORDINATE,
    "zinc-typing-recovery": "dataset:espaloma/zinc-typing@1",
    "phalkethoh-mm-energy": PHALKETHOH_COORDINATE,
    "latent-analysis": "dataset:espaloma/zinc-typing@1",
    "espaloma-mm-protocol-mini": PHALKETHOH_COORDINATE,
    "class-i-full-energy-mini": PHALKETHOH_COORDINATE,
    V0_EXPERIMENT_SLUG: TRAIN_COORDINATE,
}

EXPERIMENT_SLUGS: tuple[str, ...] = tuple(MOLHUB_COORDINATES.keys())

VALIDATION_STAGES: tuple[tuple[str, str], ...] = (
    ("1", "Dataset inventory & MolHub coordinates"),
    ("2", "B0 potential IR / Class-I kernel parity"),
    ("3", "A GAFF typing recovery (zinc-typing)"),
    ("4", "B1/B2 molecule-centered PhAlkEthOH MM energy"),
    ("5", "D latent embedding purity analysis"),
    ("6", "Synthesis & milestone gate"),
    ("7", "v0 protocol: OMol25 train / OMol25 molecule-level 8:2 val"),
)

KNOWLEDGE_NOTE_NAME: str = "mm-param-validation-stages"
PROTOCOL_NOTE_NAME: str = "class-i-protocol-architecture"

DATASET_COORDINATE_PRIMARY: str = TRAIN_COORDINATE
"""Primary train coordinate for energy-protocol experiments."""

DEFAULT_NORMALIZED_DATA_ROOT: Path = DEFAULT_VAL_DATA_ROOT
"""Back-compat alias of :data:`DEFAULT_VAL_DATA_ROOT`."""
