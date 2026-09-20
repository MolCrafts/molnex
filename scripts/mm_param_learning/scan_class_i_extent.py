"""Scan the Class-I training shard once; freeze :class:`MoleculeExtent` constants.

Does not train. Writes one npz keyed by ``source_id`` plus histograms of the
stored vectors.

**One scan serves both splits.** The extent is keyed by ``(source_id, cutoff)``
and :meth:`MoleculeExtent.for_dataset` slices a parent scan for a
:class:`~molhub.dataset.protocol.SubsetDataset` (``.source`` + ``.indices``), so
scanning the whole shard covers the train and val views
:func:`~scripts.mm_param_learning.class_i_fit.load_themol_torsion_split` cuts
from it. Do not scan the two views separately.

This replaced ``scan_omol25_extent.py`` when the Class-I dataset became THEMol
torsion-scan (2026-09-20). OMol25 is not scanned here: nothing trains on it, and
a frozen extent for a dataset no launcher opens is the kind of artifact that
later gets mistaken for the current one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from molhub.config import config as molhub_config
from molhub.dataset.themol import THEMolTorsionScan

from molzoo.rondo import RondoForceField
from scripts.mm_param_learning.class_i_fit import ClassIPrepare, MoleculeExtent
from scripts.mm_param_learning.configs import PROD_DAY

MOLNEX = Path(__file__).resolve().parents[2]
HOME = MOLNEX.parent / "workspaces" / "molhub-home"
OUT = MOLNEX / "runs" / "valence-distrib"


def _plot(name: str, extent: MoleculeExtent) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    fig.text(0.99, 0.01, f"cutoff {extent.cutoff:g} Å", ha="right", va="bottom", fontsize=8)
    getters = {
        "graph.natoms": extent.get_max_atoms_per_molecule,
        "pairs": extent.get_max_edges_per_molecule,
        "bonds": extent.get_max_bonds_per_molecule,
        "angles": extent.get_max_angles_per_molecule,
        "propers": extent.get_max_propers_per_molecule,
        "impropers": extent.get_max_impropers_per_molecule,
    }
    for ax, field in zip(axes, MoleculeExtent.FIELDS):
        raw = extent.column(field)
        hi = int(raw.max()) if raw.size else 1
        ax.hist(raw, bins=min(80, max(hi, 1)), color="#3b6ea5", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{field}\nmax={getters[field]()}  n={int(raw.size)}")
        # A row is one whole scan, so these are per *row*, not per conformer.
        ax.set_xlabel("count / row")
        ax.set_ylabel("rows")
    fig.suptitle(name)
    fig.tight_layout()
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    molhub_config.home_dir = HOME
    OUT.mkdir(parents=True, exist_ok=True)
    # The same cutoff the Class-I collate mounts: the field's own list radius.
    cutoff = float(RondoForceField().neighbor_list().cutoff)
    role = str(PROD_DAY["shard_role"])
    print(f"home={HOME} workers={workers} cutoff={cutoff} role={role}", flush=True)

    dataset = THEMolTorsionScan().load(role)
    print(f"loaded role={role} n={len(dataset)} source_id={dataset.source_id}", flush=True)
    extent = ClassIPrepare().scan_extent(dataset, cutoff=cutoff, workers=workers)
    path = extent.save()
    row: dict[str, object] = {
        "n": int(extent.column("graph.natoms").size),
        "cutoff": cutoff,
        "source_id": dataset.source_id,
        "npz": str(path),
        "max_atoms_per_row": extent.get_max_atoms_per_molecule(),
        "max_edges_per_row": extent.get_max_edges_per_molecule(),
        "max_bonds_per_row": extent.get_max_bonds_per_molecule(),
        "max_angles_per_row": extent.get_max_angles_per_molecule(),
        "max_propers_per_row": extent.get_max_propers_per_molecule(),
        "max_impropers_per_row": extent.get_max_impropers_per_molecule(),
    }
    row["plot"] = str(_plot(role, extent))
    print(json.dumps({role: row}, indent=2), flush=True)
    (OUT / "summary.json").write_text(json.dumps({role: row}, indent=2) + "\n")
    print("DONE", OUT / "summary.json", flush=True)


if __name__ == "__main__":
    main()
