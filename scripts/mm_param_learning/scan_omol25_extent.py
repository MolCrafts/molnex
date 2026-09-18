"""Scan full OMol25 roles once; freeze Class-I :class:`MoleculeExtent` constants.

Does not train. Writes npz keyed by ``source_id`` and histogram plots of the
stored vectors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from molhub.config import config as molhub_config
from molhub.dataset.omol25 import OMol25

from molzoo.rondo import RondoForceField
from scripts.mm_param_learning.class_i_fit import ClassIPrepare, MoleculeExtent

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
        ax.set_xlabel("count / molecule")
        ax.set_ylabel("molecules")
    fig.suptitle(name)
    fig.tight_layout()
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _scan_role(role: str, *, workers: int, cutoff: float) -> MoleculeExtent:
    dataset = OMol25(role=role).load()
    print(f"loaded role={role} n={len(dataset)} source_id={dataset.source_id}", flush=True)
    extent = ClassIPrepare().scan_extent(dataset, cutoff=cutoff, workers=workers)
    path = extent.save()
    print(
        f"extent saved path={path} n={len(dataset)} cutoff={cutoff} "
        f"max_atoms={extent.get_max_atoms_per_molecule()} "
        f"max_pairs={extent.get_max_edges_per_molecule()} "
        f"max_bonds={extent.get_max_bonds_per_molecule()} "
        f"max_angles={extent.get_max_angles_per_molecule()} "
        f"max_propers={extent.get_max_propers_per_molecule()} "
        f"max_impropers={extent.get_max_impropers_per_molecule()}",
        flush=True,
    )
    return extent


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    molhub_config.home_dir = HOME
    OUT.mkdir(parents=True, exist_ok=True)
    # The same cutoff the Class-I collate mounts: the field's own list radius.
    cutoff = float(RondoForceField().neighbor_list().cutoff)
    print(f"home={HOME} workers={workers} cutoff={cutoff}", flush=True)
    summary: dict[str, dict[str, object]] = {}
    for role in ("4m", "val-neutral"):
        extent = _scan_role(role, workers=workers, cutoff=cutoff)
        plot = _plot(role, extent)
        row: dict[str, object] = {
            "n": int(extent.column("graph.natoms").size),
            "cutoff": cutoff,
            "max_atoms_per_molecule": extent.get_max_atoms_per_molecule(),
            "max_edges_per_molecule": extent.get_max_edges_per_molecule(),
            "max_bonds_per_molecule": extent.get_max_bonds_per_molecule(),
            "max_angles_per_molecule": extent.get_max_angles_per_molecule(),
            "max_propers_per_molecule": extent.get_max_propers_per_molecule(),
            "max_impropers_per_molecule": extent.get_max_impropers_per_molecule(),
            "plot": str(plot),
        }
        summary[role] = row
        print(json.dumps({role: row}, indent=2), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DONE", OUT / "summary.json", flush=True)


if __name__ == "__main__":
    main()
