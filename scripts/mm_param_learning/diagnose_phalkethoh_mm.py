#!/usr/bin/env python3
"""Diagnose real PhAlkEthOH MM-small payload (no model train).

Loads :class:`molhub.dataset.PhalkethohMMDataset` from a **normalized** tree
and reports molecule-level split sizes, conformer counts, and the
molecule-mean-centered energy scale (kcal/mol) on train/val/test.

This is the first real-data gate for Validation B2: if this fails, training
cannot start.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--record-root", type=Path, default=None)
    args = p.parse_args()

    from molhub.dataset import PhalkethohMMDataset
    from molhub.dataset.meta import Targets

    report: dict = {"data_root": str(args.data_root), "splits": {}}
    for split in ("train", "val", "test"):
        ds = PhalkethohMMDataset(args.data_root, download=False, split=split)
        by: dict[str, list[float]] = defaultdict(list)
        for i in range(len(ds)):
            fr = ds[i]
            t = Targets(fr)
            by[str(t["molecule_id"])].append(float(t["mm_energy"]))
        n_mol = len(by)
        n_fr = len(ds)
        # per-molecule centering then pool RMSE of zeros vs values is std
        centered = []
        for mid, es in by.items():
            arr = np.asarray(es, dtype=np.float64)
            centered.append(arr - arr.mean())
        cat = np.concatenate(centered) if centered else np.zeros(0)
        report["splits"][split] = {
            "n_frames": n_fr,
            "n_molecules": n_mol,
            "confs_per_mol_mean": n_fr / n_mol if n_mol else 0,
            "energy_kcal_mean": float(np.mean([e for es in by.values() for e in es])) if n_fr else None,
            "energy_kcal_std": float(np.std([e for es in by.values() for e in es])) if n_fr else None,
            "centered_energy_std_kcal": float(np.std(cat)) if cat.size else None,
            "source_id": ds.source_id,
        }
    print(json.dumps(report, indent=2))
    if args.record_root is not None:
        args.record_root.mkdir(parents=True, exist_ok=True)
        (args.record_root / "artifacts").mkdir(exist_ok=True)
        (args.record_root / "artifacts" / "phalkethoh_mm_diagnose.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        metrics = args.record_root / "metrics"
        metrics.mkdir(exist_ok=True)
        # one-line jsonl for molplot
        row = {"step": 0}
        for split, d in report["splits"].items():
            row[f"{split}_n_frames"] = d["n_frames"]
            row[f"{split}_n_molecules"] = d["n_molecules"]
            row[f"{split}_centered_energy_std_kcal"] = d["centered_energy_std_kcal"]
        with (metrics / "metrics.jsonl").open("w") as f:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
