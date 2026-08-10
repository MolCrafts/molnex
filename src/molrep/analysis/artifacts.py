"""Write latent analysis metrics and point artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from molrep.analysis.latent_store import AtomLatentTable
from molrep.analysis.projection import LatentPCA2D
from molrep.analysis.type_purity import NearestNeighbourTypePurity

__all__ = ["LatentAnalysisArtifacts"]


class LatentAnalysisArtifacts:
    """Emit purity metrics + latent_points.jsonl + type_purity.txt.

    Optionally uses :class:`~molix.io.metrics.MetricsWriter` when available.
    """

    def __init__(self, out_dir: str | Path, *, k: int = 1) -> None:
        self.out_dir = Path(out_dir)
        self.k = k

    def write(self, table: AtomLatentTable) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        purity = NearestNeighbourTypePurity(k=self.k).score(table)
        coords = LatentPCA2D().project(table)

        metrics = {
            "nn_type_purity_mean": purity.mean_purity,
            "nn_type_purity_k": purity.k,
            "n_atoms": table.n_atoms,
            "n_labeled": purity.n_labeled,
            "n_scored": purity.n_scored,
        }

        # points jsonl
        points_path = self.out_dir / "latent_points.jsonl"
        with points_path.open("w") as fh:
            for i in range(table.n_atoms):
                rec = {
                    "i": i,
                    "molecule_id": table.molecule_id[i],
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                }
                if table.ref_atom_type is not None:
                    rec["ref_atom_type"] = int(table.ref_atom_type[i])
                fh.write(json.dumps(rec) + "\n")

        purity_path = self.out_dir / "type_purity.txt"
        purity_path.write_text(
            f"mean_purity={purity.mean_purity}\nk={purity.k}\n"
            f"n_scored={purity.n_scored}\nn_labeled={purity.n_labeled}\n"
        )

        # Optional MetricsWriter (record root); always also dump a plain jsonl line.
        try:
            from molix.io.metrics import MetricsWriter

            mw = MetricsWriter(self.out_dir)
            for key, value in metrics.items():
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    mw.scalar(key, float(value))
        except Exception:
            pass
        (self.out_dir / "metrics_summary.jsonl").write_text(json.dumps(metrics) + "\n")

        return metrics
