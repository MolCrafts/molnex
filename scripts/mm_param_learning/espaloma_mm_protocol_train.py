#!/usr/bin/env python3
"""Espaloma-protocol MM fitting on a small PhAlkEthOH subset (our molexp project).

Matches Espaloma's published MM-small validation protocol as closely as our stack allows:

* molecule-level shuffle/split seed **2666**, ratios **8:1:1** (80/10/10)
* loss = graph-level **MSE(E_pred, E_ref)** (absolute MM energy, not QM-centering)
* optimizer **Adam(lr=1e-4)**
* train loader **batch_size=100** conformers
* report L1 (MAE) in kcal/mol (Espaloma notebook multiplies Hartree metrics by 627.5;
  our labels are already kcal/mol after conversion)

Architecture (Espaloma ↔ ours):

| Espaloma | Ours |
|----------|------|
| SAGEConv ×3, 128, ReLU | ChemEncoder atom/bond/angle/proper dim=128 |
| Janossy → bond/angle log-coeff, torsion k×6 | Bond/Angle/ProperParamHead (n_terms=6) |
| GeometryInGraph + EnergyInGraph | ForceField.to_potentials() Class-I kernels |
| ChargeEquilibrium + NB | ``--energy-scope full``: EEM + LJ + Coulomb (algorithm-named) |

Small system (default): the **12 smallest molecules** (by atom count) in the
normalized PhAlkEthOH tree, **all 100 confs** (or ``--max-confs``).

Runs under the molexp project ``mm-param-learning`` experiment
``espaloma-mm-protocol-mini`` when ``--record-root`` points at a run dir.

Example::

    export PY=/path/to/python  # torch+molpy env
    export PYTHONPATH=src:../molhub/src
    $PY scripts/mm_param_learning/espaloma_mm_protocol_train.py \\
      --data-root .../phalkethoh-mm-small-normalized \\
      --epochs 200 --device cpu \\
      --record-root .../experiments/espaloma-mm-protocol-mini/runs/run-001
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from molhub.dataset import PhalkethohMMDataset
from tensordict import TensorDict

from molix.data import MoleculeCollator, TargetSchema
from molpot.charge import (
    AtomicChargeModel,
    ChiEtaHead,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
)
from molpot.composition.heads import LJParameterHead
from molpot.composition.mm_heads import (
    AngleParamHead,
    BondParamHead,
    ProperTorsionParamHead,
)
from molpot.forcefield import ClassicalIMM, ClassIForceField
from molrep.chem.encoder import ChemEncoder

# Espaloma notebook protocol pins
ESPALOMA_SPLIT_SEED = 2666
ESPALOMA_SPLIT_RATIOS = (8, 1, 1)  # train:val:test parts
ESPALOMA_ADAM_LR = 1e-4
ESPALOMA_BATCH_SIZE = 100
ESPALOMA_HIDDEN = 128
ESPALOMA_N_TORSION_TERMS = 6

# Energy already converted to kcal/mol in the normalized MolHub tree
# (Hartree * 627.5094740631). Espaloma notebook reports L1 * 627.5.


def enumerate_angles(
    atomi: np.ndarray, atomj: np.ndarray, n_atoms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """i-j-k angles from undirected bonds (j central)."""
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for i, j in zip(atomi.tolist(), atomj.tolist(), strict=True):
        adj[i].append(j)
        adj[j].append(i)
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []
    for j in range(n_atoms):
        nbrs = adj[j]
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                ii.append(nbrs[a])
                jj.append(j)
                kk.append(nbrs[b])
    if not ii:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    return (
        np.asarray(ii, dtype=np.int64),
        np.asarray(jj, dtype=np.int64),
        np.asarray(kk, dtype=np.int64),
    )


def enumerate_propers(
    atomi: np.ndarray, atomj: np.ndarray, n_atoms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """i-j-k-l propers with central bond j-k."""
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    edges = list(zip(atomi.tolist(), atomj.tolist(), strict=True))
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []
    ll: list[int] = []
    seen: set[tuple[int, int]] = set()
    for a, b in edges:
        for j, k in ((a, b), (b, a)):
            if (j, k) in seen:
                continue
            seen.add((j, k))
            seen.add((k, j))
            for i in adj[j]:
                if i == k:
                    continue
                for l in adj[k]:
                    if l == j or l == i:
                        continue
                    ii.append(i)
                    jj.append(j)
                    kk.append(k)
                    ll.append(l)
    if not ii:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, z
    return (
        np.asarray(ii, dtype=np.int64),
        np.asarray(jj, dtype=np.int64),
        np.asarray(kk, dtype=np.int64),
        np.asarray(ll, dtype=np.int64),
    )


def frame_to_batch(fr, device: torch.device) -> tuple[TensorDict, torch.Tensor, str]:
    """One conformer Frame → TensorDict batch + energy kcal/mol + molecule_id."""
    atoms = fr["atoms"]
    bonds = fr["bonds"]
    z = torch.as_tensor(atoms["number"], dtype=torch.long, device=device)
    pos = torch.stack(
        [torch.as_tensor(atoms[c], dtype=torch.float64, device=device) for c in ("x", "y", "z")],
        dim=-1,
    )
    atomi = np.asarray(bonds["atomi"], dtype=np.int64)
    atomj = np.asarray(bonds["atomj"], dtype=np.int64)
    n = int(z.shape[0])
    ai_t = torch.as_tensor(atomi, dtype=torch.long, device=device)
    aj_t = torch.as_tensor(atomj, dtype=torch.long, device=device)
    ang_i, ang_j, ang_k = enumerate_angles(atomi, atomj, n)
    pr_i, pr_j, pr_k, pr_l = enumerate_propers(atomi, atomj, n)

    ns: dict = {
        "atoms": TensorDict(
            {
                "Z": z,
                "pos": pos,
                "batch": torch.zeros(n, dtype=torch.long, device=device),
            },
            batch_size=[n],
        ),
        "bonds": TensorDict({"atomi": ai_t, "atomj": aj_t}, batch_size=[ai_t.shape[0]]),
    }
    if ang_i.size:
        ns["angles"] = TensorDict(
            {
                "atomi": torch.as_tensor(ang_i, dtype=torch.long, device=device),
                "atomj": torch.as_tensor(ang_j, dtype=torch.long, device=device),
                "atomk": torch.as_tensor(ang_k, dtype=torch.long, device=device),
            },
            batch_size=[ang_i.shape[0]],
        )
    if pr_i.size:
        ns["propers"] = TensorDict(
            {
                "atomi": torch.as_tensor(pr_i, dtype=torch.long, device=device),
                "atomj": torch.as_tensor(pr_j, dtype=torch.long, device=device),
                "atomk": torch.as_tensor(pr_k, dtype=torch.long, device=device),
                "atoml": torch.as_tensor(pr_l, dtype=torch.long, device=device),
            },
            batch_size=[pr_i.shape[0]],
        )
    batch = TensorDict(ns, batch_size=[])
    energy = torch.tensor(float(fr.meta["mm_energy"]), dtype=torch.float64, device=device)
    mid = str(fr.meta["molecule_id"])
    return batch, energy, mid


def frame_to_sample(fr) -> dict:
    """Convert one MolHub frame to the pre-collate molecular schema."""
    atoms = fr["atoms"]
    bonds = fr["bonds"]
    z = torch.as_tensor(np.asarray(atoms["number"]), dtype=torch.long)
    pos = torch.stack(
        [torch.as_tensor(np.asarray(atoms[c]), dtype=torch.float64) for c in ("x", "y", "z")],
        dim=-1,
    )
    atomi = np.asarray(bonds["atomi"], dtype=np.int64)
    atomj = np.asarray(bonds["atomj"], dtype=np.int64)
    ang_i, ang_j, ang_k = enumerate_angles(atomi, atomj, int(z.shape[0]))
    pr_i, pr_j, pr_k, pr_l = enumerate_propers(atomi, atomj, int(z.shape[0]))
    sample = {
        "Z": z,
        "pos": pos,
        "bond_index": torch.as_tensor(np.stack([atomi, atomj]), dtype=torch.long),
        "targets": {"mm_energy": float(fr.meta["mm_energy"])},
    }
    if ang_i.size:
        sample["angles"] = {
            "atomi": torch.as_tensor(ang_i),
            "atomj": torch.as_tensor(ang_j),
            "atomk": torch.as_tensor(ang_k),
        }
    if pr_i.size:
        sample["propers"] = {
            "atomi": torch.as_tensor(pr_i),
            "atomj": torch.as_tensor(pr_j),
            "atomk": torch.as_tensor(pr_k),
            "atoml": torch.as_tensor(pr_l),
        }
    return sample


_MM_TARGET_SCHEMA = TargetSchema(graph_level=frozenset({"mm_energy"}))
_MM_COLLATOR = MoleculeCollator(_MM_TARGET_SCHEMA)


def frames_to_batch(frames: list, device: torch.device) -> tuple[TensorDict, torch.Tensor]:
    """Pack heterogeneous conformers into one graph-aware GPU batch."""
    batch = _MM_COLLATOR([frame_to_sample(frame) for frame in frames]).to(device)
    return batch, batch["graphs", "mm_energy"]


def molecule_split(mol_ids: list[str], seed: int = ESPALOMA_SPLIT_SEED) -> dict[str, list[str]]:
    """Espaloma notebook: shuffle(seed) then split([8,1,1])."""
    ids = list(mol_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b, c = ESPALOMA_SPLIT_RATIOS
    total = a + b + c
    n_tr = max(1, int(round(n * a / total))) if n >= 3 else max(1, n - 2)
    n_vl = max(1, int(round(n * b / total))) if n >= 3 else 1
    if n_tr + n_vl >= n:
        n_tr = max(1, n - 2)
        n_vl = 1
    n_te = n - n_tr - n_vl
    if n_te < 1:
        n_te = 1
        n_tr = max(1, n - n_vl - n_te)
    return {
        "train": ids[:n_tr],
        "val": ids[n_tr : n_tr + n_vl],
        "test": ids[n_tr + n_vl :],
        "seed": seed,
        "ratios": [a, b, c],
        "scheme": "espaloma-original",
    }


def load_all_frames(
    data_root: Path, *, max_confs: int | None = None
) -> tuple[dict[str, list], dict[str, int]]:
    """Load every molecule from train/val/test into memory (Espaloma full scale).

    Returns:
        frames_by_mol: molecule_id → list of Frames (capped by *max_confs*).
        n_atoms: molecule_id → atom count.
    """
    by: dict[str, list] = defaultdict(list)
    n_atoms: dict[str, int] = {}
    for split in ("train", "val", "test"):
        ds = PhalkethohMMDataset(data_root, download=False, split=split)
        for i in range(len(ds)):
            fr = ds[i]
            mid = str(fr.meta["molecule_id"])
            n_atoms[mid] = int(len(fr["atoms"]["number"]))
            if max_confs is None or len(by[mid]) < max_confs:
                by[mid].append(fr)
    return dict(by), n_atoms


def load_official_split(data_root: Path) -> dict[str, list[str]]:
    """Molecule-level split from molhub-normalized ``splits.json`` (seed 2666)."""
    path = Path(data_root) / "splits.json"
    raw = json.loads(path.read_text())
    return {
        "train": [str(x) for x in raw["train"]],
        "val": [str(x) for x in raw["val"]],
        "test": [str(x) for x in raw["test"]],
        "seed": int(raw.get("seed", ESPALOMA_SPLIT_SEED)),
        "ratios": list(raw.get("ratios", list(ESPALOMA_SPLIT_RATIOS))),
        "scheme": str(raw.get("scheme", "espaloma-original")),
    }


def pick_molecules(
    data_root: Path,
    n_mols: int = 0,
    max_confs: int | None = None,
    *,
    selection: str = "all",
) -> dict[str, list]:
    """Load corpus and optionally subset molecules.

    Args:
        n_mols: ``0`` or negative → all molecules. Positive → subset size.
        selection: ``all`` keeps arbitrary order; ``smallest`` ranks by atom count
            (legacy mini protocol).
    """
    by, n_atoms = load_all_frames(data_root, max_confs=max_confs)
    mids = list(by.keys())
    if n_mols is not None and n_mols > 0 and n_mols < len(mids):
        if selection == "smallest":
            ranked = sorted(n_atoms.items(), key=lambda kv: (kv[1], kv[0]))
            mids = [mid for mid, _ in ranked[:n_mols]]
        else:
            mids = sorted(mids)[:n_mols]
    return {mid: by[mid] for mid in mids}


def pick_small_molecules(
    data_root: Path, n_mols: int = 12, max_confs: int | None = None
) -> dict[str, list]:
    """Legacy mini helper: *n_mols* smallest molecules by atom count."""
    return pick_molecules(data_root, n_mols=n_mols, max_confs=max_confs, selection="smallest")


def build_espaloma_like_model(
    *,
    energy_scope: str = "bonded",
    hidden: int = ESPALOMA_HIDDEN,
    n_blocks: int = 5,
) -> ClassicalIMM:
    """ChemEncoder + Class-I heads.

    Args:
        energy_scope:
            * ``bonded`` — bond/angle/proper only (legacy mini baseline)
            * ``full`` — + EEM charges + LJ + Coulomb (open Class-I NB)
    """
    if energy_scope not in ("bonded", "full"):
        raise ValueError(f"energy_scope must be bonded|full, got {energy_scope!r}")

    encoder = ChemEncoder(
        atom_dim=hidden,
        bond_dim=hidden,
        angle_dim=hidden,
        proper_dim=hidden,
        improper_dim=hidden,
        hidden_dim=hidden,
        n_blocks=n_blocks,
        local_attention=False,
    )
    ff = ClassIForceField(name="class-i-learned")
    if energy_scope == "full":
        # Cutoff lives on the Potential (via style options), not on Potentials.
        ff.def_pairstyle("lj/cut", cutoff=10.0)
        ff.def_pairstyle("coulomb")

    atom_head = None
    charge_model = None
    if energy_scope == "full":
        atom_head = LJParameterHead(feature_dim=hidden, hidden_dim=hidden)
        charge_model = AtomicChargeModel(
            ChiEtaHead(feature_dim=hidden, hidden_dim=hidden),
            ElectronegativityEqualization(DiagonalHardnessKernel()),
        )

    return ClassicalIMM(
        encoder,
        forcefield=ff,
        bond_head=BondParamHead(feature_dim=hidden, hidden_dim=hidden),
        angle_head=AngleParamHead(feature_dim=hidden, hidden_dim=hidden),
        proper_head=ProperTorsionParamHead(
            feature_dim=hidden,
            hidden_dim=hidden,
            n_terms=ESPALOMA_N_TORSION_TERMS,
            periodicity=tuple(range(1, ESPALOMA_N_TORSION_TERMS + 1)),
        ),
        atom_head=atom_head,
        charge_model=charge_model,
    ).double()


def mse_energy(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Espaloma GraphMetric MSE on graph energies."""
    return torch.mean((pred - ref) ** 2)


def mae_energy(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - ref)).detach())


@torch.no_grad()
def eval_split(
    model: ClassicalIMM,
    frames_by_mol: dict[str, list],
    mol_ids: list[str],
    device: torch.device,
    batch_size: int = ESPALOMA_BATCH_SIZE,
) -> dict[str, float]:
    """Topology once per molecule; evaluate all confs; report MAE/RMSE kcal/mol."""
    preds: list[torch.Tensor] = []
    refs: list[torch.Tensor] = []
    frames = [frame for mid in mol_ids for frame in frames_by_mol[mid]]
    for start in range(0, len(frames), batch_size):
        batch, ref = frames_to_batch(frames[start : start + batch_size], device)
        energy = model(batch).to_potentials().calc_energy(batch)
        preds.append(energy.reshape(-1))
        refs.append(ref.reshape(-1))
    pred = torch.cat(preds)
    ref = torch.cat(refs)
    return {
        "mae_kcal": mae_energy(pred, ref),
        "rmse_kcal": float(torch.sqrt(torch.mean((pred - ref) ** 2)).detach()),
        "n": int(pred.numel()),
    }


def _log_metrics(metrics_writer: object | None, row: dict) -> None:
    """Append scalars in molexp MetricsWriter format (t/k/s/v), if available."""
    if metrics_writer is None:
        return
    step = row.get("epoch")
    for key in (
        "train_mse",
        "train_mae_kcal",
        "train_rmse_kcal",
        "val_mae_kcal",
        "val_rmse_kcal",
        "elapsed_s",
        "lr",
    ):
        if key not in row:
            continue
        val = row[key]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            metrics_writer.scalar(key, float(val), step=step)  # type: ignore[attr-defined]


def train(
    model: ClassicalIMM,
    frames_by_mol: dict[str, list],
    split: dict[str, list[str]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    record_root: Path | None,
    eval_every: int = 5,
    metrics_writer: object | None = None,
    early_stop_patience: int = 0,
) -> list[dict]:
    """Espaloma-style Adam + MSE energy training.

    Args:
        early_stop_patience: Stop if val MAE does not improve for this many
            *evaluated* epochs. ``0`` disables (Espaloma notebook trains to a
            fixed budget and inspects offline; we offer patience for formal runs).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_ids = split["train"]
    val_ids = split["val"]

    # Flatten train conformers for Espaloma-style shuffled minibatches of graphs
    train_pool: list = []
    for mid in train_ids:
        train_pool.extend(frames_by_mol[mid])

    history: list[dict] = []
    best_val = float("inf")
    best_state: dict | None = None
    stale_evals = 0

    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_pool)))
        random.shuffle(order)
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            # One packed heterogeneous graph batch → one model/energy launch.
            batch, reference = frames_to_batch([train_pool[j] for j in idx], device)
            energy = model(batch).to_potentials().calc_energy(batch).reshape(-1)
            loss = mse_energy(energy, reference.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1

        do_eval = (epoch % eval_every == 0) or (epoch == epochs - 1)
        if do_eval:
            tr = eval_split(model, frames_by_mol, train_ids, device, batch_size=batch_size)
            va = eval_split(model, frames_by_mol, val_ids, device, batch_size=batch_size)
            row = {
                "epoch": epoch,
                "train_mse": epoch_loss / max(n_batches, 1),
                "train_mae_kcal": tr["mae_kcal"],
                "train_rmse_kcal": tr["rmse_kcal"],
                "val_mae_kcal": va["mae_kcal"],
                "val_rmse_kcal": va["rmse_kcal"],
                "elapsed_s": time.time() - t0,
                "lr": lr,
                "batch_size": batch_size,
            }
            # Espaloma: early stop on validation metric
            if va["mae_kcal"] < best_val:
                best_val = va["mae_kcal"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale_evals = 0
            else:
                stale_evals += 1
        else:
            row = {
                "epoch": epoch,
                "train_mse": epoch_loss / max(n_batches, 1),
                "elapsed_s": time.time() - t0,
                "lr": lr,
                "batch_size": batch_size,
            }
        history.append(row)
        print(json.dumps(row), flush=True)
        _log_metrics(metrics_writer, row)

        # Legacy free-form row dump (debug). Prefer MetricsWriter when present.
        if record_root is not None and metrics_writer is None:
            metrics_dir = record_root / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            with (metrics_dir / "metrics.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")

        if (
            early_stop_patience > 0
            and do_eval
            and stale_evals >= early_stop_patience
            and best_state is not None
        ):
            print(
                json.dumps(
                    {
                        "early_stop": True,
                        "epoch": epoch,
                        "patience": early_stop_patience,
                        "best_val_mae_kcal": best_val,
                    }
                ),
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _resolve_molexp_run(
    *,
    run_dir: Path | None,
    workspace: Path | None,
    project: str,
    experiment: str,
    run_id: str | None,
    record_root: Path | None,
):
    """Open a molexp Run from ``--run`` dir, workspace ids, or record_root.

    Returns ``(run, record_root)`` or ``(None, record_root)`` when molexp is
    unavailable / not requested.
    """
    try:
        from molexp.workspace import Workspace
        from molexp.workspace.run import Run
    except ImportError:
        return None, record_root

    if run_dir is not None:
        try:
            run = Run.load(Path(run_dir).expanduser().resolve())
            return run, Path(run.path())
        except Exception:
            return None, Path(run_dir)

    if workspace is not None and run_id is not None:
        ws = Workspace.load(Path(workspace).expanduser().resolve())
        run = ws.project(project).experiment(experiment).run(run_id)
        return run, Path(run.path())

    if record_root is not None:
        try:
            run = Run.load(Path(record_root).expanduser().resolve())
            return run, Path(run.path())
        except Exception:
            return None, Path(record_root)

    return None, record_root


def main(argv: list[str] | None = None) -> int:
    from scripts.mm_param_learning.constants import (
        DATASET_COORDINATE_PRIMARY,
        DEFAULT_NORMALIZED_DATA_ROOT,
        DEFAULT_WORKSPACE_ROOT,
        PROJECT_SLUG,
        PROTOCOL_EXPERIMENT_SLUG,
        PROTOCOL_RUN_ID,
    )

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_NORMALIZED_DATA_ROOT,
        help="Normalized PhAlkEthOH tree (molhub convert output)",
    )
    p.add_argument(
        "--n-mols",
        type=int,
        default=0,
        help="Molecule subset size; 0 = all molecules (Espaloma MM-small full = 173)",
    )
    p.add_argument(
        "--max-confs",
        type=int,
        default=100,
        help="Cap confs/mol (Espaloma MM-small uses 100)",
    )
    p.add_argument(
        "--selection",
        choices=("all", "smallest"),
        default="all",
        help="How to choose --n-mols subset (ignored when n_mols=0)",
    )
    p.add_argument(
        "--use-official-splits",
        action="store_true",
        default=True,
        help="Use data_root/splits.json (Espaloma seed 2666); default on",
    )
    p.add_argument(
        "--no-official-splits",
        action="store_false",
        dest="use_official_splits",
        help="Re-shuffle with --seed instead of splits.json",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Espaloma notebook budget is 10000; formal default 1000",
    )
    p.add_argument("--batch-size", type=int, default=ESPALOMA_BATCH_SIZE)
    p.add_argument("--hidden", type=int, default=ESPALOMA_HIDDEN)
    p.add_argument("--n-blocks", type=int, choices=(4, 5, 6), default=5)
    p.add_argument("--lr", type=float, default=ESPALOMA_ADAM_LR)
    p.add_argument("--seed", type=int, default=ESPALOMA_SPLIT_SEED)
    p.add_argument("--eval-every", type=int, default=5, help="Full train/val MAE every N epochs")
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=50,
        help="Stop after this many non-improving evals (0=off). Espaloma uses offline inspect.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--run",
        type=Path,
        default=None,
        help="molexp run directory (preferred; enables Run lifecycle + MetricsWriter)",
    )
    p.add_argument(
        "--record-root",
        type=Path,
        default=None,
        help="Alias of --run (run directory)",
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--energy-scope",
        choices=("bonded", "full"),
        default="bonded",
        help="bonded: valence only; full: + EEM charges + LJ + Coulomb",
    )
    p.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="molexp workspace root (with --run-id; prefer --run)",
    )
    p.add_argument("--project", default=PROJECT_SLUG, help="molexp project id")
    p.add_argument(
        "--experiment",
        "--experiment-slug",
        dest="experiment",
        default=PROTOCOL_EXPERIMENT_SLUG,
        help="molexp experiment id",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help=f"molexp run id (default with --workspace: {PROTOCOL_RUN_ID})",
    )
    p.add_argument(
        "--dataset-coordinate",
        default=DATASET_COORDINATE_PRIMARY,
        help="MolHub coordinate (audit; data-root is local materialization)",
    )
    p.add_argument(
        "--no-harvest",
        action="store_true",
        help="Skip post-run harvest_run Finding KnowledgeItem",
    )
    args = p.parse_args(argv)

    if args.run is None and args.record_root is not None:
        args.run = args.record_root
    if args.workspace is not None and args.run_id is None and args.run is None:
        args.run_id = PROTOCOL_RUN_ID
    if args.workspace is None and args.run_id is not None and args.run is None:
        args.workspace = DEFAULT_WORKSPACE_ROOT

    molexp_run, record_root = _resolve_molexp_run(
        run_dir=args.run,
        workspace=args.workspace,
        project=args.project,
        experiment=args.experiment,
        run_id=args.run_id,
        record_root=args.record_root,
    )
    if args.run is not None and molexp_run is None:
        record_root = Path(args.run)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    frames_by_mol = pick_molecules(
        args.data_root,
        n_mols=args.n_mols,
        max_confs=args.max_confs,
        selection=args.selection,
    )
    mol_ids = list(frames_by_mol.keys())
    if args.use_official_splits:
        official = load_official_split(args.data_root)
        # Intersect with loaded mols (subset runs stay consistent)
        have = set(mol_ids)
        split = {
            "train": [m for m in official["train"] if m in have],
            "val": [m for m in official["val"] if m in have],
            "test": [m for m in official["test"] if m in have],
            "seed": official["seed"],
            "ratios": official["ratios"],
            "scheme": official["scheme"],
        }
        # Drop molecules not in any split (should not happen)
        keep = set(split["train"]) | set(split["val"]) | set(split["test"])
        frames_by_mol = {m: frames_by_mol[m] for m in mol_ids if m in keep}
        mol_ids = list(frames_by_mol.keys())
    else:
        split = molecule_split(mol_ids, seed=args.seed)
    n_atoms = {mid: int(len(frames_by_mol[mid][0]["atoms"]["number"])) for mid in mol_ids}

    if args.energy_scope == "full":
        model_meta = {
            "encoder": f"ChemEncoder dim={args.hidden}, n_blocks={args.n_blocks}",
            "heads": "Bond+Angle+Proper(n_terms=6)+LJParameterHead+ChiEtaHead",
            "charge": "ElectronegativityEqualization(DiagonalHardnessKernel)",
            "energy": "ForceField.to_potentials Class-I (bonded+LJ+Coulomb open)",
            "gap_vs_literature": [
                "EEM not numerically audited vs espaloma ChargeEquilibrium",
                "open Coulomb (no PME default); optional cell→Ewald/PME",
                "ChemEncoder ≠ SAGE×3 backbone",
            ],
        }
    else:
        model_meta = {
            "encoder": f"ChemEncoder dim={args.hidden}, n_blocks={args.n_blocks}",
            "heads": "Bond+Angle+Proper(n_terms=6)",
            "energy": "ForceField.to_potentials Class-I (bonded only)",
            "omitted_vs_espaloma": [
                "Charge equilibration (EEM)",
                "nonbonded LJ/Coulomb",
            ],
        }

    meta = {
        "protocol": "espaloma-mm-small-notebook",
        "references": ["arXiv:2010.01196", "DOI:10.1039/D2SC02739A"],
        "split_seed": args.seed,
        "split_ratios_parts": list(ESPALOMA_SPLIT_RATIOS),
        "adam_lr": args.lr,
        "batch_size": args.batch_size,
        "energy_scope": args.energy_scope,
        "loss": "MSE(E_pred, E_ref) graph-level absolute kcal/mol",
        "metric_report": "MAE/RMSE kcal/mol (labels already kcal/mol)",
        "model": model_meta,
        "molhub": {
            "dataset_coordinate": args.dataset_coordinate,
            "data_root": str(args.data_root),
        },
        "subset": {
            "n_mols": len(mol_ids),
            "mol_ids": mol_ids,
            "n_atoms": n_atoms,
            "confs_per_mol": {mid: len(frames_by_mol[mid]) for mid in mol_ids},
            "selection": (
                "all molecules"
                if args.n_mols <= 0
                else f"{args.n_mols} molecules ({args.selection})"
            ),
            "max_confs": args.max_confs,
            "hidden": args.hidden,
            "n_blocks": args.n_blocks,
            "n_torsion_terms": ESPALOMA_N_TORSION_TERMS,
        },
        "split": {k: v for k, v in split.items() if k != "seed"},
        "early_stop_patience": args.early_stop_patience,
        "project": args.project,
        "experiment": args.experiment,
        "run_id": getattr(molexp_run, "id", args.run_id),
    }
    print(json.dumps({"setup": meta}, indent=2), flush=True)

    def _prepare_record(root: Path | None) -> None:
        if root is None:
            return
        root.mkdir(parents=True, exist_ok=True)
        (root / "artifacts").mkdir(exist_ok=True)
        (root / "metrics").mkdir(exist_ok=True)
        (root / "artifacts" / "setup.json").write_text(json.dumps(meta, indent=2) + "\n")

    def _finish_artifacts(
        root: Path | None, history: list, final: dict, model: nn.Module
    ) -> Path | None:
        if root is None:
            return None
        (root / "artifacts" / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        (root / "artifacts" / "final.json").write_text(json.dumps(final, indent=2) + "\n")
        ckpt = args.checkpoint or (root / "artifacts" / "model.pt")
        torch.save({"model": model.state_dict(), "meta": meta, "final": final}, ckpt)
        return Path(ckpt)

    def _run_body(metrics_writer: object | None) -> tuple[list, dict, nn.Module]:
        model = build_espaloma_like_model(
            energy_scope=args.energy_scope,
            hidden=args.hidden,
            n_blocks=args.n_blocks,
        ).to(device)
        history = train(
            model,
            frames_by_mol,
            split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            record_root=record_root,
            eval_every=args.eval_every,
            metrics_writer=metrics_writer,
            early_stop_patience=args.early_stop_patience,
        )
        te = eval_split(model, frames_by_mol, split["test"], device, batch_size=args.batch_size)
        val_rows = [h for h in history if "val_mae_kcal" in h]
        final = {
            "best_val_mae_kcal": min(h["val_mae_kcal"] for h in val_rows) if val_rows else None,
            "best_val_epoch": min(val_rows, key=lambda h: h["val_mae_kcal"])["epoch"]
            if val_rows
            else None,
            "test_mae_kcal": te["mae_kcal"],
            "test_rmse_kcal": te["rmse_kcal"],
            "test_n": te["n"],
            "epochs": args.epochs,
            "energy_scope": args.energy_scope,
            "dataset_coordinate": args.dataset_coordinate,
        }
        print(json.dumps({"final": final}), flush=True)
        return history, final, model

    if molexp_run is not None:
        _prepare_record(record_root)
        with molexp_run.start() as ctx:
            history, final, model = _run_body(ctx.metrics)
            ckpt = _finish_artifacts(record_root, history, final, model)
            ctx.set_result("final", final)
            ctx.set_result("setup", meta)
            if ckpt is not None:
                ctx.set_result("checkpoint", str(ckpt))
            ctx.mark_succeeded()
        if not args.no_harvest:
            try:
                from molexp.workspace import harvest_run

                narrative = (
                    f"Class-I MM protocol mini ({args.energy_scope}): "
                    f"best val MAE {final.get('best_val_mae_kcal')} kcal/mol, "
                    f"test MAE {final.get('test_mae_kcal')} kcal/mol on "
                    f"{args.dataset_coordinate}."
                )
                item = harvest_run(
                    molexp_run,
                    kind="Finding",
                    narrative=narrative,
                    created_by="mm-param-learning/espaloma_mm_protocol_train",
                    results={k: v for k, v in final.items() if v is not None},
                )
                print(json.dumps({"harvest": item.name, "status": molexp_run.status}), flush=True)
            except Exception as exc:  # noqa: BLE001 — harvest is best-effort after train
                print(json.dumps({"harvest_error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 0

    # Standalone path (no molexp Run)
    if record_root is not None:
        (record_root / "metrics" / "metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (record_root / "metrics" / "metrics.jsonl").write_text("")
    _prepare_record(record_root)
    try:
        from molexp.workspace.metrics import MetricsWriter

        writer = MetricsWriter(record_root) if record_root is not None else None
    except ImportError:
        writer = None
    history, final, model = _run_body(writer)
    _finish_artifacts(record_root, history, final, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
