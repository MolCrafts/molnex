#!/usr/bin/env python3
"""Espaloma-protocol MM fitting on a small PhAlkEthOH subset (our molexp project).

Matches Espaloma's published MM-small toy protocol as closely as our stack allows:

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
| GeometryInGraph + EnergyInGraph | ClassicalMMComposer Class-I kernels |
| ChargeEquilibrium + NB | **omitted** (documented deviation; bonded Class-I only) |

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
from tensordict import TensorDict

from molhub.dataset import PhalkethohMMDataset
from molhub.dataset.meta import Targets
from molpot.composition import ClassicalMMComposer, ClassicalMMParameterizer
from molpot.composition.mm_heads import (
    AngleParamHead,
    BondParamHead,
    ProperTorsionParamHead,
)
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


def enumerate_angles(atomi: np.ndarray, atomj: np.ndarray, n_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        "atoms": TensorDict({"Z": z, "pos": pos}, batch_size=[n]),
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
    energy = torch.tensor(float(Targets(fr)["mm_energy"]), dtype=torch.float64, device=device)
    mid = str(Targets(fr)["molecule_id"])
    return batch, energy, mid


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


def pick_small_molecules(
    data_root: Path, n_mols: int = 12, max_confs: int | None = None
) -> dict[str, list]:
    """Load all splits, pick *n_mols* smallest molecules by atom count."""
    # Use full corpus (no split filter) so we re-apply Espaloma seed ourselves.
    by: dict[str, list] = defaultdict(list)
    n_atoms: dict[str, int] = {}
    for split in ("train", "val", "test"):
        ds = PhalkethohMMDataset(data_root, download=False, split=split)
        for i in range(len(ds)):
            fr = ds[i]
            mid = str(Targets(fr)["molecule_id"])
            n_atoms[mid] = int(len(fr["atoms"]["number"]))
            if max_confs is None or len(by[mid]) < max_confs:
                by[mid].append(fr)
    ranked = sorted(n_atoms.items(), key=lambda kv: (kv[1], kv[0]))
    chosen = [mid for mid, _ in ranked[:n_mols]]
    return {mid: by[mid] for mid in chosen}


def build_espaloma_like_model() -> ClassicalMMParameterizer:
    """ChemEncoder + bond/angle/proper heads (128-wide, 6-term torsions)."""
    encoder = ChemEncoder(
        atom_dim=ESPALOMA_HIDDEN,
        bond_dim=ESPALOMA_HIDDEN,
        angle_dim=ESPALOMA_HIDDEN,
        proper_dim=ESPALOMA_HIDDEN,
        improper_dim=ESPALOMA_HIDDEN,
        hidden_dim=ESPALOMA_HIDDEN,
    )
    composer = ClassicalMMComposer(
        bond_head=BondParamHead(feature_dim=ESPALOMA_HIDDEN, hidden_dim=ESPALOMA_HIDDEN),
        angle_head=AngleParamHead(feature_dim=ESPALOMA_HIDDEN, hidden_dim=ESPALOMA_HIDDEN),
        proper_head=ProperTorsionParamHead(
            feature_dim=ESPALOMA_HIDDEN,
            hidden_dim=ESPALOMA_HIDDEN,
            n_terms=ESPALOMA_N_TORSION_TERMS,
            periodicity=tuple(range(1, ESPALOMA_N_TORSION_TERMS + 1)),
        ),
    )
    return ClassicalMMParameterizer(encoder, composer).double()


def mse_energy(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Espaloma GraphMetric MSE on graph energies."""
    return torch.mean((pred - ref) ** 2)


def mae_energy(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - ref)).detach())


@torch.no_grad()
def eval_split(
    model: ClassicalMMParameterizer,
    frames_by_mol: dict[str, list],
    mol_ids: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Topology once per molecule; evaluate all confs; report MAE/RMSE kcal/mol."""
    preds: list[torch.Tensor] = []
    refs: list[torch.Tensor] = []
    for mid in mol_ids:
        frames = frames_by_mol[mid]
        batch0, _, _ = frame_to_batch(frames[0], device)
        ir = model.parameterize(batch0)
        for fr in frames:
            batch, e_ref, _ = frame_to_batch(fr, device)
            e = model.energy(batch, ir=ir, pos=batch["atoms", "pos"]).reshape(())
            preds.append(e)
            refs.append(e_ref.reshape(()))
    pred = torch.stack(preds)
    ref = torch.stack(refs)
    return {
        "mae_kcal": mae_energy(pred, ref),
        "rmse_kcal": float(torch.sqrt(torch.mean((pred - ref) ** 2)).detach()),
        "n": int(pred.numel()),
    }


def train(
    model: ClassicalMMParameterizer,
    frames_by_mol: dict[str, list],
    split: dict[str, list[str]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    record_root: Path | None,
    eval_every: int = 5,
) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_ids = split["train"]
    val_ids = split["val"]

    # Flatten train conformers for Espaloma-style shuffled minibatches of graphs
    train_pool: list = []
    for mid in train_ids:
        train_pool.extend(frames_by_mol[mid])

    # Cache static topology batches (geometry free) for parameterize()
    topo_batch: dict[str, TensorDict] = {}
    for mid in frames_by_mol:
        b0, _, _ = frame_to_batch(frames_by_mol[mid][0], device)
        # topology-only view: keep pos but parameterize ignores geometry in encoder
        topo_batch[mid] = b0

    history: list[dict] = []
    best_val = float("inf")
    best_state: dict | None = None

    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_pool)))
        random.shuffle(order)
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            # Espaloma batches mixed graphs; mean MSE(u, u_ref) at graph level.
            losses: list[torch.Tensor] = []
            by_mid: dict[str, list] = defaultdict(list)
            for j in idx:
                fr = train_pool[j]
                mid = str(Targets(fr)["molecule_id"])
                by_mid[mid].append(fr)
            for mid, frames in by_mid.items():
                ir = model.parameterize(topo_batch[mid])
                for fr in frames:
                    batch, e_ref, _ = frame_to_batch(fr, device)
                    e = model.energy(batch, ir=ir, pos=batch["atoms", "pos"]).reshape(())
                    losses.append((e - e_ref.reshape(())) ** 2)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1

        do_eval = (epoch % eval_every == 0) or (epoch == epochs - 1)
        if do_eval:
            tr = eval_split(model, frames_by_mol, train_ids, device)
            va = eval_split(model, frames_by_mol, val_ids, device)
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

        if record_root is not None:
            metrics_dir = record_root / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            with (metrics_dir / "metrics.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True, help="Normalized PhAlkEthOH tree")
    p.add_argument("--n-mols", type=int, default=12, help="Small-system: number of smallest molecules")
    p.add_argument("--max-confs", type=int, default=20, help="Cap confs/mol (Espaloma uses 100; default 20 for mini)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=ESPALOMA_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=ESPALOMA_ADAM_LR)
    p.add_argument("--seed", type=int, default=ESPALOMA_SPLIT_SEED)
    p.add_argument("--eval-every", type=int, default=5, help="Full train/val MAE every N epochs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--record-root", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    args = p.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    frames_by_mol = pick_small_molecules(args.data_root, n_mols=args.n_mols, max_confs=args.max_confs)
    mol_ids = list(frames_by_mol.keys())
    split = molecule_split(mol_ids, seed=args.seed)
    n_atoms = {
        mid: int(len(frames_by_mol[mid][0]["atoms"]["number"])) for mid in mol_ids
    }

    meta = {
        "protocol": "espaloma-mm-small-notebook",
        "references": ["arXiv:2010.01196", "DOI:10.1039/D2SC02739A"],
        "split_seed": args.seed,
        "split_ratios_parts": list(ESPALOMA_SPLIT_RATIOS),
        "adam_lr": args.lr,
        "batch_size": args.batch_size,
        "loss": "MSE(E_pred, E_ref) graph-level absolute kcal/mol",
        "metric_report": "MAE/RMSE kcal/mol (labels already kcal/mol)",
        "model": {
            "encoder": f"ChemEncoder dim={ESPALOMA_HIDDEN}",
            "heads": "Bond+Angle+Proper(n_terms=6)",
            "energy": "ClassicalMMComposer bonded Class-I",
            "omitted_vs_espaloma": ["ChargeEquilibrium", "nonbonded LJ/Coulomb EnergyInGraph NB"],
        },
        "subset": {
            "n_mols": len(mol_ids),
            "mol_ids": mol_ids,
            "n_atoms": n_atoms,
            "confs_per_mol": {mid: len(frames_by_mol[mid]) for mid in mol_ids},
            "selection": f"{args.n_mols} smallest molecules by atom count",
        },
        "split": {k: v for k, v in split.items()},
        "project": "mm-param-learning",
        "experiment": "espaloma-mm-protocol-mini",
    }
    print(json.dumps({"setup": meta}, indent=2), flush=True)

    if args.record_root is not None:
        args.record_root.mkdir(parents=True, exist_ok=True)
        (args.record_root / "artifacts").mkdir(exist_ok=True)
        (args.record_root / "metrics").mkdir(exist_ok=True)
        # truncate metrics
        (args.record_root / "metrics" / "metrics.jsonl").write_text("")
        (args.record_root / "artifacts" / "setup.json").write_text(json.dumps(meta, indent=2) + "\n")

    model = build_espaloma_like_model().to(device)
    history = train(
        model,
        frames_by_mol,
        split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        record_root=args.record_root,
        eval_every=args.eval_every,
    )

    te = eval_split(model, frames_by_mol, split["test"], device)
    val_rows = [h for h in history if "val_mae_kcal" in h]
    final = {
        "best_val_mae_kcal": min(h["val_mae_kcal"] for h in val_rows) if val_rows else None,
        "best_val_epoch": min(val_rows, key=lambda h: h["val_mae_kcal"])["epoch"] if val_rows else None,
        "test_mae_kcal": te["mae_kcal"],
        "test_rmse_kcal": te["rmse_kcal"],
        "test_n": te["n"],
        "epochs": args.epochs,
    }
    print(json.dumps({"final": final}), flush=True)

    if args.record_root is not None:
        (args.record_root / "artifacts" / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        (args.record_root / "artifacts" / "final.json").write_text(json.dumps(final, indent=2) + "\n")
        ckpt = args.checkpoint or (args.record_root / "artifacts" / "model.pt")
        torch.save({"model": model.state_dict(), "meta": meta, "final": final}, ckpt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
