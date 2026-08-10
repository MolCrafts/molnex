#!/usr/bin/env python3
"""Train bond-head ClassicalMMParameterizer on PhAlkEthOH MM-small (Validation B2 smoke).

Requires a **normalized** data tree (from molhub convert script)::

    PYTHONPATH=src:../molhub/src python scripts/mm_param_learning/train_phalkethoh_mm_energy.py \\
      --data-root /path/to/phalkethoh-mm-small-normalized \\
      --epochs 5 --max-train-mols 8 --max-confs-per-mol 8

Primary metric: molecule-mean-centered energy RMSE/MAE (kcal/mol).
Bond-only model is a smoke path (full valence heads come later).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tensordict import TensorDict

from molhub.dataset import PhalkethohMMDataset
from molhub.dataset.meta import Targets
from molix.core.losses.molecular import center_by_group
from molpot.composition import ClassicalMMComposer, ClassicalMMParameterizer
from molpot.composition.mm_heads import BondParamHead
from molrep.chem.encoder import ChemEncoder


def _group_frames(ds: PhalkethohMMDataset, max_mols: int, max_confs: int) -> dict[str, list]:
    by: dict[str, list] = defaultdict(list)
    for i in range(len(ds)):
        fr = ds[i]
        mid = str(Targets(fr)["molecule_id"])
        if mid not in by and len(by) >= max_mols:
            continue
        if len(by[mid]) < max_confs:
            by[mid].append(fr)
        if len(by) >= max_mols and all(len(v) >= max_confs for v in by.values()):
            break
    return dict(by)


def _to_batch(fr) -> TensorDict:
    atoms = fr["atoms"]
    bonds = fr["bonds"]
    z = torch.as_tensor(atoms["number"], dtype=torch.long)
    pos = torch.stack(
        [torch.as_tensor(atoms[c], dtype=torch.float64) for c in ("x", "y", "z")],
        dim=-1,
    )
    atomi = torch.as_tensor(bonds["atomi"], dtype=torch.long)
    atomj = torch.as_tensor(bonds["atomj"], dtype=torch.long)
    return TensorDict(
        {
            "atoms": TensorDict({"Z": z, "pos": pos}, batch_size=[z.shape[0]]),
            "bonds": TensorDict({"atomi": atomi, "atomj": atomj}, batch_size=[atomi.shape[0]]),
        },
        batch_size=[],
    )


def _run_epoch(
    model: ClassicalMMParameterizer,
    by: dict[str, list],
    *,
    train: bool,
    opt: torch.optim.Optimizer | None,
) -> dict[str, float]:
    model.train(train)
    preds: list[torch.Tensor] = []
    refs: list[torch.Tensor] = []
    groups: list[int] = []
    mid_map: dict[str, int] = {}
    for mid, frames in by.items():
        mid_map.setdefault(mid, len(mid_map))
        batch0 = _to_batch(frames[0])
        with torch.set_grad_enabled(train):
            ir = model.parameterize(batch0)
            for fr in frames:
                batch = _to_batch(fr)
                e = model.energy(batch, ir=ir, pos=batch["atoms", "pos"]).reshape(())
                preds.append(e)
                refs.append(
                    torch.tensor(float(Targets(fr)["mm_energy"]), dtype=torch.float64)
                )
                groups.append(mid_map[mid])
    pred = torch.stack(preds)
    ref = torch.stack(refs)
    g = torch.tensor(groups, dtype=torch.long)
    loss = torch.mean((center_by_group(pred, g) - center_by_group(ref, g)) ** 2)
    if train and opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        cp = center_by_group(pred, g)
        cr = center_by_group(ref, g)
        rmse = float(torch.sqrt(torch.mean((cp - cr) ** 2)).detach())
        mae = float(torch.mean(torch.abs(cp - cr)).detach())
    return {"rmse_kcal": rmse, "mae_kcal": mae, "loss": float(loss.detach())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-train-mols", type=int, default=8)
    p.add_argument("--max-val-mols", type=int, default=4)
    p.add_argument("--max-confs-per-mol", type=int, default=8)
    p.add_argument("--record-root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    train_by = _group_frames(
        PhalkethohMMDataset(args.data_root, download=False, split="train"),
        args.max_train_mols,
        args.max_confs_per_mol,
    )
    val_by = _group_frames(
        PhalkethohMMDataset(args.data_root, download=False, split="val"),
        args.max_val_mols,
        args.max_confs_per_mol,
    )
    print(
        json.dumps(
            {
                "n_train_mols": len(train_by),
                "n_val_mols": len(val_by),
                "data_root": str(args.data_root),
            }
        )
    )

    encoder = ChemEncoder(atom_dim=32, bond_dim=32)
    composer = ClassicalMMComposer(bond_head=BondParamHead(feature_dim=32))
    model = ClassicalMMParameterizer(encoder, composer).double()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    for epoch in range(args.epochs):
        tr = _run_epoch(model, train_by, train=True, opt=opt)
        va = _run_epoch(model, val_by, train=False, opt=None)
        row = {
            "epoch": epoch,
            "train_rmse_kcal": tr["rmse_kcal"],
            "train_mae_kcal": tr["mae_kcal"],
            "val_rmse_kcal": va["rmse_kcal"],
            "val_mae_kcal": va["mae_kcal"],
        }
        history.append(row)
        print(json.dumps(row))

    if args.record_root is not None:
        args.record_root.mkdir(parents=True, exist_ok=True)
        (args.record_root / "metrics").mkdir(exist_ok=True)
        (args.record_root / "artifacts").mkdir(exist_ok=True)
        with (args.record_root / "metrics" / "train_smoke.jsonl").open("w") as fh:
            for row in history:
                fh.write(json.dumps(row) + "\n")
        (args.record_root / "artifacts" / "train_smoke_history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
