"""Identify which forcefield FORCE term NaNs at which training step.

Energy terms were already shown finite; this probes the per-term force paths.
"""

from __future__ import annotations

import torch

from scripts.mm_param_learning.class_i_fit import ClassIFit, load_omol25_split
from scripts.mm_param_learning.fingerprint import QUASI_PROD


def _state(t: torch.Tensor) -> str:
    if t.numel() == 0:
        return "empty"
    if not bool(torch.isfinite(t).all()):
        return "NAN"
    return f"ok maxabs={float(t.abs().max()):.4g}"


def main() -> None:
    train, val = load_omol25_split(
        role=QUASI_PROD["omol_role"],
        n_keep=0,
        max_atoms=32,
        train_frac=float(QUASI_PROD["train_frac"]),
        seed=int(QUASI_PROD["seed"]),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit = ClassIFit(
        train,
        val,
        hidden=64,
        n_blocks=4,
        max_atoms=int(QUASI_PROD["packed_max_atoms"]),
        epochs=1,
        device=device,
        record_root=None,
        eval_every_epochs=1,
        energy_scope="full",
    )
    model = fit.model()
    model.train()
    imm = model.imm
    for step, batch in enumerate(fit.datamodule().train_dataloader(), start=1):
        if step > 12:
            break
        batch = batch.to(device)
        model.zero_grad(set_to_none=True)
        out = model(batch)
        finite = bool(
            torch.isfinite(out["energy"]).all() and torch.isfinite(out["forces"]).all()
        )
        print(f"step={step} finite={finite}", flush=True)
        if finite:
            (out["energy"].square().mean() + out["forces"].square().mean()).backward()
            continue

        pots = imm(batch).to_potentials()
        pos = batch["atoms", "pos"].detach()
        n_atoms = pos.shape[0]
        num_graphs = pots._num_graphs(batch)
        zero = torch.zeros((num_graphs,) if num_graphs > 1 else (), device=device, dtype=pos.dtype)

        e_lj, f_lj = pots._lj_energy_forces(batch, pos, zero, n_atoms)
        print(f"  lj   energy={_state(e_lj)} force={_state(f_lj)}", flush=True)
        e_c, f_c = pots._coulomb_energy_forces(batch, pos, zero, n_atoms)
        print(f"  coul energy={_state(e_c)} force={_state(f_c)}", flush=True)

        atom_batch = pots._atom_batch(batch, n_atoms, device)
        batched = num_graphs > 1
        if pots.bond is not None:
            bi = pots._bond_index(batch)
            if bi is not None and bi.size(1) > 0:
                e, f = pots.bond.energy_forces(
                    pos=pos, bond_index=bi,
                    bond_types=pots._identity_types(bi.size(1), device),
                    term_batch=atom_batch[bi[0]] if batched else None,
                    num_graphs=num_graphs,
                )
                print(f"  bond force={_state(f)}", flush=True)
        if pots.angle is not None:
            ai = pots._angle_index(batch)
            if ai is not None and ai.size(1) > 0:
                e, f = pots.angle.energy_forces(
                    pos=pos, angle_index=ai,
                    angle_types=pots._identity_types(ai.size(1), device),
                    term_batch=atom_batch[ai[0]] if batched else None,
                    num_graphs=num_graphs,
                )
                print(f"  angle force={_state(f)}", flush=True)
        if pots.proper is not None:
            pi = pots._proper_index(batch)
            if pi is not None and pi.size(1) > 0:
                e, f = pots.proper.energy_forces(
                    pos=pos, proper_index=pi,
                    proper_types=pots._identity_types(pi.size(1), device),
                    term_batch=atom_batch[pi[0]] if batched else None,
                    num_graphs=num_graphs,
                )
                print(f"  proper force={_state(f)}", flush=True)
        if pots.improper is not None:
            ii = pots._improper_index(batch)
            if ii is not None and ii.size(1) > 0:
                e, f = pots.improper.energy_forces(
                    pos=pos, improper_index=ii,
                    improper_types=pots._identity_types(ii.size(1), device),
                    term_batch=atom_batch[ii[0]] if batched else None,
                    num_graphs=num_graphs,
                )
                print(f"  improper force={_state(f)}", flush=True)
        break


if __name__ == "__main__":
    main()
