"""MacePolar vs official MACE-POLAR-1-M on a periodic PEO box (hard-coded goldens).

Per-term energy parity on the k-space (``cell``-given) path: reference
energies + local interaction, local electron energy, k-space Hartree, and the
total. Goldens are the official ``PolarMACE`` (mace-torch 0.3.16 + cueq,
``graph_longrange`` 0.4.0) fp64 outputs captured on a GH200 on 2026-09-02 for
the 4995-atom, 46.36 Å cubic PEO cell — the box that exposed the on-site
self-term double count on the periodic path (``.claude/notes/notes.md`` topic
``cueq-conv-fusion``, "periodic Polar parity").

Requires the box, the Polar dump and the out-of-tree cueq overlay state;
skips when any is absent. No mace / e3nn / ASE imported here. CUDA fp64 —
the 2.9·10⁵-edge box needs a GPU.

    MOLNEX_POLAR_BOX=.../peo_box.pt MOLNEX_POLAR_DUMP=... MOLNEX_POLAR_CUEQ_STATE=... \\
    PYTHONPATH=src python regressions/mace-polar-periodic-gto.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from tensordict import TensorDict

from molix import config
from molix.md import NeighborList
from molpot.forcefield import ForceField as MolpotForceField
from molzoo.mace.checkpoint import CheckpointRemap, polar_keys
from molzoo.mace.polar import MacePolar

config.set_precision("fp64")

_BOX = Path(os.environ.get("MOLNEX_POLAR_BOX", "/tmp/mace-polar/peo_box.pt"))
_DUMP = Path(os.environ.get("MOLNEX_POLAR_DUMP", "/tmp/mace-polar/MACE-POLAR-1-M.model"))
_CUEQ = Path(os.environ.get("MOLNEX_POLAR_CUEQ_STATE", "/tmp/polar_cueq_state.pt"))

# Official PolarMACE fp64, PEO box (Q=0, M=1), captured 2026-09-02.
_E0_PLUS_INTER = -2829206.9445318216
_ELECTRON_ENERGY = 685.2709477417063
_ELECTROSTATIC_ENERGY = 9.143779770475982
_ENERGY = -2828512.52980431
_N_EDGES = 285546

_ATOL_EV = 1.0e-3


def main() -> None:
    if not (_BOX.is_file() and _DUMP.is_file() and _CUEQ.is_file()):
        print("skip: PEO box, Polar dump or cueq overlay state not present")
        return
    if not torch.cuda.is_available():
        print("skip: needs CUDA for the 2.9e5-edge box")
        return
    device = torch.device("cuda")
    dtype = torch.float64
    blob = torch.load(_BOX, map_location="cpu", weights_only=True)
    z = blob["Z"].to(device=device, dtype=torch.long)
    pos = blob["pos"].to(device=device, dtype=dtype)
    cell = blob["cell"].to(device=device, dtype=dtype)
    n = int(pos.shape[0])

    model = MacePolar.load(_DUMP, format="official", use_fallback=False, compute_forces=False)
    cueq_sd = torch.load(_CUEQ, map_location="cpu", weights_only=True)
    CheckpointRemap(polar_keys, on_unexpected="return").load(model, cueq_sd)
    model = model.to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)

    captured: dict[str, torch.Tensor] = {}

    def keep(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output.detach()

        return hook

    handles = [
        model.atomic_energies.register_forward_hook(keep("e0")),
        model.scale_shift.register_forward_hook(keep("inter")),
        model.local_electron_energy.register_forward_hook(keep("le")),
        model.coulomb_energy.register_forward_hook(keep("hartree")),
    ]

    work = TensorDict(
        {
            "atoms": TensorDict(
                {"Z": z, "pos": pos, "batch": torch.zeros(n, dtype=torch.long, device=device)},
                batch_size=[n],
            ),
            "graphs": TensorDict(
                {
                    "cell": cell.unsqueeze(0),
                    "total_charge": torch.zeros(1, dtype=dtype, device=device),
                    "total_spin": torch.ones(1, dtype=dtype, device=device),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )
    nl = NeighborList(
        cell=cell,
        cutoff=float(model._spec.r_max),
        positions=pos,
        skin=0.0,
        every=1,
        delay=0,
        check=False,
        capacity_factor=1.2,
        device=device,
    )
    nl.build(work)
    n_live = nl.num_edges
    full = work["edges"]
    work["edges"] = TensorDict(
        {key: full[key][:n_live] for key in full.keys()}, batch_size=[n_live]
    )
    assert n_live == _N_EDGES, f"n_edges={n_live} != {_N_EDGES}"

    ff = MolpotForceField("mace-polar", units="metal")
    ff.def_pairstyle(model)
    pots = ff.to_potentials()
    with torch.no_grad():
        energy = float(pots.calc_energy(work).sum())
    for handle in handles:
        handle.remove()

    got = {
        "e0_plus_inter": float(captured["e0"].sum() + captured["inter"].sum()),
        "electron_energy": float(captured["le"].sum()),
        "electrostatic_energy": float(captured["hartree"].sum()),
        "energy": energy,
    }
    gold = {
        "e0_plus_inter": _E0_PLUS_INTER,
        "electron_energy": _ELECTRON_ENERGY,
        "electrostatic_energy": _ELECTROSTATIC_ENERGY,
        "energy": _ENERGY,
    }
    bad = {k: got[k] - gold[k] for k in gold if abs(got[k] - gold[k]) >= _ATOL_EV}
    assert not bad, f"periodic Polar terms off vs official (eV): {bad}"
    print(
        "mace-polar-periodic-gto: ok  " + "  ".join(f"d{k}={got[k] - gold[k]:+.2e}" for k in gold)
    )


if __name__ == "__main__":
    main()
