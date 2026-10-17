"""Step-7 shell reference: fp64 forward of the tiny MACE fixtures (save / compare)."""

import sys

import torch
from tests.conftest import make_graph_batch  # noqa: E402
from tests.test_molzoo.test_mace.conftest import (  # noqa: E402
    ATOMIC_ENERGIES,
    ATOMIC_NUMBERS,
    CLUSTER_POS,
    CLUSTER_Z,
    PAIR_BATCH,
    PAIR_POS,
    PAIR_Z,
    SINGLE_GRAPH,
    TINY_MATPES_KWARGS,
    TINY_OMOL_KWARGS,
    TINY_POLAR_KWARGS,
    full_edge_index,
    wake_zero_init_readout,
)

from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, FORCES_KEY
from molzoo.mace.polar import MacePolar  # noqa: E402
from molzoo.mace.potential import MaceMatpes, MaceOmol  # noqa: E402
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec, MACEPolarSpec  # noqa: E402

#: Build dtype of every model here (the reference is fp64).
DTYPE = torch.float64

PATH = sys.argv[1]
MODE = sys.argv[2]  # "save" | "compare"


def cluster(conditioned: bool):
    graphs = (
        {
            "total_charge": torch.zeros(1, dtype=torch.long),
            "total_spin": torch.ones(1, dtype=torch.long),
        }
        if conditioned
        else None
    )
    return make_graph_batch(
        pos=torch.tensor(CLUSTER_POS, dtype=torch.float64),
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        pair_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
        graphs=graphs,
    )


def pair_batch():
    return make_graph_batch(
        pos=torch.tensor(PAIR_POS, dtype=torch.float64),
        Z=torch.tensor(PAIR_Z, dtype=torch.long),
        pair_index=full_edge_index(PAIR_BATCH),
        batch=torch.tensor(PAIR_BATCH, dtype=torch.long),
    )


def models():
    torch.manual_seed(0)
    matpes = MaceMatpes(
        MACEMatpesSpec(
            atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **TINY_MATPES_KWARGS
        ),
        wrt=("atoms.pos",),
        dtype=DTYPE,
    ).eval()
    torch.manual_seed(0)
    omol = MaceOmol(
        MACEOMolSpec(
            atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **TINY_OMOL_KWARGS
        ),
        wrt=("atoms.pos",),
        dtype=DTYPE,
    ).eval()
    wake_zero_init_readout(omol)
    torch.manual_seed(0)
    polar = MacePolar(
        MACEPolarSpec(
            atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **TINY_POLAR_KWARGS
        ),
        wrt=("atoms.pos",),
        dtype=DTYPE,
    ).eval()
    return {
        "matpes": (matpes, cluster(False)),
        "matpes_pair": (matpes, pair_batch()),
        "omol": (omol, cluster(True)),
        "polar": (polar, cluster(True)),
    }


def run():
    out = {}
    for name, (model, batch) in models().items():
        frame = model(batch)
        out[name] = {
            "graphs.energy": frame[ENERGY_KEY].detach().clone(),
            "atoms.energy": frame[ATOMIC_ENERGY_KEY].detach().clone(),
            "atoms.forces": frame[FORCES_KEY].detach().clone(),
        }
    return out


got = run()
if MODE == "save":
    torch.save(got, PATH)
    for name, d in got.items():
        print(name, "graphs.energy", d["graphs.energy"].tolist())
    print("saved", PATH)
else:
    ref = torch.load(PATH)
    worst = 0.0
    for name, d in ref.items():
        for key, value in d.items():
            new = got[name][key]
            denom = value.abs().max().clamp_min(1e-300)
            rel = ((new - value).abs().max() / denom).item()
            worst = max(worst, rel)
            print(
                f"{name:12s} {key:14s} max_rel={rel:.3e} "
                f"max_abs={(new - value).abs().max().item():.3e}"
            )
    print("WORST max_rel", f"{worst:.3e}", "OK" if worst <= 1e-12 else "FAIL")
    sys.exit(0 if worst <= 1e-12 else 1)
