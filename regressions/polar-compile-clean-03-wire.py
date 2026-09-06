"""Polar MD compile seam: the prescribed sequence, its parity and its break split.

The compiled core is installed on the **mlip**, not on the ``Potentials``
container, and the caller composes the sequence — the library only exposes the
primitives::

    ff.def_pairstyle(model)
    pots = ff.to_potentials()
    pots.bind_cell(cell)       # size the k-grids + capacity check (eager)
    pots.calc_energy(work)     # one eager forward: element table + grids
    pots.compile_core(Compiler(backend=..., dynamic=True))

What this pins:

a. fp64 energy / forces / virial equal hard-coded goldens to 1e-10;
b. the compiled route equals the uncompiled one to 1e-10;
c. ``mlip.energy_core`` traces to 1 graph / 0 breaks, while the whole step
   breaks **only** inside ``src/molpot/derivation/force.py`` — the named
   autograd seam (leaf creation and ``torch.autograd.grad``) that Q3 keeps
   outside the compiled region, owner molpot derivation;
d. two geometries give two different ``graphs.energy`` values;
e. a ``Potentials`` rebuilt by ``to_potentials()`` still dispatches through the
   same compiled callable;
f. ``compile_core`` without the eager forward raises;
g. the stale ``PotentialForceField`` name is gone from the touched docstrings.

Everything here runs under ``torch.no_grad()``: this script measures the
first-order, cell-bound, B=1 MD **inference** route. Buffer donation is off by
default (``molix.compile.register_inductor_fallbacks``), so a grad-enabled
``create_graph=True`` force loss is no longer refused by AOTAutograd; that path
is asserted in ``tests/test_molpot/test_forcefield/test_potentials.py``, not
here. ``Compiler(donated_buffer=True)`` re-imposes ``create_graph=False``.

Goldens: captured 2026-09-03 on the pre-change eager path of this tree
(molnex @ ``feat/oplsaa-parameterizer``), CPU, fp64, with
``PYTHONPATH=src python regressions/polar-compile-clean-03-wire.py``. They are
literals, not a live oracle: nothing here imports or shells out to ``mace`` /
``e3nn`` / ``graph_longrange`` / ASE.

Run:
    PYTHONPATH=src python regressions/polar-compile-clean-03-wire.py

Reference:
    Batatia et al. "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model
    for Molecular Chemistry" arXiv:2602.19411.
"""

from __future__ import annotations

import time

import torch
from tensordict import TensorDict

from molix import config
from molix.compile import Compiler
from molix.schema import ENERGY_KEY
from molpot.forcefield import ForceField
from molzoo.mace import potential as potential_module
from molzoo.mace.polar import MacePolar
from molzoo.mace.potential import MacePotential
from molzoo.mace.spec import MACEPolarSpec

DTYPE = torch.float64

# ---- tiny Polar backbone (l_max=3, 16 channels) --------------------------
ATOMIC_NUMBERS = [1, 6, 8]
ATOMIC_ENERGIES = [-13.6, -1029.0, -2041.0]
BACKBONE = {
    "r_max": 5.0,
    "num_bessel": 4,
    "num_polynomial_cutoff": 5,
    "l_max": 3,
    "num_features": 16,
    "max_hidden_l": 1,
    "num_interactions": 2,
    "correlation": 2,
    "mlp_dim": 8,
    "edge_channels": 8,
    "use_fallback": True,  # CPU: the fused cueq kernels need a GPU + ops wheel
}

# ---- geometry: five atoms in a 10 Å cubic cell, one graph ----------------
CLUSTER_POS = [
    [0.00, 0.00, 0.00],
    [0.95, 0.00, 0.00],
    [-0.24, 0.93, 0.00],
    [0.00, 0.00, 1.40],
    [1.20, 1.10, 0.60],
]
CLUSTER_Z = [8, 1, 1, 6, 1]
CELL_EDGE = 10.0
#: Displacement of atom 1 along +x (Å) that defines the second geometry.
NUDGE = 0.05

# ---- hard-coded fp64 goldens (see module docstring for provenance) -------
GOLDEN_N_EDGES = 20
GOLDEN_ENERGY = -2860.720879072809  # eV
GOLDEN_ENERGY_SHIFTED = -2858.189091629231  # eV, atom 1 moved +0.05 Å in x
GOLDEN_FORCES = [  # eV/Å
    [75.74452748442475, 83.26162695104391, -66.82651527243766],
    [-48.61312575847743, -93.20993281460619, 39.40566323745179],
    [-52.726911856269396, -48.07024309786945, 55.24612256806561],
    [72.631961477376, 80.7912241007415, -74.29444344148871],
    [-47.03645134705394, -22.772675139309783, 46.469172908408986],
]
GOLDEN_VIRIAL = [  # eV
    [92.69162173838573, 185.17057416977644, -122.42540937615657],
    [185.17057416977644, 92.24134246717855, -168.7560303841212],
    [-122.42540937615657, -168.7560303841212, 130.59631884206516],
]
ATOL = 1e-10

#: The one file the whole step is allowed to break in (Q3; owner molpot
#: derivation): leaf creation and ``torch.autograd.grad`` inside
#: ``autograd_forces_and_virial``.
FORCE_SEAM = "src/molpot/derivation/force.py"


def build_model() -> MacePolar:
    """A seeded tiny Polar potential, energy only, in fp64.

    cuEquivariance freezes its working precision at **construction**, so the
    global dtype has to be fp64 before the model exists — a model built at fp32
    and then ``.double()``-d still contracts in float32.
    """
    config.set_precision("fp64")
    spec = MACEPolarSpec(
        atomic_numbers=ATOMIC_NUMBERS,
        atomic_energies=ATOMIC_ENERGIES,
        **BACKBONE,
    )
    torch.manual_seed(0)
    return MacePolar(
        atomic_numbers=spec.atomic_numbers,
        atomic_energies=spec.atomic_energies,
        r_max=spec.r_max,
        num_bessel=spec.num_bessel,
        num_polynomial_cutoff=spec.num_polynomial_cutoff,
        l_max=spec.l_max,
        num_features=spec.num_features,
        max_hidden_l=spec.max_hidden_l,
        num_interactions=spec.num_interactions,
        correlation=spec.correlation,
        mlp_dim=spec.mlp_dim,
        edge_channels=spec.edge_channels,
        radial_mlp=spec.radial_mlp,
        scale=spec.scale,
        shift=spec.shift,
        num_recursion_steps=spec.num_recursion_steps,
        multipole_lmax=spec.multipole_lmax,
        multipole_sigma=spec.multipole_sigma,
        field_lmax=spec.field_lmax,
        field_widths=spec.field_widths,
        field_norms=spec.field_norms,
        add_local_electron_energy=spec.add_local_electron_energy,
        electrostatic_self_interaction=spec.electrostatic_self_interaction,
        field_self_interaction=spec.field_self_interaction,
        use_fallback=spec.use_fallback,
        compute_forces=False,
    ).eval()


def full_edge_index(n_atoms: int) -> torch.Tensor:
    """Every ordered pair ``[source, target]`` of one graph, ``(E, 2)``."""
    pairs = [[i, j] for i in range(n_atoms) for j in range(n_atoms) if i != j]
    return torch.tensor(pairs, dtype=torch.long)


def make_work(pos: torch.Tensor, cell: torch.Tensor) -> TensorDict:
    """A post-collate periodic batch: ``atoms`` / ``edges`` / ``graphs``."""
    n_atoms = pos.shape[0]
    edge_index = full_edge_index(n_atoms)
    diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "Z": torch.tensor(CLUSTER_Z, dtype=torch.long),
                    "pos": pos.clone(),
                    "batch": torch.zeros(n_atoms, dtype=torch.long),
                },
                batch_size=[n_atoms],
            ),
            "edges": TensorDict(
                {
                    "edge_index": edge_index,
                    "edge_diff": diff.clone(),
                    "edge_dist": diff.norm(dim=-1),
                },
                batch_size=[edge_index.shape[0]],
            ),
            "graphs": TensorDict(
                {
                    "cell": cell.unsqueeze(0),
                    "total_charge": torch.zeros(1, dtype=DTYPE),
                    "total_spin": torch.ones(1, dtype=DTYPE),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


def main() -> None:
    pos = torch.tensor(CLUSTER_POS, dtype=DTYPE)
    nudged = pos.clone()
    nudged[1, 0] += NUDGE
    cell = torch.eye(3, dtype=DTYPE) * CELL_EDGE

    # (g) the deleted molix.md class must not survive in a docstring.
    assert "PotentialForceField" not in (potential_module.__doc__ or "")
    assert "PotentialForceField" not in (MacePotential.__doc__ or "")

    model = build_model()
    forcefield = ForceField("mace-polar", units="metal")
    forcefield.def_pairstyle(model)
    pots = forcefield.to_potentials()

    # (f) compiling before the eager forward is refused: the element table is
    # unvalidated and the k-grids are unsized, neither of which a trace can fix.
    try:
        pots.compile_core(Compiler(backend="aot_eager", dynamic=True))
    except RuntimeError as exc:
        assert "compile_core" in str(exc), exc
    else:  # pragma: no cover - the guard is the assertion
        raise AssertionError("compile_core must refuse before the eager forward")

    work = make_work(pos, cell)
    assert work["edges", "edge_index"].shape[0] == GOLDEN_N_EDGES

    with torch.no_grad():
        # --- the prescribed sequence ------------------------------------
        pots.bind_cell(cell)
        pots.calc_energy(make_work(pos, cell))

        eager = tuple(
            tensor.detach().clone()
            for tensor in pots.calc_energy_forces_virial(make_work(pos, cell))
        )

        hits: list[int] = []
        compiler = Compiler(backend="aot_eager", dynamic=True)

        def counting_compiler(core):
            """Compile, then count every call through the installed core."""
            compiled = compiler(core)

            def wrapper(*args, **kwargs):
                hits.append(1)
                return compiled(*args, **kwargs)

            return wrapper

        pots.compile_core(counting_compiler)

        # First call pays the compilation; the warm step is timed below.
        energy, forces, virial = pots.calc_energy_forces_virial(work)
        total = float(energy.detach().reshape(()))

        # (a) hard-coded goldens
        assert abs(total - GOLDEN_ENERGY) < ATOL, total
        assert torch.allclose(
            forces.detach(), torch.tensor(GOLDEN_FORCES, dtype=DTYPE), atol=ATOL, rtol=0.0
        )
        assert torch.allclose(
            virial.detach(), torch.tensor(GOLDEN_VIRIAL, dtype=DTYPE), atol=ATOL, rtol=0.0
        )

        # (b) compiled == uncompiled
        for got, want in zip((energy, forces, virial), eager, strict=True):
            assert torch.allclose(got.detach(), want, atol=ATOL, rtol=0.0), (got, want)

        # (d) a second geometry gives a different graphs.energy — and it is the
        # warm step, so it is also the one worth putting a clock on.
        first = work[ENERGY_KEY].detach().clone()
        shifted_work = make_work(nudged, cell)
        started = time.perf_counter()
        pots.calc_energy_forces_virial(shifted_work)
        elapsed = time.perf_counter() - started
        second = shifted_work[ENERGY_KEY].detach()
        assert not torch.equal(first, second)
        assert abs(float(second.reshape(())) - GOLDEN_ENERGY_SHIFTED) < ATOL

        # (e) a rebuilt Potentials still hits the same compiled callable
        before = len(hits)
        rebuilt = forcefield.to_potentials()
        assert rebuilt is not pots
        assert rebuilt.mlip is pots.mlip
        rebuilt.calc_energy(make_work(pos, cell))
        assert len(hits) == before + 1

    # (c) the compile scope: one graph for the core, only force.py for the step
    probe = make_work(pos, cell)
    flat = Compiler.explain(
        pots.mlip.energy_core,
        pos,
        probe["atoms", "Z"],
        probe["edges", "edge_index"],
        probe["atoms", "batch"],
        1,
        edge_diff=probe["edges", "edge_diff"],
        cell=cell.unsqueeze(0),
    )
    assert (flat.graph_count, flat.graph_break_count) == (1, 0), flat

    step = Compiler.explain(pots.calc_energy_forces_virial, make_work(pos, cell))
    assert step.graph_break_count > 0, step
    strays = [site for site in step.break_sites if FORCE_SEAM not in site]
    assert strays == [], f"break sites outside {FORCE_SEAM} (owner molpot derivation): {strays}"

    print(f"warm compiled step wall-clock (CPU, recorded not asserted): {elapsed * 1e3:.1f} ms")
    print(f"whole-step break sites ({len(step.break_sites)}): {sorted(set(step.break_sites))}")
    print(
        "polar-compile-clean-03-wire: ok — "
        f"energy {total:.9f} eV, "
        f"{GOLDEN_N_EDGES} edges, core 1 graph / 0 breaks, "
        f"step breaks only in {FORCE_SEAM}"
    )


if __name__ == "__main__":
    main()
