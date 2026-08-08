"""Public-API checkpoint round trip for chain step mace-subpackage-restructure-05.

Chain step 05 merges the two hand-copied official-checkpoint loaders
(``molzoo.mace_matpes.load_matpes_state_dict`` /
``molzoo.mace_omol.load_omol_state_dict``) into the one
:class:`molzoo.mace.CheckpointRemap`, and gives step 04's
:class:`molzoo.mace.MACEPotential` a ``from_checkpoint`` classmethod. Step 07
then **deletes** the flat modules, so this file must never import them — it
uses ``molzoo.mace`` only and is expected to keep running unchanged afterwards.

Scenario (public API only), start to finish inside one ``tempfile`` directory:

1. build a tiny MatPES potential from a :class:`molzoo.mace.MACEMatpesSpec`
   (the step-04 API) and fill **every** parameter, plus the fitted-constant
   buffers, from a shape-derived deterministic rule — ``torch.linspace``, no
   RNG at all, so these goldens cannot be knocked over by a change in torch's
   random-number order or in module initialisation;
2. dump that ``state_dict`` under the **official cueq key names** and write it
   next to a minimal official-style config json whose ``hidden_irreps`` /
   ``MLP_irreps`` are the *only* statement of the channel widths — a
   ``from_checkpoint`` that hard-coded ``128 / 1 / 16`` would raise on shapes
   here;
3. read it back with ``MACEPotential.from_checkpoint(config, weights)`` and
   assert the reconstruction is exact: every ``state_dict`` tensor
   ``torch.equal`` to the source model's;
4. evaluate energy and forces on a fixed 5-atom cluster (fp64, CPU,
   ``use_fallback=True``) against the hard-coded goldens below;
5. exercise the strictness doctrine on the cheap: a checkpoint key with no home
   raises ``RuntimeError`` under :data:`molzoo.mace.MATPES_REMAP` and leaves the
   model untouched, while the same key is merely *reported* by a
   ``CheckpointRemap(MATPES_KEY_REMAP, on_unexpected="return")``.

The synthetic checkpoint is written in the real dialect, not a convenient one:
it carries the graph constants and an ``output_mask`` (both skipped on load),
the ``r_max`` / ``num_interactions`` / ``cutoff_fn.p`` / ``pair_repulsion_fn.p``
scalars and the ``atomic_energies_fn`` table (all *dropped* by the remap — E0
comes from the config). Every one of those would otherwise surface as a key
"with no home" and abort the load, so the round trip passing is itself the
assertion that the drop/skip rules still hold.

Goldens
-------
Captured by running this file at the commit below and pasting back what it
printed. There is no external oracle and none is possible: the numbers are what
*this repo* computes for a synthetic, deterministically filled model — the
point of the file is that the checkpoint round trip does not move them.

    capture command : PYTHONPATH=src python \\
                      regressions/mace-subpackage-restructure-05-checkpoint.py
    commit          : cf50d3a (cf50d3ac6af1caa151669cc36d5baba8bfc5a988) plus
                      the uncommitted step-05 working tree
                      (``src/molzoo/mace/checkpoint.py`` and
                      ``MACEPotential.from_checkpoint``)
    torch           : 2.12.1+cpu   (python 3.14.5)
    date            : 2026-08-08
    device / dtype  : CPU, float64 (``config.set_precision("fp64")``)
    oracle          : this repository, self-consistent. No mace-torch, no e3nn,
                      no ASE, no network, no subprocess — and no RNG, seeded or
                      otherwise (see :func:`fill_ramp`).
    observed        : two consecutive runs byte-identical; the loaded model
                      reproduced the source model's energy and forces exactly
                      (deviation 0.0, not merely inside the band); Σ|F| and the
                      per-atom force table are pinned below.

Tolerances follow ``tests/test_molzoo/test_mace_matpes.py:96-97``: 1e-9 eV on
energies, 1e-8 eV/Å on forces.

Run:
    PYTHONPATH=src python regressions/mace-subpackage-restructure-05-checkpoint.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

from molix import config

config.set_precision("fp64")  # before any module construction: cuEq bakes in dtype

from tensordict import TensorDict  # noqa: E402

from molzoo.mace import (  # noqa: E402
    MATPES_KEY_REMAP,
    MATPES_REMAP,
    CheckpointRemap,
    MACEMatpesSpec,
    MACEPotential,
)

#: Energy band in eV and force band in eV/Å — the fp64 tolerances of
#: ``tests/test_molzoo/test_mace_matpes.py:96-97``.
ENERGY_ATOL = 1e-9
FORCE_ATOL = 1e-8

#: The loaded model holds the *same* tensors as the source, so it must run the
#: *same* arithmetic: its energy and forces are compared to the source model's
#: at exactly zero tolerance, not at :data:`ENERGY_ATOL`.
ROUNDTRIP_ATOL = 0.0

#: Newton's third law on an isolated cluster (measured ≤ 1e-16 eV/Å).
NET_FORCE_ATOL = 1e-10

# ---------------------------------------------------------------------------
# System: the fixed 5-atom cluster of
# regressions/mace-subpackage-restructure-04-potential.py, all 20 ordered pairs
# as directed edges. Literal coordinates in Å — reproducible anywhere.
# ---------------------------------------------------------------------------
ATOMIC_NUMBERS: list[int] = [1, 6, 8]
ATOMIC_ENERGIES: list[float] = [-13.6, -1029.0, -2041.0]  # eV/atom

POSITIONS = torch.tensor(
    [
        [0.00, 0.00, 0.00],
        [1.09, 0.00, 0.00],
        [1.70, 1.15, 0.00],
        [-0.40, 0.95, 0.30],
        [2.10, -0.85, -0.50],
    ],
    dtype=torch.float64,
)
Z = torch.tensor([1, 6, 8, 1, 1], dtype=torch.long)
BATCH = torch.zeros(5, dtype=torch.long)
NUM_GRAPHS = 1
EDGE_INDEX = torch.tensor(  # (E, 2): [:, 0] = source, [:, 1] = target
    [(i, j) for i in range(5) for j in range(5) if i != j], dtype=torch.long
)

# ---------------------------------------------------------------------------
# The synthetic checkpoint's hyper-parameters. ``num_features`` /
# ``max_hidden_l`` / ``mlp_dim`` appear in the config json *only* as the irreps
# strings: ``from_checkpoint`` has to parse 16 / 1 / 8 out of them.
# ---------------------------------------------------------------------------
NUM_FEATURES = 16
MAX_HIDDEN_L = 1
MLP_DIM = 8
HIDDEN_IRREPS = "16x0e+16x1o"
MLP_IRREPS = "8x0e"

#: ``atomic_inter_scale`` / ``atomic_inter_shift`` as written into the config
#: json. The checkpoint's own ``scale_shift`` buffers are deliberately filled
#: *away* from these two values (see :func:`fill_potential`), so the loaded
#: model can only match the goldens if the checkpoint's numbers won.
CONFIG_SCALE = 0.7
CONFIG_SHIFT = -0.25

SPEC_KWARGS = dict(
    r_max=5.0,
    num_bessel=4,
    num_polynomial_cutoff=5,
    l_max=1,
    num_features=NUM_FEATURES,
    max_hidden_l=MAX_HIDDEN_L,
    num_interactions=2,
    correlation=2,
    mlp_dim=MLP_DIM,
    radial_mlp=[8],
    scale=CONFIG_SCALE,
    shift=CONFIG_SHIFT,
)

# ---------------------------------------------------------------------------
# Deterministic, RNG-free fill
# ---------------------------------------------------------------------------

#: Half-width of the ``torch.linspace`` ramp every parameter is filled with.
#: Comparable to the ``N(0, 1)`` initialisation cuEquivariance linears use, and
#: chosen empirically: this tiny model's symmetric contraction (body order 2)
#: leaves the physical regime somewhere above ~1.4 — at 1.5 the forces reach
#: 4e2 eV/Å. At 1.2 the learned readouts still carry 0.30 eV of the energy and
#: 7e-2 eV/Å of the forces, so the goldens are not a ZBL-only measurement.
FILL_SPAN = 1.2

#: Per-name phase added to the ramp, so two tensors of the same length never
#: receive the same values (a remap that swapped two same-shaped keys would
#: otherwise round-trip unnoticed). A character sum, deliberately not
#: :func:`hash`, which is salted per process.
PHASE_MODULUS = 17

#: Fitted physical constants an official checkpoint carries and the remap
#: therefore has to transport: the Agnesi transform, the ZBL pair term and the
#: affine energy normalisation. They are perturbed rather than overwritten —
#: 1e-2 keeps every covalent radius and screening coefficient positive, so the
#: model stays sane, while still moving each buffer off the value construction
#: would give it. Anything outside these three families (the cuEquivariance
#: graph constants, the symmetric-contraction projection, the E0 table) is a
#: structural constant, rebuilt identically on both sides, and is left alone.
FILLED_BUFFER_PREFIXES = ("distance_transform.", "pair_repulsion.", "scale_shift.")
BUFFER_FILL_SCALE = 0.01

# ---------------------------------------------------------------------------
# The official cueq key dialect, written out by hand
# ---------------------------------------------------------------------------

#: molnex prefix → official cueq prefix for the entries an official MatPES
#: checkpoint **carries into the model**. Hand-written rather than derived from
#: :data:`molzoo.mace.MATPES_KEY_REMAP`, so a wrong entry in the shipped table
#: cannot cancel itself out in this round trip;
#: :func:`check_official_dialect` asserts the two still agree.
CARRIED_OFFICIAL_NAMES: dict[str, str] = {
    "node_embedding.": "node_embedding.linear.",
    "bessel.freqs": "radial_embedding.bessel_fn.bessel_weights",
    "distance_transform.": "radial_embedding.distance_transform.",
    "pair_repulsion.": "pair_repulsion_fn.",
    "z_table": "atomic_numbers",
}

#: molnex prefix → official cueq prefix for entries the checkpoint carries and
#: the remap **drops**: the isolated-atom reference energies are rebuilt from
#: the config's ``atomic_energies``, Z-indexed, not read out of the weights.
DROPPED_OFFICIAL_NAMES: dict[str, str] = {
    "atomic_energies.": "atomic_energies_fn.",
}

#: Official-only entries with no molnex counterpart at all. Each is dropped by
#: a ``None`` row of the shipped table — exactly (``r_max``,
#: ``num_interactions``, ``pair_repulsion_fn.p``) or through the enclosing
#: prefix (``radial_embedding.cutoff_fn.``); if any stopped being dropped, the
#: load would abort with "no home" instead of reaching the goldens.
#: ``pair_repulsion_fn.p`` is also the longest-prefix probe — its exact row
#: must beat the ``pair_repulsion_fn.`` prefix that encloses it, or it would
#: land as ``pair_repulsion.p`` and be homeless.
OFFICIAL_ONLY_SCALARS: dict[str, float] = {
    "r_max": 5.0,
    "num_interactions": 2.0,
    "radial_embedding.cutoff_fn.p": 5.0,
    "pair_repulsion_fn.p": 5.0,
}

#: An irrep mask of the kind cuEquivariance emits and rebuilds. Skipped by
#: ``rename`` on the ``output_mask`` suffix; were it not, it would be a key
#: with no home and the load would raise.
OUTPUT_MASK_KEY = "interactions.0.conv_tp.output_mask"

#: A checkpoint key that matches no table entry and no model parameter — the
#: probe for the ``on_unexpected`` knob (the same key the unit tests use).
MYSTERY_KEY = "interactions.0.mystery_layer.weight"

# ---------------------------------------------------------------------------
# Goldens — see the module docstring for provenance.
# ---------------------------------------------------------------------------

#: Total energy of the cluster, eV, ``(B,)``.
ENERGY: list[float] = [-3111.798363383643]

#: Forces, eV/Å, ``(N, 3)``.
FORCES: list[list[float]] = [
    [0.02073978897711603, -0.014283907947197812, -0.004580298404743775],
    [-0.07674001496483682, -0.1308470223078055, -0.002087429078993083],
    [0.10544897994819648, 0.1571999433835074, 0.0003340549276333903],
    [0.022217895631466936, -0.04160730323074992, -0.012362154174687933],
    [-0.07166664959194265, 0.02953829010224585, 0.0186958267307914],
]

#: Σ|F| in eV/Å — one number that moves if any component does.
FORCE_ABS_SUM = 0.708349559401915

#: ``scale_shift`` as it comes back out of the checkpoint. Both differ from the
#: config's :data:`CONFIG_SCALE` / :data:`CONFIG_SHIFT`, which is what makes
#: "the checkpoint's value won" an observable claim.
SCALE_GOLDEN = 0.6886
SHIFT_GOLDEN = -0.2609


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def fill_ramp(name: str, count: int) -> torch.Tensor:
    """Deterministic ``count`` values for the tensor called ``name``.

    A ``torch.linspace`` ramp over ``±`` :data:`FILL_SPAN` plus a per-name
    phase. No RNG is involved anywhere in this file, so the goldens survive any
    change to torch's random-number order or to module initialisation — the
    failure mode that makes seeded goldens rot.

    ``torch.linspace(-s, s, 1)`` is ``[-s]``, so a single-element tensor gets
    ``-FILL_SPAN + phase``.

    Args:
        name: ``state_dict`` key, used only for the phase.
        count: Number of values to produce.

    Returns:
        A flat ``(count,)`` float64 tensor.
    """
    phase = (sum(ord(character) for character in name) % PHASE_MODULUS) / 100.0
    return torch.linspace(-FILL_SPAN, FILL_SPAN, count, dtype=torch.float64) + phase


def fill_potential(potential: MACEPotential) -> None:
    """Overwrite every parameter, and the fitted buffers, in place.

    Every ``nn.Parameter`` is *replaced* by :func:`fill_ramp`, so nothing the
    constructor's RNG produced survives; the three fitted-constant buffer
    families of :data:`FILLED_BUFFER_PREFIXES` are *perturbed* by
    :data:`BUFFER_FILL_SCALE` times the same ramp, which keeps them physical
    (positive radii, positive ZBL coefficients) while moving them off the value
    a fresh construction would give — without that, asserting the checkpoint
    restored them would be vacuous.

    Args:
        potential: Model to fill in place.

    Raises:
        RuntimeError: If a buffer perturbation left the tensor where it was,
            which would silently hollow out the round-trip assertion.
    """
    with torch.no_grad():
        for name, parameter in potential.named_parameters():
            parameter.copy_(fill_ramp(name, parameter.numel()).reshape(parameter.shape))
        for name, buffer in potential.named_buffers():
            if not name.startswith(FILLED_BUFFER_PREFIXES):
                continue
            ramp = fill_ramp(name, buffer.numel()).reshape(buffer.shape)
            moved = buffer + BUFFER_FILL_SCALE * ramp
            if torch.equal(moved, buffer):
                raise RuntimeError(f"buffer {name!r} did not move — the fill asserts nothing")
            buffer.copy_(moved)


def source_potential() -> MACEPotential:
    """The tiny MatPES potential the synthetic checkpoint is dumped from."""
    spec = MACEMatpesSpec(
        atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **SPEC_KWARGS
    )
    potential = MACEPotential(spec, use_fallback=True)
    fill_potential(potential)
    return potential.eval()


def official_state(potential: MACEPotential) -> dict[str, torch.Tensor]:
    """Dump ``potential``'s weights under the official cueq key names.

    The inverse of the shipped remap, plus the entries a real converted
    checkpoint carries and molnex has no home for: the dropped scalars of
    :data:`OFFICIAL_ONLY_SCALARS` and one :data:`OUTPUT_MASK_KEY`. The
    cuEquivariance graph constants keep their names — they are unlisted in the
    table and skipped on the ``.graph.c`` rule.

    Args:
        potential: Source model.

    Returns:
        Official key → the same tensor objects.
    """
    renames = sorted(
        {**CARRIED_OFFICIAL_NAMES, **DROPPED_OFFICIAL_NAMES}.items(),
        key=lambda item: -len(item[0]),
    )
    state: dict[str, torch.Tensor] = {}
    for key, value in potential.state_dict().items():
        for molnex, official in renames:
            if key == molnex or key.startswith(molnex):
                key = official + (key[len(molnex) :] if key != molnex else "")
                break
        state[key] = value
    for key, value in OFFICIAL_ONLY_SCALARS.items():
        state[key] = torch.tensor(value, dtype=torch.float64)
    state[OUTPUT_MASK_KEY] = torch.ones(4, dtype=torch.float64)
    return state


def write_checkpoint(directory: Path, state: dict[str, torch.Tensor]) -> tuple[Path, Path]:
    """Write an official-style MatPES config json + cueq ``state_dict``.

    The config states the channel widths **only** through ``hidden_irreps`` /
    ``MLP_irreps``; ``num_features`` / ``max_hidden_l`` / ``mlp_dim`` are
    nowhere in the file, so a loader that assumed the shipped ``128 / 1 / 16``
    would build a model these weights do not fit.

    Args:
        directory: Destination (a ``tempfile`` directory).
        state: Officially named weights.

    Returns:
        ``(config_path, weights_path)``.
    """
    config_path = directory / "matpes_synthetic_config.json"
    config_path.write_text(
        json.dumps(
            {
                "atomic_numbers": ATOMIC_NUMBERS,
                "atomic_energies": ATOMIC_ENERGIES,
                "r_max": SPEC_KWARGS["r_max"],
                "num_bessel": SPEC_KWARGS["num_bessel"],
                "num_polynomial_cutoff": SPEC_KWARGS["num_polynomial_cutoff"],
                "max_ell": SPEC_KWARGS["l_max"],
                "num_interactions": SPEC_KWARGS["num_interactions"],
                "correlation": SPEC_KWARGS["correlation"],
                "hidden_irreps": HIDDEN_IRREPS,
                "MLP_irreps": MLP_IRREPS,
                "radial_MLP": SPEC_KWARGS["radial_mlp"],
                "atomic_inter_scale": CONFIG_SCALE,
                "atomic_inter_shift": CONFIG_SHIFT,
            },
            indent=2,
        )
    )
    weights_path = directory / "matpes_synthetic_cueq_state.pt"
    torch.save(state, weights_path)
    return config_path, weights_path


def make_batch() -> TensorDict:
    """Post-collate batch for :data:`POSITIONS` (atoms / edges / graphs).

    A fresh object every call: ``forward`` writes in place and swaps
    ``atoms.pos`` for its own differentiation leaf.

    Returns:
        A ``TensorDict`` with ``edge_diff = pos[target] - pos[source]`` and
        ``edge_dist = ‖edge_diff‖``.
    """
    pos = POSITIONS.clone()
    edge_diff = pos[EDGE_INDEX[:, 1]] - pos[EDGE_INDEX[:, 0]]
    return TensorDict(
        atoms=TensorDict(Z=Z, pos=pos, batch=BATCH, batch_size=[pos.shape[0]]),
        edges=TensorDict(
            edge_index=EDGE_INDEX,
            edge_diff=edge_diff,
            edge_dist=edge_diff.norm(dim=-1),
            batch_size=[EDGE_INDEX.shape[0]],
        ),
        graphs=TensorDict(
            num_atoms=torch.tensor([pos.shape[0]], dtype=torch.long), batch_size=[NUM_GRAPHS]
        ),
        batch_size=[],
    )


def energy_forces(potential: MACEPotential) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy ``(B,)`` in eV and forces ``(N, 3)`` in eV/Å for the cluster."""
    out = potential(make_batch())
    return out["graphs", "energy"].detach(), out["atoms", "forces"].detach()


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


class Checker:
    """Collects every deviation so one run reports all failures, not the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def close(
        self,
        name: str,
        got: torch.Tensor,
        want: list[float] | list[list[float]],
        atol: float,
    ) -> None:
        """Assert ``max|got - want| <= atol`` against a literal nested list."""
        expected = torch.tensor(want, dtype=torch.float64)
        if got.shape != expected.shape:
            self.failures.append(f"{name}: shape {tuple(got.shape)} != {tuple(expected.shape)}")
            return
        deviation = float((got - expected).abs().max())
        if not deviation <= atol:
            self.failures.append(f"{name}: max|Δ| = {deviation:.6e} > atol {atol:.1e}")
        print(f"  {name:<38} max|Δ| {deviation:.3e}")

    def scalar(self, name: str, got: float, want: float, atol: float) -> None:
        """Assert ``|got - want| <= atol`` on a single number."""
        deviation = abs(got - want)
        if not deviation <= atol:
            self.failures.append(
                f"{name}: got {got!r}, want {want!r} (|Δ| = {deviation:.6e} > atol {atol:.1e})"
            )
        print(f"  {name:<38} |Δ| {deviation:.3e}")

    def truth(self, name: str, holds: bool, message: str) -> None:
        """Assert a boolean contract (key present / absent, error raised)."""
        if not holds:
            self.failures.append(f"{name}: {message}")
        print(f"  {name:<38} {'ok' if holds else 'FAILED'}")


def check_official_dialect(
    checker: Checker, source: MACEPotential, state: dict[str, torch.Tensor]
) -> None:
    """The synthetic checkpoint really is written in the official dialect.

    Two table-level legs against the shipped
    :data:`molzoo.mace.MATPES_KEY_REMAP` (so a wrong entry there cannot cancel
    itself out in the round trip), then two behavioural ones through the pure
    :meth:`molzoo.mace.CheckpointRemap.rename`: everything this file wrote as
    droppable leaves no trace, and everything that survives has a home.

    Args:
        checker: Failure collector.
        source: The model the checkpoint was dumped from.
        state: The officially named checkpoint contents.
    """
    print("Official key dialect (hand-written inverse vs molzoo.mace.MATPES_KEY_REMAP)")

    wrong = [
        f"{official} → {MATPES_KEY_REMAP.get(official)!r}, want {molnex!r}"
        for molnex, official in CARRIED_OFFICIAL_NAMES.items()
        if MATPES_KEY_REMAP.get(official) != molnex
    ]
    checker.truth("dialect.carried_entries_agree", not wrong, f"shipped table disagrees: {wrong}")

    carried = {value for value in MATPES_KEY_REMAP.values() if value is not None}
    checker.truth(
        "dialect.covers_every_carried_family",
        carried == set(CARRIED_OFFICIAL_NAMES),
        f"shipped table carries {sorted(carried)}, this file knows "
        f"{sorted(CARRIED_OFFICIAL_NAMES)} — a new family needs a golden",
    )

    renamed = set(MATPES_REMAP.rename(state))
    droppable = {
        *OFFICIAL_ONLY_SCALARS,
        OUTPUT_MASK_KEY,
        *(key for key in state if ".graph.c" in key),
        *(key for key in state if key.startswith(tuple(DROPPED_OFFICIAL_NAMES.values()))),
    }
    checker.truth(
        "dialect.droppable_keys_vanish",
        not droppable & renamed,
        f"expected no trace of {sorted(droppable & renamed)} after rename",
    )
    checker.truth(
        "dialect.every_survivor_has_a_home",
        not renamed - set(source.state_dict()),
        f"renamed keys with no home: {sorted(renamed - set(source.state_dict()))}",
    )


def check_roundtrip(checker: Checker, source: MACEPotential, built: MACEPotential) -> None:
    """``from_checkpoint`` reconstructs the source model tensor for tensor."""
    print("\nCheckpoint round trip (from_checkpoint vs the model it was dumped from)")
    reference = source.state_dict()
    produced = built.state_dict()

    checker.truth(
        "roundtrip.same_keys",
        set(produced) == set(reference),
        f"only in checkpoint model: {sorted(set(produced) - set(reference))}; "
        f"only in source: {sorted(set(reference) - set(produced))}",
    )
    differing = [name for name, want in reference.items() if not torch.equal(produced[name], want)]
    checker.truth(
        "roundtrip.every_tensor_equal",
        not differing,
        f"{len(differing)} tensor(s) not bit-identical: {differing[:5]}",
    )

    # The widths were parsed out of "16x0e+16x1o" / "8x0e"; had they been
    # assumed, the shapes below would not have matched and the load would have
    # raised on shape mismatch long before this line.
    checker.truth(
        "roundtrip.widths_came_from_irreps",
        tuple(built.node_embedding.weight.shape) == (1, len(ATOMIC_NUMBERS) * NUM_FEATURES)
        and built.readouts[-1].linear_2.weight.numel() == MLP_DIM,
        "the derived channel widths do not match the config's irreps strings",
    )

    scale = float(built.scale_shift.scale)
    shift = float(built.scale_shift.shift)
    checker.scalar("checkpoint.scale_shift.scale", scale, SCALE_GOLDEN, 0.0)
    checker.scalar("checkpoint.scale_shift.shift", shift, SHIFT_GOLDEN, 0.0)
    checker.truth(
        "checkpoint.beat_the_config_scale",
        scale != CONFIG_SCALE and shift != CONFIG_SHIFT,
        "scale/shift equal the config values — the checkpoint's own numbers were not loaded",
    )


def check_energy_forces(checker: Checker, source: MACEPotential, built: MACEPotential) -> None:
    """Energy and forces of the loaded model, against the goldens and the source."""
    print("\nEnergy / forces of the loaded model (fp64, CPU, use_fallback=True)")
    energy, forces = energy_forces(built)

    checker.close("loaded.energy", energy, ENERGY, ENERGY_ATOL)
    checker.close("loaded.forces", forces, FORCES, FORCE_ATOL)
    checker.scalar("loaded.force_abs_sum", float(forces.abs().sum()), FORCE_ABS_SUM, FORCE_ATOL)
    checker.scalar("loaded.net_force", float(forces.sum(0).abs().max()), 0.0, NET_FORCE_ATOL)

    source_energy, source_forces = energy_forces(source)
    checker.scalar(
        "loaded.vs_source.energy",
        float((energy - source_energy).abs().max()),
        0.0,
        ROUNDTRIP_ATOL,
    )
    checker.scalar(
        "loaded.vs_source.forces",
        float((forces - source_forces).abs().max()),
        0.0,
        ROUNDTRIP_ATOL,
    )


def check_unhoused_key(
    checker: Checker, built: MACEPotential, state: dict[str, torch.Tensor]
) -> None:
    """The ``on_unexpected`` knob: MatPES refuses, the lenient policy reports.

    One leg of the strictness doctrine, cheaply: a checkpoint key with no home
    means the mapping is stale, and loading the rest would leave a model that
    runs, looks sane and is quietly wrong (that doctrine, and the two incidents
    behind it, are quoted in the :mod:`molzoo.mace.checkpoint` docstring).

    Args:
        checker: Failure collector.
        built: The loaded model, re-checked afterwards — the refusal must not
            have left a half-loaded model behind.
        state: The officially named checkpoint contents; a tampered *copy* is
            what reaches the remap.
    """
    print("\nStrictness doctrine (a checkpoint key with no home)")
    tampered = dict(state)
    tampered[MYSTERY_KEY] = torch.zeros(3, dtype=torch.float64)

    message = ""
    try:
        MATPES_REMAP.load(built, tampered)
    except RuntimeError as error:
        message = str(error)
    checker.truth(
        "unhoused.raises_no_home",
        "no home" in message and MYSTERY_KEY in message,
        f"expected a RuntimeError naming the key, got {message!r}",
    )

    _, unexpected = CheckpointRemap(MATPES_KEY_REMAP, on_unexpected="return").load(built, tampered)
    checker.truth(
        "unhoused.returned_under_lenient_policy",
        unexpected == [MYSTERY_KEY],
        f"expected [{MYSTERY_KEY!r}], got {unexpected}",
    )

    # Neither call may have damaged the model: the refusal happens before any
    # tensor is placed, and the lenient reload puts back the same weights.
    energy, forces = energy_forces(built)
    checker.close("unhoused.energy_unchanged", energy, ENERGY, ENERGY_ATOL)
    checker.close("unhoused.forces_unchanged", forces, FORCES, FORCE_ATOL)


def main() -> int:
    """Build a synthetic checkpoint, read it back, and check every golden."""
    checker = Checker()
    source = source_potential()
    state = official_state(source)
    check_official_dialect(checker, source, state)

    with tempfile.TemporaryDirectory(prefix="molnex-mace-checkpoint-") as directory:
        config_path, weights_path = write_checkpoint(Path(directory), state)
        built = MACEPotential.from_checkpoint(config_path, weights_path, use_fallback=True).eval()

    check_roundtrip(checker, source, built)
    check_energy_forces(checker, source, built)
    check_unhoused_key(checker, built, state)

    if checker.failures:
        print("\nFAILED — the MACE checkpoint round trip no longer reproduces the goldens:")
        for failure in checker.failures:
            print(f"  {failure}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
