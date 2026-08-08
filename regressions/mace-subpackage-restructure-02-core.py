"""Public-API parity scenario for chain step mace-subpackage-restructure-02.

Builds both MACE foundation variants through the **new** core layer only —
``MACEMatpesSpec`` / ``MACEOMolSpec`` → ``MACEEncoder`` →
``molzoo.mace.geometry`` → the encoder's primitives — and composes the total
energy in this script, the way a caller has to (the library exposes primitives;
there is no ``compute_all`` façade). Every number is then checked against
literals captured from the flat ``MACEMatpes`` / ``MACEOMol`` models before the
restructure. Any deviation means the spec-driven backbone is not the same model
the flat variants were.

Goldens
-------
All literals in ``MATPES`` / ``OMOL`` below were captured from the **flat**
``molzoo.mace_matpes.MACEMatpes`` and ``molzoo.mace_omol.MACEOMol`` — the
in-repo self-oracle, deleted in chain step 07, hence hard-coded here and never
imported at run time. The capture script re-implemented each flat model's
``_compute_energy`` line by line to expose the intermediate scalars, and
asserted the recomposed total against the flat model's own public
``energy_forces(..., compute_forces=False)["energy"]`` (bit-equal) before
emitting the table. No third-party oracle, no network, no RNG-derived weights.

To re-capture while the flat variants still exist: copy this file, construct
``MACEMatpes`` / ``MACEOMol`` with the same kwargs (passing ``atomic_energies``
as a **float64** tensor — a float32 one shifts ``E0`` by ~1e-6 eV), apply
:func:`deterministic_weights`, inline each flat ``_compute_energy`` in place of
the ``MACEEncoder`` composition below, and print the scalars instead of
asserting.

    capture script : ad-hoc, per the recipe above (flat-variant decomposition)
    capture command: PYTHONPATH=src python capture_02_goldens.py
    commit         : 1ddd5ff (1ddd5ffe9772d490cbd48fd352d59456bc5b5683)
    torch          : 2.12.1+cpu
    date           : 2026-08-08
    device / dtype : CPU, float64 (``config.set_precision("fp64")``)
    oracle         : molzoo.mace_matpes.MACEMatpes / molzoo.mace_omol.MACEOMol
                     at 1ddd5ff — in-repo self-oracle, no third party
    observed       : new-vs-flat deviation 0.0 (bit-identical) on every entry,
                     and ``sorted(state_dict())`` equal for both variants

Why the energy total is not the only assertion
----------------------------------------------
Under the ``linspace(-0.1, 0.1)`` weight fill the MatPES interaction stack is
numerically tiny (node features ~5e-10, readout energies ~7e-12 eV), so its
total energy is dominated by ``E0`` and the ZBL term: a 1e-9 eV check on the
total alone would pass even with the whole interaction stack zeroed. The
per-layer feature sums-of-squares and per-layer readout energies are therefore
asserted **relatively** (1e-10) — those are the load-bearing assertions on the
interaction/product stack. They are ordinary products of small numbers, not
cancellations, so their relative accuracy is full double precision (verified
stable across ``OMP_NUM_THREADS`` 1/2/4 at capture time).

Run:
    PYTHONPATH=src python regressions/mace-subpackage-restructure-02-core.py
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import torch

from molix import config

config.set_precision("fp64")  # before any module construction: cuEq bakes in dtype

from molix.F.scatter import scatter_sum_compile_safe as scatter_sum  # noqa: E402
from molzoo.mace.encoder import MACEEncoder  # noqa: E402
from molzoo.mace.geometry import edge_lengths, edge_vectors  # noqa: E402
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec  # noqa: E402

#: Absolute tolerance on a total energy, in eV.
ENERGY_ATOL = 1e-9
#: Relative tolerance on the dimensionless per-layer checksums.
CHECKSUM_RTOL = 1e-10

# ---------------------------------------------------------------------------
# System: a fixed 5-atom cluster, all 20 ordered pairs as directed edges.
# Literal coordinates in Å — no RNG, so the goldens are reproducible anywhere.
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
Z = torch.tensor([1, 6, 8, 1, 1])
BATCH = torch.zeros(5, dtype=torch.long)
NUM_GRAPHS = 1
EDGE_INDEX = torch.tensor(  # (E, 2): [:, 0] = source, [:, 1] = target
    [(i, j) for i in range(5) for j in range(5) if i != j], dtype=torch.long
)

#: Tiny MatPES hyper-parameters (l_max=1, 16 channels) — CPU-fast, structurally
#: identical to the shipped model.
MATPES_KWARGS = dict(
    r_max=5.0,
    num_bessel=4,
    num_polynomial_cutoff=5,
    l_max=1,
    num_features=16,
    max_hidden_l=1,
    num_interactions=2,
    correlation=2,
    mlp_dim=8,
    radial_mlp=[8],
    use_fallback=True,  # CPU: the fused kernels need a GPU + the ops wheel
)
#: Tiny OMOL hyper-parameters. The flat ``MACEOMol`` had no ``use_fallback``
#: argument and hard-coded the fused path; on CPU without
#: ``cuequivariance-ops-torch`` that degrades to the same naive contraction, so
#: ``use_fallback=True`` here reproduces the flat oracle bit-for-bit (verified
#: at capture time against both settings).
OMOL_KWARGS = dict(
    r_max=5.0,
    num_bessel=4,
    num_polynomial_cutoff=5,
    l_max=1,
    num_features=16,
    num_interactions=2,
    correlation=2,
    mlp_dim=8,
    edge_channels=8,
    use_fallback=True,
)

# ---------------------------------------------------------------------------
# Goldens — flat-variant capture, see the module docstring.
# ---------------------------------------------------------------------------


class MatpesGolden(NamedTuple):
    """Flat ``MACEMatpes`` reference scalars for the system above."""

    energy: float  # eV, total
    e0: float  # eV, Σ E0[Z]
    zbl: float  # eV, Σ_i ZBL_i
    sumsq: tuple[float, ...]  # per-layer Σ h², dimensionless
    readouts: tuple[float, ...]  # eV, per-layer Σ_i readout_l(h_l)_i


class OMolGolden(NamedTuple):
    """Flat ``MACEOMol`` reference scalars for the system above."""

    energy: float  # eV, total
    e0: float  # eV, Σ E0[Z]
    emb: float  # eV, charge/spin embedding readout
    sumsq: tuple[float, ...]  # per-layer Σ h², dimensionless
    readout: float  # eV, Σ_i readout(h_last)_i (pre scale_shift)


MATPES = MatpesGolden(
    energy=-3110.794351427106,
    e0=-3110.7999999999997,
    zbl=0.0056485728867009056,
    sumsq=(1.6644954580828804e-17, 1.4248167631489696e-21),
    readouts=(6.852674614296589e-12, 3.8986176107825314e-14),
)
OMOL = OMolGolden(
    energy=-3111.2905857515216,
    e0=-3110.7999999999997,
    emb=-0.04147213673688674,
    sumsq=(0.004897576797442622, 2.02047140946756e-06),
    readout=-0.4491136147850401,
)


def deterministic_weights(model: torch.nn.Module) -> None:
    """Overwrite every parameter with a fixed ramp — no RNG anywhere.

    Applied identically to the flat oracle at capture time and to
    :class:`~molzoo.mace.encoder.MACEEncoder` here; ``state_dict`` key parity is
    what makes "identically" well defined. Buffers are left alone (they carry
    the element table and the cuEquivariance graph constants).

    Args:
        model: Module whose parameters are replaced in place.
    """
    with torch.no_grad():
        for _, parameter in sorted(model.named_parameters()):
            parameter.data.copy_(
                torch.linspace(-0.1, 0.1, parameter.numel(), dtype=torch.float64).reshape(
                    parameter.shape
                )
            )


class Checker:
    """Collects deviations of observed scalars from the embedded goldens."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def absolute(self, name: str, observed: float, golden: float, atol: float) -> None:
        """Assert ``|observed - golden| <= atol`` (energies, in eV)."""
        deviation = abs(observed - golden)
        if deviation > atol:
            self.failures.append(
                f"{name}: got {observed!r}, want {golden!r} "
                f"(deviation {deviation:.3e} > atol {atol:.1e})"
            )
        print(f"  {name:<26} deviation {deviation:.3e}")

    def relative(self, name: str, observed: float, golden: float, rtol: float) -> None:
        """Assert ``|observed - golden| <= rtol * |golden|`` (checksums)."""
        deviation = abs(observed - golden)
        if deviation > rtol * abs(golden):
            self.failures.append(
                f"{name}: got {observed!r}, want {golden!r} "
                f"(relative deviation {deviation / abs(golden):.3e} > rtol {rtol:.1e})"
            )
        print(f"  {name:<26} relative deviation {deviation / abs(golden):.3e}")


def check_matpes(checker: Checker) -> None:
    """Compose the MatPES total energy from primitives and verify it.

    Mirrors ``MACEMatpes._compute_energy``:
    ``E0 + scale_shift(ZBL + Σ_layers readout_l(h_l))``, scattered per graph.
    """
    print("MACEMatpes (density interactions, Agnesi transform, ZBL, per-layer readouts)")
    spec = MACEMatpesSpec(
        atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **MATPES_KWARGS
    )
    torch.manual_seed(0)  # module construction may draw; the fill below overwrites it
    encoder = MACEEncoder(spec).eval()
    deterministic_weights(encoder)
    encoder.validate_elements(Z)

    vectors = edge_vectors(POSITIONS, EDGE_INDEX)
    lengths = edge_lengths(vectors)
    node_attrs = encoder.node_attrs(Z, POSITIONS.dtype)
    edge_attrs = encoder.angular_features(vectors)
    edge_feats, cutoff = encoder.radial_features(lengths, Z, EDGE_INDEX)

    e0 = scatter_sum(encoder.atomic_energies(Z), BATCH, NUM_GRAPHS)
    node_energies = encoder.pair_repulsion(lengths, Z, EDGE_INDEX)
    checker.absolute("matpes.e0", float(e0[0]), MATPES.e0, ENERGY_ATOL)
    checker.absolute("matpes.zbl", float(node_energies.sum()), MATPES.zbl, ENERGY_ATOL)

    layers = encoder.layer_features(
        node_feats=encoder.initial_node_features(node_attrs),
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        edge_feats=edge_feats,
        edge_index=EDGE_INDEX,
        cutoff=cutoff,
    )
    if len(layers) != len(MATPES.sumsq):
        checker.failures.append(
            f"matpes.layer_features: got {len(layers)} layers, want {len(MATPES.sumsq)}"
        )
        return
    for i, features in enumerate(layers):
        checker.relative(
            f"matpes.sumsq[{i}]", float((features**2).sum()), MATPES.sumsq[i], CHECKSUM_RTOL
        )
        head = encoder.readouts[i](features).squeeze(-1)
        checker.relative(
            f"matpes.readout[{i}]", float(head.sum()), MATPES.readouts[i], CHECKSUM_RTOL
        )
        node_energies = node_energies + head

    energy = e0 + scatter_sum(encoder.scale_shift(node_energies), BATCH, NUM_GRAPHS)
    checker.absolute("matpes.energy", float(energy[0]), MATPES.energy, ENERGY_ATOL)


def check_omol(checker: Checker) -> None:
    """Compose the OMOL total energy from primitives and verify it.

    Mirrors ``MACEOMol._compute_energy``: ``E0 + Σ embedding_readout(h_0 +
    conditioning) + scale_shift(readout(h_last))``, scattered per graph.
    """
    print("\nMACEOMol (residual interactions, charge/spin conditioning, final readout)")
    spec = MACEOMolSpec(
        atomic_numbers=ATOMIC_NUMBERS, atomic_energies=ATOMIC_ENERGIES, **OMOL_KWARGS
    )
    torch.manual_seed(0)
    encoder = MACEEncoder(spec).eval()
    deterministic_weights(encoder)
    encoder.validate_elements(Z)

    total_charge = torch.zeros(NUM_GRAPHS, dtype=torch.long)
    total_spin = torch.ones(NUM_GRAPHS, dtype=torch.long)  # OMOL: 1 = closed-shell singlet

    vectors = edge_vectors(POSITIONS, EDGE_INDEX)
    lengths = edge_lengths(vectors, keepdim=True)  # OMOL's own (E, 1) shape
    node_attrs = encoder.node_attrs(Z, POSITIONS.dtype)
    edge_attrs = encoder.angular_features(vectors)
    edge_feats, cutoff = encoder.radial_features(lengths.squeeze(-1), Z, EDGE_INDEX)

    e0 = scatter_sum(encoder.atomic_energies(Z), BATCH, NUM_GRAPHS)
    checker.absolute("omol.e0", float(e0[0]), OMOL.e0, ENERGY_ATOL)

    node_feats = encoder.initial_node_features(node_attrs) + encoder.conditioning(
        BATCH, total_spin=total_spin, total_charge=total_charge
    )
    embedding_energy = scatter_sum(
        encoder.embedding_readout(node_feats).squeeze(-1), BATCH, NUM_GRAPHS
    )
    checker.absolute("omol.emb", float(embedding_energy[0]), OMOL.emb, ENERGY_ATOL)

    layers = encoder.layer_features(
        node_feats=node_feats,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        edge_feats=edge_feats,
        edge_index=EDGE_INDEX,
        cutoff=cutoff,
    )
    if len(layers) != len(OMOL.sumsq):
        checker.failures.append(
            f"omol.layer_features: got {len(layers)} layers, want {len(OMOL.sumsq)}"
        )
        return
    for i, features in enumerate(layers):
        checker.relative(
            f"omol.sumsq[{i}]", float((features**2).sum()), OMOL.sumsq[i], CHECKSUM_RTOL
        )

    node_energies = encoder.readout(layers[-1]).squeeze(-1)
    checker.relative("omol.readout", float(node_energies.sum()), OMOL.readout, CHECKSUM_RTOL)

    energy = (
        e0 + embedding_energy + scatter_sum(encoder.scale_shift(node_energies), BATCH, NUM_GRAPHS)
    )
    checker.absolute("omol.energy", float(energy[0]), OMOL.energy, ENERGY_ATOL)


def main() -> int:
    """Rebuild both variants on the new core layer and verify every golden."""
    checker = Checker()
    with torch.no_grad():
        check_matpes(checker)
        check_omol(checker)

    if checker.failures:
        print("\nFAILED — the spec-driven MACEEncoder is not the flat variants' model:")
        for failure in checker.failures:
            print(f"  {failure}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
