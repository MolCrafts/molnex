"""Tests for molzoo.mace.checkpoint — the official-checkpoint key tables.

Each family's table is exercised through the shared
:class:`molzoo.checkpoint.CheckpointRemap` with :data:`MACE_SKIP`: a round trip
of a model's own ``state_dict`` renamed into official cueq names — and, for
the per-element maps, put back into cueq's flat layout
(:func:`tests.test_molzoo.test_mace.conftest.official_layout`) — and loaded
back into a differently initialised model. The inverse name tables below are
written out **by hand** rather than derived from the tables under test, so a
wrong entry in the production table cannot cancel itself out. Generic remap
behaviour (strictness, reshapes, policies, the type-driven weight rule) is
asserted once in ``tests/test_molzoo/test_checkpoint.py``; the Polar table's
round trip sits in ``test_polar.py`` beside the blocks it names.

The asset-gated cases load the real official weights
(:data:`WEIGHTS_DIR_ENV`) and are ``slow``.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import cuequivariance as cue
import pytest
import torch
from tensordict import TensorDict

from molzoo.checkpoint import CheckpointRemap
from molzoo.mace.checkpoint import (
    MACE_SKIP,
    element_linear_weight,
    les_keys,
    matpes_keys,
    off_keys,
    omol_keys,
    one_hot_weight,
)
from molzoo.mace.layers import ElementLinear, OneHotLinear
from molzoo.mace.potential import MaceLes, MaceMatpes, MaceOff, MaceOmol, MacePotential
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec, MACESpec
from tests.test_molzoo.test_mace.conftest import official_layout

matpes_remap = CheckpointRemap(matpes_keys, skip=MACE_SKIP)
omol_remap = CheckpointRemap(omol_keys, on_unexpected="return", skip=MACE_SKIP)

#: Model name → official cueq name: the hand-written inverse of ``matpes_keys``.
MATPES_OFFICIAL_NAMES: dict[str, str] = {
    "node_embedding.": "node_embedding.linear.",
    "bessel.freqs": "radial_embedding.bessel_fn.bessel_weights",
    "distance_transform.": "radial_embedding.distance_transform.",
    "pair_repulsion.": "pair_repulsion_fn.",
    "z_table": "atomic_numbers",
}

#: Inverse of ``omol_keys``. ``readout.`` is the single final readout molnex
#: holds where the official checkpoint numbers a one-entry ``readouts`` list.
OMOL_OFFICIAL_NAMES: dict[str, str] = {
    "node_embedding.": "node_embedding.linear.",
    "embedding_readout.": "embedding_readout.linear.",
    "readout.": "readouts.0.",
    "bessel.freqs": "radial_embedding.bessel_fn.bessel_weights",
    "z_table": "atomic_numbers",
}

#: Inverse of ``off_keys`` (no ZBL, no transform entries).
OFF_OFFICIAL_NAMES: dict[str, str] = {
    "node_embedding.": "node_embedding.linear.",
    "bessel.freqs": "radial_embedding.bessel_fn.bessel_weights",
    "z_table": "atomic_numbers",
}

#: Inverse of ``les_keys``: MACE-OFF plus the latent-charge readouts.
LES_OFFICIAL_NAMES: dict[str, str] = {**OFF_OFFICIAL_NAMES, "charge_readouts.": "les_readouts."}

#: Official name of ``bessel.freqs`` — an ``nn.Parameter`` ending in neither
#: ``.weight`` nor ``.bias``; the table must carry it.
BESSEL_OFFICIAL_KEY = "radial_embedding.bessel_fn.bessel_weights"

#: Environment variable pointing at the offline official-weights directory.
WEIGHTS_DIR_ENV = "MOLNEX_MACE_WEIGHTS_DIR"

#: Hard-coded goldens: the dimensions of ``MACE-matpes-r2scan-omat-ft`` as
#: published (``hidden_irreps="128x0e+128x1o"``, ``MLP_irreps="16x0e"``).
OFFICIAL_NUM_FEATURES = 128
OFFICIAL_MAX_HIDDEN_L = 1
OFFICIAL_MLP_DIM = 16

#: Offline OMol asset inside :data:`WEIGHTS_DIR_ENV`: the cueq ``state_dict`` of
#: ``MACE-omol-0-extra-large-1024``, dumped out of tree from the pickled
#: ``OMOL-cueq.model`` (reading that needs ``mace`` and ``e3nn``, which
#: CLAUDE.md forbids); the converter sits beside the asset.
OMOL_WEIGHTS_FILE = "omol_cueq_state.pt"

#: Hard-coded golden: every ``nn.Parameter`` tensor of the OMol model the
#: official checkpoint has to fill.
OMOL_PARAMETER_COUNT = 104

#: Hard-coded golden: the checkpoint keys with no home in molnex's OMol model.
#: The single-head ``omol`` checkpoint shipped has none; any future key growing
#: into this list is a deliberate decision, not a silent one.
OMOL_UNEXPECTED_KEYS: list[str] = []

#: Hard-coded goldens (eV, eV/Å): energy, ``max|F|`` and ``F[0]`` of the
#: ``omol_cluster`` fixture under the official OMol weights, captured at
#: ``OMP_NUM_THREADS=1`` (see :class:`TestOfficialOMolWeights`).
OMOL_GOLDEN_ENERGY = -3122.454566006894
OMOL_GOLDEN_MAX_FORCE = 5.397722165018621
OMOL_GOLDEN_FIRST_FORCE = (-5.397722165018621, -3.9298849841895853, -3.931997122634094)

#: The repo's numerical tolerances, not exact ones: the CPU reduction order
#: inside the cuEquivariance fallback follows ``torch.get_num_threads()``, and
#: ``{1, 4, 16}`` threads and a 48-thread default disagree by ~3e-8 eV.
OMOL_ENERGY_TOL = 1e-6
OMOL_FORCE_TOL = 1e-4


def _official_weights_absent(filename: str) -> bool:
    """Whether the gated offline asset ``filename`` is unavailable."""
    directory = os.environ.get(WEIGHTS_DIR_ENV)
    return directory is None or not (Path(directory) / filename).is_file()


def _official_state(model: torch.nn.Module, names: Mapping[str, str]) -> dict[str, torch.Tensor]:
    """A model's own ``state_dict`` as an official cueq checkpoint would carry it.

    Official key names, and the per-element maps' weights in cueq's flat
    layout (:func:`official_layout`).

    Args:
        model: Source model.
        names: Molnex prefix → official prefix (longest match wins).

    Returns:
        The tensors under the names an official checkpoint would use.
    """
    ordered = sorted(names.items(), key=lambda item: -len(item[0]))
    out: dict[str, torch.Tensor] = {}
    for key, value in official_layout(model, model.state_dict()).items():
        # A cueq checkpoint holds tensors only; ``_extra_state`` is molnex's
        # own module metadata and never appears in the artifact being faked.
        if not torch.is_tensor(value):
            continue
        for prefix, official in ordered:
            if key == prefix or key.startswith(prefix):
                key = official + (key[len(prefix) :] if key != prefix else "")
                break
        out[key] = value
    return out


def _filled(product: type[MacePotential], spec: MACESpec, seed: int) -> MacePotential:
    """A tiny product whose every parameter is non-zero and seed-specific.

    Two models built from the same spec with different seeds must disagree in
    every parameter, or a round-trip assertion could pass on tensors nobody
    moved.
    """
    torch.manual_seed(seed)
    model = product(spec)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(generator=generator)
    return model


def _assert_roundtrip(
    product: type[MacePotential],
    spec: MACESpec,
    keys: Mapping[str, str | None],
    names: Mapping[str, str],
) -> None:
    """Official names load back bit-for-bit into a fresh model of ``product``."""
    source = _filled(product, spec, seed=0)
    target = _filled(product, spec, seed=1)
    CheckpointRemap(keys, on_unexpected="return", skip=MACE_SKIP).load(
        target, _official_state(source, names)
    )
    for (name, want), (_, got) in zip(
        source.named_parameters(), target.named_parameters(), strict=True
    ):
        assert torch.equal(want, got), name


class TestCheckpointRemap:
    """The MACE key tables through the shared remap."""

    def test_rename_drops_the_skip_substrings_and_renames(self) -> None:
        """``.rename`` is pure, drops :data:`MACE_SKIP` entries and ``None`` entries, renames."""
        weight = torch.zeros(2)
        freqs = torch.ones(4)
        official = {
            "node_embedding.linear.weight": weight,
            BESSEL_OFFICIAL_KEY: freqs,
            "node_embedding.linear.f.m.graphs.0.graph.c0": torch.zeros(()),
            "interactions.0.conv_tp.output_mask": torch.zeros(3),
            "products.0.symmetric_contractions.projection": torch.zeros(3),
            "interactions.0.linear.weight": weight,
            "r_max": torch.tensor(6.0),
        }
        snapshot = dict(official)

        renamed = matpes_remap.rename(official)

        assert renamed is not official
        assert all(official[key] is value for key, value in snapshot.items())
        assert set(renamed) == {
            "node_embedding.weight",
            "bessel.freqs",
            "interactions.0.linear.weight",
        }
        assert renamed["bessel.freqs"] is freqs

    def test_roundtrip_restores_every_parameter_matpes(self, tiny_matpes_spec) -> None:
        _assert_roundtrip(MaceMatpes, tiny_matpes_spec, matpes_keys, MATPES_OFFICIAL_NAMES)

    def test_roundtrip_restores_every_parameter_omol(self, tiny_omol_spec) -> None:
        _assert_roundtrip(MaceOmol, tiny_omol_spec, omol_keys, OMOL_OFFICIAL_NAMES)

    def test_roundtrip_restores_every_parameter_off(self, tiny_off_spec) -> None:
        _assert_roundtrip(MaceOff, tiny_off_spec, off_keys, OFF_OFFICIAL_NAMES)

    def test_roundtrip_restores_every_parameter_les(self, tiny_les_spec) -> None:
        _assert_roundtrip(MaceLes, tiny_les_spec, les_keys, LES_OFFICIAL_NAMES)


def test_element_linear_weight_refuses_an_equal_numel_in_another_shape() -> None:
    """The boundary takes the product's one flat row only — the module's own layout is not it."""
    linear = ElementLinear(
        cue.Irreps("O3", "4x0e"),
        cue.Irreps("O3", "3x0e"),
        cue.Irreps("O3", "5x0e"),
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="flat weight"):
        element_linear_weight(linear, linear.weight.detach())


def test_one_hot_weight_refuses_an_equal_numel_in_another_shape() -> None:
    """The boundary takes the linear's one flat row only — the ``(C, F)`` table is not it."""
    lookup = OneHotLinear(cue.Irreps("O3", "3x0e"), cue.Irreps("O3", "5x0e"), dtype=torch.float64)
    with pytest.raises(ValueError, match="flat weight"):
        one_hot_weight(lookup, lookup.weight.detach())


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get(WEIGHTS_DIR_ENV) is None,
    reason=f"{WEIGHTS_DIR_ENV} unset — the converted official weights are an offline asset",
)
def test_official_checkpoint_bit_parity() -> None:
    """The real MatPES checkpoint loads identically both ways (``science``).

    ``MaceMatpes.load`` against the hand-written "construct + remap" path on
    ``MACE-matpes-r2scan-omat-ft`` (converted offline with
    ``mace.cli.convert_e3nn_cueq``; never imported here). Bit parity, not a
    tolerance. The oracle arm states the published ``128 / 1 / 16`` dimensions
    literally, so ``load`` has to derive them from the irreps strings unaided.
    """
    directory = Path(os.environ[WEIGHTS_DIR_ENV])
    config_path = directory / "matpes_r2scan_config.json"
    weights_path = directory / "matpes_r2scan_cueq_state.pt"
    official = json.loads(config_path.read_text())

    manual = MaceMatpes(
        MACEMatpesSpec(
            atomic_numbers=official["atomic_numbers"],
            atomic_energies=official["atomic_energies"],
            r_max=official["r_max"],
            num_bessel=official["num_bessel"],
            num_polynomial_cutoff=official["num_polynomial_cutoff"],
            l_max=official["max_ell"],
            num_features=OFFICIAL_NUM_FEATURES,
            max_hidden_l=OFFICIAL_MAX_HIDDEN_L,
            num_interactions=official["num_interactions"],
            correlation=official["correlation"],
            mlp_dim=OFFICIAL_MLP_DIM,
            radial_mlp=official["radial_MLP"],
            scale=official["atomic_inter_scale"],
            shift=official["atomic_inter_shift"],
        ),
    )
    matpes_remap.load(manual, torch.load(weights_path, map_location="cpu", weights_only=True))

    built = MaceMatpes.load(config_path)

    reference = manual.state_dict()
    produced = built.state_dict()
    assert set(produced) == set(reference)
    for name, want in reference.items():
        if name.endswith("_extra_state"):  # module metadata (a dict), not a tensor
            assert produced[name] == want, name
            continue
        assert torch.equal(produced[name], want), name


def _official_omol() -> tuple[MaceOmol, dict[str, torch.Tensor]]:
    """The full-size OMol model with the official weights, and that state.

    The four fields that vary per checkpoint — element table, ``E0`` table,
    ``scale``, ``shift`` — are read out of the checkpoint itself; every other
    field is the :class:`~molzoo.mace.spec.MACEOMolSpec` default, which the
    checkpoint corroborates (``r_max`` 6.0, ``num_interactions`` 3,
    ``num_bessel`` 8, ``num_polynomial_cutoff`` 5, ``num_features`` 1024,
    ``charge_classes`` / ``spin_classes`` 201 / 101, ``mlp_dim`` 16).
    """
    weights_path = Path(os.environ[WEIGHTS_DIR_ENV]) / OMOL_WEIGHTS_FILE
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    spec = MACEOMolSpec(
        atomic_numbers=state["atomic_numbers"].tolist(),
        atomic_energies=state["atomic_energies_fn.atomic_energies"].flatten().tolist(),
        scale=float(state["scale_shift.scale"]),
        shift=float(state["scale_shift.shift"]),
    )
    model = MaceOmol(spec, wrt=("atoms.pos",))
    omol_remap.load(model, state)
    return model.eval(), state


@pytest.mark.skipif(
    _official_weights_absent(OMOL_WEIGHTS_FILE),
    reason=(
        f"{WEIGHTS_DIR_ENV} unset or {OMOL_WEIGHTS_FILE} absent — "
        "the converted official OMol weights are an offline asset"
    ),
)
@pytest.mark.slow
class TestOfficialOMolWeights:
    """The official OMol checkpoint, loaded and evaluated in tree (``science``).

    A **stability lock**, not an upstream parity record: nothing here compares
    against ``mace-torch`` or ``e3nn``. The official weights load strictly, and
    the resulting surface is the one measured on this machine on 2026-08-09
    (``torch 2.12.1+cpu``, ``cuequivariance 0.10.0``, CPU, fp64,
    ``OMP_NUM_THREADS=1``; weights ``omol_cueq_state.pt`` sha256
    ``074c86154a1d709f…``). The extraction is corroborated at value level:
    ``radial_embedding.bessel_fn.bessel_weights`` sits 2.2120e-07 Å⁻¹ from the
    analytic ``nπ/r_max`` init.
    """

    def test_official_omol_load_fills_every_learnable(self) -> None:
        """Every ``nn.Parameter`` is covered; the unhoused keys are the snapshot."""
        model, state = _official_omol()

        renamed = omol_remap.rename(state)
        parameters = {name for name, _ in model.named_parameters()}

        assert len(parameters) == OMOL_PARAMETER_COUNT
        assert parameters <= set(renamed)
        assert sorted(set(renamed) - set(model.state_dict())) == OMOL_UNEXPECTED_KEYS

    def test_official_omol_energy_and_forces_match_the_goldens(
        self, omol_cluster: TensorDict
    ) -> None:
        """The loaded surface reproduces the captured energy and forces."""
        model, _ = _official_omol()

        result = model(omol_cluster)

        energy = result["graphs", "energy"]
        forces = result["atoms", "forces"]
        assert energy.shape == (1,)
        assert forces.shape == (len(omol_cluster["atoms", "Z"]), 3)
        assert energy.item() == pytest.approx(OMOL_GOLDEN_ENERGY, abs=OMOL_ENERGY_TOL)
        assert forces.abs().max().item() == pytest.approx(OMOL_GOLDEN_MAX_FORCE, abs=OMOL_FORCE_TOL)
        assert forces[0].tolist() == pytest.approx(
            list(OMOL_GOLDEN_FIRST_FORCE), abs=OMOL_FORCE_TOL
        )
