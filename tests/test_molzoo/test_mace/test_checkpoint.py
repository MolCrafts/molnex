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
(:data:`WEIGHTS_DIR_ENV`) and are ``slow``. Per shipped family they assert
two facts: the checkpoint fills its product strictly, and the product's
energy and forces on the fixture cluster are **upstream's** — ``mace-torch``'s
own evaluation of the same checkpoint on the same cluster, saved beside the
weights as ``<family>_cluster_reference.pt`` by the asset's ``reference.py``
(fp64, CPU; ``REFERENCE.md`` there pins the versions). Nothing here is a
number molnex measured on itself.

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

import pytest
import torch
from tensordict import TensorDict

import molrep.eq as meq
from molrep.encoders.mace.layers import ElementLinear, OneHotLinear
from molzoo.checkpoint import CheckpointRemap
from molzoo.mace.checkpoint import (
    MACE_CONVERSIONS,
    MACE_SKIP,
    element_linear_weight,
    les_keys,
    matpes_keys,
    off_keys,
    omol_keys,
    one_hot_weight,
)
from molzoo.mace.potential import MaceLes, MaceMatpes, MaceOff, MaceOmol, MacePotential
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec, MACESpec
from tests.test_molzoo.test_mace.conftest import DTYPE, official_layout

matpes_remap = CheckpointRemap(matpes_keys, skip=MACE_SKIP, conversions=MACE_CONVERSIONS)
omol_remap = CheckpointRemap(
    omol_keys, on_unexpected="return", skip=MACE_SKIP, conversions=MACE_CONVERSIONS
)

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

#: Differentiate the energy with respect to the positions: the op writes
#: ``atoms.forces``, which the upstream references carry.
FORCES: tuple[str, ...] = ("atoms.pos",)

#: Offline MatPES assets inside :data:`WEIGHTS_DIR_ENV`: the config json of
#: ``MACE-matpes-r2scan-omat-ft`` with its cueq ``state_dict`` beside it (both
#: converted out of tree with ``mace.cli.convert_e3nn_cueq``), and upstream's
#: evaluation on the fixture cluster.
MATPES_CONFIG_FILE = "matpes_r2scan_config.json"
MATPES_WEIGHTS_FILE = "matpes_r2scan_cueq_state.pt"
MATPES_REFERENCE_FILE = "matpes_cluster_reference.pt"

#: Hard-coded goldens: the dimensions of ``MACE-matpes-r2scan-omat-ft`` as
#: published (``hidden_irreps="128x0e+128x1o"``, ``MLP_irreps="16x0e"``).
OFFICIAL_NUM_FEATURES = 128
OFFICIAL_MAX_HIDDEN_L = 1
OFFICIAL_MLP_DIM = 16

#: Offline OMol assets: the cueq ``state_dict`` of
#: ``MACE-omol-0-extra-large-1024``, dumped out of tree from the pickled
#: ``OMOL-cueq.model`` (reading that needs ``mace`` and ``e3nn``, which
#: CLAUDE.md forbids; the converter sits beside the asset), and upstream's
#: evaluation on the conditioned fixture cluster.
OMOL_WEIGHTS_FILE = "omol_cueq_state.pt"
OMOL_REFERENCE_FILE = "omol_cluster_reference.pt"

#: Hard-coded golden: every ``nn.Parameter`` tensor of the OMol model the
#: official checkpoint has to fill.
OMOL_PARAMETER_COUNT = 104

#: Hard-coded golden: the checkpoint keys with no home in molnex's OMol model.
#: The single-head ``omol`` checkpoint shipped has none; any future key growing
#: into this list is a deliberate decision, not a silent one.
OMOL_UNEXPECTED_KEYS: list[str] = []

#: Offline MACE-OFF24 medium assets: the config json upstream's own
#: ``extract_config_mace_model`` writes (``interaction_cls_first`` /
#: ``interaction_cls``, no block list; the asset's ``convert_config.py``),
#: its cueq ``state_dict`` beside it, and upstream's evaluation on the cluster.
OFF_CONFIG_FILE = "off24_medium_config.json"
OFF_WEIGHTS_FILE = "off24_medium_cueq_state.pt"
OFF_REFERENCE_FILE = "off24_cluster_reference.pt"

#: Hard-coded golden: every ``nn.Parameter`` tensor of MACE-OFF24 medium.
OFF_PARAMETER_COUNT = 23

#: Offline MACE-LES-OFF small assets: the same json dialect plus the ``les``
#: block ``MaceArchive`` reads off a pickled module, the cueq ``state_dict``
#: (fp32 — the model was trained in fp32), and upstream's evaluation.
LES_CONFIG_FILE = "les_off_small_config.json"
LES_WEIGHTS_FILE = "les_off_small_cueq_state.pt"
LES_REFERENCE_FILE = "les_cluster_reference.pt"

#: Hard-coded golden: every ``nn.Parameter`` tensor of MACE-LES-OFF small —
#: MACE-OFF's plus one latent-charge readout per layer.
LES_PARAMETER_COUNT = 26

#: Upstream parity tolerances (eV, eV/Å) on the fixture cluster, one pair for
#: the four families. Measured against the ``mace-torch`` references (fp64,
#: CPU, ``OMP_NUM_THREADS=1``, 2026-10-17): MatPES ΔE 1.4e-14 / max |ΔF|
#: 6.0e-14, OFF24 0 / 5.5e-15, OMol 5.4e-8 / 2.2e-7, LES-OFF 2.4e-7 / 2.0e-6.
#: The fp64-native models agree to round-off. OMol's gap is the order of the
#: CPU reduction-order scatter of the cuEquivariance fallback, which follows
#: ``torch.get_num_threads()`` (``{1, 4, 16}`` threads and a 48-thread default
#: disagree by ~3e-8 eV). LES-OFF is the one fp32-trained model: upstream
#: evaluates its ``.double()``-ed pickle, whose non-learnable constants keep
#: their fp32 rounding (the Bessel prefactor, e3nn's Clebsch–Gordan and
#: contraction tables) where molnex rebuilds them exactly in fp64. An fp64
#: re-conversion of the cueq dump reproduces the gap to the digit, so the
#: dump is not it; copying upstream's fp32-rounded prefactor alone moves ΔE
#: to 1.2e-7 and max |ΔF| to 4.0e-6, so no single constant is it either. The
#: tolerances sit a factor 4–5 above the LES gap and orders above the rest.
UPSTREAM_ENERGY_TOL = 1e-6
UPSTREAM_FORCE_TOL = 1e-5


def _official_weights_absent(*filenames: str) -> bool:
    """Whether any of the gated offline assets ``filenames`` is unavailable."""
    directory = os.environ.get(WEIGHTS_DIR_ENV)
    return directory is None or not all((Path(directory) / name).is_file() for name in filenames)


def _absent_reason(*filenames: str) -> str:
    """The skip reason naming every asset a gated case reads."""
    return f"{WEIGHTS_DIR_ENV} unset or {' / '.join(filenames)} absent — an offline asset"


def _asset(filename: str) -> Path:
    """The path of an offline asset inside :data:`WEIGHTS_DIR_ENV`."""
    return Path(os.environ[WEIGHTS_DIR_ENV]) / filename


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
    model = product(spec, dtype=DTYPE)
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
    CheckpointRemap(
        keys, on_unexpected="return", skip=MACE_SKIP, conversions=MACE_CONVERSIONS
    ).load(target, _official_state(source, names))
    for (name, want), (_, got) in zip(
        source.named_parameters(), target.named_parameters(), strict=True
    ):
        assert torch.equal(want, got), name


def _assert_upstream_parity(model: MacePotential, frame: TensorDict, reference_file: str) -> None:
    """``model`` on ``frame`` reproduces the upstream evaluation saved as ``reference_file``.

    The reference's geometry and conditioning are checked against the frame
    first, so a drifted fixture fails on the fixture, not on the energy; its
    cutoff list must be the fixture's full edge set, or the two evaluations
    would not see the same graph.
    """
    reference = torch.load(_asset(reference_file), map_location="cpu", weights_only=True)
    assert torch.equal(reference["atomic_numbers"], frame["atoms", "Z"])
    assert torch.equal(reference["positions"], frame["atoms", "pos"])
    assert reference["cutoff_equals_full"]
    for column in ("total_charge", "total_spin"):
        if ("graphs", column) in frame.keys(include_nested=True):
            assert reference[column] == int(frame["graphs", column].item())

    result = model(frame)

    energy = result["graphs", "energy"]
    forces = result["atoms", "forces"]
    assert energy.shape == (1,)
    torch.testing.assert_close(energy[0], reference["energy"], atol=UPSTREAM_ENERGY_TOL, rtol=0.0)
    torch.testing.assert_close(forces, reference["forces"], atol=UPSTREAM_FORCE_TOL, rtol=0.0)


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
        meq.Irreps("O3", "4x0e"),
        meq.Irreps("O3", "3x0e"),
        meq.Irreps("O3", "5x0e"),
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="flat weight"):
        element_linear_weight(linear, linear.weight.detach())


def test_one_hot_weight_refuses_an_equal_numel_in_another_shape() -> None:
    """The boundary takes the linear's one flat row only — the ``(C, F)`` table is not it."""
    lookup = OneHotLinear(meq.Irreps("O3", "3x0e"), meq.Irreps("O3", "5x0e"), dtype=torch.float64)
    with pytest.raises(ValueError, match="flat weight"):
        one_hot_weight(lookup, lookup.weight.detach())


@pytest.mark.slow
@pytest.mark.skipif(
    _official_weights_absent(MATPES_CONFIG_FILE, MATPES_WEIGHTS_FILE, MATPES_REFERENCE_FILE),
    reason=_absent_reason(MATPES_CONFIG_FILE, MATPES_WEIGHTS_FILE, MATPES_REFERENCE_FILE),
)
class TestOfficialMatpesWeights:
    """The official MatPES checkpoint through ``MaceMatpes.load`` (``science``)."""

    def test_official_checkpoint_bit_parity(self) -> None:
        """The real MatPES checkpoint loads identically both ways.

        ``MaceMatpes.load`` against the hand-written "construct + remap" path on
        ``MACE-matpes-r2scan-omat-ft`` (converted offline with
        ``mace.cli.convert_e3nn_cueq``; never imported here). Bit parity, not a
        tolerance. The oracle arm states the published ``128 / 1 / 16`` dimensions
        literally, so ``load`` has to derive them from the irreps strings unaided.
        """
        config_path = _asset(MATPES_CONFIG_FILE)
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
            dtype=DTYPE,
        )
        matpes_remap.load(
            manual, torch.load(_asset(MATPES_WEIGHTS_FILE), map_location="cpu", weights_only=True)
        )

        built = MaceMatpes.load(config_path, dtype=DTYPE)

        reference = manual.state_dict()
        produced = built.state_dict()
        assert set(produced) == set(reference)
        for name, want in reference.items():
            if name.endswith("_extra_state"):  # module metadata (a dict), not a tensor
                assert produced[name] == want, name
                continue
            assert torch.equal(produced[name], want), name

    def test_official_matpes_energy_and_forces_match_upstream(self, cluster: TensorDict) -> None:
        """``MaceMatpes.load``'s surface is ``mace-torch``'s on the unconditioned cluster."""
        model = MaceMatpes.load(_asset(MATPES_CONFIG_FILE), wrt=FORCES, dtype=DTYPE).eval()
        _assert_upstream_parity(model, cluster, MATPES_REFERENCE_FILE)


def _official_omol() -> tuple[MaceOmol, dict[str, torch.Tensor]]:
    """The full-size OMol model with the official weights, and that state.

    The four fields that vary per checkpoint — element table, ``E0`` table,
    ``scale``, ``shift`` — are read out of the checkpoint itself; every other
    field is the :class:`~molzoo.mace.spec.MACEOMolSpec` default, which the
    checkpoint corroborates (``r_max`` 6.0, ``num_interactions`` 3,
    ``num_bessel`` 8, ``num_polynomial_cutoff`` 5, ``num_features`` 1024,
    ``charge_classes`` / ``spin_classes`` 201 / 101, ``mlp_dim`` 16).
    """
    state = torch.load(_asset(OMOL_WEIGHTS_FILE), map_location="cpu", weights_only=True)
    spec = MACEOMolSpec(
        atomic_numbers=state["atomic_numbers"].tolist(),
        atomic_energies=state["atomic_energies_fn.atomic_energies"].flatten().tolist(),
        scale=float(state["scale_shift.scale"]),
        shift=float(state["scale_shift.shift"]),
    )
    model = MaceOmol(spec, wrt=FORCES, dtype=DTYPE)
    omol_remap.load(model, state)
    return model.eval(), state


@pytest.mark.slow
@pytest.mark.skipif(
    _official_weights_absent(OMOL_WEIGHTS_FILE, OMOL_REFERENCE_FILE),
    reason=_absent_reason(OMOL_WEIGHTS_FILE, OMOL_REFERENCE_FILE),
)
class TestOfficialOMolWeights:
    """The official OMol checkpoint, loaded and evaluated in tree (``science``).

    The weights of ``MACE-omol-0-extra-large-1024`` load strictly through
    ``omol_keys``, and the loaded surface is upstream's on the conditioned
    cluster (charge 0, spin 1 — the same ``embedding_specs`` convention on
    both sides, as the asset's ``REFERENCE.md`` traces). The extraction is
    corroborated at value level too: the checkpoint's
    ``radial_embedding.bessel_fn.bessel_weights`` sit 2.2120e-07 Å⁻¹ from the
    analytic ``nπ/r_max`` init — fp32 rounding of an untrained parameter, which
    the strict load carries in rather than re-deriving.
    """

    def test_official_omol_load_fills_every_learnable(self) -> None:
        """Every ``nn.Parameter`` is covered; the unhoused keys are the snapshot."""
        model, state = _official_omol()

        renamed = omol_remap.rename(state)
        parameters = {name for name, _ in model.named_parameters()}

        assert len(parameters) == OMOL_PARAMETER_COUNT
        assert parameters <= set(renamed)
        assert sorted(set(renamed) - set(model.state_dict())) == OMOL_UNEXPECTED_KEYS

    def test_official_omol_energy_and_forces_match_upstream(self, omol_cluster: TensorDict) -> None:
        """The loaded surface is ``mace-torch``'s on the conditioned cluster."""
        model, _ = _official_omol()
        _assert_upstream_parity(model, omol_cluster, OMOL_REFERENCE_FILE)


@pytest.mark.slow
@pytest.mark.skipif(
    _official_weights_absent(OFF_CONFIG_FILE, OFF_WEIGHTS_FILE, OFF_REFERENCE_FILE),
    reason=_absent_reason(OFF_CONFIG_FILE, OFF_WEIGHTS_FILE, OFF_REFERENCE_FILE),
)
class TestOfficialOffWeights:
    """The official MACE-OFF24 medium checkpoint through ``MaceOff.load`` (``science``)."""

    def test_official_off_load_fills_every_learnable(self) -> None:
        """``load`` is strict on both sides of the table and builds the published model.

        ``off_keys`` raises on a key with no home, the remap on an unfilled
        ``nn.Parameter``; the count is the medium model's.
        """
        model = MaceOff.load(_asset(OFF_CONFIG_FILE), dtype=DTYPE)
        assert sum(1 for _ in model.parameters()) == OFF_PARAMETER_COUNT

    def test_official_off_energy_and_forces_match_upstream(self, cluster: TensorDict) -> None:
        """``MaceOff.load``'s surface is ``mace-torch``'s on the unconditioned cluster."""
        model = MaceOff.load(_asset(OFF_CONFIG_FILE), wrt=FORCES, dtype=DTYPE).eval()
        _assert_upstream_parity(model, cluster, OFF_REFERENCE_FILE)


@pytest.mark.slow
@pytest.mark.skipif(
    _official_weights_absent(LES_CONFIG_FILE, LES_WEIGHTS_FILE, LES_REFERENCE_FILE),
    reason=_absent_reason(LES_CONFIG_FILE, LES_WEIGHTS_FILE, LES_REFERENCE_FILE),
)
class TestOfficialLesWeights:
    """The official MACE-LES-OFF small checkpoint through ``MaceLes.load`` (``science``).

    The fp32 dump fills the fp64 model; the cluster carries no cell, so the
    latent-charge term is the isolated Gaussian pair sum on both sides.
    """

    def test_official_les_load_fills_every_learnable(self) -> None:
        """``load`` is strict on both sides of the table and builds the published model.

        ``les_keys`` raises on a key with no home, the remap on an unfilled
        ``nn.Parameter``; the count is the small model's, charge readouts
        included.
        """
        model = MaceLes.load(_asset(LES_CONFIG_FILE), dtype=DTYPE)
        assert sum(1 for _ in model.parameters()) == LES_PARAMETER_COUNT

    def test_official_les_energy_and_forces_match_upstream(self, cluster: TensorDict) -> None:
        """``MaceLes.load``'s surface is ``mace-torch``'s on the isolated cluster."""
        model = MaceLes.load(_asset(LES_CONFIG_FILE), wrt=FORCES, dtype=DTYPE).eval()
        _assert_upstream_parity(model, cluster, LES_REFERENCE_FILE)
