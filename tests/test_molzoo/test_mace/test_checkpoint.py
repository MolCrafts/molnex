"""Tests for molzoo.mace.checkpoint — the one official-checkpoint key remap.

:class:`~molzoo.mace.checkpoint.CheckpointRemap` replaces the two hand-copied
loaders of the pre-cutover flat modules (now the thin compatibility aliases
:func:`molzoo.mace.variants.load_matpes_state_dict` /
:func:`~molzoo.mace.variants.load_omol_state_dict`) with a single type whose only
family-dependent behaviour is the ``on_unexpected`` knob: MatPES raises on a
checkpoint key with no home, OMol returns it (that family ships auxiliary
heads this port deliberately does not model). Everything else — the
``.graph.c`` / ``output_mask`` skip, the longest-prefix rename, the
numel-conserving reshape of ``(1,)`` frozen scalars, and above all the
strictness doctrine — is shared and must stay identical.

The doctrine is the point of most of this file. A checkpoint holds a *fitted
potential energy surface*: a tensor that is silently dropped yields a model
that runs, looks sane, and is quietly wrong (``mace_matpes.py:405-410``). So an
unfilled learnable parameter or a genuine shape disagreement raises under
*both* policies — the knob is not a laxness dial. ``test_rejects_missing_para
meter_without_weight_suffix`` is the regression lock for the second incident
(``mace_omol.py:437-439``): "learnable" means exactly ``nn.Parameter``, never a
``.weight`` / ``.bias`` name heuristic, which once excused ``bessel.freqs``
from the check and left ~2e-7 of the official weights in the parity residual.

Test data is built with the ``_official_state`` round trip — a model's own
``state_dict`` renamed into official cueq names and loaded back into a
differently initialised model — the trick the pre-cutover
``TestLoadMatpesStateDict._roundtrip_state`` used, now shared by that class
(migrated into this file by ``mace-subpackage-restructure-06-wire``) and by
:class:`TestCheckpointRemap`. The inverse name tables below are written out
**by hand** rather than derived from the tables under test, so a wrong entry in
the production table cannot cancel itself out in the round trip.

Every model here is tiny (2 layers, 16 channels, ``l_max=1``), fp64 (autouse
``fp64`` fixture in ``conftest.py``), CPU, ``use_fallback=True`` (no fused
cuEquivariance wheel on CPU) and seeded — see
``.claude/specs/mace-subpackage-restructure-05-checkpoint.md`` §Testing
strategy. Units are eV / eV·Å throughout; the remap converts nothing.

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

from molzoo.mace.checkpoint import (
    MATPES_KEY_REMAP,
    MATPES_REMAP,
    OMOL_REMAP,
    CheckpointRemap,
)
from molzoo.mace.potential import MACEPotential
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec
from molzoo.mace.variants import MACEMatpes, load_matpes_state_dict
from tests.test_molzoo.test_mace.conftest import (
    ATOMIC_ENERGIES,
    ATOMIC_NUMBERS,
    TINY_MATPES_KWARGS,
)

#: Model name → official cueq name: the hand-written inverse of
#: ``MATPES_KEY_REMAP``, longest prefix first (see the module docstring for why
#: it is not derived from the table under test).
MATPES_OFFICIAL_NAMES: dict[str, str] = {
    "node_embedding.": "node_embedding.linear.",
    "bessel.freqs": "radial_embedding.bessel_fn.bessel_weights",
    "distance_transform.": "radial_embedding.distance_transform.",
    "pair_repulsion.": "pair_repulsion_fn.",
    "z_table": "atomic_numbers",
}

#: Model name → official cueq name for the OMol family (inverse of
#: ``OMOL_KEY_REMAP``). ``readout.`` is the single final readout molnex holds
#: where the official checkpoint numbers a one-entry ``readouts`` list.
OMOL_OFFICIAL_NAMES: dict[str, str] = {
    "node_embedding.": "node_embedding.linear.",
    "embedding_readout.": "embedding_readout.linear.",
    "readout.": "readouts.0.",
    "bessel.freqs": "radial_embedding.bessel_fn.bessel_weights",
    "z_table": "atomic_numbers",
}

#: A checkpoint key that matches no table entry and no model parameter — the
#: probe for the ``on_unexpected`` knob (same key as the flat loader's test).
MYSTERY_KEY = "interactions.0.mystery_layer.weight"

#: A learnable weight every MatPES model has; deleting it must raise.
COVERED_PARAMETER = "interactions.0.linear_up.weight"

#: Official name of ``bessel.freqs`` — an ``nn.Parameter`` ending in neither
#: ``.weight`` nor ``.bias``. Deleting it is the ``mace_omol.py:437-439``
#: regression lock.
BESSEL_OFFICIAL_KEY = "radial_embedding.bessel_fn.bessel_weights"

#: MACE stores some frozen scalars as ``(1,)`` where molnex holds a 0-d buffer.
FROZEN_SCALAR_VALUE = 0.75

#: Environment variable pointing at the offline official-weights directory.
WEIGHTS_DIR_ENV = "MOLNEX_MACE_WEIGHTS_DIR"

#: The two architecture switches of the stock MatPES dump, spelled exactly as
#: ``matpes_r2scan_config.json`` spells them (read 2026-08-09 from the offline
#: weights directory): ZBL pair repulsion on, Agnesi distance transform —
#: capital ``A``. :class:`~molzoo.mace.spec.MACEMatpesSpec` hard-wires both, so
#: these are the only *present* values ``from_checkpoint`` may accept, and it
#: must accept them without caring about the case of the ``A``.
OFFICIAL_SWITCHES: dict[str, object] = {"pair_repulsion": True, "distance_transform": "Agnesi"}

#: Hard-coded goldens: the dimensions of ``MACE-matpes-r2scan-omat-ft`` as
#: published (``hidden_irreps="128x0e+128x1o"``, ``MLP_irreps="16x0e"``). The
#: bit-parity oracle states them literally so ``from_checkpoint`` has to derive
#: the same numbers from the irreps strings on its own.
OFFICIAL_NUM_FEATURES = 128
OFFICIAL_MAX_HIDDEN_L = 1
OFFICIAL_MLP_DIM = 16

#: Offline OMol asset inside :data:`WEIGHTS_DIR_ENV`: the cueq ``state_dict`` of
#: ``MACE-omol-0-extra-large-1024``. The shipped ``OMOL-cueq.model`` is a
#: *pickled* ``mace.modules.models.ScaleShiftMACE`` — reading it needs ``mace``
#: and ``e3nn`` imported, which CLAUDE.md forbids and which the toolchain does
#: not install — so it was dumped out of tree into the plain ``name -> tensor``
#: twin that ``torch.load(weights_only=True)`` reads with no third-party class
#: at all, exactly the shape MatPES already ships
#: (``matpes_r2scan_cueq_state.pt``). The converter sits beside the asset as
#: ``convert_omol_to_cueq_state.py``.
OMOL_WEIGHTS_FILE = "omol_cueq_state.pt"

#: Hard-coded golden: every ``nn.Parameter`` tensor of the OMol model the
#: official checkpoint has to fill. Pinning the count keeps
#: ``test_official_omol_load_fills_every_learnable`` from passing vacuously on
#: a model that lost half its blocks.
OMOL_PARAMETER_COUNT = 104

#: Hard-coded golden: the checkpoint keys with no home in molnex's OMol model.
#: ``OMOL_REMAP`` exists with ``on_unexpected="return"`` because the OMol family
#: *can* ship auxiliary heads this port does not model; the single-head
#: ``omol`` checkpoint actually shipped has none, so the snapshot is empty —
#: and any future key growing into this list is a deliberate decision, not a
#: silent one.
OMOL_UNEXPECTED_KEYS: list[str] = []

#: Hard-coded goldens (eV, eV/Å): energy, ``max|F|`` and ``F[0]`` of the
#: ``omol_cluster`` fixture under the official OMol weights, captured at
#: ``OMP_NUM_THREADS=1`` — see :class:`TestOfficialOMolWeights` for the full
#: provenance and :data:`OMOL_ENERGY_TOL` for why the thread count is named.
OMOL_GOLDEN_ENERGY = -3122.454566006894
OMOL_GOLDEN_MAX_FORCE = 5.397722165018621
OMOL_GOLDEN_FIRST_FORCE = (-5.397722165018621, -3.9298849841895853, -3.931997122634094)

#: The repo's *numerical* tolerances (energy 1e-6 eV, force 1e-4 eV/Å), not the
#: exact ones, because this surface is **not** bit-reproducible across thread
#: counts: the CPU reduction order inside the cuEquivariance fallback follows
#: ``torch.get_num_threads()``. Measured here it is bit-identical within a
#: thread count and across repeat processes, but ``{1, 4, 16}`` threads and
#: this node's 48-thread default disagree by 3.2e-08 eV / 9.3e-08 eV/Å.
#: Tightening to the exact pair would make the case pass or fail on how many
#: cores the runner happens to have — a flake, not a stricter lock.
OMOL_ENERGY_TOL = 1e-6
OMOL_FORCE_TOL = 1e-4

_TEST_PACKAGE = Path(__file__).resolve().parent


def _official_weights_absent(filename: str) -> bool:
    """Whether the gated offline asset ``filename`` is unavailable.

    Args:
        filename: File expected inside the :data:`WEIGHTS_DIR_ENV` directory.

    Returns:
        ``True`` when the environment variable is unset or the file is missing,
        which is the skip condition of the weights-gated cases.
    """
    directory = os.environ.get(WEIGHTS_DIR_ENV)
    return directory is None or not (Path(directory) / filename).is_file()


# --------------------------------------------------------------------------
# Precondition guard (spec §Tasks): a same-named module next to the package
# makes ``tests.test_molzoo.test_mace`` ambiguous and collection silent.
# --------------------------------------------------------------------------


def test_no_module_shadows_the_mace_test_package() -> None:
    """A leftover ``test_mace.py`` next to ``test_mace/`` hides one of them."""
    assert not (_TEST_PACKAGE.parent / "test_mace.py").exists()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _official_state(model: torch.nn.Module, names: Mapping[str, str]) -> dict[str, torch.Tensor]:
    """Rename a model's own ``state_dict`` back into official cueq key names.

    Args:
        model: Source model.
        names: Molnex prefix → official prefix (longest match wins).

    Returns:
        The same tensors under the names an official checkpoint would use.
    """
    ordered = sorted(names.items(), key=lambda item: -len(item[0]))
    out: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        for prefix, official in ordered:
            if key == prefix or key.startswith(prefix):
                key = official + (key[len(prefix) :] if key != prefix else "")
                break
        out[key] = value
    return out


def _filled(spec: MACEMatpesSpec | MACEOMolSpec, seed: int) -> MACEPotential:
    """A tiny CPU potential whose every parameter is non-zero and seed-specific.

    Two models built from the same spec with different seeds must disagree in
    every parameter, or a round-trip assertion could pass on tensors nobody
    moved. The OMol readout is zero-initialised at construction
    (``molrep.readout.mace._ScalarO3Linear``), so the explicit fill is what
    keeps those keys meaningful here.

    Args:
        spec: Validated variant configuration.
        seed: Seed of both the construction RNG and the fill generator.

    Returns:
        A ``MACEPotential`` on the pure-torch cuEquivariance path.
    """
    torch.manual_seed(seed)
    model = MACEPotential(spec, use_fallback=True)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(generator=generator)
    return model


def _write_checkpoint(
    directory: Path,
    spec: MACEMatpesSpec,
    model: MACEPotential,
    *,
    hidden_irreps: str,
    mlp_irreps: str,
    drop: tuple[str, ...] = (),
    switches: Mapping[str, object] | None = None,
) -> tuple[Path, Path]:
    """Write an official-shaped MatPES config json + cueq ``state_dict``.

    Args:
        directory: Destination directory (``tmp_path``).
        spec: The spec ``model`` was built from; supplies the config values.
        model: Source of the weights, dumped under official cueq names.
        hidden_irreps: Official ``hidden_irreps`` string, e.g. ``"32x0e+32x1o"``.
        mlp_irreps: Official ``MLP_irreps`` string, e.g. ``"8x0e"``.
        drop: Config keys to leave out, for the missing-key cases.
        switches: Extra config entries — the architecture-switch keys
            (``pair_repulsion`` / ``distance_transform``) the stock dump
            carries. Omitted by default, which is the older-dump case.

    Returns:
        ``(config_path, weights_path)``.
    """
    config: dict[str, object] = {
        "atomic_numbers": spec.atomic_numbers,
        "atomic_energies": spec.atomic_energies,
        "r_max": spec.r_max,
        "num_bessel": spec.num_bessel,
        "num_polynomial_cutoff": spec.num_polynomial_cutoff,
        "max_ell": spec.l_max,
        "correlation": spec.correlation,
        "num_interactions": spec.num_interactions,
        "hidden_irreps": hidden_irreps,
        "MLP_irreps": mlp_irreps,
        "radial_MLP": spec.radial_mlp,
        "atomic_inter_scale": spec.scale,
        "atomic_inter_shift": spec.shift,
    }
    config.update(switches or {})
    for key in drop:
        del config[key]
    config_path = directory / "matpes_config.json"
    config_path.write_text(json.dumps(config))
    weights_path = directory / "matpes_cueq_state.pt"
    torch.save(_official_state(model, MATPES_OFFICIAL_NAMES), weights_path)
    return config_path, weights_path


class TestCheckpointRemap:
    """Test the one remap that both foundation families now share."""

    def test_rename_is_pure(self) -> None:
        """``.rename`` copies, drops graph constants / masks, and renames."""
        weight = torch.zeros(2)
        freqs = torch.ones(4)
        official = {
            "node_embedding.linear.weight": weight,
            BESSEL_OFFICIAL_KEY: freqs,
            "node_embedding.linear.f.m.graphs.0.graph.c0": torch.zeros(()),
            "interactions.0.conv_tp.output_mask": torch.zeros(3),
            "interactions.0.linear.weight": weight,
            "r_max": torch.tensor(6.0),
        }
        snapshot = dict(official)

        renamed = MATPES_REMAP.rename(official)

        assert renamed is not official
        assert list(official) == list(snapshot)
        assert all(official[key] is value for key, value in snapshot.items())
        assert set(renamed) == {
            "node_embedding.weight",
            "bessel.freqs",
            "interactions.0.linear.weight",
        }
        assert renamed["bessel.freqs"] is freqs

    def test_roundtrip_restores_every_parameter_matpes(
        self, tiny_matpes_spec: MACEMatpesSpec
    ) -> None:
        """MatPES: official names load back bit-for-bit into a fresh model."""
        source = _filled(tiny_matpes_spec, seed=0)
        target = _filled(tiny_matpes_spec, seed=1)

        MATPES_REMAP.load(target, _official_state(source, MATPES_OFFICIAL_NAMES))

        for (name, want), (_, got) in zip(
            source.named_parameters(), target.named_parameters(), strict=True
        ):
            assert torch.equal(want, got), name

    def test_roundtrip_restores_every_parameter_omol(self, tiny_omol_spec: MACEOMolSpec) -> None:
        """OMol: same bijection on the family that shipped with no unit test."""
        source = _filled(tiny_omol_spec, seed=0)
        target = _filled(tiny_omol_spec, seed=1)

        OMOL_REMAP.load(target, _official_state(source, OMOL_OFFICIAL_NAMES))

        for (name, want), (_, got) in zip(
            source.named_parameters(), target.named_parameters(), strict=True
        ):
            assert torch.equal(want, got), name

    def test_raise_policy_rejects_unexpected_key(self, tiny_matpes_spec: MACEMatpesSpec) -> None:
        """MatPES policy: refuse before touching the model — no half load."""
        source = _filled(tiny_matpes_spec, seed=0)
        target = _filled(tiny_matpes_spec, seed=1)
        state = _official_state(source, MATPES_OFFICIAL_NAMES)
        state[MYSTERY_KEY] = torch.zeros(3, dtype=torch.float64)
        before = {name: parameter.detach().clone() for name, parameter in target.named_parameters()}

        with pytest.raises(RuntimeError, match="no home"):
            MATPES_REMAP.load(target, state)

        for name, parameter in target.named_parameters():
            assert torch.equal(parameter, before[name]), name

    def test_return_policy_reports_unexpected_key(self, tiny_matpes_spec: MACEMatpesSpec) -> None:
        """OMol policy: the same key is reported, not raised."""
        source = _filled(tiny_matpes_spec, seed=0)
        target = _filled(tiny_matpes_spec, seed=1)
        state = _official_state(source, MATPES_OFFICIAL_NAMES)
        state[MYSTERY_KEY] = torch.zeros(3, dtype=torch.float64)
        lenient = CheckpointRemap(MATPES_KEY_REMAP, on_unexpected="return")

        _, unexpected = lenient.load(target, state)

        assert unexpected == [MYSTERY_KEY]

    @pytest.mark.parametrize("policy", ["raise", "return"])
    def test_rejects_missing_parameter(self, tiny_matpes_spec: MACEMatpesSpec, policy: str) -> None:
        """An unfilled parameter would keep its random init — never load it."""
        source = _filled(tiny_matpes_spec, seed=0)
        target = _filled(tiny_matpes_spec, seed=1)
        state = _official_state(source, MATPES_OFFICIAL_NAMES)
        del state[COVERED_PARAMETER]

        with pytest.raises(RuntimeError, match="not covered") as raised:
            CheckpointRemap(MATPES_KEY_REMAP, on_unexpected=policy).load(target, state)

        assert COVERED_PARAMETER in str(raised.value)

    @pytest.mark.parametrize("policy", ["raise", "return"])
    def test_rejects_missing_parameter_without_weight_suffix(
        self, tiny_matpes_spec: MACEMatpesSpec, policy: str
    ) -> None:
        """``bessel.freqs`` is an ``nn.Parameter``; no name heuristic excuses it.

        Regression lock for ``mace_omol.py:437-439``: a ``.weight`` / ``.bias``
        suffix test let this one through, and the official weights differ from
        the analytic init by ~2e-7 — straight into the reported parity residual.
        """
        source = _filled(tiny_matpes_spec, seed=0)
        target = _filled(tiny_matpes_spec, seed=1)
        state = _official_state(source, MATPES_OFFICIAL_NAMES)
        del state[BESSEL_OFFICIAL_KEY]

        with pytest.raises(RuntimeError, match="not covered") as raised:
            CheckpointRemap(MATPES_KEY_REMAP, on_unexpected=policy).load(target, state)

        assert "bessel.freqs" in str(raised.value)

    @pytest.mark.parametrize("policy", ["raise", "return"])
    def test_rejects_shape_mismatch(self, tiny_matpes_spec: MACEMatpesSpec, policy: str) -> None:
        """Wrong shapes mean the model was built with the wrong config."""
        source = _filled(tiny_matpes_spec, seed=0)
        target = _filled(tiny_matpes_spec, seed=1)
        state = _official_state(source, MATPES_OFFICIAL_NAMES)
        state[COVERED_PARAMETER] = torch.zeros(1, 7, dtype=torch.float64)

        with pytest.raises(RuntimeError, match="shape mismatch"):
            CheckpointRemap(MATPES_KEY_REMAP, on_unexpected=policy).load(target, state)

    @pytest.mark.parametrize("policy", ["raise", "return"])
    def test_accepts_rank_difference_on_frozen_scalars(
        self, tiny_matpes_spec: MACEMatpesSpec, policy: str
    ) -> None:
        """MACE stores ``scale`` as ``(1,)``; molnex holds a 0-d buffer."""
        source = _filled(tiny_matpes_spec, seed=0)
        target = _filled(tiny_matpes_spec, seed=1)
        state = _official_state(source, MATPES_OFFICIAL_NAMES)
        state["scale_shift.scale"] = torch.tensor([FROZEN_SCALAR_VALUE], dtype=torch.float64)

        CheckpointRemap(MATPES_KEY_REMAP, on_unexpected=policy).load(target, state)

        assert target.scale_shift.scale.shape == ()
        assert float(target.scale_shift.scale) == pytest.approx(FROZEN_SCALAR_VALUE)

    def test_reported_lists_are_sorted(self, tiny_matpes_spec: MACEMatpesSpec) -> None:
        """Both returned lists are sorted — a deterministic normalisation.

        The flat OMol loader handed back torch's ``unexpected`` in checkpoint
        order; the merged implementation sorts both lists so a diff of two runs
        cannot move on dict ordering alone.
        """
        source = _filled(tiny_matpes_spec, seed=0)
        target = _filled(tiny_matpes_spec, seed=1)
        state = _official_state(source, MATPES_OFFICIAL_NAMES)
        state["interactions.9.zeta.weight"] = torch.zeros(3, dtype=torch.float64)
        state["interactions.0.alpha.weight"] = torch.zeros(3, dtype=torch.float64)
        lenient = CheckpointRemap(MATPES_KEY_REMAP, on_unexpected="return")

        missing, unexpected = lenient.load(target, state)

        assert unexpected == ["interactions.0.alpha.weight", "interactions.9.zeta.weight"]
        # The graph constants cueq rebuilds are the natural multi-element case;
        # torch reports them in module-registration order, which is not sorted.
        assert len(missing) > 1
        assert missing == sorted(missing)
        assert missing != [key for key in target.state_dict() if key in set(missing)]


class TestFromCheckpoint:
    """Test ``MACEPotential.from_checkpoint`` — json + weights in one call."""

    def test_builds_from_config_and_weights(
        self, tmp_path: Path, tiny_matpes_spec: MACEMatpesSpec
    ) -> None:
        """An official-shaped config + checkpoint reproduces the source model."""
        source = _filled(tiny_matpes_spec, seed=0)
        config_path, weights_path = _write_checkpoint(
            tmp_path,
            tiny_matpes_spec,
            source,
            hidden_irreps="16x0e+16x1o",
            mlp_irreps="8x0e",
        )

        built = MACEPotential.from_checkpoint(config_path, weights_path, use_fallback=True)

        for (name, want), (_, got) in zip(
            source.named_parameters(), built.named_parameters(), strict=True
        ):
            assert torch.equal(want, got), name

    def test_derives_dims_from_irreps(self, tmp_path: Path) -> None:
        """``num_features`` / ``max_hidden_l`` / ``mlp_dim`` come from the irreps.

        ``run_nve.py:154-159`` hard-codes ``128 / 1 / 16``; a checkpoint with
        any other width would then load into the wrong model — or, once the
        shapes happen to fit, be quietly wrong. These dimensions (32 / 1 / 8)
        share no value with that triple.
        """
        spec = MACEMatpesSpec(
            **{**TINY_MATPES_KWARGS, "num_features": 32, "max_hidden_l": 1, "mlp_dim": 8},
            atomic_numbers=ATOMIC_NUMBERS,
            atomic_energies=ATOMIC_ENERGIES,
        )
        config_path, weights_path = _write_checkpoint(
            tmp_path,
            spec,
            _filled(spec, seed=0),
            hidden_irreps="32x0e+32x1o",
            mlp_irreps="8x0e",
        )

        built = MACEPotential.from_checkpoint(config_path, weights_path, use_fallback=True)

        derived = built._spec
        assert isinstance(derived, MACEMatpesSpec)
        assert (derived.num_features, derived.max_hidden_l, derived.mlp_dim) == (32, 1, 8)
        # Corroborate on the built module, not only on the parsed spec.
        assert built.node_embedding.weight.numel() == len(ATOMIC_NUMBERS) * 32
        assert built.readouts[-1].linear_1.weight.numel() == 32 * 8
        assert built.readouts[-1].linear_2.weight.numel() == 8

    @pytest.mark.parametrize("absent", ["hidden_irreps", "MLP_irreps"])
    def test_rejects_config_missing_irreps(
        self, tmp_path: Path, tiny_matpes_spec: MACEMatpesSpec, absent: str
    ) -> None:
        """No silent fallback to a default width — name the missing key."""
        config_path, weights_path = _write_checkpoint(
            tmp_path,
            tiny_matpes_spec,
            _filled(tiny_matpes_spec, seed=0),
            hidden_irreps="16x0e+16x1o",
            mlp_irreps="8x0e",
            drop=(absent,),
        )

        with pytest.raises((KeyError, ValueError), match=absent):
            MACEPotential.from_checkpoint(config_path, weights_path, use_fallback=True)

    def test_tolerates_a_config_without_the_architecture_switches(
        self, tmp_path: Path, tiny_matpes_spec: MACEMatpesSpec
    ) -> None:
        """A dump that names neither switch is loaded on the spec's defaults.

        Absence is not a contradiction: older dumps predate the two keys, and
        the spec's ZBL + Agnesi are what a stock MatPES checkpoint was fitted
        with anyway. This is the case the whole existing suite (and
        :func:`_write_checkpoint`'s default config) runs on, pinned explicitly
        so a switch guard cannot be written as "absent means off".
        """
        config_path, weights_path = _write_checkpoint(
            tmp_path,
            tiny_matpes_spec,
            _filled(tiny_matpes_spec, seed=0),
            hidden_irreps="16x0e+16x1o",
            mlp_irreps="8x0e",
        )
        assert set(json.loads(config_path.read_text())).isdisjoint(OFFICIAL_SWITCHES)

        built = MACEPotential.from_checkpoint(config_path, weights_path, use_fallback=True)

        assert built._spec.pair_repulsion == "zbl"
        assert built._spec.distance_transform == "agnesi"

    def test_accepts_the_switches_the_official_dump_spells(
        self, tmp_path: Path, tiny_matpes_spec: MACEMatpesSpec
    ) -> None:
        """``pair_repulsion: true`` + ``distance_transform: "Agnesi"`` agree.

        The literal spelling of :data:`OFFICIAL_SWITCHES` is what the shipped
        ``matpes_r2scan_config.json`` holds, so a case-sensitive comparison
        against the spec's lower-case ``"agnesi"`` would reject the very
        checkpoint this constructor exists for — and
        ``test_official_checkpoint_bit_parity`` would only catch it where the
        offline weights are installed.
        """
        source = _filled(tiny_matpes_spec, seed=0)
        config_path, weights_path = _write_checkpoint(
            tmp_path,
            tiny_matpes_spec,
            source,
            hidden_irreps="16x0e+16x1o",
            mlp_irreps="8x0e",
            switches=OFFICIAL_SWITCHES,
        )

        built = MACEPotential.from_checkpoint(config_path, weights_path, use_fallback=True)

        for (name, want), (_, got) in zip(
            source.named_parameters(), built.named_parameters(), strict=True
        ):
            assert torch.equal(want, got), name

    @pytest.mark.parametrize(
        ("switches", "offender", "hard_wired", "spellings"),
        [
            ({"pair_repulsion": False}, "pair_repulsion", "zbl", ("False", "false")),
            ({"distance_transform": None}, "distance_transform", "agnesi", ("None", "null")),
            (
                {"distance_transform": "polynomial"},
                "distance_transform",
                "agnesi",
                ("polynomial",),
            ),
        ],
        ids=["pair_repulsion_off", "distance_transform_null", "distance_transform_other"],
    )
    def test_rejects_a_config_that_contradicts_a_hard_wired_switch(
        self,
        tmp_path: Path,
        tiny_matpes_spec: MACEMatpesSpec,
        switches: dict[str, object],
        offender: str,
        hard_wired: str,
        spellings: tuple[str, ...],
    ) -> None:
        """A config from another MACE family must not build a MatPES model.

        :class:`~molzoo.mace.spec.MACEMatpesSpec` hard-wires ZBL pair repulsion
        and the Agnesi distance transform, and the fitted constants of both are
        **buffers**, not ``nn.Parameter`` (``trainable=False``). So the strict
        remap cannot notice: the surplus block's constants land in the discarded
        ``missing_buffers`` list, every learnable is filled, the load returns
        happily — and the model computes a short-range repulsion (or transforms
        every distance) the checkpoint was never fitted with. That is the
        "runs, looks sane, quietly wrong" failure the module docstring is about,
        one config key away.

        The message must name the offending key **and** both values, because
        the fix is to build the spec by hand — the reader has to know which
        switch disagreed and in which direction.
        """
        config_path, weights_path = _write_checkpoint(
            tmp_path,
            tiny_matpes_spec,
            _filled(tiny_matpes_spec, seed=0),
            hidden_irreps="16x0e+16x1o",
            mlp_irreps="8x0e",
            switches={**OFFICIAL_SWITCHES, **switches},
        )

        with pytest.raises(ValueError, match=offender) as raised:
            MACEPotential.from_checkpoint(config_path, weights_path, use_fallback=True)

        message = str(raised.value)
        assert hard_wired in message, message
        assert any(spelling in message for spelling in spellings), message


class TestLoadMatpesStateDict:
    """Test the compatibility alias ``variants.load_matpes_state_dict``.

    Migrated verbatim (criteria and messages) from
    ``tests/test_molzoo/test_mace_matpes.py::TestLoadMatpesStateDict`` by
    ``mace-subpackage-restructure-06-wire``. The function lives in
    :mod:`molzoo.mace.variants`, but everything it promises — the key dialect
    and the three refusals — is :mod:`molzoo.mace.checkpoint`'s doctrine, so
    the cases sit next to :class:`TestCheckpointRemap` (spec §5). What they add
    over that class is the *forwarding*: an alias that quietly called
    ``load_state_dict`` instead would pass none of the three refusals.
    """

    def test_roundtrip_restores_every_parameter(self, matpes_variant: MACEMatpes) -> None:
        """A checkpoint written in official names loads back bit-for-bit."""
        state = _official_state(matpes_variant, MATPES_OFFICIAL_NAMES)
        torch.manual_seed(1)
        fresh = MACEMatpes(
            atomic_numbers=list(ATOMIC_NUMBERS),
            atomic_energies=torch.tensor(ATOMIC_ENERGIES),
            **{**TINY_MATPES_KWARGS, "use_fallback": False},
        )
        assert any(
            not torch.equal(want, got)
            for (_, want), (_, got) in zip(
                matpes_variant.named_parameters(), fresh.named_parameters(), strict=True
            )
        ), "the two models start out identical — the round trip would be vacuous"

        load_matpes_state_dict(fresh, state)

        for (name, want), (_, got) in zip(
            matpes_variant.named_parameters(), fresh.named_parameters(), strict=True
        ):
            assert torch.equal(want, got), name

    def test_rejects_unknown_checkpoint_key(self, matpes_variant: MACEMatpes) -> None:
        """A key with no home means the mapping is stale — do not load silently."""
        state = _official_state(matpes_variant, MATPES_OFFICIAL_NAMES)
        state[MYSTERY_KEY] = torch.zeros(3)
        with pytest.raises(RuntimeError, match="no home"):
            load_matpes_state_dict(matpes_variant, state)

    def test_rejects_missing_parameter(self, matpes_variant: MACEMatpes) -> None:
        """A parameter the checkpoint never fills would keep its random init."""
        state = _official_state(matpes_variant, MATPES_OFFICIAL_NAMES)
        del state[COVERED_PARAMETER]
        with pytest.raises(RuntimeError, match="not covered"):
            load_matpes_state_dict(matpes_variant, state)

    def test_rejects_shape_mismatch(self, matpes_variant: MACEMatpes) -> None:
        """Wrong shapes mean the model was built with the wrong config."""
        state = _official_state(matpes_variant, MATPES_OFFICIAL_NAMES)
        state[COVERED_PARAMETER] = torch.zeros(1, 7, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            load_matpes_state_dict(matpes_variant, state)

    def test_accepts_rank_difference_on_frozen_scalars(self, matpes_variant: MACEMatpes) -> None:
        """MACE stores scale/shift as ``(1,)``; molnex holds a 0-d buffer."""
        state = _official_state(matpes_variant, MATPES_OFFICIAL_NAMES)
        state["scale_shift.scale"] = torch.tensor([FROZEN_SCALAR_VALUE], dtype=torch.float64)

        load_matpes_state_dict(matpes_variant, state)

        assert float(matpes_variant.scale_shift.scale) == pytest.approx(FROZEN_SCALAR_VALUE)


@pytest.mark.skipif(
    os.environ.get(WEIGHTS_DIR_ENV) is None,
    reason=f"{WEIGHTS_DIR_ENV} unset — the converted official weights are an offline asset",
)
def test_official_checkpoint_bit_parity() -> None:
    """The real MatPES checkpoint loads identically both ways (``science``).

    ``from_checkpoint`` against the hand-written "construct + ``MATPES_REMAP``"
    path on ``MACE-matpes-r2scan-omat-ft`` (converted offline with
    ``mace.cli.convert_e3nn_cueq``; that tool is never imported here). Bit
    parity, not a tolerance: the two paths must place the very same tensors.
    The oracle arm states the published ``128 / 1 / 16`` dimensions literally,
    so ``from_checkpoint`` has to derive them from ``hidden_irreps`` /
    ``MLP_irreps`` unaided.
    """
    directory = Path(os.environ[WEIGHTS_DIR_ENV])
    config_path = directory / "matpes_r2scan_config.json"
    weights_path = directory / "matpes_r2scan_cueq_state.pt"
    official = json.loads(config_path.read_text())

    manual = MACEPotential(
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
        use_fallback=True,
    )
    MATPES_REMAP.load(manual, torch.load(weights_path, map_location="cpu", weights_only=True))

    built = MACEPotential.from_checkpoint(config_path, weights_path, use_fallback=True)

    reference = manual.state_dict()
    produced = built.state_dict()
    assert set(produced) == set(reference)
    for name, want in reference.items():
        assert torch.equal(produced[name], want), name


def _official_omol() -> tuple[MACEPotential, dict[str, torch.Tensor]]:
    """The full-size OMol model with the official weights, and that state.

    Hyper-parameters are **not** guessed: the four that vary per checkpoint —
    element table, ``E0`` table, ``scale``, ``shift`` — are read out of the
    checkpoint itself, and every other field is the
    :class:`~molzoo.mace.spec.MACEOMolSpec` default, which is exactly what the
    deleted ``scripts/omol_port/verify_e2e.py`` did (commit ``b85d12f^``:
    ``MACEOMol(atomic_numbers=ztab, atomic_energies=ae, scale=…, shift=…)``).
    The checkpoint corroborates every default it can: ``r_max`` 6.0,
    ``num_interactions`` 3, ``num_bessel`` 8 (``bessel_weights``),
    ``num_polynomial_cutoff`` 5 (``cutoff_fn.p``), ``num_features`` 1024 and
    ``charge_classes`` / ``spin_classes`` 201 / 101 (the joint-embedding
    tables), ``mlp_dim`` 16 (``readouts.0.linear_1.weight`` = 1024 × 16).

    ``use_fallback=True`` because the fused cuEquivariance kernels need a GPU
    and the ops wheel; :class:`~molzoo.mace.variants.MACEOMol` has no such
    keyword (it hard-codes the fused path), so the spec is built directly.

    Returns:
        The loaded ``MACEPotential`` in eval mode and the official state.
    """
    weights_path = Path(os.environ[WEIGHTS_DIR_ENV]) / OMOL_WEIGHTS_FILE
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    spec = MACEOMolSpec(
        atomic_numbers=state["atomic_numbers"].tolist(),
        atomic_energies=state["atomic_energies_fn.atomic_energies"].flatten().tolist(),
        scale=float(state["scale_shift.scale"]),
        shift=float(state["scale_shift.shift"]),
        use_fallback=True,
    )
    model = MACEPotential(spec, use_fallback=True)
    OMOL_REMAP.load(model, state)
    return model.eval(), state


@pytest.mark.skipif(
    _official_weights_absent(OMOL_WEIGHTS_FILE),
    reason=(
        f"{WEIGHTS_DIR_ENV} unset or {OMOL_WEIGHTS_FILE} absent — "
        "the converted official OMol weights are an offline asset"
    ),
)
class TestOfficialOMolWeights:
    """The official OMol checkpoint, loaded and evaluated in tree (``science``).

    This class re-homes the role of the deleted ``scripts/omol_port/verify_*.py``
    oracles, which ``src/molzoo/specs/mace_omol.md`` §7.1 still cites for its
    7.0e-7 eV / 4.3e-6 eV/Å parity record (deleted in ``b85d12f``; §7.1's own
    Appendix-A entry of 2026-08-09 calls the record unreproducible in-tree).
    What it is **not** is a replacement for that record: nothing here compares
    against ``mace-torch`` or ``e3nn``, and no claim about upstream parity can
    be read out of a green run. It is a **stability lock** — the official
    weights load strictly, and the resulting surface is the one measured on
    this machine on the date below. A future refactor that shifts the OMol
    energy by more than a microelectronvolt has to say so out loud.

    Provenance of the goldens (all captured 2026-08-09 on the machine this
    repository is checked out on):

    * weights ``$MOLNEX_MACE_WEIGHTS_DIR/omol_cueq_state.pt``,
      sha256 ``074c86154a1d709f…``, derived from ``OMOL-cueq.model``
      (sha256 ``8735a524e99fe80c…``, the ``mace.cli.convert_e3nn_cueq`` twin of
      ``MACE-omol-0-extra-large-1024``) by the ``convert_omol_to_cueq_state.py``
      script stored beside it — see :data:`OMOL_WEIGHTS_FILE` for why a
      re-dump was needed at all;
    * ``torch 2.12.1+cpu``, ``cuequivariance 0.10.0``, CPU, fp64 (the autouse
      ``fp64`` fixture), ``use_fallback=True``, ``OMP_NUM_THREADS=1``, no seed
      anywhere — the model is fully determined by the checkpoint and the
      geometry is a literal, but the thread count is *not* free
      (:data:`OMOL_ENERGY_TOL`), and on a many-core runner the forward is two
      orders slower than at ``OMP_NUM_THREADS=8`` (48-way oversubscription on
      a five-atom graph: ~80 s against ~0.6 s);
    * the extraction is corroborated at *value* level, not only by shapes:
      ``radial_embedding.bessel_fn.bessel_weights`` sits 2.2120e-07 Å⁻¹ from
      the analytic ``nπ/r_max`` init, the fingerprint recorded independently in
      :mod:`molzoo.mace.checkpoint`'s docstring and §7.1.
    """

    def test_official_omol_load_fills_every_learnable(self) -> None:
        """Every ``nn.Parameter`` is covered; the unhoused keys are the snapshot.

        :meth:`~molzoo.mace.checkpoint.CheckpointRemap.load` already raises on
        an unfilled learnable, so the load itself is half the assertion; the
        explicit subset check states the doctrine where a reader can see it,
        and the count keeps it from holding vacuously on a shrunken model.
        """
        model, state = _official_omol()

        renamed = OMOL_REMAP.rename(state)
        parameters = {name for name, _ in model.named_parameters()}

        assert len(parameters) == OMOL_PARAMETER_COUNT
        assert parameters <= set(renamed)
        assert sorted(set(renamed) - set(model.state_dict())) == OMOL_UNEXPECTED_KEYS

    def test_official_omol_energy_and_forces_match_the_goldens(
        self, omol_cluster: TensorDict
    ) -> None:
        """The loaded surface reproduces the captured energy and forces.

        Five atoms (``O H H C H``, all inside OMol's 83-element table) at the
        package's literal :data:`~tests.test_molzoo.test_mace.conftest.CLUSTER_POS`
        coordinates, neutral closed-shell singlet, every ordered intra-graph
        pair as an edge. Goldens are this machine's own output, not an upstream
        number — see the class docstring.
        """
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
