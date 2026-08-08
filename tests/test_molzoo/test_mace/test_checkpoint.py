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

#: Hard-coded goldens: the dimensions of ``MACE-matpes-r2scan-omat-ft`` as
#: published (``hidden_irreps="128x0e+128x1o"``, ``MLP_irreps="16x0e"``). The
#: bit-parity oracle states them literally so ``from_checkpoint`` has to derive
#: the same numbers from the irreps strings on its own.
OFFICIAL_NUM_FEATURES = 128
OFFICIAL_MAX_HIDDEN_L = 1
OFFICIAL_MLP_DIM = 16

_TEST_PACKAGE = Path(__file__).resolve().parent


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
) -> tuple[Path, Path]:
    """Write an official-shaped MatPES config json + cueq ``state_dict``.

    Args:
        directory: Destination directory (``tmp_path``).
        spec: The spec ``model`` was built from; supplies the config values.
        model: Source of the weights, dumped under official cueq names.
        hidden_irreps: Official ``hidden_irreps`` string, e.g. ``"32x0e+32x1o"``.
        mlp_irreps: Official ``MLP_irreps`` string, e.g. ``"8x0e"``.
        drop: Config keys to leave out, for the missing-key cases.

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
