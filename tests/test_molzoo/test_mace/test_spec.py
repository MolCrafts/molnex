"""Tests for molzoo.mace.spec — the torch-free MACE configuration family.

``spec.py`` is the torch-free module of the MACE sub-package, so part of this
file is about what it may *not* do: no heavy imports (an AST check of its own
source — the package ``__init__`` imports the encoder eagerly, so the module is
only reachable through torch) and no tensors in ``model_dump()``. The rest pins
each family's rules: what a spec rejects, and what its ``from_official`` reads
and refuses from a typed official config.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from molzoo.mace.spec import (
    MACELesSpec,
    MACEMatpesSpec,
    MaceOfficialConfig,
    MACEOffSpec,
    MACEOMolSpec,
    MACEPolarSpec,
    MACESpec,
    parse_irreps,
)
from tests.test_molzoo.test_mace.conftest import ATOMIC_ENERGIES, ATOMIC_NUMBERS

#: The five variant switches are required on the base and defaulted on the
#: subclasses, so every direct ``MACESpec`` construction has to spell them out.
BASE_SWITCHES: dict[str, str] = {
    "interaction": "density",
    "readout": "per_layer",
    "distance_transform": "none",
    "pair_repulsion": "none",
    "conditioning": "none",
}

#: The four conditioning sizes, as the OMOL defaults state them.
CONDITIONING_SIZES: dict[str, int] = {
    "charge_classes": 201,
    "charge_offset": 100,
    "spin_classes": 101,
    "spin_offset": 0,
}

#: Import roots that would defeat the point of a torch-free config module.
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "torch",
        "cuequivariance",
        "cuequivariance_torch",
        "tensordict",
        "molrep",
        "molix",
        "molpot",
        "molzoo",
    }
)

#: The official MatPES-shaped config every ``from_official`` case starts from.
OFFICIAL_BASE: dict[str, Any] = {
    "atomic_numbers": ATOMIC_NUMBERS,
    "atomic_energies": ATOMIC_ENERGIES,
    "r_max": 5.0,
    "num_bessel": 4,
    "num_polynomial_cutoff": 5,
    "max_ell": 1,
    "correlation": 2,
    "num_interactions": 2,
    "hidden_irreps": "16x0e+16x1o",
    "MLP_irreps": "8x0e",
    "radial_MLP": [8],
    "atomic_inter_scale": 1.5,
    "atomic_inter_shift": -0.25,
}

#: The LES block of an official MACE-LES dump this port accepts.
OFFICIAL_LES: dict[str, Any] = {
    "ewald": "Ewald",
    "sigma": 1.2,
    "dl": 2.5,
    "norm_factor": 90.0474,
    "remove_self_interaction": False,
    "output_scale": 0.5,
}


def _base_spec(**overrides: Any) -> MACESpec:
    """A valid base ``MACESpec`` with the shared table, plus ``overrides``."""
    return MACESpec(
        **{
            "atomic_numbers": list(ATOMIC_NUMBERS),
            "atomic_energies": list(ATOMIC_ENERGIES),
            **BASE_SWITCHES,
            **overrides,
        }
    )


def _official(**overrides: Any) -> MaceOfficialConfig:
    """A typed official config on :data:`OFFICIAL_BASE`, plus ``overrides``."""
    return MaceOfficialConfig(**{**OFFICIAL_BASE, **overrides})


def _off_official(**overrides: Any) -> MaceOfficialConfig:
    """The MACE-OFF shape: an agnostic stack with its neighbour normalisation."""
    return _official(
        **{
            "interaction_blocks": [
                "RealAgnosticInteractionBlock",
                "RealAgnosticResidualInteractionBlock",
            ],
            "avg_num_neighbors": 5.0,
            **overrides,
        }
    )


def _imported_roots(module: Any) -> set[str]:
    """Top-level package names imported by ``module``'s source, via ``ast``."""
    source = Path(inspect.getsourcefile(module) or "").read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — resolves inside molzoo
                roots.add("molzoo")
            elif node.module:
                roots.add(node.module.split(".")[0])
    return roots


class TestParseIrreps:
    def test_scalar_multiplicity_and_highest_order(self):
        assert parse_irreps("128x0e+128x1o", "hidden_irreps") == (128, 1)
        assert parse_irreps("16x0e", "MLP_irreps") == (16, 0)

    def test_a_non_irreps_string_is_rejected(self):
        with pytest.raises(ValueError, match="hidden_irreps"):
            parse_irreps("128", "hidden_irreps")

    def test_a_string_without_a_scalar_term_is_rejected(self):
        """The channel width is the ``0e`` multiplicity and has no default."""
        with pytest.raises(ValueError, match="0e"):
            parse_irreps("8x1o", "hidden_irreps")


class TestMaceOfficialConfig:
    def test_upstream_spellings_are_aliases(self):
        """``max_ell`` / ``atomic_inter_scale`` / ``radial_MLP`` / ``MLP_irreps`` resolve."""
        cfg = _official()
        assert (cfg.l_max, cfg.scale, cfg.shift, cfg.radial_mlp, cfg.mlp_irreps) == (
            1,
            1.5,
            -0.25,
            [8],
            "8x0e",
        )

    def test_unknown_keys_are_ignored(self):
        """An official config carries far more than the port reads."""
        assert _official(some_upstream_training_key=3).hidden_irreps == "16x0e+16x1o"

    def test_interaction_class_names_build_the_stack(self):
        """The json dialect's first class, then the other ``num_interactions - 1`` times."""
        cfg = _official(
            interaction_cls_first="RealAgnosticInteractionBlock",
            interaction_cls="RealAgnosticResidualInteractionBlock",
            num_interactions=3,
        )
        assert cfg.interaction_blocks == [
            "RealAgnosticInteractionBlock",
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticResidualInteractionBlock",
        ]

    def test_body_fields_leaves_out_what_the_archive_does_not_carry(self):
        cfg = MaceOfficialConfig(atomic_numbers=[1], atomic_energies=[0.0], r_max=4.0)
        assert cfg.body_fields() == {"r_max": 4.0}

    def test_irreps_widths_derives_the_three_widths(self):
        assert _official().irreps_widths() == (16, 1, 8)

    @pytest.mark.parametrize("absent", ["hidden_irreps", "MLP_irreps"])
    def test_irreps_widths_names_a_missing_string(self, absent):
        """No silent fallback to a default width — name the missing key."""
        with pytest.raises(KeyError, match=absent):
            _official(**{absent: None}).irreps_widths()


class TestMACESpec:
    """Test the shared MACE configuration base class."""

    def test_variant_switches_are_required(self):
        """The five variant switches have no base default — subclasses set them."""
        with pytest.raises(ValidationError):
            MACESpec(
                atomic_numbers=list(ATOMIC_NUMBERS),
                atomic_energies=list(ATOMIC_ENERGIES),
            )

    def test_unknown_fields_are_rejected(self):
        """``extra="forbid"``: a mistyped field cannot silently leave the default."""
        with pytest.raises(ValidationError):
            _base_spec(num_feature=16)

    def test_atomic_energies_length_must_match_the_table(self):
        """One ``E0`` per element — a short list would silently mis-index."""
        with pytest.raises(ValueError, match="atomic_energies"):
            MACESpec(
                atomic_numbers=[1, 6, 8],
                atomic_energies=[-13.6, -1029.0],
                **BASE_SWITCHES,
            )

    def test_non_ascending_atomic_numbers_are_rejected(self):
        """``torch.searchsorted`` needs an ordered table; unordered = wrong energy."""
        with pytest.raises(ValueError, match="atomic_numbers"):
            MACESpec(
                atomic_numbers=[8, 1, 6],
                atomic_energies=list(ATOMIC_ENERGIES),
                **BASE_SWITCHES,
            )

    def test_duplicated_atomic_numbers_are_rejected(self):
        """A duplicate row makes the one-hot ambiguous — strictly ascending only."""
        with pytest.raises(ValueError, match="atomic_numbers"):
            MACESpec(
                atomic_numbers=[1, 6, 6],
                atomic_energies=list(ATOMIC_ENERGIES),
                **BASE_SWITCHES,
            )

    def test_zero_num_bessel_is_rejected(self):
        """``num_bessel`` is a positive count (``Field(gt=0)``)."""
        with pytest.raises(ValidationError):
            _base_spec(num_bessel=0)

    def test_zero_r_max_is_rejected(self):
        """A zero cutoff has no neighbours (``Field(gt=0.0)``); units are Å."""
        with pytest.raises(ValidationError):
            _base_spec(r_max=0.0)

    def test_agnostic_interactions_need_avg_num_neighbors(self):
        with pytest.raises(ValueError, match="avg_num_neighbors"):
            _base_spec(interaction="agnostic")

    def test_atomic_energies_read_any_iterable_into_a_list(self):
        """A tensor or a tuple of ints becomes the plain ``list[float]`` the spec holds."""
        tensor = torch.tensor(ATOMIC_ENERGIES, dtype=torch.float64)
        assert _base_spec(atomic_energies=tensor).atomic_energies == list(ATOMIC_ENERGIES)
        assert _base_spec(atomic_energies=(-13, -1029, -2041)).atomic_energies == [
            -13.0,
            -1029.0,
            -2041.0,
        ]

    def test_conditioning_requires_all_four_sizes(self):
        with pytest.raises(ValueError, match="charge_classes"):
            _base_spec(
                conditioning="charge_spin", charge_classes=3, charge_offset=0, spin_classes=2
            )

    def test_an_unconditioned_spec_rejects_the_sizes(self):
        with pytest.raises(ValueError, match="charge_classes"):
            _base_spec(spin_offset=0)

    def test_a_conditioned_spec_carries_the_sizes(self):
        spec = _base_spec(conditioning="charge_spin", **CONDITIONING_SIZES)
        assert (spec.charge_classes, spec.spin_classes) == (201, 101)

    def test_the_base_states_no_body_schedule(self):
        """A bare ``MACESpec`` describes no shipped model."""
        with pytest.raises(NotImplementedError):
            _base_spec().body_schedule()

    def test_model_dump_is_json_native(self):
        """``model_dump()`` must round-trip through ``json`` — no tensors."""
        json.dumps(_base_spec().model_dump())

    def test_spec_module_imports_no_heavy_dependencies(self):
        """spec.py stays torch-free: only ``re`` / ``typing`` / ``pydantic`` and friends."""
        import molzoo.mace.spec as spec_module

        assert not (_imported_roots(spec_module) & FORBIDDEN_IMPORT_ROOTS)


class TestMACEMatpesSpec:
    """Test the MACE-MatPES configuration."""

    def test_single_interaction_is_rejected(self):
        """MatPES needs a residual second layer."""
        with pytest.raises(ValueError, match="num_interactions"):
            MACEMatpesSpec(
                atomic_numbers=list(ATOMIC_NUMBERS),
                atomic_energies=list(ATOMIC_ENERGIES),
                num_interactions=1,
            )

    def test_body_schedule_is_the_density_stack(self, tiny_matpes_spec):
        schedule = tiny_matpes_spec.body_schedule()
        assert (schedule.hidden_l, schedule.radial_mlp, schedule.shared_product) == (1, (8,), False)
        assert schedule.residual_first is False

    def test_from_official_reads_the_widths_and_the_shared_fields(self):
        spec = MACEMatpesSpec.from_official(
            _official(pair_repulsion=True, distance_transform="Agnesi")
        )
        assert (spec.num_features, spec.max_hidden_l, spec.mlp_dim) == (16, 1, 8)
        assert (spec.l_max, spec.radial_mlp, spec.scale, spec.shift) == (1, [8], 1.5, -0.25)
        assert (spec.pair_repulsion, spec.distance_transform) == ("zbl", "agnesi")

    def test_from_official_tolerates_a_config_without_the_switches(self):
        """Absence is not a contradiction: older dumps predate the two keys."""
        spec = MACEMatpesSpec.from_official(_official())
        assert (spec.pair_repulsion, spec.distance_transform) == ("zbl", "agnesi")

    @pytest.mark.parametrize(
        ("switches", "offender", "spelling"),
        [
            ({"pair_repulsion": False}, "pair_repulsion", "False"),
            ({"distance_transform": None}, "distance_transform", "None"),
            ({"distance_transform": "polynomial"}, "distance_transform", "polynomial"),
        ],
        ids=["pair_repulsion_off", "distance_transform_null", "distance_transform_other"],
    )
    def test_from_official_rejects_a_contradicted_hard_wired_switch(
        self, switches, offender, spelling
    ):
        """ZBL and Agnesi are hard-wired buffers the strict weight load cannot notice.

        A ``None`` the config *states* is a contradiction, unlike an absent key;
        the message names the key and both values, since the fix is to build
        the spec by hand.
        """
        with pytest.raises(ValueError, match=offender) as raised:
            MACEMatpesSpec.from_official(_official(**switches))
        assert spelling in str(raised.value)

    @pytest.mark.parametrize("absent", ["hidden_irreps", "MLP_irreps"])
    def test_from_official_needs_the_irreps(self, absent):
        with pytest.raises(KeyError, match=absent):
            MACEMatpesSpec.from_official(_official(**{absent: None}))


class TestMACEOMolSpec:
    """Test the MACE-OMOL configuration."""

    def test_scalar_only_angular_order_is_rejected(self):
        """OMOL's mid-layer edge irreps use ``range(l_max)``."""
        with pytest.raises(ValueError, match="l_max"):
            MACEOMolSpec(
                atomic_numbers=list(ATOMIC_NUMBERS),
                atomic_energies=list(ATOMIC_ENERGIES),
                l_max=0,
            )

    def test_body_schedule_shares_the_product(self, tiny_omol_spec):
        """OMOL's product basis is one set of weights for every element."""
        assert tiny_omol_spec.body_schedule().shared_product is True

    def test_from_official_takes_the_explicit_widths(self):
        spec = MACEOMolSpec.from_official(_official(num_features=32, mlp_dim=4, edge_channels=8))
        assert (spec.num_features, spec.mlp_dim, spec.edge_channels) == (32, 4, 8)
        assert spec.num_bessel == 4

    def test_from_official_keeps_the_defaults_for_absent_widths(self):
        spec = MACEOMolSpec.from_official(_official())
        assert (spec.num_features, spec.edge_channels, spec.charge_classes) == (1024, 128, 201)


class TestMACEOffSpec:
    def test_residual_first_reaches_the_schedule(self, tiny_off_spec):
        residual = tiny_off_spec.model_copy(update={"residual_first": True})
        assert residual.body_schedule().residual_first is True

    def test_from_official_reads_the_stack(self):
        spec = MACEOffSpec.from_official(_off_official())
        assert (spec.avg_num_neighbors, spec.residual_first) == (5.0, False)
        assert (spec.num_features, spec.max_hidden_l, spec.mlp_dim) == (16, 1, 8)

    def test_from_official_sees_a_residual_first_layer(self):
        cfg = _off_official(interaction_blocks=["RealAgnosticResidualInteractionBlock"] * 2)
        assert MACEOffSpec.from_official(cfg).residual_first is True

    def test_from_official_rejects_another_stack(self):
        cfg = _off_official(interaction_blocks=["RealAgnosticDensityInteractionBlock"] * 2)
        with pytest.raises(ValueError, match="RealAgnostic"):
            MACEOffSpec.from_official(cfg)

    @pytest.mark.parametrize(
        "switch",
        [{"pair_repulsion": True}, {"distance_transform": "Agnesi"}],
        ids=["zbl", "agnesi"],
    )
    def test_from_official_rejects_a_density_family_switch(self, switch):
        """MACE-OFF has neither; a MatPES config must not build an OFF model."""
        with pytest.raises(ValueError, match="MACE-OFF has no"):
            MACEOffSpec.from_official(_off_official(**switch))

    def test_from_official_needs_avg_num_neighbors(self):
        with pytest.raises(KeyError, match="avg_num_neighbors"):
            MACEOffSpec.from_official(_off_official(avg_num_neighbors=None))


class TestMACELesSpec:
    def test_from_official_reads_the_les_block(self):
        spec = MACELesSpec.from_official(_off_official(les=OFFICIAL_LES))
        assert (spec.les_sigma, spec.les_dl, spec.les_output_scale) == (1.2, 2.5, 0.5)
        assert spec.les_remove_self_interaction is False
        assert spec.avg_num_neighbors == 5.0

    def test_from_official_needs_the_les_block(self):
        with pytest.raises(KeyError, match="LES"):
            MACELesSpec.from_official(_off_official())

    @pytest.mark.parametrize(
        "les",
        [{**OFFICIAL_LES, "ewald": "EwaldPotential"}, {**OFFICIAL_LES, "use_dipoles": True}],
        ids=["other_ewald", "latent_dipoles"],
    )
    def test_from_official_rejects_what_is_not_ported(self, les):
        """Only the charge-only legacy LES Ewald is ported."""
        with pytest.raises(ValueError, match="charge-only"):
            MACELesSpec.from_official(_off_official(les=les))


class TestMACEPolarSpec:
    def test_body_schedule_keeps_the_last_layer_irreps(self, tiny_polar_spec):
        """Polar reads out the full hidden irreps of its last layer, not a scalar narrowing."""
        assert tiny_polar_spec.body_schedule().keep_last_layer_irreps is True

    def test_from_official_reads_the_head_and_the_runtime_cutoff(self):
        cfg = _official(
            num_recursion_steps=3,
            atomic_multipoles_max_l=1,
            multipole_sigma=1.2,
            kspace_cutoff_factor=1.5,
            kspace_cutoff=1.25,
            include_electrostatic_self_interaction=False,
            field_si=True,
            edge_channels=8,
        )
        spec = MACEPolarSpec.from_official(cfg)
        assert (spec.num_recursion_steps, spec.multipole_sigma, spec.edge_channels) == (3, 1.2, 8)
        assert (spec.kspace_cutoff, spec.kspace_cutoff_factor) == (1.25, 1.5)
        assert (spec.electrostatic_self_interaction, spec.field_self_interaction) == (False, True)
        assert (spec.num_features, spec.max_hidden_l, spec.mlp_dim) == (16, 1, 8)

    def test_from_official_takes_explicit_widths_without_irreps(self):
        cfg = _official(
            hidden_irreps=None, MLP_irreps=None, num_features=32, max_hidden_l=1, mlp_dim=4
        )
        spec = MACEPolarSpec.from_official(cfg)
        assert (spec.num_features, spec.max_hidden_l, spec.mlp_dim) == (32, 1, 4)

    def test_from_official_lets_a_single_interaction_skip_the_mlp_irreps(self):
        """A single-interaction model has no non-linear readout, hence no ``MLP_irreps``."""
        spec = MACEPolarSpec.from_official(_official(MLP_irreps=None, num_interactions=1))
        assert spec.num_interactions == 1

    def test_from_official_needs_some_widths(self):
        with pytest.raises(KeyError, match="hidden_irreps"):
            MACEPolarSpec.from_official(_official(hidden_irreps=None, MLP_irreps=None))

    @pytest.mark.parametrize(
        "switch",
        [
            {"keep_last_layer_irreps": False},
            {"quadrupole_feature_corrections": True},
            {"use_so3": True},
            {"use_agnostic_product": False},
        ],
        ids=["scalar_last_layer", "quadrupoles", "so3", "element_product"],
    )
    def test_from_official_rejects_an_unported_architecture_switch(self, switch):
        with pytest.raises(ValueError, match="unsupported Polar configuration"):
            MACEPolarSpec.from_official(_official(**switch))
