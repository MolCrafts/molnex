"""Tests for molzoo.mace.spec — the torch-free MACE configuration family.

``spec.py`` is the one module in the MACE sub-package that must stay importable
without paying for the cuEquivariance stack, so half of this file is about what
it may *not* do: no heavy imports, no tensors in ``model_dump()``, and no
transitive load of ``molzoo.mace.encoder`` when the shim hands out a spec class.

The expected defaults are hard-coded from the flat constructors they replace
(``src/molzoo/mace_matpes.py`` ``MACEMatpes.__init__`` and
``src/molzoo/mace_omol.py`` ``MACEOMol.__init__``) — that parity is the whole
point of the field table.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec, MACESpec
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

#: ``MACEMatpes.__init__`` defaults, read from src/molzoo/mace_matpes.py:89-107.
#: ``radial_mlp`` is ``None`` there and materialises as ``[64, 64, 64]`` at
#: mace_matpes.py:112; the spec carries the materialised list.
MATPES_DEFAULTS: dict[str, Any] = {
    "r_max": 6.0,
    "num_bessel": 10,
    "num_polynomial_cutoff": 5,
    "l_max": 3,
    "num_features": 128,
    "max_hidden_l": 1,
    "num_interactions": 2,
    "correlation": 3,
    "mlp_dim": 16,
    "radial_mlp": [64, 64, 64],
    "scale": 1.0,
    "shift": 0.0,
    "use_fallback": False,
    "interaction": "density",
    "readout": "per_layer",
    "distance_transform": "agnesi",
    "pair_repulsion": "zbl",
    "conditioning": "none",
}

#: ``MACEOMol.__init__`` defaults, read from src/molzoo/mace_omol.py:65-85.
#: ``use_fallback`` is not a constructor argument there — the flat model
#: hard-codes the fused cuEq path (mace_omol.py:172, 182), i.e. ``False``.
OMOL_DEFAULTS: dict[str, Any] = {
    "r_max": 6.0,
    "num_bessel": 8,
    "num_polynomial_cutoff": 5,
    "l_max": 3,
    "num_features": 1024,
    "num_interactions": 3,
    "correlation": 2,
    "mlp_dim": 16,
    "edge_channels": 128,
    "charge_classes": 201,
    "charge_offset": 100,
    "spin_classes": 101,
    "spin_offset": 0,
    "scale": 1.0,
    "shift": 0.0,
    "use_fallback": False,
    "interaction": "residual",
    "readout": "final",
    "distance_transform": "none",
    "pair_repulsion": "none",
    "conditioning": "charge_spin",
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

#: Probe run in a fresh interpreter: reaching a spec class through the package
#: shim must not drag in the encoder (and with it the whole cuEq stack).
_LAZY_PROBE = (
    "import sys\n"
    "import molzoo.mace\n"
    "assert molzoo.mace.MACEMatpesSpec is not None\n"
    "print('molzoo.mace.encoder' in sys.modules)\n"
)


def _base_spec(**overrides: Any) -> MACESpec:
    """A valid base ``MACESpec`` with the shared table, plus ``overrides``."""
    return MACESpec(
        atomic_numbers=list(ATOMIC_NUMBERS),
        atomic_energies=list(ATOMIC_ENERGIES),
        **{**BASE_SWITCHES, **overrides},
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


class TestMACESpec:
    """Test the shared MACE configuration base class."""

    def test_variant_switches_are_required(self):
        """The five variant switches have no base default — subclasses set them."""
        with pytest.raises(ValidationError):
            MACESpec(
                atomic_numbers=list(ATOMIC_NUMBERS),
                atomic_energies=list(ATOMIC_ENERGIES),
            )

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

    def test_atomic_energies_stay_a_plain_list(self):
        """Tensor conversion belongs to ``MACEEncoder.__init__``, not the spec."""
        assert _base_spec().atomic_energies == list(ATOMIC_ENERGIES)

    def test_model_dump_is_json_native(self):
        """``model_dump()`` must round-trip through ``json`` — no tensors."""
        json.dumps(_base_spec().model_dump())

    def test_spec_module_imports_no_heavy_dependencies(self):
        """spec.py stays torch-free: only ``typing`` / ``pydantic`` and friends."""
        import molzoo.mace.spec as spec_module

        assert not (_imported_roots(spec_module) & FORBIDDEN_IMPORT_ROOTS)

    def test_reaching_a_spec_class_does_not_import_the_encoder(self):
        """``molzoo.mace.MACEMatpesSpec`` must not trigger the encoder module."""
        molzoo_root = Path(inspect.getsourcefile(sys.modules["molzoo"]) or "").parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(molzoo_root), env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        )

        result = subprocess.run(
            [sys.executable, "-c", _LAZY_PROBE],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        assert result.stdout.strip() == "False"


class TestMACEMatpesSpec:
    """Test the MACE-MatPES configuration."""

    @pytest.mark.parametrize(("field", "expected"), sorted(MATPES_DEFAULTS.items()))
    def test_field_default_matches_the_flat_constructor(self, field: str, expected: Any):
        """Every default equals the ``MACEMatpes.__init__`` default it replaces."""
        spec = MACEMatpesSpec(
            atomic_numbers=list(ATOMIC_NUMBERS),
            atomic_energies=list(ATOMIC_ENERGIES),
        )
        assert getattr(spec, field) == expected

    def test_single_interaction_is_rejected(self):
        """MatPES needs a residual second layer (mace_matpes.py:109-110)."""
        with pytest.raises(ValueError, match="num_interactions"):
            MACEMatpesSpec(
                atomic_numbers=list(ATOMIC_NUMBERS),
                atomic_energies=list(ATOMIC_ENERGIES),
                num_interactions=1,
            )

    def test_tiny_spec_keeps_the_overridden_values(self, tiny_matpes_spec):
        """The shared tiny fixture round-trips its overrides (parity baseline)."""
        assert tiny_matpes_spec.num_features == 16
        assert tiny_matpes_spec.radial_mlp == [8]
        assert tiny_matpes_spec.use_fallback is True


class TestMACEOMolSpec:
    """Test the MACE-OMOL configuration."""

    @pytest.mark.parametrize(("field", "expected"), sorted(OMOL_DEFAULTS.items()))
    def test_field_default_matches_the_flat_constructor(self, field: str, expected: Any):
        """Every default equals the ``MACEOMol.__init__`` default it replaces."""
        spec = MACEOMolSpec(
            atomic_numbers=list(ATOMIC_NUMBERS),
            atomic_energies=list(ATOMIC_ENERGIES),
        )
        assert getattr(spec, field) == expected

    def test_scalar_only_angular_order_is_rejected(self):
        """OMOL's mid-layer edge irreps use ``range(l_max)`` (mace_omol.py:151-153)."""
        with pytest.raises(ValueError, match="l_max"):
            MACEOMolSpec(
                atomic_numbers=list(ATOMIC_NUMBERS),
                atomic_energies=list(ATOMIC_ENERGIES),
                l_max=0,
            )

    def test_tiny_spec_keeps_the_overridden_values(self, tiny_omol_spec):
        """The shared tiny fixture round-trips its overrides (parity baseline)."""
        assert tiny_omol_spec.num_features == 16
        assert tiny_omol_spec.edge_channels == 8
        assert tiny_omol_spec.conditioning == "charge_spin"
