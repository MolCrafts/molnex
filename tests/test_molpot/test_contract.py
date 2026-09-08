"""Interaction-contract vocabulary — mirrors ``src/molpot/contract.py``.

The contract types are the S1 declaration surface every registered style
carries: what a term reads, how it is cut off, which pairs it excludes, which
derivatives it supports, and which S-decomposition bucket
(``E = E_local + E_disp + E_perm + E_pol + E_CT``) each of its named energies
falls into. They are torch-free value objects, so this module only pins their
identity: enum spellings (they are serialised into the manifest and into
``ForceFieldSchemaView.as_dict()``), immutability, and hashability.

Reference:
    ``.claude/notes/composable-potentials.md`` §"S1 contract"
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from molpot.contract import (
    Contribution,
    CutoffKind,
    Derivative,
    ExclusionClass,
    InteractionContract,
    ParamFrame,
    ParamSpec,
    TermSpec,
)

#: This checkout's ``src/`` — the leaf-import probe below runs in a fresh
#: interpreter, which has to resolve ``molpot`` from here and not from whatever
#: else is installed.
_SRC = Path(__file__).resolve().parents[2] / "src"

#: Wall-clock ceiling (s) on that probe interpreter.
_PROBE_TIMEOUT = 300

#: Probe: execute ``src/molpot/manifest.py`` with the ``molpot`` package
#: *stubbed* — its eager ``__init__`` (which imports the whole tree, ForceField
#: included) never runs — and every ``molpot.forcefield`` import refused. What
#: is left to resolve is exactly ``molpot.manifest``'s own import list, so the
#: probe exits non-zero iff the manifest leaf reaches up into the ForceField
#: package. That is the layering ``molpot/manifest.py``'s docstring claims and
#: that ``ParamSpec`` living in :mod:`molpot.contract` is what buys: today the
#: import only survives because ``container`` happens to import
#: ``forcefield.schema`` before it imports the manifest.
_LEAF_IMPORT_PROBE = """
import importlib
import sys
import types
from pathlib import Path


class _RefuseForceField:
    def find_spec(self, name, path=None, target=None):
        if name == "molpot.forcefield" or name.startswith("molpot.forcefield."):
            raise ImportError(f"molpot.manifest must not need {name}")
        return None


package = types.ModuleType("molpot")
package.__path__ = [str(Path(sys.argv[1]) / "molpot")]
sys.modules["molpot"] = package
sys.meta_path.insert(0, _RefuseForceField())
importlib.import_module("molpot.manifest")
"""


def _contract(**overrides: object) -> InteractionContract:
    """A minimal contract: the ``pair:lj/cut`` shape, overridable field by field."""
    fields: dict[str, object] = {
        "inputs": ("atoms.pos",),
        "cutoff": CutoffKind.SWITCHED,
        "exclusion": ExclusionClass.PERMANENT,
        "derivatives": frozenset({Derivative.FORCES, Derivative.VIRIAL}),
        "terms": (TermSpec("lj_repulsion", Contribution.LOCAL),),
    }
    fields.update(overrides)
    return InteractionContract(**fields)  # ty: ignore[invalid-argument-type]


class TestContribution:
    def test_values_are_the_five_s_decomposition_labels(self) -> None:
        """``E = E_local + E_disp + E_perm + E_pol + E_CT`` — one member each."""
        assert [c.value for c in Contribution] == ["local", "disp", "perm", "pol", "ct"]
        assert Contribution.LOCAL == "local"


class TestCutoffKind:
    def test_values_name_every_supported_range_treatment(self) -> None:
        assert [k.value for k in CutoffKind] == [
            "none",
            "hard",
            "switched",
            "ewald",
            "gto_kspace",
        ]


class TestExclusionClass:
    def test_values_separate_permanent_from_induced_exclusions(self) -> None:
        assert [e.value for e in ExclusionClass] == ["none", "permanent", "induced"]


class TestDerivative:
    def test_values_name_the_three_supported_derivative_orders(self) -> None:
        assert [d.value for d in Derivative] == ["forces", "virial", "hessian"]


class TestParamFrame:
    def test_values_name_the_transformation_law_of_a_parameter(self) -> None:
        assert [f.value for f in ParamFrame] == [
            "scalar",
            "vector_lab",
            "vector_local",
            "rank2",
        ]

    def test_is_re_exported_under_its_disambiguated_name_from_the_package_root(self) -> None:
        """``molpy.Frame`` is a core repo type; the parameter law is ``ParamFrame``.

        The old ``Frame`` spelling was deliberately kept out of ``molpot``'s
        namespace to avoid shadowing the molpy frame on ``from molpot import *``
        — under the unambiguous name the enum belongs on the package surface,
        next to the rest of the contract vocabulary.
        """
        from molpot import ParamFrame as exported

        assert exported is ParamFrame


class TestTermSpec:
    def test_is_frozen_so_a_registered_label_cannot_be_rebound(self) -> None:
        term = TermSpec("lj_dispersion", Contribution.DISP)
        with pytest.raises(dataclasses.FrozenInstanceError):
            term.contribution = Contribution.LOCAL  # ty: ignore[invalid-assignment]


class TestInteractionContract:
    def test_response_defaults_to_none_for_a_non_polarizable_term(self) -> None:
        assert _contract().response is None

    def test_is_frozen_so_a_registered_contract_cannot_be_rebound(self) -> None:
        contract = _contract()
        with pytest.raises(dataclasses.FrozenInstanceError):
            contract.cutoff = CutoffKind.HARD  # ty: ignore[invalid-assignment]

    def test_is_hashable_so_contracts_can_be_deduplicated(self) -> None:
        assert len({_contract(), _contract()}) == 1

    def test_derivatives_is_an_unordered_frozen_set(self) -> None:
        """Order must not distinguish two otherwise identical contracts."""
        forward = _contract(derivatives=frozenset({Derivative.FORCES, Derivative.VIRIAL}))
        reversed_ = _contract(derivatives=frozenset({Derivative.VIRIAL, Derivative.FORCES}))
        assert isinstance(forward.derivatives, frozenset)
        assert forward == reversed_


class TestParamSpec:
    """The parameter half of a style's S1 declaration, and where it lives.

    ``ParamSpec`` belongs in this torch-free leaf next to ``ParamFrame`` — the
    type of its own ``frame`` field — and is re-exported by
    ``molpot.forcefield.schema`` for the style registry that consumes it. The
    ownership is not cosmetic: ``molpot.manifest`` records a style's
    parameters, and with the class here it can sit *below* the ForceField
    package instead of importing one of its submodules.
    """

    def test_is_owned_by_the_contract_leaf_and_re_exported_by_the_schema(self) -> None:
        """One class, two spellings — the registry's import keeps working."""
        from molpot.forcefield import schema

        assert ParamSpec.__module__ == "molpot.contract"
        assert schema.ParamSpec is ParamSpec

    def test_frame_defaults_to_scalar_so_only_directional_params_declare_a_law(self) -> None:
        """A force constant is rotation invariant; only a dipole says otherwise."""
        assert ParamSpec("k", "[n_bond_types]", "kcal/mol/Å²").frame is ParamFrame.SCALAR

    def test_is_frozen_so_a_registered_parameter_cannot_be_rebound(self) -> None:
        """Specs are serialised into the manifest; a mutable one would lie."""
        spec = ParamSpec("sigma", "[n_atom_types]", "Å")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.unit = "nm"  # ty: ignore[invalid-assignment]

    @pytest.mark.slow
    def test_the_manifest_leaf_imports_without_the_forcefield_package(self) -> None:
        """``molpot.manifest`` must resolve with every ForceField import refused.

        Marked ``slow``: the assertion is only meaningful in an interpreter
        that has not already imported ``molpot`` (this session has), so it
        costs a fresh process and a torch import.
        """
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(_SRC), environment.get("PYTHONPATH", "")) if part
        )
        completed = subprocess.run(
            [sys.executable, "-c", _LEAF_IMPORT_PROBE, str(_SRC)],
            capture_output=True,
            text=True,
            env=environment,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
        assert completed.returncode == 0, (
            f"molpot.manifest did not import as a leaf ({completed.returncode}):\n"
            f"{completed.stderr}"
        )
