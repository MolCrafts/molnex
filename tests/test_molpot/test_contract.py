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

import pytest

from molpot.contract import (
    Contribution,
    CutoffKind,
    Derivative,
    ExclusionClass,
    InteractionContract,
    ParamFrame,
    TermSpec,
)


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
