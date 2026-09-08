"""PotentialMeta style registry — schemas, contracts and composition rules.

Mirrors ``src/molpot/forcefield/schema.py``. The registry is the single place
where a style declares what it reads, how it is truncated, which pairs it
excludes, which derivatives it supports and which S-decomposition bucket each
of its named term energies belongs to (S1). Everything downstream — the
``E_local / E_disp / E_perm / E_pol / E_CT`` grouping, the refusal of
unsupported style combinations, the manifest — reads those declarations, so an
unregistered style must raise rather than be synthesised into a blank entry.
"""

from __future__ import annotations

import pytest

from molpot.forcefield import OPLSAAForceField
from molpot.forcefield.schema import PotentialMeta, StyleSchema


class TestPotentialMeta:
    def test_schema_for_improper_periodic_is_registered(self):
        view = PotentialMeta.schema_for(["improper:periodic"])
        schema = view["improper:periodic"]
        assert isinstance(schema, StyleSchema)
        assert schema.energy_formula != "(unregistered style)"

    def test_schema_for_improper_harmonic_is_registered(self):
        view = PotentialMeta.schema_for(["improper:harmonic"])
        schema = view["improper:harmonic"]
        assert isinstance(schema, StyleSchema)
        assert schema.energy_formula != "(unregistered style)"

    def test_every_registered_style_declares_at_least_one_named_term(self) -> None:
        """A style with no term can never appear in the S-decomposition."""
        termless = [key for key, s in PotentialMeta._REGISTRY.items() if not s.contract.terms]
        assert termless == []

    def test_contract_for_bond_harmonic_is_one_local_term_without_a_cutoff(self) -> None:
        from molpot.contract import Contribution, CutoffKind, Derivative, ExclusionClass

        contract = PotentialMeta.contract_for("bond:harmonic")
        assert contract.cutoff is CutoffKind.NONE
        assert contract.exclusion is ExclusionClass.NONE
        assert contract.derivatives == frozenset(
            {Derivative.FORCES, Derivative.VIRIAL, Derivative.HESSIAN}
        )
        assert [(t.key, t.contribution) for t in contract.terms] == [("bonds", Contribution.LOCAL)]

    def test_contract_for_lj_cut_splits_repulsion_from_dispersion(self) -> None:
        """S3: ``4ε(σ/r)¹²`` is local, ``−4ε(σ/r)⁶`` is dispersion."""
        from molpot.contract import Contribution, CutoffKind, ExclusionClass

        contract = PotentialMeta.contract_for("pair:lj/cut")
        assert [(t.key, t.contribution) for t in contract.terms] == [
            ("lj_repulsion", Contribution.LOCAL),
            ("lj_dispersion", Contribution.DISP),
        ]
        assert contract.exclusion is ExclusionClass.PERMANENT
        assert contract.cutoff is CutoffKind.SWITCHED

    def test_contract_for_coulomb_is_one_permanent_electrostatic_term(self) -> None:
        from molpot.contract import Contribution, CutoffKind, ExclusionClass

        contract = PotentialMeta.contract_for("pair:coulomb")
        assert [(t.key, t.contribution) for t in contract.terms] == [("coulomb", Contribution.PERM)]
        assert contract.exclusion is ExclusionClass.PERMANENT
        # Open boundaries truncate; a cell switches the same style to Ewald.
        assert contract.cutoff in {CutoffKind.HARD, CutoffKind.EWALD}

    def test_contract_for_unregistered_style_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="pair:not/a/style"):
            PotentialMeta.contract_for("pair:not/a/style")

    def test_schema_for_unregistered_style_raises_instead_of_synthesising(self) -> None:
        """F6: a typo'd style silently contributed zero energy before."""
        with pytest.raises(KeyError, match="pair:not/a/style"):
            PotentialMeta.schema_for(["bond:harmonic", "pair:not/a/style"])

    def test_check_composition_accepts_the_full_oplsaa_style_list(self) -> None:
        assert PotentialMeta.check_composition(OPLSAAForceField().style_names()) is None

    def test_check_composition_accepts_one_polarization_style_beside_coulomb(
        self, fake_pol_styles: tuple[str, str]
    ) -> None:
        """Rule 2 is "a polarization style needs a permanent partner" — nothing more.

        Pairing the two by cutoff *family* (HARD ↔ HARD, EWALD ↔ EWALD) is
        deferred to composable-a1-01: ``pair:coulomb`` declares ``EWALD`` as a
        policy label covering both the open-boundary cutoff and the lattice sum,
        so a real-space-truncated induced response composed with it has to be
        accepted here rather than refused on a label mismatch that says nothing
        about which solver actually runs.
        """
        assert (
            PotentialMeta.check_composition(
                ["bond:harmonic", "pair:lj/cut", "pair:coulomb", fake_pol_styles[0]]
            )
            is None
        )

    def test_check_composition_rejects_two_polarization_styles(
        self, fake_pol_styles: tuple[str, str]
    ) -> None:
        """Two induced-dipole responses on the same atoms double-count induction."""
        with pytest.raises(ValueError):
            PotentialMeta.check_composition(["pair:coulomb", *fake_pol_styles])

    def test_check_composition_rejects_polarization_without_permanent_electrostatics(
        self, fake_pol_styles: tuple[str, str]
    ) -> None:
        """An induced dipole needs a permanent field to respond to."""
        with pytest.raises(ValueError):
            PotentialMeta.check_composition([fake_pol_styles[0]])

    def test_check_composition_rejects_an_unregistered_style(self) -> None:
        with pytest.raises(KeyError, match="pair:not/a/style"):
            PotentialMeta.check_composition(["bond:harmonic", "pair:not/a/style"])


class TestForceFieldSchemaView:
    def test_as_dict_carries_the_contract_and_its_named_terms(self) -> None:
        view = PotentialMeta.schema_for(["pair:lj/cut"])
        contract = view.as_dict()["styles"]["pair:lj/cut"]["contract"]
        assert [term["key"] for term in contract["terms"]] == ["lj_repulsion", "lj_dispersion"]
        assert [term["contribution"] for term in contract["terms"]] == ["local", "disp"]
