"""UnitTag + CLASS_I_CANONICAL contract (learnable-classical-ff-01-ir-kernels)."""

from collections.abc import Mapping

from molpot.ir import CLASS_I_CANONICAL, UnitTag


class TestUnitTag:
    """UnitTag enumerates the physical dimensions named by the Potential IR."""

    def test_required_dimension_members_exist(self):
        names = {m.name.lower() if hasattr(m, "name") else str(m).lower() for m in UnitTag}
        # Accept either Enum members or string values for the dimension tags.
        values = set()
        for m in UnitTag:
            if hasattr(m, "value"):
                values.add(str(m.value).lower())
            values.add(str(m).lower().split(".")[-1])
        combined = names | values
        for required in (
            "energy",
            "length",
            "charge",
            "angle",
        ):
            assert any(required in c for c in combined), f"missing UnitTag for {required}"


class TestClassICanonical:
    """CLASS_I_CANONICAL freezes Class-I SI-free internal units."""

    def test_energy_is_kcal_per_mol(self):
        assert CLASS_I_CANONICAL["energy"] == "kcal/mol"

    def test_length_is_angstrom(self):
        length = CLASS_I_CANONICAL["length"]
        assert length in ("angstrom", "Å", "A")

    def test_charge_is_elementary_e(self):
        assert CLASS_I_CANONICAL["charge"] == "e"

    def test_angle_is_radian(self):
        angle = CLASS_I_CANONICAL["angle"]
        assert angle in ("radian", "rad")

    def test_mapping_is_frozen_not_silently_mutable(self):
        # MappingProxyType / frozendict / Final: mutation must raise or be a no-op
        # that leaves the public mapping unchanged.
        original = dict(CLASS_I_CANONICAL)
        raised = False
        try:
            CLASS_I_CANONICAL["energy"] = "kJ/mol"  # type: ignore[index]
        except (TypeError, AttributeError, ValueError):
            raised = True
        if not raised:
            # If assignment did not raise, the public view must still be unchanged.
            assert dict(CLASS_I_CANONICAL) == original
            assert CLASS_I_CANONICAL["energy"] == "kcal/mol"
        else:
            assert CLASS_I_CANONICAL["energy"] == "kcal/mol"

    def test_is_mapping_with_required_keys(self):
        assert isinstance(CLASS_I_CANONICAL, Mapping)
        for key in ("energy", "length", "charge", "angle"):
            assert key in CLASS_I_CANONICAL
