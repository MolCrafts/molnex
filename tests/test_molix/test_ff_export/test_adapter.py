"""BackendAdapter — the flat exclusion-scale spelling shared by every backend."""

from __future__ import annotations

from molix.ff_export.adapter import BackendAdapter
from molpot.ir import NonbondedScaling


class TestBackendAdapter:
    def test_scaling_dict_keys_follow_declared_kinds(self):
        """One ``scale_<kind>_<12|13|14>`` key per declared interaction class.

        Derived from :meth:`NonbondedScaling.kinds` / ``triple`` rather than
        from a hard-coded pair, so a class a later model declares (``induced``
        for a polarizable force field) cannot be dropped without this failing.
        """
        scaling = NonbondedScaling()
        expected = {
            f"scale_{kind}_{n}": value
            for kind in scaling.kinds()
            for n, value in zip((12, 13, 14), scaling.triple(kind), strict=True)
        }
        flat = BackendAdapter.scaling_dict(scaling)
        assert flat == expected
        assert set(flat) == {
            "scale_charge_12",
            "scale_charge_13",
            "scale_charge_14",
            "scale_lj_12",
            "scale_lj_13",
            "scale_lj_14",
        }

    def test_no_scaling_declared_stays_none(self):
        """``None`` means "no exclusion policy declared", not "no exclusions"."""
        assert BackendAdapter.scaling_dict(None) is None
