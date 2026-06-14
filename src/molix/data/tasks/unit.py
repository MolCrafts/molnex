"""Unit-conversion task — per-field explicit ``(source, target)`` pint units.

Each field in ``conversions`` declares the unit it *is in* and the unit you
want it *rescaled to*. No preset bundles; whatever you type is what you get.

The shared pint registry comes from :mod:`molpy.core.unit` (so every
MolCrafts component resolves unit strings against the same registry).

Example — QM9 U0 in Hartree, train in eV::

    UnitConvert({"U0": ("hartree", "eV")})

Multiple fields, including derived units::

    UnitConvert({
        "U0":     ("hartree",        "eV"),
        "forces": ("hartree / bohr", "eV / angstrom"),
    })

Conversion factors are resolved once at ``__init__`` via pint; the per-sample
hot path is one scalar multiply per field.
"""

from __future__ import annotations

from molpy import UnitSystem

from molix.data.task import SampleTask


class UnitConvert(SampleTask):
    """Rescale selected target tensors from one pint unit to another.

    Args:
        conversions: Mapping ``{target_key: (src_unit, dst_unit)}``. Each
            unit is any pint-parseable string (``"hartree"``, ``"eV"``,
            ``"hartree / bohr"``, …). Source and target must be
            dimensionally compatible; pint raises on mismatch.

    Raises:
        ValueError: ``conversions`` is empty, or a resolved factor is
            non-finite.
        pint.errors.DimensionalityError: ``src_unit`` and ``dst_unit`` are
            not inter-convertible.
    """

    def __init__(self, conversions: dict[str, tuple[str, str]]) -> None:
        if not conversions:
            raise ValueError("UnitConvert needs at least one field to convert.")

        factors: dict[str, float] = {}
        units_repr: dict[str, tuple[str, str]] = {}
        for key, pair in conversions.items():
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(
                    f"UnitConvert: value for '{key}' must be a 2-tuple "
                    f"(src_unit, dst_unit), got {pair!r}"
                )
            src_str, dst_str = pair
            src_unit = UnitSystem.Unit(src_str)
            dst_unit = UnitSystem.Unit(dst_str)
            factor = float((1.0 * src_unit).to(dst_unit).magnitude)
            if not _is_finite(factor):
                raise ValueError(
                    f"UnitConvert: non-finite factor for '{key}' ({src_str} → {dst_str})"
                )
            factors[str(key)] = factor
            units_repr[str(key)] = (str(src_unit), str(dst_unit))

        self.factors: dict[str, float] = factors
        self._units: dict[str, tuple[str, str]] = units_repr

    @property
    def task_id(self) -> str:
        """Cache-key identity ``unit_convert:<key>:<src>-><dst>,...`` (sorted)."""
        body = ",".join(
            f"{k}:{self._units[k][0]}->{self._units[k][1]}" for k in sorted(self.factors)
        )
        return f"unit_convert:{body}"

    def execute(self, data: dict) -> dict:
        """Rescale each configured target by its precomputed pint factor.

        Multiplies ``targets[key]`` by the conversion factor resolved at
        construction time (one scalar multiply per field).

        Args:
            data: A sample dict whose ``targets`` sub-dict holds every
                configured key.

        Returns:
            A new sample dict with the converted targets.

        Raises:
            KeyError: A configured target key is missing from the sample.
        """
        targets = dict(data.get("targets", {}))
        for key, factor in self.factors.items():
            if key not in targets:
                raise KeyError(
                    f"UnitConvert: target '{key}' not present in sample "
                    f"(available: {sorted(targets)})"
                )
            targets[key] = targets[key] * factor
        return {**data, "targets": targets}


def _is_finite(x: float) -> bool:
    return x == x and x != float("inf") and x != float("-inf")
