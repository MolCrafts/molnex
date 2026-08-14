"""Potential style meta for ``Potentials.get_schema()``.

Schema is owned by potential forms (not a free-standing ForceFieldSchema type).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ParamSpec", "StyleSchema", "ForceFieldSchemaView", "PotentialMeta"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: str
    unit: str
    description: str = ""


@dataclass(frozen=True)
class StyleSchema:
    category: str
    name: str
    index_arity: int
    params: tuple[ParamSpec, ...]
    energy_formula: str = ""

    @property
    def key(self) -> str:
        return f"{self.category}:{self.name}"


@dataclass
class ForceFieldSchemaView:
    """Aggregated style meta for styles present on one ForceField."""

    styles: dict[str, StyleSchema] = field(default_factory=dict)
    units: str = "real"

    def keys(self) -> list[str]:
        return list(self.styles.keys())

    def __contains__(self, key: object) -> bool:
        return key in self.styles

    def __getitem__(self, key: str) -> StyleSchema:
        return self.styles[key]

    def as_dict(self) -> dict[str, Any]:
        return {
            "units": self.units,
            "styles": {
                k: {
                    "category": s.category,
                    "name": s.name,
                    "index_arity": s.index_arity,
                    "params": [
                        {
                            "name": p.name,
                            "shape": p.shape,
                            "unit": p.unit,
                            "description": p.description,
                        }
                        for p in s.params
                    ],
                    "energy_formula": s.energy_formula,
                }
                for k, s in self.styles.items()
            },
        }


class PotentialMeta:
    """Registry of style → schema (class-level, not free functions)."""

    _REGISTRY: dict[str, StyleSchema] = {
        "bond:harmonic": StyleSchema(
            category="bond",
            name="harmonic",
            index_arity=2,
            energy_formula="0.5 * k * (r - r0)**2",
            params=(
                ParamSpec("k", "[n]", "kcal/mol/Å²", "force constant"),
                ParamSpec("r0", "[n]", "Å", "equilibrium length"),
            ),
        ),
        "angle:harmonic": StyleSchema(
            category="angle",
            name="harmonic",
            index_arity=3,
            energy_formula="0.5 * k * (theta - theta0)**2",
            params=(
                ParamSpec("k", "[n]", "kcal/mol/rad²", "force constant"),
                ParamSpec("theta0", "[n]", "rad", "equilibrium angle"),
            ),
        ),
        "dihedral:periodic": StyleSchema(
            category="dihedral",
            name="periodic",
            index_arity=4,
            energy_formula="sum_m (k_m / s) * [1 + cos(n_m * phi - gamma_m)]",
            params=(
                ParamSpec("k", "[n, n_terms]", "kcal/mol", "barrier heights"),
                ParamSpec("periodicity", "[n_terms]", "1", "integer periods"),
                ParamSpec("phase", "[n, n_terms]", "rad", "phase offsets"),
                ParamSpec("idivf", "[n]", "1", "AMBER idivf scale"),
            ),
        ),
        "dihedral:fourier": StyleSchema(
            category="dihedral",
            name="fourier",
            index_arity=4,
            energy_formula="sum_m (k_m / s) * [1 + cos(n_m * phi - gamma_m)]",
            params=(
                ParamSpec("k", "[n, n_terms]", "kcal/mol", "Fourier amplitudes"),
                ParamSpec("periodicity", "[n_terms]", "1", "integer periods"),
                ParamSpec("phase", "[n, n_terms]", "rad", "phase offsets"),
                ParamSpec("idivf", "[n]", "1", "scale"),
            ),
        ),
        "dihedral:opls": StyleSchema(
            category="dihedral",
            name="opls",
            index_arity=4,
            energy_formula="OPLS 4-term cosine (k1..k4)",
            params=(
                ParamSpec("k1", "[n]", "kcal/mol", "OPLS k1"),
                ParamSpec("k2", "[n]", "kcal/mol", "OPLS k2"),
                ParamSpec("k3", "[n]", "kcal/mol", "OPLS k3"),
                ParamSpec("k4", "[n]", "kcal/mol", "OPLS k4"),
            ),
        ),
        "pair:lj/cut": StyleSchema(
            category="pair",
            name="lj/cut",
            index_arity=2,
            energy_formula="4*eps*((sig/r)**12 - (sig/r)**6)",
            params=(
                ParamSpec("epsilon", "[n_atom_types]", "kcal/mol", "well depth"),
                ParamSpec("sigma", "[n_atom_types]", "Å", "length scale"),
            ),
        ),
        "pair:coulomb": StyleSchema(
            category="pair",
            name="coulomb",
            index_arity=2,
            energy_formula="ke * qi*qj/r with 1-2/1-3/1-4 scales",
            params=(ParamSpec("q", "[n_atoms]", "e", "partial charges on atoms"),),
        ),
    }

    @classmethod
    def schema_for(
        cls,
        style_keys: list[str] | tuple[str, ...],
        *,
        units: str = "real",
    ) -> ForceFieldSchemaView:
        styles: dict[str, StyleSchema] = {}
        for key in style_keys:
            meta = cls._REGISTRY.get(key)
            if meta is None:
                cat, _, name = key.partition(":")
                styles[key] = StyleSchema(
                    category=cat or "unknown",
                    name=name or key,
                    index_arity=0,
                    params=(),
                    energy_formula="(unregistered style)",
                )
            else:
                styles[key] = meta
        return ForceFieldSchemaView(styles=styles, units=units)
