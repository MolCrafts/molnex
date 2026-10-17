"""What an engine must know about a deployed model that the package cannot say.

``{name}.json`` beside the packaged model, written by :class:`DeployMetadata`.
It is engine-neutral: it states the model's own units, cutoff and receptive
field, and the native operator libraries the package calls; an engine converts
into its own unit system and decides its own ghost shell. Fields:

* ``schema_version`` — :data:`SCHEMA_VERSION`; a reader refuses another.
* ``format`` — the package format, e.g. ``"aoti"`` (``{name}.pt2``).
* ``cutoff`` — neighbour cutoff in ``length_unit``.
* ``species`` — sorted atomic numbers the model accepts.
* ``dtype`` — ``"float32"`` or ``"float64"``, the model's input dtype.
* ``receptive_field_layers`` — message-passing depth ``T``: a site energy
  depends on every atom within ``T * cutoff``.
* ``energy_unit`` / ``length_unit`` — the model's units (``"eV"``,
  ``"angstrom"``); forces are ``energy_unit / length_unit``.
* ``op_libraries`` — ``{library: requirement}``: native libraries that register
  the non-ATen operators in the package, each with the upstream version its
  binding was built against. Empty for a package of ATen operators only.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

#: Version of the ``{name}.json`` layout and of the calling convention the
#: package it describes keeps (:mod:`molix.deploy.contract`). 2: ``shifts``
#: input, ``virial`` output.
SCHEMA_VERSION = 2

#: Operator namespace → (native library that registers it for C++ callers,
#: Python package whose installed version the binding must match). ``None``:
#: molnex's own library, whose requirement is the schema version it reports.
_NATIVE_OPS: dict[str, tuple[str, str | None]] = {
    "cuequivariance": ("molnex_cueq_ops", "cuequivariance_ops_torch"),
    "molix": ("molnex_opLib", None),
}

_DTYPES = {torch.float32: "float32", torch.float64: "float64"}


@dataclass(frozen=True)
class DeployMetadata:
    """The engine-neutral description of one deployed model.

    Raises:
        ValueError: On an empty or non-positive ``species``, a non-positive
            ``cutoff`` or ``receptive_field_layers``, or an unknown ``dtype``.
    """

    format: str
    cutoff: float
    species: tuple[int, ...]
    dtype: str
    receptive_field_layers: int
    energy_unit: str = "eV"
    length_unit: str = "angstrom"
    op_libraries: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.species or any(
            isinstance(z, bool) or not isinstance(z, int) or z <= 0 for z in self.species
        ):
            raise ValueError(f"species must be atomic numbers Z > 0, got {self.species!r}")
        object.__setattr__(self, "species", tuple(sorted(set(self.species))))
        if not self.cutoff > 0:
            raise ValueError(f"cutoff must be positive, got {self.cutoff!r}")
        layers = self.receptive_field_layers
        if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
            raise ValueError(
                "receptive_field_layers must be the message-passing depth T >= 1 "
                f"(a site energy sees every atom within T * cutoff), got {layers!r}"
            )
        if self.dtype not in _DTYPES.values():
            raise ValueError(f"dtype must be one of {sorted(_DTYPES.values())}, got {self.dtype!r}")

    @staticmethod
    def dtype_name(dtype: torch.dtype) -> str:
        """``"float32"`` / ``"float64"`` for a torch floating dtype."""
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be float32 or float64, got {dtype}")
        return _DTYPES[dtype]

    @staticmethod
    def op_libraries_for(namespaces: Iterable[str]) -> dict[str, str]:
        """The native libraries a package with these operator namespaces needs.

        Args:
            namespaces: Non-ATen operator namespaces found in the package, as
                :func:`molix.export.aoti.custom_op_namespaces` returns them.

        Returns:
            ``{library: "<python package>==<installed version>"}``.

        Raises:
            ValueError: For a namespace no native library registers — such a
                package could only run inside Python.
        """
        libraries: dict[str, str] = {}
        for namespace in sorted(set(namespaces)):
            if namespace not in _NATIVE_OPS:
                raise ValueError(
                    f"operators of namespace {namespace!r} have no native registration; "
                    "the package would need Python at run time"
                )
            library, package = _NATIVE_OPS[namespace]
            libraries[library] = (
                _molix_op_requirement()
                if package is None
                else f"{package}=={_installed_version(package)}"
            )
        return libraries

    def write(self, directory: str | Path, name: str = "model") -> Path:
        """Write ``{directory}/{name}.json`` and return its path."""
        path = Path(directory) / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": SCHEMA_VERSION,
            "format": self.format,
            "cutoff": float(self.cutoff),
            "species": list(self.species),
            "dtype": self.dtype,
            "receptive_field_layers": self.receptive_field_layers,
            "energy_unit": self.energy_unit,
            "length_unit": self.length_unit,
            "op_libraries": dict(self.op_libraries),
        }
        path.write_text(json.dumps(document, indent=2) + "\n")
        return path

    @classmethod
    def read(cls, path: str | Path) -> DeployMetadata:
        """Read a ``{name}.json``.

        Raises:
            ValueError: If its ``schema_version`` is not :data:`SCHEMA_VERSION`.
        """
        document = json.loads(Path(path).read_text())
        version = document.pop("schema_version", None)
        if version != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {version!r}, expected {SCHEMA_VERSION}")
        document["species"] = tuple(document["species"])
        return cls(**document)


def _molix_op_requirement() -> str:
    """The operator-schema version molnex's loaded op library reports.

    ``molnex_op_library_requirement`` in ``src/molix/op/src/molix_library.cpp``;
    the C++ runtime checks the library it loads against the same string.
    """
    from molix import _op_lib_path

    path = _op_lib_path()
    if path is None:
        raise ValueError(
            "the package calls molix::* operators but molnex's native op library is "
            "not built here (see csrc/CMakeLists.txt), so its version cannot be recorded"
        )
    library = ctypes.CDLL(str(path))
    library.molnex_op_library_requirement.restype = ctypes.c_char_p
    return library.molnex_op_library_requirement().decode()


def _installed_version(package: str) -> str:
    """Installed version of the distribution that provides import package ``package``."""
    distributions = importlib.metadata.packages_distributions().get(package)
    if not distributions:
        raise ValueError(f"{package} is not installed")
    return importlib.metadata.version(distributions[0])
