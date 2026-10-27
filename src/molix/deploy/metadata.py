"""What an engine must know about a deployed model that the package cannot say.

``{name}.json`` beside the packaged model, written by :class:`DeployMetadata`.
It is engine-neutral: it states the model's own units, cutoff and receptive
field, and the native operator libraries the package calls; an engine converts
into its own unit system and decides its own ghost shell. Fields:

* ``format`` — the package format, e.g. ``"aoti"`` (``{name}.pt2``).
* ``cutoff`` — neighbour cutoff in ``length_unit``.
* ``species`` — sorted atomic numbers the model accepts.
* ``dtype`` — ``"float32"`` or ``"float64"``, the model's input dtype.
* ``receptive_field_layers`` — message-passing depth ``T``: an atom energy
  depends on every atom within ``T * cutoff``.
* ``energy_unit`` / ``length_unit`` — the model's units (``"eV"``,
  ``"angstrom"``); forces are ``energy_unit / length_unit``.
* ``op_libraries`` — ``{library: requirement}``: molnex's native libraries that
  register the non-ATen operators in the package (``molnex_opLib`` for
  ``molix::*``, ``molnex_equivariance`` for ``molnex_equivariance::*``), each
  with the requirement string it reports. Empty for a package of ATen
  operators only.
"""

from __future__ import annotations

import ctypes
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch

#: Operator namespace → the native library that registers it for C++ callers.
#: Both are molnex's own; each reports its requirement string through
#: ``molnex_op_library_requirement``, which the C++ runtime checks on load.
_NATIVE_OPS: dict[str, str] = {
    "molix": "molnex_opLib",
    "molnex_equivariance": "molnex_equivariance",
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
                f"(an atom energy sees every atom within T * cutoff), got {layers!r}"
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
            library = _NATIVE_OPS[namespace]
            libraries[library] = _library_requirement(library)
        return libraries

    def write(self, directory: str | Path, name: str = "model") -> Path:
        """Write ``{directory}/{name}.json`` and return its path."""
        path = Path(directory) / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
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
        """Read a ``{name}.json``; a field this reader does not know is ignored."""
        document = json.loads(Path(path).read_text())
        known = {name.name for name in fields(cls)}
        document = {key: value for key, value in document.items() if key in known}
        document["species"] = tuple(document["species"])
        return cls(**document)


def _library_requirement(library: str) -> str:
    """The requirement string a built molnex op library reports.

    ``molnex_op_library_requirement``, exported by ``csrc/molix`` and
    ``csrc/equivariance``; the C++ runtime checks the library it loads against
    the same string.
    """
    if library == "molnex_opLib":
        from molix import _op_lib_path

        path = _op_lib_path()
    else:
        from molrep.equivariance.backend import NativeBackend

        path = NativeBackend.library_path()
    if path is None:
        raise ValueError(
            f"the package calls operators of {library} but that native library is "
            "not built here (see csrc/CMakeLists.txt), so its version cannot be recorded"
        )
    loaded = ctypes.CDLL(str(path))
    loaded.molnex_op_library_requirement.restype = ctypes.c_char_p
    return loaded.molnex_op_library_requirement().decode()
