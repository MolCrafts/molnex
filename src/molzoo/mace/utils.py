"""Read official MACE archives without importing the upstream runtime.

Only already-converted cuEq weights are accepted. Basis conversion belongs
to the version-pinned upstream converter, run out of tree. Pickled modules
require explicit trust; the stubs are an import aid, NOT a security sandbox.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from torch import nn

_ALLOWED = (
    "torch",
    "collections",
    "builtins",
    "__builtin__",
    "_codecs",
    "numpy",
    "cuequivariance",
    "cuequivariance_torch",
)


class _ModStub(nn.Module):
    _pickle_args: tuple[Any, ...]

    def __new__(cls, *args: object, **kwargs: object) -> _ModStub:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_pickle_args", args)
        return instance

    def __init__(self, *args: object, **kwargs: object) -> None:
        nn.Module.__init__(self)

    def __setitem__(self, *args: object) -> None:
        return None

    def append(self, *args: object) -> None:
        return None


_STUBS: dict[tuple[str, str], type] = {}


class Unpickler(pickle.Unpickler):
    """pickle Unpickler that stubs mace/e3nn classes as empty ``nn.Module``s."""

    def find_class(self, module: str, name: str) -> Any:
        if module.split(".")[0] in _ALLOWED:
            return super().find_class(module, name)
        key = (module, name)
        if key not in _STUBS:
            _STUBS[key] = type(name, (_ModStub,), {"__module__": module})
        return _STUBS[key]


def load(file_obj: Any, **kwargs: Any) -> Any:
    """pickle.load entry for ``torch.load(..., pickle_module=this module)``."""
    return Unpickler(file_obj, **kwargs).load()


def _irreps_text(value: Any) -> str:
    """Recover inert pickled tuple metadata; no e3nn algebra is implemented."""
    if not isinstance(value, _ModStub):
        return str(value)
    try:
        terms = []
        for term in value._pickle_args[0]:
            mul, ir = term._pickle_args[0]
            ell, parity = ir._pickle_args[0]
            terms.append(f"{int(mul)}x{int(ell)}{'e' if parity == 1 else 'o'}")
        return "+".join(terms)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("unsupported pickled irreps metadata; supply a JSON config") from exc


def _module_config(module: nn.Module) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    numbers = getattr(module, "atomic_numbers", None)
    if numbers is not None:
        cfg["atomic_numbers"] = [int(z) for z in torch.as_tensor(numbers).reshape(-1).tolist()]
    energies = getattr(module, "atomic_energies_fn", None)
    if energies is not None and hasattr(energies, "atomic_energies"):
        cfg["atomic_energies"] = [float(x) for x in energies.atomic_energies.reshape(-1).tolist()]
    for name in ("r_max", "num_interactions"):
        value = getattr(module, name, None)
        if value is not None:
            cfg[name] = int(value) if name == "num_interactions" else float(value)
    interactions = getattr(module, "interactions", ())
    hidden = getattr(module, "hidden_irreps", None)
    if hidden is None and interactions:
        hidden = getattr(interactions[0], "hidden_irreps", None)
    if hidden is not None:
        cfg["hidden_irreps"] = _irreps_text(hidden)
    mlp = getattr(module, "MLP_irreps", None)
    readouts = getattr(module, "readouts", ())
    if mlp is None and readouts:
        mlp = getattr(readouts[-1], "hidden_irreps", None)
    if mlp is not None:
        cfg["MLP_irreps"] = _irreps_text(mlp)
    if interactions:
        cfg["radial_MLP"] = getattr(interactions[0], "radial_MLP", None)
        angular = getattr(interactions[0], "edge_attrs_irreps", None)
        if angular is not None:
            cfg["max_ell"] = max(
                int(t.split("x")[-1][:-1]) for t in _irreps_text(angular).split("+")
            )
    radial = getattr(module, "radial_embedding", None)
    if radial is not None:
        cfg["num_bessel"] = getattr(radial, "out_dim", None)
        cutoff = getattr(radial, "cutoff_fn", None)
        if cutoff is not None:
            cfg["num_polynomial_cutoff"] = int(cutoff.p)
    scale_shift = getattr(module, "scale_shift", None)
    if scale_shift is not None:
        cfg["scale"] = float(scale_shift.scale)
        cfg["shift"] = float(scale_shift.shift)
    products = getattr(module, "products", ())
    if products:
        sc = products[0].symmetric_contractions
        contractions = getattr(sc, "contractions", ())
        correlation = getattr(contractions[0] if contractions else sc, "correlation", None)
        if correlation is not None:
            cfg["correlation"] = int(correlation)
    aliases = {
        "num_recursion_steps": "num_recursion_steps",
        "atomic_multipoles_max_l": "multipole_lmax",
        "atomic_multipoles_smearing_width": "multipole_sigma",
        "field_feature_max_l": "field_lmax",
        "field_feature_widths": "field_widths",
        "_field_feature_norms": "field_norms",
        "kspace_cutoff_factor": "kspace_cutoff_factor",
        "include_electrostatic_self_interaction": "include_electrostatic_self_interaction",
        "field_si": "field_si",
        "add_local_electron_energy": "add_local_electron_energy",
        "keep_last_layer_irreps": "keep_last_layer_irreps",
        "quadrupole_feature_corrections": "quadrupole_feature_corrections",
        "use_so3": "use_so3",
        "use_agnostic_product": "use_agnostic_product",
    }
    for attribute, key in aliases.items():
        value = getattr(module, attribute, None)
        if value is not None:
            cfg[key] = value
    edge = getattr(module, "edge_irreps", None)
    if edge is not None:
        cfg["edge_channels"] = int(_irreps_text(edge).split("x", 1)[0])
    # Serialized runtime cutoff wins over the heuristic constructor factor.
    # Some official dumps carry a factor that no longer matches this buffer.
    cutoff = getattr(module, "kspace_cutoff", None)
    if cutoff is not None:
        cfg["kspace_cutoff"] = float(cutoff)
    return cfg


def read_archive(
    source: str | Path | nn.Module | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
    trusted: bool = False,
    converted_weights: str | Path | None = None,
) -> dict[str, Any]:
    """Read ``{"state", "config"}`` without doing a basis conversion.

    Args:
        source: Tensor state, ``state_dict``/``config`` envelope, or module.
        map_location: Forwarded to ``torch.load``.
        trusted: Allow unrestricted pickle deserialization of a local module.
            Never enable for an untrusted/downloaded archive without verification.
        converted_weights: Already-converted cuEq tensor archive. When given,
            ``source`` supplies metadata only; its weights are never installed.

    Raw e3nn contraction weights require the official upstream basis
    conversion, not concatenation or a same-numel reshape. Tensor-only
    archives plus JSON sidecars are the preferred deployment format.
    """
    loaded = source
    if isinstance(source, (str, Path)):
        try:
            loaded = torch.load(source, map_location=map_location, weights_only=True)
        except pickle.UnpicklingError as exc:
            if not trusted:
                raise ValueError(
                    "checkpoint is not a tensor-only archive; use a cuEq state_dict "
                    "plus JSON config, or trusted=True for a verified local pickle"
                ) from exc
            import molzoo.mace.utils as pickle_mod

            loaded = torch.load(
                source, map_location=map_location, weights_only=False, pickle_module=pickle_mod
            )
    if isinstance(loaded, nn.Module):
        state, config = dict(loaded.state_dict()), _module_config(loaded)
    elif isinstance(loaded, Mapping):
        state = loaded.get("state_dict", loaded)
        config = loaded.get("config", {}) if "state_dict" in loaded else {}
        if not isinstance(config, Mapping):
            raise TypeError("archive config must be a mapping")
    else:
        raise TypeError(f"cannot read an official archive from {type(loaded).__name__}")
    if not isinstance(state, Mapping) or not state:
        raise TypeError("archive state_dict must be a nonempty tensor mapping")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise TypeError("state_dict must contain only string keys and tensors")
    state = cast(Mapping[str, torch.Tensor], state)
    config = dict(config)
    for key in ("r_max", "kspace_cutoff", "num_interactions"):
        if key in state:
            config[key] = int(state[key]) if key == "num_interactions" else float(state[key])
    if "atomic_numbers" in state:
        config["atomic_numbers"] = state["atomic_numbers"].reshape(-1).tolist()
    if "atomic_energies_fn.atomic_energies" in state:
        config["atomic_energies"] = state["atomic_energies_fn.atomic_energies"].reshape(-1).tolist()
    if converted_weights is not None:
        converted = read_archive(converted_weights, map_location=map_location, trusted=trusted)
        state = converted["state"]
        config = {**config, **converted["config"]}
    if any(".symmetric_contractions.contractions." in key for key in state):
        raise ValueError(
            "unconverted e3nn contraction weights: run the version-pinned official "
            "cuEq converter out of tree and supply converted_weights= or a cuEq "
            "state_dict plus config; concatenation does not convert the CG basis"
        )
    return {"state": dict(state), "config": dict(config)}


def e3nn2cueq(source: Any, **kwargs: Any) -> dict[str, Any]:
    """Deprecated archive-reader alias; this function does NOT convert bases."""
    warnings.warn(
        "e3nn2cueq is an archive reader, not a converter; use MacePotential.load "
        "with official cuEq-converted weights",
        DeprecationWarning,
        stacklevel=2,
    )
    return read_archive(source, **kwargs)
