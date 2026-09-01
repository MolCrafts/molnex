"""Official MACE archive helpers — pickle stubs and e3nn→cueq state.

No CLI. :func:`e3nn2cueq` is what :meth:`MacePotential.load` calls for
``format="official"``. mace/e3nn classes are never imported: a stub unpickler
swaps them for inert ``nn.Module`` shells so cuEquivariance tensors still
restore for real.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

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
    def __new__(cls, *args: object, **kwargs: object) -> _ModStub:
        return object.__new__(cls)

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


def _looks_cueq(state: Mapping[str, torch.Tensor]) -> bool:
    return any(".f.m.graphs" in key or key.endswith(".f.weight") for key in state)


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
    hidden = getattr(module, "hidden_irreps", None)
    if hidden is not None:
        cfg["hidden_irreps"] = str(hidden)
    mlp = getattr(module, "MLP_irreps", None)
    if mlp is not None:
        cfg["MLP_irreps"] = str(mlp)
    return cfg


def _pack_e3nn_contractions(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold e3nn per-path contraction weights into cueq ``symmetric_contractions.weight``.

    Official Polar (e3nn) stores ``contractions.{k}.weights_max`` plus
    ``weights.0`` / ``weights.1``; molnex's cuEq block wants them concatenated
    on the coupling axis. ``U_matrix_*`` are rebuilt by cuEq and dropped.
    """
    marker = ".symmetric_contractions.contractions."
    grouped: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    drop: set[str] = set()
    for key, value in state.items():
        if marker not in key:
            continue
        head, rest = key.split(marker, 1)
        cid_str, _, tail = rest.partition(".")
        grouped.setdefault(head + ".symmetric_contractions", {}).setdefault(int(cid_str), {})[
            tail
        ] = value
        drop.add(key)
    if not grouped:
        return state
    packed = {key: value for key, value in state.items() if key not in drop}
    for layer, contractions in grouped.items():
        chunks: list[torch.Tensor] = []
        for cid in sorted(contractions):
            slot = contractions[cid]
            if "weights_max" in slot:
                chunks.append(slot["weights_max"])
            for wkey in sorted(
                (name for name in slot if name.startswith("weights.") and name[8:].isdigit()),
                key=lambda name: int(name.split(".")[-1]),
            ):
                chunks.append(slot[wkey])
        if chunks:
            packed[f"{layer}.weight"] = torch.cat(chunks, dim=1)
    return packed


def e3nn2cueq(
    source: str | Path | nn.Module | Mapping[str, torch.Tensor],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Turn an official MACE archive into ``{"state", "config"}``.

    Already-cueq pickles (OMOL-cueq, converted MatPES) pass the state through.
    e3nn pickles are stub-unpickled to a state_dict; layout conversion of
    individual Linear/TP weights is identity when the archive is already cueq
    and otherwise left to :meth:`MacePotential.load` shape checks.

    Args:
        source: Path, module, or state dict.
        map_location: Forwarded to ``torch.load``.

    Returns:
        ``state`` is ``name -> tensor``; ``config`` holds constructor fields
        scraped off a pickled module when present.
    """
    config: dict[str, Any] = {}
    if isinstance(source, Mapping):
        state = {str(k): v for k, v in source.items() if isinstance(v, torch.Tensor)}
        return {"state": _pack_e3nn_contractions(state), "config": config}
    if isinstance(source, nn.Module):
        return {
            "state": _pack_e3nn_contractions(dict(source.state_dict())),
            "config": _module_config(source),
        }
    path = Path(source)
    try:
        loaded = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        import molzoo.mace.utils as pickle_mod

        loaded = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
            pickle_module=pickle_mod,
        )
    if isinstance(loaded, nn.Module):
        return {
            "state": _pack_e3nn_contractions(dict(loaded.state_dict())),
            "config": _module_config(loaded),
        }
    if isinstance(loaded, Mapping):
        if "state_dict" in loaded and isinstance(loaded["state_dict"], Mapping):
            state = {
                str(k): v for k, v in loaded["state_dict"].items() if isinstance(v, torch.Tensor)
            }
            extra = loaded.get("config") if isinstance(loaded.get("config"), Mapping) else {}
            return {"state": _pack_e3nn_contractions(state), "config": dict(extra)}
        state = {str(k): v for k, v in loaded.items() if isinstance(v, torch.Tensor)}
        return {"state": _pack_e3nn_contractions(state), "config": config}
    raise TypeError(f"e3nn2cueq cannot read {type(loaded).__name__} from {path}")
