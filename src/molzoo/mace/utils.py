"""Read official MACE archives without importing the upstream runtime.

:class:`MaceArchive` is the MACE :class:`~molzoo.checkpoint.OfficialArchive`:
``read(path, config) -> (state, MaceOfficialConfig)`` resolves what ``path``
names (a cuEquivariance ``state_dict``, a ``{"state_dict", "config"}``
envelope, a pickled MACE module read through the shared stub unpickler, a
config json beside its weights, or a directory holding both), and types the
hyper-parameters it finds as a
:class:`~molzoo.mace.spec.MaceOfficialConfig`. Only already-converted
cuEquivariance weights are accepted; the basis conversion belongs to the
version-pinned upstream converter, run out of tree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from molzoo.checkpoint import ModuleStub, OfficialArchive
from molzoo.mace.spec import MaceOfficialConfig


def _irreps_text(value: Any) -> str:
    """Recover inert pickled irreps metadata as text (``"16x0e+16x1o"``)."""
    if not isinstance(value, ModuleStub):
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
    """Hyper-parameters off a pickled upstream module tree.

    The tree is made of :class:`~molzoo.checkpoint.ModuleStub` instances whose
    plain attributes ``nn.Module.__setstate__`` restored; which attributes a
    given upstream version carries is unknown, so each is read with ``getattr``
    and absent ones are left out of the mapping.
    """
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
    if interactions:
        # Which upstream blocks the stack is made of (stub class names).
        cfg["interaction_blocks"] = [type(block).__name__ for block in interactions]
        avg = getattr(interactions[0], "avg_num_neighbors", None)
        if avg is not None:
            cfg["avg_num_neighbors"] = float(avg)
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
    les = getattr(module, "les", None)
    if les is not None:
        # MACE-LES: the latent-charge Ewald's constants live on its submodule;
        # the multipole switches on the model (absent on older dumps = off).
        ewald = les.ewald
        cfg["les"] = {
            "ewald": type(ewald).__name__,
            "sigma": float(ewald.sigma),
            "dl": float(ewald.dl),
            "norm_factor": float(ewald.norm_factor),
            "remove_self_interaction": bool(ewald.remove_self_interaction),
            "use_atomwise": bool(getattr(les, "use_atomwise", False)),
            "use_fixed_atomic_charges": bool(getattr(les, "use_fixed_atomic_charges", False)),
            "use_epsilon_r_scaling": bool(getattr(les, "use_epsilon_r_scaling", False)),
            "use_dipoles": bool(getattr(module, "use_dipoles", False)),
            "use_quads": bool(getattr(module, "use_quads", False)),
            "use_induced_charges": bool(getattr(module, "use_induced_charges", False)),
            "use_induced_dipoles": bool(getattr(module, "use_induced_dipoles", False)),
            "output_scale": float(getattr(module, "les_output_scale", 1.0)),
        }
    # Serialized runtime cutoff wins over the heuristic constructor factor.
    # Some official dumps carry a factor that no longer matches this buffer.
    cutoff = getattr(module, "kspace_cutoff", None)
    if cutoff is not None:
        cfg["kspace_cutoff"] = float(cutoff)
    return cfg


class MaceArchive(OfficialArchive):
    """Official MACE archive reader.

    Args:
        map_location: Forwarded to ``torch.load``.
        converted_weights: An already-converted cuEquivariance tensor archive.
            When given, the archive named at :meth:`read` supplies metadata
            only; its weights are never installed.
    """

    def __init__(
        self,
        *,
        map_location: str | torch.device = "cpu",
        converted_weights: str | Path | None = None,
    ) -> None:
        self.map_location = map_location
        self.converted_weights = converted_weights

    def read(
        self,
        path: str | Path | nn.Module | Mapping[str, Any],
        config: str | Path | None = None,
    ) -> tuple[dict[str, torch.Tensor], MaceOfficialConfig]:
        """Return ``(state, config)`` without doing a basis conversion.

        Args:
            path: The weights file (``state_dict``, envelope or pickled
                module), a config json whose weights sit beside it
                (``<stem>_cueq_state.pt`` or ``<stem>.pt``), a directory
                holding one ``*config.json`` and one ``*cueq_state.pt`` /
                ``*.pt``, or the already-loaded object.
            config: An explicit sidecar json; its entries win over what the
                archive carries, and a directory is then not searched for one.

        Returns:
            The upstream-named tensors and the typed hyper-parameters.

        Raises:
            FileNotFoundError: If no weights are found next to a json or
                under a directory.
            ValueError: If a directory is ambiguous, or the weights are
                unconverted (pre-cuEquivariance) contraction weights.
            TypeError: On a malformed archive.
            pydantic.ValidationError: If the hyper-parameters a spec needs
                (element table, ``E0``, ``r_max``) are missing or mistyped.
        """
        sidecar: dict[str, Any] = {}
        if config is not None:
            sidecar = json.loads(Path(config).read_text())
        if isinstance(path, (str, Path)):
            path, found = self._locate(Path(path), search_config=config is None)
            sidecar = {**found, **sidecar}
        state, archive = self._raw(path)
        if self.converted_weights is not None:
            state, converted = self._raw(self.converted_weights)
            archive = {**archive, **converted}
        if any(".symmetric_contractions.contractions." in key for key in state):
            raise ValueError(
                "unconverted (pre-cuEquivariance) contraction weights: run the "
                "version-pinned official cuEq converter out of tree and supply "
                "converted_weights= or a cuEq state_dict plus config; concatenation "
                "does not convert the CG basis"
            )
        return state, MaceOfficialConfig(**{**archive, **sidecar})

    @staticmethod
    def _locate(path: Path, *, search_config: bool) -> tuple[Path, dict[str, Any]]:
        """The weights file ``path`` names and the json config found with it."""
        if path.is_dir():
            jsons = sorted(path.glob("*config.json")) if search_config else []
            weights = sorted(path.glob("*cueq_state.pt")) or sorted(path.glob("*.pt"))
            if len(jsons) > 1 or len(weights) > 1:
                raise ValueError(
                    f"ambiguous official archive under {path}; specify weights and config"
                )
            if not weights:
                raise FileNotFoundError(f"no official weights under {path}")
            return weights[0], json.loads(jsons[0].read_text()) if jsons else {}
        if path.suffix == ".json":
            stem = path.stem.replace("_config", "")
            for candidate in (path.with_name(stem + "_cueq_state.pt"), path.with_suffix(".pt")):
                if candidate.exists():
                    return candidate, json.loads(path.read_text())
            raise FileNotFoundError(f"no weights next to {path}")
        return path, {}

    def _raw(
        self, path: str | Path | nn.Module | Mapping[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """The tensors and the untyped hyper-parameters of one archive object."""
        loaded = (
            self.torch_load(path, map_location=self.map_location)
            if isinstance(path, (str, Path))
            else path
        )
        if isinstance(loaded, nn.Module):
            state, config = loaded.state_dict(), _module_config(loaded)
        elif isinstance(loaded, Mapping):
            state = loaded.get("state_dict", loaded)
            config = loaded.get("config", {}) if "state_dict" in loaded else {}
            if not isinstance(config, Mapping):
                raise TypeError("archive config must be a mapping")
        else:
            raise TypeError(f"cannot read an official archive from {type(loaded).__name__}")
        state = self.tensor_state(state)
        config = dict(config)
        for key in ("r_max", "kspace_cutoff", "num_interactions"):
            if key in state:
                config[key] = int(state[key]) if key == "num_interactions" else float(state[key])
        if "atomic_numbers" in state:
            config["atomic_numbers"] = state["atomic_numbers"].reshape(-1).tolist()
        if "atomic_energies_fn.atomic_energies" in state:
            config["atomic_energies"] = (
                state["atomic_energies_fn.atomic_energies"].reshape(-1).tolist()
            )
        return state, config
