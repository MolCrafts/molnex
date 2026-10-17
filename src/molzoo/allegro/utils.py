"""Read an official ``.nequip.zip`` package without importing nequip or allegro.

:class:`NequipPackage` is the Allegro :class:`~molzoo.checkpoint.OfficialArchive`.
A ``.nequip.zip`` is an uncompressed ``torch.package`` archive; every record
sits under one root directory (the file name at packaging time, so it is read
from the archive, never from the current file name):

* ``model/eager_model.pkl`` — the module tree, the source of truth for the
  architecture (``model/config.yaml`` of a foundation model only points at a
  checkpoint and carries no hyper-parameters);
* ``.data/<key>.storage`` — raw little-endian tensor bytes, the pickle's
  persistent ids.

The tree is rebuilt through the shared stub unpickler. The reader fills a
:class:`~molzoo.allegro.spec.NequipConfig` off the tree's plain attributes —
refusing every upstream option this port does not implement
(``src/molzoo/specs/allegro.md`` §4) — and normalises the state dict's layout
once, so :meth:`read` hands back tensors in molnex's shapes under upstream's
names.

Reference:
    ``nequip/scripts/package.py`` (nequip 0.19.1).
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any

import torch
from molpy import Element

from molzoo.allegro.spec import AllegroSpec, NequipConfig, NequipMLP
from molzoo.checkpoint import ModuleStub, OfficialArchive

_NONLINEARITIES = {"SiLU": "silu", "GELU": "gelu", "Tanh": "tanh", "ReLU": "relu"}


class NequipPackage(OfficialArchive):
    """Official ``.nequip.zip`` reader: ``read(path) -> (state, spec)``."""

    def read(self, path: str | Path) -> tuple[dict[str, torch.Tensor], AllegroSpec]:
        """Return the upstream ``state_dict`` (molnex layout) and the model's spec.

        Keys are as upstream names them below the package's ``GraphModel``;
        shapes are already normalised (:meth:`_normalise_layout`).

        Raises:
            ValueError: If the archive is not a little-endian nequip package, or
                it uses an option this port does not implement.
        """
        with zipfile.ZipFile(path) as archive:
            root = archive.namelist()[0].split("/")[0]
            if archive.read(f"{root}/byteorder").decode().strip() != "little":
                raise ValueError("only little-endian packages are supported")
            try:
                pickled = archive.read(f"{root}/model/eager_model.pkl")
            except KeyError as exc:
                raise ValueError(
                    f"{path} is not a nequip package (no model/eager_model.pkl)"
                ) from exc
            storages: dict[str, torch.UntypedStorage] = {}

            def persistent_load(pid: Any) -> Any:
                kind = pid[0].decode() if isinstance(pid[0], bytes) else pid[0]
                if kind != "storage":
                    raise ValueError(f"unsupported persistent id {kind!r}")
                storage_type, key = pid[1], pid[2]
                if key not in storages:
                    raw = archive.read(f"{root}/.data/{key}.storage")
                    storages[key] = (
                        torch.UntypedStorage.from_buffer(bytearray(raw), dtype=torch.uint8)
                        if raw
                        else torch.UntypedStorage(0)
                    )
                dtype = torch.uint8 if storage_type is torch.UntypedStorage else storage_type.dtype
                return torch.storage.TypedStorage(
                    wrap_storage=storages[key], dtype=dtype, _internal=True
                )

            tree = self.unpickle(pickled, persistent_load=persistent_load)
        graph_model = tree._modules["sole_model"]
        spec = AllegroSpec.from_package(self._config(graph_model))
        state = self._normalise_layout(
            self.tensor_state(graph_model.state_dict()),
            n_types=len(spec.atomic_numbers),
            num_layers=spec.num_layers,
        )
        return state, spec

    @staticmethod
    def _normalise_layout(
        state: dict[str, torch.Tensor], *, n_types: int, num_layers: int
    ) -> dict[str, torch.Tensor]:
        """Upstream tensor layouts → molnex's, in place.

        * ``per_type_energy_scale_shift.{scales,shifts}``: ``(T, 1)`` → ``(T,)``,
          a singleton broadcast to ``n_types``; an empty tensor (upstream's
          "absent") is dropped, the model already holds the identity.
        * ``allegro.tps.{layer}.weights``: a single-path ``(u,)`` → ``(u, 1)``.
        * ``allegro.tps.{layer}.w3j``: dropped; the Clebsch–Gordan constants
          are rebuilt by cuEquivariance.

        Args:
            state: Upstream-named tensors, modified and returned.
            n_types: Length of the type table.
            num_layers: Number of Allegro layers.

        Returns:
            ``state``.
        """
        prefix = "model.func.per_type_energy_scale_shift."
        for name in ("scales", "shifts"):
            key = prefix + name
            if key in state:
                value = state[key].reshape(-1)
                if value.numel() == 0:
                    del state[key]
                else:
                    state[key] = value.expand(n_types).clone() if value.numel() == 1 else value
        for layer in range(num_layers):
            key = f"model.func.allegro.tps.{layer}.weights"
            if key in state and state[key].dim() == 1:
                state[key] = state[key].unsqueeze(-1)
            state.pop(f"model.func.allegro.tps.{layer}.w3j", None)
        return state

    @staticmethod
    def _irreps(value: ModuleStub) -> list[tuple[int, int, int]]:
        """``[(mul, l, p)]`` of a stubbed upstream ``Irreps`` (a tuple of ``_MulIr``)."""
        terms = []
        for mul_ir in value._pickle_args[0]:
            mul, ir = (
                mul_ir._pickle_args if len(mul_ir._pickle_args) == 2 else mul_ir._pickle_args[0]
            )
            l, p = ir._pickle_args if len(ir._pickle_args) == 2 else ir._pickle_args[0]
            terms.append((int(mul), int(l), int(p)))
        return terms

    @staticmethod
    def _mlp(module: Any, name: str) -> NequipMLP:
        """Shape of a nequip ``ScalarMLPFunction``; rejects the variants this port lacks."""
        if type(module.mlp).__name__ == "DeepLinearMLP":
            raise ValueError(f"{name}: deep-linear MLPs are not supported")
        if module.bias:
            raise ValueError(f"{name}: biased MLPs are not supported")
        dims = list(module.dims)
        hidden = dims[1:-1]
        if len(set(hidden)) > 1:
            raise ValueError(f"{name}: non-uniform hidden widths {hidden} are not supported")
        acts = [
            type(m).__name__ for m in module.mlp._modules.values() if "weight" not in m._parameters
        ]
        nonlinearity = None
        if acts:
            if acts[0] not in _NONLINEARITIES:
                raise ValueError(f"{name}: nonlinearity {acts[0]} is not supported")
            nonlinearity = _NONLINEARITIES[acts[0]]
        linears = [m for m in module.mlp._modules.values() if "weight" in m._parameters]
        for i, layer in enumerate(linears):
            gain = 1.0 if (nonlinearity is None or i == 0) else math.sqrt(2.0)
            if not math.isclose(float(layer.alpha), gain / math.sqrt(dims[i]), rel_tol=1e-6):
                raise ValueError(f"{name}: forward_normalize=False is not supported")
        return NequipMLP(
            depth=len(hidden), width=hidden[0] if hidden else None, nonlinearity=nonlinearity
        )

    def _config(self, graph_model: Any) -> NequipConfig:
        """The hyper-parameters of a stubbed ``GraphModel``.

        Raises:
            ValueError: Naming the first unsupported upstream option.
        """
        metadata = dict(graph_model._metadata)
        modules = dict(graph_model.model.func._modules)
        for required in ("edge_norm", "radial_chemical_embed", "tensor_embed", "allegro"):
            if required not in modules:
                raise ValueError(f"not an Allegro energy model (no {required})")
        if modules["edge_norm"]._per_edge_type:
            raise ValueError("per_edge_type_cutoff is not supported")
        radial = modules["radial_chemical_embed"]
        if "bessel_encode" not in radial._modules or "type_embed" not in radial._modules:
            raise ValueError("only the TwoBodyBesselScalarEmbed radial embedding is supported")
        bessel = radial.bessel_encode
        p = float(bessel.cutoff.p)
        if p != int(p):
            raise ValueError(f"non-integer polynomial_cutoff_p {p} is not supported")

        tensor_embed = modules["tensor_embed"]
        if not tensor_embed._edge_weighter.weight_individual_irreps:
            raise ValueError("weight_individual_irreps=False is not supported")
        sh = self._irreps(tensor_embed.irreps_out["edge_attrs"])
        allegro = modules["allegro"]
        allowed = self._irreps(allegro.tensor_track_allowed_irreps)
        if any(not tp.path_channel_coupling for tp in allegro.tps):
            raise ValueError("tp_path_channel_coupling=False is not supported")
        if "avg_num_neighbors_norm" in allegro._modules:
            norm = allegro.avg_num_neighbors_norm.norm_const.reshape(-1)
        else:  # allegro 0.7.x keeps the factor on the tensor products
            norm = torch.tensor([float(allegro.tps[0].scatter_factor or 1.0)])
        if norm.numel() > 1 and not bool(torch.all(norm == norm[0])):
            raise ValueError("per-type avg_num_neighbors is not supported")
        norm_const = float(norm[0])

        type_names = str(metadata["type_names"]).split()
        atomic_numbers = [Element(name).number for name in type_names]
        scale_shift = modules["per_type_energy_scale_shift"]

        def table(values: torch.Tensor, present: bool) -> list[float] | None:
            return [float(v) for v in values.reshape(-1).tolist()] if present else None

        fields: dict[str, Any] = {
            "atomic_numbers": atomic_numbers,
            "r_max": float(modules["edge_norm"].r_max),
            "num_bessels": int(bessel.num_bessels),
            "bessel_trainable": bool(bessel.trainable),
            "polynomial_cutoff_p": int(p),
            "radial_chemical_embed_dim": int(radial.type_embed.center_embed.embedding_dim) * 2,
            "scalar_embed_mlp": self._mlp(
                modules["scalar_embed_mlp"].mlp_module, "scalar_embed_mlp"
            ),
            "l_max": max(l for _, l, _ in sh),
            "parity": len({(l, p) for _, l, p in allowed}) > len({l for _, l, _ in allowed}),
            "num_layers": int(allegro.num_layers),
            "num_scalar_features": int(allegro.num_scalar_features),
            "num_tensor_features": int(allegro.num_tensor_features),
            "allegro_mlp": self._mlp(allegro.latents[0], "allegro latent"),
            "readout_mlp": self._mlp(modules["edge_readout"].mlp_module, "edge_readout"),
            "avg_num_neighbors": 1.0 / norm_const**2,
            "per_type_energy_scales": table(scale_shift.scales, bool(scale_shift.has_scales)),
            "per_type_energy_shifts": table(scale_shift.shifts, bool(scale_shift.has_shifts)),
            "per_type_energy_scales_trainable": bool(
                getattr(scale_shift, "scales_trainable", False)
            ),
            "per_type_energy_shifts_trainable": bool(
                getattr(scale_shift, "shifts_trainable", False)
            ),
        }
        pair = modules.get("pair_potential")
        if pair is not None:
            if type(pair).__name__ != "ZBL":
                raise ValueError(f"pair potential {type(pair).__name__} is not supported")
            if not math.isclose(float(pair._qqr2exesquare), 14.399645 * 0.5, rel_tol=1e-12):
                raise ValueError("ZBL in units other than LAMMPS metal is not supported")
            zbl_numbers = [int(z) for z in pair.atomic_numbers.tolist()]
            if zbl_numbers != atomic_numbers:
                raise ValueError("ZBL atomic numbers disagree with the type table")
            fields["pair_potential"] = "ZBL"
            fields["zbl_polynomial_cutoff_p"] = int(float(pair.cutoff.p))
        return NequipConfig(**fields)
