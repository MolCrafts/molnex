"""Official Allegro key table (nequip ``GraphModel`` names → molnex names).

Keys are as :class:`molzoo.allegro.utils.NequipPackage` returns them, i.e. below
the package's ``sole_model``, with the layout already normalised by the reader
(per-type ``(T,)`` tables, ``(u, 1)`` tensor-product weights, no ``w3j``
constants). ``None`` drops an entry the model rebuilds from its constructor
(``1/r_max``, the ZBL constants). The remap itself is
:class:`molzoo.checkpoint.CheckpointRemap`.

Reference:
    ``allegro/model/allegro_models.py::FullAllegroModel`` (nequip-allegro 0.8.3).
"""

from __future__ import annotations

#: Official key prefix → molnex name. Longest prefix wins.
allegro_keys: dict[str, str | None] = {
    "model._empty": None,
    "model.func.edge_norm.": None,
    "model.func.radial_chemical_embed.bessel_encode.": "bessel.",
    "model.func.radial_chemical_embed.type_embed.": "type_embed.",
    "model.func.scalar_embed_mlp.mlp_module.": "scalar_embed_mlp.",
    "model.func.tensor_embed.env_embed_linear.": "env_embed_linear.",
    "model.func.allegro.first_layer_env_embed_projection.": "first_layer_env_embed_projection.",
    "model.func.allegro.latents.": "latents.",
    "model.func.allegro.tps.": "tps.",
    "model.func.edge_readout.mlp_module.": "edge_readout.",
    "model.func.per_type_energy_scale_shift.": "per_type_energy_scale_shift.",
    "model.func.pair_potential.": None,
}
