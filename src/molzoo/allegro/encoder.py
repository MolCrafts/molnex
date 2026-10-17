"""Feature body shared by the Allegro products.

:class:`AllegroEncoder` registers the radial basis, the two-body type
embedding and MLP, the spherical-harmonic tensor embedding and the Allegro
layers (``Contracter`` tensor products + DenseNet latent MLPs). It registers
no energy head — :class:`~molzoo.allegro.potential.AllegroPotential`
subclasses the body and registers those, so the ``state_dict`` stays flat.

Primitives, not a pipeline: ``node_types`` / ``radial_features`` /
``twobody_features`` / ``angular_features`` / ``layer_features`` (and the
:class:`~molzoo.encoder.ElementTable` gate).
:meth:`AllegroEncoder.edge_features` composes them from tensors (the one
composition :meth:`AllegroEncoder.forward` and
:meth:`~molzoo.allegro.potential.AllegroPotential.energy` share);
``forward`` reads the batch and writes the encoder contract
(``pairs.edge_features``) for heads that read features from the batch
(``molpot.composition.Sonata``, ``molpot.heads.PermMultipoleHead``), which
rely on the attributes ``output_dim``, ``num_scalar_features``, ``l_max``,
``expose_tensor_track`` and ``tensor_track_irreps``.

Mathematics: ``src/molzoo/specs/allegro.md`` §5. Hot-path discipline: every
switch is frozen into a plain attribute in ``__init__``; the spec survives only
as ``self._spec``.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nat. Commun. 14, 579 (2023).
    https://arxiv.org/abs/2204.05249
    ``allegro/nn/_allegro.py::Allegro_Module`` (nequip-allegro 0.8.3).
"""

from __future__ import annotations

import math

import cuequivariance as cue
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from molix import config
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import PolynomialCutoff
from molrep.embedding.geometry import edge_lengths, edge_vectors
from molrep.embedding.mlp import ScalarMLPFunction
from molrep.embedding.radial import SincBesselRBF
from molrep.interaction.product import sh_irreps_from_l_max
from molzoo.allegro.layers import Contracter, ProductTypeEmbedding, WeightedChannels
from molzoo.allegro.spec import AllegroSpec
from molzoo.encoder import ElementTable


class AllegroEncoder(ElementTable, TensorDictModuleBase):
    """Allegro feature body (spec §5, steps 1–6).

    Args:
        spec: A validated :class:`~molzoo.allegro.spec.AllegroSpec`. Head
            fields (readout, scales/shifts, ZBL) are ignored here.

    Raises:
        NotImplementedError: If backward irreps pruning would trim the first
            layer's input (upstream's ``V_0`` is always the full SH basis).
    """

    in_keys = [("atoms", "Z"), ("atoms", "pos"), ("pairs", "pair_index")]
    out_keys = [("pairs", "edge_features")]

    def __init__(self, spec: AllegroSpec, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        #: Provenance only. Never read on a hot path.
        self._spec = spec
        #: The build dtype of every float parameter and buffer.
        self.dtype = dtype
        n_types = len(spec.atomic_numbers)
        #: The type table in ``type_names`` order (not ascending); its membership test.
        z_table = torch.tensor(spec.atomic_numbers, dtype=torch.long)
        self.register_buffer("z_table", z_table, persistent=False)
        #: ``Z → type`` for ``Z`` in the table (``-1`` elsewhere).
        z_to_type = torch.full((max(spec.atomic_numbers) + 1,), -1, dtype=torch.long)
        z_to_type[z_table] = torch.arange(n_types)
        self.register_buffer("z_to_type", z_to_type, persistent=False)
        self.z_to_type: torch.Tensor

        # ---- frozen switches ----
        self.num_layers = spec.num_layers
        self.num_scalar_features = spec.num_scalar_features
        self.num_tensor_features = spec.num_tensor_features
        self.l_max = spec.l_max
        self.expose_tensor_track = spec.expose_tensor_track
        self.output_dim = spec.num_scalar_features * (spec.num_layers + 1)
        self.avg_norm = 1.0 / math.sqrt(spec.avg_num_neighbors) if spec.avg_num_neighbors else 1.0

        F, u = spec.num_scalar_features, spec.num_tensor_features
        # ---- two-body scalar embedding ----
        self.bessel = SincBesselRBF(
            num_basis=spec.num_bessels, r_max=spec.r_max, trainable=spec.bessel_trainable
        )
        self.cutoff = PolynomialCutoff(
            r_cut=spec.r_max, exponent=spec.polynomial_cutoff_p, dtype=dtype
        )
        self.type_embed = ProductTypeEmbedding(
            num_types=n_types, num_basis=spec.num_bessels, dim=spec.embed_dim, dtype=dtype
        )
        self.scalar_embed_mlp = ScalarMLPFunction(
            spec.embed_dim,
            F,
            hidden_layers_depth=spec.scalar_embed_mlp_hidden_layers_depth,
            hidden_layers_width=spec.scalar_embed_mlp_hidden_layers_width,
            nonlinearity=spec.scalar_embed_mlp_nonlinearity,
            dtype=dtype,
        )

        # ---- tensor embedding ----
        sh_irreps = cue.Irreps("O3", sh_irreps_from_l_max(spec.l_max))
        irrep_dims = [ir.dim for _, ir in sh_irreps]
        #: Fed unit vectors — the lengths are already in hand (no second norm).
        self.spherical_harmonics = SphericalHarmonics(
            l_max=spec.l_max, normalize=False, dtype=dtype
        )
        self.env_weighter = WeightedChannels(irrep_dims=irrep_dims, channels=u, dtype=dtype)
        env_numel = self.env_weighter.weight_numel
        self.env_embed_linear = ScalarMLPFunction(F, env_numel, nonlinearity=None, dtype=dtype)
        self.first_layer_env_embed_projection = ScalarMLPFunction(
            F, F + env_numel, nonlinearity=None, dtype=dtype
        )

        # ---- Allegro layers ----
        schedule = self._layer_irreps(spec)
        if schedule[0] != sh_irreps:
            raise NotImplementedError(
                f"irreps pruning trimmed the first layer input ({schedule[0]} vs {sh_irreps})"
            )
        # ``layer_features`` reads the scalars as ``tensor[:, :u]``.
        assert all(irreps[0].ir == cue.O3(0, 1) for irreps in schedule), schedule
        self.tps = nn.ModuleList()
        self.latents = nn.ModuleList()
        for layer, (irreps_in, irreps_out) in enumerate(zip(schedule, schedule[1:])):
            self.tps.append(
                Contracter(
                    irreps_in1=irreps_in,
                    irreps_in2=sh_irreps,
                    irreps_out=irreps_out,
                    channels=u,
                    method=spec.tp_method,
                    dtype=dtype,
                )
            )
            last = layer == spec.num_layers - 1
            self.latents.append(
                ScalarMLPFunction(
                    F * (layer + 1) + u,
                    F + (0 if last else env_numel),
                    hidden_layers_depth=spec.allegro_mlp_hidden_layers_depth,
                    hidden_layers_width=spec.allegro_mlp_hidden_layers_width,
                    nonlinearity=spec.allegro_mlp_nonlinearity,
                    dtype=dtype,
                )
            )
        #: Irreps of the last layer's output at multiplicity ``u`` (``ir_mul``).
        self.tensor_track_irreps = cue.Irreps("O3", [(u, ir) for _, ir in schedule[-1]])

    @staticmethod
    def _layer_irreps(spec: AllegroSpec) -> list[cue.Irreps]:
        """Tensor-track irreps per layer boundary, one copy each (upstream pruning).

        Forward: from the allowed set (``l ≤ l_max``; both parities when
        ``parity``) keep what ``previous ⊗ Y`` can reach, the last layer
        scalars only (unless ``expose_tensor_track``). Backward: drop what can
        no longer reach the next layer's output.

        Args:
            spec: The configuration.

        Returns:
            ``num_layers + 1`` irreps; entry 0 is the first layer's input.
        """
        sh = cue.Irreps("O3", sh_irreps_from_l_max(spec.l_max))
        allowed = (
            cue.Irreps("O3", "+".join(f"1x{l}{p}" for l in range(spec.l_max + 1) for p in "eo"))
            if spec.parity
            else sh
        )
        scalar = cue.Irreps("O3", "1x0e")

        def reachable(arg: cue.Irreps, ir: cue.Irrep) -> bool:
            return any(ir in a * e for _, a in arg for _, e in sh)

        schedule = [sh]
        for layer in range(spec.num_layers):
            last = layer == spec.num_layers - 1
            candidates = scalar if last and not spec.expose_tensor_track else allowed
            schedule.append(
                cue.Irreps("O3", [(1, ir) for _, ir in candidates if reachable(schedule[-1], ir)])
            )
        pruned = [schedule[-1]]
        for arg in reversed(schedule[:-1]):
            out = pruned[-1]
            keep = [(1, a) for _, a in arg if any(any(o in a * e for _, o in out) for _, e in sh)]
            pruned.append(cue.Irreps("O3", keep))
        return list(reversed(pruned))

    # ------------------------------------------------------------ primitives
    def node_types(self, Z: torch.Tensor) -> torch.Tensor:
        """Type index of each atom ``(N,)`` (``Z`` inside the table)."""
        return self.z_to_type[Z]

    def radial_features(self, lengths: torch.Tensor) -> torch.Tensor:
        """``sinc``-Bessel basis times the polynomial envelope ``(E, num_bessels)``."""
        return self.bessel(lengths) * self.cutoff(lengths).unsqueeze(-1)

    def twobody_features(
        self, types: torch.Tensor, edge_index: torch.Tensor, radial: torch.Tensor
    ) -> torch.Tensor:
        """Two-body scalar embedding ``(E, F)`` (type embedding, then the MLP)."""
        return self.scalar_embed_mlp(self.type_embed(types, edge_index, radial))

    def angular_features(self, vectors: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Spherical harmonics of the pair directions ``(E, (l_max + 1)²)``.

        Args:
            vectors: Pair vectors ``(E, 3)``.
            lengths: Their lengths ``(E,)`` — the harmonics are evaluated on
                ``vectors / lengths``.
        """
        return self.spherical_harmonics(vectors / lengths.unsqueeze(-1))

    def layer_features(
        self,
        *,
        twobody: torch.Tensor,
        sh: torch.Tensor,
        edge_index: torch.Tensor,
        num_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the Allegro layers.

        Args:
            twobody: Two-body scalar embedding ``(E, F)``.
            sh: Spherical harmonics ``(E, D_sh)``.
            edge_index: Source/target pairs ``(E, 2)``; the source is the centre.
            num_atoms: ``N`` (a shape, not a tensor value).

        Returns:
            ``(edge_features (E, F·(L+1)), tensor_track (E, D_out·u))``.
        """
        F = self.num_scalar_features
        u = self.num_tensor_features
        source = edge_index[:, 0].contiguous()
        tensor = self.env_weighter(sh, self.env_embed_linear(twobody))
        projection = self.first_layer_env_embed_projection(twobody)
        scalars = [projection[:, :F]]
        env_w = projection[:, F:]
        last = self.num_layers - 1
        for layer, (tp, latent) in enumerate(zip(self.tps, self.latents)):
            env = self.env_weighter(sh, env_w, index=source, num_rows=num_atoms)
            tensor = tp(tensor, env * self.avg_norm, source)
            out = latent(torch.cat(scalars + [tensor[:, :u]], dim=-1))
            scalars.append(out[:, :F])
            if layer < last:
                env_w = out[:, F:]
        return torch.cat(scalars, dim=-1), tensor

    def edge_features(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_diff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Spec §5 steps 1–6 from tensors: the primitives above, composed once.

        Args:
            positions: Atom positions ``(N, 3)`` — the differentiated variable.
            Z: Atomic numbers ``(N,)``, inside the table.
            edge_index: Source/target pairs ``(E, 2)``.
            edge_diff: Imaged pair vectors ``(E, 3)``; ``None`` is the
                open-boundary difference of ``positions``.

        Returns:
            ``(edge_features (E, F·(L+1)), tensor_track (E, D_out·u), lengths (E,))``.
        """
        vectors = edge_vectors(positions, edge_index, edge_diff=edge_diff)
        lengths = edge_lengths(vectors)
        features, tensor = self.layer_features(
            twobody=self.twobody_features(
                self.node_types(Z), edge_index, self.radial_features(lengths)
            ),
            sh=self.angular_features(vectors, lengths),
            edge_index=edge_index,
            num_atoms=Z.shape[0],
        )
        return features, tensor, lengths

    # -------------------------------------------------------------- encoder
    def forward(self, td: TensorDict) -> TensorDict:
        """Write ``pairs.edge_features`` (and the tensor track if exposed) in place.

        Reads ``atoms.Z``, ``atoms.pos``, ``pairs.pair_index`` and, when
        present, the imaged ``pairs.pair_diff``.
        """
        Z = td["atoms", "Z"]
        self._validate_once(Z)
        edge_index = td["pairs", "pair_index"].contiguous()
        keys = td.keys(include_nested=True)
        diff = td["pairs", "pair_diff"] if ("pairs", "pair_diff") in keys else None
        features, tensor, _ = self.edge_features(td["atoms", "pos"], Z, edge_index, edge_diff=diff)
        td["pairs", "edge_features"] = features
        if self.expose_tensor_track:
            td["pairs", "edge_tensor_features"] = tensor
        return td
