"""Direct permanent atomic multipole readout (no electrostatic energy).

Reads pooled atom features (from an Allegro-style edge encoder), predicts
selected atomic moments ``q`` (l=0), ``μ`` (l=1) and ``Θ`` (l=2), and applies
an optional total-charge projection. The whole permanent-multipole *readout*
surface lives in a single :class:`PermMultipoleHead` — the l=1 / l=2 paths
are inlined so the head is self-contained and the public surface is one
symbol.

Electrostatic energy is **not** this module's responsibility. The screened-
Coulomb / Ewald multipole energy lives in
:class:`molpot.potentials.EwaldMultipoleEnergy`, which consumes
``{q, μ, Θ}`` written by this head into the batch and emits a per-graph
energy plus per-atom potential / field. The two are composed via
``PotentialComposer``; see ``.claude/specs/les-electrostatics.md``.

This head is intentionally *direct* — charges are produced by an l=0 readout
on the encoder embedding, NOT by extremising an electrostatic Lagrangian
under a charge-conservation constraint. The QEq / CELLI variational variant
(Fuchs/Sanocki/Zavadlav, npj Comput. Mater. **11**, 71 (2025),
https://doi.org/10.1038/s41524-025-01790-4) is a different algorithm with a
KKT solve and per-atom Hirshfeld supervision; it is **out of scope here**
and will live in a separate ``QEqLayer``.

Scope:
    * charge head + per-graph total-charge projection (mean-residual
      subtraction; loss gradients flow into the unprojected charges so the
      head learns to predict near-neutral sums on its own);
    * equivariant ``μ`` readout — PaiNN-style scalar-gated l=1 path over
      the encoder's tensor track (slice ``1o`` block, gate per-channel,
      collapse ``u·1o → 1·1o``); output is a 3-vector that rotates as such;
    * equivariant ``Θ`` readout — same recipe at l=2 (slice ``2e``,
      collapse ``u·2e → 1·2e``); output is the 5-component traceless
      symmetric basis transforming under Wigner ``D⁽²⁾``;
    * molecular dipole ``μ_mol = Σ_i q_i r_i + Σ_i μ_i`` (PaiNN's
      ``DipoleMoment`` head) emitted automatically and supervised against
      QM9's dipole-magnitude target.

References for the components actually used here:
    * **Equivariant μ readout** — Schütt, Unke, Gastegger, *PaiNN: Equivariant
      Message Passing for the Prediction of Tensorial Properties and
      Molecular Spectra*, ICML 2021 (arXiv:2102.03150). The
      ``μ_mol = Σ q_i r_i + Σ μ_i`` molecular dipole readout used in the
      training script's auxiliary loss matches PaiNN's ``DipoleMoment``
      head; the per-channel scalar-gated l=1 message structure is the same
      as PaiNN's ``v_j ← φ(s_j) ⊙ v_j`` update.
    * **Pair-centred edge representation** — Musaelian et al.,
      *Nature Communications* **14**, 579 (2023) — the Allegro encoder
      whose tensor track this head consumes.
"""

from __future__ import annotations

import math
from typing import Sequence

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from tensordict import TensorDict

from molix import config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    """``[Linear → SiLU] × len(hidden) → Linear`` (no activation on final layer)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h, dtype=config.ftype))
        layers.append(nn.SiLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim, dtype=config.ftype))
    return nn.Sequential(*layers)


def _find_irrep_offset(
    tensor_irreps: cue.Irreps,
    target_ir: cue.Irrep,
) -> int:
    """Locate ``target_ir``'s starting offset in an ``ir_mul`` flat layout.

    Returns the offset (in scalar elements) of the first occurrence of
    ``target_ir`` within ``tensor_irreps``. Raises if absent.
    """
    offset = 0
    for mul, ir in tensor_irreps:
        if ir == target_ir:
            return offset
        offset += mul * ir.dim
    raise ValueError(f"tensor_irreps must contain a {target_ir} segment; got {tensor_irreps}.")


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class PermMultipoleHeadSpec(BaseModel):
    """Configuration snapshot for :class:`PermMultipoleHead`.

    Mirrors the ``AllegroSpec`` pattern: every constructor argument is
    captured here so a trained checkpoint carries an exact, validated
    description of the head it was built with. molcfg-driven training
    scripts can construct ``PermMultipoleHeadSpec(**cfg["multipole"])`` and
    pass it straight to ``PermMultipoleHead.from_spec(...)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    input_dim: int = Field(..., gt=0)
    avg_num_neighbors: float | None = Field(default=None, gt=0.0)
    charge: bool = True
    dipole: bool = False
    quadrupole: bool = False
    constrain_total_charge: bool = True
    total_charge_key: str = "total_charge"
    embed_moments: bool = False
    hidden_dim: int = Field(default=128, gt=0)
    out_charge_key: str = "atomic_charges"
    out_dipole_key: str = "atomic_dipoles"
    out_quadrupole_key: str = "atomic_quadrupoles"


# ---------------------------------------------------------------------------
# PermMultipoleHead
# ---------------------------------------------------------------------------


class PermMultipoleHead(nn.Module):
    """Direct permanent atomic multipole readout (q, μ, Θ).

    Pipeline::

        edge_features (E, F)
            ─ scatter (1/√⟨N⟩) by source ─→  atom_feats (N, F)
        atom_feats
            ─ q_head ─→  q (N,) ─ project total charge ─→  q
        edge_tensor_features (E, irreps_dim) + edge_features (E, F)
            ─ inlined PaiNN-style l=1 readout ─→  μ (N, 3)   [optional]
            ─ inlined PaiNN-style l=2 readout ─→  Θ (N, 5)   [optional]
        (q, μ, atoms.pos, atoms.batch)
            ─ derived ─→  μ_mol (B, 3)         (always when q is predicted)

    The two equivariant moment readouts share one structural recipe — slice
    the ``lℓ`` block of the encoder's tensor track, gate per-channel by an
    l=0 scalar projection of the edge features, collapse ``u·lℓ → 1·lℓ``
    with :class:`cuet.Linear`, scatter to source atoms — so they are
    implemented as one private method (:meth:`_equivariant_moment_readout`)
    invoked twice, not as two separate sub-modules.

    What this is NOT:
        * **Not an energy head.** The screened-Coulomb / Ewald multipole
          energy is computed downstream in
          :class:`molpot.potentials.EwaldMultipoleEnergy`, which consumes
          the moments this head writes into the batch.
        * **Not QEq.** Charges are direct l=0 predictions of the atom
          embedding, not the result of a charge-equilibration KKT solve.
          See ``QEqLayer`` (TODO) for the variational variant.

    Args:
        input_dim: Per-edge feature dim ``F`` from the encoder.
        avg_num_neighbors: Dataset-wide ⟨|N(i)|⟩ for the edge→atom pool
            normalisation. ``None`` falls back to per-source ``1/√|N(i)|``.
        charge: Predict atomic charges ``q_i``. Default ``True``.
        dipole: Predict atomic dipoles ``μ_i ∈ ℝ^3`` via the inlined
            PaiNN-style l=1 readout. Default ``False``. Requires
            ``tensor_irreps`` and the encoder built with
            ``expose_tensor_track=True``.
        quadrupole: Predict atomic quadrupoles ``Θ_i ∈ ℝ^5`` (traceless
            symmetric basis, ``2e`` irrep ordering) via the inlined l=2
            readout. Default ``False``. Requires ``tensor_irreps`` and
            the encoder built with ``l_max ≥ 2`` and
            ``expose_tensor_track=True``. Output transforms under the
            Wigner ``D⁽²⁾(R)`` representation.
        constrain_total_charge: If ``True``, project predicted charges so
            each graph sums to ``total_charge_key``. Done by subtracting
            the per-graph mean residual — the simplest unbiased
            projection (gradients still flow into the unprojected
            charges, so the head learns to predict near-neutral sums on
            its own; the projection is a safety net, not a workaround for
            a missing loss term).
        total_charge_key: Per-graph total-charge key under ``graphs``.
            **Required** when ``constrain_total_charge=True``; absent →
            ``KeyError`` at forward. For uniformly-neutral datasets
            (e.g. QM9), inject it via
            :class:`~molix.data.tasks.ConstantLabel` (``value=0.0``).
        embed_moments: Reserved. Re-injecting moment information into the
            encoder is a v1 TODO; passing ``True`` raises today.
        hidden_dim: Hidden width of the per-moment scalar MLP heads.
        out_charge_key: TensorDict key for atomic charges.
        out_dipole_key: TensorDict key for atomic dipoles.
        out_quadrupole_key: TensorDict key for atomic quadrupoles.
        tensor_irreps: Wiring (not in spec): irreps of the encoder's
            tensor-track output. Required when ``dipole=True`` (l=1
            readout) or ``quadrupole=True`` (l=2 readout). Pass
            ``Allegro.tensor_track_irreps`` (encoder must be constructed
            with ``expose_tensor_track=True``, and ``l_max ≥ 2`` if
            quadrupole is on).

    Forward output:
        Dict containing the per-atom moment writes plus
        ``"molecular_dipole"`` when charges are predicted. When total-charge
        projection is on, the output also contains ``"charge_sum_pre_proj"``
        and ``"charge_sum_post_proj"`` per-graph sums for monitoring.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        avg_num_neighbors: float | None = None,
        charge: bool = True,
        dipole: bool = False,
        quadrupole: bool = False,
        constrain_total_charge: bool = True,
        total_charge_key: str = "total_charge",
        embed_moments: bool = False,
        hidden_dim: int = 128,
        out_charge_key: str = "atomic_charges",
        out_dipole_key: str = "atomic_dipoles",
        out_quadrupole_key: str = "atomic_quadrupoles",
        tensor_irreps: cue.Irreps | None = None,
    ):
        super().__init__()
        # Build (and validate) the spec from the kwargs — single source of
        # truth for the head's configuration. The spec is frozen, so it
        # also serves as a checkpointable description.
        self.config = PermMultipoleHeadSpec(
            input_dim=input_dim,
            avg_num_neighbors=avg_num_neighbors,
            charge=charge,
            dipole=dipole,
            quadrupole=quadrupole,
            constrain_total_charge=constrain_total_charge,
            total_charge_key=total_charge_key,
            embed_moments=embed_moments,
            hidden_dim=hidden_dim,
            out_charge_key=out_charge_key,
            out_dipole_key=out_dipole_key,
            out_quadrupole_key=out_quadrupole_key,
        )
        cfg = self.config

        if not (cfg.charge or cfg.dipole or cfg.quadrupole):
            raise ValueError(
                "PermMultipoleHead needs at least one moment enabled "
                "(charge / dipole / quadrupole)."
            )
        if cfg.embed_moments:
            raise NotImplementedError(
                "embed_moments=True is reserved for a v1 release "
                "(re-inject moments into edge_features)."
            )

        # Promote a few flags onto self for hot-path access without
        # touching the Pydantic accessor on every forward.
        self.charge = cfg.charge
        self.dipole = cfg.dipole
        self.quadrupole = cfg.quadrupole
        self.constrain_total_charge = cfg.constrain_total_charge
        self.total_charge_key = cfg.total_charge_key
        self.avg_num_neighbors = cfg.avg_num_neighbors
        self.out_charge_key = cfg.out_charge_key
        self.out_dipole_key = cfg.out_dipole_key
        self.out_quadrupole_key = cfg.out_quadrupole_key

        # Charge readout: l=0 scalar MLP on pooled atom features.
        if cfg.charge:
            self.q_head = _scalar_mlp(cfg.input_dim, [cfg.hidden_dim], 1)

        # Equivariant moment readouts share one structural recipe (see
        # _equivariant_moment_readout). Both require the encoder's
        # tensor-track irreps and uniform multiplicity ``u``.
        if cfg.dipole or cfg.quadrupole:
            if tensor_irreps is None:
                needs = ", ".join(
                    name
                    for name, on in [("dipole", cfg.dipole), ("quadrupole", cfg.quadrupole)]
                    if on
                )
                raise ValueError(
                    f"{needs} requires tensor_irreps (the encoder's "
                    "tensor-track irreps). Construct your encoder with "
                    "expose_tensor_track=True and pass "
                    "tensor_irreps=encoder.tensor_track_irreps."
                )
            muls = tensor_irreps.muls
            if len(set(muls)) != 1:
                raise ValueError(
                    "PermMultipoleHead requires uniform multiplicity in "
                    f"tensor_irreps; got muls={muls}."
                )
            self._u = int(muls[0])
            self._tensor_irreps = tensor_irreps
            G = tensor_irreps.irrep_class
        else:
            self._u = 0  # unused

        # μ readout (l=1, parity odd): slice 1o block, gate, collapse u·1o → 1·1o.
        if cfg.dipole:
            self._mu_offset = _find_irrep_offset(tensor_irreps, G(1, -1))
            self._mu_size = 3 * self._u
            self.mu_proj = nn.Linear(cfg.input_dim, self._u, dtype=config.ftype)
            self.mu_collapse = cuet.Linear(
                irreps_in=cue.Irreps(G, [(self._u, "1o")]),
                irreps_out=cue.Irreps(G, [(1, "1o")]),
                layout=cue.ir_mul,
            )

        # Θ readout (l=2, parity even): slice 2e block, gate, collapse u·2e → 1·2e.
        if cfg.quadrupole:
            try:
                self._theta_offset = _find_irrep_offset(tensor_irreps, G(2, +1))
            except ValueError as e:
                raise ValueError(
                    f"quadrupole=True needs a 2e segment in tensor_irreps; "
                    f"build the encoder with l_max>=2. ({e})"
                ) from None
            self._theta_size = 5 * self._u
            self.theta_proj = nn.Linear(cfg.input_dim, self._u, dtype=config.ftype)
            self.theta_collapse = cuet.Linear(
                irreps_in=cue.Irreps(G, [(self._u, "2e")]),
                irreps_out=cue.Irreps(G, [(1, "2e")]),
                layout=cue.ir_mul,
            )

    @classmethod
    def from_spec(
        cls,
        spec: PermMultipoleHeadSpec,
        *,
        tensor_irreps: cue.Irreps | None = None,
    ) -> "PermMultipoleHead":
        """Construct from a spec.

        ``tensor_irreps`` is wiring (depends on the encoder you compose with),
        not a hyperparameter — it is *not* in the spec and must be supplied
        here when ``spec.dipole=True`` or ``spec.quadrupole=True``. Pass
        ``encoder.tensor_track_irreps`` from an Allegro built with
        ``expose_tensor_track=True`` (and ``l_max >= 2`` if quadrupole is on).
        """
        return cls(**spec.model_dump(), tensor_irreps=tensor_irreps)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Read permanent atomic multipoles ``(q, μ, Θ)`` from the batch.

        Reads ``edges.edge_features`` / ``edges.edge_index`` / ``atoms.batch``
        from the collated batch, runs the enabled charge / dipole / quadrupole
        readouts, and optionally projects the per-graph total charge.

        Args:
            batch: Post-collate :class:`~tensordict.TensorDict` with the
                ``atoms`` / ``edges`` / ``graphs`` namespaces.

        Returns:
            Dict keyed by the configured output names (e.g.
            ``out_charge_key`` ``(N,)``, ``out_dipole_key`` ``(N, 3)``,
            ``out_quadrupole_key``) plus ``molecular_dipole`` and the
            pre/post total-charge sums when charge projection is active.
        """
        edge_features = batch["edges", "edge_features"]  # (E, F)
        edge_index = batch["edges", "edge_index"]  # (E, 2)
        atom_batch = batch["atoms", "batch"]  # (N,)
        n_nodes: int = int(batch["atoms", "Z"].shape[0])
        n_graphs: int = int(batch["graphs"].batch_size[0])

        # 1. Pool edges → atoms (only needed for the l=0 charge head; the
        #    equivariant moment readouts do their own source-side scatter
        #    inside _equivariant_moment_readout).
        atom_feats = (
            self._scatter_mean_to_atoms(edge_features, edge_index, n_nodes) if self.charge else None
        )  # (N, F)

        out: dict[str, torch.Tensor] = {}

        # 2. Heads + per-graph total-charge projection.
        q: torch.Tensor | None = None
        if self.charge:
            q_raw = self.q_head(atom_feats).squeeze(-1)  # (N,)
            if self.constrain_total_charge:
                q, sum_pre, sum_post = self._project_total_charge(
                    q_raw, atom_batch, batch, n_graphs
                )
                out["charge_sum_pre_proj"] = sum_pre
                out["charge_sum_post_proj"] = sum_post
            else:
                q = q_raw
            batch[("atoms", self.out_charge_key)] = q
            out[self.out_charge_key] = q

        mu: torch.Tensor | None = None
        if self.dipole or self.quadrupole:
            tensor_feats = batch["edges", "edge_tensor_features"]  # (E, irreps_dim)
            src = edge_index[:, 0]
            if self.dipole:
                mu = self._equivariant_moment_readout(
                    tensor_feats=tensor_feats,
                    scalar_feats=edge_features,
                    src=src,
                    n_nodes=n_nodes,
                    block_offset=self._mu_offset,
                    block_size=self._mu_size,
                    out_dim=3,
                    scalar_proj=self.mu_proj,
                    collapse=self.mu_collapse,
                )  # (N, 3)
                batch[("atoms", self.out_dipole_key)] = mu
                out[self.out_dipole_key] = mu
            if self.quadrupole:
                theta = self._equivariant_moment_readout(
                    tensor_feats=tensor_feats,
                    scalar_feats=edge_features,
                    src=src,
                    n_nodes=n_nodes,
                    block_offset=self._theta_offset,
                    block_size=self._theta_size,
                    out_dim=5,
                    scalar_proj=self.theta_proj,
                    collapse=self.theta_collapse,
                )  # (N, 5)
                batch[("atoms", self.out_quadrupole_key)] = theta
                out[self.out_quadrupole_key] = theta

        # 3. Molecular dipole (always derived when charges exist).
        if q is not None:
            mu_mol = self._molecular_dipole(
                q, mu, batch["atoms", "pos"], atom_batch, n_graphs
            )  # (B, 3)
            batch[("graphs", "molecular_dipole")] = mu_mol
            out["molecular_dipole"] = mu_mol

        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _equivariant_moment_readout(
        self,
        *,
        tensor_feats: torch.Tensor,  # (E, irreps_dim) ir_mul layout
        scalar_feats: torch.Tensor,  # (E, F)
        src: torch.Tensor,  # (E,) edge source indices
        n_nodes: int,
        block_offset: int,  # start of the lℓ block in tensor_feats
        block_size: int,  # = (2ℓ+1) * u
        out_dim: int,  # = 2ℓ + 1
        scalar_proj: nn.Module,  # F → u  (l=0, rotation-invariant)
        collapse: nn.Module,  # cuet.Linear(u·lℓ → 1·lℓ)
    ) -> torch.Tensor:
        r"""Inlined PaiNN-style scalar-gated lℓ readout.

        Pipeline (per edge ``ij``)::

            v_ij^{ℓ} = slice(V_ij, lℓ).reshape(u, 2ℓ+1)   ∈ ℝ^{u·(2ℓ+1)}
            gate_ij  = scalar_proj(s_ij)                  ∈ ℝ^u            l=0
            gated_ij = gate_ij ⊙ v_ij^{ℓ}                 ∈ ℝ^{u·(2ℓ+1)}   lℓ
            m_ij     = collapse(gated_ij)                                  lℓ
            m_i      = (1/√⟨|N|⟩) · Σ_{j: src=i} m_ij     ∈ ℝ^{2ℓ+1}

        Used by both the μ (ℓ=1) and Θ (ℓ=2) paths — the only differences
        are ``block_offset / block_size / out_dim`` and which weights are
        registered as ``self.mu_*`` vs ``self.theta_*``.

        Strictly SO(3)-equivariant under Wigner ``Dˡ(R)`` because every step
        composes equivariant primitives:
            * scalar gating is l=0 (rotation-invariant).
            * channel-wise multiplication of an l=0 scalar with an lℓ
              vector lives in lℓ.
            * ``cuet.Linear`` between two ``mul × lℓ`` channel
              multiplicities is the equivariant linear map for that irrep.
            * scatter-sum is rotation-invariant in the index reduction.
        """
        E = tensor_feats.shape[0]
        u = self._u
        # Slice the lℓ block in ir_mul layout: (E, (2ℓ+1)*u) flattened
        # along the (component-outer, channel-inner) axes.
        v_flat = tensor_feats[:, block_offset : block_offset + block_size]
        v_l = v_flat.reshape(E, out_dim, u).transpose(1, 2)  # (E, u, 2ℓ+1)

        gate = scalar_proj(scalar_feats)  # (E, u)
        gated = gate.unsqueeze(-1) * v_l  # (E, u, 2ℓ+1)

        # Repack to ir_mul layout (2ℓ+1, u) for cuet.Linear.
        gated_ir_mul = gated.transpose(1, 2).reshape(E, block_size)
        edge_out = collapse(gated_ir_mul)  # (E, 2ℓ+1)

        atom_out = torch.zeros(
            n_nodes,
            out_dim,
            dtype=edge_out.dtype,
            device=edge_out.device,
        )
        atom_out.scatter_add_(
            0,
            src.unsqueeze(-1).expand_as(edge_out),
            edge_out,
        )
        if self.avg_num_neighbors is not None:
            atom_out = atom_out / math.sqrt(self.avg_num_neighbors)
        return atom_out

    def _scatter_mean_to_atoms(
        self,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        src = edge_index[:, 0]
        atom_feats = torch.zeros(
            n_nodes,
            edge_feats.shape[-1],
            dtype=edge_feats.dtype,
            device=edge_feats.device,
        )
        atom_feats.scatter_add_(0, src.unsqueeze(-1).expand_as(edge_feats), edge_feats)
        if self.avg_num_neighbors is not None:
            return atom_feats / math.sqrt(self.avg_num_neighbors)
        count = torch.zeros(n_nodes, dtype=edge_feats.dtype, device=edge_feats.device)
        count.scatter_add_(
            0, src, torch.ones(src.shape[0], dtype=edge_feats.dtype, device=src.device)
        )
        return atom_feats / count.clamp(min=1.0).sqrt().unsqueeze(-1)

    def _project_total_charge(
        self,
        q: torch.Tensor,
        atom_batch: torch.Tensor,
        batch: TensorDict,
        n_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_target = batch["graphs", self.total_charge_key].to(q.dtype)
        n_atoms_per_graph = torch.zeros(n_graphs, dtype=q.dtype, device=q.device)
        n_atoms_per_graph.scatter_add_(0, atom_batch, torch.ones_like(q))
        sum_q = torch.zeros(n_graphs, dtype=q.dtype, device=q.device)
        sum_q.scatter_add_(0, atom_batch, q)
        delta = (q_target - sum_q) / n_atoms_per_graph.clamp(min=1.0)
        q_proj = q + delta[atom_batch]
        sum_q_post = torch.zeros(n_graphs, dtype=q.dtype, device=q.device)
        sum_q_post.scatter_add_(0, atom_batch, q_proj)
        return q_proj, sum_q, sum_q_post

    def _molecular_dipole(
        self,
        q: torch.Tensor,
        mu: torch.Tensor | None,
        pos: torch.Tensor,
        atom_batch: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        # μ_mol = Σ_i q_i r_i  (+ Σ_i μ_i, if atomic dipoles are predicted).
        # For neutral systems μ_mol is origin-independent — projection
        # ensures Σ q_i = 0, so the value matches the experimental dipole
        # up to the chosen origin convention.
        per_atom = q.unsqueeze(-1) * pos  # (N, 3)
        if mu is not None:
            per_atom = per_atom + mu
        mu_mol = torch.zeros(n_graphs, 3, dtype=per_atom.dtype, device=per_atom.device)
        mu_mol.scatter_add_(0, atom_batch.unsqueeze(-1).expand_as(per_atom), per_atom)
        return mu_mol
