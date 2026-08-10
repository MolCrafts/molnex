"""ClassicalMMParameterizer: encoder → MM heads → PotentialIR → Class-I E/F.

End-to-end training-time module for learnable classical force fields.
Accepts a chemical-perception encoder **via Protocol** so ``molpot`` never
imports ``molzoo`` / ``molrep.chem``. Heads and energy evaluation are owned by
:class:`~molpot.composition.classical_mm.ClassicalMMComposer` (spec 03).

Pipeline primitives (prefer explicit methods over one opaque façade)::

    features = param.encode(batch)
    ir = param.parameterize(batch)           # or features=
    energy = param.energy(batch)             # or ir=
    out = param(batch, compute_forces=True)  # thin composition

Units: CLASS_I_CANONICAL (kcal/mol, Å, e, rad) inside IR and energy.
Optional eV conversion only via :func:`energy_kcal_to_ev` at the loss edge —
never by mutating IR units.

Forces: solely :class:`~molpot.derivation.ForceDerivation`
(``F = -∂E/∂r``). No hand-rolled force kernels.

References:
    Spec: learnable-classical-ff-05-neural-parameterizer
    OpenMM User Guide §19 "Forces"
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.composition.classical_mm import ClassicalMMComposer
from molpot.derivation import ForceDerivation
from molpot.derivation.protocol import write_energy, write_forces
from molpot.ir import PotentialIR

__all__ = [
    "ChemEmbeddingsLike",
    "ChemEncoderProtocol",
    "ClassicalMMParameterizer",
    "KCAL_MOL_TO_EV",
    "energy_kcal_to_ev",
]

# kcal/mol → eV. 1 eV = 23.060547830619026 kcal/mol (CODATA / OpenMM-style).
# Used only at the training-loss boundary — never redefines IR units.
KCAL_MOL_TO_EV: float = 1.0 / 23.060547830619026


def energy_kcal_to_ev(energy_kcal: torch.Tensor) -> torch.Tensor:
    """Convert Class-I energy from kcal/mol to eV **without mutating IR**.

    Args:
        energy_kcal: Energy tensor in kcal/mol (any shape).

    Returns:
        Same-shaped tensor in eV (``energy_kcal * KCAL_MOL_TO_EV``).

    Notes:
        Call this only at the loss / logging boundary. Parameterize and
        classical energy stay in CLASS_I_CANONICAL (kcal/mol).
    """
    return energy_kcal * KCAL_MOL_TO_EV


@runtime_checkable
class ChemEmbeddingsLike(Protocol):
    """Structural type for continuous chem feature payloads.

    Duck-typed so molpot never imports ``molrep.chem.ChemEmbeddings``.
    Implementations expose :meth:`interaction_dict` with keys consumable by
    :class:`~molpot.composition.classical_mm.ClassicalMMComposer`
    (``atoms`` / ``bonds`` / ``angles`` / ``propers`` / ``impropers``).
    """

    def interaction_dict(self) -> Mapping[str, torch.Tensor]:
        """Feature mapping keyed for ClassicalMMComposer heads."""
        ...


@runtime_checkable
class ChemEncoderProtocol(Protocol):
    """Minimal encoder surface for classical MM parameterization.

    Implementations live in molrep/molzoo; molpot only sees this Protocol.
    Either :meth:`forward` writing batch chem features, or
    :meth:`embeddings` returning a :class:`ChemEmbeddingsLike`, or both.
    """

    def forward(self, td: TensorDict) -> TensorDict:
        """Run chemical perception; may write ``*.chem_features`` on ``td``."""
        ...


# Optional embeddings method is not required by the Protocol class itself
# (runtime_checkable only checks methods declared on the Protocol). Callers
# and ClassicalMMParameterizer.encode discover embeddings via getattr.


_FEATURE_NS_KEYS: tuple[tuple[str, str], ...] = (
    ("atoms", "atoms"),
    ("bonds", "bonds"),
    ("angles", "angles"),
    ("propers", "propers"),
    ("impropers", "impropers"),
)


def _features_from_batch_chem(batch: Any) -> dict[str, torch.Tensor] | None:
    """Extract ``*.chem_features`` written by a chem encoder, if present."""
    if batch is None:
        return None
    out: dict[str, torch.Tensor] = {}
    for ns_name, feat_key in _FEATURE_NS_KEYS:
        try:
            if ns_name not in batch:
                continue
            ns = batch[ns_name]
            if isinstance(ns, (Mapping, TensorDict)) and "chem_features" in ns:
                out[feat_key] = ns["chem_features"]
        except Exception:  # noqa: BLE001 — TensorDict key miss variants
            continue
    return out or None


def _features_from_embeddings(emb: Any) -> dict[str, torch.Tensor]:
    """Normalize ChemEmbeddingsLike / Mapping into composer feature keys."""
    if isinstance(emb, Mapping):
        # Accept plural (composer) or singular (ChemEmbeddings.as_dict) keys.
        alias = {
            "atom": "atoms",
            "bond": "bonds",
            "angle": "angles",
            "proper": "propers",
            "improper": "impropers",
        }
        out: dict[str, torch.Tensor] = {}
        for k, v in emb.items():
            out[alias.get(k, k)] = v
        return out
    interaction = getattr(emb, "interaction_dict", None)
    if callable(interaction):
        return dict(interaction())
    as_dict = getattr(emb, "as_dict", None)
    if callable(as_dict):
        return _features_from_embeddings(as_dict())
    raise TypeError("encoder embeddings must be a Mapping or expose interaction_dict()/as_dict()")


def _pos_from_batch(batch: Any, pos: torch.Tensor | None) -> torch.Tensor:
    if pos is not None:
        return pos
    if isinstance(batch, (Mapping, TensorDict)):
        try:
            if "atoms" in batch:
                atoms = batch["atoms"]
                if isinstance(atoms, (Mapping, TensorDict)) and "pos" in atoms:
                    return atoms["pos"]
        except Exception:  # noqa: BLE001
            pass
        if "pos" in batch:
            return batch["pos"]
    raise ValueError("ClassicalMMParameterizer requires pos or batch['atoms','pos']")


class ClassicalMMParameterizer(nn.Module):
    """encoder (Protocol) → MM heads → PotentialIR → classical E (+ optional F).

    Args:
        encoder: Chem-perception module satisfying
            :class:`ChemEncoderProtocol` (registered if ``nn.Module``).
        composer: :class:`~molpot.composition.classical_mm.ClassicalMMComposer`
            owning heads and Class-I energy evaluation.
        force_derivation: Optional
            :class:`~molpot.derivation.ForceDerivation`. Defaults to
            autograd backend (universal). Forces are never hand-rolled.

    Notes:
        Prefer :meth:`encode`, :meth:`parameterize`, and :meth:`energy` as
        separate primitives. :meth:`forward` is a thin chain for training
        loops (``compute_forces`` optional).
    """

    def __init__(
        self,
        encoder: nn.Module,
        composer: ClassicalMMComposer,
        force_derivation: ForceDerivation | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be an nn.Module (Protocol surface)")
        if not isinstance(composer, ClassicalMMComposer):
            raise TypeError("composer must be a ClassicalMMComposer")
        self.encoder = encoder
        self.composer = composer
        self.force_derivation = (
            force_derivation if force_derivation is not None else ForceDerivation(method="autograd")
        )

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def encode(self, batch: TensorDict | Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Run the encoder Protocol and return composer-keyed features.

        Resolution order:

        1. If the encoder exposes ``embeddings(batch)``, use that payload.
        2. Else run ``encoder.forward(batch)`` and read ``*.chem_features``.
        3. Else, if ``forward`` returned a Mapping of feature tensors, use it.

        Args:
            batch: Nested TensorDict (or Mapping) with topology + geometry.

        Returns:
            Feature tensors keyed ``atoms`` / ``bonds`` / ``angles`` /
            ``propers`` / ``impropers`` (subset present).

        Raises:
            RuntimeError: If no features can be resolved from the encoder.
        """
        embeddings_fn = getattr(self.encoder, "embeddings", None)
        if callable(embeddings_fn):
            # Prefer embeddings() after a forward pass so side-effect writers
            # (e.g. ChemEncoder writing chem_features) stay consistent.
            forward = getattr(self.encoder, "forward", None)
            if callable(forward):
                try:
                    forward(batch)  # type: ignore[arg-type]
                except TypeError:
                    # Some fakes only implement embeddings; ignore.
                    pass
            emb = embeddings_fn(batch)
            # ChemEncoder.embeddings reads written chem_features; if the
            # encoder was not run (or is FakeEncoder with direct embeddings),
            # emb is still valid.
            return _features_from_embeddings(emb)

        forward = getattr(self.encoder, "forward", None)
        if not callable(forward):
            raise RuntimeError("encoder must implement forward(batch) and/or embeddings(batch)")
        result = forward(batch)  # type: ignore[misc]

        # Features written onto the batch under *.chem_features
        from_batch = _features_from_batch_chem(batch)
        if from_batch is not None:
            return from_batch
        if result is not None and result is not batch:
            from_result = _features_from_batch_chem(result)
            if from_result is not None:
                return from_result
            if isinstance(result, Mapping) and any(
                k in result
                for k in ("atoms", "bonds", "angles", "propers", "impropers", "atom", "bond")
            ):
                return _features_from_embeddings(result)

        raise RuntimeError(
            "encoder produced no features: expected embeddings() or *.chem_features on batch"
        )

    # ------------------------------------------------------------------
    # parameterize
    # ------------------------------------------------------------------

    def parameterize(
        self,
        batch: TensorDict | Mapping[str, Any],
        features: Mapping[str, torch.Tensor] | None = None,
    ) -> PotentialIR:
        """Heads → Class-I :class:`~molpot.ir.PotentialIR` (kcal/mol…).

        Args:
            batch: Topology batch (valence namespaces).
            features: Optional precomputed feature dict. When ``None``,
                :meth:`encode` runs the encoder Protocol.

        Returns:
            :class:`PotentialIR` with ``unit_system="class_i_canonical"``.
        """
        feats = features if features is not None else self.encode(batch)
        return self.composer.parameterize(feats, batch)

    # ------------------------------------------------------------------
    # energy
    # ------------------------------------------------------------------

    def energy(
        self,
        batch: TensorDict | Mapping[str, Any],
        ir: PotentialIR | None = None,
        *,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classical Class-I energy sum (kcal/mol).

        Args:
            batch: Topology + geometry batch.
            ir: Optional precomputed IR; when ``None``, :meth:`parameterize`.
            pos: Optional positions ``(N, 3)``; else read from batch.

        Returns:
            Scalar energy in kcal/mol.
        """
        bag = ir if ir is not None else self.parameterize(batch)
        return self.composer.energy(bag, batch, pos=pos)

    # ------------------------------------------------------------------
    # forward (thin composition)
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: TensorDict,
        *,
        compute_forces: bool = False,
        features: Mapping[str, torch.Tensor] | None = None,
        pos: torch.Tensor | None = None,
        return_terms: bool = False,
    ) -> dict[str, Any]:
        """Thin chain: encode → parameterize → energy (+ optional forces).

        Args:
            batch: Nested TensorDict with topology and ``atoms.pos``.
            compute_forces: If True, derive ``F = -∂E/∂pos`` via
                :class:`~molpot.derivation.ForceDerivation` and write
                ``atoms.forces``.
            features: Optional feature override (skips encoder).
            pos: Optional positions override.
            return_terms: If True, include per-term energy breakdown.

        Returns:
            Dict with ``energy`` (scalar kcal/mol), ``ir``
            (:class:`PotentialIR`), and optionally ``forces`` ``(N, 3)`` and
            ``term_energies``. Also writes ``graphs.energy`` (and
            ``atoms.forces`` when requested) onto ``batch`` in place.
        """
        ir = self.parameterize(batch, features=features)
        p = _pos_from_batch(batch, pos)
        energy = self.composer.energy(ir, batch, pos=p)

        # Peer keys on the batch (molix schema: graphs.energy).
        # Scalar energy → ensure_graphs with batch_size=[] when dim==0.
        if isinstance(batch, TensorDict):
            write_energy(batch, energy)

        out: dict[str, Any] = {"energy": energy, "ir": ir}
        if return_terms:
            out["term_energies"] = self.composer.term_energies(ir, batch, pos=p)

        if compute_forces:

            def energy_fn(positions: torch.Tensor) -> torch.Tensor:
                return self.composer.energy(ir, batch, pos=positions)

            forces = self.force_derivation(energy_fn, p)
            out["forces"] = forces
            if isinstance(batch, TensorDict):
                write_forces(batch, forces)

        return out
