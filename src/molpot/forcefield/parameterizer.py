"""Parameterizer — chem encoder + MM heads → filled ForceField.

Single responsibility: map chemistry → parameter tables on a
:class:`~molpot.forcefield.ForceField` template. Does **not** evaluate
energy (that is ``ForceField.to_potentials()``). Does **not** own IO.

Schema-gated heads: construction reads the template's registered styles
via :meth:`~molpot.forcefield.PotentialMeta.schema_for` and requires the
matching head (or ``charge_model``) for each declared namespace. OPLS-AA
templates (``dihedral:opls``) require a proper head that emits ``k1..k4``.

::

    from molpot.forcefield import OPLSAAForceField, Parameterizer

    model = Parameterizer(
        encoder,
        OPLSAAForceField(),
        bond_head=...,
        angle_head=...,
        proper_head=...,  # form="opls" → k1..k4
        improper_head=...,
        atom_head=...,
        charge_model=...,
    )
    ff = model(batch)  # ForceField copy with params filled
    E = ff.to_potentials().calc_energy(batch)  # kcal/mol
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.forcefield.container import ForceField
from molpot.forcefield.schema import PotentialMeta

__all__ = ["Parameterizer"]

_OPLS_PROPER_KEYS = frozenset({"k1", "k2", "k3", "k4"})


class Parameterizer(nn.Module):
    """Trainable force-field parameterizer: encoder + heads → ForceField.

    Predicts Class-I / OPLS-AA parameter tables as continuous functions of
    the covalent graph. The default product template is
    :class:`~molpot.forcefield.OPLSAAForceField`; any
    :class:`~molpot.forcefield.ForceField` with registered styles works —
    only styles present on ``template`` require heads.

    Units follow LAMMPS ``units="real"``: energy kcal/mol, length Å,
    charge elementary charge ``e``, angles in rad.

    Args:
        encoder: Chemical perception module (``forward`` / ``embeddings``)
            writing per-namespace features. Expected to read topology only
            and never a coordinate, so predicted parameters are geometry-
            independent and deployed forces are exactly ``-∂E/∂pos``.
        template: ForceField whose styles define the classical form and
            which heads are required.
        bond_head: Bond features → ``k`` (kcal mol⁻¹ Å⁻²) and ``r0`` (Å).
            Required when ``bond:harmonic`` is registered.
        angle_head: Angle features → ``k`` (kcal mol⁻¹ rad⁻²) and
            ``theta0`` (rad). Required when ``angle:harmonic`` is registered.
        proper_head: Proper-torsion features → OPLS ``k1..k4`` (kcal/mol)
            when ``dihedral:opls``, or Fourier amplitudes when
            ``dihedral:fourier`` / ``periodic``. Required when any dihedral
            style is registered.
        improper_head: Improper features → harmonic and/or cosine improper
            params (amplitudes in kcal/mol; ``chi0`` / ``phase`` in rad).
            Required when an improper style is registered.
        atom_head: Atom features → per-atom nonbonded params, e.g. LJ
            ``epsilon`` (kcal/mol) and ``sigma`` (Å). Required when
            ``pair:lj/cut`` is registered.
        charge_model: :class:`~molpot.charge.AtomicChargeModel` that writes
            partial charges ``atoms.q`` (``e``) via EEM / QEq. Required when
            ``pair:coulomb`` is registered.

    Raises:
        TypeError: ``encoder`` is not an ``nn.Module``, or ``template`` is
            not a :class:`ForceField`.
        ValueError: A style on ``template`` lacks its required head, or
            ``dihedral:opls`` is paired with a proper head whose ``form``
            is not ``"opls"``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        template: ForceField,
        *,
        bond_head: nn.Module | None = None,
        angle_head: nn.Module | None = None,
        proper_head: nn.Module | None = None,
        improper_head: nn.Module | None = None,
        atom_head: nn.Module | None = None,
        charge_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be an nn.Module")
        if not isinstance(template, ForceField):
            raise TypeError("template must be a ForceField")
        self.encoder = encoder
        self.template = template
        self.bond_head = bond_head
        self.angle_head = angle_head
        self.proper_head = proper_head
        self.improper_head = improper_head
        self.atom_head = atom_head
        self.charge_model = charge_model
        self._validate_schema_heads()

    def _validate_schema_heads(self) -> None:
        """Require heads for every style registered on ``template``."""
        styles = set(self.template.style_names())
        PotentialMeta.schema_for(list(styles), units=self.template.units)

        if "bond:harmonic" in styles and self.bond_head is None:
            raise ValueError("template declares bond:harmonic but bond_head is None")
        if "angle:harmonic" in styles and self.angle_head is None:
            raise ValueError("template declares angle:harmonic but angle_head is None")

        dihedral = self.template.dihedral_style()
        if dihedral is not None and self.proper_head is None:
            raise ValueError(f"template declares dihedral:{dihedral} but proper_head is None")
        if dihedral == "opls":
            form = getattr(self.proper_head, "form", None)
            if form is not None and form != "opls":
                raise ValueError(
                    "template declares dihedral:opls but proper_head.form "
                    f"is {form!r}; expected form='opls' (k1..k4)"
                )

        improper = self.template.improper_style()
        if improper is not None and self.improper_head is None:
            raise ValueError(f"template declares improper:{improper} but improper_head is None")
        if "pair:lj/cut" in styles and self.atom_head is None:
            raise ValueError("template declares pair:lj/cut but atom_head is None")
        if "pair:coulomb" in styles and self.charge_model is None:
            raise ValueError("template declares pair:coulomb but charge_model is None")

    def forward(self, batch: TensorDict | Mapping[str, Any]) -> ForceField:
        """Write continuous parameters into a copy of the template ForceField.

        Args:
            batch: Post-collate batch carrying ``atoms.Z`` and the topology
                namespaces (``bonds`` / ``angles`` / ``propers`` /
                ``impropers``) that the encoder and the parameter heads read.

        Returns:
            Filled :class:`ForceField` (copy-on-write). The stored
            ``template`` is never mutated. Energy is **not** computed here —
            call ``ff.to_potentials().calc_energy(batch)`` (kcal/mol).

        Raises:
            ValueError: A ``charge_model`` is configured but the encoder
                produced no ``features["atoms"]``; or ``dihedral:opls``
                proper output lacks ``k1..k4``.
            TypeError: The ``charge_model`` exposes no
                ``write_forcefield(ff, atom_features, batch)``.
        """
        features = self._encode(batch)
        propers = self._proper_params(features)
        filled = self.template.with_params(
            bonds=self._bond_params(features),
            angles=self._angle_params(features),
            propers=propers,
            impropers=self._improper_params(features),
            atoms=self._atom_params(features, batch),
        )
        if self.charge_model is not None:
            atom_feats = features.get("atoms")
            if atom_feats is None:
                raise ValueError("charge_model requires atom features under features['atoms']")
            write = getattr(self.charge_model, "write_forcefield", None)
            if not callable(write):
                raise TypeError(
                    "charge_model must provide write_forcefield(ff, atom_features, batch)"
                )
            filled = write(filled, atom_feats, batch)
        return filled

    # --- private: encode & heads ---

    def _encode(self, batch: TensorDict | Mapping[str, Any]) -> dict[str, torch.Tensor]:
        embeddings_fn = getattr(self.encoder, "embeddings", None)
        if callable(embeddings_fn):
            forward = getattr(self.encoder, "forward", None)
            if callable(forward):
                try:
                    forward(batch)
                except TypeError:
                    pass
            return self._normalize_features(embeddings_fn(batch))

        forward = getattr(self.encoder, "forward", None)
        if not callable(forward):
            raise RuntimeError("encoder must implement forward(batch) and/or embeddings(batch)")
        result = forward(batch)

        from_batch = self._chem_features_on(batch)
        if from_batch is not None:
            return from_batch
        if result is not None and result is not batch:
            from_result = self._chem_features_on(result)
            if from_result is not None:
                return from_result
            if isinstance(result, Mapping) and any(
                k in result
                for k in (
                    "atoms",
                    "bonds",
                    "angles",
                    "propers",
                    "impropers",
                    "atom",
                    "bond",
                )
            ):
                return self._normalize_features(result)

        raise RuntimeError(
            "encoder produced no features: expected embeddings() or *.chem_features on batch"
        )

    def _normalize_features(self, emb: Any) -> dict[str, torch.Tensor]:
        if isinstance(emb, Mapping):
            alias = {
                "atom": "atoms",
                "bond": "bonds",
                "angle": "angles",
                "proper": "propers",
                "improper": "impropers",
            }
            return {alias.get(k, k): v for k, v in emb.items()}
        interaction = getattr(emb, "interaction_dict", None)
        if callable(interaction):
            return self._normalize_features(dict(interaction()))
        as_dict = getattr(emb, "as_dict", None)
        if callable(as_dict):
            return self._normalize_features(as_dict())
        raise TypeError(
            "encoder embeddings must be a Mapping or expose interaction_dict()/as_dict()"
        )

    def _chem_features_on(self, batch: Any) -> dict[str, torch.Tensor] | None:
        if batch is None:
            return None
        out: dict[str, torch.Tensor] = {}
        for ns_name in ("atoms", "bonds", "angles", "propers", "impropers"):
            try:
                if ns_name not in batch:
                    continue
                ns = batch[ns_name]
                if isinstance(ns, (Mapping, TensorDict)) and "chem_features" in ns:
                    out[ns_name] = ns["chem_features"]
            except Exception:  # noqa: BLE001
                continue
        return out or None

    def _bond_params(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        if self.bond_head is None or "bonds" not in features:
            return None
        return dict(self.bond_head(features["bonds"]))

    def _angle_params(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        if self.angle_head is None or "angles" not in features:
            return None
        return dict(self.angle_head(features["angles"]))

    def _proper_params(
        self, features: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor] | None:
        if self.proper_head is None or "propers" not in features:
            return None
        params = dict(self.proper_head(features["propers"]))
        if self.template.dihedral_style() == "opls":
            missing = _OPLS_PROPER_KEYS - params.keys()
            if missing:
                raise ValueError(
                    f"dihedral:opls requires proper params k1..k4; missing {sorted(missing)}"
                )
        return params

    def _improper_params(
        self, features: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor] | None:
        if self.improper_head is None or "impropers" not in features:
            return None
        return dict(self.improper_head(features["impropers"]))

    def _atom_params(
        self,
        features: Mapping[str, torch.Tensor],
        batch: Any,
    ) -> dict[str, torch.Tensor] | None:
        if self.atom_head is None or "atoms" not in features:
            return None
        atoms_ns = (
            batch["atoms"]
            if isinstance(batch, (Mapping, TensorDict)) and "atoms" in batch
            else None
        )
        kwargs: dict[str, Any] = {}
        if isinstance(atoms_ns, (Mapping, TensorDict)):
            if "batch" in atoms_ns:
                kwargs["batch"] = atoms_ns["batch"]
            if "Z" in atoms_ns:
                kwargs["Z"] = atoms_ns["Z"]
        out = self.atom_head(features["atoms"], **kwargs)
        return dict(out) if out else None
