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
        proper_head=...,  # general Fourier (a, b, periodicity)
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
from molpot.manifest import RepresentationRecord

__all__ = ["Parameterizer"]


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
            partial charges ``atoms.charge`` (``e``) via EEM / QEq. Required when
            ``pair:coulomb`` is registered. Setting it makes
            ``graphs.total_charge`` a required batch field: :meth:`forward`
            forwards it as ``write_forcefield(..., total_charge=...)`` and
            never falls back to the charge model's neutral default.
        polarizability_head: Rejected — named in the signature only so a call
            written against the future polarizable profile fails with an
            explanation instead of a bare ``TypeError``. Polarizability is the
            parameter of an *induced-dipole* response (an atom's dipole
            responding to the electric field of its neighbours), which belongs
            to the A1 profile (``composable-a1``); the A0 registry these heads
            fill declares no induced style, so such a head would have no
            parameter table to write into.

    Raises:
        TypeError: ``encoder`` is not an ``nn.Module``, or ``template`` is
            not a :class:`ForceField`.
        ValueError: A style on ``template`` lacks its required head,
            ``dihedral:opls`` is paired with a proper head whose ``form``
            is not ``"opls"``, or ``polarizability_head`` is not ``None``.
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
        polarizability_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if polarizability_head is not None:
            raise ValueError(
                "polarizability_head is rejected on the A0 Parameterizer: "
                "induced-dipole / polarizability heads belong to the A1 "
                "profile (composable-a1). The A0 registry declares no "
                "induced-response style, so the head has no parameter "
                "table to fill."
            )
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
        # Read once: the template is fixed for the life of the model, and a
        # per-forward ``dihedral_style()`` is a molpy call dynamo cannot trace.
        self._dihedral_style: str | None = self.template.dihedral_style()

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

        improper = self.template.improper_style()
        if improper is not None and self.improper_head is None:
            raise ValueError(f"template declares improper:{improper} but improper_head is None")
        if "pair:lj/cut" in styles and self.atom_head is None:
            raise ValueError("template declares pair:lj/cut but atom_head is None")
        if "pair:coulomb" in styles and self.charge_model is None:
            raise ValueError("template declares pair:coulomb but charge_model is None")

    def representation_record(self) -> RepresentationRecord:
        """Name the representation that produces this model's parameters (S2).

        The parameterizer is the only object that knows which encoder generated
        the parameter tables, so it supplies the manifest's representation
        record; :meth:`ForceField.manifest
        <molpot.forcefield.ForceField.manifest>` leaves that field ``None`` and
        the caller composes the two with :func:`dataclasses.replace`.

        Returns:
            A :class:`~molpot.manifest.RepresentationRecord` with
            ``family="graph-parameterizer"`` (parameters predicted from the
            covalent graph), the encoder's class name, and the encoder's
            ``spec`` mapping when it exposes one — ``{}`` when it does not.
            ``checkpoint_id`` is left ``None``: which weights were loaded is
            the training side's to record (``composable-a0-05``).

        Raises:
            TypeError: The encoder exposes a ``spec`` that is not a mapping.
                Recording ``{}`` for it would claim the encoder is unconfigured
                when it is merely configured through an object this profile
                cannot serialise; a structured encoder spec arrives with
                ``composable-b1-01``.
        """
        spec = getattr(self.encoder, "spec", None)
        if spec is not None and not isinstance(spec, Mapping):
            raise TypeError(
                f"{type(self.encoder).__name__}.spec is a {type(spec).__name__}, not a mapping; "
                "the A0 representation record carries a JSON-native spec (composable-b1-01)"
            )
        return RepresentationRecord(
            family="graph-parameterizer",
            encoder=type(self.encoder).__name__,
            checkpoint_id=None,
            spec={} if spec is None else {str(key): value for key, value in spec.items()},
        )

    def forward(self, batch: TensorDict | Mapping[str, Any]) -> ForceField:
        """Write continuous parameters into a copy of the template ForceField.

        Args:
            batch: Post-collate batch carrying ``atoms.Z`` and the topology
                namespaces (``bonds`` / ``angles`` / ``propers`` /
                ``impropers``) that the encoder and the parameter heads read.
                When ``charge_model`` is set, ``graphs.total_charge`` ``(B,)``
                in ``e`` is required as well and is forwarded to the charge
                model.

        Returns:
            Filled :class:`ForceField` (copy-on-write). The stored
            ``template`` is never mutated. Energy is **not** computed here —
            call ``ff.to_potentials().calc_energy(batch)`` (kcal/mol).

        Raises:
            ValueError: A ``charge_model`` is configured but the encoder
                produced no ``features["atoms"]``, or the batch carries no
                ``graphs.total_charge`` — or carries it as something other
                than a ``(B,)`` tensor of charges in ``e`` (fail-closed:
                neutrality is never assumed); or ``dihedral:opls`` proper
                output lacks ``k1..k4``.
            TypeError: The ``charge_model`` exposes no
                ``write_forcefield(ff, atom_features, batch, total_charge=…)``.
        """
        filled = self.template.with_params(**self.tables(batch))
        # A charge model that only implements the ForceField writer still gets
        # to write: ``tables`` takes the tensor path when there is one, and the
        # export path honours the other. The hot path never calls this.
        if self.charge_model is not None and not callable(
            getattr(self.charge_model, "charges", None)
        ):
            write = getattr(self.charge_model, "write_forcefield", None)
            if not callable(write):
                raise TypeError(
                    "charge_model must provide charges(atom_features, batch, total_charge=…) "
                    "or write_forcefield(ff, atom_features, batch, total_charge=…)"
                )
            features = self._encode(batch)
            atom_feats = features.get("atoms")
            if atom_feats is None:
                raise ValueError("charge_model requires atom features under features['atoms']")
            filled = write(filled, atom_feats, batch, total_charge=self._total_charge(batch))
        return filled

    def tables(self, batch: TensorDict | Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
        """This batch's predicted parameters, as plain tensors per namespace.

        The **hot path**. :meth:`forward` is this plus one molpy
        ``with_params``, and that copy is what the export path wants — a
        :class:`ForceField` to hand to the LAMMPS / OpenMM writers or the
        manifest. A training step wants none of it: molpy objects cannot be
        constructed inside a compiled region (``ForceField.__new__`` is an
        untraceable call), and the tables have to be *computed* inside it
        anyway, because a table produced outside carries a ``grad_fn`` that
        ``torch.autograd.grad`` will not trace through. So the energy consumes
        this mapping directly — :meth:`molpot.forcefield.Potentials.energy`
        takes it as ``tables`` — and molpy stays on the export path.

        Args:
            batch: Post-collate batch carrying ``atoms.Z`` and the topology
                namespaces the encoder and the heads read; plus
                ``graphs.total_charge`` ``(B,)`` in ``e`` when a
                ``charge_model`` is configured.

        Returns:
            Namespace → parameter name → tensor, for the namespaces this
            model's heads fill. Namespaces with no head are absent, not empty.

        Raises:
            ValueError: A ``charge_model`` is configured but the encoder
                produced no ``features["atoms"]``; or the batch carries no
                usable ``graphs.total_charge``; or ``dihedral:opls`` proper
                output lacks ``k1..k4``.
        """
        features = self._encode(batch)
        filled: dict[str, dict[str, torch.Tensor] | None] = {
            "bonds": self._bond_params(features),
            "angles": self._angle_params(features),
            "propers": self._proper_params(features),
            "impropers": self._improper_params(features),
            "atoms": self._atom_params(features, batch),
        }
        if self.charge_model is not None:
            atom_feats = features.get("atoms")
            if atom_feats is None:
                raise ValueError("charge_model requires atom features under features['atoms']")
            charges = getattr(self.charge_model, "charges", None)
            if callable(charges):
                q = charges(atom_feats, batch, total_charge=self._total_charge(batch))
                # ``charge`` is the alias heads written against the older
                # spelling read; both name the same tensor, never two.
                filled["atoms"] = {**(filled["atoms"] or {}), "charge": q}
        return {name: table for name, table in filled.items() if table}

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

    def _total_charge(self, batch: TensorDict | Mapping[str, Any]) -> torch.Tensor:
        """Read the per-graph total charge ``(B,)`` required by ``charge_model``.

        The *total charge* of a graph is the net charge of that molecule or
        ion, the number the charge model constrains its per-atom partial
        charges to sum to. Nothing here defaults it: a neutral molecule
        declares an explicit ``0.0``.

        Args:
            batch: Post-collate batch expected to carry ``graphs.total_charge``
                ``(B,)`` in elementary charges ``e`` — ``0.0`` entries for
                neutral graphs, spelled out.

        Returns:
            Per-graph total charge ``(B,)`` in ``e``, forwarded to the charge
            model untouched.

        Raises:
            ValueError: ``graphs.total_charge`` is absent, or is present as
                something other than a tensor.
        """
        graphs_ns: object = (
            batch["graphs"]
            if isinstance(batch, (Mapping, TensorDict)) and "graphs" in batch
            else None
        )
        total_charge: object = None
        if isinstance(graphs_ns, TensorDict):
            if "total_charge" in graphs_ns:
                total_charge = graphs_ns["total_charge"]
        elif isinstance(graphs_ns, Mapping):
            total_charge = graphs_ns.get("total_charge")

        if total_charge is None:
            raise ValueError(
                "charge_model is configured but the batch carries no "
                "graphs.total_charge: a Parameterizer with a charge model "
                "never assumes neutrality, because equilibrating an ion as a "
                "neutral molecule shifts every partial charge by a constant "
                "with nothing in the run record to say so. Provide "
                "graphs.total_charge (B,) in e — write explicit zeros for "
                "neutral graphs."
            )
        if not isinstance(total_charge, torch.Tensor):
            raise ValueError(
                "graphs.total_charge must be a (B,) tensor of charges in e, "
                f"got {type(total_charge).__name__}"
            )
        return total_charge

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
        return dict(self.proper_head(features["propers"]))

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
