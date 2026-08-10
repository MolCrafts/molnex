"""ClassicalMMComposer: features → MM heads → PotentialIR → Class-I energy.

Does **not** own a chemical encoder. The caller injects feature tensors keyed
by interaction class (and optional atom features for LJ/charge). Sibling of
:class:`~molpot.composition.composer.PotentialComposer` specialized for
Class-I IR bags; does not subclass Sonata.

Pipeline primitives (caller-composed)::

    ir = composer.parameterize(features, batch)
    energy = composer.energy(ir, batch, pos=...)
    # thin convenience:
    out = composer(batch, features)  # {"energy", "ir", "term_energies"?}

Forces are **not** hand-rolled. Differentiate the energy path with
:class:`~molpot.derivation.ForceDerivation` / ``BasePotential.calc_forces``.

Units: CLASS_I_CANONICAL (kcal/mol, Å, e, rad).

References:
    Spec: learnable-classical-ff-03-mm-heads
    OpenMM User Guide §19 "Forces"
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.ir import (
    AngleBag,
    BondBag,
    ChargeBag,
    ImproperHarmonicBag,
    ImproperPeriodicBag,
    LJBag,
    NonbondedScaling,
    PotentialIR,
    ProperTorsionBag,
)
from molpot.potentials import (
    AngleHarmonic,
    BondHarmonic,
    ImproperHarmonic,
    ImproperPeriodic,
    ProperTorsionPeriodic,
)

__all__ = ["ClassicalMMComposer"]

# Evaluator: (ir, batch, *, pos) -> scalar energy
EnergyEvaluator = Callable[..., torch.Tensor]


def _get_namespace(batch: Any, name: str) -> Any | None:
    """Fetch a top-level namespace from TensorDict / Mapping."""
    if batch is None:
        return None
    if isinstance(batch, Mapping) or isinstance(batch, TensorDict):
        try:
            if name in batch:
                return batch[name]
        except Exception:  # noqa: BLE001 — TensorDict key miss variants
            return None
        return None
    return None


def _ns_tensor(ns: Any, *keys: str) -> torch.Tensor | None:
    """Read a tensor from a namespace under any of ``keys``."""
    if ns is None:
        return None
    for key in keys:
        try:
            if isinstance(ns, (Mapping, TensorDict)) and key in ns:
                return ns[key]
        except Exception:  # noqa: BLE001
            continue
    return None


def _bond_index_from_batch(batch: Any) -> torch.Tensor | None:
    """Build COO ``bond_index`` ``[2, N]`` from column namespaces or packed keys."""
    bonds = _get_namespace(batch, "bonds")
    if bonds is not None:
        packed = _ns_tensor(bonds, "bond_index")
        if packed is not None:
            return packed
        atomi = _ns_tensor(bonds, "atomi")
        atomj = _ns_tensor(bonds, "atomj")
        if atomi is not None and atomj is not None:
            return torch.stack([atomi.long(), atomj.long()], dim=0)
    # Flat legacy
    if isinstance(batch, Mapping) and "bond_index" in batch:
        return batch["bond_index"]
    return None


def _angle_index_from_batch(batch: Any) -> torch.Tensor | None:
    angles = _get_namespace(batch, "angles")
    if angles is None:
        return None
    atomi = _ns_tensor(angles, "atomi")
    atomj = _ns_tensor(angles, "atomj")
    atomk = _ns_tensor(angles, "atomk")
    if atomi is None or atomj is None or atomk is None:
        return None
    return torch.stack([atomi.long(), atomj.long(), atomk.long()], dim=0)


def _proper_index_from_batch(batch: Any) -> torch.Tensor | None:
    propers = _get_namespace(batch, "propers")
    if propers is None:
        return None
    cols = [_ns_tensor(propers, k) for k in ("atomi", "atomj", "atomk", "atoml")]
    if any(c is None for c in cols):
        return None
    return torch.stack([c.long() for c in cols], dim=0)  # type: ignore[union-attr]


def _improper_index_from_batch(batch: Any) -> torch.Tensor | None:
    impropers = _get_namespace(batch, "impropers")
    if impropers is None:
        return None
    cols = [_ns_tensor(impropers, k) for k in ("atomi", "atomj", "atomk", "atoml")]
    if any(c is None for c in cols):
        return None
    # molrs center-first: atomi = center
    return torch.stack([c.long() for c in cols], dim=0)  # type: ignore[union-attr]


def _pos_from_batch(batch: Any, pos: torch.Tensor | None) -> torch.Tensor:
    if pos is not None:
        return pos
    atoms = _get_namespace(batch, "atoms")
    if atoms is not None:
        p = _ns_tensor(atoms, "pos")
        if p is not None:
            return p
    if isinstance(batch, Mapping) and "pos" in batch:
        return batch["pos"]
    raise ValueError("ClassicalMMComposer.energy requires pos or batch atoms.pos")


def _identity_types(n: int, device: torch.device) -> torch.Tensor:
    return torch.arange(n, device=device, dtype=torch.long)


class ClassicalMMComposer(nn.Module):
    """Wire continuous MM heads into Class-I :class:`~molpot.ir.PotentialIR`.

    Args:
        bond_head: Optional :class:`~molpot.composition.mm_heads.BondParamHead`
            (or compatible module) mapping bond features → ``k``, ``r0``.
        angle_head: Optional angle parameter head.
        proper_head: Optional proper-torsion parameter head.
        improper_head: Optional improper parameter head (harmonic and/or
            periodic outputs).
        atom_head: Optional atom-level head (typically
            :class:`~molpot.composition.multihead.MultiHead` of
            :class:`~molpot.composition.heads.LJParameterHead` and
            :class:`~molpot.composition.heads.ChargeHead`).
        scaling: Optional nonbonded scaling; defaults to Class-I AMBER/GAFF.
        evaluator: Optional callable ``(ir, batch, *, pos) -> energy``. When
            set, :meth:`energy` delegates to it instead of built-in kernels.

    Notes:
        Prefer :meth:`parameterize` and :meth:`energy` as separate primitives.
        :meth:`forward` is a thin chain for ``nn.Module`` convenience only.
    """

    def __init__(
        self,
        *,
        bond_head: nn.Module | None = None,
        angle_head: nn.Module | None = None,
        proper_head: nn.Module | None = None,
        improper_head: nn.Module | None = None,
        atom_head: nn.Module | None = None,
        scaling: NonbondedScaling | None = None,
        evaluator: EnergyEvaluator | None = None,
    ) -> None:
        super().__init__()
        self.bond_head = bond_head
        self.angle_head = angle_head
        self.proper_head = proper_head
        self.improper_head = improper_head
        self.atom_head = atom_head
        self.scaling = scaling if scaling is not None else NonbondedScaling()
        self.evaluator = evaluator

    # ------------------------------------------------------------------
    # parameterize
    # ------------------------------------------------------------------

    def parameterize(
        self,
        features: Mapping[str, torch.Tensor],
        batch: Any,
    ) -> PotentialIR:
        """Run enabled heads and assemble a Class-I :class:`PotentialIR`.

        Args:
            features: Feature tensors keyed by interaction class. Supported
                keys: ``"bonds"``, ``"angles"``, ``"propers"``,
                ``"impropers"``, ``"atoms"``. Missing keys skip that bag.
            batch: Topology batch (TensorDict or Mapping) with valence
                namespaces. Used for atom ``batch`` index when charge
                neutrality is required; topology counts are taken from
                feature row counts.

        Returns:
            :class:`PotentialIR` with ``unit_system="class_i_canonical"`` and
            bags populated for enabled terms. Default
            :class:`~molpot.ir.NonbondedScaling` is always attached.
        """
        bonds = self._run_bond_head(features)
        angles = self._run_angle_head(features)
        propers = self._run_proper_head(features)
        imp_h, imp_p = self._run_improper_head(features)
        lj, charges = self._run_atom_head(features, batch)

        return PotentialIR(
            bonds=bonds,
            angles=angles,
            propers=propers,
            impropers_harmonic=imp_h,
            impropers_periodic=imp_p,
            lj=lj,
            charges=charges,
            scaling=self.scaling,
            unit_system="class_i_canonical",
        )

    def _run_bond_head(self, features: Mapping[str, torch.Tensor]) -> BondBag | None:
        if self.bond_head is None or "bonds" not in features:
            return None
        out = self.bond_head(features["bonds"])
        return BondBag(k=out["k"], r0=out["r0"])

    def _run_angle_head(self, features: Mapping[str, torch.Tensor]) -> AngleBag | None:
        if self.angle_head is None or "angles" not in features:
            return None
        out = self.angle_head(features["angles"])
        return AngleBag(k=out["k"], theta0=out["theta0"])

    def _run_proper_head(self, features: Mapping[str, torch.Tensor]) -> ProperTorsionBag | None:
        if self.proper_head is None or "propers" not in features:
            return None
        out = self.proper_head(features["propers"])
        return ProperTorsionBag(
            k=out["k"],
            periodicity=out["periodicity"],
            phase=out["phase"],
            idivf=out["idivf"],
        )

    def _run_improper_head(
        self, features: Mapping[str, torch.Tensor]
    ) -> tuple[ImproperHarmonicBag | None, ImproperPeriodicBag | None]:
        if self.improper_head is None or "impropers" not in features:
            return None, None
        out = self.improper_head(features["impropers"])
        harm: ImproperHarmonicBag | None = None
        peri: ImproperPeriodicBag | None = None

        if "chi0" in out:
            k_h = out.get("k_harmonic", out.get("k"))
            if k_h is not None and k_h.ndim == 1:
                harm = ImproperHarmonicBag(k=k_h, chi0=out["chi0"])

        k_p = out.get("k_periodic")
        if k_p is None and "phase" in out and "periodicity" in out:
            k_candidate = out.get("k")
            if k_candidate is not None and k_candidate.ndim == 2:
                k_p = k_candidate
        if k_p is not None and "phase" in out and "periodicity" in out:
            peri = ImproperPeriodicBag(
                k=k_p,
                periodicity=out["periodicity"],
                phase=out["phase"],
                idivf=out["idivf"],
            )
        return harm, peri

    def _run_atom_head(
        self,
        features: Mapping[str, torch.Tensor],
        batch: Any,
    ) -> tuple[LJBag | None, ChargeBag | None]:
        if self.atom_head is None or "atoms" not in features:
            return None, None
        atom_feats = features["atoms"]
        atoms_ns = _get_namespace(batch, "atoms")
        batch_idx = _ns_tensor(atoms_ns, "batch") if atoms_ns is not None else None
        if batch_idx is None and isinstance(batch, Mapping):
            batch_idx = batch.get("batch")
        kwargs: dict[str, Any] = {}
        if batch_idx is not None:
            kwargs["batch"] = batch_idx
        z = _ns_tensor(atoms_ns, "Z") if atoms_ns is not None else None
        if z is not None:
            kwargs["Z"] = z
        out = self.atom_head(atom_feats, **kwargs)
        lj = None
        charges = None
        if "epsilon" in out and "sigma" in out:
            lj = LJBag(epsilon=out["epsilon"], sigma=out["sigma"])
        q = out.get("charge", out.get("q"))
        if q is not None:
            charges = ChargeBag(q=q)
        return lj, charges

    # ------------------------------------------------------------------
    # energy
    # ------------------------------------------------------------------

    def energy(
        self,
        ir: PotentialIR,
        batch: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate Class-I energy from an IR and geometry.

        Args:
            ir: Parameter bags from :meth:`parameterize` (or constructed).
            batch: Topology batch with valence namespaces.
            pos: Optional positions ``(N, 3)``; else read from batch.

        Returns:
            Scalar energy (kcal/mol).

        Notes:
            Built-in path uses per-interaction continuous parameters with
            identity type indices into the IR bag tables (no discrete type
            table inside the composer). When ``evaluator`` was provided at
            construction, that callable is used instead.
        """
        if self.evaluator is not None:
            return self.evaluator(ir, batch, pos=pos)

        p = _pos_from_batch(batch, pos)
        device = p.device
        dtype = p.dtype
        total = torch.zeros((), device=device, dtype=dtype)
        terms = self._term_energies(ir, batch, pos=p)
        for e in terms.values():
            total = total + e
        return total

    def term_energies(
        self,
        ir: PotentialIR,
        batch: Any,
        *,
        pos: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-term energy breakdown (kcal/mol scalars)."""
        p = _pos_from_batch(batch, pos)
        return self._term_energies(ir, batch, pos=p)

    def _term_energies(
        self,
        ir: PotentialIR,
        batch: Any,
        *,
        pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        terms: dict[str, torch.Tensor] = {}
        device = pos.device

        if ir.bonds is not None:
            bond_index = _bond_index_from_batch(batch)
            if bond_index is not None and bond_index.size(1) > 0:
                n = bond_index.size(1)
                pot = BondHarmonic(k=ir.bonds.k, r0=ir.bonds.r0)
                terms["bonds"] = pot(
                    pos=pos,
                    bond_index=bond_index,
                    bond_types=_identity_types(n, device),
                )
            else:
                terms["bonds"] = torch.zeros((), device=device, dtype=pos.dtype)

        if ir.angles is not None:
            angle_index = _angle_index_from_batch(batch)
            if angle_index is not None and angle_index.size(1) > 0:
                n = angle_index.size(1)
                pot = AngleHarmonic(k=ir.angles.k, theta0=ir.angles.theta0)
                terms["angles"] = pot(
                    pos=pos,
                    angle_index=angle_index,
                    angle_types=_identity_types(n, device),
                )
            else:
                terms["angles"] = torch.zeros((), device=device, dtype=pos.dtype)

        if ir.propers is not None:
            proper_index = _proper_index_from_batch(batch)
            if proper_index is not None and proper_index.size(1) > 0:
                n = proper_index.size(1)
                pot = ProperTorsionPeriodic(
                    k=ir.propers.k,
                    periodicity=ir.propers.periodicity,
                    phase=ir.propers.phase,
                    idivf=ir.propers.idivf,
                )
                terms["propers"] = pot(
                    pos=pos,
                    proper_index=proper_index,
                    proper_types=_identity_types(n, device),
                )
            else:
                terms["propers"] = torch.zeros((), device=device, dtype=pos.dtype)

        if ir.impropers_harmonic is not None:
            improper_index = _improper_index_from_batch(batch)
            if improper_index is not None and improper_index.size(1) > 0:
                n = improper_index.size(1)
                pot = ImproperHarmonic(
                    k=ir.impropers_harmonic.k,
                    chi0=ir.impropers_harmonic.chi0,
                )
                terms["impropers_harmonic"] = pot(
                    pos=pos,
                    improper_index=improper_index,
                    improper_types=_identity_types(n, device),
                )
            else:
                terms["impropers_harmonic"] = torch.zeros((), device=device, dtype=pos.dtype)

        if ir.impropers_periodic is not None:
            improper_index = _improper_index_from_batch(batch)
            if improper_index is not None and improper_index.size(1) > 0:
                n = improper_index.size(1)
                pot = ImproperPeriodic(
                    k=ir.impropers_periodic.k,
                    periodicity=ir.impropers_periodic.periodicity,
                    phase=ir.impropers_periodic.phase,
                    idivf=ir.impropers_periodic.idivf,
                )
                terms["impropers_periodic"] = pot(
                    pos=pos,
                    improper_index=improper_index,
                    improper_types=_identity_types(n, device),
                )
            else:
                terms["impropers_periodic"] = torch.zeros((), device=device, dtype=pos.dtype)

        return terms

    # ------------------------------------------------------------------
    # forward (thin composition)
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: Any,
        features: Mapping[str, torch.Tensor],
        *,
        pos: torch.Tensor | None = None,
        return_terms: bool = False,
    ) -> dict[str, Any]:
        """Thin chain: :meth:`parameterize` then :meth:`energy`.

        Args:
            batch: Topology + geometry batch.
            features: Feature tensors keyed by interaction class.
            pos: Optional positions override.
            return_terms: If True, include ``term_energies`` breakdown.

        Returns:
            Dict with ``energy`` (scalar), ``ir`` (:class:`PotentialIR`), and
            optionally ``term_energies``.
        """
        ir = self.parameterize(features, batch)
        p = _pos_from_batch(batch, pos)
        energy = self.energy(ir, batch, pos=p)
        out: dict[str, Any] = {"energy": energy, "ir": ir}
        if return_terms:
            out["term_energies"] = self.term_energies(ir, batch, pos=p)
        return out
