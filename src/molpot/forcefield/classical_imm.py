"""ClassicalIMM — chem encoder + MM heads → continuous Class-I ForceField.

Single responsibility: map chemistry → parameter tables on a ForceField.
Does **not** evaluate energy (that is ``ForceField.to_potentials()``).
Does **not** own IO.

::

    ff = ClassIForceField()
    model = ClassicalIMM(encoder, forcefield=ff, bond_head=..., ...)
    ff_t = model(batch)                          # → ForceField (copy-on-write)
    E = ff_t.to_potentials().calc_energy(batch)
    ff_opls = model.project(OPLSForceField())    # general → concrete form only
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.forcefield.container import ClassIForceField, ForceField
from molpot.ir import ProperTorsionBag

__all__ = ["ClassicalIMM"]


class ClassicalIMM(nn.Module):
    """Class-I continuous parameterization: encoder + heads → ForceField.

    Args:
        encoder: Chemical perception module (``forward`` / ``embeddings``).
        forcefield: Template ForceField (styles define the classical form).
        bond_head / angle_head / proper_head / improper_head / atom_head:
            Parameter heads (features → continuous MM params).
        charge_model: Optional :class:`~molpot.charge.AtomicChargeModel`
            that writes partial charges ``atoms.q`` via EEM/QEq-family solve.
    """

    def __init__(
        self,
        encoder: nn.Module,
        forcefield: ForceField | None = None,
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
        self.encoder = encoder
        self.forcefield = forcefield if forcefield is not None else ClassIForceField()
        self.bond_head = bond_head
        self.angle_head = angle_head
        self.proper_head = proper_head
        self.improper_head = improper_head
        self.atom_head = atom_head
        self.charge_model = charge_model
        self._last: ForceField | None = None

    def forward(
        self,
        batch: TensorDict | Mapping[str, Any],
        *,
        forcefield: ForceField | None = None,
    ) -> ForceField:
        """Write continuous parameters into a copy of the template ForceField.

        Returns:
            Filled :class:`ForceField` (copy-on-write). Energy is **not**
            computed here — call ``ff.to_potentials().calc_energy(batch)``.
        """
        template = forcefield if forcefield is not None else self.forcefield
        features = self._encode(batch)
        filled = template.with_params(
            bonds=self._bond_params(features),
            angles=self._angle_params(features),
            propers=self._proper_params(features),
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
        self._last = filled
        return filled

    def project(
        self,
        target: ForceField,
        source: ForceField | None = None,
        *,
        atol: float = 1e-8,
    ) -> ForceField:
        """Project continuous/general parameters onto a concrete ForceField form.

        Single responsibility: map source param tables into ``target`` styles.
        Does not train or evaluate energy.

        Raises:
            RuntimeError: No source ForceField available.
            ValueError: Target style cannot be filled from source (missing
                params, incompatible functional form, or non-representable
                Fourier residual when targeting OPLS).
        """
        src = source if source is not None else self._last
        if src is None:
            raise RuntimeError("project requires source=... or a prior forward()")

        out = target.clone()
        out._params = {}

        # Bonds
        if out.bond_style() is not None:
            if out.bond_style() != "harmonic":
                raise ValueError(f"project: unsupported target bond style {out.bond_style()!r}")
            bp = src.params("bonds")
            if bp is None:
                raise ValueError("project: target has bond:harmonic but source has no bonds params")
            if src.bond_style() not in (None, "harmonic"):
                raise ValueError(
                    f"project: cannot map source bond style {src.bond_style()!r} → harmonic"
                )
            out = out.with_params(bonds=dict(bp))

        # Angles
        if out.angle_style() is not None:
            if out.angle_style() != "harmonic":
                raise ValueError(f"project: unsupported target angle style {out.angle_style()!r}")
            ap = src.params("angles")
            if ap is None:
                raise ValueError(
                    "project: target has angle:harmonic but source has no angles params"
                )
            if src.angle_style() not in (None, "harmonic"):
                raise ValueError(
                    f"project: cannot map source angle style {src.angle_style()!r} → harmonic"
                )
            out = out.with_params(angles=dict(ap))

        # Propers / dihedrals
        tgt_d = out.dihedral_style()
        if tgt_d is not None:
            sp = src.params("propers")
            if sp is None:
                raise ValueError(
                    f"project: target has dihedral:{tgt_d} but source has no propers params"
                )
            if tgt_d == "opls":
                out = out.with_params(propers=self._propers_to_opls(sp, atol=atol))
            elif tgt_d in ("fourier", "periodic"):
                if "k1" in sp:
                    # OPLS → multi-term cosine (inverse of _propers_to_opls amplitudes)
                    out = out.with_params(propers=self._opls_to_periodic(sp))
                else:
                    out = out.with_params(propers=dict(sp))
            else:
                raise ValueError(f"project: unsupported target dihedral style {tgt_d!r}")

        # Impropers (pass-through same family only)
        if out.improper_style() is not None:
            ip = src.params("impropers")
            if ip is None:
                raise ValueError(
                    f"project: target has improper:{out.improper_style()} "
                    "but source has no impropers params"
                )
            out = out.with_params(impropers=dict(ip))

        # Pair / atom nonbonded
        if out.pair_style() is not None:
            atoms = src.params("atoms")
            pairs = src.params("pairs")
            if atoms is None and pairs is None:
                raise ValueError(
                    f"project: target has pair:{out.pair_style()} "
                    "but source has no atoms/pairs params"
                )
            if atoms is not None:
                out = out.with_params(atoms=dict(atoms))
            if pairs is not None:
                out = out.with_params(pairs=dict(pairs))

        return out

    # --- private: encode & heads (not free functions) ---

    def _encode(self, batch: TensorDict | Mapping[str, Any]) -> dict[str, torch.Tensor]:
        embeddings_fn = getattr(self.encoder, "embeddings", None)
        if callable(embeddings_fn):
            forward = getattr(self.encoder, "forward", None)
            if callable(forward):
                try:
                    forward(batch)  # type: ignore[arg-type]
                except TypeError:
                    pass
            return self._normalize_features(embeddings_fn(batch))

        forward = getattr(self.encoder, "forward", None)
        if not callable(forward):
            raise RuntimeError("encoder must implement forward(batch) and/or embeddings(batch)")
        result = forward(batch)  # type: ignore[misc]

        from_batch = self._chem_features_on(batch)
        if from_batch is not None:
            return from_batch
        if result is not None and result is not batch:
            from_result = self._chem_features_on(result)
            if from_result is not None:
                return from_result
            if isinstance(result, Mapping) and any(
                k in result
                for k in ("atoms", "bonds", "angles", "propers", "impropers", "atom", "bond")
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

    def _propers_to_opls(
        self,
        propers: dict[str, torch.Tensor],
        *,
        atol: float,
    ) -> dict[str, torch.Tensor]:
        """Map multi-term cosine tables → OPLS ``k1..k4`` (LAMMPS Ki).

        Source form (Class-I)::

            E = Σ_m (k_m / s) [1 + cos(n_m φ − γ_m)]

        OPLS (LAMMPS)::

            E = ½ K1(1+cos φ) + ½ K2(1−cos 2φ) + ½ K3(1+cos 3φ) + ½ K4(1−cos 4φ)

        which is the same multi-term form with n=1..4, γ=(0,π,0,π), k_m = K_m/2.

        Terms with n∉{1,2,3,4} or phases not representable as 0/π must be ~0
        (within ``atol``) or projection fails.
        """
        if "k1" in propers:
            return {
                "k1": propers["k1"],
                "k2": propers["k2"],
                "k3": propers["k3"],
                "k4": propers["k4"],
                **({"idivf": propers["idivf"]} if "idivf" in propers else {}),
            }

        propers = ProperTorsionBag._cosine_fields(propers)
        k = propers["k"]
        if k.ndim != 2:
            raise ValueError(f"expected proper k [n, n_terms], got {tuple(k.shape)}")
        phase = propers.get("phase")
        periodicity = propers.get("periodicity")
        if phase is None or periodicity is None:
            raise ValueError("project to OPLS requires propers k, phase, periodicity")

        n_int, n_terms = k.shape
        if periodicity.shape[0] != n_terms:
            raise ValueError("periodicity length must equal n_terms")
        if phase.shape != k.shape:
            raise ValueError("phase must match k shape")

        # Accumulate into OPLS K1..K4 (LAMMPS), starting from multi-term k.
        K = torch.zeros(n_int, 4, device=k.device, dtype=k.dtype)
        periods = periodicity.detach().long().tolist()
        for m, n_per in enumerate(periods):
            if n_per not in (1, 2, 3, 4):
                if not torch.all(torch.abs(k[:, m]) <= atol):
                    raise ValueError(
                        f"project to OPLS: non-zero Fourier term with n={n_per} "
                        f"(max |k|={float(k[:, m].abs().max())}); cannot represent in OPLS k1..k4"
                    )
                continue
            col = n_per - 1
            # Fold phase into signed amplitude: γ≈0 → +k, γ≈π → +k for even-n OPLS
            # convention (even terms already use γ=π in evaluation).
            # General: k_eff = k * cos(γ) for γ near 0/π only; sin component must vanish.
            g = phase[:, m]
            cos_g = torch.cos(g)
            sin_g = torch.sin(g)
            if not torch.all(torch.abs(sin_g * k[:, m]) <= atol):
                raise ValueError(
                    f"project to OPLS: phase has sin component for n={n_per} "
                    "(only γ ≈ 0 or π supported)"
                )
            # For odd n: OPLS uses (1+cos) → γ=0, k_term = K/2 → K = 2 k cosγ
            # For even n: OPLS uses (1−cos)= (1+cos(·−π)) → γ=π, k_term = K/2
            # If source γ≈0 on even n, sign flips relative to OPLS: K = −2k
            if n_per in (1, 3):
                K[:, col] = K[:, col] + 2.0 * k[:, m] * cos_g
            else:
                # prefer γ≈π (cosγ≈−1): k_term positive with γ=π equals K/2
                # if cosγ≈−1: K = 2k; if cosγ≈+1: K = −2k
                K[:, col] = K[:, col] + 2.0 * k[:, m] * (-cos_g)

        out: dict[str, torch.Tensor] = {
            "k1": K[:, 0],
            "k2": K[:, 1],
            "k3": K[:, 2],
            "k4": K[:, 3],
        }
        if "idivf" in propers:
            out["idivf"] = propers["idivf"]
        return out

    def _opls_to_periodic(self, propers: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """OPLS k1..k4 → multi-term k/phase/periodicity (inverse of OPLS eval)."""
        k1, k2, k3, k4 = propers["k1"], propers["k2"], propers["k3"], propers["k4"]
        n = k1.shape[0]
        device, dtype = k1.device, k1.dtype
        k = 0.5 * torch.stack([k1, k2, k3, k4], dim=-1)
        phase = torch.zeros(n, 4, device=device, dtype=dtype)
        phase[:, 1] = math.pi
        phase[:, 3] = math.pi
        periodicity = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long)
        out: dict[str, torch.Tensor] = {
            "k": k,
            "phase": phase,
            "periodicity": periodicity,
            "idivf": propers.get("idivf", torch.ones(n, device=device, dtype=dtype)),
        }
        return out
