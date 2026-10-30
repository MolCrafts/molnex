"""ByteFF: the covalent-graph encoder, the bonded heads and the bond-charge / direct-LJ pair heads.

ByteFF is Rondo's architecture (:class:`molzoo.rondo.model.Rondo`) with the
published pair heads: the charge of ``pair coul/cut`` comes from bond charge
corrections (:class:`~molpot.classical.BondChargeCorrection`) and σ, ε of
``pair lj/cut`` are predicted directly (:class:`~molpot.classical.LJCutDirect`).
:meth:`ByteFF.load` reads one ByteFF-26 member into those modules. With a force
field that declares ``pair lj/cut`` (arithmetic mixing, special ``0 0 0.5``)
and ``pair coul/cut`` (``coulomb 332.0637141491396``, special ``0 0
0.8333333333``) and a cutoff beyond every intramolecular distance, this is the
published model's functional form; a loaded member then predicts what
``byteff2@v1.1.0`` predicts. The five-member mean is the caller's: average
:meth:`~molzoo.rondo.model.Rondo.predict` over the members, then
:meth:`~molzoo.rondo.model.Rondo.price` once.

Spec: ``.claude/specs/rondo-enc-01-graph-encoder.md``.

Reference:
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." Chem. Sci. 2025.
    https://arxiv.org/abs/2408.12817
"""

from __future__ import annotations

from pathlib import Path

import torch
from molpy.ff.forcefield import ForceField

from molpot.classical import BondChargeCorrection, LJCutDirect, Term
from molzoo.byteff.checkpoint import ByteFFKeys
from molzoo.checkpoint import CheckpointRemap, OfficialArchive
from molzoo.rondo.model import Rondo

__all__ = ["ByteFF"]

#: The released window's lowest bond ``K`` (LAMMPS; the head's ``½k`` clamp of 80).
PUBLISHED_BOND_K_FLOOR = 40.0


class ByteFF(Rondo):
    """Rondo with bond-charge-correction charges and directly predicted σ, ε.

    The pair styles get :class:`~molpot.classical.BondChargeCorrection` (charge
    moved along each bond, antisymmetric in its direction, on top of the
    equivalence-averaged formal charges) and :class:`~molpot.classical.LJCutDirect`
    (σ and ε in a ``tanh`` window, no reference) instead of Rondo's EEM (D1)
    and GAFF2-anchored window (D2). Everything else is Rondo's. With a force
    field that declares ``pair lj/cut`` (arithmetic mixing, special ``0 0
    0.5``) and ``pair coul/cut`` (``coulomb 332.0637141491396``, special ``0 0
    0.8333333333``) and a cutoff beyond every intramolecular distance, this is
    the published model's functional form.

    Reference:
        Zheng, T. et al. "Data-driven parametrization of molecular mechanics
        force fields for expansive chemical space coverage." Chem. Sci. 2025.
        https://arxiv.org/abs/2408.12817
    """

    def _new(self, style: str) -> Term:
        heads = self.spec.heads
        if style == "pair:lj/cut":
            return LJCutDirect(
                self.graph.width,
                hidden_dim=heads.lj_direct_hidden,
                layers=heads.lj_direct_layers,
                tanh_output=heads.pair_tanh_output,
                dtype=self.dtype,
            )
        if style == "pair:coul/cut":
            return BondChargeCorrection(
                self.graph.width,
                self.graph.bond_width,
                hidden_dim=heads.bcc_hidden,
                width=heads.bcc_width,
                layers=heads.bcc_layers,
                out_hidden_dim=heads.bcc_out_hidden,
                out_layers=heads.bcc_out_layers,
                tanh_output=heads.pair_tanh_output,
                dtype=self.dtype,
            )
        return super()._new(style)

    @classmethod
    def load(
        cls, path: str | Path, forcefield: ForceField, *, dtype: torch.dtype, **settings: object
    ) -> ByteFF:
        """A ByteFF whose every head is one ByteFF-26 member.

        Reads ``ByteFF-26/models/member-<k>/optimal.pt`` (HF
        ``ByteDance-Seed/byteff2``, revision ``v1.1.0``) through
        :class:`~molzoo.checkpoint.CheckpointRemap` over
        :class:`~molzoo.byteff.checkpoint.ByteFFKeys`. The settings must describe
        ByteFF-26's sizes, which are the defaults.

        Args:
            path: A member's ``optimal.pt``.
            forcefield: The force field; every style it declares must be one
                the checkpoint has a head for.
            dtype: Floating dtype of the model.
            **settings: As for the constructor. ``bond_k_floor`` defaults to the
                published window's 40, not Rondo's 80, so the member is
                reproduced as released.

        Returns:
            The model, in training mode like any freshly built module.

        Raises:
            ValueError: A tensor of the checkpoint has no home in the model, or a
                tensor of the model is not in the checkpoint.
        """
        model = cls(forcefield, dtype=dtype, **{"bond_k_floor": PUBLISHED_BOND_K_FLOOR, **settings})
        archive = torch.load(Path(path), map_location="cpu", weights_only=True)
        state = OfficialArchive.tensor_state(archive["model_state_dict"])
        heads = model.spec.heads
        table = ByteFFKeys(
            n_layers=model.spec.encoder.n_layers,
            head_layers=heads.bonded_layers,
            out_layers=heads.bonded_out_layers,
            pair_layers={
                "lj_direct_layers": heads.lj_direct_layers,
                "bcc_layers": heads.bcc_layers,
                "bcc_out_layers": heads.bcc_out_layers,
            },
            slots=tuple(model.heads),
        ).table()
        own = model.state_dict()
        renamed = CheckpointRemap(table).rename(state)
        unexpected = sorted(set(renamed) - set(own))
        absent = sorted(set(own) - set(renamed))
        if unexpected or absent:
            raise ValueError(
                f"ByteFF-26 tensors with no home: {unexpected}; "
                f"tensors it does not provide: {absent}"
            )
        model.load_state_dict({key: value.to(own[key].dtype) for key, value in renamed.items()})
        return model
