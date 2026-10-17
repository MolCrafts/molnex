"""Rondo: a graph network that predicts a Class-I force field, priced through molpot potentials.

Spec: ``src/molzoo/specs/rondo.md``.

``Rondo(forcefield, spec, wrt=("atoms.pos",), dtype=torch.float64)`` is shaped like every other
molzoo model (:class:`molix.autograd.Potential`): ``forward`` is the
:class:`~molix.autograd.Derivative` of :meth:`Rondo.energy`, writing
``graphs.energy`` and, for ``atoms.pos``, the force at ``atoms.forces_pred`` —
a training batch holds the DFT force at ``atoms.forces``. Where the energy
must sit beside its own label, the training step wraps the call in
:func:`molix.frame.beside_labels` (``graphs.energy → graphs.energy_pred``), as
:class:`molix.core.steps.JointTrainStep` does; the model itself moves nothing.
Three primitives sit underneath it —

* :meth:`tables` — the encoder and one head per parameter table turn the
  covalent graph into this batch's parameters (``bonds.k`` …), never reading a
  coordinate, so every parameter is geometry-independent and the force is
  exactly ``−dE/dpos``;
* :meth:`energy` — the force field's styles, as the torch potentials
  :meth:`molpot.potentials.collection.Potentials.from_forcefield` built from
  the IR, priced on those tables (or on tables the caller hands in — a training
  step detaches the ones its loss does not train), every style or the ones the
  caller names (ByteFF's ``calc_terms``: the bonded strain at a QM minimum, the
  non-bonded force alone);
* :meth:`partial_hessian` — the MM ``3×3`` blocks on the pairs a Hessian
  objective names.

The force field **is** the Hamiltonian: which terms exist, their cutoff, mixing
rule, Coulomb constant and 1-4 scales are the molpy ``ForceField``'s, declared
once. Rondo adds the heads that supply the parameters the IR leaves open and
nothing else. Conventions are molrs's: LAMMPS ``K`` (no ½), angles in degrees.

References:
    Wang, Y. et al. "End-to-end differentiable construction of molecular
    mechanics force fields." (Espaloma) https://arxiv.org/abs/2010.01196
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." (ByteFF)
    https://arxiv.org/abs/2408.12817
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from molpy.ff.forcefield import ForceField
from tensordict import TensorDict

from molix.autograd import Derivative, Potential
from molix.schema import ENERGY_KEY, FORCES_PRED_KEY, POS_KEY
from molpot.charge import (
    AtomicChargeModel,
    ChiEtaHead,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
)
from molpot.forcefield.partial_hessian import TermHessian
from molpot.heads.classical import (
    AngleHead,
    BondHead,
    CovalentRadiusSum,
    CvffHead,
    LJHead,
    PeriodicHead,
)
from molpot.potentials.collection import Potentials
from molpot.potentials.potential import BondedPotential
from molrep.chem.encoder import ChemEncoder
from molrep.chem.hybridization import Hybridization

from .spec import RondoSpec

__all__ = ["Rondo"]

Tables = dict[str, dict[str, torch.Tensor]]


class Rondo(Potential):
    """Predict every open Class-I parameter from the covalent graph and evaluate the energy.

    Args:
        forcefield: The molpy ``ForceField`` whose styles are priced. Every
            style must have a registered molpot potential **and** a head here
            that supplies its parameters: ``bond harmonic``, ``angle
            harmonic``, ``dihedral periodic`` or ``opls``, ``improper cvff``,
            ``pair lj/cut``, ``pair coul/cut``. A style outside this set is
            refused at construction, by name.
        spec: Encoder and head settings (:class:`~molzoo.rondo.spec.RondoSpec`).
            Defaults to the spec's defaults.
        wrt: Variables :meth:`forward` differentiates against, as for every
            molzoo potential (``("atoms.pos",)`` writes ``atoms.forces_pred``;
            ``()`` makes ``forward`` the energy alone).
        dtype: Floating dtype every parameter and buffer is built at.

    Raises:
        ValueError: A declared style has no potential or no head, or a
            ``dihedral opls`` is declared with a proper periodicity other than
            ``(1, 2, 3, 4)``.

    Example::

        ff = mp.ff.ForceField(name="class-i", units="real")
        ff.def_style("bond", "harmonic")
        ...
        model = Rondo(ff, RondoSpec(), wrt=("atoms.pos",), dtype=torch.float64)
        out = model(batch)      # graphs.energy, atoms.forces_pred, atoms.charge, bonds.k …
    """

    #: Styles Rondo can supply parameters for, with the head each one needs.
    SUPPORTED_STYLES: tuple[str, ...] = (
        "bond:harmonic",
        "angle:harmonic",
        "dihedral:periodic",
        "dihedral:opls",
        "improper:cvff",
        "pair:lj/cut",
        "pair:coul/cut",
    )

    def __init__(
        self,
        forcefield: ForceField,
        spec: RondoSpec | None = None,
        *,
        wrt: Sequence[str] = ("atoms.pos",),
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        #: The build dtype of every float parameter and buffer.
        self.dtype = dtype
        spec = RondoSpec() if spec is None else spec
        self.spec = spec
        # Plain attribute: the IR is data, not a module.
        self.forcefield = forcefield
        width = spec.encoder.width
        hidden = spec.heads.hidden

        self.encoder = ChemEncoder(
            atom_dim=width,
            bond_dim=width,
            angle_dim=width,
            proper_dim=width,
            improper_dim=width,
            hidden_dim=width,
            num_elements=spec.encoder.num_elements,
            atom_channels=spec.encoder.atom_channels,
            bond_channels=spec.encoder.bond_channels,
            n_blocks=spec.encoder.n_blocks,
            local_attention=spec.encoder.local_attention,
            dtype=dtype,
        )
        self.hybridization = Hybridization()
        self.covalent_radius = CovalentRadiusSum(spec.encoder.num_elements, dtype=dtype)
        self.potentials = Potentials.from_forcefield(forcefield)

        unknown = [key for key in self.potentials.keys() if key not in self.SUPPORTED_STYLES]
        if unknown:
            raise ValueError(
                f"Rondo has no head for the styles {unknown}; it supplies parameters for "
                f"{self.SUPPORTED_STYLES}"
            )
        self.heads = nn.ModuleDict()
        for key in self.potentials.keys():
            if key == "bond:harmonic":
                self.heads["bonds"] = BondHead(
                    width,
                    hidden,
                    k_ref=spec.heads.bond_k_ref,
                    r0_gain=spec.heads.bond_r0_gain,
                    dtype=dtype,
                )
            elif key == "angle:harmonic":
                self.heads["angles"] = AngleHead(
                    width,
                    hidden,
                    k_ref=spec.heads.angle_k_ref,
                    theta0_gain_deg=spec.heads.angle_theta0_gain_deg,
                    dtype=dtype,
                )
            elif key == "dihedral:periodic":
                self.heads["propers"] = PeriodicHead(
                    width, hidden, periodicity=spec.heads.proper_periodicity, dtype=dtype
                )
            elif key == "dihedral:opls":
                if tuple(spec.heads.proper_periodicity) != (1, 2, 3, 4):
                    raise ValueError(
                        "dihedral opls is the four-term series n = 1..4; set "
                        f"proper_periodicity=(1, 2, 3, 4), got {spec.heads.proper_periodicity}"
                    )
                self.heads["propers"] = PeriodicHead(
                    width, hidden, periodicity=(1, 2, 3, 4), dtype=dtype
                )
            elif key == "improper:cvff":
                self.heads["impropers"] = CvffHead(
                    width,
                    hidden,
                    k_ref=spec.heads.improper_k_ref,
                    periodicity=spec.heads.improper_periodicity,
                    dtype=dtype,
                )
            elif key == "pair:lj/cut":
                self.heads["lj"] = LJHead(
                    width,
                    hidden,
                    sigma_gain=spec.heads.lj_sigma_gain,
                    epsilon_gain=spec.heads.lj_epsilon_gain,
                    dtype=dtype,
                )
            elif key == "pair:coul/cut":
                self.heads["charge"] = AtomicChargeModel(
                    ChiEtaHead(feature_dim=width, hidden_dim=hidden, dtype=dtype),
                    ElectronegativityEqualization(
                        DiagonalHardnessKernel(),
                        max_atoms=spec.heads.eem_max_atoms,
                        jitter=spec.heads.eem_jitter,
                    ),
                )
        # One bonded potential per namespace: ``tables`` reads their indices.
        self._bonded_keys = {
            potential.namespace: key
            for key, potential in self.potentials.items()
            if isinstance(potential, BondedPotential)
        }
        # The op ``forward`` runs, built once like every molzoo potential's.
        # A bound method, so no submodule is registered. The force lands beside
        # the label (``atoms.forces_pred``); the energy is the op's ``of``.
        keys = tuple(wrt)
        outputs = tuple(FORCES_PRED_KEY if key == "atoms.pos" else None for key in keys)
        self._derivative = Derivative(self.energy, wrt=keys, outputs=outputs)

    # ------------------------------------------------------------------
    # tables
    # ------------------------------------------------------------------

    def tables(self, frame: TensorDict) -> Tables:
        """This batch's predicted parameters, one row per term or atom.

        Args:
            frame: Batch with ``atoms.Z``, the encoder's declared channel
                columns, the covalent namespaces (``bonds.bond_type`` when
                angles or propers are priced), ``atoms.gaff2_sigma`` /
                ``atoms.gaff2_epsilon`` when Lennard-Jones is priced, and
                ``graphs.total_charge`` ``(B,)`` when charges are.

        Returns:
            Namespace → name → tensor. Parameters come with their references
            (``k_ref``, ``r0_ref``, ``theta0_ref``, ``sigma_ref``,
            ``epsilon_ref``) so a prior can read both from one table.
        """
        emb = self.encoder.compose(frame)
        atoms = frame["atoms"]
        tables: Tables = {}
        if "bonds" in self.heads:
            index = self._bonded("bonds").index(frame)
            r0_ref = self.covalent_radius(atoms["Z"][index[0]], atoms["Z"][index[1]])
            tables["bonds"] = dict(self.heads["bonds"](emb.bond, r0_ref=r0_ref.to(emb.bond.dtype)))
        if "angles" in self.heads or "propers" in self.heads:
            degree, pi = self._hybridization(frame)
        if "angles" in self.heads:
            centre = self._bonded("angles").index(frame)[1]
            theta0_ref = torch.rad2deg(self.hybridization.angle_reference(degree, pi))[centre]
            tables["angles"] = dict(
                self.heads["angles"](emb.angle, theta0_ref=theta0_ref.to(emb.angle.dtype))
            )
        if "propers" in self.heads:
            table = dict(self.heads["propers"](emb.proper))
            # A dihedral about an sp centre is undefined (its cross products
            # vanish); ByteFF zeroes those terms by a SMARTS match, here the
            # centre's hybridization comes from the bond orders.
            index = self._bonded("propers").index(frame)
            linear = self.hybridization.linear(degree, pi)
            bent = (~(linear[index[1]] | linear[index[2]])).unsqueeze(-1)
            table["k"] = table["k"] * bent.to(table["k"].dtype)
            tables["propers"] = table
        if "impropers" in self.heads:
            tables["impropers"] = dict(self.heads["impropers"](emb.improper))
        per_atom: dict[str, torch.Tensor] = {}
        if "lj" in self.heads:
            for column in ("gaff2_sigma", "gaff2_epsilon"):
                if column not in atoms.keys():
                    raise KeyError(
                        f"Rondo prices pair lj/cut around atoms.{column}; "
                        "the batch has no such column"
                    )
            per_atom.update(
                self.heads["lj"](
                    emb.atom,
                    sigma_ref=atoms["gaff2_sigma"].to(emb.atom.dtype),
                    epsilon_ref=atoms["gaff2_epsilon"].to(emb.atom.dtype),
                )
            )
        if "charge" in self.heads:
            per_atom["charge"] = self.heads["charge"].charges(
                emb.atom, frame, total_charge=frame["graphs", "total_charge"]
            )
        if per_atom:
            tables["atoms"] = per_atom
        return tables

    # ------------------------------------------------------------------
    # energy
    # ------------------------------------------------------------------

    def energy(
        self,
        frame: TensorDict,
        tables: Tables | None = None,
        *,
        styles: Sequence[str] | None = None,
    ) -> TensorDict:
        """Write ``graphs.energy`` ``(B,)`` kcal/mol — the traced seam.

        The heads run inside the energy (``tables=None``) so that everything
        below the caller's compiled region is one graph; a caller that has
        already predicted the tables — a training step that detaches the ones
        its loss does not train — hands them in. The predicted tables are
        written onto their namespaces as columns (``bonds.k``, ``atoms.charge``,
        …) so losses and priors read them from the batch.

        Args:
            frame: Padded batch with ``atoms.pos``, ``atoms.batch``, ``graphs``
                and the namespaces the force field's styles read.
            tables: Parameters from :meth:`tables`, or ``None`` to predict them
                here.
            styles: The styles summed, as the force field names them
                (``"bond:harmonic"``, ``"pair:lj/cut"`` …); ``None`` (default)
                prices every declared style. ByteFF's ``calc_terms``: the bond
                and angle energy on a QM minimum is the strain ``r0`` / ``θ0``
                remove, the pair styles' force alone tells a clashing geometry.
                A fixed tuple of names, so the sum traces as straight-line code.

        Returns:
            The same frame with ``graphs.energy`` and the table columns written.

        Raises:
            ValueError: *styles* is empty or names a style the force field does
                not declare.
        """
        if tables is None:
            tables = self.tables(frame)
        keys = tuple(self.potentials.keys()) if styles is None else tuple(styles)
        unknown = [key for key in keys if key not in self.potentials]
        if not keys or unknown:
            raise ValueError(
                f"energy(styles=...) names {unknown or 'no style'}; the force field declares "
                f"{tuple(self.potentials.keys())}"
            )
        total: torch.Tensor | None = None
        for key in keys:
            term = self.potentials[key](frame, **self._style_params(key, tables))
            total = term if total is None else total + term
        frame[ENERGY_KEY] = total
        for namespace, table in tables.items():
            if namespace in frame.keys():
                for name, value in table.items():
                    frame[namespace, name] = value
        return frame

    def partial_hessian(self, frame: TensorDict, tables: Tables | None = None) -> torch.Tensor:
        """The MM ``3×3`` blocks on the pairs ``hessian_pairs`` names ``(P, 9)``.

        A pair's block is the sum over every bonded term containing it, so each
        term's own blocks (:class:`~molpot.forcefield.partial_hessian.TermHessian`
        on the gathered coordinates) are accumulated into the pair row its
        ``hpair_<i>_<j>`` column names.

        Args:
            frame: Batch with ``hessian_pairs`` and the ``hpair_*`` slot columns;
                ``atoms.pos`` must require grad (marked before the traced body).
            tables: Predicted tables, when the caller already has them.

        Returns:
            ``(P, 9)`` kcal mol⁻¹ Å⁻², row-major blocks in ``hessian_pairs`` order.

        Raises:
            ValueError: A priced valence namespace has rows but no ``hpair_*``
                column — its term would silently receive no gradient.
        """
        if tables is None:
            tables = self.tables(frame)
        pos = frame[POS_KEY]
        blocks = pos.new_zeros((frame["hessian_pairs"].batch_size[0], 9))
        hessian = TermHessian()
        for key, potential in self.potentials.items():
            if not isinstance(potential, BondedPotential):
                continue
            namespace = potential.namespace
            if namespace not in frame.keys():
                continue
            block = frame[namespace]
            index = potential.index(frame)
            choices = sorted(name for name in block.keys() if name.startswith("hpair_"))
            if not choices:
                if int(index.shape[1]):
                    raise ValueError(
                        f"{namespace} has {int(index.shape[1])} rows and no hpair_<i>_<j> column; "
                        f"{key} would contribute nothing to the partial Hessian"
                    )
                continue
            params = {
                name: value.to(pos.dtype) for name, value in self._style_params(key, tables).items()
            }
            mask = self._mask(block, pos)

            def term_energy(x: torch.Tensor, _p=potential, _params=params, _m=mask) -> torch.Tensor:
                return (_p.term_energy(x, **_params) * _m).sum()

            with torch.enable_grad():
                per_term = hessian.blocks(term_energy, pos[index.T])
            n_terms = per_term.shape[0]
            for name in choices:
                i, j = (int(part) for part in name.removeprefix("hpair_").split("_"))
                blocks = blocks.index_add(
                    0, block[name].long(), per_term[:, i, :, j, :].reshape(n_terms, 9)
                )
        return blocks

    # ------------------------------------------------------------------
    # private
    # ------------------------------------------------------------------

    def _bonded(self, namespace: str) -> BondedPotential:
        """The bonded potential priced on *namespace*; the owner of its term index."""
        return self.potentials[self._bonded_keys[namespace]]

    def _style_params(self, key: str, tables: Tables) -> dict[str, torch.Tensor]:
        """The parameters one style's potential reads, from the predicted tables."""
        if key == "bond:harmonic":
            return {"k": tables["bonds"]["k"], "r0": tables["bonds"]["r0"]}
        if key == "angle:harmonic":
            return {"k": tables["angles"]["k"], "theta0": tables["angles"]["theta0"]}
        if key in ("dihedral:periodic", "dihedral:opls"):
            head = self.heads["propers"]
            k = tables["propers"]["k"]
            if key == "dihedral:opls":
                return {f"k{m + 1}": 2.0 * k[:, m] for m in range(4)}
            rows = k.shape[0]
            return {
                "k": k,
                "periodicity": head.periodicity.to(k.dtype).expand(rows, -1),
                "phase": head.phase.to(k.dtype).expand(rows, -1),
            }
        if key == "improper:cvff":
            k = tables["impropers"]["k"]
            head = self.heads["impropers"]
            return {
                "k": k,
                "sign": head.sign.to(k.dtype).expand(k.shape),
                "periodicity": head.periodicity.to(k.dtype).expand(k.shape),
            }
        if key == "pair:lj/cut":
            return {"epsilon": tables["atoms"]["epsilon"], "sigma": tables["atoms"]["sigma"]}
        if key == "pair:coul/cut":
            return {"charge": tables["atoms"]["charge"]}
        raise ValueError(f"no parameter mapping for style {key!r}")  # pragma: no cover

    def _hybridization(self, frame: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Degree and π count per atom from ``bonds.bond_type``; a batch without it is refused."""
        n_atoms = int(frame["atoms"].batch_size[0])
        if "bonds" not in frame.keys() or "bond_type" not in frame["bonds"].keys():
            raise KeyError(
                "Rondo reads bonds.bond_type to tell sp, sp2 and sp3 centres apart; "
                "the batch has no such column"
            )
        index = self._bonded("bonds").index(frame)
        bonds = frame["bonds"]
        mask = bonds["mask"] if "mask" in bonds.keys() else None
        return self.hybridization(index, bonds["bond_type"], mask, n_atoms)

    @staticmethod
    def _mask(block: TensorDict, like: torch.Tensor) -> torch.Tensor:
        """The namespace's live-row mask as a float, or ``1`` for an unpadded batch."""
        if "mask" in block.keys():
            return block["mask"].to(like.dtype)
        return like.new_ones(())
