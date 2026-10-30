"""Rondo: a graph encoder and one parameter head per style, priced through molpot potentials.

Spec: ``src/molzoo/specs/rondo.md``; the encoder and the bonded heads are
``.claude/specs/rondo-enc-01-graph-encoder.md``.

``Rondo(forcefield, *, dtype=torch.float64)`` is shaped like every other
molzoo model (:class:`molix.autograd.Potential`): ``forward`` is the
:class:`~molix.autograd.Derivative` of :meth:`Rondo.energy` with respect to
``atoms.pos``, writing ``graphs.energy`` and the force at ``atoms.forces_pred``.
A training batch holds the DFT force at ``atoms.forces``. Where the energy must
sit beside its own label, the training step wraps the call in
:func:`molix.frame.beside_labels` (``graphs.energy → graphs.energy_pred``), as
:class:`molix.core.steps.JointTrainStep` does; the model itself moves nothing.

Rondo owns the two interfaces and nothing between them:

* encoder → heads: :attr:`graph`
  (:class:`~molrep.encoders.covalent_graph.encoder.CovalentGraphEncoder`) returns
  :class:`~molrep.encoders.features.EncoderFeatures`, one row per atom and per bond.
  :meth:`features` is that call and evaluates no head.
* heads → potentials: one :class:`~molpot.classical.Term` per style the
  :class:`~molpot.potentials.collection.Potentials` prices. The bonded heads
  assemble angle, torsion and improper features from the atom and bond rows
  themselves; the pair heads are EEM charges (D1) and LJ around GAFF2 (D2)
  (:class:`molzoo.byteff.potential.ByteFF` overrides them). Their columns are
  in molrs's conventions (LAMMPS ``K`` without ½, degrees).

:meth:`predict` (parameters only) and :meth:`price` (energy at given
parameters) are the two primitives :meth:`energy` composes.

The force field **is** the catalogue of styles: cutoff, mixing, Coulomb
constant and 1-4 scales are the molpy ``ForceField``'s. A declared style with
no registered potential, and a priced style with no parameter head, are refused
at construction.

References:
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." Chem. Sci. 2025.
    https://arxiv.org/abs/2408.12817
    Wang, Y. et al. "End-to-end differentiable construction of molecular
    mechanics force fields." (Espaloma) https://arxiv.org/abs/2010.01196
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np
import torch
from molpy import Block, Element, Frame
from molpy.ff.forcefield import ForceField
from tensordict import TensorDict

from molix.autograd import Derivative, Potential
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, FORCES_PRED_KEY, POS_KEY
from molpot.classical import TERMS, Term
from molpot.forcefield.partial_hessian import TermHessian
from molpot.potentials.collection import Potentials
from molpot.potentials.potential import BondedPotential
from molrep.encoders.covalent_graph import CovalentGraphEncoder, CovalentGraphEncoderSpec
from molrep.encoders.features import EncoderFeatures
from molrep.encoders.mace_conformer import MaceConformerEncoder
from molrep.encoders.message_passing import MessagePassingEncoder

from .spec import validate

__all__ = ["Rondo"]

Tables = dict[str, dict[str, torch.Tensor]]

#: molpot namespace of a bonded term -> the molpy frame block that lists its rows.
_MOLPY_BLOCKS = {
    "bonds": "bonds",
    "angles": "angles",
    "propers": "dihedrals",
    "impropers": "impropers",
}
_ENDS = ("atomi", "atomj", "atomk", "atoml")


def _term_style(kind: type[Term]) -> str:
    """The force-field key of a term class."""
    if isinstance(kind, type) and issubclass(kind, Term):
        return kind.style
    raise TypeError(f"a term is a Term class, got {kind!r}")


def _row(params: dict[str, torch.Tensor], row: int) -> dict[str, float]:
    """One row of *params* as molrs type parameters; a vector column becomes ``name1 … namem``."""
    values: dict[str, float] = {}
    for name, column in params.items():
        entry = column[row]
        if entry.ndim == 0:
            values[name] = float(entry)
        else:
            values.update({f"{name}{m + 1}": float(x) for m, x in enumerate(entry)})
    return values


def _style_of(kind: str | type[Term]) -> str:
    """A term class, or a style key a training objective already holds."""
    if isinstance(kind, str):
        return kind
    return _term_style(kind)


class Rondo(Potential):
    """Predict the open parameters of the force field's styles and evaluate their energy.

    Args:
        forcefield: The molpy ``ForceField``. Its energy is one
            :class:`~molpot.potentials.collection.Potentials`
            (:meth:`~molpot.potentials.collection.Potentials.from_forcefield`).
            Rondo adds one parameter head per style that object prices. A
            later style is a force field that declares it; a subclass
            overrides :meth:`load_state_dict` to accept this checkpoint.
        dtype: Floating dtype every parameter and buffer is built at.
        encoder: The encoder the heads read, built by the caller at *dtype*
            (rondo-enc phase 4: a
            :class:`~molrep.encoders.mace_conformer.MaceConformerEncoder` under
            ByteFF-26's heads). ``None`` builds the covalent-graph encoder from
            *settings*; an injected encoder takes no covalent-graph setting.
            It is registered as :attr:`graph` either way, so the heads load by
            the same keys.
        **settings: Fields of
            :class:`~molrep.encoders.covalent_graph.spec.CovalentGraphEncoderSpec`
            and :class:`~molzoo.rondo.spec.HeadsSpec`. Omitted fields take the
            spec's defaults.

    Raises:
        ValueError: The field declares no priced style, a declared style has no
            registered potential, or a priced style has no parameter head.
        TypeError: A covalent-graph setting beside an injected *encoder*.

    Example::

        from molzoo.rondo import Rondo

        model = Rondo(ff, dtype=torch.float64)
        model.load_state_dict(torch.load("rondo.pt"))
        features = model.features(batch)          # EncoderFeatures, no head
        out = model(batch)                        # graphs.energy, atoms.forces_pred, bonds.k …
    """

    SUPPORTED_STYLES: tuple[str, ...] = tuple(TERMS)

    def __init__(
        self,
        forcefield: ForceField,
        *,
        dtype: torch.dtype,
        encoder: CovalentGraphEncoder | MaceConformerEncoder | MessagePassingEncoder | None = None,
        **settings: object,
    ) -> None:
        super().__init__()
        #: The build dtype of every float parameter and buffer.
        self.dtype = dtype
        if encoder is not None:
            covalent = sorted(set(settings) & set(CovalentGraphEncoderSpec.model_fields))
            if covalent:
                raise TypeError(
                    f"covalent-graph settings {covalent} beside an injected "
                    f"{type(encoder).__name__}, which they would not configure"
                )
        spec = validate(**settings)
        #: Rondo's spec. ``spec.encoder`` describes the covalent-graph encoder
        #: only when no encoder was injected; an injected one carries its own.
        self.spec = spec
        # Plain attribute: the IR is data, not a module.
        self.forcefield = forcefield.canonical()
        self.graph: CovalentGraphEncoder | MaceConformerEncoder | MessagePassingEncoder = (
            CovalentGraphEncoder(spec.encoder, dtype=dtype) if encoder is None else encoder
        )
        self.heads = torch.nn.ModuleDict()
        self.potentials = Potentials.from_forcefield(self.forcefield)
        if len(self.potentials) == 0:
            raise ValueError(
                "the force field declares no style that prices an energy; "
                "Rondo is built from those styles"
            )
        missing = [style for style in self.potentials if style not in TERMS]
        if missing:
            raise ValueError(
                f"Rondo has no parameters for the styles {missing}; it supplies parameters for "
                f"{self.SUPPORTED_STYLES}"
            )
        self._terms: dict[str, Term] = {}
        for style in self.potentials:
            self._add(style)
        # The op ``forward`` runs, built once. A bound method, so no submodule
        # is registered. The force is −∂E/∂pos and lands beside the label
        # (``atoms.forces_pred``); the energy is the op's ``of``.
        self._derivative = Derivative(self.energy, wrt=("atoms.pos",), outputs=(FORCES_PRED_KEY,))

    @property
    def terms(self) -> Mapping[type[Term], Term]:
        """The terms this model carries, keyed by their class, in price order."""
        return MappingProxyType({type(term): term for term in self._terms.values()})

    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        strict: bool = True,
        assign: bool = False,
        prefixes: Sequence[str] | None = None,
    ):
        """Load parameters. A subclass on a richer force field overrides this.

        The base is strict: every key of this module is in *state_dict*, and
        *state_dict* has no key this module lacks. The subclass is constructed
        on the force field that declares the new styles and overrides this
        method to accept the parent's checkpoint (the new heads stay at their
        initial values)::

            def load_state_dict(self, state_dict, strict=True, assign=False):
                return super().load_state_dict(state_dict, strict=False, assign=assign)

        Args:
            state_dict: A ``state_dict`` from this class or, in an override,
                from the parent.
            strict: Require the keys to match exactly.
            assign: Copy by assignment rather than into the existing storage.
            prefixes: Copy only these prefixes (``"graph."``, ``"heads."``).
                Every other tensor is left in its existing storage.

        Returns:
            Missing and unexpected keys, as
            :meth:`torch.nn.Module.load_state_dict` does.
        """
        if prefixes is not None:
            return self._load_prefixes(state_dict, prefixes)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _load_prefixes(self, state_dict: Mapping[str, object], prefixes: Sequence[str]):
        """Copy only these prefixes into the existing tensors.

        Every other parameter keeps its storage. A missing key under a named
        prefix raises. A key under a named prefix that this model does not
        have raises. Keys outside the prefixes are ignored.
        """
        allowed = tuple(prefixes)
        current = self.state_dict()
        incoming = {
            str(key): value for key, value in state_dict.items() if str(key).startswith(allowed)
        }
        owned = [key for key in current if key.startswith(allowed)]
        missing = [key for key in owned if key not in incoming]
        unexpected = [key for key in incoming if key not in current]
        if missing or unexpected:
            raise KeyError(
                f"prefix load missing {missing or '[]'}, unexpected {unexpected or '[]'}"
            )
        return super().load_state_dict(incoming, strict=False, assign=False)

    # ------------------------------------------------------------------
    # features / energy
    # ------------------------------------------------------------------

    def features(self, frame: TensorDict) -> EncoderFeatures:
        """The encoder's atom and bond features. No parameter head runs.

        Args:
            frame: Batch with the encoder's columns (``atoms.Z`` /
                ``formal_charge`` / ``ring_connectivity`` / ``min_ring_size`` /
                ``equi_index``; ``bonds.bond_index`` / ``bond_type`` /
                ``in_ring`` / ``equi_index``).

        Returns:
            One row per atom and per bond.
        """
        return self.graph(frame)

    def predict(
        self, frame: TensorDict, *, styles: Sequence[str | type[Term]] | None = None
    ) -> Tables:
        """The parameters of *styles* (every carried style when ``None``), nothing priced.

        Args:
            frame: Batch with the encoder's columns and the namespaces the
                styles read.
            styles: Members of :attr:`potentials`, as in :meth:`energy`.

        Returns:
            ``{namespace: {column: tensor}}`` in molrs's conventions.

        Raises:
            ValueError: *styles* is empty or names a style the model does not
                carry.
        """
        features = self.features(frame)
        predicted: Tables = {}
        for key in self._styles(styles):
            term = self._terms[key]
            columns = term.predict(features, frame, self.potentials[key])
            predicted.setdefault(term.namespace, {}).update(columns)
        return predicted

    def price(
        self,
        frame: TensorDict,
        tables: Tables,
        *,
        styles: Sequence[str | type[Term]] | None = None,
    ) -> torch.Tensor:
        """The energy ``(B,)`` kcal/mol of *styles* at the parameters in *tables*.

        The graph sum of :meth:`atom_energy` — one path for both.

        Args:
            frame: Batch with ``atoms.pos``, ``atoms.batch``, ``graphs`` and the
                namespaces the styles read.
            tables: Parameters as :meth:`predict` returns them (or a mean of
                several models' tables).
            styles: Members of :attr:`potentials`, as in :meth:`energy`.
        """
        return self._graph_sum(frame, self.atom_energy(frame, tables, styles=styles))

    def atom_energy(
        self,
        frame: TensorDict,
        tables: Tables,
        *,
        styles: Sequence[str | type[Term]] | None = None,
    ) -> torch.Tensor:
        """The energy ``(N,)`` kcal/mol per atom: each term split equally over its atoms.

        ``molpot.potentials.Potentials.atom_energy`` (LAMMPS ``compute pe/atom``'s
        partition) at the parameters in *tables*; ghost atoms (``atoms.mask``)
        carry 0.

        Args:
            frame: As :meth:`price`.
            tables: As :meth:`price`.
            styles: As :meth:`price`.
        """
        keys = self._styles(styles)
        params = {key: self._terms[key].potential_params(tables) for key in keys}
        energy = self.potentials.atom_energy(frame, params, styles=keys)
        atoms = frame["atoms"]
        if "mask" in atoms.keys():
            energy = energy * atoms["mask"].to(energy.dtype)
        return energy

    @staticmethod
    def _graph_sum(frame: TensorDict, atom_energy: torch.Tensor) -> torch.Tensor:
        """``(B,)``: *atom_energy* summed by ``atoms.batch`` over the graph slots."""
        graphs = frame["graphs"].batch_size[0]
        return atom_energy.new_zeros(graphs).index_add(
            0, frame["atoms", "batch"].long(), atom_energy
        )

    @staticmethod
    def write_tables(frame: TensorDict, tables: Tables) -> None:
        """Write every predicted column onto the namespace of *frame* that holds it."""
        for namespace, columns in tables.items():
            if namespace in frame.keys():
                for name, value in columns.items():
                    frame[namespace, name] = value

    def parameterize(self, frame: TensorDict, *, graph: int = 0) -> tuple[Frame, ForceField]:
        """One molecule's static force field: its parameters predicted once, as molpy data.

        Every atom and every term row of *graph* becomes its own molpy type at
        the predicted parameters — nothing is averaged — on the styles, cutoffs
        and 1-4 scales of :attr:`forcefield`; charges go on the atoms. molpy
        serialises the pair (for LAMMPS, ``molpy.io.write_lammps_data`` and
        ``molpy.io.lammps.LammpsForcefieldWriter``). Each term's
        :meth:`~molpot.classical.Term.potential_params` already speaks molrs's
        parameter names; a ``(rows, m)`` column (the periodic series) becomes
        molrs's ``name1 … namem``. Runs once per topology, eagerly: not a hot
        path.

        Args:
            frame: Collated batch as :meth:`predict` reads it. Padding rows
                (ghost atoms, dead terms on them) are not written.
            graph: The molecule to write.

        Returns:
            ``(structure, forcefield)``: an ``atoms`` block (positions, charge,
            mass, ``type``) and one block per bonded namespace with ``type``
            columns (``propers`` as molpy's ``dihedrals``); the typed field.
        """
        with torch.no_grad():
            tables = self.predict(frame)
        atoms = frame["atoms"]
        own = torch.nonzero(atoms["batch"] == graph).flatten()
        local = torch.full_like(atoms["batch"], -1)
        local[own] = torch.arange(own.numel(), device=own.device, dtype=local.dtype)
        local = local.cpu()
        field = ForceField(name=f"{self.forcefield.name}-static", units=self.forcefield.units)
        lj, coul = self.forcefield.special_bonds
        field.set_special_bonds(lj=list(lj), coul=list(coul))
        atom_style = field.def_style("atom", "full")
        mass = [Element.by_number(int(z)).mass for z in atoms["Z"][own].tolist()]
        types = [atom_style.def_type(f"atom{i}", mass=m) for i, m in enumerate(mass)]
        pos = atoms["pos"][own].detach().cpu().double().numpy()
        columns = {
            "id": np.arange(1, len(types) + 1),
            "type": np.array([t.name for t in types]),
            "x": pos[:, 0],
            "y": pos[:, 1],
            "z": pos[:, 2],
            "mass": np.array(mass),
            "mol_id": np.ones(len(types), dtype=np.int64),
        }
        structure = Frame()
        for key in self.potentials.keys():
            category, name = key.split(":", 1)
            source = next(s for s in self.forcefield.get_styles(category) if s.name == name)
            style = field.def_style(category, name, params=dict(source.params))
            params = {
                column: value.detach().cpu().double()
                for column, value in self._terms[key].potential_params(tables).items()
            }
            if category == "pair":
                if "charge" in params:
                    columns["charge"] = params.pop("charge")[own.cpu()].numpy()
                for i, atom_type in enumerate(types if params else ()):
                    style.def_type(atom_type.name, atom_type, **_row(params, int(own[i])))
                continue
            namespace = self._terms[key].namespace
            block = frame[namespace]
            if namespace == "bonds":
                index = block["bond_index"].T
            else:
                index = torch.stack([block[end] for end in _ENDS if end in block.keys()], dim=1)
            index = index.long().cpu()
            rows = torch.nonzero(local[index[:, 0]] >= 0).flatten().tolist()
            names = [f"{category}{n}" for n in range(len(rows))]
            for label, row in zip(names, rows, strict=True):
                ends = (types[int(local[atom])] for atom in index[row])
                style.def_type(label, *ends, **_row(params, row))
            ends = local[index[rows]].numpy()
            listed = {end: ends[:, m] for m, end in enumerate(_ENDS[: index.shape[1]])}
            structure[_MOLPY_BLOCKS[namespace]] = Block({**listed, "type": np.array(names)})
        structure["atoms"] = Block(columns)
        return structure, field

    def _styles(self, styles: Sequence[str | type[Term]] | None) -> tuple[str, ...]:
        if styles is None:
            return tuple(self.potentials.keys())
        keys = tuple(_style_of(item) for item in styles)
        unknown = [key for key in keys if key not in self.potentials]
        if not keys or unknown:
            raise ValueError(
                f"styles names {unknown or 'no style'}; the model carries "
                f"{tuple(self.potentials.keys())}"
            )
        return keys

    def energy(
        self,
        frame: TensorDict,
        *,
        styles: Sequence[str | type[Term]] | None = None,
    ) -> TensorDict:
        """Write ``graphs.energy`` ``(B,)`` kcal/mol — the traced seam.

        The heads of the priced styles run here, and their columns are written
        onto the batch (``bonds.k``, ``atoms.charge``, …): :meth:`predict` then
        :meth:`energy_at`. Which columns a loss may move is the training step's
        (``molix.core.steps.JointTrainStep.keep``), not the model's.

        Args:
            frame: Padded batch with ``atoms.pos``, ``atoms.batch``, ``graphs``
                and the namespaces the priced styles read.
            styles: Members of :attr:`potentials` to price — the classes in
                :mod:`molpot.classical` (``BondHarmonic``, ``LJCut``, …)
                — or ``None`` for every member. A training objective may pass
                the force field's style key instead. The tuple is fixed, so
                the call traces as straight-line code.

        Returns:
            The same frame with ``graphs.energy`` and the predicted columns written.

        Raises:
            ValueError: *styles* is empty or names a style the model does not
                carry.
        """
        return self.energy_at(frame, self.predict(frame, styles=styles), styles=styles)

    def energy_at(
        self,
        frame: TensorDict,
        tables: Tables,
        *,
        styles: Sequence[str | type[Term]] | None = None,
    ) -> TensorDict:
        """Write ``atoms.energy`` ``(N,)``, ``graphs.energy`` ``(B,)`` and the columns at *tables*.

        :meth:`energy` without the prediction: a caller that scores several
        objectives on one batch predicts once (:meth:`predict`), detaches per
        objective and prices each here — the encoder and the
        heads run once per batch, not once per objective.

        Args:
            frame: As :meth:`energy`.
            tables: Parameters as :meth:`predict` returns them.
            styles: As :meth:`energy`.

        Returns:
            The same frame with ``graphs.energy`` and the table columns written.
        """
        atom_energy = self.atom_energy(frame, tables, styles=styles)
        frame[ATOMIC_ENERGY_KEY] = atom_energy
        frame[ENERGY_KEY] = self._graph_sum(frame, atom_energy)
        self.write_tables(frame, tables)
        return frame

    def tables(
        self,
        frame: TensorDict,
        *,
        styles: Sequence[str | type[Term]] | None = None,
    ) -> TensorDict:
        """Write the predicted parameter columns of *styles* onto the batch — no energy.

        The heads :meth:`energy` runs, without pricing: an objective that reads
        the parameters themselves (charges against MBIS, the torsion prior)
        needs nothing else, and ``graphs.energy`` — the label — is not touched.

        Args:
            frame: Batch with the namespaces the styles' heads read.
            styles: As :meth:`energy`.

        Returns:
            The same frame with the predicted columns written.
        """
        self.write_tables(frame, self.predict(frame, styles=styles))
        return frame

    def partial_hessian(self, frame: TensorDict, tables: Tables) -> torch.Tensor:
        """The MM ``3×3`` blocks on the pairs ``hessian_pairs`` names ``(P, 9)``.

        A pair's block is the sum over every carried bonded term containing it.

        Args:
            frame: Batch with ``hessian_pairs`` and the ``hpair_*`` slot columns;
                ``atoms.pos`` must require grad (marked before the traced body).
            tables: Parameters as :meth:`predict` returns them, detached as the
                caller wants.

        Returns:
            ``(P, 9)`` kcal mol⁻¹ Å⁻², row-major blocks in ``hessian_pairs`` order.

        Raises:
            ValueError: A priced valence namespace has rows but no ``hpair_*``
                column — its term would silently receive no gradient.
        """
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
                name: value.to(pos.dtype)
                for name, value in self._terms[key].potential_params(tables).items()
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

    def _new(self, style: str) -> Term:
        """Build the parameter head of *style*. The numbers live on the spec, not on molpot."""
        kind = TERMS[style]
        width = self.graph.width
        heads = self.spec.heads
        dtype = self.dtype
        if style in ("bond:harmonic", "angle:harmonic", "dihedral:periodic", "improper:cvff"):
            floor = {"k_floor": heads.bond_k_floor} if style == "bond:harmonic" else {}
            return kind(
                width,
                self.graph.bond_width,
                **floor,
                hidden_dim=heads.bonded_hidden,
                width=heads.bonded_width,
                layers=heads.bonded_layers,
                out_hidden_dim=heads.bonded_out_hidden,
                out_layers=heads.bonded_out_layers,
                tanh_output=heads.tanh_output,
                dtype=dtype,
            )
        if style == "pair:lj/cut":
            return kind(
                width,
                heads.hidden,
                sigma_gain=heads.lj_sigma_gain,
                epsilon_gain=heads.lj_epsilon_gain,
                dtype=dtype,
            )
        if style == "pair:coul/cut":
            return kind(
                width,
                heads.hidden,
                max_atoms=heads.eem_max_atoms,
                jitter=heads.eem_jitter,
                dtype=dtype,
            )
        raise ValueError(
            f"Rondo has no parameters for {style}; it supplies {self.SUPPORTED_STYLES}"
        )

    def _add(self, style: str) -> Term:
        """Register the parameter head of *style*.

        The style is already a member of :attr:`potentials`.
        """
        term = self._new(style)
        if term.head_key in self.heads:
            raise ValueError(
                f"{style} writes heads.{term.head_key}, which "
                f"{self._occupant(term.head_key)} already writes"
            )
        self.heads[term.head_key] = term
        self._terms[style] = term
        return term

    def _occupant(self, head_key: str) -> str:
        """The style whose head is registered at *head_key*."""
        for style, term in self._terms.items():
            if term.head_key == head_key:
                return style
        return head_key

    @staticmethod
    def _mask(block: TensorDict, like: torch.Tensor) -> torch.Tensor:
        """The namespace's live-row mask as a float, or ``1`` for an unpadded batch."""
        if "mask" in block.keys():
            return block["mask"].to(like.dtype)
        return like.new_ones(())
