"""Write LAMMPS data and settings via :mod:`molpy.io`.

The Class-I :class:`~molpot.forcefield.ForceField` **is** a molpy
``ForceField``. This type builds a :class:`~molpy.core.atomistic.Atomistic`
frame, calls :meth:`~molpot.forcefield.ForceField.realize` to fill native
types, then delegates files to molpy. Independent of ``pair_style molnex``.

Required system contract (post-collate nested TensorDict)::

    system["atoms"]["pos"]                 # (N, 3) Å
    system["atoms"]["q"]                   # optional (N,) e — a *fallback*
                                           #   charge source; see below
    system["graphs"]["cell"]               # optional (3, 3) or (1, 3, 3) Å —
                                           #   periodic box; open padded box otherwise
    system["bonds"]["atomi" / "atomj"]     # optional
    system["angles"]["atomi" / "atomj" / "atomk"]
    system["propers"]["atomi" … "atoml"]

Charges are **not** read from the system here. They arrive on the
:class:`~molpot.forcefield.TypeAssignment` every call takes
(``types.atom_charges``, in ``e``), which is where
:meth:`~molpot.forcefield.ForceField.assign_types` put the values it keyed the
atom types on: the force field's own ``atoms.q`` parameter table when it has
one, else the system's ``atoms.q`` column, else zeros. Reading the system a
second time on this side wrote zeros for every field that keeps its charges in
its own table — a deck whose types exist *because* the atoms differ in charge
and whose atoms are then all neutral. ``charges=`` on
:meth:`LammpsWriter.write_data` / :meth:`LammpsWriter.to_frame` overrides that
column and nothing else, so an override that disagrees with the typing states
two models in one deck.

The bonded include is molpy's; the *pair* section is molnex's
(:meth:`LammpsWriter.write_pair_block`), because a periodic Class-I model has
no faithful spelling on the molrs side — see that method for the two defects.

Impropers are a **known gap**: they are neither written into the data file nor
registered by :meth:`~molpot.forcefield.ForceField.realize`, because molpy's
LAMMPS include writer takes the harmonic improper form only and
``improper:periodic`` — the form :class:`~molpot.forcefield.OPLSAAForceField`
declares — has no spelling to be registered under. A field whose impropers carry
energy therefore exports a deck that omits them (ledger debt ``F25``). They are
not typed either: :class:`~molpot.forcefield.TypeAssignment` carried an
``improper_type`` / ``improper_labels`` pair that no file ever read, and it was
deleted on 2026-09-08 rather than kept in step with a section nothing writes.

References:
    LAMMPS docs: pair_style lj/mdf, pair_style lj/cut/coul/long, pair_style
    hybrid/overlay, kspace_style pppm, special_bonds, pair_modify mix — the
    combining rule this writer refuses to leave to the deck.
    Spec: composable-a0-05-export
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from molix.ff_export.force_spec import ForceSpec
from molix.ff_export.lammps_adapter import LammpsAdapter
from molpot import MixingRule
from molpot.forcefield.container import ForceField, TypeAssignment

__all__ = ["LammpsWriter"]


class LammpsWriter:
    """Serialize a Class-I :class:`~molpot.forcefield.ForceField` through molpy IO.

    Every call takes the same :class:`~molpot.forcefield.TypeAssignment` — the
    condensed typing :meth:`~molpot.forcefield.ForceField.assign_types` computes
    once for a (force field, system) pair — because the three files address the
    types by the *same* numbers: the data file labels them, the bonded include
    names its coefficients after them and the pair block writes them as
    ``pair_coeff i j``. A second condensation computed per file is a numbering
    that can differ once, and LAMMPS reads a shifted one without complaint::

        types = ff.assign_types(batch, masses=masses)
        writer.write_data("system.data", batch, types=types, masses=masses)
        writer.write_settings(
            "system.in.settings", ff, batch, types=types, masses=masses, skip_pair_style=True
        )
        writer.write_pair_block("system.in.pair", spec, types=types)

    The bonded half of a deck is the first two calls and the nonbonded half the
    third, kept separate because molpy has no faithful spelling for it (see
    :meth:`write_pair_block`). ``skip_pair_style=True`` is what keeps them from
    both claiming the nonbonded section: without it the include also carries
    ``pair_style`` / ``pair_coeff`` / ``special_bonds`` lines, and a deck that
    sources both files states its nonbonded model twice. A **bonded-only** field
    is the one case that omits the flag and the third call together.
    """

    def write_data(
        self,
        path: str | Path,
        system: TensorDict,
        *,
        types: TypeAssignment,
        masses: torch.Tensor | list[float] | None = None,
        charges: torch.Tensor | list[float] | None = None,
        box_pad_A: float = 20.0,
    ) -> Path:
        """Write an ``atom_style full`` data file via :func:`molpy.io.write_lammps_data`.

        Args:
            path: Destination file.
            system: Nested TensorDict with ``atoms.pos`` and optional bonded
                namespaces (``atomi`` … columns).
            types: The system's condensed typing, from
                :meth:`~molpot.forcefield.ForceField.assign_types` on the force
                field this data file will be read with. Its ``atom_charges``
                fill the ``Atoms`` section's charge column.
            masses: Per-atom mass in amu ``(N,)``. Default: 12.011.
            charges: Per-atom charge ``(N,)`` in ``e``, overriding
                ``types.atom_charges``. See :meth:`to_frame` — the default is
                the charges the types were keyed on, and an override that
                disagrees with them writes a deck stating two models.
            box_pad_A: Padding around the molecule for an open box (Å). Ignored
                when ``system`` carries ``graphs.cell`` — see :meth:`to_frame`.

        Returns:
            The written path.

        Raises:
            TypeError: A column read off ``system`` is not a tensor.
            ValueError: ``graphs.cell`` holds more than one cell — one data file
                describes one box (both raised through :meth:`to_frame`).
        """
        from molpy.io.writers import write_lammps_data

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_lammps_data(
            dest,
            self.to_frame(system, types=types, masses=masses, charges=charges, box_pad_A=box_pad_A),
        )
        return dest

    def write_settings(
        self,
        path: str | Path,
        forcefield: ForceField,
        system: TensorDict,
        *,
        types: TypeAssignment,
        masses: torch.Tensor | list[float] | None = None,
        box_pad_A: float = 20.0,
        skip_pair_style: bool = False,
    ) -> Path:
        """Write a LAMMPS include via :func:`molpy.io.write_lammps_forcefield`.

        The include carries **no** ``units`` line (``skip_units=True``). It is
        sourced after ``read_data``, and LAMMPS refuses a ``units`` command once
        a simulation box exists — which it always does by the time a
        ``*_coeff`` line can be read. The deck states the unit style itself,
        before the box; :attr:`~molpot.forcefield.ForceField.units` is what the
        coefficients written here are expressed in.

        Args:
            path: Destination file.
            forcefield: Filled Class-I container.
            system: Topology whose types the coefficients are written for.
            types: That system's condensed typing under ``forcefield``, and the
                names the coefficients are written under. It is passed straight
                to :meth:`~molpot.forcefield.ForceField.realize`, which
                registers the molpy types from it and condenses nothing — so
                this must be the assignment ``forcefield`` itself makes for
                ``system`` at these ``masses``. A foreign one names
                ``bond_coeff`` lines the data file's labelmap does not contain,
                and nothing here re-derives it to say so.
            masses: Per-atom mass in amu (must match :meth:`write_data`, and the
                ``masses`` ``types`` was assigned with — mass is part of an atom
                type).
            box_pad_A: Open-box padding (must match :meth:`write_data`); ignored
                under ``graphs.cell`` — see :meth:`to_frame`.
            skip_pair_style: Write a **bonded-only** include — no
                ``pair_style``, no ``pair_coeff``, no ``special_bonds`` — so
                that :meth:`write_pair_block` owns the whole nonbonded section
                and the deck cannot end up with two disagreeing pair
                definitions. Every ``pair:*`` style is dropped from the
                realized clone (the caller's force field is untouched) because
                molpy emits the ``pair_coeff`` lines from the styles that
                remain even when asked to skip the ``pair_style`` line.

        Returns:
            The written path.

        Raises:
            TypeError: ``system["atoms"]["pos"]`` is not a tensor.
            ValueError: ``masses`` has a different length than ``atoms.pos``
                (raised through :meth:`to_frame`).
        """
        from molpy import PairStyle
        from molpy.io.writers import write_lammps_forcefield

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame(system, types=types, masses=masses, box_pad_A=box_pad_A)
        # The caller's assignment *is* the numbering, handed to the one object
        # that registers molpy types from it. Condensing again here — to compare
        # with what was given, and again inside ``realize`` — made three passes
        # produce the answer this method was already holding.
        ff = forcefield.clone().realize(system, masses=masses, types=types)
        if skip_pair_style:
            for name in list(ff.pair_styles()):
                ff.remove_style(PairStyle, name)
        write_lammps_forcefield(
            dest,
            ff,
            frame=frame,
            units=ff.units,
            skip_pair_style=skip_pair_style,
            skip_units=True,
        )
        return dest

    def write_pair_block(
        self,
        path: str | Path,
        spec: ForceSpec,
        *,
        types: TypeAssignment,
    ) -> Path:
        """Write the nonbonded section of a LAMMPS deck from ``spec``.

        molnex owns this block rather than molpy because a periodic Class-I
        model has no faithful spelling on the molrs side: its include keeps
        bare ``pair_coeff`` lines, and its hybrid path writes plain ``hybrid``,
        under which each pair of atom types is handled by **one** sub-style —
        so every LJ pair would silently lose its Coulomb term. The correct
        style is ``hybrid/overlay``, which sums both sub-styles over every
        pair, and there the sub-style name in each ``pair_coeff`` line is
        mandatory.

        **Every** ``i <= j`` pair of LJ types gets its own line —
        ``pair_coeff 1 1``, ``pair_coeff 1 2``, ``pair_coeff 2 2``, … — and none
        of the mixing is left to LAMMPS. Both halves of that are measured
        against ``lmp`` 30 Mar 2026, not inferred:

        * Under ``hybrid/overlay`` the ``pair_coeff * * <coulomb>`` line below
          maps *every* type pair to the Coulomb sub-style, and LAMMPS then never
          mixes the LJ sub-style onto an ``i ≠ j`` pair it was not explicitly
          assigned to. With diagonal lines only, **every cross-type
          Lennard-Jones interaction is silently zero** — no warning, no error.
        * Even without the hybrid, LAMMPS's default combining rule (``geometric``
          in both ε and σ unless the deck says ``pair_modify mix``) is not
          necessarily the model's: molnex mixes Lorentz–Berthelot by default and
          geometrically for OPLS-AA. So the cross terms of an exported deck would
          be a different model even where LAMMPS does mix.

        The values therefore come from the model's own rule — the ``mixing``
        field the adapter read off
        :class:`~molpot.ir.nonbonded.NonbondedMethodRecord`, applied through
        :meth:`MixingRule.mix
        <molpot.potentials.vdw.lj126.MixingRule.mix>`, the same call
        :class:`~molpot.forcefield.Potentials` evaluates with — over the
        per-type ``ε`` / ``σ`` on an ``edge_index`` of the ``i <= j`` pairs.
        Dispatching on what the model declared, rather than restating ``σ_ij =
        ½(σ_i + σ_j)`` here, is what makes "the deck uses molnex's mixing" true
        by construction instead of by review, and it is why an OPLS-AA export
        stopped silently changing rules on 2026-09-08. (This is the one
        sanctioned ``molix.ff_export`` → ``molpot`` edge beyond the IR,
        alongside ``molix.ff_export``'s existing import of ``molpot.ir``.)

        ``pair_coeff * * <coulomb sub-style>`` carries no parameters — charges
        come from the data file — and is also what a **charges-only** model
        (no ``lj`` bag, so no per-type lines at all) is left with, there without
        a sub-style token unless a hybrid named one. Numbers go through
        :meth:`~molix.ff_export.LammpsAdapter.number`, which owns the deck's
        numeric spelling, so these coefficients and the ``pair_style`` line
        above them cannot be written at different precisions.

        After the coefficients come the adapter's ``kspace_style``,
        ``kspace_modify`` and ``special_bonds`` records, each rendered as the
        adapter already spelled it. ``kspace_modify gewald`` states the model's
        Ewald splitting parameter, so the deck solves the decomposition molnex
        ran rather than one LAMMPS picked from the requested accuracy.

        **Cost, and what pays it.** The block is ``T(T+1)/2`` lines for ``T``
        distinct atom types. Until 2026-09-08 molnex exported *identity* types —
        one per atom — which made that ``N(N+1)/2``: 26 atoms cost 351 lines and
        1000 atoms half a million (ledger debt **F25**). The rows the adapter
        hands over are still per atom, because the Lennard-Jones bag of a
        Class-I :class:`~molpot.ir.PotentialIR` is per atom; they are collapsed
        here onto ``types``, whose classes are equal-parameter atoms by
        construction — the 26-atom water/ion box has four (O, H, Na⁺, Cl⁻) and
        writes ten lines. The fix was never a return to LAMMPS-side mixing,
        which is wrong at any size for the two reasons above.

        Args:
            path: Destination file, typically ``include``-d after the bonded
                one written by :meth:`write_settings` with
                ``skip_pair_style=True``.
            spec: Force-spec from
                :class:`~molix.ff_export.LammpsAdapter`, carrying the ``pair``
                record and optionally the ``kspace`` / ``special_bonds`` ones.
            types: The condensed typing the data file was written with. LAMMPS
                type ``i`` is class ``i - 1`` here, because
                :meth:`~molpot.forcefield.ForceField.assign_types` labels the
                classes in an order molrs reproduces when it sorts the labels as
                text.

        Returns:
            The written path.

        Raises:
            KeyError: ``spec`` has no ``pair`` record — an all-bonded force
                field has no pair block to write.
            ValueError: ``spec`` and ``types`` describe different systems —
                either a different atom count, or two atoms of one type carrying
                different ``ε`` / ``σ``, which means the assignment was made by
                another force field than the one that was translated. One line
                would then have to state two models.
        """
        pair = next((f for f in spec.forces if f.get("category") == "pair"), None)
        if pair is None:
            raise KeyError(f"force-spec for {spec.backend!r} carries no 'pair' record")

        lj_tag = ""
        coulomb_tag = ""
        if pair["style"] == "hybrid/overlay":
            for sub in pair["sub_styles"]:
                if sub["style"].startswith("coul"):
                    coulomb_tag = sub["style"]
                else:
                    lj_tag = sub["style"]

        lines: list[str] = [pair["type"]]
        entries = pair["parameters"]
        if entries:
            epsilon, sigma = self._lj_per_type(entries, types)
            n_types = len(types.atom_labels)
            rows = [(a, b) for a in range(n_types) for b in range(a, n_types)]
            mixed = MixingRule(pair["mixing"]).mix(
                {
                    "epsilon": torch.tensor(epsilon, dtype=torch.float64),
                    "sigma": torch.tensor(sigma, dtype=torch.float64),
                },
                torch.tensor(rows, dtype=torch.long),
            )
            epsilon_ij = mixed["epsilon_ij"].tolist()
            sigma_ij = mixed["sigma_ij"].tolist()
            for row, (a, b) in enumerate(rows):
                coeff = ["pair_coeff", str(a + 1), str(b + 1)]
                if lj_tag:
                    coeff.append(lj_tag)
                coeff += [
                    LammpsAdapter.number(epsilon_ij[row]),
                    LammpsAdapter.number(sigma_ij[row]),
                ]
                lines.append(" ".join(coeff))
        if coulomb_tag or not entries:
            # Two reasons for a ``* *`` line, and both need it. Under
            # ``hybrid/overlay`` it is what hands every type pair to the Coulomb
            # sub-style, which the per-type LJ lines above never mention. With no
            # LJ entries at all — a charges-only model — it is the *only* line
            # there is, and LAMMPS leaves ``pair_coeff`` unset without it; the
            # sub-style token then appears only if there is a hybrid to name one.
            lines.append(f"pair_coeff * * {coulomb_tag}".rstrip())
        for category in ("kspace", "kspace_modify", "special_bonds"):
            lines += [f["type"] for f in spec.forces if f.get("category") == category]

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(f"{line}\n" for line in lines))
        return dest

    @staticmethod
    def _lj_per_type(
        entries: list[dict[str, Any]],
        types: TypeAssignment,
    ) -> tuple[list[float], list[float]]:
        """Collapse the adapter's per-atom ``ε`` / ``σ`` rows onto ``types``.

        Args:
            entries: The ``pair`` record's ``parameters``, one row per atom in
                atom order (``LammpsAdapter._lj_parameters``).
            types: The condensed typing of the same system.

        Returns:
            ``(epsilon, sigma)`` in kcal/mol and Å, one entry per atom type, in
            class order.

        Raises:
            ValueError: There is not one row per atom, or two atoms of one type
                disagree about ``(ε, σ)``. Both mean ``entries`` and ``types``
                came from different force fields: an atom type is *defined* by
                parameter equality, so within one field the disagreement cannot
                happen, and picking either value would export a model neither
                side holds. The message names the two atoms.
        """
        if len(entries) != len(types.atom_type):
            raise ValueError(
                f"the pair record carries {len(entries)} Lennard-Jones rows but the "
                f"type assignment covers {len(types.atom_type)} atoms; they describe "
                "different systems"
            )
        epsilon = [0.0] * len(types.atom_labels)
        sigma = [0.0] * len(types.atom_labels)
        owner = [-1] * len(types.atom_labels)
        for atom, (entry, cls) in enumerate(zip(entries, types.atom_type, strict=True)):
            if owner[cls] < 0:
                owner[cls] = atom
                epsilon[cls] = float(entry["epsilon"])
                sigma[cls] = float(entry["sigma"])
            elif (epsilon[cls], sigma[cls]) != (float(entry["epsilon"]), float(entry["sigma"])):
                raise ValueError(
                    f"atoms {owner[cls]} and {atom} share LAMMPS type "
                    f"{types.atom_labels[cls]!r} but carry different Lennard-Jones "
                    f"parameters: (epsilon, sigma) = "
                    f"({epsilon[cls]!r}, {sigma[cls]!r}) vs "
                    f"({float(entry['epsilon'])!r}, {float(entry['sigma'])!r})"
                )
        return epsilon, sigma

    def to_frame(
        self,
        system: TensorDict,
        *,
        types: TypeAssignment,
        masses: torch.Tensor | list[float] | None = None,
        charges: torch.Tensor | list[float] | None = None,
        box_pad_A: float = 20.0,
    ) -> Any:
        """Build a molpy ``Frame`` (Atomistic → ``to_frame``) from ``system``.

        Bonds, angles and proper dihedrals are written, each labelled with its
        class name from ``types``. **Impropers are not** — neither here nor in
        the include :meth:`write_settings` realizes, because molpy's LAMMPS
        writer takes the harmonic improper form only and OPLS-AA's
        ``improper:periodic`` has no spelling to be registered under. A field
        whose impropers carry energy exports a deck that silently omits that
        term (ledger debt ``F25``); ``types`` carries no improper class map
        either, since nothing on this side would read one.

        The **box** is the periodic/open switch of the whole export, so it is
        read from the system rather than assumed: ``system["graphs", "cell"]``
        — ``(3, 3)`` or ``(1, 3, 3)`` in Å, rows being the lattice vectors as
        everywhere in molnex — becomes a periodic :class:`molpy.Box`, and a
        system carrying no cell keeps the open box
        ``Box.from_bounds(xyz, padding=box_pad_A)``. A cell written as padded
        open bounds would leave LAMMPS summing a different lattice than the one
        the model was evaluated in, which ``coul/long`` + ``kspace_style pppm``
        cannot survive.

        Args:
            system: Nested TensorDict with ``atoms.pos`` ``(N, 3)`` Å, optional
                ``graphs.cell`` and the bonded namespaces.
            types: The system's condensed typing; every atom and interaction is
                labelled with its class name (``t1``, ``b1``, …) rather than with
                one name of its own, and its ``atom_charges`` are the default
                charge column (below).
            masses: Per-atom mass in amu ``(N,)``. Default: 12.011.
            charges: Per-atom charge ``(N,)`` in ``e``, overriding
                ``types.atom_charges`` — which is what the ``Atoms`` section
                states otherwise, being the charges
                :meth:`~molpot.forcefield.ForceField.assign_types` keyed the
                atom types on (the field's ``atoms.q`` table, else the system's
                ``atoms.q``, else zeros). Pass this only to write a column the
                typing was *not* computed from; two atoms of one type given
                different charges here make a deck whose ``Atoms`` section and
                whose type list describe different models.
            box_pad_A: Padding around the molecule (Å) for the open box. Unused
                when a cell is present.

        Returns:
            The molpy ``Frame``.

        Raises:
            TypeError: ``atoms.pos`` or ``graphs.cell`` is not a tensor.
            ValueError: ``graphs.cell`` is batched with ``B > 1`` — one data
                file describes one box, so there is no faithful single lattice
                to write for a batch of several. Or ``masses`` has a different
                length than ``atoms.pos``.
        """
        from molpy import Box
        from molpy.core.atomistic import Atomistic

        pos = self._tensor(system, "atoms", "pos")
        xyz = pos.detach().cpu().double().numpy()
        n_atoms = int(xyz.shape[0])
        mass_list = ForceField._mass_list(masses, n_atoms)
        # Default: the charges the typing was computed from, carried on the
        # assignment. Reading ``atoms.q`` again here instead wrote zeros for
        # every field that keeps its charges in its own ``atoms`` table — the
        # deck's types would then differ *by* a charge its atoms do not carry.
        if charges is not None:
            charge_list = [float(x) for x in torch.as_tensor(charges).reshape(-1).tolist()]
        else:
            charge_list = list(types.atom_charges)

        # One assignment reaches both sides of the molpy seam — this frame and
        # the force field ``write_settings`` realizes — so the labels are the
        # same strings, and molrs then numbers the types in class order, which is
        # the numbering ``write_pair_block`` addresses (see
        # ``ForceField.assign_types``).
        world = Atomistic()
        handles = []
        for i in range(n_atoms):
            handles.append(
                world.def_atom(
                    {
                        "type": types.atom_labels[types.atom_type[i]],
                        "x": float(xyz[i, 0]),
                        "y": float(xyz[i, 1]),
                        "z": float(xyz[i, 2]),
                        "charge": charge_list[i],
                        # ``mol_id`` — not ``mol`` — is the column molrs reads to
                        # infer ``atom_style full``. Under any other spelling the
                        # molecule id is dropped, the data file comes out
                        # ``Atoms # charge`` (6 fields, no molecule column), and
                        # LAMMPS then refuses to read it under the
                        # ``atom_style full`` the settings include declares:
                        # "Incorrect format in Atoms section". Verified against
                        # lmp 30 Mar 2026 while capturing the parity goldens.
                        "mol_id": 1,
                        "mass": mass_list[i],
                    }
                )
            )
        for n, (i, j) in enumerate(ForceField._arity(system, "bonds", 2)):
            world.def_bond(handles[i], handles[j], type=types.bond_labels[types.bond_type[n]])
        for n, (i, j, k) in enumerate(ForceField._arity(system, "angles", 3)):
            world.def_angle(
                handles[i],
                handles[j],
                handles[k],
                type=types.angle_labels[types.angle_type[n]],
            )
        for n, (i, j, k, l) in enumerate(ForceField._arity(system, "propers", 4)):
            world.def_dihedral(
                handles[i],
                handles[j],
                handles[k],
                handles[l],
                type=types.proper_labels[types.proper_type[n]],
            )
        frame = world.to_frame()
        cell = self._cell(system)
        frame.box = (
            Box(cell)
            if cell is not None
            else Box.from_bounds(np.asarray(xyz, dtype=float), padding=box_pad_A)
        )
        return frame

    @staticmethod
    def _cell(system: TensorDict) -> np.ndarray | None:
        """Lattice vectors ``(3, 3)`` in Å from ``graphs.cell``, or ``None``.

        Raises:
            ValueError: The cell is batched over more than one graph.
        """
        if ("graphs", "cell") not in system.keys(include_nested=True):
            return None
        cell = LammpsWriter._tensor(system, "graphs", "cell")
        if cell.ndim == 3:
            if cell.shape[0] != 1:
                raise ValueError(
                    f"graphs.cell holds {cell.shape[0]} cells but one LAMMPS data file "
                    "describes one box; write one system per file"
                )
            cell = cell[0]
        return cell.detach().cpu().double().numpy()

    @staticmethod
    def _tensor(system: TensorDict, *keys: str) -> torch.Tensor:
        value = system[keys] if len(keys) > 1 else system[keys[0]]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"system{list(keys)} must be a tensor, got {type(value)!r}")
        return value
