"""Write LAMMPS data and settings via :mod:`molpy.io`.

The Class-I :class:`~molpot.forcefield.ForceField` **is** a molpy
``ForceField``. This type builds a :class:`~molpy.core.atomistic.Atomistic`
frame, calls :meth:`~molpot.forcefield.ForceField.realize` to fill native
types, then delegates files to molpy. Independent of ``pair_style molnex``.

Required system contract (post-collate nested TensorDict)::

    system["atoms"]["pos"]                 # (N, 3) Å
    system["atoms"]["q"]                   # optional (N,) e
    system["graphs"]["cell"]               # optional (3, 3) or (1, 3, 3) Å —
                                           #   periodic box; open padded box otherwise
    system["bonds"]["atomi" / "atomj"]     # optional
    system["angles"]["atomi" / "atomj" / "atomk"]
    system["propers"]["atomi" … "atoml"]

The bonded include is molpy's; the *pair* section is molnex's
(:meth:`LammpsWriter.write_pair_block`), because a periodic Class-I model has
no faithful spelling on the molrs side — see that method for the two defects.

References:
    LAMMPS docs: pair_style lj/mdf, pair_style lj/cut/coul/long, pair_style
    hybrid/overlay, kspace_style pppm, special_bonds
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
from molpot import lorentz_berthelot
from molpot.forcefield.container import ForceField

__all__ = ["LammpsWriter"]


class LammpsWriter:
    """Serialize a Class-I :class:`~molpot.forcefield.ForceField` through molpy IO.

    Identity types: one atom type per atom, one bonded type per interaction.
    The bonded half of a deck is two calls — :meth:`write_data` and
    :meth:`write_settings` — and the nonbonded half is a third,
    :meth:`write_pair_block`, kept separate because molpy has no faithful
    spelling for it (see that method).
    """

    def write_data(
        self,
        path: str | Path,
        system: TensorDict,
        *,
        masses: torch.Tensor | list[float] | None = None,
        charges: torch.Tensor | list[float] | None = None,
        box_pad_A: float = 20.0,
    ) -> Path:
        """Write an ``atom_style full`` data file via :func:`molpy.io.write_lammps_data`.

        Args:
            path: Destination file.
            system: Nested TensorDict with ``atoms.pos`` and optional bonded
                namespaces (``atomi`` … columns).
            masses: Per-atom mass in amu ``(N,)``. Default: 12.011.
            charges: Per-atom charge ``(N,)`` in ``e``. Default: ``atoms.q``
                or zeros.
            box_pad_A: Padding around the molecule for an open box (Å). Ignored
                when ``system`` carries ``graphs.cell`` — see :meth:`to_frame`.

        Returns:
            The written path.
        """
        from molpy.io.writers import write_lammps_data

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_lammps_data(
            dest, self.to_frame(system, masses=masses, charges=charges, box_pad_A=box_pad_A)
        )
        return dest

    def write_settings(
        self,
        path: str | Path,
        forcefield: ForceField,
        system: TensorDict,
        *,
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
            system: Topology used to assign identity types.
            masses: Per-atom mass in amu (must match :meth:`write_data`).
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
        """
        from molpy import PairStyle
        from molpy.io.writers import write_lammps_forcefield

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame(system, masses=masses, box_pad_A=box_pad_A)
        ff = forcefield.clone().realize(system, masses=masses)
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

    def write_pair_block(self, path: str | Path, spec: ForceSpec) -> Path:
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
        * Even without the hybrid, LAMMPS's default combining rule is
          *geometric* in σ, while molnex mixes Lorentz–Berthelot. So the cross
          terms of an exported deck would be a different model even where LAMMPS
          does mix.

        The values therefore come from
        :func:`~molpot.potentials.vdw.lj126.lorentz_berthelot` itself — the same
        function :class:`~molpot.forcefield.Potentials` evaluates with — applied
        to the per-type ``ε`` / ``σ`` over an ``edge_index`` of the ``i <= j``
        pairs. Calling molnex's own rule rather than restating ``σ_ij =
        ½(σ_i + σ_j)`` here is what makes "the deck uses molnex's mixing" true by
        construction instead of by review. (This is the one sanctioned
        ``molix.ff_export`` → ``molpot`` edge beyond the IR, alongside
        ``molix.ff_export``'s existing import of ``molpot.ir``.)

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

        **Cost, and the debt it carries.** With identity types (one type per
        atom) the block is ``N(N+1)/2`` lines, i.e. quadratic in the atom count:
        26 atoms give 351 ``pair_coeff`` lines, 1000 atoms give half a million.
        That is fine for the export sizes A0 targets and is *not* fine as a
        general path; it is recorded as ledger debt **F25** in
        ``.claude/notes/composable-potentials.md``. The fix is condensed typing
        (equivalent atoms sharing a type), not a return to LAMMPS-side mixing —
        which would be wrong at any size.

        Args:
            path: Destination file, typically ``include``-d after the bonded
                one written by :meth:`write_settings` with
                ``skip_pair_style=True``.
            spec: Force-spec from
                :class:`~molix.ff_export.LammpsAdapter`, carrying the ``pair``
                record and optionally the ``kspace`` / ``special_bonds`` ones.

        Returns:
            The written path.

        Raises:
            KeyError: ``spec`` has no ``pair`` record — an all-bonded force
                field has no pair block to write.
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
            rows = [(a, b) for a in range(len(entries)) for b in range(a, len(entries))]
            mixed = lorentz_berthelot(
                {
                    "epsilon": torch.tensor([e["epsilon"] for e in entries], dtype=torch.float64),
                    "sigma": torch.tensor([e["sigma"] for e in entries], dtype=torch.float64),
                },
                torch.tensor(rows, dtype=torch.long),
            )
            epsilon_ij = mixed["epsilon_ij"].tolist()
            sigma_ij = mixed["sigma_ij"].tolist()
            for row, (a, b) in enumerate(rows):
                coeff = [
                    "pair_coeff",
                    str(entries[a]["type_index"]),
                    str(entries[b]["type_index"]),
                ]
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

    def to_frame(
        self,
        system: TensorDict,
        *,
        masses: torch.Tensor | list[float] | None = None,
        charges: torch.Tensor | list[float] | None = None,
        box_pad_A: float = 20.0,
    ) -> Any:
        """Build a molpy ``Frame`` (Atomistic → ``to_frame``) from ``system``.

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
                ``atoms.q``, optional ``graphs.cell`` and the bonded namespaces.
            masses: Per-atom mass in amu ``(N,)``. Default: 12.011.
            charges: Per-atom charge ``(N,)`` in ``e``. Default: ``atoms.q``
                or zeros.
            box_pad_A: Padding around the molecule (Å) for the open box. Unused
                when a cell is present.

        Returns:
            The molpy ``Frame``.

        Raises:
            ValueError: ``graphs.cell`` is batched with ``B > 1`` — one data
                file describes one box, so there is no faithful single lattice
                to write for a batch of several.
        """
        from molpy import Box
        from molpy.core.atomistic import Atomistic

        pos = self._tensor(system, "atoms", "pos")
        xyz = pos.detach().cpu().double().numpy()
        n_atoms = int(xyz.shape[0])
        mass_list = ForceField._mass_list(masses, n_atoms)
        if charges is not None:
            charge_list = [float(x) for x in torch.as_tensor(charges).reshape(-1).tolist()]
        elif ("atoms", "q") in system.keys(include_nested=True):
            charge_list = [
                float(x)
                for x in self._tensor(system, "atoms", "q").detach().cpu().reshape(-1).tolist()
            ]
        else:
            charge_list = [0.0] * n_atoms

        # The same atom count both sides of the molpy seam pass, so the frame's
        # labels and the realized force field's are the same strings — and so
        # molrs numbers the types in atom order, which is the numbering
        # ``write_pair_block`` addresses (see ``ForceField.identity_type_label``).
        def label(index: int) -> str:
            return ForceField.identity_type_label(index, n_atoms)

        world = Atomistic()
        handles = []
        for i in range(n_atoms):
            handles.append(
                world.def_atom(
                    {
                        "type": label(i),
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
        for i, j in ForceField._arity(system, "bonds", 2):
            world.def_bond(handles[i], handles[j], type="-".join((label(i), label(j))))
        for i, j, k in ForceField._arity(system, "angles", 3):
            world.def_angle(
                handles[i],
                handles[j],
                handles[k],
                type="-".join((label(i), label(j), label(k))),
            )
        for i, j, k, l in ForceField._arity(system, "propers", 4):
            world.def_dihedral(
                handles[i],
                handles[j],
                handles[k],
                handles[l],
                type="-".join((label(i), label(j), label(k), label(l))),
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
