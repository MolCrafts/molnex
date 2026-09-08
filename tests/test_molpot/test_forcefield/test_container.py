"""OPLSAAForceField inherits ForceField; product templates are named subclasses."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from molpy import ForceField as MolpyForceField
from tensordict import TensorDict
from torch import nn

from molix.batch import NeighborList, TensorFrame, collate_molecules
from molpot.forcefield import ForceField, OPLSAAForceField, TypeAssignment
from molpot.ir import ExclusionScales, NonbondedScaling
from molpot.manifest import ModelManifest, NumericsRecord, ParamTableRecord
from tests.conftest import (
    BUTANE_PAIR_LIST_CUTOFF,
    butane_pair_class_i_forcefield,
    frame,
)

#: Per-species ``(ε kcal/mol, σ Å, q e, mass amu)`` of the periodic parity case
#: — TIP3P water with OPLS-shaped ions, copied from
#: ``interface/tests/parity_classical.py`` rather than imported (``interface/``
#: is not on the unit import path), so the box typed here is the one the LAMMPS
#: parity run exports.
WATER_ION_SPECIES: dict[str, tuple[float, float, float, float]] = {
    "O": (0.1521, 3.1507, -0.834, 15.999),
    "H": (0.0, 0.0, 0.417, 1.008),
    "Na": (0.0028, 3.328, 1.0, 22.990),
    "Cl": (0.1, 4.417, -1.0, 35.453),
}

#: TIP3P internal geometry: O–H bond length (Å) and H–O–H angle (deg).
OH_A = 0.9572
HOH_DEG = 104.52


def _line_of_atoms(n_atoms: int, bonds: list[tuple[int, int]] | None = None) -> TensorDict:
    """``n_atoms`` collinear carbons, optionally bonded — a topology, no parameters.

    One molpy ``Frame`` through ``collate_molecules``: the ``bonds`` block
    therefore arrives as the ``(2, N)`` COO ``bonds.bond_index``, which is what
    a batch's covalent topology looks like everywhere downstream of the collate.
    """
    built = frame(
        Z=(6,) * n_atoms,
        pos=[[1.53 * i, 0.0, 0.0] for i in range(n_atoms)],
        bonds=None if not bonds else ([b[0] for b in bonds], [b[1] for b in bonds]),
    )
    return collate_molecules([built])


def _water_ion_box() -> tuple[ForceField, TensorDict, torch.Tensor]:
    """Two waters plus Na⁺ and Cl⁻ — eight atoms, four parameter classes.

    The species order is ``O H H O H H Na Cl``, so first-occurrence order
    (``O H Na Cl``) is neither atom order nor alphabetical: an assignment that
    numbered types by atom index or by species name would disagree with this
    fixture rather than pass it by luck.

    Returns:
        The filled force field, the system, and the per-atom masses in amu.
    """
    labels = ["O", "H", "H", "O", "H", "H", "Na", "Cl"]
    half = math.radians(HOH_DEG) / 2.0
    arms = (
        [OH_A * math.cos(half), OH_A * math.sin(half), 0.0],
        [OH_A * math.cos(half), -OH_A * math.sin(half), 0.0],
    )
    xyz: list[list[float]] = []
    for origin in ([0.0, 0.0, 0.0], [4.0, 0.0, 0.0]):
        xyz.append(origin)
        xyz += [[origin[k] + arm[k] for k in range(3)] for arm in arms]
    xyz += [[0.0, 4.0, 0.0], [4.0, 4.0, 0.0]]  # Na, Cl

    def column(index: int) -> torch.Tensor:
        return torch.tensor([WATER_ION_SPECIES[t][index] for t in labels], dtype=torch.float64)

    # Two water molecules and two ions, each its own molpy ``Frame``, collated
    # the way a dataset reaches a model: the per-molecule bond and angle indices
    # below are molecule-local and the collate rebases them into the box.
    numbers = {"O": 8, "H": 1, "Na": 11, "Cl": 17}
    waters = [
        frame(
            Z=(8, 1, 1),
            pos=xyz[3 * molecule : 3 * molecule + 3],
            bonds=([0, 0], [1, 2]),
            terms={"angles": {"atomi": [1], "atomj": [0], "atomk": [2]}},
        )
        for molecule in range(2)
    ]
    ions = [frame(Z=(numbers[label],), pos=[xyz[6 + ion]]) for ion, label in enumerate(labels[6:])]
    system = collate_molecules([*waters, *ions])
    ff = (
        ForceField(name="water-ion", units="real")
        .def_atomstyle("full")
        .def_bondstyle("harmonic")
        .def_anglestyle("harmonic")
        .with_params(
            bonds={
                "k": torch.full((4,), 1000.0, dtype=torch.float64),
                "r0": torch.full((4,), OH_A, dtype=torch.float64),
            },
            angles={
                "k": torch.full((2,), 200.0, dtype=torch.float64),
                "theta0": torch.full((2,), math.radians(HOH_DEG), dtype=torch.float64),
            },
            atoms={"epsilon": column(0), "sigma": column(1), "q": column(2)},
        )
    )
    return ff, system, column(3)


class _StubMlip(nn.Module):
    """The minimal shape ``def_pairstyle`` accepts as an MLIP pair term.

    An mlip is not a named style — it *is* the module — so identity is the
    only thing that distinguishes one from another, and it is exactly what
    :meth:`~molpot.forcefield.ForceField.clone` has to preserve. No energy is
    evaluated here: the copy contract is about which object arrives, not what
    it computes.
    """

    def _write_energy(self, system: TensorDict) -> TensorDict:
        return system


class TestForceField:
    def test_types_are_the_equivalence_classes_of_the_parameters(self):
        """Eight atoms, four species — four atom types, one bond type, one angle.

        A LAMMPS type *is* an equivalence class of parameters: two atoms
        sharing ``(mass, q, ε, σ)`` are one atom type, and the four O–H bonds of
        two waters — same atom types, same ``(k, r0)`` — are one bond type.
        Identity typing (one type per atom, the a0 shortcut this replaces) gave
        this box eight atom types, four bond types and two angle types, and an
        exported pair block of ``N(N+1)/2`` lines rather than ``T(T+1)/2``.

        Labels follow **first occurrence** — ``O H Na Cl`` here — so the
        assignment can be read straight off the system, and the interaction
        lists index the type tables rather than repeating their parameters.
        """
        ff, system, masses = _water_ion_box()
        types = ff.assign_types(system, masses=masses)
        assert types.atom_labels == ("t1", "t2", "t3", "t4")
        assert types.atom_type == (0, 1, 1, 0, 1, 1, 2, 3)
        assert types.bond_labels == ("b1",)
        assert types.bond_type == (0, 0, 0, 0)
        assert types.angle_labels == ("a1",)
        assert types.angle_type == (0, 0)

    def test_assign_types_reads_a_typed_view_like_the_plain_batch(self):
        """``TensorFrame`` in, the same assignment out.

        ``TensorFrame.from_tensordict`` is a zero-copy typed view of the
        post-collate batch (CLAUDE.md § TensorDict Contract) — the same leaves,
        declared. Typing reads ``atoms.pos``, ``atoms.q`` and the bonded
        namespaces off whichever of the two it is handed, so an assignment that
        moved between them would mean some reader branched on the container's
        *type* instead of on the system it describes.
        """
        ff, system, masses = _water_ion_box()
        view = TensorFrame.from_tensordict(system)

        assert ff.assign_types(view, masses=masses) == ff.assign_types(system, masses=masses)

    def test_one_atom_type_pair_can_carry_two_bond_types(self):
        """Same atom types, different ``(k, r0)`` — two bond types, not one.

        The bonded key is the ordered atom-type tuple **and** the parameter
        tuple. Keyed on the atom types alone, these two bonds would collapse
        into a single type carrying whichever ``r0`` was registered last, and
        both bonds would then be exported at that length — a silent change of
        model, which is why the bonded types are named explicitly instead of
        being derived from their atoms' labels.
        """
        ff = (
            ForceField(name="x", units="real")
            .def_atomstyle("full")
            .def_bondstyle("harmonic")
            .with_params(
                bonds={
                    "k": torch.tensor([200.0, 200.0], dtype=torch.float64),
                    "r0": torch.tensor([1.0, 1.5], dtype=torch.float64),
                }
            )
        )
        types = ff.assign_types(_line_of_atoms(4, bonds=[(0, 1), (2, 3)]))
        assert types.atom_type == (0, 0, 0, 0)  # four parameterless, equal atoms
        assert types.bond_labels == ("b1", "b2")
        assert types.bond_type == (0, 1)

    def test_labels_are_zero_padded_to_the_type_count(self):
        """Ten types label as ``t01``…``t10``, so text order is type order.

        The labels are sorted as text downstream (molrs writes the type tables
        in that order), and unpadded ``t10`` sorts before ``t2`` — which
        renumbers the types against the parameters they came from and corrupts
        every coeff line that follows. Measured once on a 26-atom water/ion box
        exported unpadded: ``E_vdwl`` came out ``-0.2251`` kcal/mol against
        molnex's ``-0.8163``, with no warning from LAMMPS.
        """
        ff = ForceField(name="x", units="real").def_atomstyle("full")
        labels = ff.assign_types(
            _line_of_atoms(10), masses=torch.arange(1.0, 11.0, dtype=torch.float64)
        ).atom_labels
        assert labels == tuple(f"t{i:02d}" for i in range(1, 11))
        assert sorted(labels) == list(labels)

    def test_realize_registers_one_native_type_per_class(self):
        """``realize`` fills molpy's tables from that same assignment.

        The type names are what cross the molpy seam — into the data file's
        label sections and onto every ``*_coeff`` line — so a second typing
        here would be a second numbering, and the pair block's numeric
        ``i j`` would address someone else's ε and σ. They also have to start
        with a letter: ``read_data`` refuses a bare ``1`` as a type label.
        """
        from molpy import AngleType, AtomType, BondType

        ff, system, masses = _water_ion_box()
        ff.realize(system, masses=masses)
        assert [t.name for t in ff.get_types(AtomType)] == ["t1", "t2", "t3", "t4"]
        assert [t.name for t in ff.get_types(BondType)] == ["b1"]
        assert [t.name for t in ff.get_types(AngleType)] == ["a1"]

    def test_realize_registers_the_assignment_it_is_given(self):
        """One typing per export, computed once and passed down.

        :meth:`~molpot.forcefield.ForceField.realize` and
        :class:`~molix.ff_export.LammpsWriter` both need the same
        :class:`~molpot.forcefield.TypeAssignment`: the include's coefficient
        lines are stated per type and the data file's atoms are labelled by
        type, so two condensations of the same system are two numberings that
        can differ once and produce a deck whose ``pair_coeff`` reads another
        type's ``ε`` / ``σ``. Passing the assignment in is what makes "the same
        one" a fact rather than a convention — the writer already holds one by
        the time it realizes a clone.

        The assignment handed over here is deliberately **not** what this field
        would compute (which is four atom types — see
        :meth:`test_realize_registers_one_native_type_per_class`): one class for
        everything, so a ``realize`` that quietly typed the system again would
        register four types where one was asked for.
        """
        from molpy import AngleType, AtomType, BondType

        ff, system, masses = _water_ion_box()
        one_class = TypeAssignment(
            atom_type=(0,) * 8,
            atom_labels=("t1",),
            atom_charges=(0.0,) * 8,
            bond_type=(0,) * 4,
            bond_labels=("b1",),
            angle_type=(0, 0),
            angle_labels=("a1",),
        )

        ff.realize(system, masses=masses, types=one_class)

        assert [t.name for t in ff.get_types(AtomType)] == ["t1"]
        assert [t.name for t in ff.get_types(BondType)] == ["b1"]
        assert [t.name for t in ff.get_types(AngleType)] == ["a1"]

    def test_the_assignment_declares_exactly_the_types_it_can_realize(self):
        """Its fields are the atom / bond / angle / proper typings, plus charges.

        An improper typing lived here as two more fields that nothing ever
        wrote a coefficient from: :meth:`realize` registers no improper type
        (molpy's include writer takes the harmonic form only), so the labels
        were a numbering with no coefficients behind it — a data file that can
        state ``i1`` on an improper the include never defines, which is the
        shape of a silent zero term. A declaration you cannot realize does not
        belong on the type that *is* the realization's contract; when the
        improper form lands, the fields come back with the coefficients that
        justify them.

        The whole field set is pinned rather than the absence of one name: a
        ``not hasattr`` assertion says nothing about what the type does carry,
        and it is the carried set the writer and molpy both read.
        """
        assert tuple(f.name for f in dataclasses.fields(TypeAssignment)) == (
            "atom_type",
            "atom_labels",
            "atom_charges",
            "bond_type",
            "bond_labels",
            "angle_type",
            "angle_labels",
            "proper_type",
            "proper_labels",
        )

    def test_nonbonded_scaling_is_mirrored_into_molpy_special_bonds(self):
        """The exclusion policy reaches molpy, which is what writes the deck.

        molpy formats the ``special_bonds`` line from its own arrays, so a
        policy that lived only on the Python side would export an unscaled 1–4
        (``1.0``) — the most expensive silent error a Class-I deck can carry.
        """
        opls = NonbondedScaling(
            q=ExclusionScales(0.0, 0.0, 0.5),
            lj=ExclusionScales(0.0, 0.0, 0.5),
        )
        ff = ForceField(name="x", units="real", nonbonded_scaling=opls)
        assert list(ff.special_bonds_lj) == [0.0, 0.0, 0.5]
        assert list(ff.special_bonds_coul) == [0.0, 0.0, 0.5]

        ff.nonbonded_scaling = NonbondedScaling()  # AMBER: q 1–4 = 5/6
        assert ff.special_bonds_coul[2] == pytest.approx(5.0 / 6.0)
        assert ff.special_bonds_lj[2] == 0.5

        clone = ff.clone()
        assert list(clone.special_bonds_lj) == list(ff.special_bonds_lj)
        assert list(clone.special_bonds_coul) == list(ff.special_bonds_coul)

    def test_setting_molpy_special_bonds_updates_the_scaling(self):
        """The mirror runs both ways, so the two spellings cannot drift apart."""
        ff = ForceField(name="x", units="real")  # AMBER default: q 1–4 = 5/6
        ff.set_special_bonds((0.0, 0.0, 0.5), (0.0, 0.0, 0.5))
        assert ff.nonbonded_scaling.triple("q") == (0.0, 0.0, 0.5)
        assert ff.nonbonded_scaling.triple("lj") == (0.0, 0.0, 0.5)

    def test_an_option_the_style_does_not_declare_is_refused(self):
        """A misspelled option is a typo, not a silent "left at the default".

        Options are free-form keyword arguments, so ``mixng="geometric"`` used
        to land in the options dict under a key nothing reads: the resolver saw
        no ``mixing``, applied the Lorentz–Berthelot fallback, and the model ran
        — and was *described* — as a force field the caller never asked for.
        The registry now names each style's vocabulary
        (``PotentialMeta.schema_for``), so the key is checked where it is
        declared. The message carries the offending key and the options there
        are, because a typo is only useful next to the spelling it missed.

        The same rule catches an option offered to the wrong style: ``mixing``
        is a Lennard-Jones combining rule and means nothing to ``pair:coulomb``,
        whose real-space/reciprocal split has no ``ε`` / ``σ`` to combine.
        """
        ff = ForceField(name="x", units="real")
        with pytest.raises(ValueError) as typo:
            ff.def_pairstyle("lj/cut", mixng="geometric")
        assert "mixng" in str(typo.value)
        assert "mixing" in str(typo.value)

        with pytest.raises(ValueError, match="mixing"):
            ForceField(name="y", units="real").def_pairstyle("coulomb", mixing="geometric")

    def test_an_unknown_lj_mixing_rule_is_refused_where_it_is_declared(self):
        """Fail closed at ``def_pairstyle``, not at the first energy evaluation.

        The combining rule is a closed vocabulary
        (:class:`~molpot.potentials.vdw.lj126.MixingRule`). A typo'd rule
        accepted here would either fall back to Lorentz–Berthelot — a silently
        different force field on every cross pair — or raise from inside a
        kernel one backward pass into a training run, far from the declaration
        that caused it. The message names the offending string so the typo is
        the first thing read.
        """
        ff = ForceField(name="x", units="real")
        with pytest.raises(ValueError, match="lorenz-berthelot"):
            ff.def_pairstyle("lj/cut", mixing="lorenz-berthelot")

    def test_re_declaring_a_style_keeps_the_options_it_does_not_restate(self):
        """Naming one option must not drop the others — F24, in one line.

        A product template declares the options that *are* the model: OPLS-AA
        combines both ``ε`` and ``σ`` geometrically, so it declares
        ``pair:lj/cut`` with ``mixing="geometric"``. A caller then re-declares
        the same style to set the cutoff their box needs, which is the ordinary
        way to say "this model, at 12 Å" — and replacing the option dict
        wholesale reverted the combining rule to the Lorentz–Berthelot default
        underneath them. Every cross pair of unequal size is then priced at
        another ``σ`` (``3.5`` with ``2.5`` gives ``3.0`` Å rather than
        ``2.958``), with the manifest and the exported deck agreeing with the
        wrong run rather than with the template.

        Merging is the only reading under which a re-declaration means what it
        says. An option restated explicitly still wins — that is the caller
        overriding the template, which is a decision rather than an accident.
        """
        method = OPLSAAForceField().def_pairstyle("lj/cut", cutoff=12.0).to_ir().nonbonded_method
        assert method.lj_cutoff == 12.0
        assert method.lj_mixing == "geometric"

        overridden = (
            OPLSAAForceField()
            .def_pairstyle("lj/cut", mixing="lorentz-berthelot")
            .to_ir()
            .nonbonded_method
        )
        assert overridden.lj_mixing == "lorentz-berthelot"

    def test_neighbor_list_reaches_the_largest_radius_a_periodic_run_prices(self):
        """The field declares the pair list its own styles are priced against.

        Under a cell OPLS-AA runs Lennard-Jones out to 10 Å and the real-space
        half of its Ewald split out to ``3 × smearing = 15 Å``, so the graph
        has to reach the **larger** of the two: a list built at the
        Lennard-Jones radius drops every Coulomb pair between 10 and 15 Å, and
        the reciprocal half compensates only for what the real-space sum saw.
        Both radii are resolved once, on
        :meth:`~molpot.forcefield.ForceField._nonbonded_method_record`, so the
        field is the only place that can name this number without a second
        reading of the style options — which is what the four fitting scripts
        did with a private ``LJ_CUT_CUTOFF`` constant.

        ``symmetry`` is not a caller's choice: molpot consumes a placed
        ``edges`` namespace and never re-symmetrises a half list, so a list
        handed out here is always full bidirectional.
        """
        nlist = OPLSAAForceField().neighbor_list(pbc=True)
        assert isinstance(nlist, NeighborList)
        assert nlist.cutoff == pytest.approx(15.0)
        assert nlist.symmetry is True
        assert nlist.pbc is True

    def test_neighbor_list_forwards_the_properties_of_the_data(self):
        """Radius is the model's; periodicity and buffer size are the dataset's.

        The same parameters price an isolated molecule and a periodic box, and
        how many pairs the C++ buffer must hold is a property of the systems
        being batched, not of the force field. Forwarding them means a caller
        never rebuilds the list by hand to change one — the path by which a
        script ends up enumerating a graph at a radius its force field does
        not run at.
        """
        nlist = (
            ForceField(name="lj", units="real")
            .def_pairstyle("lj/cut", cutoff=8.0)
            .neighbor_list(pbc=True, max_num_pairs=64)
        )
        assert nlist.cutoff == pytest.approx(8.0)
        assert nlist.pbc is True
        assert nlist.max_num_pairs == 64

    def test_which_coulomb_radius_the_list_covers_is_decided_by_the_boundary(self):
        """Whichever Coulomb term will run is the term the list is sized for.

        One ``pair:coulomb`` declaration is two solvers: without a cell the
        truncated open sum, with one the Ewald split whose real-space half runs
        to ``3 × smearing = 15 Å``. Which of them runs is a property of the
        data, so it is ``pbc`` — not the declaration — that says which radius
        the graph has to cover. The open ``cutoff`` plays no part under a cell:
        sizing the periodic list at 12 Å would truncate the real-space sum
        short of the split the reciprocal half was written against, which is a
        different electrostatics rather than a cheaper one.
        """
        ff = ForceField(name="q", units="real").def_pairstyle("coulomb", cutoff=12.0)
        assert ff.neighbor_list().cutoff == pytest.approx(12.0)
        assert ff.neighbor_list(pbc=True).cutoff == pytest.approx(15.0)

    def test_the_butane_pair_field_places_its_list_at_the_fixture_radius(self):
        """The shared butane-pair box is priced at the radius its field owns (F21).

        ``butane_pair_batch`` places its list through ``neighbor_list(pbc=True)``
        rather than a number of its own, and every symmetry / NVE gate on that
        box reads it. A style change that moved the radius would move those
        goldens with nothing naming why — this pin does.
        """
        neighbors = butane_pair_class_i_forcefield().neighbor_list(pbc=True)
        assert neighbors.cutoff == BUTANE_PAIR_LIST_CUTOFF

    def test_an_open_boundary_coulomb_without_a_cutoff_is_refused(self):
        """An untruncated open Coulomb needs every pair — say so, do not guess.

        OPLS-AA declares ``pair:coulomb`` bare, which is a lattice sum under a
        cell and the *untruncated* open sum without one: at ``pbc=False`` the
        term is the full ``N(N-1)/2`` sum, no finite radius reproduces it, and
        any number this method invented would be a silently different model.
        Fail closed where the list is asked for, naming both ways out — declare
        an open cutoff, or say the system is periodic.
        """
        with pytest.raises(ValueError, match="cutoff"):
            OPLSAAForceField().neighbor_list()

    def test_a_bonded_only_field_has_no_pair_list_to_declare(self):
        """No pair style is nothing to price, not a default radius.

        A bonded-only field's energy is a function of the topology alone; a
        neighbour list built for it would be an empty term's graph, and the
        fallback radius it was built at would be the first thing a caller
        forgot they never chose.
        """
        ff = ForceField(name="b", units="real").def_bondstyle("harmonic")
        with pytest.raises(ValueError, match="pair"):
            ff.neighbor_list()

    def test_an_mlip_pair_term_has_no_radius_this_force_field_can_name(self):
        """The encoder owns its cutoff, and the refusal has to say so.

        An ``nn.Module`` pair term is not a registered style: its radius is the
        ``r_max`` on the encoder's own config, which this container never sees
        and its nonbonded record therefore cannot state. Refusing is right —
        but refusing as *"no pair style declared"* sends the caller looking for
        a missing ``def_pairstyle`` they already wrote, on a field whose whole
        energy is the module they passed to it. The message has to name the
        mlip, so the one place the radius does live is the first thing read.
        """
        ff = ForceField(name="mlip", units="real").def_pairstyle(_StubMlip(), cutoff=5.0)
        with pytest.raises(ValueError, match="(?i)mlip"):
            ff.neighbor_list()

    def test_the_assignment_carries_the_charges_its_types_were_keyed_on(self):
        """The typing and the data file's charge column come from one reading.

        Charge is part of the atom key, so two atoms are one LAMMPS type only
        when they carry the same one; the exporter then has to write *that*
        charge next to *that* type. Carrying the values on the assignment is
        what makes the two impossible to disagree — reading them a second time
        on the writer's side let a field-supplied ``q`` type the atoms while
        the data file's fourth column stayed at the system's (here absent)
        ``atoms.q``, i.e. a deck whose types say one charge and whose atoms
        carry another.
        """
        system = _line_of_atoms(2)  # no ``atoms.q``: the field is the only source
        ff = (
            ForceField(name="q", units="real")
            .def_atomstyle("full")
            .with_params(atoms={"q": torch.tensor([0.3, -0.3], dtype=torch.float64)})
        )
        assert ff.assign_types(system).atom_charges == (0.3, -0.3)

        # No charge source at all: the types were keyed on mass alone and the
        # exported charge column is zero, not absent — LAMMPS ``atom_style
        # full`` has a column there either way.
        bare = ForceField(name="x", units="real").def_atomstyle("full")
        assert bare.assign_types(system).atom_charges == (0.0, 0.0)

        # The system is the second source, read only where the field has no
        # ``q`` table: a batch loaded from disk carries the charges it was
        # stored with, and a field that has not been given any must type — and
        # export — those rather than zeros. This is the same reading the data
        # file's fourth column is written from, which is why it is asserted
        # here on the assignment and not again on the writer.
        from_system = _line_of_atoms(2)
        from_system["atoms", "q"] = torch.tensor([0.2, -0.2], dtype=torch.float64)
        assert bare.assign_types(from_system).atom_charges == (0.2, -0.2)

    def test_clone_carries_the_mlip(self):
        """A copy of an mlip field is still that mlip — dropping it is zero energy.

        ``clone`` rebuilds the declared styles *by name*, and an mlip pair term
        has no name: it is the ``nn.Module`` itself, kept on the container by
        ``def_pairstyle``. Rebuilding therefore reconstructs everything about
        the field except the only term that carries its energy, and nothing
        raises — ``Potentials`` reads the mlip with ``getattr(..., None)``, so
        the copy assembles cleanly with no classical style either and prices
        every configuration at exactly zero. ``to_dtype`` already carries it
        across explicitly; ``clone`` itself must, or every other copy path
        built on it (``with_params``, and so the whole copy-on-write fitting
        loop) silently loses the model.
        """
        mlip = _StubMlip()
        ff = ForceField(name="mlip", units="real").def_pairstyle(mlip, cutoff=5.0)
        assert ff.to_potentials().mlip is mlip

        clone = ff.clone()
        assert clone.to_potentials().mlip is ff.to_potentials().mlip

        # ``with_params`` is ``clone`` plus a table, so it inherits the same
        # loss: the fitted field that comes back out of a training step is the
        # one that would be exported and run.
        refilled = ff.with_params(atoms={"q": torch.zeros(2, dtype=torch.float64)})
        assert refilled.to_potentials().mlip is mlip


class TestOPLSAAForceField:
    def test_default_styles_are_oplsaa_set(self):
        from molpot.forcefield import OPLSAAForceField

        ff = OPLSAAForceField()
        assert isinstance(ff, ForceField)
        assert isinstance(ff, MolpyForceField)
        for key in (
            "bond:harmonic",
            "angle:harmonic",
            "dihedral:opls",
            "improper:periodic",
            "pair:lj/cut",
            "pair:coulomb",
        ):
            assert ff.has_style(key), key

    def test_default_nonbonded_scaling_f14_is_half(self):
        from molpot.forcefield import OPLSAAForceField

        scaling = OPLSAAForceField().nonbonded_scaling
        assert scaling.q.s14 == 0.5
        assert scaling.lj.s14 == 0.5


def _filled_oplsaa() -> OPLSAAForceField:
    """OPLS-AA template with the bond and per-atom nonbonded tables filled.

    Two of the six declared styles carry parameters; the manifest still has to
    describe all six, because a declared style with no table is an
    unparameterised model, not an absent term.
    """
    return OPLSAAForceField().with_params(
        bonds={
            "k": torch.tensor([340.0], dtype=torch.float64),
            "r0": torch.tensor([1.09], dtype=torch.float64),
        },
        atoms={
            "epsilon": torch.tensor([0.066, 0.030], dtype=torch.float64),
            "sigma": torch.tensor([3.50, 2.50], dtype=torch.float64),
            "q": torch.tensor([-0.12, 0.06], dtype=torch.float64),
        },
    )


class TestForceFieldManifest:
    """``ForceField.manifest`` — the S2 manifest's owner.

    The force field is where the styles, the style options, the parameter
    tables and the exclusion scaling all already live, so it fills the
    manifest; ``Potentials`` is only its torch view and owns no second copy.
    ``representation`` and ``provenance`` are left unset on purpose — the
    caller composes them in with ``dataclasses.replace``, because composition
    is the caller's job.
    """

    def test_interactions_are_the_energy_bearing_declared_styles(self):
        """Every declared style but ``atom:*``, which is a record layout."""
        ff = _filled_oplsaa()
        manifest = ff.manifest(profile="A0")
        assert isinstance(manifest, ModelManifest)
        assert [record.style for record in manifest.interactions] == [
            key for key in ff.style_names() if not key.startswith("atom:")
        ]

    def test_nonbonded_record_is_the_one_to_ir_reports(self):
        """One nonbonded-method record, shared with ``PotentialIR`` — not two."""
        ff = _filled_oplsaa()
        assert ff.manifest(profile="A0").nonbonded == ff.to_ir().nonbonded_method

    def test_electrostatics_carries_the_permanent_exclusion_scaling(self):
        """S5: the permanent scaling is model identity, not a runtime detail."""
        ff = _filled_oplsaa()
        electrostatics = ff.manifest(profile="A0").electrostatics
        assert electrostatics is not None
        assert electrostatics.permanent_scaling == ff.nonbonded_scaling

    def test_electrostatics_reports_only_what_the_force_field_owns(self):
        """A ``pair:coulomb`` style is not a charge model, so it constrains nothing.

        The force field owns the multipole order (0 — point charges) and the
        permanent scaling. *How* the charges were obtained is the charge
        model's: ``AtomicChargeModel.electrostatics_record`` names the
        constraint (``"total-charge-kkt"``) and the solve it ran, and the
        caller composes the two with ``dataclasses.replace``. Claiming the KKT
        constraint here describes a solver this container never ran — and
        says it even for a field whose ``atoms.q`` came from a file.
        """
        electrostatics = _filled_oplsaa().manifest(profile="A0").electrostatics
        assert electrostatics is not None
        assert electrostatics.multipole_order == 0
        assert electrostatics.charge_constraint == "none"
        assert electrostatics.solver is None

    def test_a_bonded_only_field_still_carries_both_nonbonded_records(self):
        """Records are never ``None``: a consumer branches on a field, not presence.

        A field with no pair style at all gets an all-``None``
        ``NonbondedMethodRecord`` and an electrostatics record all the same,
        so reading ``manifest.nonbonded.lj_cutoff`` is always legal.
        """
        manifest = (
            ForceField(name="bonded", units="real").def_bondstyle("harmonic").manifest(profile="A0")
        )
        nonbonded = manifest.nonbonded
        assert nonbonded is not None
        assert [getattr(nonbonded, f.name) for f in dataclasses.fields(nonbonded)] == [None] * len(
            dataclasses.fields(nonbonded)
        )
        assert manifest.electrostatics is not None

    def test_profile_must_be_one_this_molnex_knows(self):
        """``A1`` is not a spelling choice — the polarization records do not exist.

        Accepted verbatim, the string would label a manifest as a model
        profile whose induced scaling and response solver this schema has no
        fields for, and the mislabelling would only surface in whatever reads
        it back. The refusal names the profile asked for and the ones there
        are.
        """
        ff = _filled_oplsaa()
        with pytest.raises(ValueError) as excinfo:
            ff.manifest(profile="A1")
        assert "A1" in str(excinfo.value)
        assert "A0" in str(excinfo.value)
        ff.manifest(profile="A0")

    def test_numerics_is_the_runtime_record_rather_than_a_second_reading(self):
        """One reader of the global precision, and the force field defers to it.

        What ``NumericsRecord.from_runtime`` reports is pinned in
        ``tests/test_molpot/test_manifest.py``; this pins only that the
        container does not take its own reading of ``molix.config["ftype"]``
        and ``torch.get_float32_matmul_precision()``.
        """
        assert _filled_oplsaa().manifest(profile="A0").numerics == NumericsRecord.from_runtime()

    def test_tables_content_hash_the_filled_parameter_tables(self):
        """Each filled category is hashed by ``ParamTableRecord``, not re-derived."""
        ff = _filled_oplsaa()
        tables = {record.category: record for record in ff.manifest(profile="A0").tables}
        assert set(tables) == {"bonds", "atoms"}
        bonds = ff.params("bonds")
        assert bonds is not None  # filled by ``_filled_oplsaa``; narrows ``dict | None``
        assert tables["bonds"] == ParamTableRecord.from_tables("bonds", bonds)

    def test_profile_is_recorded_and_composition_is_left_to_the_caller(self):
        """No façade fills representation / provenance; the caller replaces them."""
        manifest = _filled_oplsaa().manifest(profile="A0")
        assert manifest.profile == "A0"
        assert manifest.representation is None
        assert manifest.provenance is None
