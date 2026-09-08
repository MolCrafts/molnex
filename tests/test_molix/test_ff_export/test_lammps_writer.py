"""LammpsWriter — molpy.io data/settings from ForceField + TensorDict."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from molix.batch import TensorFrame, collate_molecules
from molix.ff_export import LammpsAdapter, LammpsWriter
from molpot.forcefield import ForceField
from molpot.ir import ExclusionScales, NonbondedScaling
from tests.conftest import frame


def _two_atom_system(*, box: np.ndarray | None = None) -> tuple[ForceField, TensorDict]:
    """One bonded C-C pair through the collate; *box* (rows, Å) makes it periodic."""
    # OPLS proper table without improper/pair (LAMMPS *.ff writer: harmonic improper only)
    ff = (
        ForceField(name="OPLS", units="real")
        .def_atomstyle("full")
        .def_bondstyle("harmonic")
        .def_anglestyle("harmonic")
        .def_dihedralstyle("opls")
        .with_params(
            bonds={
                "k": torch.tensor([200.0], dtype=torch.float64),
                "r0": torch.tensor([1.53], dtype=torch.float64),
            },
            propers={
                "k1": torch.tensor([1.0], dtype=torch.float64),
                "k2": torch.tensor([2.0], dtype=torch.float64),
                "k3": torch.tensor([0.0], dtype=torch.float64),
                "k4": torch.tensor([0.0], dtype=torch.float64),
            },
        )
    )
    # One bonded molecule at rest, put through the only Frame -> batch
    # conversion there is: its ``bonds`` therefore reach the writer as the
    # ``(2, N)`` COO block ``collate_molecules`` writes, not as ``atomi`` /
    # ``atomj`` columns a fixture spelled by hand.
    system = collate_molecules(
        [frame(Z=(6, 6), pos=((0.0, 0.0, 0.0), (1.53, 0.0, 0.0)), bonds=([0], [1]), box=box)]
    )
    return ff, system


def _nonbonded_forcefield(
    *,
    switch_start: float | None = 10.0,
    mixing: str | None = None,
) -> ForceField:
    """Two-type field declaring LJ (optionally switched) + PME Coulomb pair styles.

    Carries the OPLS 1–4 convention (0.5 for both classes) rather than the
    container's AMBER default, so the ``special_bonds`` golden below reads as
    the force field it names. ``mixing`` is left undeclared by default, so the
    field mixes Lorentz–Berthelot; passing ``"geometric"`` changes exactly one
    thing about the model and therefore about the deck.
    """
    lj_options: dict[str, float | str] = {"cutoff": 12.0}
    if switch_start is not None:
        lj_options["switch_start"] = switch_start
    if mixing is not None:
        lj_options["mixing"] = mixing
    return (
        ForceField(
            name="OPLS",
            units="real",
            nonbonded_scaling=NonbondedScaling(
                q=ExclusionScales(0.0, 0.0, 0.5),
                lj=ExclusionScales(0.0, 0.0, 0.5),
            ),
        )
        .def_atomstyle("full")
        .def_bondstyle("harmonic")
        .def_pairstyle("lj/cut", **lj_options)
        .def_pairstyle("coulomb", calculator="pme")
        .with_params(
            bonds={
                "k": torch.tensor([200.0], dtype=torch.float64),
                "r0": torch.tensor([1.53], dtype=torch.float64),
            },
            atoms={
                "q": torch.tensor([0.5, -0.5], dtype=torch.float64),
                "epsilon": torch.tensor([0.066, 0.03], dtype=torch.float64),
                "sigma": torch.tensor([3.5, 2.5], dtype=torch.float64),
            },
        )
    )


def _four_atom_nonbonded(
    epsilon: tuple[float, float, float, float] = (0.066, 0.03, 0.066, 0.05),
) -> tuple[ForceField, TensorDict]:
    """Four atoms in a row, the **first and third identical** in every parameter.

    Their ``(mass, q, ε, σ)`` keys coincide — nothing else about them does — so
    the box carries three atom types over four atoms, which is what separates
    typing by parameter equality from the identity typing it replaces. Masses
    are the writer's default (12.011 amu for every atom), so the whole
    distinction rests on the tables declared here.

    Args:
        epsilon: The per-atom Lennard-Jones well depths (kcal/mol). The default
            keeps atoms 1 and 3 identical; a caller passing an ``ε`` that
            differs between them builds a force field whose parameters
            contradict that typing.

    Returns:
        The filled force field and the system.
    """
    ff = (
        ForceField(
            name="OPLS",
            units="real",
            nonbonded_scaling=NonbondedScaling(
                q=ExclusionScales(0.0, 0.0, 0.5),
                lj=ExclusionScales(0.0, 0.0, 0.5),
            ),
        )
        .def_atomstyle("full")
        .def_pairstyle("lj/cut", cutoff=12.0, switch_start=10.0)
        .def_pairstyle("coulomb", calculator="pme")
        .with_params(
            atoms={
                "q": torch.tensor([0.5, -0.5, 0.5, -0.25], dtype=torch.float64),
                "epsilon": torch.tensor(epsilon, dtype=torch.float64),
                "sigma": torch.tensor([3.5, 2.5, 3.5, 3.0], dtype=torch.float64),
            },
        )
    )
    atoms = ff.params("atoms")
    assert atoms is not None  # just filled above; narrows the ``dict | None``
    built = frame(
        Z=(6, 6, 6, 6),
        pos=[[3.0 * i, 0.0, 0.0] for i in range(4)],
    )
    # ``frame()`` takes no per-atom extras, so the charge column is written onto
    # the block it built; the collate carries it through as ``atoms.q``.
    built["atoms"]["q"] = atoms["q"].detach().cpu().double().numpy()
    return ff, collate_molecules([built])


def _charges_only_forcefield() -> ForceField:
    """Coulomb-only field: charges under PME, no Lennard-Jones table at all."""
    return (
        ForceField(
            name="OPLS",
            units="real",
            nonbonded_scaling=NonbondedScaling(
                q=ExclusionScales(0.0, 0.0, 0.5),
                lj=ExclusionScales(0.0, 0.0, 0.5),
            ),
        )
        .def_atomstyle("full")
        .def_pairstyle("coulomb", calculator="pme")
        .with_params(atoms={"q": torch.tensor([0.5, -0.5], dtype=torch.float64)})
    )


class TestLammpsWriter:
    def test_write_settings_bond_coeff(self, tmp_path):
        ff, system = _two_atom_system()
        types = ff.assign_types(system)
        text = LammpsWriter().write_settings(tmp_path / "s.ff", ff, system, types=types).read_text()
        assert "bond_style harmonic" in text
        assert "bond_coeff" in text
        assert "100" in text  # K = k/2

    def test_write_data_carries_the_topology_counts(self, tmp_path):
        ff, system = _two_atom_system()
        types = ff.assign_types(system)
        text = LammpsWriter().write_data(tmp_path / "s.data", system, types=types).read_text()
        assert "2 atoms" in text
        assert "1 bonds" in text
        assert "Atoms" in text

    def test_write_data_charges_are_the_ones_the_types_were_keyed_on(self, tmp_path):
        """One charge reading types the atoms and fills the data file's column.

        The field's own ``atoms.q`` table is what
        :meth:`~molpot.forcefield.ForceField.assign_types` keys the atom types
        on — here it is the only charge source, since the system carries no
        ``atoms.q`` — so it is also what the ``Atoms`` section has to state.
        The writer used to read the charges again from the system and wrote
        zeros whenever the field carried the table instead: a deck whose two
        types exist *because* the atoms differ in charge, and whose atoms are
        then both neutral. LAMMPS accepts that file and runs a model with no
        electrostatics in it.
        """
        system = collate_molecules([frame(Z=(6, 6), pos=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))])
        ff = (
            ForceField(name="q", units="real")
            .def_atomstyle("full")
            .with_params(atoms={"q": torch.tensor([0.3, -0.3], dtype=torch.float64)})
        )
        types = ff.assign_types(system)
        text = LammpsWriter().write_data(tmp_path / "s.data", system, types=types).read_text()
        # ``Atoms # full`` rows are ``id mol type q x y z``.
        rows = [
            line.split()
            for line in text.split("Atoms # full", 1)[1].splitlines()
            if line.split() and line.split()[0].isdigit()
        ]
        assert [float(row[3]) for row in rows] == pytest.approx([0.3, -0.3], abs=1e-12)

    def test_write_data_requires_atoms_pos(self, tmp_path):
        # The assignment is built from the *placed* two-atom box: what is being
        # refused here is a system with no coordinates, not one with no types.
        ff, placed = _two_atom_system()
        types = ff.assign_types(placed)
        system = TensorDict(
            {"atoms": TensorDict({"Z": torch.tensor([6, 6])}, batch_size=[2])},
            batch_size=[],
        )
        with pytest.raises(KeyError):
            LammpsWriter().write_data(tmp_path / "s.data", system, types=types)

    def test_write_settings_can_skip_pair_styles(self, tmp_path):
        """The bonded include drops every pair line so the pair block owns them."""
        _, system = _two_atom_system()
        ff = _nonbonded_forcefield()
        text = (
            LammpsWriter()
            .write_settings(
                tmp_path / "s.ff",
                ff,
                system,
                types=ff.assign_types(system),
                skip_pair_style=True,
            )
            .read_text()
        )
        assert not [
            line
            for line in text.splitlines()
            if line.startswith(("pair_style", "pair_coeff", "special_bonds"))
        ]
        assert "bond_coeff" in text

    def test_write_pair_block_hybrid_overlay(self, tmp_path):
        """The writer composes the block; every style string is the adapter's.

        Its own contribution is the coefficient table: every ``i <= j`` pair
        stated explicitly, mixed by the rule the ``pair`` record names — here
        the Lorentz–Berthelot this field leaves undeclared, so ``σ_12`` is the
        arithmetic ``½(3.5 + 2.5) = 3``. LAMMPS never mixes an LJ sub-style onto
        an ``i != j`` pair under ``hybrid/overlay`` (the cross term measures
        ``E_vdwl = 0``), and its own default mixing is geometric in ``sigma``
        where this model takes the arithmetic mean — two independent reasons
        the lines cannot be left to the engine.
        """
        ff = _nonbonded_forcefield()
        _, system = _two_atom_system()
        spec = LammpsAdapter().translate(ff.to_ir())
        record = {f["category"]: f for f in spec.forces if "category" in f}
        path = LammpsWriter().write_pair_block(
            tmp_path / "pair.in", spec, types=ff.assign_types(system)
        )
        assert path == tmp_path / "pair.in"
        lines = path.read_text().splitlines()

        assert lines[0] == record["pair"]["type"]
        assert lines[1:4] == [
            "pair_coeff 1 1 lj/mdf 0.066 3.5",
            "pair_coeff 1 2 lj/mdf 0.04449719 3",
            "pair_coeff 2 2 lj/mdf 0.03 2.5",
        ]
        assert lines[4] == "pair_coeff * * coul/long"
        assert lines[5] == record["kspace"]["type"]
        assert lines[6] == record["kspace_modify"]["type"]
        assert lines[7] == record["special_bonds"]["type"]
        assert len(lines) == 8

    def test_write_pair_block_mixes_by_the_declared_geometric_rule(self, tmp_path):
        """A model declaring OPLS-AA mixing exports OPLS-AA cross terms.

        Only the ``1 2`` line can move — the two diagonal lines mix a type with
        itself, where every rule agrees — and it moves to the geometric
        ``σ_12``.

        **Hard-coded golden**, seven significant digits because
        ``LammpsAdapter.number`` renders the deck at ``{:.7g}`` and the deck is
        parsed as text::

            python - <<'EOF'
            import math
            print(f"{math.sqrt(0.066 * 0.03):.7g}", f"{math.sqrt(3.5 * 2.5):.7g}")
            EOF
            0.04449719 2.95804

        Same ``ε_12`` as the Lorentz–Berthelot golden above (both rules take the
        geometric mean of ``ε``), so the difference is attributable to ``σ``
        alone.
        """
        ff = _nonbonded_forcefield(mixing="geometric")
        _, system = _two_atom_system()
        lines = (
            LammpsWriter()
            .write_pair_block(
                tmp_path / "pair.in",
                LammpsAdapter().translate(ff.to_ir()),
                types=ff.assign_types(system),
            )
            .read_text()
            .splitlines()
        )
        assert lines[1:4] == [
            "pair_coeff 1 1 lj/mdf 0.066 3.5",
            "pair_coeff 1 2 lj/mdf 0.04449719 2.95804",
            "pair_coeff 2 2 lj/mdf 0.03 2.5",
        ]

    def test_write_pair_block_single_style_has_untagged_coeffs(self, tmp_path):
        """One fused style: no sub-style token on any coeff line, and no ``* *``.

        ``pair_coeff * *`` exists only to hand every type pair to the Coulomb
        sub-style of a ``hybrid/overlay``; a fused ``lj/cut/coul/long`` needs
        parameters on every line it takes, so the bare form would be rejected.
        """
        ff = _nonbonded_forcefield(switch_start=None)
        _, system = _two_atom_system()
        lines = (
            LammpsWriter()
            .write_pair_block(
                tmp_path / "pair.in",
                LammpsAdapter().translate(ff.to_ir()),
                types=ff.assign_types(system),
            )
            .read_text()
            .splitlines()
        )
        coeffs = [line.split() for line in lines if line.startswith("pair_coeff")]

        assert [tokens[1:3] for tokens in coeffs] == [["1", "1"], ["1", "2"], ["2", "2"]]
        assert all(len(tokens) == 5 for tokens in coeffs)  # i j eps sigma — no token
        assert len(lines) == 7

    def test_write_pair_block_charges_only_maps_every_pair(self, tmp_path):
        """No LJ table still needs one coeff line, or no pair is assigned at all."""
        ff = _charges_only_forcefield()
        _, system = _two_atom_system()
        lines = (
            LammpsWriter()
            .write_pair_block(
                tmp_path / "pair.in",
                LammpsAdapter().translate(ff.to_ir()),
                types=ff.assign_types(system),
            )
            .read_text()
            .splitlines()
        )
        assert [line for line in lines if line.startswith("pair_coeff")] == ["pair_coeff * *"]

    def test_write_data_uses_graphs_cell_as_the_box(self, tmp_path):
        """A periodic export carries ``graphs.cell``, not the padded open bounds.

        A ``pair_style … coul/long`` + ``kspace_style pppm`` run is only
        meaningful under the cell the model was evaluated in; writing the
        open-boundary padded bounds instead would silently change the lattice
        sum LAMMPS performs.
        """
        ff, periodic = _two_atom_system(box=np.eye(3) * 12.0)
        types = ff.assign_types(periodic)
        text = LammpsWriter().write_data(tmp_path / "cell.data", periodic, types=types).read_text()
        assert "0 12 xlo xhi" in text
        assert "0 12 ylo yhi" in text
        assert "0 12 zlo zhi" in text

        # No box on the frame → no graphs.cell → the open box pads the bounds by box_pad_A.
        _, system = _two_atom_system()
        open_text = (
            LammpsWriter().write_data(tmp_path / "open.data", system, types=types).read_text()
        )
        assert "-20 21.53 xlo xhi" in open_text
        assert "-20 20 ylo yhi" in open_text

    def test_write_settings_omits_the_units_line(self, tmp_path):
        """The include is sourced after ``read_data``; LAMMPS refuses ``units`` then."""
        ff, system = _two_atom_system()
        text = (
            LammpsWriter()
            .write_settings(tmp_path / "s.ff", ff, system, types=ff.assign_types(system))
            .read_text()
        )
        assert not [line for line in text.splitlines() if line.startswith("units")]

    def test_write_data_labels_types_so_lammps_accepts_them(self, tmp_path):
        """Type labels are ``t1`` / ``b1``: a LAMMPS label may not start with a digit.

        This field declares no per-atom table and the system carries no charge,
        so both atoms key on the default mass alone and the box has **one**
        atom type — and its single bond one bond type, named for the bond class
        rather than for the atoms it joins.
        """
        ff, system = _two_atom_system()
        text = (
            LammpsWriter()
            .write_data(tmp_path / "s.data", system, types=ff.assign_types(system))
            .read_text()
        )
        assert "Atom Type Labels\n\n1 t1\n" in text
        assert "Bond Type Labels\n\n1 b1\n" in text
        assert "Atoms # full" in text

    def test_write_data_gives_atoms_with_equal_parameters_one_type(self, tmp_path):
        """Four atoms, three types — and the Atoms table says which is which.

        The data file is where the typing becomes LAMMPS's: the label section
        declares the types and the third column of each ``Atoms`` row assigns
        one. Atoms 1 and 3 carry the same ``(mass, q, ε, σ)``, so they share a
        type id here — and the ``pair_coeff`` lines of the block written beside
        this file are indexed by exactly these ids, which is why the two are
        pinned against the same assignment rather than each deriving its own.
        """
        ff, system = _four_atom_nonbonded()
        text = (
            LammpsWriter()
            .write_data(tmp_path / "s.data", system, types=ff.assign_types(system))
            .read_text()
        )
        assert "3 atom types" in text
        labels = text.split("Atom Type Labels\n\n", 1)[1].split("\n\n", 1)[0].splitlines()
        assert labels == ["1 t1", "2 t2", "3 t3"]
        rows = text.split("Atoms # full\n\n", 1)[1].strip().splitlines()
        assert [row.split()[2] for row in rows] == ["1", "2", "1", "3"]

    def test_write_pair_block_emits_one_line_per_type_pair(self, tmp_path):
        """``T(T+1)/2`` lines for ``T`` types, not ``N(N+1)/2`` for ``N`` atoms.

        Four atoms in three types give six ``pair_coeff`` lines where identity
        typing gave ten; at a thousand atoms that is the difference between a
        few hundred lines and half a million (ledger **F25**). The collapse has
        to pick each type's own coefficients — the diagonal lines below are the
        four atoms' ``(ε, σ)`` seen through three types, where any mixing rule
        is the identity, so what they measure is the row selection and not the
        combining rule the hybrid golden above pins.
        """
        ff, system = _four_atom_nonbonded()
        lines = (
            LammpsWriter()
            .write_pair_block(
                tmp_path / "pair.in",
                LammpsAdapter().translate(ff.to_ir()),
                types=ff.assign_types(system),
            )
            .read_text()
            .splitlines()
        )
        coeffs = [
            line.split() for line in lines if line.startswith("pair_coeff") and "*" not in line
        ]
        assert [(t[1], t[2]) for t in coeffs] == [
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("2", "2"),
            ("2", "3"),
            ("3", "3"),
        ]
        assert [" ".join(t) for t in coeffs if t[1] == t[2]] == [
            "pair_coeff 1 1 lj/mdf 0.066 3.5",
            "pair_coeff 2 2 lj/mdf 0.03 2.5",
            "pair_coeff 3 3 lj/mdf 0.05 3",
        ]

    def test_write_pair_block_refuses_a_type_its_parameters_disagree_on(self, tmp_path):
        """Two atoms of one type with different ``ε`` is a contradiction, not a choice.

        Collapsing the per-atom rows by type means one of them supplies the
        line; if the rows disagree, whichever the writer happened to read first
        would silently become the model for the other atom too. That can only
        arise from an assignment and an IR built from different force fields —
        exactly the mistake worth failing on, since nothing downstream can see
        it: the deck stays well-formed and LAMMPS runs it.
        """
        ff, system = _four_atom_nonbonded()
        drifted, _ = _four_atom_nonbonded(epsilon=(0.066, 0.03, 0.077, 0.05))
        with pytest.raises(ValueError):
            LammpsWriter().write_pair_block(
                tmp_path / "pair.in",
                LammpsAdapter().translate(drifted.to_ir()),
                types=ff.assign_types(system),
            )

    def test_write_settings_special_bonds_follow_the_scaling(self, tmp_path):
        """The include's exclusion line is the field's policy, not molpy's default.

        ``_two_atom_system`` carries the container default (AMBER: ``lj`` 1–4
        ``0.5``, ``coul`` 1–4 ``5/6``); molpy's own default is an unscaled
        ``1.0`` on both, which would run every 1–4 pair at full strength.
        """
        ff, system = _two_atom_system()
        text = (
            LammpsWriter()
            .write_settings(tmp_path / "s.ff", ff, system, types=ff.assign_types(system))
            .read_text()
        )
        assert "special_bonds lj 0.000000 0.000000 0.500000 coul 0.000000 0.000000 0.833333" in text

    def test_write_data_reads_a_typed_view_like_the_plain_batch(self, tmp_path):
        """``TensorFrame`` in, the same deck out — byte for byte.

        The writer reads the post-collate batch: ``atoms.pos``, the bonded
        namespaces and ``graphs.cell``. Wrapping that batch in its typed view
        (CLAUDE.md § TensorDict Contract) replaces no storage — ``pos`` is the
        same tensor — so the only thing a difference in the text could mean is
        that some reader took a different branch on the container's *type*
        rather than on its content.
        """
        ff, system = _four_atom_nonbonded()
        types = ff.assign_types(system)
        view = TensorFrame.from_tensordict(system)

        plain = LammpsWriter().write_data(tmp_path / "plain.data", system, types=types).read_text()
        viewed = LammpsWriter().write_data(tmp_path / "view.data", view, types=types).read_text()

        assert viewed == plain
