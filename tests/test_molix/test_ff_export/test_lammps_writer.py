"""LammpsWriter — molpy.io data/settings from ForceField + TensorDict."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.ff_export import LammpsAdapter, LammpsWriter
from molpot.forcefield import ForceField
from molpot.ir import ExclusionScales, NonbondedScaling


def _two_atom_system() -> tuple[ForceField, TensorDict]:
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
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.53, 0.0, 0.0]], dtype=torch.float64)
    system = TensorDict(
        {
            "atoms": TensorDict({"pos": pos}, batch_size=[2]),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )
    return ff, system


def _nonbonded_forcefield(*, switch_start: float | None = 10.0) -> ForceField:
    """Two-type field declaring LJ (optionally switched) + PME Coulomb pair styles.

    Carries the OPLS 1–4 convention (0.5 for both classes) rather than the
    container's AMBER default, so the ``special_bonds`` golden below reads as
    the force field it names.
    """
    lj_options: dict[str, float] = {"cutoff": 12.0}
    if switch_start is not None:
        lj_options["switch_start"] = switch_start
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
        text = LammpsWriter().write_settings(tmp_path / "s.ff", ff, system).read_text()
        assert "bond_style harmonic" in text
        assert "bond_coeff" in text
        assert "100" in text  # K = k/2

    def test_write_data_identity_types(self, tmp_path):
        _, system = _two_atom_system()
        text = LammpsWriter().write_data(tmp_path / "s.data", system).read_text()
        assert "2 atoms" in text
        assert "1 bonds" in text
        assert "Atoms" in text

    def test_write_data_reads_atoms_q(self, tmp_path):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        system = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": pos,
                        "q": torch.tensor([0.2, -0.2], dtype=torch.float64),
                    },
                    batch_size=[2],
                ),
            },
            batch_size=[],
        )
        text = LammpsWriter().write_data(tmp_path / "s.data", system).read_text()
        assert "0.2" in text
        assert "-0.2" in text

    def test_write_data_requires_atoms_pos(self, tmp_path):
        system = TensorDict(
            {"atoms": TensorDict({"Z": torch.tensor([6, 6])}, batch_size=[2])},
            batch_size=[],
        )
        with pytest.raises(KeyError):
            LammpsWriter().write_data(tmp_path / "s.data", system)

    def test_write_settings_can_skip_pair_styles(self, tmp_path):
        """The bonded include drops every pair line so the pair block owns them."""
        _, system = _two_atom_system()
        text = (
            LammpsWriter()
            .write_settings(
                tmp_path / "s.ff", _nonbonded_forcefield(), system, skip_pair_style=True
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
        stated explicitly, Lorentz–Berthelot mixed by
        :func:`~molpot.potentials.vdw.lj126.lorentz_berthelot`. LAMMPS never
        mixes an LJ sub-style onto an ``i != j`` pair under ``hybrid/overlay``
        (the cross term measures ``E_vdwl = 0``), and its own default mixing is
        geometric in ``sigma`` where molnex takes the arithmetic mean.
        """
        spec = LammpsAdapter().translate(_nonbonded_forcefield().to_ir())
        record = {f["category"]: f for f in spec.forces if "category" in f}
        path = LammpsWriter().write_pair_block(tmp_path / "pair.in", spec)
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

    def test_write_pair_block_single_style_has_untagged_coeffs(self, tmp_path):
        """One fused style: no sub-style token on any coeff line, and no ``* *``.

        ``pair_coeff * *`` exists only to hand every type pair to the Coulomb
        sub-style of a ``hybrid/overlay``; a fused ``lj/cut/coul/long`` needs
        parameters on every line it takes, so the bare form would be rejected.
        """
        spec = LammpsAdapter().translate(_nonbonded_forcefield(switch_start=None).to_ir())
        lines = LammpsWriter().write_pair_block(tmp_path / "pair.in", spec).read_text().splitlines()
        coeffs = [line.split() for line in lines if line.startswith("pair_coeff")]

        assert [tokens[1:3] for tokens in coeffs] == [["1", "1"], ["1", "2"], ["2", "2"]]
        assert all(len(tokens) == 5 for tokens in coeffs)  # i j eps sigma — no token
        assert len(lines) == 7

    def test_write_pair_block_charges_only_maps_every_pair(self, tmp_path):
        """No LJ table still needs one coeff line, or no pair is assigned at all."""
        spec = LammpsAdapter().translate(_charges_only_forcefield().to_ir())
        lines = LammpsWriter().write_pair_block(tmp_path / "pair.in", spec).read_text().splitlines()
        assert [line for line in lines if line.startswith("pair_coeff")] == ["pair_coeff * *"]

    def test_write_data_uses_graphs_cell_as_the_box(self, tmp_path):
        """A periodic export carries ``graphs.cell``, not the padded open bounds.

        A ``pair_style … coul/long`` + ``kspace_style pppm`` run is only
        meaningful under the cell the model was evaluated in; writing the
        open-boundary padded bounds instead would silently change the lattice
        sum LAMMPS performs.
        """
        _, system = _two_atom_system()
        periodic = system.clone()
        periodic["graphs"] = TensorDict(
            {"cell": torch.eye(3, dtype=torch.float64).unsqueeze(0) * 12.0}, batch_size=[1]
        )
        text = LammpsWriter().write_data(tmp_path / "cell.data", periodic).read_text()
        assert "0 12 xlo xhi" in text
        assert "0 12 ylo yhi" in text
        assert "0 12 zlo zhi" in text

        # No cell declared → the open box still pads the bounds by box_pad_A.
        open_text = LammpsWriter().write_data(tmp_path / "open.data", system).read_text()
        assert "-20 21.53 xlo xhi" in open_text
        assert "-20 20 ylo yhi" in open_text

    def test_write_settings_omits_the_units_line(self, tmp_path):
        """The include is sourced after ``read_data``; LAMMPS refuses ``units`` then."""
        ff, system = _two_atom_system()
        text = LammpsWriter().write_settings(tmp_path / "s.ff", ff, system).read_text()
        assert not [line for line in text.splitlines() if line.startswith("units")]

    def test_write_data_labels_types_so_lammps_accepts_them(self, tmp_path):
        """Type labels are ``t1``-style: a LAMMPS label may not start with a digit."""
        _, system = _two_atom_system()
        text = LammpsWriter().write_data(tmp_path / "s.data", system).read_text()
        assert "Atom Type Labels\n\n1 t1\n" in text
        assert "Bond Type Labels\n\n1 t1-t2\n" in text
        assert "Atoms # full" in text

    def test_write_settings_special_bonds_follow_the_scaling(self, tmp_path):
        """The include's exclusion line is the field's policy, not molpy's default.

        ``_two_atom_system`` carries the container default (AMBER: ``lj`` 1–4
        ``0.5``, ``coul`` 1–4 ``5/6``); molpy's own default is an unscaled
        ``1.0`` on both, which would run every 1–4 pair at full strength.
        """
        ff, system = _two_atom_system()
        text = LammpsWriter().write_settings(tmp_path / "s.ff", ff, system).read_text()
        assert "special_bonds lj 0.000000 0.000000 0.500000 coul 0.000000 0.000000 0.833333" in text
