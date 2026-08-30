"""Tests for ``molix.engine.export`` — argument validation + the meta block.

``export_for_lammps`` AOT-compiles a model into ``{name}.pt2`` and stamps a
sibling ``{name}.meta.json`` whose ``lammps`` block is the contract a
``pair_style molnex`` C++ run reads. The force-export path is refused on CPU
(``molix.export.Exporter``), so the block assembly itself is unit-tested
through the private pure function ``_lammps_meta_block`` — that is the only
way the schema contract can be covered on a CPU CI box.

-- the three ``export_for_lammps`` cases migrated verbatim from
   ``tests/test_molix/test_engine.py`` --
"""

from __future__ import annotations

import json

import pytest
import torch

from molix.engine import export_for_lammps
from molix.export import Exporter
from tests.test_molix.test_engine.conftest import _FlatEFPotential


def test_export_for_lammps_validates_inputs(tmp_path):
    with pytest.raises(ValueError, match="species"):
        export_for_lammps(_FlatEFPotential(), tmp_path / "x", species=[], cutoff=4.5)
    with pytest.raises(ValueError, match="units"):
        export_for_lammps(_FlatEFPotential(), tmp_path / "x", species=[1], cutoff=4.5, units="lj")


def test_export_for_lammps_cpu_refused(tmp_path):
    """CPU force export is disabled (torch 2.12 miscompiles the backward scatter)."""
    with pytest.raises(RuntimeError, match="disabled on CPU"):
        export_for_lammps(
            _FlatEFPotential(),
            tmp_path / "x",
            species=[1, 6, 8],
            cutoff=4.5,
            units="real",
            adapter="flat",
            device="cpu",
        )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="force export requires CUDA")
def test_export_for_lammps_roundtrip(tmp_path):
    """AOTI export writes the lammps meta block and a dynamic-shape ``.pt2``."""
    outdir = export_for_lammps(
        _FlatEFPotential(),
        tmp_path / "flat_export",
        species=[8, 1, 6],  # deliberately unsorted → stored sorted
        cutoff=4.5,
        units="real",
        adapter="flat",
        device="cuda",
    )
    meta = json.loads((outdir / "model.meta.json").read_text())
    assert (outdir / "model.pt2").exists()
    lm = meta["lammps"]
    assert lm["cutoff"] == 4.5
    assert lm["units"] == "real"
    assert lm["species"] == [1, 6, 8]
    assert lm["adapter"] == "flat"
    assert lm["inputs"] == ["Z", "pos", "edge_index"]
    assert lm["outputs"] == ["energy", "forces"]

    # the exported .pt2 serves N/E different from the trace (MD needs this)
    runner = Exporter.load(outdir / "model.pt2")
    Z = torch.tensor([1, 6, 8, 1], device="cuda")
    pos = torch.randn(4, 3, device="cuda")
    ei = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], device="cuda")
    e, f = runner(Z, pos, ei)
    assert f.shape == (4, 3)


class TestLammpsMetaBlock:
    """``molix.engine.export._lammps_meta_block`` — schema-v2 block assembly.

    Pure dict assembly, no AOTI export, so the ``lammps`` meta contract is
    covered on CPU (the surrounding export path is CPU-refused, see
    ``test_export_for_lammps_cpu_refused``). Field set and semantics are the
    spec ``lammps-iface-01-meta`` §"meta.json v2 字段" list.

    Signature this class calls (for the implementer)::

        _lammps_meta_block(
            *,
            species: list[int],
            cutoff: float,
            units: str,
            adapter_name: str,
            model_dtype: str,
            supports_pbc: bool,
            supports_virial: bool,
            cuda_graph: bool,
            n_atoms: int | None = None,
            e_max: int | None = None,
        ) -> dict

    ``contract`` / ``execution`` are *derived* (not parameters):
    ``contract == "cutoff-graph"`` for both current export paths, and
    ``execution`` is ``"cuda-graph"`` when ``cuda_graph`` else ``"dynamic"``.
    The private symbol is imported inside each test so only this class is red
    while it is missing.
    """

    #: Every key the v2 block must carry, dynamic-execution branch.
    BASE_KEYS = {
        "schema_version",
        "contract",
        "execution",
        "cutoff",
        "units",
        "energy_unit",
        "length_unit",
        "energy_conv",
        "species",
        "model_dtype",
        "inputs",
        "outputs",
        "adapter",
        "supports_pbc",
        "supports_virial",
    }

    @staticmethod
    def block(**overrides: object) -> dict:
        """Assemble a block with sane defaults; ``overrides`` replace kwargs."""
        from molix.engine.export import _lammps_meta_block

        kwargs: dict[str, object] = {
            "species": [8, 1, 6],
            "cutoff": 4.5,
            "units": "real",
            "adapter_name": "molnex-tensordict",
            "model_dtype": "float32",
            "supports_pbc": False,
            "supports_virial": False,
            "cuda_graph": False,
        }
        kwargs.update(overrides)
        return _lammps_meta_block(**kwargs)

    def test_dynamic_key_set_is_exact(self):
        assert set(self.block().keys()) == self.BASE_KEYS

    def test_cuda_graph_key_set_adds_fixed_shapes(self):
        block = self.block(cuda_graph=True, n_atoms=4, e_max=8)
        assert set(block.keys()) == self.BASE_KEYS | {"n_atoms", "e_max"}

    def test_cuda_graph_discriminator_key_is_gone(self):
        """``cuda_graph`` the boolean was renamed to ``execution`` in v2."""
        assert "cuda_graph" not in self.block()
        assert "cuda_graph" not in self.block(cuda_graph=True, n_atoms=4, e_max=8)

    def test_schema_version_is_two(self):
        from molix.engine.export import LAMMPS_META_SCHEMA

        assert LAMMPS_META_SCHEMA == 2
        assert self.block()["schema_version"] == 2

    def test_contract_is_cutoff_graph(self):
        """Both current export paths derive the edge set from a cutoff."""
        assert self.block()["contract"] == "cutoff-graph"
        assert self.block(cuda_graph=True, n_atoms=4, e_max=8)["contract"] == "cutoff-graph"

    def test_execution_is_dynamic_without_cuda_graph(self):
        assert self.block()["execution"] == "dynamic"

    def test_execution_is_cuda_graph_with_fixed_shapes(self):
        assert self.block(cuda_graph=True, n_atoms=4, e_max=8)["execution"] == "cuda-graph"

    def test_energy_conv_covers_every_unit_style(self):
        conv = self.block()["energy_conv"]
        assert set(conv) == {"real", "metal"}
        assert all(isinstance(v, float) for v in conv.values())

    def test_energy_conv_is_unity_for_the_native_style(self):
        assert self.block(units="real")["energy_conv"]["real"] == 1.0
        assert self.block(units="metal")["energy_conv"]["metal"] == 1.0

    def test_energy_conv_real_to_metal_matches_si_value(self):
        """1 kcal/mol in eV = 4184 / (N_A * e), 2019 SI exact constants."""
        assert self.block(units="real")["energy_conv"]["metal"] == pytest.approx(
            0.043364104241800934, rel=1e-6
        )

    def test_length_unit_is_angstrom(self):
        """Guards ``force_conv = energy_conv`` on the C++ side."""
        assert self.block()["length_unit"] == "angstrom"

    def test_outputs_omit_virial_when_unsupported(self):
        assert self.block(supports_virial=False)["outputs"] == ["energy", "forces"]

    def test_outputs_include_virial_when_supported(self):
        assert self.block(supports_virial=True)["outputs"] == ["energy", "forces", "virial"]

    def test_inputs_are_the_flat_triple_for_dynamic(self):
        assert self.block()["inputs"] == ["Z", "pos", "edge_index"]

    def test_inputs_carry_the_mask_for_cuda_graph(self):
        block = self.block(cuda_graph=True, n_atoms=4, e_max=8)
        assert block["inputs"] == ["Z", "pos", "edge_index", "mask"]

    def test_adapter_is_a_registered_name(self):
        """Provenance must name a real adapter, never ``molnex-tensordict-static``."""
        from molix.engine.adapter import EngineAdapter

        assert self.block()["adapter"] in EngineAdapter.names()


class TestExportForLammpsValidation:
    """``molix.engine.export.export_for_lammps`` argument validation (schema v2)."""

    def test_cuda_graph_with_non_default_adapter_raises(self, tmp_path):
        """The static path inlines the molnex-tensordict convention: a foreign
        adapter used to be silently ignored, which then lied in ``meta.json``.

        The ``ValueError`` must fire during argument validation — i.e. *before*
        the CPU force-export ``RuntimeError`` raised deeper in the pipeline.
        """
        with pytest.raises(ValueError, match="adapter"):
            export_for_lammps(
                _FlatEFPotential(),
                tmp_path / "x",
                species=[1],
                cutoff=4.5,
                adapter="flat",
                device="cpu",
                cuda_graph=True,
                n_atoms=4,
                e_max=8,
            )
