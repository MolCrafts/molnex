"""Tests for ``molix.engine.export`` — argument validation + the meta block.

``export_for_lammps`` AOT-compiles a model into ``{name}.pt2`` and stamps a
sibling ``{name}.meta.json`` whose ``lammps`` block is the contract a
``pair_style molnex`` C++ run reads. The force-export path is refused on CPU
(``molix.export.Exporter``), so the block assembly itself is unit-tested
through the private pure function ``_lammps_meta_block`` — that is the only
way the schema contract can be covered on a CPU CI box.

Schema v3 (spec ``lammps-iface-05-pbc``) is a hard break of that contract: the
model gains a ``local_mask`` input and an ``atomic_energy`` output so a
domain-decomposed rank can sum **only local** site energies (D1), the
CUDA-graph edge-validity mask is renamed ``edge_mask`` so the two masks cannot
be mis-wired, and two new fields — ``receptive_field_layers`` (mandatory, no
default) and ``requires_sorted_edges`` (opt-in, default ``False``) — turn the
fat-ghost cutoff condition (D3.2) and the sorted-edge requirement (D6) into
declared, fail-closed contract terms.

-- the three ``export_for_lammps`` cases migrated verbatim from
   ``tests/test_molix/test_engine.py``, then updated to schema v3 --
"""

from __future__ import annotations

import inspect
import json

import pytest
import torch
import torch.nn as nn

from molix.engine import export_for_lammps
from molix.export import Exporter
from tests.test_molix.test_engine.conftest import _FlatEFPotential, pair_site_energies_and_forces


class _IndecomposablePotential(nn.Module):
    """Flat-convention model whose scalar energy is **not** its site-energy sum.

    Under v3 the exported energy is defined as
    ``(atomic_energy * local_mask).sum()``, so a model carrying a term that has
    no per-site attribution (a constant offset here) cannot be partitioned
    across ranks — the total it reports and the total the export computes would
    differ by that term. Export must refuse such a model eagerly, on the example
    system and *before* tracing, rather than silently shipping a different
    energy function.

    Local to this module: it exists only for the refusal case, so it stays here
    rather than in ``conftest.py`` (whose stubs are decomposable by design).
    """

    def forward(self, Z, pos, edge_index):
        edge_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
        atomic, forces = pair_site_energies_and_forces(pos, edge_index, edge_diff)
        return atomic.sum().reshape(1) + 1000.0, forces, atomic


def test_export_for_lammps_validates_inputs(tmp_path):
    with pytest.raises(ValueError, match="species"):
        export_for_lammps(
            _FlatEFPotential(),
            tmp_path / "x",
            species=[],
            cutoff=4.5,
            receptive_field_layers=1,
        )
    with pytest.raises(ValueError, match="units"):
        export_for_lammps(
            _FlatEFPotential(),
            tmp_path / "x",
            species=[1],
            cutoff=4.5,
            units="lj",
            receptive_field_layers=1,
        )


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
            receptive_field_layers=1,
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
        receptive_field_layers=1,
    )
    meta = json.loads((outdir / "model.meta.json").read_text())
    assert (outdir / "model.pt2").exists()
    lm = meta["lammps"]
    assert lm["schema_version"] == 3
    assert lm["cutoff"] == 4.5
    assert lm["units"] == "real"
    assert lm["species"] == [1, 6, 8]
    assert lm["adapter"] == "flat"
    assert lm["inputs"] == ["Z", "pos", "edge_index", "local_mask"]
    assert lm["outputs"] == ["energy", "forces", "atomic_energy"]
    assert lm["receptive_field_layers"] == 1
    assert lm["requires_sorted_edges"] is False

    # the exported .pt2 serves N/E different from the trace (MD needs this)
    runner = Exporter.load(outdir / "model.pt2")
    Z = torch.tensor([1, 6, 8, 1], device="cuda")
    pos = torch.randn(4, 3, device="cuda")
    ei = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], device="cuda")
    local_mask = torch.ones(4, dtype=torch.bool, device="cuda")
    e, f, ea = runner(Z, pos, ei, local_mask)
    assert f.shape == (4, 3)
    assert ea.shape == (4,)


class TestLammpsMetaBlock:
    """``molix.engine.export._lammps_meta_block`` — schema-v3 block assembly.

    Pure dict assembly, no AOTI export, so the ``lammps`` meta contract is
    covered on CPU (the surrounding export path is CPU-refused, see
    ``test_export_for_lammps_cpu_refused``). Field set and semantics are the
    spec ``lammps-iface-05-pbc`` §"Python 导出契约 schema v3" list — the
    acceptance criterion ac-010 calls these "the v3 expectations". The class
    keeps its name because it still tests the same unit; a parallel ``…V3``
    class would fossilise the v2 assertions next to it.

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
            receptive_field_layers: int,          # v3, mandatory
            requires_sorted_edges: bool = False,  # v3, opt-in
            n_atoms: int | None = None,
            e_max: int | None = None,
        ) -> dict

    ``contract`` / ``execution`` are *derived* (not parameters):
    ``contract == "cutoff-graph"`` for both current export paths, and
    ``execution`` is ``"cuda-graph"`` when ``cuda_graph`` else ``"dynamic"``.
    The private symbol is imported inside each test so only this class is red
    while it is missing.
    """

    #: Every key the v3 block must carry, dynamic-execution branch.
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
        "receptive_field_layers",
        "requires_sorted_edges",
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
            "receptive_field_layers": 2,
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

    def test_schema_version_is_three(self):
        from molix.engine.export import LAMMPS_META_SCHEMA

        assert LAMMPS_META_SCHEMA == 3
        assert self.block()["schema_version"] == 3

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
        assert self.block(supports_virial=False)["outputs"] == [
            "energy",
            "forces",
            "atomic_energy",
        ]

    def test_outputs_include_virial_when_supported(self):
        """virial moves from index 2 to 3 — the C++ decoder resolves it by name."""
        assert self.block(supports_virial=True)["outputs"] == [
            "energy",
            "forces",
            "atomic_energy",
            "virial",
        ]

    def test_inputs_are_the_flat_quadruple_for_dynamic(self):
        assert self.block()["inputs"] == ["Z", "pos", "edge_index", "local_mask"]

    def test_inputs_carry_both_masks_for_cuda_graph(self):
        block = self.block(cuda_graph=True, n_atoms=4, e_max=8)
        assert block["inputs"] == ["Z", "pos", "edge_index", "edge_mask", "local_mask"]

    def test_bare_mask_spelling_is_gone(self):
        """``mask`` next to ``local_mask`` is a predictable mis-wiring source."""
        assert "mask" not in self.block()["inputs"]
        assert "mask" not in self.block(cuda_graph=True, n_atoms=4, e_max=8)["inputs"]

    def test_receptive_field_layers_is_a_positive_int(self):
        block = self.block(receptive_field_layers=3)
        assert block["receptive_field_layers"] == 3
        assert isinstance(block["receptive_field_layers"], int)
        assert block["receptive_field_layers"] > 0

    def test_requires_sorted_edges_defaults_to_false(self):
        """Scatter reductions are permutation-invariant, so sorting is opt-in (D6)."""
        assert self.block()["requires_sorted_edges"] is False

    def test_requires_sorted_edges_is_recorded_when_declared(self):
        assert self.block(requires_sorted_edges=True)["requires_sorted_edges"] is True

    def test_adapter_is_a_registered_name(self):
        """Provenance must name a real adapter, never ``molnex-tensordict-static``."""
        from molix.engine.adapter import EngineAdapter

        assert self.block()["adapter"] in EngineAdapter.names()


class TestExportForLammpsValidation:
    """``molix.engine.export.export_for_lammps`` argument validation (schema v3)."""

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
                receptive_field_layers=1,
            )

    def test_missing_receptive_field_layers_raises(self, tmp_path):
        """D3.2 needs ``T`` and cannot infer it; defaulting to 1 would declare a
        3-layer MACE strictly local and let the fat-ghost check pass on a far
        too small threshold — exactly what fail-closed exists to prevent.

        The message must carry both the keyword and a copy-pasteable re-export
        command, because the reader hits this *after* a finished training run.
        The pinned form is the keyword name plus ``export_for_lammps(``.
        """
        with pytest.raises(ValueError) as excinfo:
            export_for_lammps(
                _FlatEFPotential(),
                tmp_path / "x",
                species=[1],
                cutoff=4.5,
                adapter="flat",
                device="cpu",
            )
        message = str(excinfo.value)
        assert "receptive_field_layers" in message
        assert "export_for_lammps(" in message

    def test_receptive_field_layers_is_keyword_only_without_a_default(self):
        """``= None`` + an explicit ``ValueError`` (not a bare required kwarg): a
        missing value must be reported as the message above, not as a
        ``TypeError`` from the call machinery."""
        param = inspect.signature(export_for_lammps).parameters["receptive_field_layers"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default in (None, inspect.Parameter.empty)

    def test_requires_sorted_edges_is_keyword_only_and_defaults_to_false(self):
        """Signature is the only CPU-observable surface for this default: the
        export path that would stamp it into ``meta.json`` is CPU-refused."""
        param = inspect.signature(export_for_lammps).parameters["requires_sorted_edges"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is False

    def test_indecomposable_model_is_refused(self, tmp_path):
        """Eager, pre-trace consistency check: ``|E - Σ atomic_energy|`` rel 1e-6.

        Runs in eager mode on the example system, so it never enters the traced
        graph — and it must fire before the CPU force-export ``RuntimeError``,
        i.e. before tracing at all.
        """
        with pytest.raises(ValueError, match="atomic_energy|per-atom"):
            export_for_lammps(
                _IndecomposablePotential(),
                tmp_path / "x",
                species=[1, 6],
                cutoff=4.5,
                adapter="flat",
                device="cpu",
                receptive_field_layers=1,
            )
