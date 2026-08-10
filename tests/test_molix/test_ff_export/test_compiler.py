"""ForceFieldCompiler + registry integration tests."""

from __future__ import annotations

import pytest
import torch

from molix.ff_export import ForceFieldCompiler, OpenMMAdapter, UnsupportedTermError
from molix.ff_export.adapter import BackendAdapter
from molix.ff_export.cases import TranslationCase
from molix.ff_export.force_spec import ForceSpec
from molpot.ir import BondBag, ImproperHarmonicBag, PotentialIR


class TestForceFieldCompiler:
    def test_default_adapter_is_openmm_by_name(self):
        compiler = ForceFieldCompiler()
        assert isinstance(compiler.adapter, OpenMMAdapter)

    def test_adapter_instance_accepted(self):
        compiler = ForceFieldCompiler(adapter=OpenMMAdapter())
        ir = PotentialIR(bonds=BondBag(k=torch.tensor([100.0]), r0=torch.tensor([1.0])))
        spec = compiler.compile(ir)
        assert isinstance(spec, ForceSpec)
        assert spec.backend == "openmm"
        bond = next(f for f in spec.forces if f["type"] == "HarmonicBondForce")
        assert bond["parameters"][0]["k"] == 41840.0

    def test_unknown_adapter_name_raises(self):
        with pytest.raises(ValueError, match="unknown adapter"):
            ForceFieldCompiler(adapter="gromacs")

    def test_unsupported_term_raises_structured_error(self):
        compiler = ForceFieldCompiler("openmm")
        ir = PotentialIR(
            impropers_harmonic=ImproperHarmonicBag(
                k=torch.tensor([1.0]),
                chi0=torch.tensor([0.0]),
            )
        )
        with pytest.raises(UnsupportedTermError) as ei:
            compiler.compile(ir)
        err = ei.value
        assert err.term == "improper_harmonic"
        assert err.case is TranslationCase.UNSUPPORTED
        assert "improper_harmonic" in str(err)

    def test_empty_ir_compiles_to_empty_forces(self):
        spec = ForceFieldCompiler().compile(PotentialIR())
        assert spec.forces == []
        assert spec.backend == "openmm"

    def test_meta_passthrough(self):
        meta = {"type_systems": {"bond": ["C-C"]}, "symbolic": None}
        spec = ForceFieldCompiler().compile(
            PotentialIR(),
            type_systems=meta["type_systems"],
            symbolic=meta["symbolic"],
        )
        assert spec.metadata.get("type_systems") == {"bond": ["C-C"]}

    def test_registry_lists_openmm(self):
        assert "openmm" in BackendAdapter.names()
