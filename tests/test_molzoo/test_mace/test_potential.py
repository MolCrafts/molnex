"""Tests for molzoo.mace.potential — the unified MACE energy/force potential.

:class:`~molzoo.mace.potential.MacePotential` merges the two pre-cutover flat
foundation models — now the keyword aliases
:class:`molzoo.mace.potential.MaceMatpes` /
:class:`~molzoo.mace.potential.MaceOmol` — into one spec-driven class with two
seams:

* ``energy_core(...) -> (B,)`` — flat, public, compilable (1 dynamo graph);
* ``_write_energy(batch) -> batch`` — the ``molpot.derivation.protocol``
  hook that ``call_energy`` dispatches to.

Every branch (forces on/off, charge/spin conditioning) is resolved in
``__init__`` into ``self._pipeline``; ``forward`` is a one-line dispatch and no
public signature carries ``compute_forces``.

The two keyword variants are imported here as **parity oracles**: the numbers
must not move when the code moves house. Since the cutover
(``mace-subpackage-restructure-06-wire``) they are subclasses of the class under
test, so the parity is no longer "old module vs new module" but the two seams a
caller actually has — ``MacePotential.forward`` (batch schema, shared force
kernel, own position leaf policy) against ``MaceMatpes.energy_forces`` /
``MaceOmol.energy_forces`` (raw tensors, the variants' own leaf). Those are two
genuinely different code paths on one energy, and
``regressions/mace-subpackage-restructure-04-potential.py`` freezes the same
numbers as hard-coded literals with no import of either.

Every model is tiny (2 layers, 16 channels, ``l_max=1``), fp64 (autouse
``fp64`` fixture in ``conftest.py``), CPU, seeded — see
``.claude/specs/mace-subpackage-restructure-04-potential.md`` §Testing strategy.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from molix.compile import Compiler
from molpot.derivation.protocol import call_energy
from molzoo.mace.encoder import MACEEncoder
from molzoo.mace.potential import MaceMatpes, MaceOmol, MacePotential
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec
from tests.conftest import make_graph_batch, translate_graph
from tests.test_molzoo.test_mace.conftest import (
    CLUSTER_POS,
    CLUSTER_Z,
    PAIR_Z,
    SINGLE_GRAPH,
    full_edge_index,
    raw_tensors,
)

#: Iron (26) is outside the ``[1, 6, 8]`` table — ``searchsorted`` would snap it.
OFF_TABLE_Z = [8, 1, 1, 6, 26]

#: Rigid translation of the whole system (Å); the energy must not notice.
TRANSLATION = [1.0, 0.0, 0.0]

#: Re-homing the code must not move a bit: fp64 parity between the batch path
#: and the variants' raw-tensor ``energy_forces`` (spec §Domain basis —
#: measured bitwise equal at the same operator order).
ENERGY_PARITY_ATOL = 1e-12  # eV
FORCE_PARITY_ATOL = 1e-12  # eV/Å

#: Newton's third law on an isolated cluster (measured 1.4e-17 on this model).
NET_FORCE_ATOL = 1e-10  # eV/Å

#: Central-difference step and tolerance (measured worst error 2.8e-7).
FD_STEP = 1e-4  # Å
FD_ATOL = 1e-6  # eV/Å

#: Atom/axis pairs probed by the finite-difference test.
FD_PROBES = ((0, 0), (2, 1), (4, 2))

#: Hard-coded golden: the parameter names of the pre-cutover flat energy core
#: (``MaceMatpes._compute_energy``, ``mace_matpes.py:229-237`` at 0e05959).
#: ``energy_core`` must keep them so 07's call-site re-point is a pure rename.
FLAT_ENERGY_CORE_PARAMETERS = ("self", "positions", "Z", "edge_index", "batch", "num_graphs")

#: The keyword tail after the flat six: imaged geometry, the rejected legacy
#: ``shifts``, the two per-graph conditioning tensors only the OMOL variant
#: consumes, and the ``cell`` only the Polar k-space path reads.
CONDITIONING_PARAMETERS = ("edge_diff", "shifts", "total_charge", "total_spin", "cell")

#: A third force path in this file is forbidden (CLAUDE.md "Force derivation").
FORBIDDEN_GRAD_CALLS = frozenset(
    {"torch.autograd.grad", "autograd.grad", "torch.func.grad", "func.grad", "grad"}
)


# --------------------------------------------------------------------------
# Precondition guard (spec §Testing strategy): the package must not be shadowed
# by a same-named module, or the whole directory is silently skipped.
# --------------------------------------------------------------------------

_TEST_PACKAGE = Path(__file__).resolve().parent


def test_mace_test_directory_is_a_package() -> None:
    """``tests/test_molzoo/test_mace/`` must carry an ``__init__.py``."""
    assert (_TEST_PACKAGE / "__init__.py").is_file()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _transfer(potential: MacePotential, variant: torch.nn.Module) -> MacePotential:
    """Move the oracle's weights into the potential by a strict key match."""
    potential.load_state_dict(variant.state_dict(), strict=True)
    return potential


def _dotted_name(node: ast.expr) -> str:
    """``ast.Attribute``/``ast.Name`` chain as ``"torch.autograd.grad"``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _called_names(tree: ast.AST) -> set[str]:
    """Every dotted callee name appearing in ``tree``."""
    return {_dotted_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}


def _potential_module_tree() -> ast.Module:
    """Parse ``src/molzoo/mace/potential.py``."""
    source = Path(inspect.getsourcefile(MacePotential) or "").read_text(encoding="utf-8")
    return ast.parse(source)


def _function_body(function: Callable[..., object]) -> list[ast.stmt]:
    """Statements of ``function``, with a leading docstring dropped."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition = tree.body[0]
    assert isinstance(definition, ast.FunctionDef)
    body = definition.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def leaf_cluster() -> TensorDict:
    """The cluster whose ``atoms.pos`` is already a live ``requires_grad`` leaf."""
    pos = torch.tensor(CLUSTER_POS, dtype=torch.float64).requires_grad_(True)
    return make_graph_batch(
        pos=pos,
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        edge_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
    )


@pytest.fixture
def branch_cluster() -> TensorDict:
    """The cluster whose ``atoms.pos`` requires grad but is **not** a leaf."""
    pos = torch.tensor(CLUSTER_POS, dtype=torch.float64).requires_grad_(True) * 1.0
    return make_graph_batch(
        pos=pos,
        Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
        edge_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
    )


@pytest.fixture
def matpes_potential(tiny_matpes_spec: MACEMatpesSpec) -> MacePotential:
    """Energy + force MatPES potential on the tiny spec."""
    torch.manual_seed(0)
    return MacePotential(tiny_matpes_spec).eval()


@pytest.fixture
def energy_only_potential(tiny_matpes_spec: MACEMatpesSpec) -> MacePotential:
    """Energy-only MatPES potential — ``compute_forces=False`` is a *construction*."""
    torch.manual_seed(0)
    return MacePotential(tiny_matpes_spec, compute_forces=False).eval()


@pytest.fixture
def omol_potential(tiny_omol_spec: MACEOMolSpec) -> MacePotential:
    """Energy + force OMOL potential on the tiny spec."""
    torch.manual_seed(0)
    return MacePotential(tiny_omol_spec).eval()


@pytest.fixture
def matpes_parity(
    tiny_matpes_spec: MACEMatpesSpec, matpes_variant: MaceMatpes
) -> tuple[MacePotential, MaceMatpes]:
    """MatPES potential holding the oracle variant's weights, plus the oracle."""
    return _transfer(MacePotential(tiny_matpes_spec).eval(), matpes_variant), matpes_variant


@pytest.fixture
def omol_parity(
    tiny_omol_spec: MACEOMolSpec, omol_variant: MaceOmol
) -> tuple[MacePotential, MaceOmol]:
    """OMOL potential holding the oracle variant's weights, plus the oracle."""
    return _transfer(MacePotential(tiny_omol_spec).eval(), omol_variant), omol_variant


class TestMacePotential:
    """Test the unified MACE energy/force potential."""

    # -- construction ---------------------------------------------------------

    def test_inherits_the_encoder_so_state_dict_keys_stay_flat(self) -> None:
        """Holding an encoder would prefix every key with ``encoder.`` (ac-007)."""
        assert issubclass(MacePotential, MACEEncoder)

    @pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
    def test_user_selected_energy_accumulation(self, matpes_variant: MaceMatpes, dtype) -> None:
        matpes_variant.energy_dtype = dtype
        values = torch.tensor([1e8, 1.0, -1e8], dtype=torch.float32, requires_grad=True)
        result = matpes_variant._sum_energy(values, torch.zeros(3, dtype=torch.long), 1)
        assert result.dtype == (dtype or values.dtype)
        assert result.item() == (1.0 if dtype == torch.float64 else 0.0)
        assert torch.equal(torch.autograd.grad(result.sum(), values)[0], torch.ones_like(values))

    def test_invalid_energy_dtype_is_rejected(self, tiny_matpes_spec: MACEMatpesSpec) -> None:
        with pytest.raises(ValueError, match="energy_dtype"):
            MacePotential(tiny_matpes_spec, energy_dtype=torch.float16)

    def test_input_cast_keeps_force_chain_rule(self, matpes_variant: MaceMatpes) -> None:
        model = matpes_variant.float().eval()
        original_keys = set(model.state_dict())
        pos = torch.tensor(CLUSTER_POS, dtype=torch.float64)
        z = torch.tensor(CLUSTER_Z)
        edges = full_edge_index(SINGLE_GRAPH)
        batch = torch.zeros(len(z), dtype=torch.long)
        expected = model.energy_forces(pos.float(), z, edges, batch, 1)
        model.configure_input_dtype(torch.float32)
        actual = model.energy_forces(pos, z, edges, batch, 1)
        assert actual["forces"].dtype == torch.float64
        torch.testing.assert_close(actual["forces"], expected["forces"].double(), atol=0, rtol=0)
        torch.testing.assert_close(actual["energy"], expected["energy"], atol=0, rtol=0)
        assert set(model.state_dict()) == original_keys

    def test_input_precision_cannot_change_after_compile(self, matpes_variant: MaceMatpes) -> None:
        matpes_variant._core = matpes_variant.energy_core
        with pytest.raises(RuntimeError, match="before compile_core"):
            matpes_variant.configure_input_dtype(torch.float32)

    def test_energy_forces_rebinds_live_edges_to_its_position_leaf(self, matpes_variant):
        pos = torch.tensor(CLUSTER_POS, dtype=torch.float64, requires_grad=True)
        z = torch.tensor(CLUSTER_Z)
        edges = full_edge_index(SINGLE_GRAPH)
        batch = torch.zeros(len(z), dtype=torch.long)
        live = pos[edges[:, 1]] - pos[edges[:, 0]]
        expected = matpes_variant.energy_forces(pos, z, edges, batch, 1, edge_diff=live.detach())
        actual = matpes_variant.energy_forces(pos, z, edges, batch, 1, edge_diff=live)
        torch.testing.assert_close(actual["forces"], expected["forces"], atol=0, rtol=0)

    def test_forces_are_computed_by_default(self) -> None:
        """MD / evaluation is the main use — unlike ``PiNetPotential``."""
        parameters = inspect.signature(MacePotential.__init__).parameters
        assert parameters["compute_forces"].default is True
        assert parameters["use_fallback"].default is False

    def test_the_spec_decides_the_cuequivariance_path(
        self, matpes_potential: MacePotential
    ) -> None:
        """``spec.use_fallback=True`` (the tiny CPU spec) builds pure-torch blocks."""
        assert matpes_potential.interactions[0].use_fallback is True
        assert matpes_potential.products[0].use_fallback is True

    def test_the_constructor_can_escalate_a_fused_spec_to_the_fallback(
        self, tiny_omol_spec: MACEOMolSpec
    ) -> None:
        """``use_fallback=True`` at the call site wins over a fused spec.

        The knob is escalation-only: the ``False`` default never downgrades a
        spec that asked for the pure-torch path (a ``bool`` cannot express
        "not given"), while a CPU / test call site can always force it on.
        """
        assert tiny_omol_spec.use_fallback is False
        potential = MacePotential(tiny_omol_spec, use_fallback=True)
        assert potential.interactions[0].use_fallback is True
        assert potential.products[0].use_fallback is True

    # -- state_dict parity: the chain gate ------------------------------------

    def test_loads_the_matpes_variant_state_dict_strictly(
        self, matpes_potential: MacePotential, matpes_variant: MaceMatpes
    ) -> None:
        """No missing, no unexpected: a direct hand-over, not a fuzzy remap."""
        report = matpes_potential.load_state_dict(matpes_variant.state_dict(), strict=True)
        assert (list(report.missing_keys), list(report.unexpected_keys)) == ([], [])

    def test_loads_the_omol_variant_state_dict_strictly(
        self, omol_potential: MacePotential, omol_variant: MaceOmol
    ) -> None:
        """The OMOL arm of the strict hand-over."""
        report = omol_potential.load_state_dict(omol_variant.state_dict(), strict=True)
        assert (list(report.missing_keys), list(report.unexpected_keys)) == ([], [])

    # -- forward: the molix.md.forcefield contract ----------------------------

    def test_forward_returns_the_same_batch_object(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """Writes are in place — ``Potentials`` keeps its own handle."""
        assert matpes_potential(cluster) is cluster

    def test_forward_writes_one_energy_per_graph(
        self, matpes_potential: MacePotential, pair_batch: TensorDict
    ) -> None:
        """``graphs.energy`` is ``(B,)`` in eV."""
        out = matpes_potential(pair_batch)
        assert out["graphs", "energy"].shape == (2,)

    def test_forward_writes_one_force_vector_per_atom(
        self, matpes_potential: MacePotential, pair_batch: TensorDict
    ) -> None:
        """``atoms.forces`` is ``(N, 3)`` in eV/Å."""
        out = matpes_potential(pair_batch)
        assert out["atoms", "forces"].shape == (len(PAIR_Z), 3)

    def test_forward_builds_the_graphs_namespace_with_the_batch_size(
        self, matpes_potential: MacePotential, pair_batch: TensorDict
    ) -> None:
        """A missing ``graphs`` must come back as ``batch_size=[B]``, never ``[]``.

        Since 07-cleanup ``write_energy`` sizes the namespace itself, but the
        potential still creates it *before* calling ``write_energy`` so the
        schema holds even for callers that bypass the protocol helpers.
        """
        del pair_batch["graphs"]
        out = matpes_potential(pair_batch)
        assert out["graphs"].batch_size == torch.Size([2])

    def test_energy_only_forward_also_builds_graphs_with_the_batch_size(
        self, energy_only_potential: MacePotential, pair_batch: TensorDict
    ) -> None:
        """The energy-only pipeline is ``_write_energy`` — same schema duty."""
        del pair_batch["graphs"]
        out = energy_only_potential(pair_batch)
        assert out["graphs"].batch_size == torch.Size([2])

    def test_forward_consumes_the_periodic_shifts(
        self, matpes_potential: MacePotential, periodic_cluster: TensorDict
    ) -> None:
        """``edges.shifts`` must reach the edge vectors, not be dropped."""
        pos, Z, edge_index, batch, num_graphs = raw_tensors(periodic_cluster)
        edge_diff = periodic_cluster["edges", "edge_diff"]
        with torch.no_grad():
            expected = matpes_potential.energy_core(
                pos, Z, edge_index, batch, num_graphs, edge_diff=edge_diff
            )
        out = matpes_potential(periodic_cluster)
        assert torch.allclose(out["graphs", "energy"], expected, atol=ENERGY_PARITY_ATOL, rtol=0.0)

    def test_energy_only_instance_writes_no_forces(
        self, energy_only_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """ "Energy only" is a constructed object, and it must not pay for forces."""
        out = energy_only_potential(cluster)
        assert ("atoms", "forces") not in out.keys(include_nested=True)

    def test_energy_only_instance_still_writes_the_energy(
        self, energy_only_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """The energy-only pipeline is still a full energy pipeline."""
        out = energy_only_potential(cluster)
        assert out["graphs", "energy"].shape == (1,)

    # -- needs_leaf policy ----------------------------------------------------

    def test_detached_positions_yield_a_detached_energy(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """The potential owns the leaf, so nobody downstream can lose a graph."""
        out = matpes_potential(cluster)
        assert out["graphs", "energy"].requires_grad is False

    def test_non_leaf_positions_yield_a_detached_energy(
        self, matpes_potential: MacePotential, branch_cluster: TensorDict
    ) -> None:
        """``needs_leaf = not (requires_grad and is_leaf)`` — a branch is not a leaf."""
        out = matpes_potential(branch_cluster)
        assert out["graphs", "energy"].requires_grad is False

    def test_grad_leaf_positions_keep_the_energy_attached(
        self, matpes_potential: MacePotential, leaf_cluster: TensorDict
    ) -> None:
        """Detaching here would cut the training path at the first potential."""
        out = matpes_potential(leaf_cluster)
        assert out["graphs", "energy"].requires_grad is True

    def test_grad_leaf_energy_loss_reaches_the_parameters(
        self, matpes_potential: MacePotential, leaf_cluster: TensorDict
    ) -> None:
        """The whole point of not detaching: ``loss.backward()`` trains the model."""
        out = matpes_potential(leaf_cluster)
        out["graphs", "energy"].sum().backward()
        weight = matpes_potential.readouts[0].linear.weight
        assert weight.grad is not None and torch.any(weight.grad != 0)

    # -- energy_core: the flat compile seam -----------------------------------

    def test_energy_core_keeps_the_flat_parameter_names(self) -> None:
        """07's call-site re-point must be a pure rename of ``_compute_energy``."""
        parameters = tuple(inspect.signature(MacePotential.energy_core).parameters)
        assert parameters[: len(FLAT_ENERGY_CORE_PARAMETERS)] == FLAT_ENERGY_CORE_PARAMETERS

    def test_energy_core_takes_the_conditioning_tensors_last(self) -> None:
        """``shifts`` then the two per-graph OMOL tensors, all optional."""
        parameters = inspect.signature(MacePotential.energy_core).parameters
        assert tuple(parameters)[len(FLAT_ENERGY_CORE_PARAMETERS) :] == CONDITIONING_PARAMETERS
        assert all(parameters[name].default is None for name in CONDITIONING_PARAMETERS)

    def test_energy_core_first_six_parameters_are_positional(self) -> None:
        """``run_nve.py`` calls it positionally on the MD hot path."""
        parameters = inspect.signature(MacePotential.energy_core).parameters
        assert all(
            parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for name in FLAT_ENERGY_CORE_PARAMETERS[1:]
        )

    def test_energy_core_returns_one_energy_per_graph(
        self, matpes_potential: MacePotential, pair_batch: TensorDict
    ) -> None:
        """Flat tensors in, ``(B,)`` eV out — no TensorDict on this seam."""
        with torch.no_grad():
            energy = matpes_potential.energy_core(*raw_tensors(pair_batch))
        assert energy.shape == (2,)

    def test_energy_core_agrees_with_the_forward_energy(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """One energy, two seams: the TensorDict path must add nothing."""
        with torch.no_grad():
            direct = matpes_potential.energy_core(*raw_tensors(cluster))
        out = matpes_potential(cluster)
        assert torch.allclose(out["graphs", "energy"], direct, atol=ENERGY_PARITY_ATOL, rtol=0.0)

    def test_matpes_energy_core_rejects_a_total_charge(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """Silently ignoring the conditioning would return a wrong energy."""
        with pytest.raises(ValueError):
            matpes_potential.energy_core(
                *raw_tensors(cluster), total_charge=torch.zeros(1, dtype=torch.long)
            )

    def test_matpes_energy_core_rejects_a_total_spin(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """Same for the spin channel — this variant has no joint embedding."""
        with pytest.raises(ValueError):
            matpes_potential.energy_core(
                *raw_tensors(cluster), total_spin=torch.ones(1, dtype=torch.long)
            )

    def test_omol_energy_core_consumes_the_total_charge(
        self, omol_parity: tuple[MacePotential, MaceOmol], omol_cluster: TensorDict
    ) -> None:
        """A different charge must reach the joint embedding and move the energy."""
        potential, _ = omol_parity
        pos, Z, edge_index, batch, num_graphs = raw_tensors(omol_cluster)
        spin = torch.ones(1, dtype=torch.long)
        with torch.no_grad():
            neutral = potential.energy_core(
                pos,
                Z,
                edge_index,
                batch,
                num_graphs,
                total_charge=torch.zeros(1, dtype=torch.long),
                total_spin=spin,
            )
            cation = potential.energy_core(
                pos,
                Z,
                edge_index,
                batch,
                num_graphs,
                total_charge=torch.ones(1, dtype=torch.long),
                total_spin=spin,
            )
        assert float((cation - neutral).abs().max()) > 1e-6

    # -- _write_energy: the protocol hook -------------------------------------

    def test_call_energy_dispatches_to_the_energy_core(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """``EnergyReadout`` must reach the core, not the force pipeline."""
        out = call_energy(matpes_potential, cluster)
        assert ("graphs", "energy") in out.keys(include_nested=True)
        assert ("atoms", "forces") not in out.keys(include_nested=True)

    def test_call_energy_keeps_the_energy_on_the_callers_leaf(
        self, matpes_potential: MacePotential, leaf_cluster: TensorDict
    ) -> None:
        """``_write_energy`` must not detach — ``ForceReadout`` differentiates it."""
        out = call_energy(matpes_potential, leaf_cluster)
        assert out["graphs", "energy"].requires_grad is True

    # -- monomorphism ---------------------------------------------------------

    @pytest.mark.parametrize("method", ["forward", "energy_core", "_write_energy"])
    def test_no_compute_forces_on_the_public_surface(self, method: str) -> None:
        """ "Energy only" is a construction, never a per-call flag."""
        parameters = inspect.signature(getattr(MacePotential, method)).parameters
        assert "compute_forces" not in parameters

    def test_forward_is_a_single_pipeline_dispatch(self) -> None:
        """``return self._pipeline(batch)`` — every branch resolved in ``__init__``."""
        body = _function_body(MacePotential.forward)
        assert len(body) == 1
        statement = body[0]
        assert isinstance(statement, ast.Return)
        assert isinstance(statement.value, ast.Call)
        assert _dotted_name(statement.value.func) == "self._pipeline"

    def test_no_hand_rolled_force_pass_in_this_module(self) -> None:
        """Forces come from the shared kernel; a third force path is forbidden."""
        tree = _potential_module_tree()
        called = _called_names(tree)
        assert not called & FORBIDDEN_GRAD_CALLS
        assert not {name for name in called if name.split(".")[-1] == "backward"}

    # -- numerical parity with the keyword variants (fp64, hard tolerance) ----

    def test_matpes_energy_matches_the_variant_energy_forces(
        self, matpes_parity: tuple[MacePotential, MaceMatpes], cluster: TensorDict
    ) -> None:
        """Moving house must not move a bit (1e-12 eV on a −3.1 keV total)."""
        potential, variant = matpes_parity
        pos, Z, edge_index, batch, num_graphs = raw_tensors(cluster)
        reference = variant.energy_forces(pos, Z, edge_index, batch, num_graphs=num_graphs)
        energy = potential(cluster)["graphs", "energy"]
        assert torch.allclose(energy, reference["energy"], atol=ENERGY_PARITY_ATOL, rtol=0.0), (
            f"bitwise equal: {torch.equal(energy, reference['energy'])}"
        )

    def test_matpes_forces_match_the_variant_energy_forces(
        self, matpes_parity: tuple[MacePotential, MaceMatpes], cluster: TensorDict
    ) -> None:
        """Same for ``F = -dE/dx`` (1e-12 eV/Å)."""
        potential, variant = matpes_parity
        pos, Z, edge_index, batch, num_graphs = raw_tensors(cluster)
        reference = variant.energy_forces(pos, Z, edge_index, batch, num_graphs=num_graphs)
        forces = potential(cluster)["atoms", "forces"]
        assert torch.allclose(forces, reference["forces"], atol=FORCE_PARITY_ATOL, rtol=0.0), (
            f"bitwise equal: {torch.equal(forces, reference['forces'])}"
        )

    def test_matpes_periodic_energy_matches_the_variant_energy_forces(
        self, matpes_parity: tuple[MacePotential, MaceMatpes], periodic_cluster: TensorDict
    ) -> None:
        """The ``edges.shifts`` path is the MD path — it gets its own parity check."""
        potential, variant = matpes_parity
        pos, Z, edge_index, batch, num_graphs = raw_tensors(periodic_cluster)
        edge_diff = periodic_cluster["edges", "edge_diff"]
        reference = variant.energy_forces(
            pos, Z, edge_index, batch, num_graphs=num_graphs, edge_diff=edge_diff
        )
        energy = potential(periodic_cluster)["graphs", "energy"]
        assert torch.allclose(energy, reference["energy"], atol=ENERGY_PARITY_ATOL, rtol=0.0), (
            f"bitwise equal: {torch.equal(energy, reference['energy'])}"
        )

    def test_matpes_periodic_forces_match_the_variant_energy_forces(
        self, matpes_parity: tuple[MacePotential, MaceMatpes], periodic_cluster: TensorDict
    ) -> None:
        """``S_ij`` is constant w.r.t. ``pos``, so the forces stay exact."""
        potential, variant = matpes_parity
        pos, Z, edge_index, batch, num_graphs = raw_tensors(periodic_cluster)
        edge_diff = periodic_cluster["edges", "edge_diff"]
        reference = variant.energy_forces(
            pos, Z, edge_index, batch, num_graphs=num_graphs, edge_diff=edge_diff
        )
        forces = potential(periodic_cluster)["atoms", "forces"]
        assert torch.allclose(forces, reference["forces"], atol=FORCE_PARITY_ATOL, rtol=0.0), (
            f"bitwise equal: {torch.equal(forces, reference['forces'])}"
        )

    def test_omol_energy_matches_the_variant_energy_forces(
        self, omol_parity: tuple[MacePotential, MaceOmol], omol_cluster: TensorDict
    ) -> None:
        """The conditioned arm: charge 0 / spin 1, the flat model's own defaults."""
        potential, variant = omol_parity
        pos, Z, edge_index, batch, _ = raw_tensors(omol_cluster)
        reference = variant.energy_forces(
            pos,
            Z,
            edge_index,
            batch,
            total_charge=omol_cluster["graphs", "total_charge"],
            total_spin=omol_cluster["graphs", "total_spin"],
        )
        energy = potential(omol_cluster)["graphs", "energy"]
        assert torch.allclose(energy, reference["energy"], atol=ENERGY_PARITY_ATOL, rtol=0.0), (
            f"bitwise equal: {torch.equal(energy, reference['energy'])}"
        )

    def test_omol_forces_match_the_variant_energy_forces(
        self, omol_parity: tuple[MacePotential, MaceOmol], omol_cluster: TensorDict
    ) -> None:
        """Non-vacuous by construction — see :func:`.conftest.wake_zero_init_readout`."""
        potential, variant = omol_parity
        pos, Z, edge_index, batch, _ = raw_tensors(omol_cluster)
        reference = variant.energy_forces(
            pos,
            Z,
            edge_index,
            batch,
            total_charge=omol_cluster["graphs", "total_charge"],
            total_spin=omol_cluster["graphs", "total_spin"],
        )
        largest = float(reference["forces"].detach().abs().max())
        assert largest > 0.0, "vacuous oracle: every reference force is exactly zero"
        forces = potential(omol_cluster)["atoms", "forces"]
        assert torch.allclose(forces, reference["forces"], atol=FORCE_PARITY_ATOL, rtol=0.0), (
            f"bitwise equal: {torch.equal(forces, reference['forces'])}"
        )

    def test_omol_defaults_to_a_neutral_closed_shell_singlet(
        self, omol_parity: tuple[MacePotential, MaceOmol], omol_cluster: TensorDict
    ) -> None:
        """Unconditioned batch → charge 0 / spin 1, as the flat model defaults.

        Spin 0 would index an untrained embedding row and quietly return
        garbage, so the default must stay ``1`` (``mace_omol.py:330-335``).
        """
        potential, _ = omol_parity
        # Build the unconditioned twin first: ``forward`` swaps ``atoms.pos``
        # for its own leaf, and reading it back afterwards would compare a
        # different tensor object.
        bare = make_graph_batch(
            pos=omol_cluster["atoms", "pos"].clone(),
            Z=omol_cluster["atoms", "Z"],
            edge_index=omol_cluster["edges", "edge_index"],
            batch=omol_cluster["atoms", "batch"],
        )
        conditioned = potential(omol_cluster)["graphs", "energy"].clone()
        assert torch.allclose(
            potential(bare)["graphs", "energy"], conditioned, atol=ENERGY_PARITY_ATOL, rtol=0.0
        )

    # -- physics --------------------------------------------------------------

    def test_net_force_vanishes(self, matpes_potential: MacePotential, cluster: TensorDict) -> None:
        """Newton's third law: the energy depends on ``pos`` only through ``r_ij``."""
        forces = matpes_potential(cluster)["atoms", "forces"]
        assert float(forces.detach().sum(0).abs().max()) <= NET_FORCE_ATOL

    def test_energy_is_translation_invariant(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """A rigid 1 Å shift cannot change a physical energy."""
        moved = translate_graph(cluster, torch.tensor(TRANSLATION, dtype=torch.float64))
        before = matpes_potential(cluster)["graphs", "energy"].clone()
        after = matpes_potential(moved)["graphs", "energy"]
        assert float((after - before).abs().max()) <= ENERGY_PARITY_ATOL

    def test_forces_match_central_differences(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """``F = -dE/dx`` against a numerical derivative of ``energy_core``."""
        forces = matpes_potential(cluster)["atoms", "forces"]
        pos, Z, edge_index, batch, num_graphs = raw_tensors(cluster)
        pos = pos.detach()
        for atom, axis in FD_PROBES:
            shifted = pos.clone()
            shifted[atom, axis] += FD_STEP
            with torch.no_grad():
                plus = float(
                    matpes_potential.energy_core(shifted, Z, edge_index, batch, num_graphs).sum()
                )
            shifted[atom, axis] -= 2 * FD_STEP
            with torch.no_grad():
                minus = float(
                    matpes_potential.energy_core(shifted, Z, edge_index, batch, num_graphs).sum()
                )
            numerical = -(plus - minus) / (2 * FD_STEP)
            assert float(forces[atom, axis]) == pytest.approx(numerical, abs=FD_ATOL)

    # -- element table gate ---------------------------------------------------

    def test_the_element_gate_runs_once_per_instance(
        self,
        matpes_potential: MacePotential,
        cluster: TensorDict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Re-validating per call costs a host sync and a dynamo graph break.

        ``Z`` is constant over a trajectory and a wrongly wired model/dataset
        pair fails on the very first batch, so the gate is spent after one
        forward. Callers binding a *new* system to an existing instance call
        ``validate_elements`` themselves (``mace_matpes.py:206-222`` semantics,
        preserved deliberately).
        """
        checks: list[torch.Tensor] = []
        validate = matpes_potential.validate_elements

        def counting(Z: torch.Tensor) -> None:
            checks.append(Z)
            validate(Z)

        monkeypatch.setattr(matpes_potential, "validate_elements", counting)
        matpes_potential(cluster)
        matpes_potential(cluster)
        assert len(checks) == 1

    # -- compile smoke --------------------------------------------------------

    @pytest.mark.slow
    def test_energy_core_compiles_to_one_graph(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """The flat seam exists to be compiled: 1 dynamo graph, 0 breaks."""
        explanation = Compiler.explain(matpes_potential.energy_core, *raw_tensors(cluster))
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0


class TestMacePotentialCompileCore:
    """``compile_core`` — one precondition, one action, installed on the mlip.

    The compiled build lives on the potential (``self._core``) and not on the
    ``Potentials`` container, so it survives every ``to_potentials()`` rebuild;
    what that buys the MD path is asserted in
    ``tests/test_molix/test_md/test_driver.py``. Here only the local contract:
    when installation is legal, that it happens exactly once, and that
    ``_write_energy`` keeps doing all of its unpacking around the installed
    core.
    """

    def test_compile_core_requires_one_eager_forward_then_installs_once(
        self, matpes_variant: MaceMatpes, cluster: TensorDict
    ) -> None:
        """Fresh instance refuses; after one eager forward it compiles once.

        The single precondition is ``_elements_validated`` — the element table
        is checked eagerly and never inside the trace, and on the polarisable
        variant the same eager forward is what sizes the k-space grids.
        Installation is idempotent: a second call must not re-invoke the
        compiler, because the mlip is shared by every ``Potentials`` built
        from the same ForceField.
        """
        compiled: list[object] = []
        replacement = matpes_variant.energy_core  # a bound method: one object

        def stub(core: object) -> object:
            compiled.append(core)
            return replacement

        with pytest.raises(RuntimeError, match="compile_core") as excinfo:
            matpes_variant.compile_core(stub)
        assert "eager" in str(excinfo.value)
        assert compiled == []
        assert matpes_variant._core is None

        matpes_variant._write_energy(cluster)
        matpes_variant.compile_core(stub)
        assert len(compiled) == 1
        assert matpes_variant._core is replacement

        matpes_variant.compile_core(stub)
        assert len(compiled) == 1

    def test_write_energy_dispatches_through_the_core_and_keeps_its_unpacking(
        self,
        matpes_variant: MaceMatpes,
        cluster: TensorDict,
        leaf_cluster: TensorDict,
    ) -> None:
        """Installing a core moves the call site only — never the unpacking.

        ``_write_energy`` still resolves ``num_graphs``, still creates a missing
        ``graphs`` namespace at ``batch_size=[B]`` through its one documented
        host sync, still rejects a batch without ``edges.edge_diff``, and still
        writes the energy back **undetached** so an outer force session can
        differentiate it.
        """
        hits: list[int] = []
        eager_core = matpes_variant.energy_core

        def counting(*args: object, **kwargs: object) -> torch.Tensor:
            hits.append(1)
            return eager_core(*args, **kwargs)

        matpes_variant._write_energy(cluster)  # spends the eager element gate
        matpes_variant.compile_core(lambda _: counting)

        with torch.enable_grad():
            out = matpes_variant._write_energy(leaf_cluster)
        assert hits == [1]
        assert out["graphs", "energy"].grad_fn is not None

        no_graphs = cluster.clone()
        del no_graphs["graphs"]
        rebuilt = matpes_variant._write_energy(no_graphs)
        assert len(hits) == 2
        assert rebuilt["graphs"].batch_size == torch.Size([1])

        no_diff = cluster.clone()
        no_diff["edges"] = no_diff["edges"].exclude("edge_diff")
        with pytest.raises(ValueError, match="requires edges.edge_diff"):
            matpes_variant._write_energy(no_diff)

    def test_the_element_gate_is_eager_only_and_uncounted(
        self,
        matpes_variant: MaceMatpes,
        cluster: TensorDict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Under tracing the gate folds away and stays unspent.

        ``torch.compiler.is_compiling()`` is a tracing-time constant, so dynamo
        deletes this branch and the host-syncing element check never enters the
        graph. It must also leave ``_elements_validated`` alone: a traced call
        validated nothing, so it must not claim the gate was spent — that is
        exactly why the fact is a single flag and not a call counter, which
        dynamo would replay and advance. Whether the check *itself* rejects an
        off-table ``Z``, and that it runs once per eager instance, are owned by
        ``TestMacePotential`` and ``test_encoder.py``.
        """
        compiling = [True]
        checks: list[torch.Tensor] = []
        validate = matpes_variant.validate_elements
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling[0])
        monkeypatch.setattr(matpes_variant, "validate_elements", lambda Z: checks.append(Z))

        matpes_variant._write_energy(cluster)
        assert checks == []
        assert matpes_variant._elements_validated is False

        compiling[0] = False
        matpes_variant._write_energy(cluster)
        assert len(checks) == 1
        assert matpes_variant._elements_validated is True
        validate(cluster["atoms", "Z"])  # the real check still passes this batch

        assert not hasattr(matpes_variant, "_eager_calls")
        assert "no element validation" in (MacePotential.energy_core.__doc__ or "")
        assert "eager" in (MACEEncoder.validate_elements.__doc__ or "")
