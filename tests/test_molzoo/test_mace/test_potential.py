"""Tests for molzoo.mace.potential — the unified MACE energy/force potential.

:class:`~molzoo.mace.potential.MacePotential` merges the two pre-cutover flat
foundation models — now the keyword aliases
:class:`molzoo.mace.potential.MaceMatpes` /
:class:`~molzoo.mace.potential.MaceOmol` — into one spec-driven class with two
seams:

* ``energy_core(...) -> (B,)`` — flat, public, compilable (1 dynamo graph);
* ``energy(batch) -> batch`` — the energy on the batch schema, what
  ``molix.autograd.Derivative`` differentiates and ``call_energy`` reaches.

Every branch (what is derived, charge/spin conditioning) is resolved in
``__init__``; ``forward`` is a one-line dispatch through the op and no public
signature carries a per-call switch.

The two keyword variants are imported here as **parity oracles**: the numbers
must not move when the code moves house. The parity is between the two seams
a caller actually has — ``MacePotential.forward`` (batch schema, the op owning
the position leaf) against ``energy_core`` differentiated on a leaf of the
test's own (:func:`_raw_energy_forces`). Those are two genuinely different
code paths on one energy.

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

from molix.autograd import Derivative
from molix.compile import Compiler
from molix.schema import ENERGY_KEY, FORCES_KEY, VIRIAL_KEY
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

#: Rigid translation of the whole system (Å); the energy must not notice.
TRANSLATION = [1.0, 0.0, 0.0]

#: Re-homing the code must not move a bit: fp64 parity between the batch path
#: and ``energy_core`` on a raw leaf (spec §Domain basis — measured bitwise
#: equal at the same operator order).
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
#: consumes, the ``cell`` only the Polar k-space path reads, and the per-atom
#: ``atom_mask`` a padded batch (``molix.data.MoleculePadder``) carries.
CONDITIONING_PARAMETERS = (
    "edge_diff",
    "shifts",
    "total_charge",
    "total_spin",
    "cell",
    "atom_mask",
)

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


def _raw_energy_forces(potential: MacePotential, batch: TensorDict) -> dict[str, torch.Tensor]:
    """``{"energy", "forces"}`` from ``energy_core`` on a leaf of this test's own.

    The raw-tensor oracle: everything is read off ``batch`` as flat tensors,
    the positions are re-leafed here, ``edge_diff`` is rebuilt from that leaf
    (plus the batch's own lattice remainder) and ``torch.autograd.grad`` is
    called directly — no op, no frame, no restore.
    """
    pos, Z, edge_index, membership, num_graphs = raw_tensors(batch)
    nested = batch.keys(include_nested=True)
    leaf = pos.detach().clone().requires_grad_(True)
    shifts = batch["pairs", "pair_diff"].detach() - (pos[edge_index[:, 1]] - pos[edge_index[:, 0]])
    conditioning = {
        name: batch["graphs", name]
        for name in ("total_charge", "total_spin")
        if ("graphs", name) in nested
    }
    energy = potential.energy_core(
        leaf,
        Z,
        edge_index,
        membership,
        num_graphs,
        edge_diff=leaf[edge_index[:, 1]] - leaf[edge_index[:, 0]] + shifts,
        **conditioning,
    )
    (grad,) = torch.autograd.grad(energy.sum(), leaf)
    return {"energy": energy.detach(), "forces": -grad}


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
        pair_index=full_edge_index(SINGLE_GRAPH),
        batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
    )


FORCES = ("atoms.pos",)
FORCES_AND_VIRIAL = ("atoms.pos", "graphs.strain")


@pytest.fixture
def matpes_potential(tiny_matpes_spec: MACEMatpesSpec) -> MacePotential:
    """Energy + force MatPES potential on the tiny spec."""
    torch.manual_seed(0)
    return MacePotential(tiny_matpes_spec, wrt=FORCES).eval()


@pytest.fixture
def energy_only_potential(tiny_matpes_spec: MACEMatpesSpec) -> MacePotential:
    """Energy-only MatPES potential — ``wrt=()`` is a *construction*."""
    torch.manual_seed(0)
    return MacePotential(tiny_matpes_spec).eval()


@pytest.fixture
def omol_potential(tiny_omol_spec: MACEOMolSpec) -> MacePotential:
    """Energy + force OMOL potential on the tiny spec."""
    torch.manual_seed(0)
    return MacePotential(tiny_omol_spec, wrt=FORCES).eval()


@pytest.fixture
def matpes_parity(
    tiny_matpes_spec: MACEMatpesSpec, matpes_variant: MaceMatpes
) -> tuple[MacePotential, MaceMatpes]:
    """MatPES potential holding the oracle variant's weights, plus the oracle."""
    potential = _transfer(MacePotential(tiny_matpes_spec, wrt=FORCES).eval(), matpes_variant)
    return potential, matpes_variant


@pytest.fixture
def omol_parity(
    tiny_omol_spec: MACEOMolSpec, omol_variant: MaceOmol
) -> tuple[MacePotential, MaceOmol]:
    """OMOL potential holding the oracle variant's weights, plus the oracle."""
    potential = _transfer(MacePotential(tiny_omol_spec, wrt=FORCES).eval(), omol_variant)
    return potential, omol_variant


class TestMacePotential:
    """Test the unified MACE energy/force potential."""

    # -- construction ---------------------------------------------------------

    def test_inherits_the_encoder_so_state_dict_keys_stay_flat(self) -> None:
        """Holding an encoder would prefix every key with ``encoder.``."""
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

    def test_input_cast_keeps_force_chain_rule(
        self, matpes_variant: MaceMatpes, cluster: TensorDict
    ) -> None:
        """The boundary cast stays differentiable and the forces come back at the caller's dtype.

        An fp64 caller with ``configure_input_dtype(float32)`` and an fp32
        caller feed the same fp32 model; they agree to fp32 precision, not to
        the bit — the op rebuilds the edge geometry from the fp64 leaf and
        the model casts it at its boundary, so the subtraction rounds once in
        fp64 rather than twice in fp32 (measured 4e-8 eV/Å).
        """
        model = matpes_variant.float().eval()
        original_keys = set(model.state_dict())
        single = make_graph_batch(
            pos=torch.tensor(CLUSTER_POS, dtype=torch.float32),
            Z=torch.tensor(CLUSTER_Z, dtype=torch.long),
            pair_index=full_edge_index(SINGLE_GRAPH),
            batch=torch.zeros(len(CLUSTER_Z), dtype=torch.long),
        )
        expected = model(single)
        model.configure_input_dtype(torch.float32)
        actual = model(cluster)
        assert actual[FORCES_KEY].dtype == torch.float64
        torch.testing.assert_close(
            actual[FORCES_KEY], expected[FORCES_KEY].double(), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(actual[ENERGY_KEY], expected[ENERGY_KEY], atol=1e-6, rtol=1e-6)
        assert set(model.state_dict()) == original_keys

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
        """The energy-only op is ``energy`` — same schema duty."""
        del pair_batch["graphs"]
        out = energy_only_potential(pair_batch)
        assert out["graphs"].batch_size == torch.Size([2])

    def test_forward_consumes_the_periodic_shifts(
        self, matpes_potential: MacePotential, periodic_cluster: TensorDict
    ) -> None:
        """``edges.shifts`` must reach the edge vectors, not be dropped."""
        pos, Z, edge_index, batch, num_graphs = raw_tensors(periodic_cluster)
        edge_diff = periodic_cluster["pairs", "pair_diff"]
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

    # -- the derivative op ----------------------------------------------------

    def test_forward_is_the_same_op_composed_outside(
        self,
        matpes_potential: MacePotential,
        energy_only_potential: MacePotential,
        cluster: TensorDict,
    ) -> None:
        """``wrt`` at construction and ``Derivative(model, wrt=...)`` are one code path."""
        energy_only_potential.load_state_dict(matpes_potential.state_dict(), strict=True)
        composed = Derivative(energy_only_potential, wrt=FORCES)
        with torch.no_grad():
            inside = matpes_potential(cluster.clone())
            outside = composed(cluster.clone())
        assert torch.equal(inside[FORCES_KEY], outside[FORCES_KEY])
        assert torch.equal(inside[ENERGY_KEY], outside[ENERGY_KEY])

    def test_derivative_exposes_the_init_fixed_op(self, matpes_potential: MacePotential) -> None:
        """The training step marks the leaves through ``derivative`` — the op ``forward`` runs."""
        assert matpes_potential.derivative.wrt == (("atoms", "pos"),)

    def test_energy_loss_through_forward_reaches_the_parameters(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """With grad enabled the op leaves the energy attached: ``loss.backward()`` trains."""
        out = matpes_potential(cluster)
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

    def test_energy_core_atom_mask_multiplies_every_per_atom_energy(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """``atom_mask`` scales ``e0`` and the readout node energies before the graph sum.

        An all-false mask therefore prices the whole graph at exactly zero — a
        multiply, never a slice, so the traced shape never depends on the mask.
        """
        pos, Z, edge_index, atom_batch, num_graphs = raw_tensors(cluster)
        with torch.no_grad():
            unmasked = matpes_potential.energy_core(pos, Z, edge_index, atom_batch, num_graphs)
            masked = matpes_potential.energy_core(
                pos,
                Z,
                edge_index,
                atom_batch,
                num_graphs,
                atom_mask=torch.zeros_like(Z, dtype=torch.bool),
            )
        assert float(unmasked.abs().sum()) > 0.0
        assert torch.equal(masked, torch.zeros_like(unmasked))

    def test_a_padded_batch_returns_the_unpadded_energy_and_forces(
        self, matpes_potential: MacePotential, tiny_matpes_spec: MACEMatpesSpec, cluster: TensorDict
    ) -> None:
        """``MoleculePadder`` ghosts add nothing: ``energy`` forwards ``atoms.mask``.

        Ghost atoms carry ``e0`` and a readout energy of their own, and the
        padding edges are self-loops on the first ghost at the dead-edge
        displacement; the per-atom multiply by ``atoms.mask`` and the dead
        geometry together leave the per-graph energy, the real atoms' forces
        and (zero) ghost forces exactly as on the unpadded batch.
        """
        from molix.data import MoleculePadder

        n_atoms = int(cluster["atoms"].batch_size[0])
        n_pairs = int(cluster["pairs"].batch_size[0])
        reference = matpes_potential(cluster.clone())
        padded = MoleculePadder(
            max_atoms=n_atoms + 3, max_pairs=n_pairs + 7, cutoff=tiny_matpes_spec.r_max
        )(cluster.clone())
        out = matpes_potential(padded)
        assert torch.allclose(
            out[ENERGY_KEY], reference[ENERGY_KEY], atol=ENERGY_PARITY_ATOL, rtol=0.0
        )
        assert torch.allclose(
            out[FORCES_KEY][:n_atoms], reference[FORCES_KEY], atol=FORCE_PARITY_ATOL, rtol=0.0
        )
        assert torch.equal(
            out[FORCES_KEY][n_atoms:], torch.zeros(3, 3, dtype=out[FORCES_KEY].dtype)
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

    # -- energy: the seam a container reaches ---------------------------------

    def test_call_energy_dispatches_to_the_energy_not_the_op(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """A container's energy call must reach ``energy``, never the force op."""
        out = call_energy(matpes_potential, cluster)
        assert ("graphs", "energy") in out.keys(include_nested=True)
        assert ("atoms", "forces") not in out.keys(include_nested=True)

    def test_call_energy_keeps_the_energy_on_the_callers_leaf(
        self, matpes_potential: MacePotential, leaf_cluster: TensorDict
    ) -> None:
        """``energy`` must not detach — an op composed outside differentiates it."""
        out = call_energy(matpes_potential, leaf_cluster)
        assert out["graphs", "energy"].requires_grad is True

    # -- monomorphism ---------------------------------------------------------

    def test_forward_is_a_single_op_dispatch(self) -> None:
        """``return self._derivative(batch)`` — every branch resolved in ``__init__``."""
        body = _function_body(MacePotential.forward)
        assert len(body) == 1
        statement = body[0]
        assert isinstance(statement, ast.Return)
        assert isinstance(statement.value, ast.Call)
        assert _dotted_name(statement.value.func) == "self._derivative"

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
        reference = _raw_energy_forces(variant, cluster)
        energy = potential(cluster)["graphs", "energy"]
        assert torch.allclose(energy, reference["energy"], atol=ENERGY_PARITY_ATOL, rtol=0.0), (
            f"bitwise equal: {torch.equal(energy, reference['energy'])}"
        )

    def test_matpes_forces_match_the_variant_energy_forces(
        self, matpes_parity: tuple[MacePotential, MaceMatpes], cluster: TensorDict
    ) -> None:
        """Same for ``F = -dE/dx`` (1e-12 eV/Å)."""
        potential, variant = matpes_parity
        reference = _raw_energy_forces(variant, cluster)
        forces = potential(cluster)["atoms", "forces"]
        assert torch.allclose(forces, reference["forces"], atol=FORCE_PARITY_ATOL, rtol=0.0), (
            f"bitwise equal: {torch.equal(forces, reference['forces'])}"
        )

    def test_matpes_periodic_energy_matches_the_variant_energy_forces(
        self, matpes_parity: tuple[MacePotential, MaceMatpes], periodic_cluster: TensorDict
    ) -> None:
        """The ``edges.shifts`` path is the MD path — it gets its own parity check."""
        potential, variant = matpes_parity
        reference = _raw_energy_forces(variant, periodic_cluster)
        energy = potential(periodic_cluster)["graphs", "energy"]
        assert torch.allclose(energy, reference["energy"], atol=ENERGY_PARITY_ATOL, rtol=0.0), (
            f"bitwise equal: {torch.equal(energy, reference['energy'])}"
        )

    def test_matpes_periodic_forces_match_the_variant_energy_forces(
        self, matpes_parity: tuple[MacePotential, MaceMatpes], periodic_cluster: TensorDict
    ) -> None:
        """``S_ij`` is constant w.r.t. ``pos``, so the forces stay exact."""
        potential, variant = matpes_parity
        reference = _raw_energy_forces(variant, periodic_cluster)
        forces = potential(periodic_cluster)["atoms", "forces"]
        assert torch.allclose(forces, reference["forces"], atol=FORCE_PARITY_ATOL, rtol=0.0), (
            f"bitwise equal: {torch.equal(forces, reference['forces'])}"
        )

    def test_omol_energy_matches_the_variant_energy_forces(
        self, omol_parity: tuple[MacePotential, MaceOmol], omol_cluster: TensorDict
    ) -> None:
        """The conditioned arm: charge 0 / spin 1, the flat model's own defaults."""
        potential, variant = omol_parity
        reference = _raw_energy_forces(variant, omol_cluster)
        energy = potential(omol_cluster)["graphs", "energy"]
        assert torch.allclose(energy, reference["energy"], atol=ENERGY_PARITY_ATOL, rtol=0.0), (
            f"bitwise equal: {torch.equal(energy, reference['energy'])}"
        )

    def test_omol_forces_match_the_variant_energy_forces(
        self, omol_parity: tuple[MacePotential, MaceOmol], omol_cluster: TensorDict
    ) -> None:
        """Non-vacuous by construction — see :func:`.conftest.wake_zero_init_readout`."""
        potential, variant = omol_parity
        reference = _raw_energy_forces(variant, omol_cluster)
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
        # Build the unconditioned twin from a clone so the two batches share
        # nothing but their values.
        bare = make_graph_batch(
            pos=omol_cluster["atoms", "pos"].clone(),
            Z=omol_cluster["atoms", "Z"],
            pair_index=omol_cluster["pairs", "pair_index"],
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

    def test_virial_matches_central_differences_of_the_strained_energy(
        self, matpes_variant: MaceMatpes, cluster: TensorDict
    ) -> None:
        """``W = ∂E/∂ε`` against a finite difference of this model's own energy.

        The virial rides the same backward as the forces, so it costs almost
        nothing next to the convolution — which is exactly why it needs an
        independent oracle rather than a plausibility check. The reference
        strains the same tensors the derivation strains, ``v @ (I + ε_sym)ᵀ``,
        and differences the energy centrally in fp64.
        """
        from molpot.forcefield import ForceField

        pots = ForceField("mace", units="metal").def_pairstyle(matpes_variant).to_potentials()
        cell = torch.eye(3, dtype=torch.float64) * 12.0
        cluster["graphs", "cell"] = cell.unsqueeze(0)
        with torch.no_grad():
            virial = Derivative(pots, wrt=FORCES_AND_VIRIAL)(cluster.clone())[VIRIAL_KEY][0]

        pos = cluster["atoms", "pos"].detach()
        edge_diff = cluster["pairs", "pair_diff"].detach()
        eps = 2.0e-4

        def strained_energy(delta: torch.Tensor) -> float:
            frame = cluster.clone()
            transform = (torch.eye(3, dtype=torch.float64) + 0.5 * (delta + delta.T)).T
            frame["atoms", "pos"] = pos @ transform
            frame["graphs", "cell"] = (cell @ transform).unsqueeze(0)
            # The energy reads its geometry from the imaged edge displacements,
            # so the reference has to strain exactly what the derivation
            # strains — positions alone leave the energy unchanged.
            frame["pairs", "pair_diff"] = edge_diff @ transform
            with torch.no_grad():
                return float(pots.calc_energy(frame).sum())

        # Cauchy convention throughout: ``W_αβ = ∂E/∂ε_αβ`` with no doubled
        # shear, so ``σ = W / V`` is the stress a paper or LAMMPS means. The
        # off-diagonals are the half that used to be twice this.
        for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2)):
            delta = torch.zeros(3, 3, dtype=torch.float64)
            delta[i, j] = eps
            fd = (strained_energy(delta) - strained_energy(-delta)) / (2 * eps)
            assert virial[i, j].item() == pytest.approx(fd, abs=2e-4, rel=1e-3)

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

    @pytest.mark.slow
    def test_forward_compiles_to_one_graph_in_inference(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """The op is the compile boundary: unpacking, leaf, energy and grad are one graph.

        One eager forward first — the documented precondition that spends the
        element gate outside the trace.
        """
        matpes_potential(cluster)
        with torch.no_grad():
            explanation = Compiler.explain(matpes_potential, cluster)
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            explanation.break_sites
        )

    @pytest.mark.slow
    def test_training_trace_breaks_only_at_the_leaf(
        self, matpes_potential: MacePotential, cluster: TensorDict
    ) -> None:
        """With grad enabled the attached forces leak the leaf: stock torch breaks there only."""
        matpes_potential(cluster)
        explanation = Compiler.explain(matpes_potential, cluster)
        assert explanation.graph_break_count >= 1
        assert all("leaked" in reason for reason in explanation.break_reasons), (
            explanation.break_reasons
        )
        assert all(
            site.rsplit(":", 1)[0].endswith("molix/autograd/variables.py")
            for site in explanation.break_sites
        ), explanation.break_sites


class TestMacePotentialEnergy:
    """``energy`` — the seam on the batch schema, and the eager element gate around it."""

    def test_energy_keeps_its_unpacking(
        self,
        matpes_variant: MaceMatpes,
        cluster: TensorDict,
        leaf_cluster: TensorDict,
    ) -> None:
        """``energy`` resolves ``num_graphs``, builds a missing ``graphs`` namespace at
        ``batch_size=[B]`` through its one documented host sync, rejects a batch
        without ``edges.edge_diff``, and writes the energy back **undetached**
        so an op composed outside can differentiate it.
        """
        with torch.enable_grad():
            out = matpes_variant.energy(leaf_cluster)
        assert out["graphs", "energy"].grad_fn is not None

        no_graphs = cluster.clone()
        del no_graphs["graphs"]
        rebuilt = matpes_variant.energy(no_graphs)
        assert rebuilt["graphs"].batch_size == torch.Size([1])

        no_diff = cluster.clone()
        no_diff["pairs"] = no_diff["pairs"].exclude("pair_diff")
        with pytest.raises(ValueError, match="requires edges.edge_diff"):
            matpes_variant.energy(no_diff)

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
        dynamo would replay and advance, and why one eager forward precedes
        any compile. Whether the check *itself* rejects an off-table ``Z``, and
        that it runs once per eager instance, are owned by ``TestMacePotential``
        and ``test_encoder.py``.
        """
        compiling = [True]
        checks: list[torch.Tensor] = []
        validate = matpes_variant.validate_elements
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling[0])
        monkeypatch.setattr(matpes_variant, "validate_elements", lambda Z: checks.append(Z))

        matpes_variant.energy(cluster)
        assert checks == []
        assert matpes_variant._elements_validated is False

        compiling[0] = False
        matpes_variant.energy(cluster)
        assert len(checks) == 1
        assert matpes_variant._elements_validated is True
        validate(cluster["atoms", "Z"])  # the real check still passes this batch

        assert not hasattr(matpes_variant, "_eager_calls")
