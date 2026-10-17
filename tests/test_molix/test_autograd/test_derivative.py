"""Unit tests for molix.autograd.derivative.Derivative.

Mirrors ``src/molix/autograd/derivative.py``. Two closed-form toys:

* a weighted quadratic ``E_g = s Σ_{i∈g} w_i ‖r_i‖²`` (``F_i = -2 s w_i r_i``,
  ``∂E/∂w_i = s ‖r_i‖²``) carrying one parameter ``s``, so a force loss has
  somewhere to go and a second column (``atoms.weight``) can be differentiated
  against;
* a periodic pair spring ``E = ½ k Σ_e ‖d_e‖²`` reading ``edges.pair_diff`` off
  the frame, with explicit ``edges.shifts`` under a cell — the second edge
  reaches its partner through the +x lattice vector, so a virial that failed to
  strain the shifts would miss a term of order the energy itself. In closed
  form ``W_αβ = k Σ_e d_α d_β``.

Units: positions Å, energy eV, forces eV/Å, virial eV. Everything runs in
float64 at ``atol=1e-12, rtol=0`` (the "exact" band for a closed-form
gradient) except the finite-difference oracles, which use ``1e-6``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.autograd import Derivative, Potential
from molix.schema import (
    ATOMIC_ENERGY_KEY,
    ENERGY_KEY,
    FORCES_KEY,
    POS_KEY,
    STRAIN_KEY,
    VIRIAL_KEY,
)

EXACT_ATOL: float = 1e-12
FD_ATOL: float = 1e-6

#: ``pos = arange(12).reshape(4, 3) * 0.1`` Å with unit weights and ``s = 1``:
#: ``F = -2 s w r = -0.2 * arange(12)`` eV/Å. Hard-coded, not recomputed.
EXPECTED_FORCES: list[list[float]] = [
    [-0.0, -0.2, -0.4],
    [-0.6, -0.8, -1.0],
    [-1.2, -1.4, -1.6],
    [-1.8, -2.0, -2.2],
]
#: ``∂E/∂w_i = s ‖r_i‖²`` on the same positions: ``0.01 * (0+1+4, 9+16+25, …)``.
EXPECTED_WEIGHT_GRAD: list[float] = [0.05, 0.5, 1.49, 3.02]

#: Periodic pair spring: two atoms in a 4 Å cube; edge 0 is ``0 → 1`` inside
#: the box, edge 1 is ``1 → 0`` through the +x lattice vector. ``d_0 = (1.5,
#: 0, 0)``, ``d_1 = (2.5, 0, 0)``; with ``k = 2``: ``E = ½ k (1.5² + 2.5²) =
#: 8.5``, ``F_1 = -k (d_0 - d_1) = (2, 0, 0)``, ``W_xx = k (1.5² + 2.5²) = 17``
#: and every other component 0.
PAIR_K: float = 2.0
PAIR_CELL_EDGE: float = 4.0
PAIR_POS: list[list[float]] = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
PAIR_EDGE_INDEX: list[list[int]] = [[0, 1], [1, 0]]
PAIR_SHIFTS: list[list[float]] = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
PAIR_ENERGY: float = 8.5
PAIR_FORCES: list[list[float]] = [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
PAIR_VIRIAL_XX: float = 17.0
#: Symmetric-strain step for the virial oracle (float64: round-off ~1e-10 eV).
STRAIN_STEP: float = 1e-5

FORCES = ("atoms.pos",)
FORCES_AND_VIRIAL = ("atoms.pos", "graphs.strain")


class Quadratic(nn.Module):
    """``E_g = s Σ_{i∈g} w_i ‖r_i‖²`` written onto the frame; counts calls.

    Args:
        scale: Initial value of the one parameter ``s`` (eV/Å²).
        write_atomic: Also write the per-atom energies at ``atoms.energy``.
    """

    def __init__(self, *, scale: float = 1.0, write_atomic: bool = True) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float64))
        self.write_atomic = write_atomic
        self.calls = 0

    def forward(self, frame: TensorDict) -> TensorDict:
        self.calls += 1
        pos = frame["atoms", "pos"]
        atomic = self.scale.to(pos.dtype) * frame["atoms", "weight"] * (pos * pos).sum(dim=-1)
        energy = torch.zeros(frame["graphs"].batch_size[0], dtype=pos.dtype, device=pos.device)
        frame[ENERGY_KEY] = energy.index_add(0, frame["atoms", "batch"], atomic)
        if self.write_atomic:
            frame[ATOMIC_ENERGY_KEY] = atomic
        return frame

    def total_energy(self, pos: torch.Tensor) -> torch.Tensor:
        """Scalar total at unit weights — the finite-difference oracle."""
        return self.scale.to(pos.dtype) * (pos * pos).sum()


def quadratic_frame(n_atoms: int = 4, n_graphs: int = 2) -> TensorDict:
    """Post-collate batch with ``pos = arange(3 N).reshape(N, 3) * 0.1`` Å, unit weights."""
    per_graph = n_atoms // n_graphs
    pos = torch.arange(n_atoms * 3, dtype=torch.float64).reshape(n_atoms, 3) * 0.1
    return TensorDict(
        atoms=TensorDict(
            pos=pos,
            weight=torch.ones(n_atoms, dtype=torch.float64),
            batch=torch.arange(n_graphs).repeat_interleave(per_graph),
            batch_size=[n_atoms],
        ),
        graphs=TensorDict(batch_size=[n_graphs]),
        batch_size=[],
    )


def _pair_energy(pair_diff: torch.Tensor) -> torch.Tensor:
    """``½ k Σ_e ‖d_e‖²`` as a scalar — the raw-tensor form of the pair spring."""
    return 0.5 * PAIR_K * pair_diff.pow(2).sum()


class PairSpring:
    """The pair spring read off the frame's ``edges.pair_diff``; a plain callable, not a module."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frame: TensorDict) -> TensorDict:
        self.calls += 1
        frame[ENERGY_KEY] = _pair_energy(frame["pairs", "pair_diff"]).reshape(1)
        return frame


def pair_frame() -> TensorDict:
    """The periodic pair-spring batch: two atoms, two edges, one cell."""
    pos = torch.tensor(PAIR_POS, dtype=torch.float64)
    pair_index = torch.tensor(PAIR_EDGE_INDEX, dtype=torch.long)
    shifts = torch.tensor(PAIR_SHIFTS, dtype=torch.float64)
    diff = pos[pair_index[:, 1]] - pos[pair_index[:, 0]] + shifts
    return TensorDict(
        atoms=TensorDict(pos=pos, batch=torch.zeros(2, dtype=torch.long), batch_size=[2]),
        pairs=TensorDict(
            pair_index=pair_index,
            pair_diff=diff,
            pair_dist=diff.norm(dim=-1),
            shifts=shifts,
            batch_size=[2],
        ),
        graphs=TensorDict(
            cell=(torch.eye(3, dtype=torch.float64) * PAIR_CELL_EDGE).unsqueeze(0),
            batch_size=[1],
        ),
        batch_size=[],
    )


def _strained_pair_energy(delta: torch.Tensor) -> float:
    """The pair spring at strain ``delta``: positions, cell and shifts all strained."""
    transform = (torch.eye(3, dtype=torch.float64) + 0.5 * (delta + delta.T)).T
    pos = torch.tensor(PAIR_POS, dtype=torch.float64) @ transform
    shifts = torch.tensor(PAIR_SHIFTS, dtype=torch.float64) @ transform
    pair_index = torch.tensor(PAIR_EDGE_INDEX, dtype=torch.long)
    return float(_pair_energy(pos[pair_index[:, 1]] - pos[pair_index[:, 0]] + shifts))


def central_difference_forces(
    energy_fn: Callable[[torch.Tensor], torch.Tensor], pos: torch.Tensor, *, h: float = 1e-5
) -> torch.Tensor:
    """``F = -dE/dr`` by central differences — the domain oracle for the forces."""
    base = pos.detach().clone()
    forces = torch.zeros_like(base)
    with torch.no_grad():
        for atom in range(base.shape[0]):
            for axis in range(base.shape[1]):
                plus = base.clone()
                plus[atom, axis] += h
                minus = base.clone()
                minus[atom, axis] -= h
                forces[atom, axis] = -(energy_fn(plus) - energy_fn(minus)) / (2.0 * h)
    return forces


def central_difference_virial(
    energy_at: Callable[[torch.Tensor], float], *, h: float = STRAIN_STEP
) -> torch.Tensor:
    """``W = dE/dε`` by six symmetric-strain central differences (Cauchy convention).

    A shear moves ``ε_ab`` and ``ε_ba`` together, so the quotient counts that
    pair twice and the off-diagonals are halved.
    """
    virial = torch.zeros(3, 3, dtype=torch.float64)
    for row in range(3):
        for column in range(row, 3):
            delta = torch.zeros(3, 3, dtype=torch.float64)
            delta[row, column] = h
            delta[column, row] = h
            derivative = (energy_at(delta) - energy_at(-delta)) / (2.0 * h)
            virial[row, column] = derivative / (1.0 if row == column else 2.0)
            virial[column, row] = virial[row, column]
    return virial


def _expected_forces() -> torch.Tensor:
    return torch.tensor(EXPECTED_FORCES, dtype=torch.float64)


def _expected_pair_virial() -> torch.Tensor:
    expected = torch.zeros(1, 3, 3, dtype=torch.float64)
    expected[0, 0, 0] = PAIR_VIRIAL_XX
    return expected


class Spring(nn.Module):
    """The pair spring with its constant as the one parameter — a force loss reaches it."""

    def __init__(self) -> None:
        super().__init__()
        self.k = nn.Parameter(torch.tensor(PAIR_K, dtype=torch.float64))

    def forward(self, frame: TensorDict) -> TensorDict:
        pair_diff = frame["pairs", "pair_diff"]
        frame[ENERGY_KEY] = (0.5 * self.k * pair_diff.pow(2).sum()).reshape(1)
        return frame


def _spring_loss(frame: TensorDict, wrt: tuple[str, ...]) -> torch.Tensor:
    """Energy plus squared forces (plus squared virial when it is a variable)."""
    total = frame[ENERGY_KEY].sum() + frame[FORCES_KEY].pow(2).sum()
    if "graphs.strain" in wrt:
        total = total + frame[VIRIAL_KEY].pow(2).sum()
    return total


class TestDerivative:
    """``Derivative`` — one energy forward, one ``autograd.grad``, in place."""

    def test_forces_match_the_hardcoded_analytic_gradient(self) -> None:
        frame = Derivative(Quadratic(), wrt=FORCES)(quadratic_frame())
        torch.testing.assert_close(frame[FORCES_KEY], _expected_forces(), atol=EXACT_ATOL, rtol=0.0)

    def test_force_only_wrt_does_not_read_graphs_cell(self) -> None:
        """Force-only ``wrt`` must not probe ``graphs`` (GH200 2367461)."""
        frame = quadratic_frame()
        frame["graphs", "task"] = torch.zeros(2, dtype=torch.long)
        frame["graphs", "energy"] = torch.zeros(2, dtype=torch.float64)
        Derivative(Quadratic(write_atomic=False), wrt=FORCES)(frame)
        torch.testing.assert_close(frame[FORCES_KEY], _expected_forces(), atol=EXACT_ATOL, rtol=0.0)

    def test_forces_match_when_graphs_keys_are_inserted_in_another_order(self) -> None:
        """Collate key order must not change the derivative (GH200 2366877)."""
        op = Derivative(Quadratic(write_atomic=False), wrt=FORCES)
        first = quadratic_frame()
        first["graphs", "energy"] = torch.zeros(2, dtype=torch.float64)
        first["graphs", "total_charge"] = torch.zeros(2, dtype=torch.float64)
        second = quadratic_frame()
        second["graphs", "total_charge"] = torch.zeros(2, dtype=torch.float64)
        second["graphs", "energy"] = torch.zeros(2, dtype=torch.float64)
        assert list(first["graphs"].keys()) != list(second["graphs"].keys())
        torch.testing.assert_close(
            op(first)[FORCES_KEY], op(second)[FORCES_KEY], atol=EXACT_ATOL, rtol=0.0
        )

    def test_forces_match_central_finite_differences(self) -> None:
        toy = Quadratic()
        frame = quadratic_frame()
        pos = frame[POS_KEY].detach().clone()

        Derivative(toy, wrt=FORCES)(frame)
        numerical = central_difference_forces(toy.total_energy, pos)

        assert float((frame[FORCES_KEY].detach() - numerical).abs().max()) < FD_ATOL

    def test_virial_matches_the_closed_form_on_a_periodic_pair(self) -> None:
        frame = Derivative(PairSpring(), wrt=FORCES_AND_VIRIAL)(pair_frame())
        torch.testing.assert_close(
            frame[VIRIAL_KEY], _expected_pair_virial(), atol=EXACT_ATOL, rtol=0.0
        )

    def test_virial_matches_a_central_strain_difference(self) -> None:
        frame = Derivative(PairSpring(), wrt=FORCES_AND_VIRIAL)(pair_frame())
        numerical = central_difference_virial(_strained_pair_energy)
        assert float((frame[VIRIAL_KEY][0].detach() - numerical).abs().max()) < FD_ATOL

    def test_straining_at_zero_leaves_energy_and_forces_exact(self) -> None:
        frame = Derivative(PairSpring(), wrt=FORCES_AND_VIRIAL)(pair_frame())
        torch.testing.assert_close(
            frame[ENERGY_KEY],
            torch.tensor([PAIR_ENERGY], dtype=torch.float64),
            atol=EXACT_ATOL,
            rtol=0.0,
        )
        torch.testing.assert_close(
            frame[FORCES_KEY],
            torch.tensor(PAIR_FORCES, dtype=torch.float64),
            atol=EXACT_ATOL,
            rtol=0.0,
        )

    def test_energy_only_runs_the_energy_and_writes_nothing_else(self) -> None:
        """``wrt=()`` is the energy alone: no leaf, no forces, the frame otherwise untouched."""
        toy = Quadratic()
        frame = quadratic_frame()
        pos = frame[POS_KEY]

        out = Derivative(toy)(frame)

        assert out is frame
        assert toy.calls == 1
        assert frame[POS_KEY] is pos
        assert ENERGY_KEY in frame.keys(include_nested=True)
        assert FORCES_KEY not in frame.keys(include_nested=True)

    @pytest.mark.parametrize("wrt", [FORCES, FORCES_AND_VIRIAL])
    def test_frame_comes_back_as_handed_in(self, wrt: tuple[str, ...]) -> None:
        """The caller's own tensor objects are back, and none of them requires grad."""
        frame = pair_frame()
        originals = {
            key: frame[key]
            for key in (
                POS_KEY,
                ("graphs", "cell"),
                ("pairs", "pair_diff"),
                ("pairs", "pair_dist"),
                ("pairs", "shifts"),
            )
        }

        out = Derivative(PairSpring(), wrt=wrt)(frame)

        assert out is frame
        for key, tensor in originals.items():
            assert frame[key] is tensor, key
            assert not frame[key].requires_grad, key

    def test_the_strain_column_is_created_once_and_left_on_the_frame(self) -> None:
        """``graphs.strain`` is the variable: a ``(1, 3, 3)`` zero column, or the caller's."""
        created = pair_frame()
        Derivative(PairSpring(), wrt=FORCES_AND_VIRIAL)(created)
        strain = created[STRAIN_KEY]
        assert strain.shape == (1, 3, 3)
        assert not strain.requires_grad
        torch.testing.assert_close(strain, torch.zeros(1, 3, 3, dtype=torch.float64))

        given = pair_frame()
        column = torch.zeros(1, 3, 3, dtype=torch.float64)
        given[STRAIN_KEY] = column
        Derivative(PairSpring(), wrt=FORCES_AND_VIRIAL)(given)
        assert given[STRAIN_KEY] is column

    def test_geometry_is_rebuilt_from_the_leaf_not_the_callers_graph(self) -> None:
        """A cached ``pair_diff`` hanging off a foreign graph is re-derived, not trusted."""
        frame = pair_frame()
        pair_index = frame["pairs", "pair_index"]
        branch = frame[POS_KEY].clone().requires_grad_(True)
        frame["pairs", "pair_diff"] = (
            branch[pair_index[:, 1]] - branch[pair_index[:, 0]] + frame["pairs", "shifts"]
        )

        out = Derivative(PairSpring(), wrt=FORCES)(frame)

        torch.testing.assert_close(
            out[FORCES_KEY].detach(),
            torch.tensor(PAIR_FORCES, dtype=torch.float64),
            atol=EXACT_ATOL,
            rtol=0.0,
        )
        (through_branch,) = torch.autograd.grad(out[FORCES_KEY].sum(), branch, allow_unused=True)
        assert through_branch is None

    def test_a_live_leaf_column_is_differentiated_as_is_and_left_in_the_frame(self) -> None:
        """The op makes a leaf unless the column already is one: then that leaf is the variable.

        A caller that marks ``atoms.pos`` before the traced region owns it — it
        stays in the frame after the call, and the forces hang off it.
        """
        frame = quadratic_frame()
        leaf = frame[POS_KEY].detach().requires_grad_(True)
        frame[POS_KEY] = leaf

        out = Derivative(Quadratic(), wrt=FORCES)(frame)

        assert out[POS_KEY] is leaf
        torch.testing.assert_close(out[FORCES_KEY], _expected_forces(), atol=EXACT_ATOL, rtol=0.0)
        (through_leaf,) = torch.autograd.grad(out[FORCES_KEY].sum(), leaf, allow_unused=True)
        assert through_leaf is not None

    def test_an_attached_non_leaf_column_still_gets_a_fresh_leaf(self) -> None:
        """Only a *leaf* is reused; an attached column is detached and nothing flows back."""
        frame = quadratic_frame()
        upstream = frame[POS_KEY].clone().requires_grad_(True)
        column = upstream * 1.0
        frame[POS_KEY] = column

        out = Derivative(Quadratic(), wrt=FORCES)(frame)

        assert out[POS_KEY] is column
        torch.testing.assert_close(out[FORCES_KEY], _expected_forces(), atol=EXACT_ATOL, rtol=0.0)
        (through_upstream,) = torch.autograd.grad(
            out[FORCES_KEY].sum(), upstream, allow_unused=True
        )
        assert through_upstream is None

    def test_mark_leaves_replaces_the_columns_by_fresh_leaves(self) -> None:
        """Each call: new leaf objects carrying the columns' values; ``graphs.strain`` created.

        Fresh per call, so a ``.grad`` never accumulates across training steps.
        """
        op = Derivative(PairSpring(), wrt=FORCES_AND_VIRIAL)
        frame = pair_frame()
        pos = frame[POS_KEY]

        out = op.mark_leaves(frame)

        assert out is frame
        first = {key: frame[key] for key in (POS_KEY, STRAIN_KEY)}
        assert first[POS_KEY] is not pos
        assert torch.equal(first[POS_KEY], pos)
        assert first[STRAIN_KEY].shape == (1, 3, 3)
        assert torch.equal(first[STRAIN_KEY], torch.zeros(1, 3, 3, dtype=torch.float64))
        for key, column in first.items():
            assert column.is_leaf and column.requires_grad, key

        op.mark_leaves(frame)
        for key, column in first.items():
            assert frame[key] is not column, key
            assert frame[key].is_leaf and frame[key].requires_grad, key

    def test_a_marked_strain_column_is_the_variable_itself(self) -> None:
        """The ``(1, 3, 3)`` column marked by ``mark_leaves`` is differentiated as is.

        A reshape of a leaf is not a leaf, so the strain leaf stays the column
        and only ``apply`` sees it as ``(3, 3)``; the closed-form virial is
        unchanged and hangs off the caller's leaf.
        """
        op = Derivative(PairSpring(), wrt=FORCES_AND_VIRIAL)
        frame = op.mark_leaves(pair_frame())
        strain = frame[STRAIN_KEY]

        op(frame)

        assert frame[STRAIN_KEY] is strain
        torch.testing.assert_close(
            frame[VIRIAL_KEY], _expected_pair_virial(), atol=EXACT_ATOL, rtol=0.0
        )
        torch.testing.assert_close(
            frame[FORCES_KEY],
            torch.tensor(PAIR_FORCES, dtype=torch.float64),
            atol=EXACT_ATOL,
            rtol=0.0,
        )
        (through_strain,) = torch.autograd.grad(frame[VIRIAL_KEY].sum(), strain, allow_unused=True)
        assert through_strain is not None

    def test_mark_leaves_is_harmless_under_no_grad(self) -> None:
        """Inference never needs the marking; done anyway, the outputs are still detached."""
        op = Derivative(Quadratic(), wrt=FORCES)
        with torch.no_grad():
            frame = op(op.mark_leaves(quadratic_frame()))
        torch.testing.assert_close(frame[FORCES_KEY], _expected_forces(), atol=EXACT_ATOL, rtol=0.0)
        assert frame[FORCES_KEY].requires_grad is False
        assert frame[ENERGY_KEY].requires_grad is False

    def test_outputs_are_detached_under_no_grad(self) -> None:
        with torch.no_grad():
            frame = Derivative(Quadratic(), wrt=FORCES)(quadratic_frame())
        assert frame[ENERGY_KEY].requires_grad is False
        assert frame[ATOMIC_ENERGY_KEY].requires_grad is False
        assert frame[FORCES_KEY].requires_grad is False

    def test_energies_stay_attached_when_grad_is_enabled(self) -> None:
        frame = Derivative(Quadratic(), wrt=FORCES)(quadratic_frame())
        assert frame[ENERGY_KEY].requires_grad is True
        assert frame[ATOMIC_ENERGY_KEY].requires_grad is True

    def test_force_loss_reaches_the_parameters_when_grad_is_enabled(self) -> None:
        toy = Quadratic()
        frame = Derivative(toy, wrt=FORCES)(quadratic_frame())
        frame[FORCES_KEY].pow(2).mean().backward()
        assert toy.scale.grad is not None
        assert toy.scale.grad.abs().item() > 0.0

    @pytest.mark.parametrize("wrt", [FORCES, FORCES_AND_VIRIAL])
    def test_evaluates_the_energy_once(self, wrt: tuple[str, ...]) -> None:
        spring = PairSpring()
        Derivative(spring, wrt=wrt)(pair_frame())
        assert spring.calls == 1

    def test_strain_requires_a_cell(self) -> None:
        frame = pair_frame()
        del frame["graphs", "cell"]
        with pytest.raises(ValueError, match=r"graphs\.cell"):
            Derivative(PairSpring(), wrt=FORCES_AND_VIRIAL)(frame)

    def test_strain_refuses_a_multi_graph_batch(self) -> None:
        """One strain leaf is one lattice; two graphs under it would share a box."""
        frame = pair_frame()
        frame["graphs"] = TensorDict(
            cell=(torch.eye(3, dtype=torch.float64) * PAIR_CELL_EDGE).expand(2, 3, 3).clone(),
            batch_size=[2],
        )
        with pytest.raises(ValueError, match="one graph"):
            Derivative(PairSpring(), wrt=FORCES_AND_VIRIAL)(frame)

    @pytest.mark.parametrize("wrt", [("graphs.strain", "atoms.pos"), ("graphs.strain",)])
    def test_strain_must_follow_pos(self, wrt: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match=r"atoms\.pos"):
            Derivative(PairSpring(), wrt=wrt)

    def test_a_repeated_variable_is_refused(self) -> None:
        with pytest.raises(ValueError, match="once"):
            Derivative(Quadratic(), wrt=("atoms.pos", ("atoms", "pos")))

    def test_keys_are_normalised_to_tuples_and_outputs_follow_the_table(self) -> None:
        dotted = Derivative(PairSpring(), wrt=FORCES_AND_VIRIAL)
        tupled = Derivative(PairSpring(), wrt=(("atoms", "pos"), ("graphs", "strain")))
        assert dotted.wrt == tupled.wrt == (POS_KEY, STRAIN_KEY)
        assert dotted.outputs == tupled.outputs == (FORCES_KEY, VIRIAL_KEY)
        assert dotted.of == ENERGY_KEY

    def test_a_generic_column_lands_under_its_grad_key(self) -> None:
        """Any other column: the leaf replaces it, ``∂E/∂column`` goes to ``<ns>.<name>_grad``."""
        frame = quadratic_frame()
        weight = frame["atoms", "weight"]

        op = Derivative(Quadratic(), wrt=("atoms.weight",))
        op(frame)

        assert op.outputs == (("atoms", "weight_grad"),)
        torch.testing.assert_close(
            frame["atoms", "weight_grad"],
            torch.tensor(EXPECTED_WEIGHT_GRAD, dtype=torch.float64),
            atol=EXACT_ATOL,
            rtol=0.0,
        )
        assert frame["atoms", "weight"] is weight
        assert not weight.requires_grad

    def test_outputs_override_the_table_key_by_key(self) -> None:
        """``outputs=`` renames where a derivative lands; the sign stays the variable's."""
        frame = quadratic_frame()
        op = Derivative(
            Quadratic(), wrt=("atoms.pos", "atoms.weight"), outputs=(None, "atoms.dE_dw")
        )
        op(frame)
        assert op.outputs == (FORCES_KEY, ("atoms", "dE_dw"))
        torch.testing.assert_close(frame[FORCES_KEY], _expected_forces(), atol=EXACT_ATOL, rtol=0.0)
        torch.testing.assert_close(
            frame["atoms", "dE_dw"],
            torch.tensor(EXPECTED_WEIGHT_GRAD, dtype=torch.float64),
            atol=EXACT_ATOL,
            rtol=0.0,
        )

    def test_outputs_must_pair_with_wrt(self) -> None:
        with pytest.raises(ValueError, match="outputs"):
            Derivative(Quadratic(), wrt=FORCES, outputs=("atoms.a", "atoms.b"))

    def test_a_module_energy_is_registered_and_a_bound_method_is_not(self) -> None:
        """Parameters flow through the op under ``energy.``; a bound method adds none."""
        toy = Quadratic()
        over_module = Derivative(toy, wrt=FORCES)
        assert dict(over_module.named_children()) == {"energy": toy}
        assert list(over_module.state_dict()) == ["energy.scale"]

        over_method = Derivative(toy.forward, wrt=FORCES)
        assert list(over_method.children()) == []
        assert list(over_method.state_dict()) == []
        torch.testing.assert_close(
            over_method(quadratic_frame())[FORCES_KEY],
            _expected_forces(),
            atol=EXACT_ATOL,
            rtol=0.0,
        )

    def test_an_op_over_a_potentials_energy_matches_its_forward(self) -> None:
        """``Derivative(pot.energy, wrt=...)`` and ``pot.forward`` are one code path.

        The op calls exactly what it is handed; a potential is passed as its
        ``energy`` method, never as the object, whose ``forward`` is an op of
        its own.
        """
        toy = Quadratic()

        class _Potential(Potential):
            def __init__(self) -> None:
                super().__init__()
                self.toy = toy
                self._derivative = Derivative(self.energy, wrt=FORCES)

            def energy(self, frame: TensorDict) -> TensorDict:
                return self.toy(frame)

        pot = _Potential()
        outside = Derivative(pot.energy, wrt=FORCES)(quadratic_frame())
        inside = pot(quadratic_frame())
        assert toy.calls == 2
        torch.testing.assert_close(
            outside[FORCES_KEY], inside[FORCES_KEY], atol=EXACT_ATOL, rtol=0.0
        )

    def test_a_non_callable_energy_is_refused(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            Derivative(object(), wrt=FORCES)  # type: ignore[arg-type]

    @pytest.mark.slow
    @pytest.mark.parametrize("wrt", [FORCES, FORCES_AND_VIRIAL])
    def test_inference_traces_to_one_graph(self, wrt: tuple[str, ...]) -> None:
        """Leaves, strain, geometry, energy, ``autograd.grad`` and the writes: one dynamo graph.

        Inference shape (ambient ``no_grad``): every output is detached and the
        caller's buffers are back, so no ``requires_grad`` intermediate escapes
        the traced region.
        """
        from molix.compile import Compiler

        op = Derivative(PairSpring(), wrt=wrt)
        with torch.no_grad():
            summary = Compiler.explain(op, pair_frame())
        assert (summary.graph_count, summary.graph_break_count) == (1, 0), summary.break_sites

    @pytest.mark.slow
    @pytest.mark.parametrize("wrt", [FORCES, FORCES_AND_VIRIAL])
    def test_training_backward_through_the_compiled_op_matches_eager(
        self, wrt: tuple[str, ...]
    ) -> None:
        """The leaves are made before ``enable_grad``: the region keeps its ``autograd.grad``.

        Made inside that block, the stock-torch leak break resumes with the
        inner grad outside the compiled region and ``loss.backward()`` is a
        double backward (``aot_autograd does not currently support double
        backward``). Pinned at the op, where the leaf is placed.
        """
        from molix.compile import Compiler

        spring = Spring()
        _spring_loss(Derivative(spring, wrt=wrt)(pair_frame()), wrt).backward()
        expected = spring.k.grad.detach().clone()
        spring.k.grad = None

        compiled = Compiler(backend="aot_eager")(Derivative(spring, wrt=wrt))
        _spring_loss(compiled(pair_frame()), wrt).backward()

        assert expected.abs().item() > 0.0
        torch.testing.assert_close(spring.k.grad, expected, atol=EXACT_ATOL, rtol=0.0)

    @pytest.mark.slow
    def test_training_trace_breaks_only_at_the_leaf(self) -> None:
        """Unmarked, attached outputs leak the leaf: stock torch breaks there and nowhere else.

        Until pytorch/pytorch#196653 lands, returning a tensor that still
        requires grad and derives from a leaf made inside the traced function
        is a graph break, reported against the module that owns the seam —
        one reason, at ``leaf`` in ``molix/autograd/variables.py``. A caller
        that wants one graph marks the columns first (the two tests below).
        """
        from molix.compile import Compiler

        summary = Compiler.explain(Derivative(Quadratic(), wrt=FORCES), quadratic_frame())
        assert len(summary.break_reasons) == 1, summary.break_reasons
        assert "leaked" in summary.break_reasons[0]
        site_file = summary.break_sites[0].rsplit(":", 1)[0]
        assert site_file.endswith("molix/autograd/variables.py"), summary.break_sites

    @pytest.mark.slow
    @pytest.mark.parametrize("wrt", [FORCES, FORCES_AND_VIRIAL])
    def test_training_traces_to_one_graph_after_mark_leaves(self, wrt: tuple[str, ...]) -> None:
        """The caller-owned leaf is a graph input: on stock torch the attached outputs
        derive from an input, nothing leaks, and the training op is one graph."""
        from molix.compile import Compiler

        op = Derivative(PairSpring(), wrt=wrt)
        summary = Compiler.explain(op, op.mark_leaves(pair_frame()))
        assert (summary.graph_count, summary.graph_break_count) == (1, 0), summary.break_sites

    @pytest.mark.slow
    @pytest.mark.parametrize("wrt", [FORCES, FORCES_AND_VIRIAL])
    def test_fullgraph_training_backward_after_mark_leaves_matches_eager(
        self, wrt: tuple[str, ...]
    ) -> None:
        """``Compiler(fullgraph=True)`` on stock torch: parameter gradients equal eager to 1e-12."""
        from molix.compile import Compiler

        spring = Spring()
        _spring_loss(Derivative(spring, wrt=wrt)(pair_frame()), wrt).backward()
        expected = spring.k.grad.detach().clone()
        spring.k.grad = None

        op = Derivative(spring, wrt=wrt)
        compiled = Compiler(fullgraph=True, backend="aot_eager")(op)
        _spring_loss(compiled(op.mark_leaves(pair_frame())), wrt).backward()

        assert expected.abs().item() > 0.0
        torch.testing.assert_close(spring.k.grad, expected, atol=EXACT_ATOL, rtol=0.0)


class _Gained(nn.Module):
    """``E_g = gain · s Σ_{i∈g} w_i ‖r_i‖²`` with ``gain`` a keyword argument of the energy.

    What a training step hands in when it predicted the parameters once and
    differentiates the energy against the geometry with them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.seen: list[torch.Tensor] = []

    def forward(self, frame: TensorDict, *, gain: torch.Tensor) -> TensorDict:
        self.seen.append(gain)
        pos = frame["atoms", "pos"]
        atomic = gain * self.scale * frame["atoms", "weight"] * (pos * pos).sum(dim=-1)
        energy = torch.zeros(frame["graphs"].batch_size[0], dtype=pos.dtype)
        frame[ENERGY_KEY] = energy.index_add(0, frame["atoms", "batch"], atomic)
        return frame


class TestDerivativeEnergyKwargs:
    """``op(frame, **kwargs)`` is ``energy(frame, **kwargs)``; the derivative stays on ``wrt``."""

    def test_kwargs_reach_the_energy_and_scale_the_forces(self) -> None:
        toy = _Gained()
        gain = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

        frame = Derivative(toy, wrt=FORCES)(quadratic_frame(), gain=gain)

        assert toy.seen == [gain]
        torch.testing.assert_close(
            frame[FORCES_KEY], 2.0 * _expected_forces(), atol=EXACT_ATOL, rtol=0.0
        )

    def test_the_forces_keep_the_graph_the_kwargs_carry(self) -> None:
        """A force loss reaches the tables the step predicted outside the op."""
        gain = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        frame = Derivative(_Gained(), wrt=FORCES)(quadratic_frame(), gain=gain)

        frame[FORCES_KEY].pow(2).sum().backward()

        # F = -2 gain s w r, so Σ F² = 4 gain² Σ (s w r)²: d/dgain = 8 gain Σ r² at s = w = 1.
        pos = quadratic_frame()[POS_KEY]
        torch.testing.assert_close(
            gain.grad, 8.0 * gain.detach() * (pos * pos).sum(), atol=EXACT_ATOL, rtol=0.0
        )

    def test_energy_only_hands_the_kwargs_on_too(self) -> None:
        toy = _Gained()
        gain = torch.tensor(3.0, dtype=torch.float64)

        frame = Derivative(toy)(quadratic_frame(), gain=gain)

        assert toy.seen == [gain]
        torch.testing.assert_close(
            frame[ENERGY_KEY].sum(), 3.0 * (quadratic_frame()[POS_KEY] ** 2).sum()
        )
