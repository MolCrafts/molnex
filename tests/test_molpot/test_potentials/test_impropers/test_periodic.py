"""ImproperPeriodic — Class-I multi-term cosine improper torsion.

Same cosine form as ProperTorsionPeriodic, on improper_index [4, N].
Central atom is at row index 0 (molrs Topology layout [center, i, j, k]).
"""

import math

import pytest
import torch

from molpot.potentials.impropers import ImproperPeriodic


def _cis_pos() -> torch.Tensor:
    """phi = 0 planar fixture (same geometry as proper cis)."""
    return torch.tensor(
        [
            [0.0, 1.0, 0.0],  # atom 0 = i (peripheral)
            [0.0, 0.0, 0.0],  # atom 1 = center
            [1.0, 0.0, 0.0],  # atom 2 = j
            [1.0, 1.0, 0.0],  # atom 3 = k
        ],
        dtype=torch.float64,
    )


def _trans_pos() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # center
            [1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
        dtype=torch.float64,
    )


def _perp_pos() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # center
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def _improper_index_center_first() -> torch.Tensor:
    """COO [4, 1]; molrs layout [center, i, j, k] with center = atom 1."""
    return torch.tensor([[1], [0], [2], [3]], dtype=torch.long)


def _single_term(*, k: float = 1.0, n: int = 1, gamma: float = 0.0):
    return ImproperPeriodic(
        k=torch.tensor([[k]], dtype=torch.float64),
        periodicity=torch.tensor([n], dtype=torch.long),
        phase=torch.tensor([[gamma]], dtype=torch.float64),
    )


class TestImproperPeriodic:
    def test_an_empty_table_takes_its_depth_from_the_table_width(self):
        """The batch that assembles the container may carry no improper at all.

        Depth 1 would survive here and then gather past the ladder on the next
        batch that does bring ``n = 2`` — a CUDA index assert nowhere near the
        cause.
        """
        pot = ImproperPeriodic(
            k=torch.zeros(0, 2, dtype=torch.float64),
            periodicity=torch.zeros(0, 2, dtype=torch.long),
            phase=torch.zeros(0, 2, dtype=torch.float64),
        )
        assert pot.n_max == 2
        later = pot(
            pos=_perp_pos(),
            improper_index=_improper_index_center_first(),
            improper_types=torch.tensor([0], dtype=torch.long),
            k=torch.ones(1, 2, dtype=torch.float64),
            periodicity=torch.tensor([[1, 2]], dtype=torch.long),
            phase=torch.zeros(1, 2, dtype=torch.float64),
        )
        assert torch.isfinite(later).all()

    def test_an_empty_table_is_a_molecule_with_no_impropers(self):
        """Zero rows is ordinary — a small molecule has no improper at all."""
        pot = ImproperPeriodic(
            k=torch.zeros(0, 1, dtype=torch.float64),
            periodicity=torch.zeros(0, 1, dtype=torch.long),
            phase=torch.zeros(0, 1, dtype=torch.float64),
        )
        assert pot.n_max == 1
        energy = pot(
            pos=_perp_pos(),
            improper_index=torch.zeros((4, 0), dtype=torch.long),
            improper_types=torch.zeros(0, dtype=torch.long),
        )
        assert float(energy) == 0.0

    @pytest.mark.slow
    def test_energy_forces_trace_as_one_graph(self):
        """Same gate as the proper torsion: closed-form E + F is one graph."""
        from molix.compile import Compiler

        pot = _single_term(k=1.0, n=2, gamma=0.0)
        kwargs = dict(
            pos=_perp_pos(),
            improper_index=_improper_index_center_first(),
            improper_types=torch.tensor([0], dtype=torch.long),
        )
        pot.energy_forces(**kwargs)
        explanation = Compiler.explain(pot.energy_forces, **kwargs)
        assert (explanation.graph_count, explanation.graph_break_count) == (1, 0), (
            f"break sites: {explanation.break_sites}"
        )

    def test_cis_phi0_k1_gives_energy_2(self):
        pot = _single_term(k=1.0, n=1, gamma=0.0)
        e = pot(
            pos=_cis_pos(),
            improper_index=_improper_index_center_first(),
            improper_types=torch.tensor([0], dtype=torch.long),
        )
        assert math.isclose(float(e), 2.0, rel_tol=1e-10, abs_tol=1e-10)

    def test_trans_phi_pi_gives_energy_0(self):
        pot = _single_term(k=1.0, n=1, gamma=0.0)
        e = pot(
            pos=_trans_pos(),
            improper_index=_improper_index_center_first(),
            improper_types=torch.tensor([0], dtype=torch.long),
        )
        assert math.isclose(float(e), 0.0, rel_tol=1e-10, abs_tol=1e-10)

    def test_perp_phi_half_pi_gives_energy_1(self):
        pot = _single_term(k=1.0, n=1, gamma=0.0)
        e = pot(
            pos=_perp_pos(),
            improper_index=_improper_index_center_first(),
            improper_types=torch.tensor([0], dtype=torch.long),
        )
        assert math.isclose(float(e), 1.0, rel_tol=1e-10, abs_tol=1e-10)

    def test_central_atom_is_row_index_0(self):
        """Lock molrs layout: central lives at row 0 of improper_index.

        Center-first [center, i, j, k] with center=atom 1 yields the cis
        cosine golden; a silent trefoil reorder would break the energy.
        """
        pot = _single_term(k=1.0, n=1, gamma=0.0)
        index = _improper_index_center_first()
        assert int(index[0, 0]) == 1  # central atom id at row 0
        e = pot(
            pos=_cis_pos(),
            improper_index=index,
            improper_types=torch.tensor([0], dtype=torch.long),
        )
        assert math.isclose(float(e), 2.0, rel_tol=1e-10, abs_tol=1e-10)

    def test_empty_improper_index_returns_zero(self):
        pot = _single_term()
        e = pot(
            pos=_cis_pos(),
            improper_index=torch.zeros(4, 0, dtype=torch.long),
            improper_types=torch.zeros(0, dtype=torch.long),
        )
        assert float(e) == 0.0

    def test_rejects_row_major_n_by_4_shape(self):
        pot = _single_term()
        wrong = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)  # [N, 4]
        with pytest.raises(ValueError, match=r"\[4"):
            pot(
                pos=_cis_pos(),
                improper_index=wrong,
                improper_types=torch.tensor([0], dtype=torch.long),
            )

    def test_multi_term_sums_component_energies(self):
        pot = ImproperPeriodic(
            k=torch.tensor([[1.0, 0.5]], dtype=torch.float64),
            periodicity=torch.tensor([1, 2], dtype=torch.long),
            phase=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        )
        e = pot(
            pos=_cis_pos(),
            improper_index=_improper_index_center_first(),
            improper_types=torch.tensor([0], dtype=torch.long),
        )
        # 2.0 + 1.0 = 3.0 at phi=0
        assert math.isclose(float(e), 3.0, rel_tol=1e-10, abs_tol=1e-10)
