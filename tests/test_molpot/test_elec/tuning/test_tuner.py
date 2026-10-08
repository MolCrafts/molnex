"""Tuner input validation and neighbour filtering.

Whether the tuned parameters reach the requested accuracy is a timing-driven
search over full Ewald / PME / P3M sums; that parity check lives in
``regressions/elec-reference-parity.py``.
"""

import pytest
import torch

from molpot.potentials.elec.tuning import tune_ewald, tune_p3m, tune_pme
from molpot.potentials.elec.tuning.tuner import TunerBase
from tests.test_molpot.test_elec.conftest import DEVICES, DTYPES, periodic_neighbor_list

DEFAULT_CUTOFF = 4.4


def system(device=None, dtype=None):
    charges = torch.ones((4, 1), dtype=dtype, device=device)
    cell = torch.eye(3, dtype=dtype, device=device)
    positions = 0.3 * torch.arange(12, dtype=dtype, device=device).reshape((4, 3))

    return charges, cell, positions


def cscl():
    """CsCl in a unit cube: ``(positions, charges (2, 1), cell)``, float32."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    charges = torch.tensor([[-1.0], [1.0]])
    return positions, charges, torch.eye(3)


def neighbor_list(positions, box, cutoff, full_neighbor_list=False):
    pairs, _, dist = periodic_neighbor_list(
        positions.to(torch.float64),
        box.to(torch.float64),
        cutoff,
        full_list=full_neighbor_list,
        periodic=True,
    )
    return pairs.to(positions.device), dist.to(dtype=positions.dtype, device=positions.device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_TunerBase_init(device, dtype):
    """
    Check that `TunerBase` initilizes correctly.

    We are using dummy `neighbor_indices` and `neighbor_distances` to verify types. Have
    to be sure that these dummy variables are initilized correctly.
    """
    charges, cell, positions = system(device, dtype)
    TunerBase(
        charges=charges,
        cell=cell,
        positions=positions,
        cutoff=DEFAULT_CUTOFF,
        calculator=1.0,
        exponent=1,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("full_neighbor_list", [True, False])
def test_cutoff_filter(device, dtype, full_neighbor_list):
    """Filtering a longer-range list to the cutoff equals building it at the cutoff."""
    _, cell, positions = system(device, dtype)
    neighbor_indices, neighbor_distances = neighbor_list(
        positions=positions,
        box=cell,
        cutoff=DEFAULT_CUTOFF * 2,
        full_neighbor_list=full_neighbor_list,
    )
    _, filtered_distances = TunerBase.filter_neighbors(
        DEFAULT_CUTOFF, neighbor_indices, neighbor_distances
    )
    assert filtered_distances.max() < DEFAULT_CUTOFF

    _, distance_from_calculation = neighbor_list(
        positions=positions,
        box=cell,
        cutoff=DEFAULT_CUTOFF,
        full_neighbor_list=full_neighbor_list,
    )
    assert torch.allclose(filtered_distances, distance_from_calculation)


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_accuracy_error(tune):
    pos, charges, cell = cscl()

    match = "'foo' is not a float."
    neighbor_indices, neighbor_distances = neighbor_list(
        positions=pos, box=cell, cutoff=DEFAULT_CUTOFF
    )
    with pytest.raises(ValueError, match=match):
        tune(
            charges=charges,
            cell=cell,
            positions=pos,
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            accuracy="foo",
        )


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_exponent_not_1_error(tune):
    pos, charges, cell = cscl()
    neighbor_indices, neighbor_distances = neighbor_list(
        positions=pos, box=cell, cutoff=DEFAULT_CUTOFF
    )

    match = "Only exponent = 1 is supported but got 2."
    with pytest.raises(NotImplementedError, match=match):
        tune(
            charges=charges,
            cell=cell,
            positions=pos,
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            exponent=2,
        )


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_invalid_shape_positions(tune):
    charges, cell, _ = system()
    match = (
        r"`positions` must be a tensor with shape \[n_atoms, 3\], got tensor with "
        r"shape \[4, 5\]"
    )
    with pytest.raises(ValueError, match=match):
        tune(
            charges=charges,
            cell=cell,
            positions=torch.ones((4, 5)),
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=None,
            neighbor_distances=None,
        )


# Tests for invalid shape, dtype and device of cell
@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_invalid_shape_cell(tune):
    charges, _, positions = system()
    match = r"`cell` must be a tensor with shape \[3, 3\], got tensor with shape \[2, 2\]"
    with pytest.raises(ValueError, match=match):
        tune(
            charges=charges,
            cell=torch.ones([2, 2]),
            positions=positions,
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=None,
            neighbor_distances=None,
        )


@pytest.mark.parametrize("tune", [tune_ewald, tune_pme, tune_p3m])
def test_invalid_dtype_cell(tune):
    charges, _, positions = system()
    match = (
        r"type of `cell` \(torch.float64\) must be same as that of the "
        r"`positions` class \(torch.float32\)"
    )
    with pytest.raises(TypeError, match=match):
        tune(
            charges=charges,
            cell=torch.eye(3, dtype=torch.float64),
            positions=positions,
            cutoff=DEFAULT_CUTOFF,
            neighbor_indices=None,
            neighbor_distances=None,
        )
