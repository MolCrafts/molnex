"""PiNN's atomic dress: per-element energies by least squares on element counts.

Each training frame ``n`` is one row: ``X[n, k]`` is the number of atoms of
element ``elems[k]`` and ``y[n]`` its energy label. With no intercept,

``beta = pinv(X^T X) X^T y``

and ``c[elems[k]] = beta[k]``. The cutoff of ``pinv`` is pinned at
``rtol=1e-15`` (NumPy 1.x's ``rcond`` default, which PiNN's ``np.linalg.pinv``
uses) rather than torch's size-dependent default, so a rank-deficient fit — and
with it the table's digest — does not move between torch builds.

Reference:
    Shao et al. "PiNN: A Python Library for Building Atomic Neural Networks of
    Molecules and Materials" J. Chem. Inf. Model. 60, 1184 (2020).
    https://doi.org/10.1021/acs.jcim.9b00994 (``pinn/utils.py``
    ``get_atomic_dress``, Teoroo-CMC/PiNN ``b592996``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from molpy import Frame

from molix.data.collate import ATOMIC_NUMBER_COLUMNS, GRAPH_LEAF
from molix.data.dataset import BaseDataset
from molix.data.dress import AtomicDress

#: ``pinv`` singular-value cutoff relative to the largest; see the module docstring.
PINV_RTOL = 1e-15


class LeastSquaresDress:
    """Fit ``c[Z]`` to a training split by no-intercept least squares.

    ``e0`` is identified only on the span of the training compositions: on a
    fixed-composition set (one molecule's conformers) the Gram matrix is
    rank-deficient and ``pinv`` returns the minimum-norm split, which may
    price a different composition arbitrarily. An element of *elems* that no
    training frame contains is refused rather than fitted to 0.

    Args:
        elems: Atomic numbers, the column order of ``X``. The resulting table
            supports exactly these.

    Raises:
        ValueError: If *elems* is empty, has a negative or a repeated entry.
    """

    def __init__(self, elems: Sequence[int]) -> None:
        elems = tuple(int(z) for z in elems)
        if not elems:
            raise ValueError("LeastSquaresDress needs at least one element")
        if min(elems) < 0:
            raise ValueError(f"LeastSquaresDress elements must be >= 0 (got {list(elems)})")
        if len(set(elems)) != len(elems):
            raise ValueError(f"LeastSquaresDress elements repeat (got {list(elems)})")
        self.elems = elems

    def fit(self, counts: torch.Tensor, energy: torch.Tensor) -> AtomicDress:
        """Solve for ``c[Z]`` from a count matrix and its energies.

        Args:
            counts: ``(N, K)`` integer element counts, column ``k`` counting
                ``elems[k]``; one row per training frame.
            energy: ``(N,)`` energy labels, in the unit the table will carry
                (that of ``graphs.energy``, eV).

        Returns:
            The dense Z-indexed table of length ``max(elems) + 1``, supported
            exactly at *elems*.

        Raises:
            ValueError: On a shape mismatch, no rows, a non-integer *counts*,
                a negative count, or an element with no atom in any row.
        """
        if counts.ndim != 2 or counts.shape[1] != len(self.elems):
            raise ValueError(
                f"counts must be (N, {len(self.elems)}) for elems {list(self.elems)} "
                f"(got shape {tuple(counts.shape)})"
            )
        n_rows = counts.shape[0]
        if n_rows < 1:
            raise ValueError("LeastSquaresDress.fit needs at least one row")
        if energy.shape != (n_rows,):
            raise ValueError(
                f"energy must be ({n_rows},) to match counts (got shape {tuple(energy.shape)})"
            )
        if counts.dtype.is_floating_point or counts.dtype.is_complex or counts.dtype == torch.bool:
            raise ValueError(f"counts must be an integer tensor (got {counts.dtype})")
        counts = counts.to(dtype=torch.int64)
        if bool((counts < 0).any()):
            raise ValueError("counts must be non-negative")
        totals = counts.sum(dim=0).tolist()
        absent = [z for z, total in zip(self.elems, totals, strict=True) if total == 0]
        if absent:
            raise ValueError(
                f"elements {absent} have no atom in any training row; their coefficient "
                "would be a null-space choice, not a fit"
            )
        x = counts.to(device="cpu", dtype=torch.float64)
        y = energy.detach().to(device="cpu", dtype=torch.float64)
        return self._solve(x, y)

    def _solve(self, x: torch.Tensor, y: torch.Tensor) -> AtomicDress:
        """``pinv`` normal equations on the float64 ``(N, K)`` design matrix."""
        beta = torch.linalg.pinv(x.T @ x, rtol=PINV_RTOL) @ x.T @ y
        return self._table(beta, 0.0)

    def _table(self, beta: torch.Tensor, intercept: float) -> AtomicDress:
        index = torch.tensor(self.elems, dtype=torch.long)
        e0 = torch.zeros(max(self.elems) + 1, dtype=torch.float64)
        support = torch.zeros(max(self.elems) + 1, dtype=torch.bool)
        e0[index] = beta
        support[index] = True
        return AtomicDress(e0=e0, support=support, intercept=intercept)

    def fit_dataset(self, dataset: BaseDataset) -> AtomicDress:
        """Fit on every frame of *dataset* — pass the training split, nothing else.

        One pass over ``range(len(dataset))``; a record that is a sequence of
        frames (a scan) gives one row per frame. Only atomic numbers and the
        energy label are read. The label is the meta key collate turns into
        ``graphs.energy`` (``pe`` or ``energy``); a frame carrying both has no
        ``graphs.energy`` after collate, so it is refused here too.

        Args:
            dataset: The training split, e.g. a
                :class:`~molix.data.SubsetDataset` of the training indices.

        Returns:
            The fitted table (see :meth:`fit`).

        Raises:
            KeyError: A frame has no atomic-number column or no energy label.
            ValueError: A frame has two energy labels, or an atom whose
                element is not in *elems*; and everything :meth:`fit` raises.
        """
        rows: list[list[int]] = []
        labels: list[float] = []
        for index in range(len(dataset)):
            record = dataset[index]
            frames = [record] if isinstance(record, Frame) else list(record)
            for frame in frames:
                atoms = frame["atoms"]
                name = next((c for c in ATOMIC_NUMBER_COLUMNS if c in atoms), None)
                if name is None:
                    raise KeyError(f"dataset[{index}] has no atomic-number column")
                numbers = np.asarray(atoms[name]).astype(np.int64).reshape(-1)
                unknown = sorted(set(numbers.tolist()) - set(self.elems))
                if unknown:
                    raise ValueError(
                        f"dataset[{index}] has elements {unknown} outside {list(self.elems)}"
                    )
                rows.append([int(np.count_nonzero(numbers == z)) for z in self.elems])
                labels.append(self._label(frame, index))
        counts = torch.tensor(rows, dtype=torch.int64).reshape(-1, len(self.elems))
        return self.fit(counts, torch.tensor(labels, dtype=torch.float64))

    @staticmethod
    def _label(frame: Frame, index: int) -> float:
        keys = [key for key in frame.meta if GRAPH_LEAF.get(key, key) == "energy"]
        if not keys:
            raise KeyError(f"dataset[{index}] has no energy label (meta 'pe' or 'energy')")
        if len(keys) > 1:
            raise ValueError(
                f"dataset[{index}] carries two energy labels {keys}; collate drops "
                "graphs.energy for such a frame, so it cannot be fitted"
            )
        raw = frame.meta[keys[0]]
        # molrs >= 0.14 wraps meta entries in MetaValue; .value unwraps.
        return float(getattr(raw, "value", raw))


class InterceptLeastSquaresDress(LeastSquaresDress):
    """Least squares with a per-graph intercept: ``y = sum_k X[n, k] c[elems[k]] + b``.

    The design matrix gains a column of ones; ``b`` lands on
    :attr:`AtomicDress.intercept <molix.data.AtomicDress>`. For a label that is
    not extensive in the atom count (a glass-transition temperature, say) the
    intercept carries the size-independent part an elemental sum cannot. Same
    validation, the same ``pinv`` cutoff and the same dataset pass as
    :class:`LeastSquaresDress`. A dress with an intercept is a training
    residual only; it cannot be exported as atom energies.

    Args:
        elems: Atomic numbers, the column order of ``X``.
    """

    def _solve(self, x: torch.Tensor, y: torch.Tensor) -> AtomicDress:
        design = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
        beta = torch.linalg.pinv(design.T @ design, rtol=PINV_RTOL) @ design.T @ y
        return self._table(beta[:-1], float(beta[-1]))
