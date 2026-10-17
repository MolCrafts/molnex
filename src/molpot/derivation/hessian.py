"""Cartesian Hessian derivation: ``H = ∂²E/∂pos²``.

The **Hessian** of a molecule is the matrix of *second* derivatives of its
potential energy ``E`` with respect to the Cartesian coordinates of its atoms —
the force-constant matrix. Near a minimum of the **potential energy surface**
(PES: the energy seen as a function of the nuclear coordinates) the energy is
quadratic in a displacement ``Δx``, ``E ≈ E₀ + ½ Δxᵀ H Δx``, so ``H`` says how
stiff every distortion is and, after mass weighting, fixes the harmonic
vibrational frequencies. Projecting it onto bond / angle coordinates with a
Wilson ``B`` matrix — the construction of Wilson (1941), cited below and
implemented by :class:`molix.core.losses.ProjectedInternalHessianLoss` — gives
exactly the force constants a **Class-I** force field is parameterised by
("Class-I": the classical harmonic-bond, harmonic-angle, Fourier-torsion,
pairwise-nonbonded functional form of OPLS / AMBER / CHARMM).

Single responsibility: the second derivative of a potential energy surface with
respect to atomic positions, in the flat Cartesian layout ``i = 3a + c`` for
atom ``a`` and component ``c ∈ {x, y, z}`` (``x0, y0, z0, x1, …``) that
:class:`molix.core.losses.ProjectedInternalHessianLoss` documents for
``graphs.hessian`` and consumes. Throughout,
``N`` is the number of atoms, ``B`` the number of molecules in a batch, and a
"graph" is one molecule inside a packed (concatenated) batch.

Where this sits relative to its siblings:

* :mod:`molpot.derivation.force` owns the **first** derivative
  (``F = -∂E/∂pos``) at the tensor level;
  :class:`molix.autograd.Derivative` owns it at the batch level.
* This module never re-derives it. Both :meth:`HessianDerivation.forward` and
  :meth:`HessianDerivation.graph_blocks` compose
  :func:`~molpot.derivation.force.autograd_forces_from_energy` on a leaf of
  their own (:func:`molix.autograd.leaf`) — the op cannot serve them, because
  the rows below need that leaf *after* the first derivative is taken and the
  op never lets its leaf out; the only new differentiation here is the
  **second** ``torch.autograd.grad`` walking the ``3N`` rows of ``∂E/∂pos``.

Scope — **autograd double-backward only**. "Autograd" is PyTorch's reverse-mode
automatic differentiation; a *double* backward differentiates a quantity that is
itself a gradient, which is what a second derivative needs. Unlike
:class:`~molpot.derivation.force.ForceDerivation` there is deliberately no
``method=`` dispatch and no functorch (``torch.func.jacrev ∘ grad``, PyTorch's
functional-transform API) backend: differentiating twice through the *fused*
cuEquivariance kernels (the GPU library behind this repo's equivariant encoders)
is owned by the active ``cuet-force-doublebackward`` spec
(``.claude/specs/cuet-force-doublebackward.md``), so this module targets
pure-PyTorch (Class-I shaped) energy graphs. Finite differences — rebuilding the
Hessian from many displaced energy evaluations — are a test oracle at most,
never a production path.

Differentiability policy mirrors ``force.py``. ``create_graph`` is PyTorch's
flag for "keep the derivative itself differentiable"; for the *second*
derivative it resolves from the ambient :func:`torch.is_grad_enabled` **at
entry**, so training keeps ``∂³E/∂pos²∂θ`` alive — with ``θ`` the model
parameters, so a Hessian loss can reach a Class-I parameter head — while
``torch.no_grad()`` yields a detached Hessian for reporting. The first
derivative always keeps its graph — it is what the second one differentiates.

Units: the producer is unit-agnostic, exactly like the derivative op — it
differentiates whatever the energy emits. With energies in kcal/mol and
positions in Å the Hessian is
kcal·mol⁻¹·Å⁻² (eV/Å² for the repo-default eV convention); keeping the energy on
one convention is the caller's job.

Reference:
    Wilson, E. B. "A Method of Obtaining the Expanded Secular Equation for the
    Vibration Frequency of a Molecule" *J. Chem. Phys.* **9**, 76 (1941).
    https://doi.org/10.1063/1.1750829

Example:
    >>> import torch
    >>> from molpot import HessianDerivation
    >>> deriv = HessianDerivation()
    >>> pos = torch.randn(5, 3)  # 5 atoms, Å
    >>> hessian = deriv(lambda p: p.pow(2).sum(), pos)  # (15, 15) = (3N, 3N)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.autograd import leaf as position_leaf
from molix.schema import ENERGY_KEY, FORCES_KEY, POS_KEY, has_energy
from molpot.derivation.force import autograd_forces_from_energy

__all__ = ["HessianDerivation"]

#: An energy: ``energy(batch) -> batch``, reading ``atoms.pos`` off the batch
#: and writing ``graphs.energy`` (optionally ``atoms.energy``) in place — a
#: potential's ``energy`` method.
Energy = Callable[[TensorDict], TensorDict]


class HessianDerivation(nn.Module):
    """Cartesian second derivative ``H = ∂²E/∂pos²`` by autograd double backward.

    Stateless (no parameters, no buffers, no session side-channel) and
    constructed with no arguments, like its sibling
    :class:`~molpot.derivation.force.ForceDerivation`. One instance is reusable
    across batches.

    Two primitive entry points: :meth:`forward` for a dense ``(3N, 3N)`` Hessian
    of a tensor-level energy function, :meth:`graph_blocks` for one
    ``(3Nᵢ, 3Nᵢ)`` block per graph of a packed batch.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Dense Cartesian Hessian of a scalar energy function.

        Evaluates ``energy_fn`` on a private ``requires_grad`` *leaf* — an input
        tensor of the autograd graph, i.e. the thing derivatives are taken with
        respect to — taken from ``pos.detach()`` inside
        :func:`torch.enable_grad`, takes ``g = ∂E/∂pos`` via
        :func:`~molpot.derivation.force.autograd_forces_from_energy` (negated
        back from forces), then differentiates each of the ``3N`` entries of
        ``g.reshape(-1)`` w.r.t. that leaf and stacks the resulting rows.

        The result is returned **as computed — not symmetrized**, so a broken
        energy graph shows up as asymmetry (``max|H - Hᵀ|``) instead of being
        silently averaged away.

        Args:
            energy_fn: Maps positions ``(N, 3)`` of the ``N`` atoms to a
                **scalar** total energy (a per-molecule energy vector ``(B,)``
                over the ``B`` molecules of a batch is summed to that scalar),
                closing over the model parameters and the rest of the batch.
                Must recompute every position-derived quantity internally so the
                gradient flows through it.
            pos: Atomic positions ``(N, 3)`` in Å. Does not need
                ``requires_grad`` — a private leaf is taken from
                ``pos.detach()``.

        Returns:
            Hessian ``(3N, 3N)`` with row/column index ``i = 3a + c`` for atom
            ``a`` and Cartesian component ``c ∈ {x, y, z}``
            (``x0, y0, z0, x1, …``), in energy per Å²: kcal·mol⁻¹·Å⁻² for a
            kcal/mol energy, eV/Å² for an eV one.

        Raises:
            ValueError: If ``pos`` is not a ``(N, 3)`` tensor.
        """
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must be atomic positions (N, 3); got shape {tuple(pos.shape)}")

        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            leaf = position_leaf(pos)
            # create_graph=True on the first derivative regardless of the ambient
            # grad mode: the second derivative is exactly what differentiates it.
            gradient = -autograd_forces_from_energy(energy_fn(leaf), leaf, create_graph=True)
            flat = gradient.reshape(-1)
            rows = [
                self._row(flat[i], leaf, create_graph=create_graph) for i in range(flat.shape[0])
            ]
        return torch.stack(rows)

    def graph_blocks(self, energy: Energy, batch: TensorDict) -> list[torch.Tensor]:
        """Per-graph Hessian blocks of a packed batch, in graph order.

        A packed batch stores several molecules as one long atom axis; a
        "graph" is one of those molecules, with ``Nᵢ`` atoms out of ``N`` in
        total.

        Installs its own position leaf on the batch and runs **one** energy
        forward on it, takes ``∂E/∂pos`` with ``create_graph=True``
        (:func:`~molpot.derivation.force.autograd_forces_from_energy`), then
        for each global row ``i`` differentiates ``(-forces).reshape(-1)[i]``
        w.r.t. the position leaf and keeps only the columns belonging to
        ``i``'s own graph, so the result held is ``Σ (3Nᵢ)²`` numbers rather
        than the ``(3N)²`` of a dense batch-wide Hessian. (Each row is still
        evaluated at full width ``3N`` before being sliced, and the slices are
        views of it, so assembling one block transiently holds ``3Nᵢ × 3N``.)
        The collate contract (per-sample rebased ``pair_index`` /
        ``bond_index``, so no energy term couples two molecules) makes the
        discarded cross-graph columns exactly zero — the blocks are exact, not
        an approximation.

        Graph extents come from ``("atoms", "batch")`` alone — the single source
        of truth for membership. Nothing is written back under a ``hessian``
        key: batch-key writes stay with the derivative op and a future Hessian
        readout. The batch comes back with ``atoms.pos`` replaced by this
        method's ``requires_grad`` leaf (the energy and forces hang off it, and
        the rows below differentiate against exactly this tensor) plus
        ``graphs.energy`` / ``atoms.forces``, both attached.

        Args:
            energy: ``energy(batch) -> batch`` — a potential's ``energy``
                method — writing the per-molecule energies ``graphs.energy``
                ``(B,)`` back onto the batch.
            batch: Post-collate batch carrying ``atoms.pos`` ``(N, 3)`` Å and
                the graph-membership vector ``atoms.batch`` ``(N,)`` (which
                molecule each atom belongs to), which must be non-decreasing
                (atoms of a graph contiguous).

        Returns:
            One Hessian block ``(3Nᵢ, 3Nᵢ)`` per graph, in graph order, each in
            the ``x0, y0, z0, x1, …`` Cartesian layout (index ``i = 3a + c``
            within the graph) and in energy per Å²: kcal·mol⁻¹·Å⁻² for a
            kcal/mol energy, eV/Å² for an eV one.

        Raises:
            ValueError: If ``atoms.batch`` is not non-decreasing — atoms of one
                graph would then be spread across disjoint row ranges and the
                blocks would mix graph identities.
        """
        membership = batch["atoms", "batch"]
        if membership.numel() > 1 and bool((membership[1:] < membership[:-1]).any()):
            raise ValueError(
                "atoms.batch must be non-decreasing so each graph owns a contiguous "
                f"atom range; got {membership.tolist()}. Sort the batch by graph "
                "before taking per-graph Hessian blocks."
            )

        # Resolved at entry, before grad is forced on for the first derivative:
        # no_grad callers get detached blocks.
        create_graph = torch.is_grad_enabled()
        # This method owns the leaf — the derivative op would never let its
        # own out, and the rows below differentiate w.r.t. exactly this tensor.
        leaf = position_leaf(cast(torch.Tensor, batch[POS_KEY]))
        batch[POS_KEY] = leaf
        with torch.enable_grad():
            # ``energy`` writes in place (molix.autograd.Potential.energy): the
            # columns are read back off the same batch the leaf sits in.
            energy(batch)
            if not has_energy(batch):
                raise RuntimeError("energy must write batch['graphs', 'energy']")
            batch[FORCES_KEY] = autograd_forces_from_energy(
                batch[ENERGY_KEY], leaf, create_graph=True
            )

        counts = torch.unique_consecutive(membership, return_counts=True)[1].tolist()
        blocks: list[torch.Tensor] = []
        start = 0
        with torch.enable_grad():
            # Negate inside the scope: under an ambient no_grad the negation
            # would otherwise strip the first derivative's graph.
            flat = (-batch[FORCES_KEY]).reshape(-1)
            for count in counts:
                stop = start + count
                columns = slice(3 * start, 3 * stop)
                blocks.append(
                    torch.stack(
                        [
                            self._row(flat[i], leaf, create_graph=create_graph)[columns]
                            for i in range(3 * start, 3 * stop)
                        ]
                    )
                )
                start = stop
        return blocks

    def _row(
        self,
        entry: torch.Tensor,
        leaf: torch.Tensor,
        *,
        create_graph: bool,
    ) -> torch.Tensor:
        """One Hessian row: ``∂(∂E/∂pos)_i/∂pos`` flattened to ``(3N,)``.

        Args:
            entry: Scalar entry ``i`` of the flattened first derivative,
                still connected to ``leaf``.
            leaf: The ``requires_grad`` position leaf ``(N, 3)`` the energy was
                evaluated on.
            create_graph: Keep the row connected to the model parameters
                (``∂³E/∂pos²∂θ``) for a Hessian-supervised ``loss.backward()``.

        Returns:
            Hessian row ``(3N,)`` in the ``x0, y0, z0, x1, …`` layout.
        """
        (row,) = torch.autograd.grad(entry, leaf, retain_graph=True, create_graph=create_graph)
        return row.reshape(-1)
