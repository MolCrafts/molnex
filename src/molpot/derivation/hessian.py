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
:class:`molix.data.pes_bundle.PesBundle` fixes for ``graphs.hessian`` and that
:class:`molix.core.losses.ProjectedInternalHessianLoss` consumes. Throughout,
``N`` is the number of atoms, ``B`` the number of molecules in a batch, and a
"graph" is one molecule inside a packed (concatenated) batch.

Where this sits relative to its siblings:

* :mod:`molpot.derivation.force` owns the **first** derivative
  (``F = -∂E/∂pos``) at the tensor level;
  :mod:`molpot.derivation.kernels` owns it at the batch level (position-leaf
  ownership + the ``graphs.energy`` / ``atoms.forces`` writes).
* This module never re-derives either of those. :meth:`HessianDerivation.forward`
  composes :func:`~molpot.derivation.force.autograd_forces_from_energy` and
  :meth:`HessianDerivation.graph_blocks` composes
  :func:`~molpot.derivation.kernels.grad_force_pass`; the only new
  differentiation here is the **second** ``torch.autograd.grad`` walking the
  ``3N`` rows of ``∂E/∂pos``.

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

Units: the producer is unit-agnostic, exactly like
:mod:`molpot.derivation.kernels` — it differentiates whatever the energy core
emits. With energies in kcal/mol and positions in Å the Hessian is
kcal·mol⁻¹·Å⁻² (eV/Å² for the repo-default eV convention); keeping the core on
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

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.derivation.force import autograd_forces_from_energy
from molpot.derivation.kernels import EnergyCore, grad_force_pass
from molpot.derivation.protocol import FORCES_KEY, POS_KEY

__all__ = ["HessianDerivation"]


class HessianDerivation(nn.Module):
    """Cartesian second derivative ``H = ∂²E/∂pos²`` by autograd double backward.

    Stateless (no parameters, no buffers, no session side-channel) and
    constructed with no arguments, like its siblings
    :class:`~molpot.derivation.stress.StressDerivation` and
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
            kcal/mol energy core, eV/Å² for an eV one.

        Raises:
            ValueError: If ``pos`` is not a ``(N, 3)`` tensor.
        """
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must be atomic positions (N, 3); got shape {tuple(pos.shape)}")

        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            leaf = pos.detach().requires_grad_(True)
            # create_graph=True on the first derivative regardless of the ambient
            # grad mode: the second derivative is exactly what differentiates it.
            gradient = -autograd_forces_from_energy(energy_fn(leaf), leaf, create_graph=True)
            flat = gradient.reshape(-1)
            rows = [
                self._row(flat[i], leaf, create_graph=create_graph) for i in range(flat.shape[0])
            ]
        return torch.stack(rows)

    def graph_blocks(self, energy_core: EnergyCore, batch: TensorDict) -> list[torch.Tensor]:
        """Per-graph Hessian blocks of a packed batch, in graph order.

        A packed batch stores several molecules as one long atom axis; a
        "graph" is one of those molecules, with ``Nᵢ`` atoms out of ``N`` in
        total.

        Runs **one** energy forward through
        :func:`~molpot.derivation.kernels.grad_force_pass`
        (``create_graph=True, detach_energy=False``), then for each global row
        ``i`` differentiates ``(-forces).reshape(-1)[i]`` w.r.t. the position
        leaf and keeps only the columns belonging to ``i``'s own graph, so the
        result held is ``Σ (3Nᵢ)²`` numbers rather than the ``(3N)²`` of a dense
        batch-wide Hessian. (Each row is still evaluated at full width ``3N``
        before being sliced, and the slices are views of it, so assembling one
        block transiently holds ``3Nᵢ × 3N``.) The collate contract (per-sample
        rebased ``edge_index`` / ``bond_index``, so no energy term couples two
        molecules) makes the discarded cross-graph columns exactly zero — the
        blocks are exact, not an approximation.

        Graph extents come from ``("atoms", "batch")`` alone — the single source
        of truth for membership. Nothing is written back under a ``hessian``
        key: batch-key writes and readout modes stay with
        :mod:`molpot.derivation.kernels` and a future Hessian readout. The batch
        is mutated exactly as :func:`~molpot.derivation.kernels.grad_force_pass`
        documents (position leaf plus ``graphs.energy`` / ``atoms.forces``).

        Args:
            energy_core: Energy core — a callable ``energy_core(batch)`` that
                runs the model on the batch and writes the per-molecule energies
                ``graphs.energy`` ``(B,)`` back onto it, as consumed by
                :func:`~molpot.derivation.kernels.grad_force_pass`.
            batch: Post-collate batch carrying ``atoms.pos`` ``(N, 3)`` Å and
                the graph-membership vector ``atoms.batch`` ``(N,)`` (which
                molecule each atom belongs to), which must be non-decreasing
                (atoms of a graph contiguous).

        Returns:
            One Hessian block ``(3Nᵢ, 3Nᵢ)`` per graph, in graph order, each in
            the ``x0, y0, z0, x1, …`` Cartesian layout (index ``i = 3a + c``
            within the graph) and in energy per Å²: kcal·mol⁻¹·Å⁻² for a
            kcal/mol energy core, eV/Å² for an eV one.

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

        # Resolved at entry, before grad_force_pass forces grad on for the first
        # derivative: no_grad callers get detached blocks.
        create_graph = torch.is_grad_enabled()
        batch = grad_force_pass(energy_core, batch, create_graph=True, detach_energy=False)
        leaf = batch[POS_KEY]

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
