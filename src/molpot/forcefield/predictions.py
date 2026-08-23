"""ForceFieldPredictions — Class-I model path → the per-task prediction mapping.

A **force field** is the set of parameters of a fixed algebraic expression for the
potential energy of a molecule as a function of its atomic positions. **Class-I** is
the classical *molecular-mechanics* (MM) shape of that expression, the one used by
OPLS / AMBER / CHARMM: harmonic springs on bond lengths and bend angles, cosine
(Fourier) terms on torsion angles, plus pairwise Lennard-Jones and Coulomb terms
between non-bonded atoms. Here the parameters are not looked up in a discrete
atom-type table — they are *predicted* from the covalent graph (which atom is bonded
to which) by a neural model, so the whole force field is differentiable.

:class:`~molpot.forcefield.Parameterizer` produces a filled
:class:`~molpot.forcefield.ForceField`; the Class-I task losses consume a plain
mapping keyed ``energy`` / ``forces`` / ``hessian`` / ``q`` (``q`` = per-atom partial
charges). This module is that one hop and nothing else — it evaluates no new
functional form, and it owns no derivative of its own: forces come from
:meth:`~molpot.forcefield.Potentials.calc_energy_forces` (which delegates to
:func:`~molpot.derivation.force.autograd_forces_from_energy`), the Cartesian
**Hessian** — the matrix of second derivatives of the energy with respect to the
atomic coordinates, i.e. the force constants of every small distortion — from an
**injected** :class:`~molpot.derivation.HessianDerivation`, and ``q`` is *read* from
the ForceField the model already filled.

The declared keys are fixed at construction and drive a call plan built there,
so a prediction the task did not declare is never computed and never appears in
the returned dict.

Units follow the Class-I parameter tables, which use the LAMMPS ``units="real"``
convention (the :class:`~molpot.forcefield.ForceField` default): kcal/mol for energy,
kcal mol⁻¹ Å⁻¹ for forces, kcal mol⁻¹ Å⁻² for the Hessian, elementary charge ``e``
for ``q``, Å for positions.

Reference:
    Cartesian Hessian and its projection onto bond / angle coordinates —
    Wilson, E. B. "A Method of Obtaining the Expanded Secular Equation for the
    Vibration Frequency of a Molecule" *J. Chem. Phys.* **9**, 76 (1941).
    https://doi.org/10.1063/1.1750829
    Electronegativity equalization, the charge model whose output ``q`` is read here —
    Mortier, W. J.; Ghosh, S. K.; Shankar, S. "Electronegativity-equalization
    method for the calculation of atomic charges in molecules"
    *J. Am. Chem. Soc.* **108**, 4315 (1986).
    https://doi.org/10.1021/ja00275a013
    Charge equilibration, the same family extended to molecular dynamics —
    Rappé, A. K.; Goddard, W. A. "Charge equilibration for molecular dynamics
    simulations" *J. Phys. Chem.* **95**, 3358 (1991).
    https://doi.org/10.1021/j100161a070
    The multi-task fitting protocol (energies, forces, charges and torsion profiles on
    one parameterization) these predictions serve —
    Wang, Y.; Fass, J.; Chodera, J. D. "End-to-end differentiable construction
    of molecular mechanics force fields" *Chem. Sci.* **13**, 12016 (2022).
    https://doi.org/10.1039/D2SC02739A
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import torch
from tensordict import TensorDict

from molpot.forcefield.container import ForceField
from molpot.forcefield.parameterizer import Parameterizer
from molpot.forcefield.potentials import Potentials

__all__ = ["ForceFieldPredictions"]


class _HessianDerivation(Protocol):
    """Call shape of the injected Hessian derivation.

    Satisfied by :class:`~molpot.derivation.HessianDerivation` and by anything else
    mapping ``(energy_fn, pos)`` to a dense Cartesian Hessian.
    """

    def __call__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        pos: torch.Tensor,
        /,
    ) -> torch.Tensor: ...


_EnergyStep = Callable[[TensorDict, Potentials, dict[str, torch.Tensor]], None]
_ChargeStep = Callable[[ForceField, dict[str, torch.Tensor]], None]


class ForceFieldPredictions:
    """Map one :class:`~molpot.forcefield.Parameterizer` plus a batch onto
    the predictions one task declares.

    ``keys`` is the task's prediction contract and is fixed at construction:
    ``__init__`` turns it into a call plan, and :meth:`__call__` walks that plan
    with exactly **one** ``model(batch)`` forward and at most one
    ``ForceField.to_potentials()`` assembly (zero for a ``("q",)``-only task,
    which needs no energy at all). There is no ``keys="all"``, no default key
    set, and no method that computes everything.

    Why these forces are the deployed ones. Write ``x`` for the ``3N`` Cartesian
    coordinates of the ``N`` atoms (Å), ``E`` for the Class-I potential energy
    (kcal/mol), ``G`` for the covalent graph (which atoms are bonded to which) and
    ``θ`` for the Class-I parameters the heads predict (force constants, equilibrium
    lengths, charges, …). The encoder this path is built with reads ``G`` alone and
    never a coordinate (:class:`~molrep.chem.ChemEncoder`), so ``θ = θ(G)``,
    ``dθ/dx = 0``, and the chain rule ``dE/dx = ∂E/∂x + (∂E/∂θ)(dθ/dx)`` collapses to
    ``F = −∂E(x; θ(G))/∂x`` — exactly the force the exported force field produces.
    An encoder that did read positions would add the ``(∂E/∂θ)(dθ/dx)`` term this
    path does not carry. The same argument covers the charges only for a *diagonal*
    hardness kernel (:class:`~molpot.charge.DiagonalHardnessKernel`, hardness matrix
    ``A = diag(η)``), whose charge solve reads no positions either, so ``dq/dx = 0``
    and no charge-response term is missing from ``F`` — geometry-dependent charge
    kernels are out of scope.

    Args:
        model: :class:`~molpot.forcefield.Parameterizer`; ``model(batch)``
            returns the filled :class:`~molpot.forcefield.ForceField`.
        keys: Predictions this task declares, a subset of
            :attr:`SUPPORTED_KEYS`, each declared once.
        hessian_derivation: Cartesian Hessian derivation
            (:class:`~molpot.derivation.HessianDerivation`), called as
            ``derivation(energy_fn, pos)``. Required iff ``"hessian"`` is
            declared.

    Attributes:
        SUPPORTED_KEYS: The four predictions a task may declare —
            ``("energy", "forces", "hessian", "q")``.
        model: The :class:`~molpot.forcefield.Parameterizer` passed at
            construction.
        keys: The declared keys in declaration order — exactly the keys
            :meth:`__call__` returns.
        hessian_derivation: The injected Hessian derivation, or ``None`` when
            ``"hessian"`` was not declared.

    Raises:
        ValueError: ``keys`` is empty, repeats a key, or names a key outside
            :attr:`SUPPORTED_KEYS`; ``"q"`` is declared while the model has no
            ``charge_model``; ``"hessian"`` is declared without a derivation.

    Example:
        >>> predictions = ForceFieldPredictions(model, keys=("energy", "forces"))  # doctest: +SKIP
        >>> out = predictions(batch)  # doctest: +SKIP
        >>> sorted(out)  # doctest: +SKIP
        ['energy', 'forces']
    """

    SUPPORTED_KEYS: tuple[str, ...] = ("energy", "forces", "hessian", "q")

    def __init__(
        self,
        model: Parameterizer,
        *,
        keys: Sequence[str],
        hessian_derivation: _HessianDerivation | None = None,
    ) -> None:
        declared = tuple(keys)
        if not declared:
            raise ValueError(
                f"keys must declare at least one prediction out of {self.SUPPORTED_KEYS}; got ()"
            )
        duplicates = sorted({key for key in declared if declared.count(key) > 1})
        if duplicates:
            raise ValueError(f"keys has duplicate entries {duplicates}; declare each key once")
        unsupported = [key for key in declared if key not in self.SUPPORTED_KEYS]
        if unsupported:
            raise ValueError(
                f"unsupported prediction keys {unsupported}; "
                f"supported keys are {self.SUPPORTED_KEYS}"
            )
        if "q" in declared and model.charge_model is None:
            raise ValueError(
                'keys declares "q" but model.charge_model is None; '
                "build Parameterizer(..., charge_model=AtomicChargeModel(...))"
            )
        if "hessian" in declared and hessian_derivation is None:
            raise ValueError(
                'keys declares "hessian" but no Hessian derivation was injected; '
                "pass hessian_derivation=HessianDerivation()"
            )

        self.model = model
        self.keys = declared
        self.hessian_derivation = hessian_derivation

        energy_plan: list[_EnergyStep] = []
        if "energy" in declared and "forces" in declared:
            energy_plan.append(self._put_energy_forces)
        elif "energy" in declared:
            energy_plan.append(self._put_energy)
        elif "forces" in declared:
            energy_plan.append(self._put_forces)
        if "hessian" in declared:
            energy_plan.append(self._put_hessian)
        self._energy_plan: tuple[_EnergyStep, ...] = tuple(energy_plan)
        self._charge_plan: tuple[_ChargeStep, ...] = (self._put_charges,) if "q" in declared else ()
        self._evaluate = self._walk_energy_plan if energy_plan else self._skip_energy_plan

    def __call__(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Run the declared predictions for one batch.

        One ``model(batch)`` forward fills the ForceField; the plan built at
        construction then reads exactly the declared predictions off it.

        Args:
            batch: Post-collate batch with ``atoms.pos`` ``(N, 3)`` Å and the
                Class-I topology namespaces (``bonds`` / ``angles`` / …).

        Returns:
            Exactly the declared keys — ``energy`` ``(B,)`` (``()`` for a single
            graph) kcal/mol, ``forces`` ``(N, 3)`` kcal mol⁻¹ Å⁻¹, ``hessian``
            ``(3N, 3N)`` kcal mol⁻¹ Å⁻², ``q`` ``(N,)`` ``e``.

        Raises:
            RuntimeError: ``"q"`` was declared but the filled ForceField carries
                no ``atoms["q"]``.
        """
        forcefield: ForceField = self.model(batch)
        predictions: dict[str, torch.Tensor] = {}
        self._evaluate(batch, forcefield, predictions)
        for step in self._charge_plan:
            step(forcefield, predictions)
        return predictions

    def energy(self, batch: TensorDict) -> torch.Tensor:
        """Class-I potential energy of ``batch``.

        Only the terms the ForceField declares are assembled, so a bonded-only form
        evaluates no Lennard-Jones (LJ) and no Coulomb term. LJ is the pair term for
        atoms that are *not* bonded — steep repulsion at short range, weak attraction
        further out — parameterized per atom by a well depth ``ε`` (kcal/mol) and the
        distance ``σ`` (Å) at which the pair energy crosses zero. Even when an atom
        head has already written ``ε``/``σ`` into the parameter tables, no LJ term is
        evaluated unless a ``pair:lj/cut`` style is declared.

        Args:
            batch: Post-collate batch with ``atoms.pos`` ``(N, 3)`` Å and the
                Class-I topology namespaces (``bonds`` / ``angles`` / …).

        Returns:
            Energy ``(B,)`` over the ``B`` molecules of the batch, in kcal/mol —
            shape ``()`` for a single graph (one molecule).
        """
        forcefield: ForceField = self.model(batch)
        return forcefield.to_potentials().calc_energy(batch)

    def forces(self, batch: TensorDict) -> torch.Tensor:
        """Atomic forces ``F = −∂E/∂x`` at fixed graph-derived parameters.

        One energy evaluation happens inside; only the forces are returned (use
        :meth:`energy_forces` to keep both). Symbols ``E``, ``x``, ``θ`` are the ones
        defined in the class docstring.

        Args:
            batch: Post-collate batch with ``atoms.pos`` ``(N, 3)`` Å and the
                Class-I topology namespaces.

        Returns:
            Forces ``(N, 3)`` in kcal mol⁻¹ Å⁻¹, still connected to the parameter
            heads (the mixed second derivative ``∂²E/∂x∂θ`` is kept) whenever grad is
            enabled, so a force loss back-propagates into those heads.
        """
        return self.energy_forces(batch)[1]

    def energy_forces(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy and forces from one energy evaluation.

        Args:
            batch: Post-collate batch with ``atoms.pos`` ``(N, 3)`` Å and the
                Class-I topology namespaces.

        Returns:
            Energy ``(B,)`` — ``()`` for a single graph — in kcal/mol and forces
            ``(N, 3)`` in kcal mol⁻¹ Å⁻¹.
        """
        forcefield: ForceField = self.model(batch)
        return forcefield.to_potentials().calc_energy_forces(batch)

    def hessian(self, batch: TensorDict) -> torch.Tensor:
        """Dense Cartesian Hessian ``H_ij = ∂²E/∂x_i∂x_j`` via the injected derivation.

        ``x_i`` is one of the ``3N`` Cartesian coordinates, so ``H`` collects the
        force constants of every small distortion of the molecule. It is returned as
        the derivation computes it — **not** symmetrized — so a broken energy graph
        shows up as asymmetry instead of being averaged away.

        A loss defined in *internal* coordinates ``q_int`` (bond lengths in Å, bend
        angles in rad) first contracts ``H`` with the Wilson ``B`` matrix,
        ``B = ∂q_int/∂x`` of shape ``(n_internal, 3N)``, built by
        :class:`~molpot.forcefield.WilsonBMatrix` and consumed by
        :class:`molix.core.losses.ProjectedInternalHessianLoss`. (Wilson writes those
        internal coordinates ``q``; they are *not* the partial charges this module
        calls ``q``.)

        Args:
            batch: Post-collate batch with ``atoms.pos`` ``(N, 3)`` Å and the
                Class-I topology namespaces.

        Returns:
            Hessian ``(3N, 3N)`` in kcal mol⁻¹ Å⁻², rows/columns ordered
            ``x0, y0, z0, x1, …`` (index ``3a + c`` for atom ``a`` and component
            ``c ∈ {x, y, z}``).

        Raises:
            ValueError: No Hessian derivation was injected at construction.

        Reference:
            Wilson, E. B. *J. Chem. Phys.* **9**, 76 (1941).
            https://doi.org/10.1063/1.1750829
        """
        forcefield: ForceField = self.model(batch)
        return self._hessian(batch, forcefield.to_potentials())

    def charges(self, batch: TensorDict) -> torch.Tensor:
        """Partial charges the model's charge equilibration wrote on the ForceField.

        *Charge equilibration* replaces fixed per-element charges by a small
        optimization: every atom gets an electronegativity ``χ`` (kcal mol⁻¹ e⁻¹, how
        strongly it pulls electrons) and a hardness ``η`` (kcal mol⁻¹ e⁻², how much it
        resists being charged) from the graph, and the charges ``q`` (``e``) minimize
        the electrostatic model energy ``Σ_i (χ_i q_i + ½ η_i q_i²)`` (kcal/mol)
        subject to a fixed total charge ``Q`` (``e``). The minimum of that constrained
        quadratic is a linear system — the Karush–Kuhn–Tucker (KKT) conditions, the
        standard name for the stationarity plus constraint equations of a constrained
        optimum::

            [ A   1 ] [ q ]   [ -χ ]
            [ 1ᵀ  0 ] [ λ ] = [  Q ]

        with ``A`` the hardness matrix and ``λ`` the Lagrange multiplier that enforces
        ``Σ q = Q``. This is the electronegativity-equalization method (EEM) /
        charge-equilibration (QEq) family, solved by
        :class:`~molpot.charge.ElectronegativityEqualization` once inside the model
        forward; this method *reads* that result rather than re-solving it.

        With a diagonal hardness kernel (``A = diag(η)``) the solve reads no positions,
        so the charges depend on the covalent graph only and are bit-identical across
        conformers — different geometries — of one topology.

        Args:
            batch: Post-collate batch carrying the topology namespaces the encoder
                and the charge solve read (``atoms.Z``, ``bonds``, …).

        Returns:
            Partial charges ``(N,)`` in elementary charge ``e``.

        Raises:
            RuntimeError: The filled ForceField carries no ``atoms["q"]``.
        """
        forcefield: ForceField = self.model(batch)
        return self._charges(forcefield)

    # --- private: the call plan ---

    def _walk_energy_plan(
        self,
        batch: TensorDict,
        forcefield: ForceField,
        predictions: dict[str, torch.Tensor],
    ) -> None:
        """Assemble the declared Class-I terms once, then run the energy steps."""
        potentials = forcefield.to_potentials()
        for step in self._energy_plan:
            step(batch, potentials, predictions)

    def _skip_energy_plan(
        self,
        batch: TensorDict,
        forcefield: ForceField,
        predictions: dict[str, torch.Tensor],
    ) -> None:
        """No declared prediction needs an energy — assemble no Potentials."""
        del batch, forcefield, predictions

    def _put_energy(
        self,
        batch: TensorDict,
        potentials: Potentials,
        predictions: dict[str, torch.Tensor],
    ) -> None:
        predictions["energy"] = potentials.calc_energy(batch)

    def _put_forces(
        self,
        batch: TensorDict,
        potentials: Potentials,
        predictions: dict[str, torch.Tensor],
    ) -> None:
        predictions["forces"] = potentials.calc_energy_forces(batch)[1]

    def _put_energy_forces(
        self,
        batch: TensorDict,
        potentials: Potentials,
        predictions: dict[str, torch.Tensor],
    ) -> None:
        energy, forces = potentials.calc_energy_forces(batch)
        predictions["energy"] = energy
        predictions["forces"] = forces

    def _put_hessian(
        self,
        batch: TensorDict,
        potentials: Potentials,
        predictions: dict[str, torch.Tensor],
    ) -> None:
        predictions["hessian"] = self._hessian(batch, potentials)

    def _put_charges(
        self,
        forcefield: ForceField,
        predictions: dict[str, torch.Tensor],
    ) -> None:
        predictions["q"] = self._charges(forcefield)

    def _hessian(self, batch: TensorDict, potentials: Potentials) -> torch.Tensor:
        """Hand the injected derivation the same ``(energy_fn, pos)`` shape ForceDerivation uses."""
        derivation = self.hessian_derivation
        if derivation is None:
            raise ValueError(
                "hessian requires a Hessian derivation injected at construction; "
                "pass hessian_derivation=HessianDerivation()"
            )
        return derivation(lambda p: potentials.calc_energy(batch, pos=p), batch["atoms", "pos"])

    def _charges(self, forcefield: ForceField) -> torch.Tensor:
        """Read ``atoms["q"]`` off the filled ForceField — never a silent zero vector."""
        atoms = forcefield.params("atoms")
        q = None if atoms is None else atoms.get("q")
        if q is None:
            raise RuntimeError(
                'the filled ForceField carries no atoms["q"]: the charge model wrote no '
                "charges for this batch, so no q prediction can be made"
            )
        return q

    def __repr__(self) -> str:
        return f"ForceFieldPredictions(model={type(self.model).__name__}, keys={self.keys})"
