"""Rondo's energy and forces: the host-side bind and the traced derivative.

Spec: ``src/molzoo/specs/rondo.md`` §3.7.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.autograd import Derivative
from molix.autograd.variables import get_column
from molix.schema import ENERGY_KEY, FORCES_KEY, VIRIAL_KEY
from molix.units import units
from molpot.heads.energy import AtomicReferenceEnergy

__all__ = ["RondoPotential"]


class RondoPotential(nn.Module):
    """Rondo's energy and forces as one prediction mapping.

    **Two stages, and only the second one is traced.** A Class-I step is a
    host-side half — the encoder and heads predict this batch's parameter
    tables, a :class:`~molpot.forcefield.ForceField` is filled with them, a
    :class:`~molpot.forcefield.Potentials` is assembled from it and bound to
    this batch's topology — followed by one tensor half: the
    :class:`molix.autograd.Derivative` over that container's energy.

    Running both halves inside one compiled region costs 22 graph breaks on a
    real batch (measured 2026-09-11): ``ForceField.__new__`` is an untraceable
    method call, the pair calculators branch on Python options while they are
    being constructed, and the eager preconditions the bind owns — the
    net-charge report, the edge-orientation refusal, the unwrapped-bond gate,
    the exclusion table — are all host syncs or data-dependent shapes. None of
    them is physics. Split out, the same op on the same batch is **1 graph /
    0 breaks** in inference (forces, and forces plus virial) and one break in
    training, at the autograd leaf (``molix/autograd/variables.py``,
    pytorch/pytorch#196653).

    Rebuilding the container per batch is not itself a cost to the compiler:
    dynamo caches by code object, so the second and every later container
    compiles **zero** new graphs (measured ``[8, 0, 0]``), and the predicted
    tables arrive as ordinary tensor inputs.

    Args:
        parameterizer: The :class:`~molzoo.rondo.parameterizer.Rondo` model;
            calling it returns the filled force field.
        dressing: Optional frozen per-element reference energies added to
            ``E_MM`` after the unit conversion. Fitting the ~O(10) kcal/mol
            residual instead of the ~1e6 kcal/mol dressed total is what keeps
            the loss well conditioned in float32; the table itself belongs to
            whoever fitted it (a dataset), not to this model.
        energy_unit: Pint unit of the traced energy (and therefore of autograd
            forces). Container energy is kcal mol⁻¹; the scale is a Python
            float from :data:`molix.units.units`, not a buffer.
        wrt: What the op differentiates the energy against.
            ``("atoms.pos",)`` — forces — is the training path.
            ``("atoms.pos", "graphs.strain")`` adds the virial and is an
            inference path only: one strain leaf is one lattice, so it needs a
            single-graph batch carrying ``graphs.cell`` and a placed list with
            ``pairs.shifts``, which a packed training batch is not.
        trainable: Namespaces of the predicted parameter tables that keep a
            gradient path; every other table is detached before the energy is
            built. ``None`` (default) trains everything.

            This is how one objective is kept from driving parameters it has
            no business constraining. A torsion profile constrains torsion
            barriers; letting it also move Lennard-Jones sigma lets the two
            compensate each other, which is the classical degeneracy of
            molecular-mechanics fitting and the reason a single combined
            objective under-determines the parameters (spec § 6.1, risk R2).
            ByteFF's ``set_trainable_terms`` is the same device.

            Detaching is not the same as excluding the parameters from the
            optimizer: the encoder is shared, so detaching a table removes
            *that path's* contribution to the encoder gradient while leaving
            the others, which a per-parameter-group optimizer cannot express.
        compiler: Applied to the bound container before it is called, i.e. to
            the traced half alone. ``None`` runs it eagerly.

    Raises:
        ValueError: ``dressing`` is a kcal mol⁻¹ table but ``energy_unit`` is
            something else.
    """

    def __init__(
        self,
        parameterizer: nn.Module,
        dressing: AtomicReferenceEnergy | None = None,
        *,
        energy_unit: Any = units.kilocalorie_per_mole,
        wrt: Sequence[str] = ("atoms.pos",),
        trainable: Sequence[str] | None = None,
        compiler: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__()
        self.trainable = None if trainable is None else tuple(trainable)
        self.parameterizer = parameterizer
        self.dressing = dressing
        self.energy_unit = energy_unit
        self._kcal_to_unit = float((1.0 * units.kilocalorie_per_mole).to(energy_unit).magnitude)
        if dressing is not None and self._kcal_to_unit != 1.0:
            raise ValueError(
                "RondoPotential dressing is a kcal mol^-1 table and cannot be "
                f"added to energy_unit={energy_unit!r} (scale {self._kcal_to_unit})"
            )
        self.wrt = tuple(wrt)
        self._compiler = compiler
        # Both built once, on the first batch: the container because assembling
        # it is untraceable molpy work, the op because a compiler applied per
        # batch would wrap a new callable every step.
        self._physics: Any = None
        self._op: Any = None

    def _hold(self, name: str, value: Any) -> None:
        """Keep an ``nn.Module`` off the parameter tree.

        ``_physics`` is the seed topology and ``_op`` is the derivative
        wrapper. Registering either puts first-batch buffers into
        ``state_dict``, so ``Trainer.smoke`` cannot restore a pre-bind
        snapshot (profile 2026-09-13).

        Args:
            name: Attribute name to set.
            value: The object to hold outside ``_modules``.
        """
        object.__setattr__(self, name, value)
        object.__getattribute__(self, "_modules").pop(name, None)

    def physics(self, batch: TensorDict) -> Any:
        """The assembled Class-I container for this field, bound to this topology.

        Built **once** and reused: assembling it is molpy work
        (``ForceField.__new__``, the pair-style option branches) that cannot be
        traced, and rebuilding it per batch was worth 20 graph breaks. It is
        assembled from one filled field so every declared style gets its term;
        the tables that field carries are dead from then on, because
        :meth:`energy` passes this batch's own on every call.

        The topology bind *is* per batch — a new batch is a new topology — and
        is host-side by nature (the exclusion table, the edge-orientation
        refusal, the net-charge report). It runs here, outside anything traced.
        The default steps call this from their eager prologue so a compiled
        ``forward`` never sees it.

        Args:
            batch: Post-collate batch carrying this step's topology.

        Returns:
            The bound :class:`~molpot.forcefield.Potentials` container.
        """
        if self._physics is None:
            with torch.no_grad():
                # Seed on a **copy**. The encoder writes its per-namespace
                # ``chem_features`` into whatever batch it is handed, so
                # seeding on the caller's batch gives the first batch one more
                # ``atoms`` column than every later one — and a column count is
                # a dynamo guard, so the first step compiles at 6 keys and the
                # second raises ``RecompileError: len(...['atoms']) != 6``
                # (GH200 job 2461680). The copy is built once per run.
                seed = self.parameterizer(batch.clone())
                # Warmup must stay in no_grad: otherwise the seed term
                # buffers keep a ``grad_fn`` from this eager call and
                # ``autograd.grad`` inside the compiled region is
                # "external grad_fn" on ``graphs.energy``.
                self._hold("_physics", seed.to_potentials(wrt=self.wrt).warmup(batch))
        else:
            self._physics.bind_topology(batch)
        if self._op is None:
            op = Derivative(self.energy, wrt=self.wrt)
            self._hold("_op", op if self._compiler is None else self._compiler(op))
        return self._physics

    @property
    def derivative(self) -> Any:
        """The :class:`~molix.autograd.Derivative` the steps mark leaves on.

        Unwraps a ``torch.compile`` wrapper so ``RondoPotential(compiler=...)``
        still exposes the op the prologue needs. ``None`` until
        :meth:`physics` has built it.
        """
        op = getattr(self._op, "_orig_mod", self._op)
        return op if isinstance(op, Derivative) else None

    def energy(self, frame: TensorDict) -> TensorDict:
        """``graphs.energy`` from this batch's own parameters — the traced seam.

        The encoder and the heads run **here**, inside whatever region the
        caller compiles, and that placement is the point. Predicting the tables
        outside and handing them in does not work: a table with a ``grad_fn``
        from outside makes ``torch.autograd.grad`` untraceable
        ("autograd.grad with external grad_fn"), and putting it on the frame
        rather than on a module changes nothing (measured 2026-09-11).
        Computed here, the tables are ordinary intermediates. When
        :attr:`trainable` names a subset of namespaces, the others are
        detached right here — after prediction, before the energy — so the
        parameters are still *used* at their predicted values and only their
        gradient path is cut.

        Args:
            frame: The batch to price.

        Returns:
            The same frame with ``graphs.energy`` written.
        """
        assert self._physics is not None, "call physics(batch) first"
        tables = self.parameterizer.tables(frame)
        if self.trainable is not None:
            tables = {
                namespace: (
                    params
                    if namespace in self.trainable
                    else {name: value.detach() for name, value in params.items()}
                )
                for namespace, params in tables.items()
            }
        out = self._physics.energy(frame, tables)
        out[ENERGY_KEY] = out[ENERGY_KEY] * self._kcal_to_unit
        return out

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Predict energy, forces, and optionally the virial.

        Args:
            batch: Post-collate batch. Its ``graphs.energy`` and
                ``atoms.forces`` columns are preserved (see below).

        Returns:
            ``energy`` ``(B,)``, ``forces`` ``(N, 3)``, ``energy_mm`` ``(B,)``,
            plus ``virial`` and ``e0`` when those are configured.
        """
        # Host-side bind stays out of the compiled region. The default steps
        # already called :meth:`physics` in their prologue; eager callers
        # (probes, benches) still go through here.
        if torch.compiler.is_compiling():
            if self._op is None:
                raise RuntimeError(
                    "RondoPotential.physics(batch) must run before the compiled forward"
                )
        else:
            self.physics(batch)
        # **The op writes in place, onto the label columns.** ``graphs.energy``
        # and ``atoms.forces`` are where a Derivative puts its results *and*
        # where a labelled batch carries its DFT targets. Handing it the batch
        # therefore replaces the labels with the predictions, and a loss that
        # reads the target after the forward compares the prediction with
        # itself: the force term is then identically zero and the energy
        # gradient is identically zero too, because the target moves with the
        # prediction. That is not a small error — it is no training signal at
        # all, and it is silent. So the caller's label columns are kept here
        # and put back, the same way the op itself restores the geometry
        # buffers it perturbs. Direct key reads, not a keys() walk: dynamo
        # specialises the two constants and does not graph-break on a
        # membership test over a NestedKey set.
        energy_label = batch[ENERGY_KEY]
        forces_label = batch[FORCES_KEY]

        # The op resolves ``create_graph`` from the ambient grad mode and
        # detaches its outputs when grad is off, so inference needs no
        # enable_grad/detach dance around it.
        out = self._op(batch)
        energy, forces = out[ENERGY_KEY], out[FORCES_KEY]
        virial = out[VIRIAL_KEY] if "graphs.strain" in self.wrt else None
        batch[ENERGY_KEY] = energy_label
        batch[FORCES_KEY] = forces_label
        energy_mm = energy.reshape(-1)
        e0 = None
        z = get_column(batch, ("atoms", "Z"))
        if self.dressing is not None and z is not None:
            atom_batch = get_column(batch, ("atoms", "batch"))
            # Sum the frozen dressing in float64. Per-graph sums are ~1e5
            # kcal/mol; float32 ``index_add`` is order-dependent on GPU, so
            # packed vs single differ by ~1 ULP (~0.03) and that noise leaks
            # into the residualized energy target, breaking gradient parity.
            e0_per_atom = self.dressing.atomic_energies[z.to(torch.long)].double()
            mask = get_column(batch, ("atoms", "mask"))
            if mask is not None:
                e0_per_atom = e0_per_atom * mask.to(dtype=e0_per_atom.dtype)
            if atom_batch is None:
                e0 = e0_per_atom.sum()
            else:
                n_graphs = batch["graphs", "num_atoms"].shape[0]
                e0 = torch.zeros(n_graphs, dtype=torch.float64, device=e0_per_atom.device)
                e0.index_add_(0, atom_batch.to(torch.long), e0_per_atom)
            energy = energy_mm + e0.float()
        else:
            energy = energy_mm
        predictions: dict[str, torch.Tensor] = {
            "energy": energy,
            "forces": forces,
            "energy_mm": energy_mm,
        }
        if virial is not None:
            predictions["virial"] = cast(torch.Tensor, virial)
        if e0 is not None:
            predictions["e0"] = e0
        return predictions

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Trainable parameters grouped by the head that owns them.

        Returns:
            Child name → its trainable parameters, skipping children with none.
        """
        groups = {
            name: tuple(p for p in child.parameters() if p.requires_grad)
            for name, child in self.parameterizer.named_children()
        }
        return {name: params for name, params in groups.items() if params}


class RondoPredictions(nn.Module):
    """Rondo answering a declared prediction contract, without the energy wrapper.

    :class:`RondoPotential` is Rondo as a potential: one energy with
    :class:`molix.autograd.Derivative` built over it, which is what stages 2 and
    3 fit. This is Rondo answering
    :class:`~molpot.forcefield.predictions.ForceFieldPredictions` keys directly,
    which is what stage 1 needs — its target is the **partial Hessian**, the
    ``3x3`` mixed blocks on the atom pairs that share a bonded term, and no
    Cartesian energy or force appears in it at all.

    How much work a declared key costs varies, and that is the point of letting
    the contract choose: ``"bond_k"`` / ``"angle_k"`` / ``"charge"`` are reads of
    a table the heads already wrote, assembling no potential; ``"partial_hessian"``
    assembles the bonded terms and takes a term-local second derivative
    (:meth:`~molpot.forcefield.potentials.Potentials.partial_hessian`), which is
    still far cheaper than a ``(3N, 3N)`` Hessian.

    **Its ``state_dict`` is interchangeable with** :class:`RondoPotential`'s. Both
    hold the parameterizer under the attribute name ``parameterizer``, so both
    produce ``parameterizer.*`` keys and a staged fit can carry one stage's
    checkpoint into the next. A wrapper that stored it under any other name would
    load partially and silently, which is the failure that made three earlier
    production checkpoints unusable.

    Args:
        parameterizer: The Rondo recipe whose predictions are read.
        keys: Predictions to declare, as
            :class:`~molpot.forcefield.predictions.ForceFieldPredictions` names
            them — ``("partial_hessian",)`` is ByteFF's first stage.
        trainable: Namespaces of the predicted parameter tables that keep a
            gradient path, as for :class:`RondoPotential`. Stage 1 passes
            ``("bonds", "angles", "impropers")``: the propers that share those
            pairs are still priced, so the pair blocks are complete, but the
            partial-Hessian residual cannot move them.

    Example::

        model = RondoPredictions(Rondo(spec))
        predictions = model(batch)      # {"partial_hessian": (n_pairs, 9)}
    """

    def __init__(
        self,
        parameterizer: nn.Module,
        *,
        keys: Sequence[str] = ("partial_hessian",),
        trainable: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        unknown = [key for key in keys if key not in self.TABLE_KEYS and key != "partial_hessian"]
        if unknown:
            raise ValueError(
                f"RondoPredictions cannot answer {unknown}; it declares "
                f"{sorted((*self.TABLE_KEYS, 'partial_hessian'))}. An energy or a force "
                "is RondoPotential's."
            )
        self.parameterizer = parameterizer
        self.keys = tuple(keys)
        self.trainable = None if trainable is None else tuple(trainable)
        # Assembled once on the first batch, off the parameter tree — as in
        # RondoPotential, and for the same reason.
        self._physics: Any = None

    #: Predictions that are a read of one parameter table: ``key -> (namespace, name)``.
    TABLE_KEYS: dict[str, tuple[str, str]] = {
        "charge": ("atoms", "charge"),
        "bond_k": ("bonds", "k"),
        "angle_k": ("angles", "k"),
    }

    def _hold(self, name: str, value: Any) -> None:
        """Keep an ``nn.Module`` off the parameter tree (see RondoPotential)."""
        object.__setattr__(self, name, value)

    def physics(self, batch: TensorDict) -> Any:
        """The assembled Class-I container, bound to this batch's topology.

        Host-side and untraceable, exactly as :meth:`RondoPotential.physics` is and
        for the same reason: assembling it constructs a molpy ``ForceField``, and
        ``ForceField.__new__`` inside a compiled region is
        ``Unsupported: Unsupported method call``. The first per-stage calibration
        found that the hard way — every cell of the ``hessian`` and ``mbis`` grids
        (jobs 2751981, 2751982), because this class used to reach the container
        through ``ForceFieldPredictions``, whose contract is one
        ``parameterizer(batch)`` forward *returning a filled ForceField*. That is
        the right shape for the export and one-shot paths it serves, and the wrong
        one for a traced training step, which must see tensors only.

        Only ``"partial_hessian"`` needs it. A pure table read declares no container
        and assembles none.
        """
        if self._physics is None:
            with torch.no_grad():
                # Seed on a copy: the encoder writes per-namespace features into
                # whatever batch it is handed, and a column count is a dynamo guard.
                seed = self.parameterizer(batch.clone())
                self._hold("_physics", seed.to_potentials(wrt=()).warmup(batch))
        else:
            self._physics.bind_topology(batch)
        # The curvature is taken in the positions, so they have to be a variable of
        # the graph — and marking one inside a compiled region is
        # `Unsupported: returning intermediate with requires_grad_()`. Marked here,
        # on the host, exactly as `Derivative.mark_leaves` marks the columns it
        # differentiates. Nothing reads the gradient *with respect to* the
        # positions; stage 1 still fits only parameters.
        pos = batch["atoms", "pos"]
        if not pos.requires_grad:
            pos.requires_grad_(True)
        return self._physics

    def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """The declared predictions for this batch.

        The tables are computed **here**, inside whatever region this runs in, from
        :meth:`~molpot.forcefield.Parameterizer.tables` — plain tensors, no molpy.
        A table produced outside would carry a ``grad_fn`` that
        ``torch.autograd.grad`` refuses to trace through, which is the same reason
        :meth:`RondoPotential.energy` computes them itself.

        :attr:`trainable` detaches the namespaces it does not name, after prediction
        and before anything reads them, so their parameters are still *used* at
        their predicted values and only their gradient path is cut.

        Args:
            batch: Post-collate batch carrying the Class-I topology namespaces.

        Returns:
            Exactly the declared keys, in each prediction's own units —
            ``(n_pairs, 9)`` kcal mol⁻¹ Å⁻² for ``partial_hessian``, ``(n_atoms,)``
            ``e`` for ``charge``, ``(n_terms,)`` kcal mol⁻¹ Å⁻² for ``bond_k`` and
            kcal mol⁻¹ rad⁻² for ``angle_k``.

        Raises:
            RuntimeError: ``"partial_hessian"`` is declared and :meth:`physics` has
                not run — a compiled forward cannot assemble the container itself.
            KeyError: The model's heads wrote no table a declared key reads.
        """
        needs_physics = "partial_hessian" in self.keys
        if needs_physics and not torch.compiler.is_compiling():
            self.physics(batch)
        if needs_physics and self._physics is None:
            raise RuntimeError(
                "RondoPredictions.physics(batch) must run before the compiled forward"
            )
        tables = self.parameterizer.tables(batch)
        if self.trainable is not None:
            tables = {
                namespace: (
                    params
                    if namespace in self.trainable
                    else {name: value.detach() for name, value in params.items()}
                )
                for namespace, params in tables.items()
            }
        out: dict[str, torch.Tensor] = {}
        for key in self.keys:
            if key == "partial_hessian":
                out[key] = self._physics.partial_hessian(batch, tables)
                continue
            namespace, name = self.TABLE_KEYS[key]
            table = tables.get(namespace)
            if table is None or name not in table:
                raise KeyError(
                    f"{key!r} reads {namespace}[{name!r}], which this model's heads did "
                    f"not write for this batch (namespaces: {sorted(tables)})"
                )
            out[key] = table[name]
        return out
