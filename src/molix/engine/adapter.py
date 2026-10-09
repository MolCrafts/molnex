"""Model adapters for the generic ``pair_style molnex`` LAMMPS interface.

The LAMMPS C++ pair style talks to an AOT-Inductor ``.so`` through
:class:`molnex.interface.ModelRunner` with a *flat tensor* calling convention —
``TensorDict`` never crosses into C++. The contract is therefore the **export
calling convention**, not a model's in-memory ``forward`` (schema v3):

    inputs  : Z ``(N,)`` int64, pos ``(N, 3)`` float, pair_index ``(E, 2)`` int64,
              local_mask ``(N,)`` bool
    outputs : energy ``()`` scalar float, forces ``(N, 3)`` float,
              atomic_energy ``(N,)`` float

``local_mask`` marks the atoms this MPI rank *owns*; the rest are ghosts —
copies of atoms owned by a periodic image or a neighbouring rank. A partitioned
run must sum only local site energies, ``E⁽ᵖ⁾ = Σ_{i∈L} E_i`` (spec
``lammps-iface-05-pbc`` D1), because a site energy carries a per-species offset
of order tens of eV and counting a ghost's would corrupt the total outright.
That is why the per-atom energy — not the scalar — is the authoritative output
here: a scalar total cannot be partitioned after the fact.

An :class:`EngineAdapter` is the Python-side shim that maps this flat convention
onto a specific model's real input/output shapes. A molnex-native model (nested
``TensorDict`` forward) uses :class:`MolnexTensorDictAdapter`; a third-party model
that already speaks the flat convention uses :class:`FlatTensorAdapter`; anything
else gets a ~15-line subclass implementing :meth:`build_inputs` / :meth:`read_outputs`.

Adapters self-register under a string name (the same strategy-pattern + registry
used by :mod:`molix.quant`), so :func:`molix.engine.export_for_lammps` can select
one by name and stamp it into ``meta.json`` for provenance.

Edge convention follows the repo-wide rule (see ``CLAUDE.md`` → Edge Convention):
``pair_index[:, 0]`` is the source, ``[:, 1]`` the target, and
``pair_diff = pos[target] - pos[source]``.

Edge **order** is unconstrained by default: molnex aggregation is scatter-class
(``index_add_`` / ``scatter_add_``), which is permutation-invariant, so the C++
side hands over the neighbour list in its raw order and pays nothing per step.
A model whose aggregation *is* order-sensitive — segmented or CSR reductions
require rows grouped by index — must declare ``requires_sorted_edges=True`` at
export; then, and only then, does the C++ side sort by ``(source, target)``
before the call. Adapters must not assume an order the export did not declare.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from molix.autograd import leaf
from molix.schema import (
    ATOMIC_ENERGY_FLAT_KEY,
    ATOMIC_ENERGY_KEY,
    ENERGY_KEY,
    FORCES_KEY,
    PAIR_DIST_CLAMP_MIN,
)


class EngineAdapter(ABC):
    """Strategy mapping the flat LAMMPS calling convention onto one model.

    Concrete subclasses set a class-level :attr:`name` (auto-registered) and
    implement :meth:`build_inputs` (flat tensors → model input) and
    :meth:`read_outputs` (model output → ``(energy_scalar, forces)``). The whole
    adapter is wrapped in :class:`EngineForward`, which is the ``nn.Module`` that
    actually gets AOT-exported.
    """

    name: str = ""
    _registry: dict[str, type[EngineAdapter]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            EngineAdapter._registry[cls.name] = cls

    @classmethod
    def from_name(cls, name: str) -> EngineAdapter:
        """Instantiate the registered adapter for ``name`` (e.g. ``"molnex-tensordict"``)."""
        try:
            return cls._registry[name]()
        except KeyError:
            valid = ", ".join(sorted(cls._registry))
            raise ValueError(f"unknown adapter {name!r}; valid adapters: {valid}") from None

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """All registered adapter names."""
        return tuple(sorted(cls._registry))

    @abstractmethod
    def build_inputs(
        self, model: nn.Module, Z: torch.Tensor, pos: torch.Tensor, pair_index: torch.Tensor
    ) -> object:
        """Convert flat LAMMPS tensors into whatever ``model`` consumes.

        Args:
            model: The wrapped potential.
            Z: Atomic numbers ``(N,)`` int64.
            pos: Positions ``(N, 3)`` float (model dtype).
            pair_index: Source-target pairs ``(E, 2)`` int64, in whatever order
                the caller produced them — the C++ side sorts by
                ``(source, target)`` only for exports declaring
                ``requires_sorted_edges=True`` (see the module docstring).

        Returns:
            The object passed to ``model(...)`` (e.g. a nested ``TensorDict``).
        """

    @abstractmethod
    def read_outputs(self, out: object) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Extract ``(energy (), forces (N, 3), atomic_energy (N,))`` from the output.

        The per-atom energy is mandatory under schema v3: it is what the LAMMPS
        side sums over local sites, so an adapter that cannot supply it must
        raise rather than return a placeholder.
        """

    @staticmethod
    def read_energy_triple(out: object) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Decode a molnex ``TensorDict`` forward into the v3 triple.

        The per-atom energy is ``("atoms", "energy")``. Forces are optional
        and live at ``("atoms", "forces")``; the total is ``("graphs", "energy")``.
        A third-party flat dict is :class:`FlatTensorAdapter`'s own reader,
        not this one.

        Forces are **optional**: the export wrappers derive the deployed forces
        from the site energies themselves, so an energy-only model (e.g.
        ``PiNetPotential()`` with the default ``wrt=()``) is a complete v3
        citizen — its slot comes back ``None`` and the export's
        force-consistency check is vacuous. When a model does emit forces they
        are read and checked. Energy-only is also the *safe* construction:
        the wrapper differentiates the site energies itself, and a model whose
        own op has already consumed the energy graph under ``no_grad`` would
        die with "backward through the graph a second time" at trace
        preparation.

        Lives on the base class because :class:`MolnexTensorDictAdapter` and
        :class:`molix.engine.static.StaticForward` share it.

        Args:
            out: The model's return value, a nested ``TensorDict``.

        Returns:
            ``(energy (), forces (N, 3), atomic_energy (N,))``. The energy is
            summed to a scalar; a ``(1,)`` single-graph energy is the usual input.

        Raises:
            ValueError: If ``out`` is not a ``TensorDict``, or carries no
                per-atom energy. Refusing is deliberate: the per-atom energy
                is the only source of the LAMMPS partition energy, so dropping
                it silently would deploy a model that reports zero.
        """
        from tensordict import TensorDict

        if not isinstance(out, TensorDict):
            raise ValueError(f"model output must be a nested TensorDict; got {type(out).__name__}")
        has_atomic = "atoms" in out.keys() and "energy" in out["atoms"].keys()
        atomic = out[ATOMIC_ENERGY_KEY] if has_atomic else None
        has_forces = "atoms" in out.keys() and "forces" in out["atoms"].keys()
        energy = out[ENERGY_KEY]
        forces = out[FORCES_KEY] if has_forces else None
        if atomic is None:
            raise ValueError(
                "model output carries no per-atom energy, which schema v3 requires: "
                f"expected {ATOMIC_ENERGY_KEY}. A LAMMPS rank sums only the site "
                "energies it owns, so a model without them cannot be partitioned."
            )
        return energy.sum(), forces, atomic

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MolnexTensorDictAdapter(EngineAdapter):
    """Adapter for molnex-native potentials (PiNet, MACE, … via ``PiNetPotential``-style).

    Builds the post-collate nested ``TensorDict`` (``atoms`` / ``pairs`` / ``graphs``
    per ``CLAUDE.md``), runs ``model(batch)``, and reads ``graphs.energy``,
    ``atoms.forces``, and ``atoms.energy``. The per-atom energy is the
    one the C++ pair style partitions over local sites; the scalar energy is
    read too, but only so :func:`molix.engine.export_for_lammps` can check
    eagerly that the model is decomposable before exporting it.
    """

    name = "molnex-tensordict"

    def build_inputs(
        self, model: nn.Module, Z: torch.Tensor, pos: torch.Tensor, pair_index: torch.Tensor
    ) -> object:
        from tensordict import TensorDict

        n = Z.shape[0]
        src = pair_index[:, 0]
        tgt = pair_index[:, 1]
        pair_diff = pos[tgt] - pos[src]  # pos[target] - pos[source]
        pair_dist = pair_diff.norm(dim=-1).clamp(min=PAIR_DIST_CLAMP_MIN)
        return TensorDict(
            atoms=TensorDict(
                Z=Z,
                pos=pos,
                batch=torch.zeros(n, dtype=torch.long, device=pos.device),
                batch_size=[n],
            ),
            pairs=TensorDict(
                pair_index=pair_index,
                pair_diff=pair_diff,
                pair_dist=pair_dist,
                batch_size=[pair_index.shape[0]],
            ),
            graphs=TensorDict(
                num_atoms=torch.full((1,), n, dtype=torch.long, device=pos.device),
                batch_size=[1],
            ),
            batch_size=[],
        )

    def read_outputs(self, out: object) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        return self.read_energy_triple(out)


class FlatTensorAdapter(EngineAdapter):
    """Adapter for third-party models already speaking the flat convention.

    The wrapped model must implement ``forward(Z, pos, pair_index)`` returning
    either the ``(energy, forces, atomic_energy)`` triple or a dict with
    ``"energy"`` / ``"forces"`` / ``"atomic_energy"`` keys. Use this as the
    zero-glue path; write a bespoke :class:`EngineAdapter` subclass when a model
    needs real input/output translation.

    A model still returning schema v2's ``(energy, forces)`` pair is **refused**,
    not padded with zeros: without per-site energies a LAMMPS rank cannot tell
    its own sites from its ghosts', and the only alternatives to refusing are
    over-counting every ghost site or reporting zero. Both are silent physics
    errors, so the failure is moved to export time where it is loud and cheap.
    """

    name = "flat"

    def build_inputs(
        self, model: nn.Module, Z: torch.Tensor, pos: torch.Tensor, pair_index: torch.Tensor
    ) -> object:
        return (Z, pos, pair_index)

    def read_outputs(self, out: object) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if isinstance(out, dict):
            atomic = out.get(ATOMIC_ENERGY_FLAT_KEY)
            if "energy" not in out or atomic is None:
                raise ValueError(
                    "flat model dict must carry 'energy' and "
                    f"{ATOMIC_ENERGY_FLAT_KEY!r}; got keys {sorted(out)}"
                )
            return out["energy"].sum(), out.get("forces"), atomic
        # Narrow to a real sequence before counting: a bare Tensor also has a
        # length, so `len(out) != 3` alone would happily unpack the first three
        # ROWS of an (n, ...) tensor into energy/forces/atomic_energy.
        if not isinstance(out, tuple | list):
            raise ValueError(
                "flat model must return the (energy, forces, atomic_energy) triple or a "
                f"dict with those keys; got {type(out).__name__}"
            )
        if len(out) != 3:
            raise ValueError(
                f"flat model returned {len(out)} tensors; schema v3 requires the triple "
                "(energy, forces, atomic_energy). A model that cannot emit per-atom "
                "energies cannot be deployed into a partitioned LAMMPS run — see the "
                "spec lammps-iface-05-pbc, or wrap it in an EngineAdapter subclass that "
                "attributes its energy to sites."
            )
        energy, forces, atomic = out
        return energy.sum(), forces, atomic


class EngineForward(nn.Module):
    """Flat-convention ``nn.Module`` wrapper that is the actual AOT-export target.

    Holds a potential and an :class:`EngineAdapter`. Its ``forward(Z, pos,
    pair_index, local_mask)`` returns ``(energy, forces, atomic_energy)`` — the
    exact signature the C++ ``pair_style molnex`` invokes through the AOTI runner
    under schema v3.

    **The returned energy and forces belong to the partition, not to the whole
    input.** Given site energies :math:`E_i` and the ownership mask
    :math:`m_i \\in \\{0, 1\\}`,

    .. math::

        E^{(p)} = \\sum_i m_i E_i, \\qquad
        \\mathbf F_j = -\\frac{\\partial E^{(p)}}{\\partial \\mathbf r_j}

    for **every** :math:`j`, masked-out sites included. Those cross-terms are
    the whole point: a ghost owns no site energy on this rank yet its
    coordinates enter its neighbours' site energies, so it carries a real force
    that LAMMPS reverse-communicates to the owning rank. Conversely a rank
    owning nothing must produce no force at all.

    This is why the forces are differentiated *here* rather than taken from the
    model: the model's own forces are :math:`-\\partial E_{\\text{total}}/
    \\partial \\mathbf r`, which is the right answer only for an all-ones mask.
    Under any partial mask they are simply a different quantity.

    Two consequences worth knowing:

    * ``pos`` is re-leafed inside ``forward`` (:func:`molix.autograd.leaf`), so
      any autograd graph the caller had through ``pos`` is *not* continued —
      this wrapper returns values for an MD engine, it is not a training path.
      The partitioned energy — a masked sum of site energies read from a
      model output — is not the frame-in/frame-out
      shape :class:`molix.autograd.Derivative` differentiates, so the wrapper
      keeps the leaf and the one ``autograd.grad`` itself.
    * the mask is applied by this wrapper, never by an adapter: adapters
      translate shapes, partitioning is engine semantics.

    Reference:
        Spec ``lammps-iface-05-pbc`` D1/D2; Musaelian et al., "Learning Local
        Equivariant Representations for Large-Scale Atomistic Dynamics",
        Nat. Commun. 14, 579 (2023). https://arxiv.org/abs/2204.05249
    """

    def __init__(self, model: nn.Module, adapter: EngineAdapter | str = "molnex-tensordict"):
        super().__init__()
        self.model = model
        self.adapter = (
            adapter if isinstance(adapter, EngineAdapter) else EngineAdapter.from_name(adapter)
        )

    def model_outputs(
        self, Z: torch.Tensor, pos: torch.Tensor, pair_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Run the model once and read its unmasked ``(energy, forces, atomic_energy)``.

        The seam :func:`molix.engine.export_for_lammps` uses for its eager
        pre-trace checks, which need the model's *own* scalar energy and forces
        — the two quantities :meth:`forward` deliberately replaces with their
        partitioned counterparts.
        """
        inputs = self.adapter.build_inputs(self.model, Z, pos, pair_index)
        if isinstance(inputs, tuple):
            out = self.model(*inputs)
        else:
            # Monomorphic forward: what a potential derives is a
            # construction-time property (``wrt=``), not a call kwarg.
            out = self.model(inputs)
        return self.adapter.read_outputs(out)

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        pair_index: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: ``Z (N,)`` int64, ``pos (N,3)``, ``pair_index (E,2)`` int64,
        ``local_mask (N,)`` bool. Returns ``(energy (), forces (N,3),
        atomic_energy (N,))``."""
        pos = leaf(pos)
        with torch.enable_grad():
            _, _, atomic_energy = self.model_outputs(Z, pos, pair_index)
            # Mask-multiply, never a slice: `n_local` is a runtime scalar and
            # torch.export cannot trace a shape that depends on one.
            energy = (atomic_energy * local_mask.to(atomic_energy.dtype)).sum()
            (grad,) = torch.autograd.grad(energy, pos)
        return energy.detach().reshape(()), -grad, atomic_energy.detach()
