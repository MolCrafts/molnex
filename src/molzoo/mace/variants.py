"""Named MACE foundation models: the keyword surface the scripts already bind.

A *foundation* model is one shipped with weights already fitted on a large,
chemically broad dataset; molnex carries two, MatPES (periodic materials,
arXiv:2503.04070) and OMol (molecules, additionally conditioned on the total
charge and total spin of the system). Everything that *defines* them lives
elsewhere — the hyper-parameters and architecture switches in
:mod:`molzoo.mace.spec`, the module graph in :mod:`molzoo.mace.encoder`, the
energy and forces in :mod:`molzoo.mace.potential`, the official-checkpoint key
dialect in :mod:`molzoo.mace.checkpoint`. This module adds **no** physics, no
hyper-parameter, no *irreps* string and no ``E0`` table of its own: it is the
adapter that lets the pre-restructure call sites keep working unchanged. (An
*irrep*, short for irreducible representation, labels how a feature transforms
when the molecule is rotated — a scalar stays put, a vector turns with it — and
an irreps string such as ``128x0e+128x1o`` says how many features of each kind
a layer carries; ``E0`` is the table of isolated-atom reference energies, one
per element, in eV/atom.)

One such call site is still real and out of this spec's reach::

    benchmarks/bench_mace_matpes.py:41    MACEMatpes(atomic_numbers=…, …)

``scripts/matpes_port/run_nve.py`` was the second until
``mace-subpackage-restructure-07-cleanup`` collapsed its ``build_model`` onto
:meth:`~molzoo.mace.potential.MACEPotential.from_checkpoint`, which builds the
spec itself. So :class:`MACEMatpes` and :class:`MACEOMol` stay constructible by
keyword, and :func:`load_matpes_state_dict` / :func:`load_omol_state_dict` stay
importable as free functions. Both loaders are *only* here for that
compatibility — CLAUDE.md's "no factory functions" rule would otherwise reject
them, and they forward, unchanged, to the two preset
:class:`~molzoo.mace.checkpoint.CheckpointRemap` instances.

**New code should not use anything in this module.** Build the configuration
and the potential directly, and load a checkpoint through the remap::

    from molzoo.mace import MACEMatpesSpec, MACEPotential
    from molzoo.mace.checkpoint import MATPES_REMAP

    potential = MACEPotential(MACEMatpesSpec(atomic_numbers=…, atomic_energies=…))
    MATPES_REMAP.load(potential, state)

That form says which variant it means in the *type* of the spec, takes the
hyper-parameters from one validated place, and is what
:meth:`~molzoo.mace.potential.MACEPotential.from_checkpoint` already does for a
stock MatPES checkpoint.

The keyword vocabulary, once, for both classes below
----------------------------------------------------

MACE describes each atom by the neighbours inside a cutoff radius ``r_max``
(Å) and refines that description over ``num_interactions`` rounds of *message
passing* — one round lets an atom mix in features of its neighbours. Each
neighbour enters through two expansions: its distance in ``num_bessel``
*Bessel* radial functions (the shapes ``sin(ω_n r) / r``, with ``r`` the
interatomic distance in Å and the frequencies ``ω_n`` in Å⁻¹), and its
direction in *spherical harmonics* — the standard angular functions on the
sphere — up to angular order ``l_max`` (order 0 is a plain scalar, 1 turns like
a vector, and so on). ``num_features`` is how many rotation-invariant numbers
each atom carries, ``max_hidden_l`` the highest angular order kept in the node
state between layers, and ``correlation`` the *body order* of the symmetric
contraction — how many neighbours may enter one term of the many-body
expansion. ``num_polynomial_cutoff`` is the exponent ``p`` of the polynomial
envelope that takes an edge's weight smoothly to zero as its length approaches
``r_max``. ``scale`` and ``shift`` are the fitted affine normalisation of the
learned per-atom energy (eV), on top of which the ``E0`` table contributes the
isolated-atom references; the full energy expression, its ``F_i = -∂E/∂r_i``
force convention and its units are written out in
:mod:`molzoo.mace.potential`.

Resolved debt — the private energy core
---------------------------------------

``scripts/matpes_port/run_nve.py`` used to bind ``model._compute_energy`` (and
``benchmarks/bench_mace_matpes.py`` to hand it to ``Compiler``) as the compiled
energy core of the MD loop — a **private** method crossing a package boundary,
discovered by the ``mace-subpackage-restructure`` cutover. Both consumers were
re-pointed at the public
:meth:`~molzoo.mace.potential.MACEPotential.energy_core` by
``mace-subpackage-restructure-07-cleanup``; no in-tree caller binds the private
name any more. :attr:`MACEMatpes._compute_energy` is kept as a *name alias* of
``energy_core`` — the same function object, so the two can never disagree —
purely as back-compat for out-of-tree callers, and the positional signature
``(pos, Z, edge_index, batch, num_graphs, shifts)`` stays a contract for them.
:class:`MACEOMol` deliberately has **no** such alias: the flat OMol model's
``_compute_energy`` took ``(…, batch, total_charge, total_spin, shifts)``, so an
alias of ``energy_core`` there would silently reinterpret the fifth positional
argument as ``num_graphs``.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
    Batatia et al. "A foundation model for atomistic materials chemistry"
    (MACE-MP-0). https://arxiv.org/abs/2401.00096
    Kaplan et al. "A foundational potential energy surface dataset for
    materials" (MatPES). https://arxiv.org/abs/2503.04070
    Levine et al. "The Open Molecules 2025 (OMol25) Dataset, Evaluations, and
    Models" https://arxiv.org/abs/2505.08762
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from molpot.derivation.force import autograd_forces_from_energy
from molzoo.mace.checkpoint import MATPES_REMAP, OMOL_REMAP
from molzoo.mace.potential import MACEPotential
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec

__all__ = [
    "MACEMatpes",
    "MACEOMol",
    "load_matpes_state_dict",
    "load_omol_state_dict",
]


def _supplied(**fields: object) -> dict[str, object]:
    """Drop the ``None`` placeholders, leaving only the caller's own values.

    The keyword constructors below take ``None`` for "not given" instead of
    repeating the shipped defaults. Restating ``num_bessel=10`` here would put a
    second copy of every foundation hyper-parameter one module away from
    :mod:`molzoo.mace.spec` — where the defaults are validated and
    gate-checked against the flat constructors they replace (the
    ``test_field_default_matches_the_flat_constructor`` cases in
    ``tests/test_molzoo/test_mace/test_spec.py``).
    Filtering instead means the spec stays the single source.

    Args:
        **fields: Spec field name → the caller's value, or ``None``.

    Returns:
        The subset whose value is not ``None``, ready to splat into a spec.
    """
    return {name: value for name, value in fields.items() if value is not None}


def _energy_table(atomic_energies: Sequence[float] | torch.Tensor) -> list[float]:
    """Normalise the ``E0`` table to the plain ``list[float]`` the spec holds.

    ``benchmarks/bench_mace_matpes.py:43`` passes a ``torch.Tensor``
    (``torch.zeros(len(_Z_TABLE), dtype=config.ftype)``),
    while :class:`~molzoo.mace.spec.MACESpec` keeps the table torch-free so a
    configuration can be read without importing torch. Both shapes are accepted
    here; the tensor is rebuilt inside
    :class:`~molzoo.mace.encoder.MACEEncoder`.

    Args:
        atomic_energies: Per-element reference energies ``E0`` in eV/atom, in
            ``atomic_numbers`` order — a sequence or a 1-D tensor.

    Returns:
        The same energies as a ``list[float]`` (eV/atom).
    """
    return [float(energy) for energy in atomic_energies]


class MACEMatpes(MACEPotential):
    """MACE-MatPES foundation model, constructed by keyword.

    A thin adapter over :class:`~molzoo.mace.potential.MACEPotential`: the
    keywords below are mapped one-for-one onto
    :class:`~molzoo.mace.spec.MACEMatpesSpec` and nothing else happens. The
    energy it evaluates, the ``F_i = -∂E/∂r_i`` convention and the units
    (eV, eV/Å, Å) are documented on :mod:`molzoo.mace.potential`; the shipped
    defaults on :class:`~molzoo.mace.spec.MACEMatpesSpec`. Prefer the spec form
    in new code — see the module docstring, which also glosses the MACE
    vocabulary the keywords below are named in.

    Only ``atomic_numbers`` and ``atomic_energies`` are required; every other
    argument defaults to ``None``, meaning *not given*, in which case the
    spec's own default applies. Those defaults are the ones this class'
    predecessor hard-coded (the pre-cutover flat ``MACEMatpes``, deleted in
    06-wire; see git history).

    Args:
        atomic_numbers: Element table (z-table) in checkpoint order, strictly
            ascending.
        atomic_energies: Per-element reference energies ``E0`` in eV/atom, same
            order. A sequence or a 1-D ``torch.Tensor``.
        r_max: Radial cutoff in Å.
        num_bessel: Number of Bessel radial basis functions.
        num_polynomial_cutoff: Polynomial cutoff exponent ``p`` (also the
            envelope exponent of the ZBL — Ziegler–Biersack–Littmark —
            short-range nuclear repulsion term, matching MACE).
        l_max: Maximum spherical-harmonics order.
        num_features: Scalar channel multiplicity (``hidden_irreps`` 0e count).
        max_hidden_l: Highest ``l`` carried in the node state between layers.
        num_interactions: Number of interaction/product layers.
        correlation: Body-order correlation of the symmetric contraction.
        mlp_dim: Hidden width of the final non-linear readout, the small
            multi-layer perceptron (MLP) that turns features into an energy.
        radial_mlp: Hidden widths of the radial weight MLP.
        scale: ``atomic_inter_scale`` — multiplies the interaction energy (eV).
        shift: ``atomic_inter_shift`` — per-atom energy offset in eV.
        use_fallback: ``True`` puts the equivariant blocks on the pure-torch
            path of cuEquivariance — NVIDIA's GPU library for the equivariant
            tensor algebra MACE is built from — instead of its *fused* kernels
            (``False``, the shipped default): single GPU kernels that do a
            whole tensor product at once, which need a GPU plus the
            ``cuequivariance-ops-torch`` wheel and are ~36x faster per step of
            molecular dynamics (MD).

    Raises:
        ValueError: From :class:`~molzoo.mace.spec.MACEMatpesSpec` — a table
            that is not strictly ascending, an ``E0`` list of the wrong length,
            or fewer than two interaction layers.
    """

    #: Name alias of :meth:`~molzoo.mace.potential.MACEPotential.energy_core`.
    #: Same function object, so the positional signature
    #: ``(pos, Z, edge_index, batch, num_graphs, shifts)`` cannot drift from the
    #: public one. ``mace-subpackage-restructure-07-cleanup`` re-pointed the two
    #: in-tree consumers (``scripts/matpes_port/run_nve.py``,
    #: ``benchmarks/bench_mace_matpes.py``) onto ``energy_core``; the alias
    #: remains as back-compat for out-of-tree callers and may be dropped by a
    #: future spec.
    _compute_energy = MACEPotential.energy_core

    def __init__(
        self,
        *,
        atomic_numbers: Sequence[int],
        atomic_energies: Sequence[float] | torch.Tensor,
        r_max: float | None = None,
        num_bessel: int | None = None,
        num_polynomial_cutoff: int | None = None,
        l_max: int | None = None,
        num_features: int | None = None,
        max_hidden_l: int | None = None,
        num_interactions: int | None = None,
        correlation: int | None = None,
        mlp_dim: int | None = None,
        radial_mlp: Sequence[int] | None = None,
        scale: float | None = None,
        shift: float | None = None,
        use_fallback: bool | None = None,
    ) -> None:
        super().__init__(
            MACEMatpesSpec(
                atomic_numbers=list(atomic_numbers),
                atomic_energies=_energy_table(atomic_energies),
                **_supplied(
                    r_max=r_max,
                    num_bessel=num_bessel,
                    num_polynomial_cutoff=num_polynomial_cutoff,
                    l_max=l_max,
                    num_features=num_features,
                    max_hidden_l=max_hidden_l,
                    num_interactions=num_interactions,
                    correlation=correlation,
                    mlp_dim=mlp_dim,
                    radial_mlp=None if radial_mlp is None else list(radial_mlp),
                    scale=scale,
                    shift=shift,
                    use_fallback=use_fallback,
                ),
            )
        )

    def energy_forces(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int | None = None,
        shifts: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Energy and forces from raw tensors, off the batch schema.

        The raw-tensor twin of :meth:`~molzoo.mace.potential.MACEPotential.forward`,
        kept because ``benchmarks/bench_mace_matpes.py:115`` calls it. One energy
        forward, then one ``torch.autograd.grad`` on a position *leaf* — the
        tensor autograd differentiates with respect to — created and owned by
        this call. That is MACE's own ``get_outputs`` shape, and the same
        tensor-level kernel
        (:func:`molpot.derivation.force.autograd_forces_from_energy`) that
        :func:`molpot.derivation.kernels.grad_force_pass` uses on the batch
        path. No third force path is derived here.

        The returned energy is always detached: this method creates the leaf, so
        the caller has no graph to lose (the same ``needs_leaf`` policy
        ``forward`` gets from ``detach_energy=None``). Differentiating the energy
        w.r.t. the parameters needs the batch path, not this one.

        Args:
            positions: Atom positions ``(N, 3)`` in Å, for the ``N`` atoms of
                the system.
            Z: Atomic numbers ``(N,)``. Must lie inside this model's element
                table; **not** checked on this path — an off-table element is
                snapped onto a neighbouring row by ``torch.searchsorted`` and
                yields a plausible, wrong energy, so call
                :meth:`~molzoo.mace.encoder.MACEEncoder.validate_elements` once
                per (model, system) pair.
            edge_index: ``(E, 2)`` for the ``E`` neighbour pairs, with
                ``[:, 0]`` = source, ``[:, 1]`` = target.
            batch: Graph index per atom ``(N,)`` — ``batch[i]`` says which
                graph atom ``i`` belongs to.
            num_graphs: Number of graphs ``B`` (one graph = one molecule or one
                periodic cell). Inferred from ``batch`` when omitted, which
                costs a *host sync* — the CPU waits for the GPU to hand the
                value back — so pass it on hot paths.
            shifts: Optional periodic shift vectors ``(E, 3)`` in Å, added to
                the edge displacements so a neighbour across a periodic
                boundary enters at its imaged position.

        Returns:
            ``{"energy": (B,) eV, "forces": (N, 3) eV/Å}``.
        """
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1
        leaf = positions.detach().requires_grad_(True)
        with torch.enable_grad():
            energy = self.energy_core(leaf, Z, edge_index, batch, num_graphs, shifts)
        return {"energy": energy.detach(), "forces": autograd_forces_from_energy(energy, leaf)}


class MACEOMol(MACEPotential):
    """MACE-OMol foundation model, constructed by keyword.

    A thin adapter over :class:`~molzoo.mace.potential.MACEPotential`, mapping
    its keywords onto :class:`~molzoo.mace.spec.MACEOMolSpec`; see
    :class:`MACEMatpes` and the module docstring for the shape, for the keyword
    vocabulary, and for why new code should build the spec directly. OMol adds
    the per-graph charge/spin *conditioning*: the integer total charge and
    total spin of a molecule index a learned vector in an *embedding* table —
    a lookup from an integer to a trainable vector — which is added to every
    atom's initial features. Its energy therefore carries one further per-atom
    term outside the ``scale``/``shift`` normalisation
    (:mod:`molzoo.mace.potential`).

    ``use_fallback`` is deliberately absent, as it was on the predecessor (the
    pre-cutover flat ``MACEOMol``, deleted in 06-wire; see git history): that
    model hard-coded the fused cuEquivariance path, which is what
    :class:`~molzoo.mace.spec.MACEOMolSpec` defaults to.

    Args:
        atomic_numbers: Element table (z-table) in checkpoint order, strictly
            ascending.
        atomic_energies: Per-element reference energies ``E0`` in eV/atom, same
            order. A sequence or a 1-D ``torch.Tensor``.
        r_max: Radial cutoff in Å.
        num_bessel: Number of Bessel radial basis functions.
        num_polynomial_cutoff: Polynomial cutoff exponent ``p``.
        l_max: Maximum spherical-harmonics order (at least 1).
        num_features: Scalar channel multiplicity (``hidden_irreps`` 0e count).
        num_interactions: Number of interaction/product layers.
        correlation: Body-order correlation of the symmetric contraction.
        mlp_dim: Hidden width of the final non-linear readout, the small
            multi-layer perceptron (MLP) that turns features into an energy.
        edge_channels: Per-``l`` channel count of the mid-layer edge irreps and
            the radial-MLP hidden width.
        charge_classes: Number of rows in the total-charge embedding table.
        charge_offset: Added to the total charge to get the row index (row =
            ``total_charge + charge_offset``), so that negative charges still
            land on a valid row — with the shipped ``100``, charge −100 → row 0.
        spin_classes: Number of rows in the total-spin embedding table.
        spin_offset: Added to the total spin to get the row index, on the same
            convention as ``charge_offset``.
        scale: ``atomic_inter_scale`` — multiplies the interaction energy (eV).
        shift: ``atomic_inter_shift`` — per-atom energy offset in eV.

    Raises:
        ValueError: From :class:`~molzoo.mace.spec.MACEOMolSpec` — a table that
            is not strictly ascending, an ``E0`` list of the wrong length, or
            ``l_max`` below 1.
    """

    def __init__(
        self,
        *,
        atomic_numbers: Sequence[int],
        atomic_energies: Sequence[float] | torch.Tensor,
        r_max: float | None = None,
        num_bessel: int | None = None,
        num_polynomial_cutoff: int | None = None,
        l_max: int | None = None,
        num_features: int | None = None,
        num_interactions: int | None = None,
        correlation: int | None = None,
        mlp_dim: int | None = None,
        edge_channels: int | None = None,
        charge_classes: int | None = None,
        charge_offset: int | None = None,
        spin_classes: int | None = None,
        spin_offset: int | None = None,
        scale: float | None = None,
        shift: float | None = None,
    ) -> None:
        super().__init__(
            MACEOMolSpec(
                atomic_numbers=list(atomic_numbers),
                atomic_energies=_energy_table(atomic_energies),
                **_supplied(
                    r_max=r_max,
                    num_bessel=num_bessel,
                    num_polynomial_cutoff=num_polynomial_cutoff,
                    l_max=l_max,
                    num_features=num_features,
                    num_interactions=num_interactions,
                    correlation=correlation,
                    mlp_dim=mlp_dim,
                    edge_channels=edge_channels,
                    charge_classes=charge_classes,
                    charge_offset=charge_offset,
                    spin_classes=spin_classes,
                    spin_offset=spin_offset,
                    scale=scale,
                    shift=shift,
                ),
            )
        )

    def energy_forces(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        total_charge: torch.Tensor,
        total_spin: torch.Tensor,
        shifts: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Energy and forces from raw tensors, conditioning tensors included.

        The OMol counterpart of :meth:`MACEMatpes.energy_forces`; the leaf
        ownership, the detach policy and the force kernel are identical (see
        there). ``num_graphs`` is read off ``total_charge``'s static shape
        rather than ``batch.max().item()``, so this path costs no host sync.

        Args:
            positions: Atom positions ``(N, 3)`` in Å, for the ``N`` atoms of
                the system.
            Z: Atomic numbers ``(N,)``. Must lie inside this model's element
                table; **not** checked on this path — see
                :meth:`MACEMatpes.energy_forces` for why that matters and
                :meth:`~molzoo.mace.encoder.MACEEncoder.validate_elements` for
                the once-per-system check.
            edge_index: ``(E, 2)`` for the ``E`` neighbour pairs, with
                ``[:, 0]`` = source, ``[:, 1]`` = target.
            batch: Graph index per atom ``(N,)`` — ``batch[i]`` says which
                graph atom ``i`` belongs to.
            total_charge: Per-graph total charge ``(B,)`` in units of the
                elementary charge ``e``, for the ``B`` graphs in the batch. Its
                length is what fixes ``B`` here.
            total_spin: Per-graph total spin ``(B,)``, dimensionless, on the
                multiplicity ``2S+1`` convention with ``S`` the total electron
                spin (``1`` = closed-shell singlet, every electron paired).
            shifts: Optional periodic shift vectors ``(E, 3)`` in Å, added to
                the edge displacements so a neighbour across a periodic
                boundary enters at its imaged position.

        Returns:
            ``{"energy": (B,) eV, "forces": (N, 3) eV/Å}``.
        """
        leaf = positions.detach().requires_grad_(True)
        with torch.enable_grad():
            energy = self.energy_core(
                leaf,
                Z,
                edge_index,
                batch,
                total_charge.shape[0],
                shifts,
                total_charge=total_charge,
                total_spin=total_spin,
            )
        return {"energy": energy.detach(), "forces": autograd_forces_from_energy(energy, leaf)}


def load_matpes_state_dict(model: MACEPotential, cueq_state: dict[str, torch.Tensor]) -> None:
    """Load a cueq-converted MACE-MatPES ``state_dict``, strictly.

    A ``state_dict`` is PyTorch's flat ``name -> tensor`` dump of a model's
    weights; *cueq-converted* means it has already been rewritten, out of tree
    by ``mace.cli.convert_e3nn_cueq``, into the layout of cuEquivariance —
    NVIDIA's GPU library for the equivariant tensor algebra MACE is built from.

    Compatibility alias, forwarding to :data:`molzoo.mace.checkpoint.MATPES_REMAP`
    — the ``on_unexpected="raise"`` preset of
    :class:`~molzoo.mace.checkpoint.CheckpointRemap`, which owns the key dialect
    and the strictness doctrine: every learnable tensor of ``model`` must be
    filled by the checkpoint and every checkpoint weight must land somewhere. A
    silently dropped tensor is the failure mode that produces a model which
    runs, looks sane, and is quietly wrong. New code should call
    ``MATPES_REMAP.load(model, state)`` directly, or build the model with
    :meth:`~molzoo.mace.potential.MACEPotential.from_checkpoint`.

    Args:
        model: Target model, constructed with the checkpoint's
            hyper-parameters — a :class:`MACEMatpes` or any
            :class:`~molzoo.mace.potential.MACEPotential` built from a
            :class:`~molzoo.mace.spec.MACEMatpesSpec`.
        cueq_state: ``state_dict`` of the cueq-converted official model
            (``mace.cli.convert_e3nn_cueq``, run out of tree).

    Returns:
        ``None`` — the legacy contract. The remap's
        ``(missing_buffers, unexpected)`` report is dropped; under this policy
        ``unexpected`` is empty by construction and ``missing_buffers`` holds
        only entries cuEquivariance rebuilds. Call the remap directly to see it.

    Raises:
        RuntimeError: If a checkpoint key has no home, a shape disagrees, or a
            model parameter is left unfilled.
    """
    MATPES_REMAP.load(model, cueq_state)


def load_omol_state_dict(
    model: MACEPotential, cueq_state: dict[str, torch.Tensor]
) -> tuple[list[str], list[str]]:
    """Load a cueq-converted MACE-OMol ``state_dict``, strictly.

    Compatibility alias, forwarding to :data:`molzoo.mace.checkpoint.OMOL_REMAP`
    — the ``on_unexpected="return"`` preset of
    :class:`~molzoo.mace.checkpoint.CheckpointRemap`. Unhoused checkpoint keys
    are *returned* rather than raised, because the OMol checkpoint family
    carries auxiliary heads this port deliberately does not model; an unfilled
    parameter or a shape disagreement still raises. New code should call
    ``OMOL_REMAP.load(model, state)`` directly.

    Args:
        model: Target model, constructed with the checkpoint's
            hyper-parameters — a :class:`MACEOMol` or any
            :class:`~molzoo.mace.potential.MACEPotential` built from a
            :class:`~molzoo.mace.spec.MACEOMolSpec`.
        cueq_state: ``state_dict`` of the cueq-converted official model.

    Returns:
        ``(missing_buffers, unexpected)`` — the non-learnable entries the
        checkpoint did not provide (buffers, not ``nn.Parameter``) and the
        renamed checkpoint keys with no home in ``model``. Both lists are
        sorted by :meth:`~molzoo.mace.checkpoint.CheckpointRemap.load`; the
        legacy contract, passed through unchanged.

    Raises:
        RuntimeError: If a learnable parameter is left unfilled or a shape
            disagrees.
    """
    return OMOL_REMAP.load(model, cueq_state)
