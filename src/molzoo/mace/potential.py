"""Unified MACE energy/force potential for the foundation-model variants.

One host class for every MACE foundation variant; which variant it is comes
entirely from the :class:`~molzoo.mace.spec.MACESpec` it is built from — the
MatPES density stack or the OMOL charge/spin-conditioned residual stack.

Per graph ``g`` (one molecule or one periodic cell)::

    E_g = Σ_{i∈g} E0(Z_i)  +  Σ_{i∈g} (scale · ε_i + shift)     [eV]
    ε_i = ZBL_i + Σ_layer readout_layer(h_i^layer)              [eV]

``E0`` is the frozen isolated-atom reference, ``ZBL`` the short-range nuclear
repulsion (only with ``pair_repulsion="zbl"``, zero otherwise), and
``scale`` / ``shift`` the fitted per-atom normalisation. MatPES reads out every
layer; OMol reads out the last one and adds its charge/spin embedding next to
``E0``, outside the normalisation. Forces are the autograd derivative of this
same energy, ``F_i = -∂E/∂r_i`` (eV/Å, the repo-wide sign).

It **inherits** :class:`~molzoo.mace.encoder.MACEEncoder` rather than holding
one, mirroring upstream's ``ScaleShiftMACE(MACE)``: holding an encoder would
prefix every ``state_dict`` key with ``encoder.`` and official checkpoints
would stop loading.

Three seams, deliberately coexisting:

``energy_core(positions, Z, edge_index, batch, num_graphs, …) -> (B,)``
    Flat, public, compilable. Raw tensors in, per-graph energy out; no
    ``TensorDict`` and no ``.item()``, so a traced region stays in one graph —
    hence ``num_graphs`` as an argument. It performs **no element validation**:
    that check host-syncs, so whoever drives this seam calls
    :meth:`~molzoo.mace.encoder.MACEEncoder.validate_elements` once beforehand.
    Export (``molix.engine``) drives it directly.

``energy(batch) -> batch``
    The energy on the batch schema — what :class:`molix.autograd.Derivative`
    differentiates. It detaches neither positions nor energy: the op that
    calls it owns the position leaf, and a detach here would sever
    ``F = -∂E/∂r`` before anyone differentiates.

``forward(batch) -> batch``
    The :class:`~molix.autograd.Derivative` built once in ``__init__`` over
    :meth:`energy`, from the ``wrt`` keys given at construction:
    ``MaceMatpes(...)`` writes the energy alone, ``MaceMatpes(...,
    wrt=("atoms.pos",))`` the energy and forces, and ``Derivative(model,
    wrt=...)`` composed outside is the same op. Compile it with
    ``Compiler(...)(model)`` after one eager forward (the element-table check
    below and, on the polarisable variant, the k-space grid sizing are
    host-side work a trace cannot do).

Every branch — conditioning, per-layer vs final readout, what is derived —
is resolved once in ``__init__`` into a bound attribute; ``forward`` is a
single dispatch through the op. There is no ``compute_forces`` flag on any
public signature: what a potential derives is a *construction*.

Units: ``graphs.energy`` eV ``(B,)``, ``atoms.forces`` eV/Å ``(N, 3)``,
positions and ``r_max`` in Å. Nothing here converts units.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
    Batatia et al. "A foundation model for atomistic materials chemistry"
    (MACE-MP-0). https://arxiv.org/abs/2401.00096
    Kaplan et al. "A foundational potential energy surface dataset for
    materials" (MatPES). https://arxiv.org/abs/2503.04070
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
from tensordict import TensorDict

from molix.autograd import Derivative
from molix.F.scatter import scatter_sum_compile_safe as _scatter_sum
from molix.schema import POS_KEY
from molpot.derivation.protocol import write_energy
from molzoo.mace.checkpoint import CheckpointRemap, matpes_keys, omol_keys
from molzoo.mace.encoder import MACEEncoder
from molzoo.mace.geometry import edge_lengths, edge_vectors
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec, MACESpec
from molzoo.mace.utils import read_archive

#: One term of an official irreps string, e.g. ``"128x1o"`` — multiplicity,
#: angular order ``l``, parity. Parity is fixed by ``l`` in every MACE
#: configuration, so only the first two groups are read.
_IRREPS_TERM = re.compile(r"(\d+)x(\d+)([eo])")


def _parse_irreps(text: str, key: str) -> tuple[int, int]:
    """Scalar multiplicity and highest angular order of an official irreps string.

    ``"128x0e+128x1o"`` → ``(128, 1)``; ``"16x0e"`` → ``(16, 0)``. Upstream MACE
    states channel widths this way, so a checkpoint's ``num_features`` /
    ``max_hidden_l`` / ``mlp_dim`` are readable from its config instead of being
    assumed (``run_nve.py``'s former ``build_model``, removed in
    ``mace-subpackage-restructure-07-cleanup``, hard-coded ``128 / 1 / 16``,
    which loads a differently sized checkpoint into the wrong model).

    Args:
        text: Irreps string from the official config.
        key: Config key it came from, for the error message.

    Returns:
        ``(multiplicity of the scalar 0e term, highest l over all terms)``.

    Raises:
        ValueError: If a term is not ``<mul>x<l><parity>``, or the string has
            no scalar term — the channel width would then be a guess.
    """
    terms: list[tuple[int, int]] = []
    for term in str(text).split("+"):
        matched = _IRREPS_TERM.fullmatch(term.strip())
        if matched is None:
            raise ValueError(f"config key {key} is not an irreps string: {text!r} (term {term!r})")
        terms.append((int(matched.group(1)), int(matched.group(2))))
    scalars = [multiplicity for multiplicity, order in terms if order == 0]
    if not scalars:
        raise ValueError(
            f"config key {key} has no scalar (0e) term: {text!r} — the channel "
            "width is its multiplicity and there is no default"
        )
    return scalars[0], max(order for _, order in terms)


class MacePotential(MACEEncoder):
    """MACE foundation-model energy (and forces) on the molnex batch schema.

    Given a batch of atomic systems it writes the per-graph potential energy
    ``graphs.energy`` ``(B,)`` in eV and, when built with
    ``wrt=("atoms.pos",)``, the per-atom forces ``atoms.forces`` ``(N, 3)`` in
    eV/Å; the energy expression it evaluates and the ``F_i = -∂E/∂r_i``
    convention are spelled out in the module docstring. The variant is the
    spec's business; the only two switches here decide *what is derived* and
    *which cuEquivariance path* the blocks take::

        potential = MaceMatpes(atomic_numbers=..., atomic_energies=...)
        potential = MaceOmol(..., wrt=("atoms.pos",))
        potential = MacePolar(..., use_fallback=True)

    A complete, runnable use of this public surface (both variants, free and
    periodic, energy-only and energy+forces, with hard-coded fp64 goldens) lives
    in the mirrored unit tests under ``tests/test_molzoo/test_mace/`` (the
    ``regressions/`` tier was deleted on 2026-09-07).

    Args:
        spec: Validated variant configuration —
            :class:`~molzoo.mace.spec.MACEMatpesSpec` or
            :class:`~molzoo.mace.spec.MACEOMolSpec`.
        wrt: What :meth:`forward` derives beside the energy, as TensorFrame
            keys (:class:`molix.autograd.Derivative`): ``()`` (the default)
            writes ``graphs.energy`` alone, ``("atoms.pos",)`` adds
            ``atoms.forces``, ``("atoms.pos", "graphs.strain")`` adds
            ``graphs.virial`` on a one-graph periodic batch.
        use_fallback: Put the equivariant blocks on the pure-torch
            cuEquivariance path instead of its *fused* kernels — single GPU
            kernels that do a whole tensor product at once, and which need a
            GPU plus the ``cuequivariance-ops-torch`` wheel. The flag only ever
            escalates: ``True`` overrides a spec that asked for fused kernels,
            while the ``False`` default never downgrades a spec that already
            asked for the fallback (a ``bool`` cannot express "not given").
            Leave it alone on GPU — the foundation force path is always
            autograd, so the functorch-compatibility argument for the
            pure-torch blocks never applies here, and the fallback measured
            **35.7x slower per MD step** with no correctness upside
            (``.claude/notes/notes.md`` ``mol:note:topic:cueq-use-fallback``;
            the guard that keeps it honest is
            ``benchmarks/bench_mace_matpes.py``). CPU and test call sites,
            which have no fused kernels available, pass ``True`` explicitly.
        energy_dtype: Accumulation dtype for graph energies, float64 by
            default even with fp32 model weights. ``None`` uses model dtype.
            This does not promote the learned calculation or recover digits
            lost when loading weights into a lower-precision model.
    """

    _keys: ClassVar[dict[str, str | None]] = {}
    _unexpected: ClassVar[Literal["raise", "return"]] = "raise"

    def __init__(
        self,
        spec: MACESpec,
        *,
        wrt: Sequence[str | Sequence[str]] = (),
        use_fallback: bool = False,
        energy_dtype: torch.dtype | None = torch.float64,
    ) -> None:
        super().__init__(spec.model_copy(update={"use_fallback": True}) if use_fallback else spec)
        if energy_dtype not in (None, torch.float32, torch.float64):
            raise ValueError(
                "energy_dtype must be torch.float32, torch.float64 or None (model dtype)"
            )
        self.energy_dtype = energy_dtype
        self._input_dtype: torch.dtype | None = None

        # ---- frozen switches: no pydantic attribute reads on the hot path ----
        self._conditioning = (
            self._condition_charge_spin
            if spec.conditioning == "charge_spin"
            else self._reject_conditioning
        )
        self._readout_energy = (
            self._per_layer_readout_energy
            if spec.readout == "per_layer"
            else self._final_readout_energy
        )
        #: The one derivative route: the op over :meth:`energy`, fixed at
        #: construction. Its ``energy`` is a bound method, so it registers no
        #: submodule and the checkpoint keys stay flat.
        self._derivative = Derivative(self.energy, wrt=wrt)
        #: Element-table gate, spent after the first eager batch (see ``energy``).
        self._elements_validated = False

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        format: str = "official",
        config: str | Path | None = None,
        map_location: str | torch.device = "cpu",
        trusted: bool = False,
        converted_weights: str | Path | None = None,
        **kwargs: Any,
    ) -> MacePotential:
        """Construct from an official archive and load its weights.

        ``format="official"`` is the only format: a cueq ``state_dict``, a
        pickled MACE module, or a directory / json config that sits next to
        the weights. Key dialect translation lives here, not as a public
        ``*_REMAP`` constant.

        Args:
            path: Weights file, pickled ``.model``, config json, or a directory.
            format: Must be ``"official"``.
            config: Optional sidecar json when ``path`` is a plain state dict.
            map_location: Forwarded to ``torch.load``.
            trusted: Explicitly permit pickle loading of verified local modules.
                Tensor state plus JSON config does not need this.
            converted_weights: Official cuEq-converted tensor state; ``path``
                then supplies metadata only, never an intermediate weight load.
            **kwargs: Constructor overrides (``use_fallback``, ``wrt``,
                architecture fields).

        Returns:
            A model of ``cls`` holding the archive's weights.
        """
        if format != "official":
            raise ValueError(f"format must be 'official', got {format!r}")
        weights, official = _resolve_official(Path(path), config)
        archive = read_archive(
            weights, map_location=map_location, trusted=trusted, converted_weights=converted_weights
        )
        merged = {**archive["config"], **official}
        model = cls._from_official(merged, **kwargs)
        numbers = archive["state"].get("atomic_numbers")
        if numbers is not None and not torch.equal(numbers.cpu().reshape(-1), model.z_table.cpu()):
            raise ValueError(
                "checkpoint atomic_numbers disagree with the configuration element table"
            )
        CheckpointRemap(cls._keys, on_unexpected=cls._unexpected).load(model, archive["state"])
        return model

    @classmethod
    def _from_official(cls, official: dict[str, Any], **kwargs: Any) -> MacePotential:
        raise NotImplementedError(f"{cls.__name__} does not implement official construction")

    def configure_input_dtype(self, dtype: torch.dtype | None) -> None:
        """Cast geometry at the model boundary, preserving the caller's MD state.

        Set to the model parameter dtype to run, for example, fp64 MD state
        with an fp32 potential. Casts remain differentiable: forces still
        refer to the original position leaf. ``None`` (default) leaves inputs
        untouched. Configure before compilation; no checkpoint keys change.
        This does not cast parameters or select the process-wide GEMM policy.
        """
        if dtype not in (None, torch.float32, torch.float64):
            raise ValueError("input dtype must be torch.float32, torch.float64 or None")
        self._input_dtype = dtype

    def _evaluate_core(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        *,
        edge_diff: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        total_spin: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One dispatch/precision boundary for flat and TensorDict callers."""
        dtype = self._input_dtype
        if dtype is not None:
            positions = positions.to(dtype=dtype)
            edge_diff = None if edge_diff is None else edge_diff.to(dtype=dtype)
            cell = None if cell is None else cell.to(dtype=dtype)
            total_charge = None if total_charge is None else total_charge.to(dtype=dtype)
            total_spin = None if total_spin is None else total_spin.to(dtype=dtype)
        return self.energy_core(
            positions,
            Z,
            edge_index,
            batch,
            num_graphs,
            edge_diff=edge_diff,
            total_charge=total_charge,
            total_spin=total_spin,
            cell=cell,
            atom_mask=atom_mask,
        )

    def _sum_energy(
        self,
        values: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-graph reduction of a per-atom energy, at the user's accumulation precision.

        The one place a padded batch's ``atom_mask`` enters: every per-atom
        energy (``e0``, the readouts, OMOL's embedding readout, Polar's local
        electron energy) is multiplied by it **before** the graph sum, so a
        ghost atom adds exactly zero to whichever graph it is assigned to. A
        multiply, never a slice: the traced shape does not depend on the mask.

        Args:
            values: Per-atom energies ``(N,)`` in eV.
            batch: Graph index per atom ``(N,)``.
            num_graphs: Number of graphs ``B``.
            atom_mask: Optional ``(N,)`` bool / float, ``True`` for real atoms.

        Returns:
            Per-graph energies ``(B,)`` in eV.
        """
        if atom_mask is not None:
            values = values * atom_mask.to(dtype=values.dtype)
        if self.energy_dtype is not None:
            values = values.to(dtype=self.energy_dtype)
        return _scatter_sum(values, batch, num_graphs)

    def forward(self, batch: TensorDict) -> TensorDict:
        """Run the construction-fixed op, mutating ``batch`` in place.

        Reads ``atoms.{Z,pos,batch}``, ``edges.edge_index`` ``(E, 2)`` (``[:,0]``
        source, ``[:,1]`` target — never transposed on the way in),
        ``edges.edge_diff`` ``(E, 3)`` and, for the OMOL variant, optional
        ``graphs.{total_charge,total_spin}``. Writes ``graphs.energy`` ``(B,)``
        in eV and, per the ``wrt`` keys of construction, ``atoms.forces``
        ``(N, 3)`` in eV/Å and ``graphs.virial`` ``(1, 3, 3)`` in eV. Under
        ``torch.no_grad()`` every output is detached; with grad enabled they
        stay attached to the parameters, so an energy or force loss trains.

        Args:
            batch: Post-collate ``TensorDict``.

        Returns:
            The same ``batch`` object, with the outputs written in place.

        Raises:
            ValueError: If the batch contains an atomic number outside this
                model's element table (checked once per instance), or if it
                carries ``graphs.total_charge`` / ``graphs.total_spin`` for a
                variant that has no charge/spin conditioning.
        """
        return self._derivative(batch)

    @property
    def derivative(self) -> Derivative:
        """The op :meth:`forward` runs: ``Derivative(self.energy, wrt=...)``, fixed in ``__init__``.

        What a training step marks its leaves through
        (:meth:`molix.autograd.Derivative.mark_leaves`).
        """
        return self._derivative

    # --------------------------------------------------------- compile seam
    def bind_cell(self, cell: torch.Tensor) -> None:
        """Take the live simulation cell — a no-op on this class.

        Satisfies :class:`molix.schema.CellBound` so the MD driver can hand
        every potential the lattice through one call without asking what kind
        of potential it is. Non-polarisable variants (MatPES, OMol) have no
        reciprocal-space grid to size, so there is nothing to bind: the local
        message-passing energy reads the already-imaged ``edge_diff`` and never
        looks at the lattice itself. :meth:`molzoo.mace.polar.MacePolar.bind_cell`
        overrides this and forwards to its two GTO k-space grids.

        Args:
            cell: Lattice vectors ``(3, 3)`` — or ``(1, 3, 3)`` — in Å, rows
                being the lattice vectors. Ignored here.
        """
        del cell

    # ------------------------------------------------------------ energy core
    def energy_core(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        *,
        edge_diff: torch.Tensor | None = None,
        shifts: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        total_spin: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-graph total energy ``(B,)`` as a pure function of ``positions``.

        Evaluates the ``E_g`` of the module docstring: isolated-atom reference
        energies, plus the scale/shift-normalised sum of the readouts (and the
        ZBL pair term, where the variant has one), summed over the atoms of
        each graph.

        This is the flat compile seam: raw tensors in, energy out, no
        ``TensorDict`` and no host sync. Every quantity that depends on the
        positions is recomputed here — the edge displacement vectors, their
        lengths, the angular (spherical-harmonic) features of their directions
        and the radial basis expansion of their lengths — so the result is a
        differentiable function of ``positions`` (``torch.autograd.grad`` gives
        the forces) and traces into a single TorchDynamo graph.

        **The seam performs no element validation.** ``Z`` is taken as already
        inside the element table; an atomic number outside it is snapped onto a
        neighbouring row here and yields a plausible, wrong energy. The check
        costs a host sync and so cannot live in a traced region, which makes it
        the *seam caller's* duty: whoever drives this method directly —
        ``torch.compile``, ``torch.export``,
        ``molix.engine.export_for_lammps`` — calls
        :meth:`~molzoo.mace.encoder.MACEEncoder.validate_elements` once first.
        On the batch path :meth:`energy` is that caller and does it on its
        first *eager* call.

        The whole op — :meth:`forward` unpacking, leaf, this core and the
        ``autograd.grad`` — compiles as one region under
        :class:`molix.compile.Compiler` once one eager forward has spent the
        element gate; this seam alone is what the exporters drive.

        Args:
            positions: Atom positions ``(N, 3)`` in Å.
            Z: Atomic numbers ``(N,)``; must lie inside the element table (see
                :meth:`~molzoo.mace.encoder.MACEEncoder.validate_elements`).
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target.
            batch: Graph index per atom ``(N,)`` — ``batch[i]`` says which of
                the ``B`` graphs atom ``i`` belongs to.
            num_graphs: Number of graphs ``B``. An argument, not
                ``int(batch.max().item()) + 1``: that host sync would break the
                traced graph and stall the MD hot path.
            edge_diff: Optional already-imaged displacements ``(E, 3)`` in Å.
            shifts: Rejected. Raises ``TypeError``.
            total_charge: Per-graph total charge ``(B,)`` in units of the
                elementary charge ``e`` (OMOL only; defaults to neutral). Used
                as an integer index into an embedding table, offset by the
                spec's ``charge_offset``.
            total_spin: Per-graph total spin ``(B,)``, dimensionless (OMOL
                only; defaults to ``1``, the closed-shell singlet — every
                electron paired — so the values follow the spin-multiplicity
                ``2S+1`` convention, not a count of unpaired electrons). Also
                an embedding-table index, offset by ``spin_offset``.
            cell: Ignored on this base implementation — local MACE (MatPES /
                OMol) has no explicit lattice; the already-imaged ``edge_diff``
                already carries every periodic image it needs.
                :meth:`~molzoo.mace.polar.MacePolar.energy_core` overrides
                this method and uses ``cell`` to select its periodic
                reciprocal-space path.
            atom_mask: Optional ``(N,)`` boolean, ``True`` for real atoms — what
                a padded batch (:class:`molix.data.MoleculePadder`) carries
                as ``atoms.mask``. Every per-atom energy is multiplied by it
                before the graph sum (:meth:`_sum_energy`), so ghost atoms add
                nothing; padding edges are dead by their geometry. ``None``
                means every atom is real.

        Returns:
            Total energy per graph ``(B,)`` in eV.

        Raises:
            TypeError: If ``shifts`` is given — it is no longer a PBC channel;
                pass the already-imaged ``edge_diff`` instead.
            ValueError: If a variant without charge/spin conditioning is handed
                ``total_charge`` or ``total_spin`` — silently ignoring them
                would return a plausible, wrong energy.
        """
        if shifts is not None:
            raise TypeError("energy_core no longer takes shifts; pass edge_diff=")
        del cell  # local MACE has no explicit cell; Polar overrides this path.
        vectors = edge_vectors(positions, edge_index, edge_diff=edge_diff)
        lengths = edge_lengths(vectors)

        node_attrs = self.node_attrs(Z, positions.dtype)
        edge_feats, cutoff = self.radial_features(lengths, Z, edge_index)
        edge_attrs = self.angular_features(vectors)

        node_feats = self.initial_node_features(node_attrs)
        e0 = self._sum_energy(self.atomic_energies(Z), batch, num_graphs, atom_mask)
        node_feats, e0 = self._conditioning(
            node_feats, e0, batch, num_graphs, total_charge, total_spin, atom_mask
        )

        per_layer = self.layer_features(
            node_feats=node_feats,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=cutoff,
        )
        # The short-range pair term is a per-atom energy on the same footing as
        # the readouts, and it is summed first (MatPES operator order).
        node_es = (
            self.pair_repulsion(lengths, Z, edge_index)
            if self.pair_repulsion is not None
            else node_feats.new_zeros(node_feats.shape[0])
        )
        node_es = self._readout_energy(per_layer, node_es)
        return e0 + self._sum_energy(self.scale_shift(node_es), batch, num_graphs, atom_mask)

    # -------------------------------------------------------------- variants
    def _reject_conditioning(
        self,
        node_feats: torch.Tensor,
        e0: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        total_charge: torch.Tensor | None,
        total_spin: torch.Tensor | None,
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass the node state through — this variant has no charge/spin input.

        The ``None`` comparisons are constant-folded by dynamo, so the guard
        costs no graph break on the compiled path.

        Args:
            node_feats: Initial node features ``(N, num_features)``.
            e0: Per-graph reference energy ``(B,)`` in eV.
            batch: Graph index per atom ``(N,)`` (unused).
            num_graphs: Number of graphs ``B`` (unused).
            total_charge: Must be ``None``.
            total_spin: Must be ``None``.
            atom_mask: Unused — nothing per-atom is added here.

        Returns:
            ``(node_feats, e0)`` unchanged.

        Raises:
            ValueError: If either conditioning tensor was supplied.
        """
        if total_charge is not None or total_spin is not None:
            raise ValueError(
                "this MACE variant has no charge/spin conditioning "
                '(conditioning="none"): total_charge / total_spin would be '
                "ignored and the energy would be quietly wrong — build the "
                "potential from a MACEOMolSpec instead"
            )
        return node_feats, e0

    def _condition_charge_spin(
        self,
        node_feats: torch.Tensor,
        e0: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        total_charge: torch.Tensor | None,
        total_spin: torch.Tensor | None,
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the OMOL charge/spin embedding and its scalar readout to ``e0``.

        Args:
            node_feats: Initial node features ``(N, num_features)``.
            e0: Per-graph reference energy ``(B,)`` in eV.
            batch: Graph index per atom ``(N,)``.
            num_graphs: Number of graphs ``B``.
            total_charge: Per-graph total charge ``(B,)`` in units of ``e``;
                ``None`` → neutral.
            total_spin: Per-graph total spin ``(B,)``, dimensionless; ``None``
                → ``1``, the closed-shell singlet (all electrons paired, i.e.
                multiplicity ``2S+1 = 1``). Spin ``0`` would index an untrained
                embedding row and return garbage
                (``src/molzoo/mace_omol.py:330-335`` at 0e05959, before
                deletion).
            atom_mask: Optional ``(N,)`` real-atom mask; the per-atom embedding
                readout is multiplied by it before the graph sum.

        Returns:
            ``(conditioned node features (N, num_features), e0 (B,) eV)``.
        """
        if total_charge is None:
            total_charge = torch.zeros(num_graphs, dtype=torch.long, device=node_feats.device)
        if total_spin is None:
            total_spin = torch.ones(num_graphs, dtype=torch.long, device=node_feats.device)
        node_feats = node_feats + self.conditioning(
            batch, total_spin=total_spin, total_charge=total_charge
        )
        embedded = self.embedding_readout(node_feats).squeeze(-1)
        return node_feats, e0 + self._sum_energy(embedded, batch, num_graphs, atom_mask)

    def _per_layer_readout_energy(
        self, per_layer: list[torch.Tensor], node_es: torch.Tensor
    ) -> torch.Tensor:
        """Sum one readout per interaction layer (MACE-MP / MatPES).

        Args:
            per_layer: Node features of every layer, from
                :meth:`~molzoo.mace.encoder.MACEEncoder.layer_features`.
            node_es: Per-atom energy accumulated so far ``(N,)`` in eV.

        Returns:
            Per-atom interaction energy ``(N,)`` in eV, before scale/shift.
        """
        for readout, node_feats in zip(self.readouts, per_layer, strict=True):
            node_es = node_es + readout(node_feats).squeeze(-1)
        return node_es

    def _final_readout_energy(
        self, per_layer: list[torch.Tensor], node_es: torch.Tensor
    ) -> torch.Tensor:
        """Read out the last layer only (OMOL).

        Args:
            per_layer: Node features of every layer, from
                :meth:`~molzoo.mace.encoder.MACEEncoder.layer_features`.
            node_es: Per-atom energy accumulated so far ``(N,)`` in eV.

        Returns:
            Per-atom interaction energy ``(N,)`` in eV, before scale/shift.
        """
        return node_es + self.readout(per_layer[-1]).squeeze(-1)

    # ---------------------------------------------------------------- energy
    def energy(self, batch: TensorDict) -> TensorDict:
        """The energy on the batch schema — what :class:`~molix.autograd.Derivative` differentiates.

        Also the whole of :meth:`forward` on an instance built with ``wrt=()``.
        Deliberately detaches **nothing**: the op that calls it (this
        potential's own, or one composed outside) owns the position leaf this
        energy hangs off, and detaching would cut ``F = -∂E/∂r``.

        The ``graphs`` namespace is created here — ``TensorDict({},
        batch_size=[num_graphs])`` — *before* ``protocol.write_energy`` runs, so
        the post-collate schema (``graphs`` is ``batch_size=[B]``, see
        CLAUDE.md) holds from this ``B``, the one this method already resolved.
        It used to be load-bearing: ``protocol.ensure_graphs`` built
        ``batch_size=[]`` and silently degraded the schema until
        ``mace-subpackage-restructure-07-cleanup`` taught it ``num_graphs`` and
        had ``write_energy`` pass ``energy.shape[0]``. The two now agree, and
        this line stays as the explicit statement of the shape.

        Args:
            batch: Post-collate ``TensorDict`` carrying ``atoms.{Z,pos,batch}``
                and ``edges.edge_index`` ``(E, 2)``. Optional and consumed when
                present: ``edges.shifts`` ``(E, 3)`` in Å (periodic images),
                ``atoms.mask`` ``(N,)`` (a padded batch's real-atom mask, passed
                to :meth:`energy_core` as ``atom_mask``) and, for the OMOL
                variant, ``graphs.{total_charge,total_spin}``
                ``(B,)``. A ``graphs`` namespace that is already there supplies
                ``B`` from its ``batch_size``; without one, ``B`` costs the one
                host sync on this path — hand the op a batch with a ``graphs``
                namespace, as every collate produces, before compiling.

        Returns:
            The same ``batch``, with ``graphs.energy`` ``(B,)`` in eV written.

        Raises:
            ValueError: On the first **eager** call, if any atomic number lies
                outside this model's element table — a traced call never
                validates, since the check host-syncs, which is why one eager
                forward precedes any compile; on every call, if a variant without
                charge/spin conditioning was handed ``graphs.total_charge`` or
                ``graphs.total_spin``; on every call, if ``edges.edge_diff`` is
                absent from ``batch`` (``edges.shifts`` alone is no longer a
                PBC channel — pass the already-imaged displacement instead).
        """
        Z = batch["atoms", "Z"]
        positions = batch[POS_KEY]
        atom_batch = batch["atoms", "batch"]
        edge_index = batch["pairs", "pair_index"]  # (E, 2) — the core's own convention

        # Once per instance: Z is constant over a trajectory and a wrongly
        # wired model/dataset pair fails on the very first batch, while a
        # per-call check costs a host sync and a dynamo graph break.
        # torch.compiler.is_compiling() is a tracing-time constant: dynamo folds
        # this branch away, so the host-syncing element check never enters the
        # graph.
        if not self._elements_validated and not torch.compiler.is_compiling():
            self.validate_elements(Z)
            self._elements_validated = True

        nested = batch.keys(include_nested=True)
        has_graphs = "graphs" in batch.keys()
        if has_graphs and batch["graphs"].batch_size:
            num_graphs = int(batch["graphs"].batch_size[0])
        else:
            # The only host sync on this path; ``forward`` is not the compile
            # target — ``energy_core`` is, and it takes ``num_graphs`` directly.
            num_graphs = int(atom_batch.max().item()) + 1

        if ("pairs", "pair_diff") not in nested:
            raise ValueError(
                "MacePotential.energy requires edges.edge_diff; "
                "edges.shifts is no longer a PBC channel"
            )
        cell = None
        if ("graphs", "cell") in nested:
            cell = batch["graphs", "cell"]
        energy = self._evaluate_core(
            positions,
            Z,
            edge_index,
            atom_batch,
            num_graphs,
            edge_diff=batch["pairs", "pair_diff"],
            total_charge=(
                batch["graphs", "total_charge"] if ("graphs", "total_charge") in nested else None
            ),
            total_spin=(
                batch["graphs", "total_spin"] if ("graphs", "total_spin") in nested else None
            ),
            cell=cell,
            atom_mask=batch["atoms", "mask"] if ("atoms", "mask") in nested else None,
        )

        if not has_graphs:
            batch["graphs"] = TensorDict({}, batch_size=[num_graphs])
        write_energy(batch, energy)
        return batch


def _energy_table(atomic_energies: Sequence[float] | torch.Tensor) -> list[float]:
    return [float(energy) for energy in atomic_energies]


def _resolve_official(path: Path, config: str | Path | None) -> tuple[Path, dict[str, Any]]:
    official: dict[str, Any] = {}
    if config is not None:
        official = json.loads(Path(config).read_text())
    if path.is_dir():
        jsons = sorted(path.glob("*config.json")) if config is None else []
        weights = sorted(path.glob("*cueq_state.pt")) or sorted(path.glob("*.pt"))
        if len(jsons) > 1 or len(weights) > 1:
            raise ValueError(f"ambiguous official archive under {path}; specify weights and config")
        if jsons:
            official = json.loads(jsons[0].read_text())
        if not weights:
            raise FileNotFoundError(f"no official weights under {path}")
        return weights[0], official
    if path.suffix == ".json":
        official = json.loads(path.read_text())
        stem = path.stem.replace("_config", "")
        for candidate in (path.with_name(stem + "_cueq_state.pt"), path.with_suffix(".pt")):
            if candidate.exists():
                return candidate, official
        raise FileNotFoundError(f"no weights next to {path}")
    return path, official


#: Fields every variant reads from an official config, with the upstream
#: spellings seen for each. The first alias present wins; a field the config
#: does not carry is dropped rather than passed as ``None``, so the spec's own
#: default stands.
_OFFICIAL_ALIASES: dict[str, tuple[str, ...]] = {
    "r_max": ("r_max",),
    "num_bessel": ("num_bessel",),
    "num_interactions": ("num_interactions",),
    "correlation": ("correlation",),
    "l_max": ("max_ell", "l_max"),
    "scale": ("atomic_inter_scale", "scale"),
    "shift": ("atomic_inter_shift", "shift"),
}


def official_ctor(official: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Constructor arguments shared by every MACE variant, plus the caller's.

    Args:
        official: The archive's config mapping.
        extra: Variant-specific arguments; ``None`` values are dropped, the
            same way an absent common field is.

    Returns:
        Keyword arguments for the product class. Caller overrides are applied
        by the caller *after* this, so an explicit ``None`` from a user (say
        ``energy_dtype=None``) still reaches the constructor.
    """
    ctor: dict[str, Any] = {
        "atomic_numbers": official["atomic_numbers"],
        "atomic_energies": official["atomic_energies"],
    }
    for name, aliases in _OFFICIAL_ALIASES.items():
        value = next((official[a] for a in aliases if official.get(a) is not None), None)
        if value is not None:
            ctor[name] = value
    ctor.update({key: value for key, value in extra.items() if value is not None})
    return ctor


def _require_irreps(official: dict[str, Any]) -> tuple[int, int, int]:
    absent = [key for key in ("hidden_irreps", "MLP_irreps") if key not in official]
    if absent:
        raise KeyError(
            f"official config is missing {absent}: "
            "num_features / max_hidden_l / mlp_dim are derived from the irreps "
            "strings and have no default"
        )
    num_features, max_hidden_l = _parse_irreps(official["hidden_irreps"], "hidden_irreps")
    mlp_dim, _ = _parse_irreps(official["MLP_irreps"], "MLP_irreps")
    return num_features, max_hidden_l, mlp_dim


class MaceMatpes(MacePotential):
    """MACE-MatPES. Flat constructor; spec is assembled and validated inside."""

    _keys = matpes_keys
    _unexpected = "raise"

    def __init__(
        self,
        *,
        atomic_numbers: Sequence[int],
        atomic_energies: Sequence[float] | torch.Tensor,
        r_max: float = 6.0,
        num_bessel: int = 10,
        num_polynomial_cutoff: int = 5,
        l_max: int = 3,
        num_features: int = 128,
        max_hidden_l: int = 1,
        num_interactions: int = 2,
        correlation: int = 3,
        mlp_dim: int = 16,
        radial_mlp: Sequence[int] = (64, 64, 64),
        scale: float = 1.0,
        shift: float = 0.0,
        wrt: Sequence[str | Sequence[str]] = (),
        use_fallback: bool = False,
        energy_dtype: torch.dtype | None = torch.float64,
    ) -> None:
        spec = MACEMatpesSpec(
            atomic_numbers=list(atomic_numbers),
            atomic_energies=_energy_table(atomic_energies),
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            l_max=l_max,
            num_features=num_features,
            max_hidden_l=max_hidden_l,
            num_interactions=num_interactions,
            correlation=correlation,
            mlp_dim=mlp_dim,
            radial_mlp=list(radial_mlp),
            scale=scale,
            shift=shift,
            use_fallback=use_fallback,
        )
        super().__init__(
            spec,
            wrt=wrt,
            use_fallback=use_fallback,
            energy_dtype=energy_dtype,
        )

    @classmethod
    def _from_official(cls, official: dict[str, Any], **kwargs: Any) -> MaceMatpes:
        if "pair_repulsion" in official and not official["pair_repulsion"]:
            raise ValueError(
                f"official config sets pair_repulsion={official['pair_repulsion']!r}, "
                "but MACEMatpesSpec hard-wires pair_repulsion='zbl' and the ZBL "
                "constants are buffers, so the load would not object to the surplus "
                "term"
            )
        if (
            "distance_transform" in official
            and str(official["distance_transform"]).lower() != "agnesi"
        ):
            raise ValueError(
                f"official config sets distance_transform="
                f"{official['distance_transform']!r}, but MACEMatpesSpec hard-wires "
                "distance_transform='agnesi' and the Agnesi constants are buffers, "
                "so the load would not object to the surplus transform"
            )
        num_features, max_hidden_l, mlp_dim = _require_irreps(official)
        ctor = official_ctor(
            official,
            num_features=num_features,
            max_hidden_l=max_hidden_l,
            mlp_dim=mlp_dim,
            num_polynomial_cutoff=official.get("num_polynomial_cutoff"),
            radial_mlp=official.get("radial_MLP", official.get("radial_mlp")),
        )
        ctor.update(kwargs)
        return cls(**ctor)


class MaceOmol(MacePotential):
    """MACE-OMol. Flat constructor; spec is assembled and validated inside."""

    _keys = omol_keys
    _unexpected = "return"

    def __init__(
        self,
        *,
        atomic_numbers: Sequence[int],
        atomic_energies: Sequence[float] | torch.Tensor,
        r_max: float = 6.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        l_max: int = 3,
        num_features: int = 1024,
        num_interactions: int = 3,
        correlation: int = 2,
        mlp_dim: int = 16,
        edge_channels: int = 128,
        charge_classes: int = 201,
        charge_offset: int = 100,
        spin_classes: int = 101,
        spin_offset: int = 0,
        scale: float = 1.0,
        shift: float = 0.0,
        wrt: Sequence[str | Sequence[str]] = (),
        use_fallback: bool = False,
        energy_dtype: torch.dtype | None = torch.float64,
    ) -> None:
        spec = MACEOMolSpec(
            atomic_numbers=list(atomic_numbers),
            atomic_energies=_energy_table(atomic_energies),
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
            use_fallback=use_fallback,
        )
        super().__init__(
            spec,
            wrt=wrt,
            use_fallback=use_fallback,
            energy_dtype=energy_dtype,
        )

    @classmethod
    def _from_official(cls, official: dict[str, Any], **kwargs: Any) -> MaceOmol:
        ctor = official_ctor(
            official,
            num_features=official.get("num_features"),
            mlp_dim=official.get("mlp_dim"),
            edge_channels=official.get("edge_channels"),
        )
        ctor.update(kwargs)
        return cls(**ctor)
