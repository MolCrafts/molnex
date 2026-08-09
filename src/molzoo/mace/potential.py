"""Unified MACE energy/force potential for the foundation-model variants.

An *interatomic potential* is a function from an arrangement of atoms — their
chemical element numbers ``Z_i`` and Cartesian positions ``r_i``, in ångström
(Å) — to the potential energy ``E`` of that arrangement, in electronvolt (eV),
and from there to the force on each atom, which is the negative gradient of
that energy::

    F_i = -∂E/∂r_i          [eV/Å]

That sign is the repo-wide convention (``molix.schema.FORCES_KEY``). MACE
*learns* such a function: it describes each atom's neighbourhood by the
displacement vectors to the neighbours inside a cutoff radius ``r_max`` and
turns that description into a per-atom energy through several rounds of
*message passing* — each round lets an atom mix in features of its neighbours.
A *foundation* variant is one whose weights were fitted once on a large,
chemically broad dataset and are then used unchanged; molnex ships two, MatPES
(periodic materials, arXiv:2503.04070) and OMol (molecules, additionally
conditioned on the total charge and total spin of the system).

Per *graph* ``g`` — one graph is one molecule, or one periodic cell, inside a
batch of ``B`` of them — this class evaluates exactly::

    E_g = Σ_{i∈g} E0(Z_i)  +  Σ_{i∈g} (scale · ε_i + shift)     [eV]
    ε_i = ZBL_i + Σ_layer readout_layer(h_i^layer)              [eV]

``E0(Z)`` is the frozen isolated-atom reference energy of element ``Z``,
``h_i^layer`` is atom ``i``'s feature vector after each message-passing layer,
``ZBL_i`` is the short-range Ziegler–Biersack–Littmark nuclear repulsion —
present only when the spec asks for it (``pair_repulsion="zbl"``, the MatPES
default; identically zero otherwise) — and ``scale`` / ``shift`` are the fitted
affine normalisation of the learned part, applied *per atom*
(:class:`molpot.heads.rescale.GlobalRescale`). MatPES reads out every layer, so
the inner sum runs over all of them; OMol reads out the last layer only, so its
inner sum has a single term, and it adds one further per-atom contribution from
its charge/spin embedding next to ``E0`` — that is, *outside* the
``scale``/``shift`` normalisation. The forces are never hand-coded: they are
the autograd derivative of this same ``E_g``.

:class:`MACEPotential` is the one energy + force host for every MACE
foundation variant. Which variant it is comes entirely from the
:class:`~molzoo.mace.spec.MACESpec` it is built from — the MatPES density
stack (arXiv:2503.04070) or the OMOL charge/spin-conditioned residual stack —
so the two pre-cutover flat modules that each duplicated this forward
(deleted in the restructure's wiring step, 06-wire; see git history) collapse
into this single class.

It **inherits** :class:`~molzoo.mace.encoder.MACEEncoder` rather than holding
one, mirroring upstream's ``ScaleShiftMACE(MACE)``: holding an encoder would
prefix every key of the ``state_dict`` (PyTorch's flat ``name -> tensor`` dump
of a model's weights) with ``encoder.``, and an official checkpoint would stop
loading without a key rewrite.

Two seams, deliberately coexisting:

``energy_core(positions, Z, edge_index, batch, num_graphs, …) -> (B,)``
    The **flat, public, compilable** one. Raw tensors in, per-graph energy in
    eV out; no ``TensorDict`` (the nested batch container of the post-collate
    schema, see CLAUDE.md) and no ``.item()`` *host sync* — reading a tensor
    value back into Python makes the CPU wait for the GPU and forces
    TorchDynamo, the tracer behind ``torch.compile``, to cut the traced region
    in two (a *graph break*, which costs back the launch overhead compilation
    was meant to remove). ``num_graphs`` is an argument for exactly that
    reason: one traced graph, no break. This is the entry a molecular-dynamics
    (MD) driver or an engine export binds to.

``_write_energy(batch) -> batch``
    The **protocol hook**, one level up, on the batch schema.
    ``molpot.derivation.protocol.call_energy`` prefers ``model._write_energy``,
    so implementing it wires MACE into ``EnergyReadout`` / ``ForceReadout`` for
    free. It reads the tensors off the post-collate batch, calls
    :meth:`MACEPotential.energy_core`, and writes the result back through
    ``protocol.write_energy`` — **without** detaching either the positions or
    the energy. Detaching a tensor cuts it out of the autograd graph; an outer
    readout session may have created the position *leaf* (the tensor autograd
    differentiates with respect to) itself, so a detach here would sever
    ``F = -∂E/∂r`` before anyone takes the derivative. (The flat models
    detached inside their ``energy_forces(compute_forces=False)``; that duty
    belongs to :class:`molix.md.forcefield.PotentialForceField`, not to the
    potential.)

Every branch — forces on/off, charge/spin conditioning, per-layer vs final
readout — is resolved once in ``__init__`` into a bound attribute, and
``forward`` is a single dispatch through ``self._pipeline``. There is no
``compute_forces`` flag on any public signature: "energy only" is a
*construction* (``MACEPotential(spec, compute_forces=False)``), which is also
what ``molix.md.forcefield.PotentialForceField.calc_energy`` already assumes.

Units: ``graphs.energy`` in eV ``(B,)``, ``atoms.forces`` in eV/Å ``(N, 3)``,
positions and ``r_max`` in Å. The bridge to the MD integrator's unit system
(mass in atomic mass units, length in Å, time in fs, hence energy in
amu·Å²/fs²) lives in ``molix.md.forcefield``; nothing here converts units.

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
from pathlib import Path

import torch
from tensordict import TensorDict

from molix.F.scatter import scatter_sum_compile_safe as _scatter_sum
from molix.schema import POS_KEY
from molpot.derivation.kernels import grad_force_pass
from molpot.derivation.protocol import write_energy
from molzoo.mace.checkpoint import MATPES_REMAP, CheckpointRemap
from molzoo.mace.encoder import MACEEncoder
from molzoo.mace.geometry import edge_lengths, edge_vectors
from molzoo.mace.spec import MACEMatpesSpec, MACEOMolSpec

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


class MACEPotential(MACEEncoder):
    """MACE foundation-model energy (and forces) on the molnex batch schema.

    Given a batch of atomic systems it writes the per-graph potential energy
    ``graphs.energy`` ``(B,)`` in eV and, unless built energy-only, the per-atom
    forces ``atoms.forces`` ``(N, 3)`` in eV/Å; the energy expression it
    evaluates and the ``F_i = -∂E/∂r_i`` convention are spelled out in the
    module docstring. The variant is the spec's business; the only two switches
    here decide *how much* is computed and *which cuEquivariance path* the
    blocks take::

        potential = MACEPotential(MACEMatpesSpec(...))          # E + F
        potential = MACEPotential(spec, compute_forces=False)   # E only
        potential = MACEPotential(spec, use_fallback=True)      # CPU / no ops wheel

    A complete, runnable use of this public surface (both variants, free and
    periodic, energy-only and energy+forces, with hard-coded fp64 goldens) is
    ``regressions/mace-subpackage-restructure-04-potential.py``.

    Args:
        spec: Validated variant configuration —
            :class:`~molzoo.mace.spec.MACEMatpesSpec` or
            :class:`~molzoo.mace.spec.MACEOMolSpec`.
        compute_forces: Build the energy **and** force pipeline. Defaults to
            ``True``: both foundation checkpoints are used for MD and
            evaluation, where the force is the point.
            (:class:`~molzoo.pinet.potential.PiNetPotential` defaults to
            ``False`` because its main road is energy-only training — the
            disagreement is deliberate, not an oversight.)
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
    """

    def __init__(
        self,
        spec: MACEMatpesSpec | MACEOMolSpec,
        *,
        compute_forces: bool = True,
        use_fallback: bool = False,
    ) -> None:
        super().__init__(spec.model_copy(update={"use_fallback": True}) if use_fallback else spec)

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
        #: One bound call path; ``forward`` never branches (cf. PiNetPotential).
        self._pipeline = self._pipeline_ef if compute_forces else self._write_energy
        #: Element-table gate, spent after the first batch (see ``_write_energy``).
        self._elements_validated = False

    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | Path,
        weights_path: str | Path,
        *,
        remap: CheckpointRemap = MATPES_REMAP,
        map_location: str | torch.device = "cpu",
        **ctor_kwargs,
    ) -> MACEPotential:
        """Build a MatPES potential from an official config json + cueq weights.

        Three steps, nothing hidden: read the config and translate it into a
        :class:`~molzoo.mace.spec.MACEMatpesSpec`, construct the model, load the
        weights through ``remap``. It replaced the hand-written construction in
        ``run_nve.py``'s ``build_model`` (removed in
        ``mace-subpackage-restructure-07-cleanup``, which re-pointed the script
        here)::

            potential = MACEPotential.from_checkpoint(
                weights_dir / "matpes_r2scan_config.json",
                weights_dir / "matpes_r2scan_cueq_state.pt",
            )
            potential.eval()   # the caller's step, deliberately not done here

        A runnable end-to-end use — a synthetic official-dialect checkpoint
        written to a temporary directory, read back through this classmethod,
        and checked against hard-coded energy / force goldens — is
        ``regressions/mace-subpackage-restructure-05-checkpoint.py``.

        ``num_features`` / ``max_hidden_l`` / ``mlp_dim`` are **derived** from
        the config's ``hidden_irreps`` / ``MLP_irreps`` (see
        :func:`_parse_irreps`); a missing key raises instead of falling back to
        the shipped ``128 / 1 / 16``, which would load a differently sized
        checkpoint into the wrong model. The config keys translate as
        ``max_ell → l_max``, ``radial_MLP → radial_mlp``, ``atomic_inter_scale
        → scale``, ``atomic_inter_shift → shift``; ``r_max``, ``num_bessel``,
        ``num_polynomial_cutoff``, ``num_interactions``, ``correlation``,
        ``atomic_numbers`` and ``atomic_energies`` keep their names. Nothing on
        this path converts units: ``r_max`` is read in Å, ``atomic_energies`` in
        eV/atom, ``atomic_inter_shift`` in eV, and the checkpoint's weights are
        already in that same system (see :mod:`molzoo.mace.checkpoint`).

        **Validated when present: the two architecture switches the config
        spells.** :class:`~molzoo.mace.spec.MACEMatpesSpec` hard-wires
        ``pair_repulsion="zbl"`` and ``distance_transform="agnesi"``, so a
        config that names either key with a contradicting value is refused here,
        before the model is built. It has to be refused at *this* seam because
        the load cannot see it: the fitted constants of both blocks are
        registered as buffers, not ``nn.Parameter`` (``ZBLRepulsion`` and
        ``AgnesiTransform`` are both built with ``trainable=False``), so the
        unfilled-parameter guard of
        :meth:`~molzoo.mace.checkpoint.CheckpointRemap.load` never fires — the
        surplus term would keep its default constants and the model would run,
        look sane, and compute a short-range repulsion (or transform every
        distance) the checkpoint was never fitted with. ``distance_transform``
        is compared case-insensitively: the stock dump spells it ``"Agnesi"``.

        **Absence is the documented boundary.** A config naming neither key is
        accepted on the spec's defaults — older dumps predate the two entries,
        and ZBL + Agnesi is what a stock MatPES checkpoint was fitted with
        anyway. The remaining variant flags (density interactions, per-layer
        readout, conditioning) are likewise taken from the spec and compared
        against nothing. For anything but a stock MatPES checkpoint, build the
        spec explicitly and call ``MATPES_REMAP.load`` yourself.

        The ``(missing_buffers, unexpected)`` report of
        :meth:`~molzoo.mace.checkpoint.CheckpointRemap.load` is dropped here:
        under the default policy ``unexpected`` is empty by construction (an
        unhoused key raises instead), and ``missing_buffers`` holds only entries
        cuEquivariance rebuilds. Pass a ``remap`` with ``on_unexpected="return"``
        and that list is lost — load by hand if you need to inspect it.

        This constructor covers the **MatPES** family only. An OMol checkpoint
        is loaded by constructing a :class:`~molzoo.mace.spec.MACEOMolSpec`
        explicitly and calling
        ``molzoo.mace.checkpoint.OMOL_REMAP.load(potential, state)`` — its
        config carries charge/spin fields with no counterpart here.

        Args:
            config_path: Official config json (the ``model.config`` dumped
                beside the converted weights).
            weights_path: ``state_dict`` of the cueq-converted official model,
                saved by ``torch.save`` and read back with
                ``weights_only=True``.
            remap: Key-dialect translation and unexpected-key policy. Defaults
                to :data:`~molzoo.mace.checkpoint.MATPES_REMAP`, which refuses
                a checkpoint key with no home.
            map_location: Device the weights are read onto, forwarded to
                ``torch.load``. Defaults to CPU: the model is built on the
                ambient device and moved by the caller.
            **ctor_kwargs: Passed straight to ``cls`` — ``compute_forces`` /
                ``use_fallback`` are runtime-environment switches, not
                scientific content of the checkpoint, so they stay out of the
                config translation.

        Returns:
            A ``MACEPotential`` holding the checkpoint's weights, in training
            mode (call ``.eval()`` yourself).

        Raises:
            KeyError: If the config lacks a key the spec needs — including
                ``hidden_irreps`` / ``MLP_irreps``, which are named explicitly
                because no width default is acceptable.
            ValueError: If the config contradicts one of the two hard-wired
                architecture switches, if an irreps string cannot be parsed, or
                if the spec rejects the translated configuration.
            RuntimeError: From ``remap`` — an unhoused checkpoint key, a shape
                disagreement, or an unfilled parameter.
        """
        official = json.loads(Path(config_path).read_text())
        absent = [key for key in ("hidden_irreps", "MLP_irreps") if key not in official]
        if absent:
            raise KeyError(
                f"official config {Path(config_path).name} is missing {absent}: "
                "num_features / max_hidden_l / mlp_dim are derived from the irreps "
                "strings and have no default"
            )
        if "pair_repulsion" in official and not official["pair_repulsion"]:
            raise ValueError(
                f"official config {Path(config_path).name} sets pair_repulsion="
                f"{official['pair_repulsion']!r}, but MACEMatpesSpec hard-wires "
                "pair_repulsion='zbl' and the ZBL constants are buffers, so the load "
                "would not object to the surplus term: build the spec explicitly and "
                "call MATPES_REMAP.load yourself"
            )
        if (
            "distance_transform" in official
            and str(official["distance_transform"]).lower() != "agnesi"
        ):
            raise ValueError(
                f"official config {Path(config_path).name} sets distance_transform="
                f"{official['distance_transform']!r}, but MACEMatpesSpec hard-wires "
                "distance_transform='agnesi' and the Agnesi constants are buffers, so "
                "the load would not object to the surplus transform: build the spec "
                "explicitly and call MATPES_REMAP.load yourself"
            )

        num_features, max_hidden_l = _parse_irreps(official["hidden_irreps"], "hidden_irreps")
        mlp_dim, _ = _parse_irreps(official["MLP_irreps"], "MLP_irreps")

        spec = MACEMatpesSpec(
            atomic_numbers=official["atomic_numbers"],
            atomic_energies=official["atomic_energies"],
            r_max=official["r_max"],
            num_bessel=official["num_bessel"],
            num_polynomial_cutoff=official["num_polynomial_cutoff"],
            l_max=official["max_ell"],
            num_features=num_features,
            max_hidden_l=max_hidden_l,
            num_interactions=official["num_interactions"],
            correlation=official["correlation"],
            mlp_dim=mlp_dim,
            radial_mlp=official["radial_MLP"],
            scale=official["atomic_inter_scale"],
            shift=official["atomic_inter_shift"],
        )

        model = cls(spec, **ctor_kwargs)
        remap.load(model, torch.load(weights_path, map_location=map_location, weights_only=True))
        return model

    def forward(self, batch: TensorDict) -> TensorDict:
        """Run the construction-fixed pipeline, mutating ``batch`` in place.

        Reads ``atoms.{Z,pos,batch}``, ``edges.edge_index`` ``(E, 2)`` (``[:,0]``
        source, ``[:,1]`` target — never transposed on the way in), optional
        ``edges.shifts`` ``(E, 3)`` and, for the OMOL variant, optional
        ``graphs.{total_charge,total_spin}``. Writes ``graphs.energy`` ``(B,)``
        in eV and — unless the instance was built with ``compute_forces=False``
        — ``atoms.forces`` ``(N, 3)`` in eV/Å.

        Args:
            batch: Post-collate ``TensorDict``.

        Returns:
            The same ``batch`` object, with the outputs written in place.

        Raises:
            ValueError: If the batch contains an atomic number outside this
                model's element table (checked once per instance), or if it
                carries ``graphs.total_charge`` / ``graphs.total_spin`` for a
                variant that has no charge/spin conditioning.
            RuntimeError: From the force kernel, if forces were requested but
                the energy did not reach ``graphs.energy``.
        """
        return self._pipeline(batch)

    # ------------------------------------------------------------ energy core
    def energy_core(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        shifts: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        total_spin: torch.Tensor | None = None,
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

        The first six parameters keep the names and positions of the flat
        MatPES model's ``_compute_energy``
        (``src/molzoo/mace_matpes.py:229-237`` at 0e05959, before deletion) so
        those call sites move over by a pure rename. The flat OMOL model's core
        (``src/molzoo/mace_omol.py:228-237`` at 0e05959) took ``total_charge`` /
        ``total_spin`` positionally instead of ``num_graphs`` and does **not**
        line up.

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
            shifts: Optional periodic shift vectors ``(E, 3)`` in Å
                (``unit_shifts @ cell``), added to the edge displacements so a
                neighbour across a periodic boundary enters at its imaged
                position. Constant w.r.t. ``positions``, so the forces stay
                exact.
            total_charge: Per-graph total charge ``(B,)`` in units of the
                elementary charge ``e`` (OMOL only; defaults to neutral). Used
                as an integer index into an embedding table, offset by the
                spec's ``charge_offset``.
            total_spin: Per-graph total spin ``(B,)``, dimensionless (OMOL
                only; defaults to ``1``, the closed-shell singlet — every
                electron paired — so the values follow the spin-multiplicity
                ``2S+1`` convention, not a count of unpaired electrons). Also
                an embedding-table index, offset by ``spin_offset``.

        Returns:
            Total energy per graph ``(B,)`` in eV.

        Raises:
            ValueError: If a variant without charge/spin conditioning is handed
                ``total_charge`` or ``total_spin`` — silently ignoring them
                would return a plausible, wrong energy.
        """
        vectors = edge_vectors(positions, edge_index, shifts)
        lengths = edge_lengths(vectors)

        node_attrs = self.node_attrs(Z, positions.dtype)
        edge_feats, cutoff = self.radial_features(lengths, Z, edge_index)
        edge_attrs = self.angular_features(vectors)

        node_feats = self.initial_node_features(node_attrs)
        e0 = _scatter_sum(self.atomic_energies(Z), batch, num_graphs)
        node_feats, e0 = self._conditioning(
            node_feats, e0, batch, num_graphs, total_charge, total_spin
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
        return e0 + _scatter_sum(self.scale_shift(node_es), batch, num_graphs)

    # -------------------------------------------------------------- variants
    def _reject_conditioning(
        self,
        node_feats: torch.Tensor,
        e0: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        total_charge: torch.Tensor | None,
        total_spin: torch.Tensor | None,
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
        return node_feats, e0 + _scatter_sum(embedded, batch, num_graphs)

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

    # ------------------------------------------------------------- pipelines
    def _write_energy(self, batch: TensorDict) -> TensorDict:
        """Energy core on the batch schema — the ``protocol.call_energy`` hook.

        Also the whole pipeline of a ``compute_forces=False`` instance.
        Deliberately detaches **nothing**: an outer ``EnergyReadout`` /
        ``ForceReadout`` session may own the position leaf this energy hangs
        off, and detaching would cut ``F = -∂E/∂r``.

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
                present: ``edges.shifts`` ``(E, 3)`` in Å (periodic images) and,
                for the OMOL variant, ``graphs.{total_charge,total_spin}``
                ``(B,)``. A ``graphs`` namespace that is already there supplies
                ``B`` from its ``batch_size``; without one, ``B`` costs the one
                host sync on this path (``forward`` is not the compile target,
                :meth:`energy_core` is).

        Returns:
            The same ``batch``, with ``graphs.energy`` ``(B,)`` in eV written.

        Raises:
            ValueError: On the first call, if any atomic number lies outside
                this model's element table; on every call, if a variant without
                charge/spin conditioning was handed ``graphs.total_charge`` or
                ``graphs.total_spin``.
        """
        Z = batch["atoms", "Z"]
        positions = batch[POS_KEY]
        atom_batch = batch["atoms", "batch"]
        edge_index = batch["edges", "edge_index"]  # (E, 2) — the core's own convention

        # Once per instance: Z is constant over a trajectory and a wrongly
        # wired model/dataset pair fails on the very first batch, while a
        # per-call check costs a host sync and a dynamo graph break.
        if not self._elements_validated:
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

        energy = self.energy_core(
            positions,
            Z,
            edge_index,
            atom_batch,
            num_graphs,
            shifts=batch["edges", "shifts"] if ("edges", "shifts") in nested else None,
            total_charge=(
                batch["graphs", "total_charge"] if ("graphs", "total_charge") in nested else None
            ),
            total_spin=(
                batch["graphs", "total_spin"] if ("graphs", "total_spin") in nested else None
            ),
        )

        if not has_graphs:
            batch["graphs"] = TensorDict({}, batch_size=[num_graphs])
        write_energy(batch, energy)
        return batch

    def _pipeline_ef(self, batch: TensorDict) -> TensorDict:
        """Energy + forces: one energy forward, one ``autograd.grad`` backward.

        Delegates to the shared batch-level kernel
        (:func:`molpot.derivation.kernels.grad_force_pass`) — the autograd
        backend, the only one cuEquivariance's fused kernels support. This
        module hand-rolls no force path of its own.

        ``detach_energy=None`` is the ``needs_leaf`` policy the flat models
        implemented by hand: the kernel detaches ``graphs.energy`` exactly when
        it had to create the position leaf itself, and leaves it attached when
        the caller handed in a live ``requires_grad`` leaf, so a training
        ``loss.backward()`` still reaches the parameters. ``create_graph``
        follows the ambient grad mode.

        Args:
            batch: Post-collate ``TensorDict`` carrying ``atoms.pos``.

        Returns:
            The same ``batch``, with ``graphs.energy`` ``(B,)`` in eV and
            ``atoms.forces`` ``(N, 3)`` in eV/Å.
        """
        return grad_force_pass(self._write_energy, batch, detach_energy=None)
