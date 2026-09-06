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

:class:`MacePotential` is the one energy + force host for every MACE
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

    The seam performs **no element validation**. The check host-syncs (see
    :meth:`~molzoo.mace.encoder.MACEEncoder.validate_elements`), so it cannot
    live inside a traced region; whoever drives the flat seam directly —
    ``torch.compile``, ``torch.export``, ``molix.engine.export_for_lammps`` —
    calls ``validate_elements`` once itself beforehand. On the batch path
    :meth:`MacePotential._write_energy` is that caller, on its first *eager*
    call.

    :meth:`MacePotential.compile_core` installs a compiled build of this same
    method, which is what makes the compiled path exist at all. That build
    covers **first-order, cell-bound, B = 1 MD inference only**: forces come
    from one ``torch.autograd.grad`` *outside* the compiled region, and the
    second backward a force-supervised training loss needs is not compilable
    under AOTAutograd (spec ``cuet-force-doublebackward``) — training keeps
    running the eager core.

``_write_energy(batch) -> batch``
    The **protocol hook**, one level up, on the batch schema.
    ``molpot.derivation.protocol.call_energy`` prefers ``model._write_energy``,
    so implementing it wires MACE into ``EnergyReadout`` / ``ForceReadout`` for
    free. It reads the tensors off the post-collate batch, calls
    :meth:`MacePotential.energy_core`, and writes the result back through
    ``protocol.write_energy`` — **without** detaching either the positions or
    the energy. Detaching a tensor cuts it out of the autograd graph; an outer
    readout session may have created the position *leaf* (the tensor autograd
    differentiates with respect to) itself, so a detach here would sever
    ``F = -∂E/∂r`` before anyone takes the derivative. (The flat models
    detached inside their ``energy_forces(compute_forces=False)``; that duty
    belongs to :class:`molpot.forcefield.potentials.Potentials`, not to the
    potential.)

Every branch — forces on/off, charge/spin conditioning, per-layer vs final
readout — is resolved once in ``__init__`` into a bound attribute, and
``forward`` is a single dispatch through ``self._pipeline``. There is no
``compute_forces`` flag on any public signature: "energy only" is a
*construction* (``MacePotential(spec, compute_forces=False)``), which is also
what ``molpot.forcefield.potentials.Potentials.calc_energy`` already assumes.

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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
from tensordict import TensorDict

from molix.F.scatter import scatter_sum_compile_safe as _scatter_sum
from molix.schema import POS_KEY
from molpot.derivation.force import autograd_forces_from_energy
from molpot.derivation.kernels import grad_force_pass
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
    ``graphs.energy`` ``(B,)`` in eV and, unless built energy-only, the per-atom
    forces ``atoms.forces`` ``(N, 3)`` in eV/Å; the energy expression it
    evaluates and the ``F_i = -∂E/∂r_i`` convention are spelled out in the
    module docstring. The variant is the spec's business; the only two switches
    here decide *how much* is computed and *which cuEquivariance path* the
    blocks take::

        potential = MaceMatpes(atomic_numbers=..., atomic_energies=...)
        potential = MaceOmol(..., compute_forces=False)
        potential = MacePolar(..., use_fallback=True)

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
        compute_forces: bool = True,
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
        #: One bound call path; ``forward`` never branches (cf. PiNetPotential).
        self._pipeline = self._pipeline_ef if compute_forces else self._write_energy
        #: Element-table gate, spent after the first batch (see ``_write_energy``).
        self._elements_validated = False
        #: Compiled build of :meth:`energy_core`, or ``None`` while eager.
        #: A plain attribute, never a buffer: it is a callable, not model state,
        #: and must stay out of ``state_dict``. Installed by
        #: :meth:`compile_core`, read by :meth:`_write_energy`.
        self._core: Callable[..., torch.Tensor] | None = None

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
            **kwargs: Constructor overrides (``use_fallback``, ``compute_forces``,
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

    def energy_forces(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int | None = None,
        *,
        edge_diff: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        total_spin: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Energy and forces from raw tensors.

        Returns:
            ``{"energy": (B,) eV, "forces": (N, 3) eV/Å}``.
        """
        if num_graphs is None:
            num_graphs = (
                int(total_charge.shape[0])
                if total_charge is not None
                else int(batch.max().item()) + 1
            )
        leaf = positions.detach().requires_grad_(True)
        with torch.enable_grad():
            energy = self._evaluate_core(
                leaf,
                Z,
                edge_index,
                batch,
                num_graphs,
                # A new position leaf owns this inference derivative. An
                # incoming live edge graph belongs to the caller's old leaf.
                edge_diff=None if edge_diff is None else edge_diff.detach(),
                total_charge=total_charge,
                total_spin=total_spin,
                cell=cell,
            )
        return {"energy": energy.detach(), "forces": autograd_forces_from_energy(energy, leaf)}

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
        if self._core is not None:
            raise RuntimeError("configure input dtype before compile_core")
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
    ) -> torch.Tensor:
        """One dispatch/precision boundary for flat and TensorDict callers."""
        dtype = self._input_dtype
        if dtype is not None:
            positions = positions.to(dtype=dtype)
            edge_diff = None if edge_diff is None else edge_diff.to(dtype=dtype)
            cell = None if cell is None else cell.to(dtype=dtype)
            total_charge = None if total_charge is None else total_charge.to(dtype=dtype)
            total_spin = None if total_spin is None else total_spin.to(dtype=dtype)
        core = self._core if self._core is not None else self.energy_core
        return core(
            positions,
            Z,
            edge_index,
            batch,
            num_graphs,
            edge_diff=edge_diff,
            total_charge=total_charge,
            total_spin=total_spin,
            cell=cell,
        )

    def _sum_energy(
        self, values: torch.Tensor, batch: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        """Per-graph energy reduction with the user's accumulation precision."""
        if self.energy_dtype is not None:
            values = values.to(dtype=self.energy_dtype)
        return _scatter_sum(values, batch, num_graphs)

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

    def compile_core(self, compiler: Callable[[Callable[..., Any]], Callable[..., Any]]) -> None:
        """Install a compiled build of :meth:`energy_core` as the evaluated core.

        The single action is ``self._core = compiler(self.energy_core)``.
        :meth:`_write_energy` then calls ``self._core`` instead of
        :meth:`energy_core`, so no caller ever branches on "is it compiled?".
        ``compiler`` is **structural** — any ``callable -> callable``, such as
        :class:`molix.compile.Compiler`, which this module deliberately does
        not import.

        **Installation is idempotent and shared.** The mlip is one
        ``nn.Module`` held by the ForceField, and every
        :meth:`~molpot.forcefield.container.ForceField.to_potentials` call
        builds a *fresh* ``Potentials`` around that same object — which is
        exactly why the core is installed here and not on the container: it
        survives every rebuild. A second call while a core is installed returns
        without invoking ``compiler``, so a second, *different* compiler is
        silently ignored. There is no ``uninstall``: to get the eager core back
        (a force-supervised training run, say, whose double backward is not
        compilable at all), build a new potential.

        **Prescribed sequence.** The caller composes it — scripts, docs and
        ``regressions/`` do, the library does not::

            ff.def_pairstyle(model)
            pots = ff.to_potentials()
            pots.bind_cell(cell)      # size the grids + capacity check
            pots.calc_energy(work)    # one eager forward: elements + grids
            pots.compile_core(Compiler(dynamic=True, backend=...))

        ``dynamic=True`` on the MD path: the live edge count varies with every
        neighbour-list rebuild, and a static shape would recompile each time.

        **What the batch must carry** is unchanged by compiling, because
        :meth:`_write_energy` keeps doing the unpacking either way: it resolves
        ``num_graphs``, creates the ``graphs`` namespace when absent, and writes
        the energy back through ``protocol.write_energy`` **undetached**. A
        batch without a ``graphs`` namespace does *not* raise — it takes the one
        documented host sync (``int(atom_batch.max().item()) + 1``), which is
        affordable because ``_write_energy`` itself is never traced. A batch
        without ``edges.edge_diff`` does raise, with the existing
        ``ValueError``.

        Args:
            compiler: Any ``callable -> callable``; applied once to the bound
                :meth:`energy_core`.

        Raises:
            RuntimeError: If no eager forward has happened yet — the element
                table is unvalidated and, on the polarisable variant, the
                k-space grids are unsized, neither of which a traced call can
                repair.
        """
        if not self._elements_validated:
            raise RuntimeError(
                "compile_core needs a validated element table: run one eager "
                "forward first (bind_cell(cell) then calc_energy(batch)) so the "
                "element check and any k-space grid sizing happen outside the trace"
            )
        if self._core is not None:
            return
        self._core = compiler(self.energy_core)

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
        On the batch path :meth:`_write_energy` is that caller and does it on
        its first *eager* call.

        This method is also what :meth:`compile_core` compiles. The compiled
        build covers **first-order, cell-bound, B = 1 MD inference only** — the
        force backward runs outside it, and double backward (force-supervised
        training) is not compilable under AOTAutograd, spec
        ``cuet-force-doublebackward``.

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
        e0 = self._sum_energy(self.atomic_energies(Z), batch, num_graphs)
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
        return e0 + self._sum_energy(self.scale_shift(node_es), batch, num_graphs)

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
        return node_feats, e0 + self._sum_energy(embedded, batch, num_graphs)

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
            ValueError: On the first **eager** call, if any atomic number lies
                outside this model's element table — the compiled seam never
                validates, since the check host-syncs (see
                :meth:`compile_core`); on every call, if a variant without
                charge/spin conditioning was handed ``graphs.total_charge`` or
                ``graphs.total_spin``; on every call, if ``edges.edge_diff`` is
                absent from ``batch`` (``edges.shifts`` alone is no longer a
                PBC channel — pass the already-imaged displacement instead).
        """
        Z = batch["atoms", "Z"]
        positions = batch[POS_KEY]
        atom_batch = batch["atoms", "batch"]
        edge_index = batch["edges", "edge_index"]  # (E, 2) — the core's own convention

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

        if ("edges", "edge_diff") not in nested:
            raise ValueError(
                "MacePotential.forward requires edges.edge_diff; "
                "edges.shifts is no longer a PBC channel"
            )
        cell = None
        if ("graphs", "cell") in nested:
            cell = batch["graphs", "cell"]
        # One evaluation path: the compiled build when compile_core installed
        # one, the eager bound method otherwise. No caller branches on this.
        energy = self._evaluate_core(
            positions,
            Z,
            edge_index,
            atom_batch,
            num_graphs,
            edge_diff=batch["edges", "edge_diff"],
            total_charge=(
                batch["graphs", "total_charge"] if ("graphs", "total_charge") in nested else None
            ),
            total_spin=(
                batch["graphs", "total_spin"] if ("graphs", "total_spin") in nested else None
            ),
            cell=cell,
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
        compute_forces: bool = True,
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
            compute_forces=compute_forces,
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
        ctor: dict[str, Any] = dict(
            atomic_numbers=official["atomic_numbers"],
            atomic_energies=official["atomic_energies"],
            r_max=official.get("r_max"),
            num_bessel=official.get("num_bessel"),
            num_polynomial_cutoff=official.get("num_polynomial_cutoff"),
            l_max=official.get("max_ell", official.get("l_max")),
            num_features=num_features,
            max_hidden_l=max_hidden_l,
            num_interactions=official.get("num_interactions"),
            correlation=official.get("correlation"),
            mlp_dim=mlp_dim,
            radial_mlp=official.get("radial_MLP", official.get("radial_mlp")),
            scale=official.get("atomic_inter_scale", official.get("scale")),
            shift=official.get("atomic_inter_shift", official.get("shift")),
        )
        ctor = {k: v for k, v in ctor.items() if v is not None}
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
        compute_forces: bool = True,
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
            compute_forces=compute_forces,
            use_fallback=use_fallback,
            energy_dtype=energy_dtype,
        )

    @classmethod
    def _from_official(cls, official: dict[str, Any], **kwargs: Any) -> MaceOmol:
        ctor: dict[str, Any] = dict(
            atomic_numbers=official["atomic_numbers"],
            atomic_energies=official["atomic_energies"],
            r_max=official.get("r_max"),
            num_bessel=official.get("num_bessel"),
            num_interactions=official.get("num_interactions"),
            l_max=official.get("max_ell", official.get("l_max")),
            num_features=official.get("num_features"),
            correlation=official.get("correlation"),
            mlp_dim=official.get("mlp_dim"),
            edge_channels=official.get("edge_channels"),
            scale=official.get("atomic_inter_scale", official.get("scale")),
            shift=official.get("atomic_inter_shift", official.get("shift")),
        )
        ctor = {k: v for k, v in ctor.items() if v is not None}
        ctor.update(kwargs)
        return cls(**ctor)
