"""The MACE products on the molnex potential contract.

:class:`MacePotential` is :class:`molix.autograd.Potential` over
:class:`~molzoo.mace.encoder.MACEEncoder`. It **inherits** the body rather than
holding one — upstream's ``ScaleShiftMACE(MACE)`` shape — so the ``state_dict``
keys stay flat and an official checkpoint loads onto the product. Which
variant it is comes entirely from the spec it is built from; the products
:class:`MaceMatpes`, :class:`MaceOmol`, :class:`MaceOff` and :class:`MaceLes`
(and :class:`~molzoo.mace.polar.MacePolar`) type that spec and name the key
table of their official archives.

Two public methods, as :mod:`molix.autograd.potential` prescribes:

``energy(frame) -> frame``
    Reads ``atoms.{Z,pos,batch}``, ``pairs.pair_index`` ``(E, 2)`` (``[:, 0]``
    source, ``[:, 1]`` target), the imaged ``pairs.pair_diff`` when present,
    ``atoms.mask`` when present and ``graphs.{total_charge,total_spin}`` when
    present (see :meth:`MACEEncoder.conditioning`); casts the geometry to the
    ``input_dtype`` of construction; runs the encoder primitives and the
    readouts; writes ``atoms.energy`` ``(N,)`` and ``graphs.energy`` ``(B,)``
    with ``B = frame["graphs"].batch_size[0]``. Per graph ``g``::

        E_g = Σ_{i∈g} E_i
        E_i = E0(Z_i) + scale · ε_i + shift  [+ r(h_i^0)]           [eV]
        ε_i = ZBL_i + Σ_layer readout_layer(h_i^layer)              [eV]

    ``E0`` is the frozen isolated-atom reference, ``ZBL`` the short-range
    nuclear repulsion (``pair_repulsion="zbl"`` only, zero otherwise),
    ``scale`` / ``shift`` the fitted per-atom normalisation, and ``r`` OMol's
    readout of the charge/spin-conditioned initial state. MatPES, OFF and
    Polar read out every layer; OMol reads out the last one. A product with a
    graph-level term — MACE-LES's latent Ewald sum, Polar's Hartree energy —
    adds it to ``graphs.energy`` after the per-atom scatter, in its own
    ``energy``. Every per-atom term is multiplied by ``atoms.mask`` before the
    scatter (a padded batch's ghosts add exactly zero; a multiply, never a
    slice), and cast to ``energy_dtype`` before it is added — ``E0`` is
    thousands of eV and would swallow the interaction in float32.

``forward(frame) -> frame``
    The :class:`~molix.autograd.Derivative` built once in ``__init__`` from
    the ``wrt`` keys: ``MaceMatpes(spec, dtype=torch.float64)`` writes the
    energy alone, ``MaceMatpes(spec, wrt=("atoms.pos",), dtype=torch.float64)``
    the energy and ``atoms.forces``
    (``F_i = -∂E/∂r_i``, eV/Å), ``("atoms.pos", "graphs.strain")`` adds
    ``graphs.virial``. Compile it with ``Compiler(...)(model)`` after one eager
    forward.

Two eager preconditions a trace cannot establish: the element-table check,
spent on the first eager call (:meth:`MACEEncoder._validate_once`), and — on a
periodic MACE-LES run — one :meth:`MaceLes.bind_cell` before compiling.

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
    Cheng, B. "Latent Ewald summation for machine learning of long-range
    interactions." npj Comput. Mater. 11, 80 (2025).
    https://doi.org/10.1038/s41524-025-01577-7
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

import cuequivariance as cue
import torch
import torch.nn as nn
from tensordict import TensorDictBase

from molix.autograd import Derivative, Potential
from molix.F.scatter import scatter_sum_compile_safe
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY
from molpot.heads.energy import AtomicReferenceEnergy
from molpot.heads.rescale import GlobalRescale
from molpot.potentials.elec.les import LatentEwaldEnergy
from molpot.potentials.repulsion import ZBLRepulsion
from molrep import cueq
from molrep.embedding.geometry import edge_lengths, edge_vectors
from molzoo.checkpoint import CheckpointRemap
from molzoo.mace.checkpoint import MACE_SKIP, les_keys, matpes_keys, off_keys, omol_keys
from molzoo.mace.encoder import MACEEncoder
from molzoo.mace.heads import LinearReadout, NonLinearBiasReadout, NonLinearReadout
from molzoo.mace.spec import (
    MACELesSpec,
    MACEMatpesSpec,
    MaceOfficialConfig,
    MACEOffSpec,
    MACEOMolSpec,
    MACESpec,
)
from molzoo.mace.utils import MaceArchive

#: The ``wrt`` keys of :class:`molix.autograd.Derivative`.
Wrt = Sequence[str | Sequence[str]]

#: The dtypes a potential accepts for its accumulation and its input cast.
_FLOAT_DTYPES = (None, torch.float32, torch.float64)


class MacePotential(Potential, MACEEncoder):
    """A MACE body with its energy heads: the shared product.

    Args:
        spec: Validated variant configuration; the product classes narrow it
            to their own spec type.
        wrt: What :meth:`forward` derives beside the energy, as TensorFrame
            keys (:class:`molix.autograd.Derivative`): ``()`` (the default)
            writes ``graphs.energy`` alone, ``("atoms.pos",)`` adds
            ``atoms.forces``, ``("atoms.pos", "graphs.strain")`` adds
            ``graphs.virial`` on a one-graph periodic batch.
        energy_dtype: Accumulation dtype of the per-atom and graph energies,
            float64 by default even with fp32 model weights. ``None`` uses
            the model dtype. This does not promote the learned calculation or
            recover digits lost when loading weights into a lower-precision
            model.
        input_dtype: Dtype the geometry (``atoms.pos``, ``pairs.pair_diff``,
            ``graphs.cell``) is cast to at the top of :meth:`energy`, so an
            fp64 MD state can drive an fp32 model. The cast is differentiable:
            forces still refer to the caller's leaf and come back at its
            dtype. ``None`` leaves the inputs untouched.
        dtype: Floating dtype every parameter and buffer is built at.

    Raises:
        ValueError: If ``energy_dtype`` or ``input_dtype`` is not
            ``torch.float32``, ``torch.float64`` or ``None``.
    """

    readouts: nn.ModuleList | None
    readout: NonLinearBiasReadout | None

    def __init__(
        self,
        spec: MACESpec,
        *,
        wrt: Wrt = (),
        energy_dtype: torch.dtype | None = torch.float64,
        input_dtype: torch.dtype | None = None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(spec, dtype=dtype)
        if energy_dtype not in _FLOAT_DTYPES:
            raise ValueError(
                "energy_dtype must be torch.float32, torch.float64 or None (model dtype)"
            )
        if input_dtype not in _FLOAT_DTYPES:
            raise ValueError("input_dtype must be torch.float32, torch.float64 or None")
        self.energy_dtype = energy_dtype
        self._input_dtype = input_dtype

        # The heads live on the product, not on a nested body, so the
        # checkpoint prefixes ``readouts`` / ``readout`` / ``atomic_energies``
        # / ``scale_shift`` / ``pair_repulsion`` / ``embedding_readout`` stay
        # at the module root.
        self.pair_repulsion = (
            ZBLRepulsion(exponent=spec.num_polynomial_cutoff, dtype=self.dtype)
            if spec.pair_repulsion == "zbl"
            else None
        )
        self.embedding_readout = (
            cueq.Linear(
                cue.Irreps("O3", self.scalar_irreps),
                cue.Irreps("O3", "1x0e"),
                layout=cue.ir_mul,
                dtype=self.dtype,
            )
            if self.joint_embedding is not None
            else None
        )
        self.atomic_energies = AtomicReferenceEnergy(
            atomic_energies=spec.atomic_energies,
            atomic_numbers=spec.atomic_numbers,
        )
        self.readouts = None
        self.readout = None
        if spec.readout == "per_layer":
            self.readouts = self._readout_stack(spec.mlp_dim, spec.last_readout_bias)
        else:
            self.readout = NonLinearBiasReadout(
                irreps_in=self.scalar_irreps, mlp_dim=spec.mlp_dim, dtype=self.dtype
            )
        self.scale_shift = GlobalRescale(scale=spec.scale, shift=spec.shift, dtype=self.dtype)
        self._derivative = Derivative(self.energy, wrt=wrt)

    # ------------------------------------------------------------ construction
    def _readout_stack(self, mlp_dim: int, last_bias: bool) -> nn.ModuleList:
        """One scalar readout per interaction layer.

        Upstream MACE: readout 0 is always linear; the non-linear readout is
        the last of *several* layers, so a single-interaction model has one
        linear readout.

        Args:
            mlp_dim: Hidden width of the last, non-linear readout.
            last_bias: Whether that readout carries a bias (Polar's does).
        """
        n_layers = self.num_interactions
        last = NonLinearBiasReadout if last_bias else NonLinearReadout
        return nn.ModuleList(
            [
                (
                    last(irreps_in=self.layer_irreps[i], mlp_dim=mlp_dim, dtype=self.dtype)
                    if i == n_layers - 1 and n_layers > 1
                    else LinearReadout(irreps_in=self.layer_irreps[i], dtype=self.dtype)
                )
                for i in range(n_layers)
            ]
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config: str | Path | None = None,
        wrt: Wrt = (),
        energy_dtype: torch.dtype | None = torch.float64,
        input_dtype: torch.dtype | None = None,
        dtype: torch.dtype,
    ) -> Self:
        """Construct from an official archive and load its weights.

        Each product implements it over its own spec's ``from_official`` and
        its family's key table, through :meth:`_load`.

        Args:
            path: Weights file, pickled ``.model``, config json, or a directory
                (:meth:`molzoo.mace.utils.MaceArchive.read`).
            config: Optional sidecar json when ``path`` is a plain state dict.
            wrt: As the constructor's.
            energy_dtype: As the constructor's.
            input_dtype: As the constructor's.
            dtype: As the constructor's.

        Returns:
            A model of ``cls`` holding the archive's weights.

        Raises:
            NotImplementedError: On the shared base class, which names no
                archive family.
        """
        del path, config, wrt, energy_dtype, input_dtype, dtype
        raise NotImplementedError(f"{cls.__name__} names no official archive family")

    @classmethod
    def _load(
        cls,
        path: str | Path,
        config: str | Path | None,
        spec_from: Callable[[MaceOfficialConfig], MACESpec],
        keys: Mapping[str, str | None],
        *,
        on_unexpected: Literal["raise", "return"] = "raise",
        wrt: Wrt,
        energy_dtype: torch.dtype | None,
        input_dtype: torch.dtype | None,
        dtype: torch.dtype,
    ) -> Self:
        """Read the archive, build the spec, construct, remap the weights in.

        Args:
            path: As :meth:`load`.
            config: As :meth:`load`.
            spec_from: The product's spec constructor from an official config.
            keys: The family's key table (:mod:`molzoo.mace.checkpoint`).
            on_unexpected: The family's policy on an unhoused key.
            wrt: As the constructor's.
            energy_dtype: As the constructor's.
            input_dtype: As the constructor's.
            dtype: As the constructor's.

        Raises:
            ValueError: If the checkpoint's element table disagrees with the
                configuration's — the ``E0`` table would be misaligned.
        """
        state, cfg = MaceArchive().read(path, config)
        model = cls(
            spec_from(cfg), wrt=wrt, energy_dtype=energy_dtype, input_dtype=input_dtype, dtype=dtype
        )
        numbers = state.get("atomic_numbers")
        if numbers is not None and not torch.equal(numbers.cpu().reshape(-1), model.z_table.cpu()):
            raise ValueError(
                "checkpoint atomic_numbers disagree with the configuration element table"
            )
        CheckpointRemap(keys, on_unexpected=on_unexpected, skip=MACE_SKIP).load(model, state)
        return model

    # ----------------------------------------------------------------- energy
    def energy(self, frame: TensorDictBase) -> TensorDictBase:
        """Write ``atoms.energy`` ``(N,)`` and ``graphs.energy`` ``(B,)`` in place.

        Args:
            frame: Post-collate batch carrying ``atoms.{Z,pos,batch}``,
                ``pairs.pair_index`` and the ``graphs`` namespace at
                ``batch_size=[B]``; ``pairs.pair_diff`` (the imaged
                displacement — required on a periodic batch), ``atoms.mask``
                and ``graphs.{total_charge,total_spin}`` when present.

        Returns:
            The same ``frame``, both energy columns written and attached to
            whatever the caller's leaves are.

        Raises:
            ValueError: On the first eager call, if any atomic number lies
                outside this model's element table; on every call, if an
                unconditioned variant was handed ``graphs.total_charge`` or
                ``graphs.total_spin``.
        """
        Z = frame["atoms", "Z"]
        self._validate_once(Z)
        keys = frame.keys(include_nested=True)
        pos = frame["atoms", "pos"]
        edge_diff = frame["pairs", "pair_diff"] if ("pairs", "pair_diff") in keys else None
        if self._input_dtype is not None:
            pos = pos.to(dtype=self._input_dtype)
            edge_diff = None if edge_diff is None else edge_diff.to(dtype=self._input_dtype)
        batch = frame["atoms", "batch"]
        num_graphs = frame["graphs"].batch_size[0]
        atom_energy, _ = self._atom_energy(
            pos,
            Z,
            frame["pairs", "pair_index"].contiguous(),
            batch,
            num_graphs,
            edge_diff=edge_diff,
            total_charge=(
                frame["graphs", "total_charge"] if ("graphs", "total_charge") in keys else None
            ),
            total_spin=frame["graphs", "total_spin"] if ("graphs", "total_spin") in keys else None,
            atom_mask=frame["atoms", "mask"] if ("atoms", "mask") in keys else None,
        )
        frame[ATOMIC_ENERGY_KEY] = atom_energy
        frame[ENERGY_KEY] = scatter_sum_compile_safe(atom_energy, batch, num_graphs)
        return frame

    def _atom_energy(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        *,
        edge_diff: torch.Tensor | None,
        total_charge: torch.Tensor | None,
        total_spin: torch.Tensor | None,
        atom_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The tensor body of :meth:`energy`: per-atom energies and the layer features.

        Shared with :meth:`MaceLes.energy`, whose latent charges read the
        same layer features. Arguments are the columns :meth:`energy` reads,
        already cast; ``num_graphs`` is ``B``.

        Returns:
            ``(atom energy (N,) at the accumulation dtype, masked; per-layer node
            features)``.
        """
        vectors = edge_vectors(positions, edge_index, edge_diff=edge_diff)
        lengths = edge_lengths(vectors)
        elements = self.element_index(Z)
        edge_feats, cutoff = self.radial_features(lengths, Z, edge_index)
        node_feats = self.conditioning(
            self.initial_node_features(elements), total_charge, total_spin, batch, num_graphs
        )
        per_layer = self.layer_features(
            node_feats=node_feats,
            elements=elements,
            edge_attrs=self.angular_features(vectors),
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=cutoff,
        )
        interaction = (
            self._sum_readouts(self.readouts, per_layer)
            if self.readouts is not None
            else self.readout(per_layer[-1]).squeeze(-1)
        )
        if self.pair_repulsion is not None:
            # The short-range pair term is a per-atom energy on the same
            # footing as the readouts, summed first (MatPES operator order).
            interaction = self.pair_repulsion(lengths, Z, edge_index) + interaction
        atom_energy = self._accumulate(self.atomic_energies(Z)) + self._accumulate(
            self.scale_shift(interaction)
        )
        if self.embedding_readout is not None:
            atom_energy = atom_energy + self._accumulate(
                self.embedding_readout(node_feats).squeeze(-1)
            )
        if atom_mask is not None:
            atom_energy = atom_energy * atom_mask.to(dtype=atom_energy.dtype)
        return atom_energy, per_layer

    @staticmethod
    def _sum_readouts(readouts: nn.ModuleList, per_layer: list[torch.Tensor]) -> torch.Tensor:
        """Sum one scalar readout per interaction layer into ``(N,)``."""
        total = per_layer[0].new_zeros(per_layer[0].shape[0])
        for readout, node_feats in zip(readouts, per_layer, strict=True):
            total = total + readout(node_feats).squeeze(-1)
        return total

    def _accumulate(self, values: torch.Tensor) -> torch.Tensor:
        """``values`` at the energy accumulation dtype (``energy_dtype``)."""
        return values if self.energy_dtype is None else values.to(dtype=self.energy_dtype)


class MaceMatpes(MacePotential):
    """MACE-MatPES: the density-normalised stack with ZBL repulsion and the Agnesi transform.

    ``MaceMatpes.load(".../matpes_r2scan_config.json")`` loads the official
    checkpoints (spec ``src/molzoo/specs/mace_matpes.md``).
    """

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config: str | Path | None = None,
        wrt: Wrt = (),
        energy_dtype: torch.dtype | None = torch.float64,
        input_dtype: torch.dtype | None = None,
        dtype: torch.dtype,
    ) -> Self:
        """Load an official MatPES archive (:meth:`MacePotential.load`)."""
        return cls._load(
            path,
            config,
            MACEMatpesSpec.from_official,
            matpes_keys,
            wrt=wrt,
            energy_dtype=energy_dtype,
            input_dtype=input_dtype,
            dtype=dtype,
        )


class MaceOmol(MacePotential):
    """MACE-OMol: the charge/spin-conditioned residual stack with a final readout.

    ``MaceOmol.load(path)`` loads the official checkpoints (spec
    ``src/molzoo/specs/mace_omol.md``); their auxiliary heads are returned,
    not raised, by the remap.
    """

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config: str | Path | None = None,
        wrt: Wrt = (),
        energy_dtype: torch.dtype | None = torch.float64,
        input_dtype: torch.dtype | None = None,
        dtype: torch.dtype,
    ) -> Self:
        """Load an official OMol archive (:meth:`MacePotential.load`)."""
        return cls._load(
            path,
            config,
            MACEOMolSpec.from_official,
            omol_keys,
            on_unexpected="return",
            wrt=wrt,
            energy_dtype=energy_dtype,
            input_dtype=input_dtype,
            dtype=dtype,
        )


class MaceOff(MacePotential):
    """MACE-OFF: the organic force fields (standard MACE, no ZBL, no distance transform).

    ``MaceOff.load(path)`` loads the official checkpoints (spec
    ``src/molzoo/specs/mace_off.md``).
    """

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config: str | Path | None = None,
        wrt: Wrt = (),
        energy_dtype: torch.dtype | None = torch.float64,
        input_dtype: torch.dtype | None = None,
        dtype: torch.dtype,
    ) -> Self:
        """Load an official MACE-OFF archive (:meth:`MacePotential.load`)."""
        return cls._load(
            path,
            config,
            MACEOffSpec.from_official,
            off_keys,
            wrt=wrt,
            energy_dtype=energy_dtype,
            input_dtype=input_dtype,
            dtype=dtype,
        )


class MaceLes(MaceOff):
    """MACE-LES: MACE-OFF plus the latent-charge long-range energy.

    One charge readout per layer, the same kind as that layer's energy readout
    (linear, then the non-linear one last); their sum times
    ``les_output_scale`` is each atom's latent charge ``q_i`` in e. The
    long-range energy is :class:`~molpot.potentials.elec.les.LatentEwaldEnergy`
    of those charges — an isolated Gaussian pair sum, or with ``graphs.cell``
    the reciprocal sum — a graph-level term with no per-atom attribution, added
    to ``graphs.energy`` after the per-atom scatter. A periodic run calls
    :meth:`bind_cell` once, eagerly, before compiling.

    ``MaceLes.load(path)`` loads the official checkpoints; only the charge-only
    LES (no latent dipoles, quadrupoles or induced terms) is ported.
    """

    def __init__(
        self,
        spec: MACELesSpec,
        *,
        wrt: Wrt = (),
        energy_dtype: torch.dtype | None = torch.float64,
        input_dtype: torch.dtype | None = None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            spec,
            wrt=wrt,
            energy_dtype=energy_dtype,
            input_dtype=input_dtype,
            dtype=dtype,
        )
        self.charge_readouts = self._readout_stack(spec.mlp_dim, last_bias=False)
        self.les_output_scale = spec.les_output_scale
        self.les = LatentEwaldEnergy(
            sigma=spec.les_sigma,
            dl=spec.les_dl,
            norm_factor=spec.les_norm_factor,
            remove_self_interaction=spec.les_remove_self_interaction,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config: str | Path | None = None,
        wrt: Wrt = (),
        energy_dtype: torch.dtype | None = torch.float64,
        input_dtype: torch.dtype | None = None,
        dtype: torch.dtype,
    ) -> Self:
        """Load an official MACE-LES archive (:meth:`MacePotential.load`)."""
        return cls._load(
            path,
            config,
            MACELesSpec.from_official,
            les_keys,
            wrt=wrt,
            energy_dtype=energy_dtype,
            input_dtype=input_dtype,
            dtype=dtype,
        )

    def bind_cell(self, cell: torch.Tensor) -> None:
        """Size (first call) or capacity-check the LES reciprocal box. Eager only.

        A construction-time precondition of the periodic path, like the
        element-table check: sizing reads the lattice norms back to the host,
        which a traced call cannot do, so it precedes ``Compiler(...)(model)``.
        Idempotent; later calls only re-check the capacity the first bought
        (:class:`molix.schema.CellBound`).

        Args:
            cell: Lattice vectors ``(3, 3)`` or ``(1, 3, 3)`` in Å.
        """
        self.les.bind_cell(cell)

    def energy(self, frame: TensorDictBase) -> TensorDictBase:
        """The MACE-OFF atom energies plus the latent Ewald sum on ``graphs.energy``.

        Reads what :meth:`MacePotential.energy` reads and ``graphs.cell``
        ``(1, 3, 3)`` when present (the periodic LES path; single system).
        ``atoms.energy`` holds the short-range per-atom part alone, so its
        graph sum differs from ``graphs.energy`` by the long-range term.
        """
        Z = frame["atoms", "Z"]
        self._validate_once(Z)
        keys = frame.keys(include_nested=True)
        pos = frame["atoms", "pos"]
        edge_diff = frame["pairs", "pair_diff"] if ("pairs", "pair_diff") in keys else None
        cell = frame["graphs", "cell"] if ("graphs", "cell") in keys else None
        if self._input_dtype is not None:
            pos = pos.to(dtype=self._input_dtype)
            edge_diff = None if edge_diff is None else edge_diff.to(dtype=self._input_dtype)
            cell = None if cell is None else cell.to(dtype=self._input_dtype)
        batch = frame["atoms", "batch"]
        num_graphs = frame["graphs"].batch_size[0]
        atom_mask = frame["atoms", "mask"] if ("atoms", "mask") in keys else None
        atom_energy, per_layer = self._atom_energy(
            pos,
            Z,
            frame["pairs", "pair_index"].contiguous(),
            batch,
            num_graphs,
            edge_diff=edge_diff,
            total_charge=(
                frame["graphs", "total_charge"] if ("graphs", "total_charge") in keys else None
            ),
            total_spin=frame["graphs", "total_spin"] if ("graphs", "total_spin") in keys else None,
            atom_mask=atom_mask,
        )
        charges = self._sum_readouts(self.charge_readouts, per_layer) * self.les_output_scale
        if atom_mask is not None:
            charges = charges * atom_mask.to(dtype=charges.dtype)
        energy = scatter_sum_compile_safe(atom_energy, batch, num_graphs)
        long_range = self.les(pos, charges, batch, num_graphs, cell=cell)
        frame[ATOMIC_ENERGY_KEY] = atom_energy
        frame[ENERGY_KEY] = energy + long_range.to(dtype=energy.dtype)
        return frame
