"""Allegro energy model: the feature body plus upstream's energy heads.

:class:`AllegroPotential` is a :class:`~molix.autograd.Potential` over the
:class:`~molzoo.allegro.encoder.AllegroEncoder` body: it registers, at the
module root, upstream's ``edge_readout`` MLP, ``per_type_energy_scale_shift``
and the optional ZBL ``pair_potential`` — the energy of ``FullAllegroModel``
(``src/molzoo/specs/allegro.md`` §5 step 7):

.. math::

    E_i = s_{t_i} \\frac{1}{\\sqrt 2\\,\\sqrt{\\langle N\\rangle}}
          \\sum_j \\mathrm{MLP}_r(h_{ij}) + b_{t_i} \\; [+\\, E^{ZBL}_i]

Two public methods, the potential contract: :meth:`AllegroPotential.energy`
reads the frame's columns and writes ``atoms.energy`` and ``graphs.energy``;
``forward`` is the one :class:`molix.autograd.Derivative` over ``energy``,
fixed by ``wrt=`` at construction. :meth:`AllegroPotential.load` builds the
model an official ``.nequip.zip`` describes and loads its weights.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nat. Commun. 14, 579 (2023).
    https://arxiv.org/abs/2204.05249
    ``allegro/model/allegro_models.py::FullAllegroModel`` (nequip-allegro 0.8.3).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch
from tensordict import TensorDictBase

from molix.autograd import Derivative, Potential
from molix.F.scatter import scatter_sum_compile_safe
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY
from molpot.heads.rescale import PerSpeciesScaleShift
from molpot.potentials.repulsion import UniversalZBL
from molrep.embedding.mlp import ScalarMLPFunction
from molzoo.allegro.checkpoint import allegro_keys
from molzoo.allegro.encoder import AllegroEncoder
from molzoo.allegro.spec import AllegroSpec
from molzoo.allegro.utils import NequipPackage
from molzoo.checkpoint import CheckpointRemap


class AllegroPotential(Potential, AllegroEncoder):
    """Allegro energy (+ derivatives fixed by ``wrt``), written into the frame.

    ::

        spec = AllegroSpec(atomic_numbers=[1, 6, 8], r_max=5.0)
        model = AllegroPotential(spec, wrt=("atoms.pos",))
        model = AllegroPotential.load("Allegro-OAM-L-0.1.nequip.zip", wrt=("atoms.pos",))

    Args:
        spec: The configuration (upstream ``FullAllegroModel`` names and defaults).
        wrt: TensorFrame keys :meth:`forward` differentiates the energy
            against (``"atoms.pos"`` → forces, ``"graphs.strain"`` → virial);
            ``()`` is energy only.
        dtype: Floating dtype every parameter and buffer is built at (the
            per-type scale/shift stays float64, as upstream); ``None`` is the
            (transitional) process-wide default.

    Raises:
        ValueError: If ``spec.expose_tensor_track`` is set — the exposed track
            exists for feature heads; an energy model built on it would carry
            weights no official package has.
    """

    def __init__(
        self,
        spec: AllegroSpec,
        *,
        wrt: Sequence[str | Sequence[str]] = (),
        dtype: torch.dtype | None = None,
    ) -> None:
        if spec.expose_tensor_track:
            raise ValueError("expose_tensor_track is an encoder option; use AllegroEncoder")
        super().__init__(spec, dtype=dtype)
        n_types = len(spec.atomic_numbers)
        self.edge_readout = ScalarMLPFunction(
            self.output_dim,
            1,
            hidden_layers_depth=spec.readout_mlp_hidden_layers_depth,
            hidden_layers_width=spec.readout_mlp_hidden_layers_width,
            nonlinearity=spec.readout_mlp_nonlinearity,
            dtype=self.dtype,
        )
        self.per_type_energy_scale_shift = PerSpeciesScaleShift(
            num_species=n_types,
            scales=spec.per_type_energy_scales,
            shifts=spec.per_type_energy_shifts,
            scales_trainable=spec.per_type_energy_scales_trainable,
            shifts_trainable=spec.per_type_energy_shifts_trainable,
            dtype=torch.float64,
        )
        self.pair_potential = (
            UniversalZBL(r_max=spec.r_max, exponent=spec.zbl_polynomial_cutoff_p, dtype=self.dtype)
            if spec.zbl
            else None
        )
        #: EdgewiseReduce factor ``1/√⟨N⟩ / √2``.
        self.edge_sum_norm = self.avg_norm / math.sqrt(2.0)
        self._derivative = Derivative(self.energy, wrt=wrt)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        wrt: Sequence[str | Sequence[str]] = (),
        tp_method: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> AllegroPotential:
        """Build the model an official ``.nequip.zip`` describes and load its weights.

        The architecture comes from the package's module tree
        (:class:`~molzoo.allegro.utils.NequipPackage`); nothing upstream is
        imported. Build precision is ``dtype`` — an fp32 package loaded at
        ``torch.float64`` evaluates its fp32 weights in fp64.

        Args:
            path: The ``.nequip.zip`` file.
            wrt: As for the constructor.
            tp_method: Tensor-product kernel (``"uniform_1d"`` / ``"fused_tp"``)
                in place of the spec default — the one field that may differ
                from the package without changing the weights.
            dtype: As for the constructor.

        Returns:
            The loaded model.

        Raises:
            ValueError: On an unsupported upstream option or a key / shape
                mismatch.
        """
        state, spec = NequipPackage().read(path)
        if tp_method is not None:
            spec = spec.model_copy(update={"tp_method": tp_method})
        model = cls(spec, wrt=wrt, dtype=dtype)
        CheckpointRemap(allegro_keys).load(model, state)
        return model

    def energy(self, frame: TensorDictBase) -> TensorDictBase:
        """Write ``atoms.energy`` ``(N,)`` and ``graphs.energy`` ``(B,)`` in eV (float64), in place.

        Reads ``atoms.pos``, ``atoms.Z``, ``pairs.pair_index`` and, when
        present, the imaged ``pairs.pair_diff`` and the real-atom
        ``atoms.mask`` (ghost atoms get 0). What
        :class:`~molix.autograd.Derivative` differentiates; detaches nothing.

        Raises:
            ValueError: On the first eager call, if an atomic number lies
                outside the type table.
        """
        Z = frame["atoms", "Z"]
        self._validate_once(Z)
        num_atoms = Z.shape[0]
        types = self.node_types(Z)
        edge_index = frame["pairs", "pair_index"].contiguous()
        keys = frame.keys(include_nested=True)
        diff = frame["pairs", "pair_diff"] if ("pairs", "pair_diff") in keys else None
        features, _, lengths = self.edge_features(
            frame["atoms", "pos"], Z, edge_index, edge_diff=diff
        )
        pair_energy = self.edge_readout(features).squeeze(-1)
        per_atom = scatter_sum_compile_safe(pair_energy, edge_index[:, 0], num_atoms)
        per_atom = self.per_type_energy_scale_shift(per_atom * self.edge_sum_norm, types)
        if self.pair_potential is not None:
            per_atom = per_atom + self.pair_potential(lengths, Z, edge_index).to(per_atom.dtype)
        if ("atoms", "mask") in frame.keys(include_nested=True):
            per_atom = per_atom * frame["atoms", "mask"].to(per_atom.dtype)
        frame[ATOMIC_ENERGY_KEY] = per_atom
        frame[ENERGY_KEY] = scatter_sum_compile_safe(
            per_atom, frame["atoms", "batch"], frame["graphs"].batch_size[0]
        )
        return frame
