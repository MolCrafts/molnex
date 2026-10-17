"""PiNet2 energy model: the feature body plus PiNN's per-block energy readout.

:class:`PiNetPotential` is a :class:`~molix.autograd.Potential` over the
:class:`~molzoo.pinet.encoder.PiNetEncoder` body and registers its heads at
the module root (``out_layers``, ``atomic_energies``), so its ``state_dict``
is flat. It mirrors PiNN's ``pot_model`` over ``PiNet2``:

.. math::

    E_g = s \\sum_{i \\in g} m_i\\, o_{\\mathrm{depth}, i}
        + \\sum_{i \\in g} m_i\\, e_0(Z_i),
    \\qquad o_t = W_t\\, \\mathrm{FF}_t(p^1_t) + o_{t-1}

with ``s = 1 / e_scale`` and ``e_0`` PiNN's ``e_dress`` (spec
``src/molzoo/specs/pinet2.md`` §5).

Two public methods, the potential contract: :meth:`PiNetPotential.energy`
reads the frame's columns and writes ``atoms.energy`` and ``graphs.energy``;
``forward`` is the one :class:`molix.autograd.Derivative` over ``energy``,
fixed by ``wrt=`` at construction.

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems" J. Chem. Theory Comput. 21, 1382 (2025).
    https://doi.org/10.1021/acs.jctc.4c01570
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from tensordict import TensorDictBase

from molix.autograd import Derivative, Potential
from molix.F.scatter import scatter_sum_compile_safe
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY
from molpot.heads.energy import AtomicReferenceEnergy
from molzoo.pinet.encoder import PiNetEncoder
from molzoo.pinet.layers import OutLayer
from molzoo.pinet.spec import PiNetSpec


class PiNetPotential(Potential, PiNetEncoder):
    """PiNet2 energy (+ derivatives fixed by ``wrt``), written into the frame.

    ::

        model = PiNetPotential(
            PiNetSpec(atomic_numbers=[1, 6, 8]), wrt=("atoms.pos",), dtype=torch.float64
        )
        frame = model(frame)  # graphs.energy, atoms.energy, atoms.forces

    Args:
        spec: The configuration; the head fields are ``out_nodes`` /
            ``out_activation`` (the ``OutLayer`` readout), ``atomic_energies``
            (PiNN ``e_dress``, table order, eV; ``None`` registers no reference
            head) and ``scale`` (PiNN ``1 / e_scale``).
        wrt: TensorFrame keys :meth:`forward` differentiates the energy
            against (``"atoms.pos"`` → forces, ``"graphs.strain"`` → virial);
            ``()`` is energy only.
        dtype: Floating dtype every parameter and buffer is built at.
    """

    def __init__(
        self,
        spec: PiNetSpec = PiNetSpec(),
        *,
        wrt: Sequence[str | Sequence[str]] = (),
        dtype: torch.dtype,
    ) -> None:
        super().__init__(spec, dtype=dtype)
        self.out_layers = nn.ModuleList(
            OutLayer(
                spec.out_nodes,
                in_dim=self.feature_dim,
                activation=spec.out_activation,
                dtype=self.dtype,
            )
            for _ in range(spec.depth)
        )
        self.atomic_energies = (
            AtomicReferenceEnergy(
                atomic_energies=spec.atomic_energies, atomic_numbers=spec.atomic_numbers
            )
            if spec.atomic_energies is not None
            else None
        )
        self.scale = spec.scale
        self._derivative = Derivative(self.energy, wrt=wrt)

    def energy(self, frame: TensorDictBase) -> TensorDictBase:
        """Write ``atoms.energy`` ``(N,)`` and ``graphs.energy`` ``(B,)`` in eV, in place.

        Reads ``atoms.pos``, ``atoms.Z``, ``pairs.pair_index`` and, when
        present, the imaged ``pairs.pair_diff`` and the real-atom
        ``atoms.mask`` (ghost atoms get 0). What
        :class:`~molix.autograd.Derivative` differentiates; detaches nothing.

        Raises:
            ValueError: On the first eager call, if an atomic number lies
                outside the element table.
        """
        layers, _, _ = self._batch_layers(frame)
        positions = frame["atoms", "pos"]
        output = positions.new_zeros(positions.shape[0], 1)
        for out_layer, layer in zip(self.out_layers, layers):
            output = out_layer(layer.block.p1, output)
        per_atom = self.scale * output.squeeze(-1)
        if self.atomic_energies is not None:
            per_atom = per_atom + self.atomic_energies(frame["atoms", "Z"])
        if ("atoms", "mask") in frame.keys(include_nested=True):
            per_atom = per_atom * frame["atoms", "mask"].to(per_atom.dtype)
        frame[ATOMIC_ENERGY_KEY] = per_atom
        frame[ENERGY_KEY] = scatter_sum_compile_safe(
            per_atom, frame["atoms", "batch"], frame["graphs"].batch_size[0]
        )
        return frame
