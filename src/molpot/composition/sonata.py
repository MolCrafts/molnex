"""Sonata — the static permanent-electrostatics potential.

Sonata is a sibling (not subclass) of :class:`PotentialComposer`, specialised
for the multipole-Ewald path, on the molnex potential contract
(:class:`molix.autograd.Potential`). It owns an Allegro-style encoder built
with ``expose_tensor_track=True``, a :class:`PermMultipoleHead` predicting
``(q, μ, Θ)`` and an :class:`EwaldMultipoleEnergy` evaluator, plus any
short-range energy heads. Two public methods, as
:mod:`molix.autograd.potential` prescribes:

``energy(frame) -> frame``
    encoder → multipole head → Ewald → short-range heads, in place on the
    frame. Reads ``atoms.{pos,batch}``, ``pairs.pair_index`` and the imaged
    ``pairs.pair_diff`` when present, ``graphs.cell`` when present (the
    reciprocal path) and ``graphs.total_charge`` when the head projects
    charges. Writes ``graphs.energy = energy_short + energy_es``,
    ``graphs.energy_short``, ``graphs.energy_es``, ``atoms.phi``,
    ``atoms.field``, and whatever the heads write (``atoms.atomic_charges``
    / ``atomic_dipoles`` / ``atomic_quadrupoles``,
    ``graphs.molecular_dipole``, the charge-sum diagnostics).

``forward(frame) -> frame``
    The :class:`~molix.autograd.Derivative` built once in ``__init__`` from
    the ``wrt`` keys: ``("atoms.pos",)`` adds ``atoms.forces`` ``(N, 3)``,
    ``("atoms.pos", "graphs.strain")`` adds ``graphs.virial`` ``(1, 3, 3)`` on
    a one-graph periodic batch (``W = ∂E/∂ε``, Cauchy stress ``σ = W / V``).

One eager precondition on a periodic run: :meth:`Sonata.bind_cell` freezes
the reciprocal integer grid on the lattice, like
:meth:`molzoo.mace.polar.MacePolar.bind_cell`. Unbound, the Ewald term
enumerates the grid from each frame's cell on every call.

What this is NOT
    * **Not a composer for induced-response models.** The forthcoming
      ``LesPolarizable`` line wraps :class:`EwaldMultipoleEnergy` with
      ``kappa_head`` / ``alpha_head`` to add inline non-self-consistent
      linear response. Mixing those with the self-consistent CG Thole solve
      in :class:`molpot.potentials.Polarization` would double-count
      induction, so a :class:`Polarization` among the short-range heads is
      refused at construction.
    * **Not invoking** :class:`PotentialComposer` — the latter's
      ``mixing_fn(atom_params, pair_index)`` per-pair contract does not
      match the global multipole-Ewald signature ``(q, pos, cell, batch,
      mu, Q)``.
    * **Not for QEq / CELLI.** Charges come from the direct l=0 readout
      of :class:`PermMultipoleHead`, not from a charge-equilibration
      KKT solve.

References:
    * Cheng B., *Latent Ewald summation for machine-learning potentials*,
      npj Comput. Mater. **11**, 80 (2025).
      https://doi.org/10.1038/s41524-025-01577-7
    * King D. S. et al., *Latent equivariant ML force fields with
      long-range electrostatics*, Nat. Commun. **16**, 8763 (2025).
      https://doi.org/10.1038/s41467-025-63852-x
    * Aguado A. & Madden P. A., *Ewald summation of electrostatic
      multipole interactions up to the quadrupolar level*,
      J. Chem. Phys. **119**, 7471 (2003).
      https://doi.org/10.1063/1.1605941
    * Stone A. J., *The Theory of Intermolecular Forces*, 2nd ed.
      (Oxford, 2013), §3.
    * Upstream LES code: https://github.com/ChengUCB/les
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDictBase

import molrep.eq as meq
from molix.autograd import Derivative, Potential
from molix.schema import ENERGY_KEY
from molpot.heads.multipole import PermMultipoleHead, PermMultipoleHeadSpec
from molpot.potentials.elec.ewald_multipole import (
    EwaldMultipoleEnergy,
    EwaldMultipoleEnergySpec,
)
from molpot.potentials.polarization import Polarization

__all__ = ["Sonata", "SonataSpec", "TensorTrackEncoder"]

_POLARIZATION_REFUSE_MSG = (
    "Sonata refuses `Polarization` among its short-range heads: it would "
    "double-count induction against any future inline LES α-mode. "
    "Polarization belongs to the `LesPolarizable` line; use an "
    "`EdgeEnergyHead` (or any head that writes `graphs.energy_short`) here."
)


class TensorTrackEncoder(Protocol):
    """What Sonata reads off its encoder: an Allegro body exposing its tensor track.

    Attributes:
        expose_tensor_track: Whether the body writes the equivariant
            ``pairs.edge_tensor_features`` the multipole head reads.
        l_max: Highest spherical-harmonic degree of that track; ``≥ 2`` for
            dipoles and quadrupoles.
        output_dim: Width of the scalar ``pairs.edge_features`` the heads read.
        tensor_track_irreps: Irreps of the tensor track, the wiring of the
            equivariant moment readouts.
    """

    expose_tensor_track: bool
    l_max: int
    output_dim: int
    tensor_track_irreps: meq.Irreps

    def __call__(self, frame: TensorDictBase) -> TensorDictBase:
        """Write the edge features onto ``frame`` and return it."""
        ...


def _theta_to_cartesian_quadrupole(theta: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, 5)`` ℓ=2 real-spherical components to ``(N, 3, 3)``.

    The result is a symmetric, traceless 3×3 tensor that is rotation-
    equivariant with the input under cuequivariance's ``D⁽²⁾`` Wigner
    matrix: ``M(D⁽²⁾(R) · θ) = R · M(θ) · Rᵀ``.

    The cuequivariance ``2e`` real-spherical-harmonic convention
    (extracted from ``cuequivariance.group_theory.descriptors.
    spherical_harmonics_.sympy_spherical_harmonics``) maps the five
    components in ``ir_mul`` layout to these polynomial bases:

    * index 0 (m=-2): ``√15 · xz``
    * index 1 (m=-1): ``√15 · xy``
    * index 2 (m=0):  ``(√5/2) · (-x² + 2y² - z²)``
    * index 3 (m=+1): ``√15 · yz``
    * index 4 (m=+2): ``(√15/2) · (z² - x²)``

    Note that index 0 is ``xz`` (not ``xy``) and the m=0 diagonal is
    ``diag(-1, 2, -1)`` (not ``diag(-1, -1, 2)``) — both depart from
    the older "physics convention" that is wrong for cuequivariance.

    The basis tensors below are orthonormal in the Frobenius inner
    product (any consistent global scaling preserves the equivariance).

    Args:
        theta: ``(N, 5)`` spherical components in cuequivariance's
            ``2e`` ir_mul layout (``mul=1``, last axis = component).

    Returns:
        ``(N, 3, 3)`` symmetric traceless Cartesian quadrupole tensor.
    """
    s2 = 0.5**0.5
    s6 = 6.0**-0.5
    n = theta.shape[0]
    q = torch.zeros(n, 3, 3, dtype=theta.dtype, device=theta.device)
    # index 0 (m=-2 ↔ xz): xz off-diagonal
    q[:, 0, 2] = q[:, 2, 0] = theta[:, 0] * s2
    # index 1 (m=-1 ↔ xy): xy off-diagonal
    q[:, 0, 1] = q[:, 1, 0] = theta[:, 1] * s2
    # index 2 (m=0 ↔ -x²+2y²-z²): diag = (-1, 2, -1) / √6
    q[:, 0, 0] = -theta[:, 2] * s6
    q[:, 1, 1] = 2.0 * theta[:, 2] * s6
    q[:, 2, 2] = -theta[:, 2] * s6
    # index 3 (m=+1 ↔ yz): yz off-diagonal
    q[:, 1, 2] = q[:, 2, 1] = theta[:, 3] * s2
    # index 4 (m=+2 ↔ z²-x²): diag = (-1, 0, 1) / √2  (added on top of m=0)
    q[:, 0, 0] = q[:, 0, 0] - theta[:, 4] * s2
    q[:, 2, 2] = q[:, 2, 2] + theta[:, 4] * s2
    return q


class SonataSpec(BaseModel):
    """Frozen Pydantic snapshot of a :class:`Sonata`: the head and the Ewald term.

    Mirrors the :class:`PermMultipoleHeadSpec` /
    :class:`EwaldMultipoleEnergySpec` pattern: every hyperparameter is
    captured here so a trained checkpoint carries an exact, validated
    description of the model it was built with. The encoder and the
    short-range heads are wiring (they depend on the dataset and the
    training script) and are handed to the constructor as modules.

    Attributes:
        head: The :class:`PermMultipoleHeadSpec` for the multipole readout;
            its ``input_dim`` must equal the encoder's ``output_dim``.
        ewald: The :class:`EwaldMultipoleEnergySpec` for the σ-screened
            multipole-Coulomb evaluator.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    head: PermMultipoleHeadSpec
    ewald: EwaldMultipoleEnergySpec


class Sonata(Potential):
    """Static permanent-electrostatics potential on the two-method contract.

    Args:
        spec: Head and Ewald configuration (:class:`SonataSpec`).
        encoder: Allegro-style body built with ``expose_tensor_track=True``
            and ``l_max ≥ 2`` when the head predicts dipoles or
            quadrupoles; its ``tensor_track_irreps`` wire the equivariant
            moment readouts.
        short_range_heads: Modules that each consume the frame and write
            ``graphs.energy_short`` ``(B,)``; their contributions are summed.
            A :class:`Polarization` among them is refused — it would
            double-count induction against a future inline LES α-mode.
        wrt: What :meth:`forward` derives beside the energy, as TensorFrame
            keys (:class:`molix.autograd.Derivative`): ``("atoms.pos",)``
            (the default) adds ``atoms.forces``, ``("atoms.pos",
            "graphs.strain")`` adds ``graphs.virial`` on a one-graph
            periodic batch, ``()`` writes the energy alone.
        dtype: Floating dtype the multipole head is built at. The encoder
            and the short-range heads arrive built.

    Raises:
        ValueError: The encoder does not expose its tensor track; the head
            predicts dipoles or quadrupoles on an encoder with ``l_max < 2``;
            ``spec.head.input_dim`` differs from ``encoder.output_dim``; a
            short-range head is a :class:`Polarization`.
    """

    _kvec_indices: torch.Tensor | None

    def __init__(
        self,
        spec: SonataSpec,
        encoder: TensorTrackEncoder,
        *,
        short_range_heads: Sequence[nn.Module] = (),
        wrt: Sequence[str] = ("atoms.pos",),
        dtype: torch.dtype,
    ) -> None:
        if not encoder.expose_tensor_track:
            raise ValueError(
                "Sonata requires an encoder built with `expose_tensor_track=True` "
                "to expose the equivariant tensor-track features the multipole "
                "head consumes. Reconstruct your encoder with this flag set."
            )
        if (spec.head.dipole or spec.head.quadrupole) and encoder.l_max < 2:
            raise ValueError(
                "Sonata requires `encoder.l_max >= 2` when the head predicts dipoles "
                f"or quadrupoles; got `l_max={encoder.l_max}`."
            )
        if spec.head.input_dim != encoder.output_dim:
            raise ValueError(
                f"spec.head.input_dim={spec.head.input_dim} must equal the encoder's "
                f"output_dim={encoder.output_dim}: the head reads the encoder's edge features."
            )
        if any(isinstance(head, Polarization) for head in short_range_heads):
            raise ValueError(_POLARIZATION_REFUSE_MSG)

        super().__init__()
        self.spec = spec
        self.encoder = encoder
        self.perm_multipole_head = PermMultipoleHead.from_spec(
            spec.head, tensor_irreps=encoder.tensor_track_irreps, dtype=dtype
        )
        self.ewald = EwaldMultipoleEnergy.from_spec(spec.ewald)
        self.short_range_heads = nn.ModuleList(short_range_heads)
        # The frozen reciprocal grid of ``bind_cell``; follows ``.to(device)``
        # and stays out of the checkpoint — it is a property of the lattice.
        self.register_buffer("_kvec_indices", None, persistent=False)
        self._derivative = Derivative(self.energy, wrt=wrt)

    def bind_cell(self, cell: torch.Tensor) -> None:
        """Freeze the reciprocal integer grid on ``cell``. Eager only.

        A precondition of the periodic path, like the element-table check of
        an encoder: enumerating the keep-set ``0 < k² ≤ k_max²`` reads the
        lattice norms back to the host, which a traced call cannot do, and
        the cutoff is a step discontinuity in the cell — differentiating the
        energy against a strain is only smooth on a grid fixed before the
        cell moves. Every later :meth:`energy` uses this grid, whatever cell
        the frame carries; calling again replaces it.

        Args:
            cell: Lattice vectors ``(3, 3)`` or ``(1, 3, 3)`` in Å, rows being
                the lattice vectors.

        Raises:
            ValueError: ``cell`` is neither ``(3, 3)`` nor ``(1, 3, 3)``.
        """
        if cell.shape == (1, 3, 3):
            cell = cell[0]
        if cell.shape != (3, 3):
            raise ValueError(f"cell must be (3, 3) or (1, 3, 3); got {tuple(cell.shape)}")
        self._kvec_indices = self.ewald.enumerate_kvec_indices(cell)

    def energy(self, frame: TensorDictBase) -> TensorDictBase:
        """Write ``graphs.energy`` ``(B,)`` and the electrostatic columns in place.

        Args:
            frame: Post-collate batch carrying ``atoms.{Z,pos,batch}``,
                ``pairs.pair_index`` and the ``graphs`` namespace at
                ``batch_size=[B]``; ``pairs.pair_diff`` (the imaged
                displacement) and ``graphs.cell`` ``(B, 3, 3)`` or ``(3, 3)``
                when periodic; ``graphs.total_charge`` ``(B,)`` when the head
                projects the charge sums.

        Returns:
            The same ``frame`` with ``graphs.energy``, ``graphs.energy_short``,
            ``graphs.energy_es``, ``atoms.phi`` ``(N,)``, ``atoms.field``
            ``(N, 3)`` and the heads' own columns written, attached to
            whatever the caller's leaves are.
        """
        self.encoder(frame)
        head = self.perm_multipole_head
        head(frame)
        keys = frame.keys(include_nested=True)
        theta = frame["atoms", head.out_quadrupole_key] if head.quadrupole else None
        # Θ lives in the (N, 5) real-spherical 2e basis (cuequivariance
        # output); the Ewald term takes Q in (N, 3, 3) symmetric traceless
        # Cartesian form. The basis swap is this potential's, so neither
        # the head nor the term knows about it.
        ewald_out = self.ewald(
            q=frame["atoms", head.out_charge_key],
            pos=frame["atoms", "pos"],
            cell=frame["graphs", "cell"] if ("graphs", "cell") in keys else None,
            batch=frame["atoms", "batch"],
            mu=frame["atoms", head.out_dipole_key] if head.dipole else None,
            Q=None if theta is None else _theta_to_cartesian_quadrupole(theta),
            kvec_indices=self._kvec_indices,
        )
        energy_es = ewald_out["pot"]
        energy_short = torch.zeros_like(energy_es)
        for short_range_head in self.short_range_heads:
            short_range_head(frame)
            energy_short = energy_short + frame["graphs", "energy_short"]
        frame["graphs", "energy_short"] = energy_short
        frame["graphs", "energy_es"] = energy_es
        frame[ENERGY_KEY] = energy_short + energy_es
        frame["atoms", "phi"] = ewald_out["phi"]
        frame["atoms", "field"] = ewald_out["field"]
        return frame
