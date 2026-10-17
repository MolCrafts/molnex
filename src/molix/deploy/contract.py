"""The engine-neutral calling convention of a deployed atom-energy model.

A deployed model is one flat function:

    Z (N,) int64, pos (N, 3), pair_index (E, 2) int64, shifts (E, 3),
    local_mask (N,) bool
        -> energy (), forces (N, 3), atom_energy (N,), virial (3, 3)

An edge's displacement is ``pos[target] - pos[source] + shifts`` (repo-wide
convention: ``pair_index[:, 0]`` is the source). Two ways to hand over a
periodic system, one package for both:

* **Ghost copies** (a domain-decomposed engine): the caller stages its
  neighbours' atoms as extra rows, ``shifts`` is zero, and ``local_mask``
  marks the atoms it owns. The returned energy is the partition
  :math:`E^{(p)} = \\sum_i m_i E_i`; the forces are
  :math:`F_j = -\\partial E^{(p)} / \\partial r_j` for every atom, ghosts
  included — a ghost owns no atom energy, but its coordinates enter its
  neighbours' atom energies.
* **Folded images** (one process owns every atom): ``N`` is the atom count,
  each pair names the image's owner and carries the lattice translation in
  ``shifts``, and the mask is all true. Nothing is evaluated twice.

The virial is :math:`\\partial E^{(p)} / \\partial \\varepsilon` under a
homogeneous strain, :math:`\\sum_e \\operatorname{sym}(d_e \\otimes
\\partial E / \\partial d_e)` — molnex's ``graphs.virial`` sign
(:class:`molix.autograd.Derivative`); an engine converts. It is exact in both
modes because an atom's energy depends on positions only through its edges.
Nothing in the contract names an engine or a file format; an engine adapter
and a format writer each build on it. The neighbour list is handed over
unsorted.

Reference:
    Musaelian et al., "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics", Nat. Commun. 14, 579 (2023).
    https://arxiv.org/abs/2204.05249
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
from tensordict import TensorDict

from molix.autograd import leaf
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY, FORCES_KEY, PAIR_DIST_CLAMP_MIN

#: Dynamic dimensions of the five contract inputs, for ``torch.export``: the
#: atom count (at least 2) and the pair count (at least 1). One package then
#: serves every frame of a trajectory.
N_ATOMS = torch.export.Dim("n_atoms", min=2, max=1 << 20)
N_PAIRS = torch.export.Dim("n_pairs", min=1, max=1 << 24)
DYNAMIC_SHAPES = ({0: N_ATOMS}, {0: N_ATOMS}, {0: N_PAIRS}, {0: N_PAIRS}, {0: N_ATOMS})

#: Relative tolerance of the atom-energy partition checks.
PARTITION_RTOL = 1e-6
#: Relative tolerance of the force-consistency check. Looser than the energy
#: check: it compares a backward against an independently computed field.
FORCE_RTOL = 1e-5


class AtomEnergyModel(Protocol):
    """Structural type of a deployable model; nothing subclasses it.

    A module satisfies it when ``forward`` has the signature below.
    :func:`verify` checks one on an example system.
    """

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        pair_index: torch.Tensor,
        shifts: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate one neighbour list.

        Args:
            Z: Atomic numbers ``(N,)`` int64.
            pos: Positions ``(N, 3)`` in Å, model dtype.
            pair_index: Source-target pairs ``(E, 2)`` int64, unsorted.
            shifts: Lattice translation of each pair's target image ``(E, 3)``
                in Å, model dtype; zero for staged ghost copies.
            local_mask: ``(N,)`` bool. True for the atoms the caller owns.

        Returns:
            ``energy ()``, ``forces (N, 3)``, ``atom_energy (N,)``,
            ``virial (3, 3)``, with ``energy == (atom_energy * local_mask).sum()``,
            ``forces == -d(energy)/d(pos)`` and ``virial == d(energy)/d(strain)``.
        """
        ...


class AtomEnergyAdapter(nn.Module):
    """A molnex potential on the atom-energy calling convention.

    The potential is called as ``potential(batch)`` on the post-collate batch
    and must write ``atoms.energy``; the adapter replaces the potential's own
    total and forces with the partition. ``pos`` is re-leafed inside
    ``forward``, so a caller's autograd graph through ``pos`` is not continued:
    the outputs are values for an integrator. The mask is multiplied, never
    sliced — the local count is a runtime scalar a traced shape cannot depend on.

    Args:
        model: A molnex potential whose output carries ``atoms.energy``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def build_batch(
        Z: torch.Tensor, pos: torch.Tensor, pair_index: torch.Tensor, shifts: torch.Tensor
    ) -> TensorDict:
        """The one-graph post-collate batch the potential reads.

        Args:
            Z: Atomic numbers ``(N,)`` int64.
            pos: Positions ``(N, 3)``.
            pair_index: Source-target pairs ``(E, 2)`` int64, in the caller's
                order (not sorted here).
            shifts: Lattice translations ``(E, 3)``.

        Returns:
            A nested ``TensorDict`` with ``atoms`` / ``pairs`` / ``graphs``;
            ``pair_diff = pos[target] - pos[source] + shifts`` and ``pair_dist``
            clamped at :data:`molix.schema.PAIR_DIST_CLAMP_MIN`. A ``pair_diff``
            built from leaves is live, so the models read it as is
            (:func:`molrep.embedding.geometry.edge_vectors`).
        """
        n = Z.shape[0]
        pair_diff = pos[pair_index[:, 1]] - pos[pair_index[:, 0]] + shifts
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

    def model_outputs(
        self, Z: torch.Tensor, pos: torch.Tensor, pair_index: torch.Tensor, shifts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """The potential's own ``(energy (), forces or None, atom_energy (N,))``.

        What :meth:`check` compares the partition against.

        Raises:
            ValueError: If the potential does not return a ``TensorDict`` or
                writes no per-atom energy — the only source of the partition.
        """
        return self._outputs_of(self.build_batch(Z, pos, pair_index, shifts))

    def _outputs_of(
        self, batch: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        out = self.model(batch)
        if not isinstance(out, TensorDict):
            raise ValueError(f"model output must be a nested TensorDict; got {type(out).__name__}")
        atoms = out["atoms"].keys() if "atoms" in out.keys() else ()
        if "energy" not in atoms:
            raise ValueError(
                f"model output carries no per-atom energy: expected {ATOMIC_ENERGY_KEY}. "
                "A caller sums only the atom energies it owns, so a model without "
                "them cannot be partitioned."
            )
        forces = out[FORCES_KEY] if "forces" in atoms else None
        return out[ENERGY_KEY].sum(), forces, out[ATOMIC_ENERGY_KEY]

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        pair_index: torch.Tensor,
        shifts: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(energy (), forces (N, 3), atom_energy (N,), virial (3, 3))``."""
        pos = leaf(pos)
        with torch.enable_grad():
            # The shifts as a leaf of their own: dE/dshift_e is dE/dd_e.
            shifts = leaf(shifts)
            batch = self.build_batch(Z, pos, pair_index, shifts)
            diff = batch["pairs", "pair_diff"]
            _, _, atom_energy = self._outputs_of(batch)
            energy = (atom_energy * local_mask.to(atom_energy.dtype)).sum()
            grad_pos, grad_diff = torch.autograd.grad(energy, (pos, shifts))
        virial = diff.detach().transpose(0, 1) @ grad_diff
        virial = 0.5 * (virial + virial.transpose(0, 1))
        return energy.detach().reshape(()), -grad_pos, atom_energy.detach(), virial

    def check(
        self, Z: torch.Tensor, pos: torch.Tensor, pair_index: torch.Tensor, shifts: torch.Tensor
    ) -> None:
        """Refuse a potential whose energy the partition would change.

        On top of :func:`verify`: the potential's own energy must equal the sum
        of its atom energies (a long-range term with no atom attribution
        fails here), and its own forces, when it writes them, must equal the
        partition's at a full mask.

        Raises:
            ValueError: On any of those disagreements.
        """
        energy, forces, atom_energy = self.model_outputs(Z, pos, pair_index, shifts)
        _require_close(
            float(energy.detach().sum()),
            float(atom_energy.detach().sum()),
            PARTITION_RTOL,
            "model energy",
            "the sum of its atom energies",
        )
        local_mask = torch.ones(Z.shape[0], dtype=torch.bool, device=Z.device)
        _, derived, _, _ = verify(self, Z, pos, pair_index, shifts, local_mask)
        if forces is None:
            return
        reference = forces.detach()
        deviation = float((derived.detach() - reference).abs().max())
        scale = float(reference.abs().max())
        if deviation > FORCE_RTOL * max(scale, 1e-12):
            raise ValueError(
                f"model forces disagree with -d(sum of atom energies)/dpos by {deviation!r} "
                f"(max |F| {scale!r}, tolerance rel {FORCE_RTOL}). The deployed forces are "
                "derived from the atom energies, so the two must agree when every atom is local."
            )


def verify(
    model: nn.Module,
    Z: torch.Tensor,
    pos: torch.Tensor,
    pair_index: torch.Tensor,
    shifts: torch.Tensor,
    local_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ``model`` once and check it keeps the atom-energy contract.

    Args:
        model: Any module with the :class:`AtomEnergyModel` signature.
        Z: Atomic numbers ``(N,)``.
        pos: Positions ``(N, 3)``.
        pair_index: Pairs ``(E, 2)``.
        shifts: Lattice translations ``(E, 3)``.
        local_mask: ``(N,)`` bool.

    Returns:
        The model's ``(energy, forces, atom_energy, virial)``.

    Raises:
        ValueError: If it does not return four tensors of shapes ``()``,
            ``(N, 3)``, ``(N,)``, ``(3, 3)``, if the energy is not the masked
            atom-energy sum (rel :data:`PARTITION_RTOL`), or if the forces or
            the virial are not finite.
    """
    out = model(Z, pos, pair_index, shifts, local_mask)
    if not isinstance(out, tuple) or len(out) != 4:
        count = len(out) if isinstance(out, tuple | list) else None
        raise ValueError(
            "forward must return (energy, forces, atom_energy, virial), four tensors; "
            f"got {type(out).__name__}" + (f" of length {count}" if count is not None else "")
        )
    energy, forces, atom_energy, virial = out
    if not all(isinstance(tensor, torch.Tensor) for tensor in out):
        raise ValueError("forward must return (energy, forces, atom_energy, virial) as tensors")
    n = Z.shape[0]
    if energy.shape != ():
        raise ValueError(f"energy must have shape (), got {tuple(energy.shape)}")
    if forces.shape != (n, 3):
        raise ValueError(f"forces must have shape ({n}, 3), got {tuple(forces.shape)}")
    if atom_energy.shape != (n,):
        raise ValueError(f"atom_energy must have shape ({n},), got {tuple(atom_energy.shape)}")
    if virial.shape != (3, 3):
        raise ValueError(f"virial must have shape (3, 3), got {tuple(virial.shape)}")
    _require_close(
        float(energy.detach()),
        float((atom_energy.detach() * local_mask.to(atom_energy.dtype)).sum()),
        PARTITION_RTOL,
        "returned energy",
        "the masked sum of its atom energies",
    )
    if not bool(torch.isfinite(forces).all()):
        raise ValueError("forces are not finite on the example system")
    if not bool(torch.isfinite(virial).all()):
        raise ValueError("the virial is not finite on the example system")
    return energy, forces, atom_energy, virial


def example_inputs(
    species: list[int], cutoff: float, dtype: torch.dtype, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A dense example system covering every species, for tracing and checks.

    At least eight atoms: a two-atom graph specializes badly and an exported
    package then fails at a larger ``N``. The box is ``0.45 * cutoff``, so the
    cube diagonal stays inside the cutoff and every pair is an edge. The pair
    index is contiguous row-major ``(E, 2)`` — an ahead-of-time compiler fixes
    strides from the example.

    Returns:
        ``(Z, pos, pair_index, shifts, local_mask)`` with zero shifts (an open
        cluster) and an all-true mask.
    """
    n = max(8, 2 * len(species))
    generator = torch.Generator().manual_seed(0)
    box = 0.45 * cutoff
    pos = ((torch.rand(n, 3, generator=generator, dtype=torch.float64) - 0.5) * box).to(
        dtype=dtype, device=device
    )
    Z = torch.tensor([species[i % len(species)] for i in range(n)], dtype=torch.long, device=device)
    dist = torch.cdist(pos.double(), pos.double())
    # The diagonal by index: cdist's matmul path leaves it slightly above zero.
    others = ~torch.eye(n, dtype=torch.bool, device=dist.device)
    pair_index = ((dist < cutoff) & others).nonzero().to(torch.long).contiguous()
    shifts = torch.zeros(pair_index.shape[0], 3, dtype=dtype, device=device)
    return Z, pos, pair_index, shifts, torch.ones(n, dtype=torch.bool, device=device)


def _require_close(left: float, right: float, rtol: float, left_name: str, right_name: str) -> None:
    scale = max(abs(left), abs(right), 1e-12)
    if abs(left - right) > rtol * scale:
        raise ValueError(
            f"{left_name} {left!r} is not {right_name} {right!r} "
            f"(difference {abs(left - right)!r}, tolerance rel {rtol}). "
            "The deployed energy is the masked atom-energy sum, so a term with "
            "no per-atom attribution would change the energy at deployment."
        )
