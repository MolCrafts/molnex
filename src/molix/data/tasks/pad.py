"""Dense fixed-length padding for ``torch.compile`` / CUDA graphs.

Pads a collated molecular batch to a fixed atom count and edge count so every
training step sees identical tensor shapes — the precondition for CUDA-graph
capture (``torch.compile(mode="reduce-overhead")``). On small molecular batches
the per-step time is dominated by hundreds of tiny kernel *launches* (GPU
~20-30% utilised); replaying one recorded graph removes that overhead and
saturates the GPU (measured ~5x on PiNet/aspirin at batch_size=32).

Correctness contract — padded ("ghost") atoms and padding edges contribute
EXACTLY ZERO to per-graph energy and to forces on real atoms:

* A boolean ``("atoms", "mask")`` (True for real atoms) is added. The model
  multiplies it into per-atom energy *before* the per-graph scatter-sum, so
  ghost atoms add 0 to whichever graph they are assigned to.
* Every padding edge is a self-loop on the FIRST ghost atom, so no padding edge
  ever gathers from / scatters to a real atom. Real atoms' features — and hence
  their energy and forces — are therefore identical to the unpadded batch, and
  forces on ghost atoms are 0 (total energy does not depend on ghost positions).

The graph count is NOT padded; pair this with ``drop_last=True`` so every batch
has the same number of graphs (the third static dimension CUDA graphs needs).

Overflow raises rather than truncating — a batch larger than the configured cap
is a misconfiguration, never silently dropped.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molix.data.task import BatchTask


class PadMolecularBatch(BatchTask):
    """Pad a collated batch to ``(max_atoms, max_edges)`` fixed shapes.

    Args:
        max_atoms: Fixed atom-dimension length. Must exceed the largest real
            atom count in any batch (at least one ghost atom is required to
            anchor padding edges).
        max_edges: Fixed edge-dimension length. Must be >= the largest real
            edge count in any batch.
        pad_atom_type: Atomic number assigned to ghost atoms. Must be a valid
            embedding index for the model (default 1 = H). Ghost energy is
            masked to 0 regardless, so the value only needs to be embeddable.
    """

    def __init__(self, *, max_atoms: int, max_edges: int, pad_atom_type: int = 1) -> None:
        self.max_atoms = int(max_atoms)
        self.max_edges = int(max_edges)
        self.pad_atom_type = int(pad_atom_type)

    @property
    def task_id(self) -> str:
        return f"PadMolecularBatch(a={self.max_atoms},e={self.max_edges})"

    def execute(self, data: dict) -> dict:
        atoms = data["atoms"]
        edges = data["edges"]
        n = int(atoms.batch_size[0])
        e = int(edges.batch_size[0])
        if n >= self.max_atoms:
            raise ValueError(
                f"PadMolecularBatch: batch has {n} atoms but max_atoms={self.max_atoms}; "
                "raise max_atoms (need >= max real atoms + 1 ghost slot)."
            )
        if e > self.max_edges:
            raise ValueError(
                f"PadMolecularBatch: batch has {e} edges but max_edges={self.max_edges}; "
                "raise max_edges."
            )

        dev = atoms["Z"].device
        n_pad = self.max_atoms - n
        e_pad = self.max_edges - e
        ghost0 = n  # index of the first ghost atom; padding edges self-loop here

        # --- atoms: append ghost rows (Z = valid type, everything else 0) ---
        new_atoms: dict[str, torch.Tensor] = {}
        for k, t in atoms.items():
            pad = t.new_zeros((n_pad, *t.shape[1:]))
            if k == "Z":
                pad = pad + self.pad_atom_type
            # "batch" pads to 0 -> ghosts nominally belong to graph 0, but their
            # energy is masked to 0 so they add nothing to it. "pos" and atom
            # targets (e.g. forces) pad to 0.
            new_atoms[k] = torch.cat([t, pad], dim=0)
        mask = torch.zeros(self.max_atoms, dtype=torch.bool, device=dev)
        mask[:n] = True
        new_atoms["mask"] = mask
        data["atoms"] = TensorDict(new_atoms, batch_size=[self.max_atoms], device=atoms.device)

        # --- edges: padding edges are (ghost0, ghost0) self-loops ---
        new_edges: dict[str, torch.Tensor] = {}
        for k, t in edges.items():
            if k == "edge_index":
                pad = torch.full((e_pad, *t.shape[1:]), ghost0, dtype=t.dtype, device=dev)
            else:
                pad = t.new_zeros((e_pad, *t.shape[1:]))
            new_edges[k] = torch.cat([t, pad], dim=0)
        data["edges"] = TensorDict(new_edges, batch_size=[self.max_edges], device=edges.device)

        return data
