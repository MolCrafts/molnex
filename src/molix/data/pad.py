"""Dense fixed-length padding for ``torch.compile`` / CUDA graphs.

Pads a collated molecular batch to a fixed atom count, edge count, valence-term
count and — when ``max_graphs`` is set — graph count, so every training step
sees identical tensor shapes: the precondition for one dynamo graph and for
CUDA-graph capture (``torch.compile(mode="reduce-overhead")``). On small
molecular batches the per-step time is dominated by hundreds of tiny kernel
*launches* (GPU ~20-30% utilised); replaying one recorded graph removes that
overhead and saturates the GPU (measured ~5x on PiNet/aspirin at
batch_size=32).

Correctness contract — padded ("ghost") atoms, padding edges and dead graph
slots contribute EXACTLY ZERO to per-graph energy and to forces on real atoms:

* A boolean ``("atoms", "mask")`` (True for real atoms) is added. The model
  multiplies it into per-atom energy *before* the per-graph scatter-sum, so
  ghost atoms add 0 to whichever graph they are assigned to. With
  ``max_graphs`` set they are assigned to the **sink** — the last graph slot,
  which is dead by construction — so a per-atom head that forgets the mask
  pollutes a slot the loss weights out, never a real molecule.
* Dummy valence rows (``bonds`` / ``angles`` / ``propers`` / ``impropers``)
  index four offset ghost atoms and carry ``mask=False``; the potential
  multiplies predicted ``k`` by that mask so dummy terms add 0.
* Every padding edge is a self-loop on the FIRST ghost atom, so no padding edge
  ever gathers from / scatters to a real atom, **and it is dead**: its
  ``pair_diff`` (and ``shifts``, when the batch carries them) is the schema's
  dead-edge vector ``[dead_pair_length(cutoff), 0, 0]`` and its ``pair_dist``
  that length, so every cutoff envelope is exactly zero on it and no ``1/r``
  term sees ``r = 0`` — the convention the MD neighbour list and the static
  export forward pad with (:func:`molix.schema.dead_pair_shift`). Real atoms'
  features — and hence their energy and forces — are therefore identical to
  the unpadded batch, and forces on ghost atoms are 0 (total energy does not
  depend on ghost positions).
* Dead graph slots carry ``num_atoms = 0``, zeros in every other ``graphs``
  column, and ``("graphs", "mask") = False``; losses and metrics weight by
  that mask (``molix.core.losses.energy_loss(weight_key="mask")``).

**Capacity-first packing.** The padder is the one owner of the static shape.
:meth:`MoleculePadder.budgets` hands the same caps to
:class:`~molix.data.sampler.TokenBudgetPacker` as per-batch budgets — atoms,
edges, every padded valence family, and the graph count — so no packed batch
can exceed any cap: overflow is impossible by construction, not improbable.
:meth:`MoleculePadder.from_counts` derives the caps from a dataset's
per-molecule counts (each cap is the dataset density times the atom budget,
and never below the largest single molecule), and :meth:`MoleculePadder.check`
refuses a count table with a molecule that could not be packed alone. Fixing
the molecule count per batch instead would force every cap up to the sum of
the largest molecules (OMol25 4m: 17 M proper rows at 2048 molecules, 50x a
random batch) — the waste is structural to fixing ``B``, not a tuning problem.

Overflow raises rather than truncating — a batch larger than the configured cap
is a misconfiguration, never silently dropped.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from tensordict import TensorDict

from molix.schema import dead_pair_length, dead_pair_shift

#: Dummy valence needs four distinct ghosts with finite geometry. A
#: ghost–ghost bond at the same position has ``r = 0``; autograd of
#: ``‖x‖`` at the origin is NaN even when ``k = 0``.
N_GHOSTS = 4
_VALENCE = ("bonds", "angles", "propers", "impropers", "hessian_pairs")
_INDEX_GHOST = {"atomi": 0, "atomj": 1, "atomk": 2, "atoml": 3}
#: Count tables :meth:`MoleculePadder.from_counts` understands, in the order
#: :meth:`MoleculePadder.caps` reports them.
_TABLES = ("atoms", "pairs", *_VALENCE, "graphs")


def _live_count(block: TensorDict) -> int:
    """Real term count of a valence namespace before padding."""
    if "bond_index" in block.keys():
        return int(block["bond_index"].shape[1])
    if "atomi" in block.keys():
        return int(block["atomi"].shape[0])
    if "mask" in block.keys():
        return int(block["mask"].shape[0])
    if block.batch_size:
        return int(block.batch_size[0])
    return 0


def _pad_valence_tensor(
    key: str, tensor: torch.Tensor, n_pad: int, ghosts: torch.Tensor
) -> torch.Tensor:
    """Append dummy rows/columns. Index columns point at distinct ghosts."""
    if n_pad == 0:
        return tensor
    if key == "bond_index":
        arity = int(tensor.shape[0])
        pad = ghosts[:arity].to(dtype=tensor.dtype).unsqueeze(1).expand(arity, n_pad)
        return torch.cat([tensor, pad], dim=1)
    if key in _INDEX_GHOST:
        ghost = ghosts[min(_INDEX_GHOST[key], int(ghosts.numel()) - 1)]
        pad = tensor.new_full((n_pad, *tensor.shape[1:]), int(ghost))
        return torch.cat([tensor, pad], dim=0)
    pad = tensor.new_zeros((n_pad, *tensor.shape[1:]))
    return torch.cat([tensor, pad], dim=0)


def _as_long_vector(name: str, values: Any) -> torch.Tensor:
    """One per-molecule count table as a ``(n,)`` long tensor."""
    vector = torch.as_tensor(values).reshape(-1).to(torch.long)
    if vector.numel() == 0:
        raise ValueError(f"MoleculePadder: count table {name!r} is empty")
    if bool((vector < 0).any()):
        raise ValueError(f"MoleculePadder: count table {name!r} has a negative entry")
    return vector


def _round_up(value: float, multiple: int) -> int:
    """Smallest multiple of *multiple* that is ``>= value``."""
    return int(math.ceil(value / multiple)) * multiple


class MoleculePadder:
    """Pad a collated batch to fixed atom, edge, valence and graph shapes.

    Not a :class:`~molix.data.task.BatchTask`. Compose with
    :class:`~molix.data.FrameCollator` as the DataLoader
    ``collate_fn``. Dummy valence terms index the four ghost atoms
    (finite dummy geometry); ``*.mask`` is False on those rows so
    :class:`~molpot.forcefield.potentials.Potentials` zeros their ``k``.

    The caps double as the per-batch budgets of the packer that composes
    the batches (:meth:`budgets`), which is what makes a single static
    shape sufficient for a whole dataset: build the padder from the
    dataset's per-molecule counts with :meth:`from_counts`, hand
    :meth:`budgets` to :class:`~molix.data.sampler.TokenBudgetBatchSampler`
    (:class:`~molix.data.DataModule` does so under ``batch_counts``),
    and every batch fits.

    Args:
        max_atoms: Fixed atom-dimension length. Need one ghost for dead
            edges, four when any valence family is padded.
        max_pairs: Fixed edge-dimension length. Must be >= the largest real
            edge count in any batch.
        cutoff: Interaction cutoff ``r_cut`` in Å of the model the batch feeds;
            padding edges are displaced by :func:`molix.schema.dead_pair_shift`
            of it, which is what makes them dead.
        pad_atom_type: Atomic number assigned to ghost atoms. Must be a valid
            embedding index for the model (default 1 = H). Ghost energy is
            masked to 0 regardless, so the value only needs to be embeddable.
        max_bonds: Cap for ``bonds.bond_index`` count axis. Default
            ``2 * max_atoms``. Independent of the atom cap — dummy ``atomi``
            is a ghost index, never ``max_atoms``. ``0`` leaves the
            namespace untouched.
        max_angles: Cap for ``angles`` rows. Default ``4 * max_atoms``.
        max_propers: Cap for ``propers`` rows. Default ``8 * max_atoms``
            (OMol25 8-mol batches hit ~2.2 propers/atom).
        max_impropers: Cap for ``impropers`` rows. Default ``2 * max_atoms``.
        max_hessian_pairs: Cap for ``hessian_pairs`` rows — the atom pairs a
            bonded term couples, carrying the stage-1 partial-Hessian labels.
            Default ``0`` (the namespace is not padded), because only the
            partial-Hessian stage has one. An exact upper bound is
            ``2 * (max_bonds + max_angles)``: the pairs of a proper or an
            improper are 1-2 and 1-3 pairs, which bonds and angles already
            enumerate.
        max_graphs: Fixed graph-slot count, or ``None`` to leave the graph
            count alone (a fixed-``batch_size`` loader with ``drop_last``
            then supplies the static graph dimension). Live graphs occupy
            ``[0, B)``; the last slot is the ghost atoms' sink, so a batch
            needs ``B <= max_graphs - 1``.
    """

    def __init__(
        self,
        *,
        max_atoms: int,
        max_pairs: int,
        cutoff: float,
        pad_atom_type: int = 1,
        max_bonds: int | None = None,
        max_angles: int | None = None,
        max_propers: int | None = None,
        max_impropers: int | None = None,
        max_hessian_pairs: int | None = None,
        max_graphs: int | None = None,
    ) -> None:
        self.max_atoms = int(max_atoms)
        self.max_pairs = int(max_pairs)
        self.cutoff = float(cutoff)
        self.pad_atom_type = int(pad_atom_type)
        self.max_bonds = int(2 * max_atoms if max_bonds is None else max_bonds)
        self.max_angles = int(4 * max_atoms if max_angles is None else max_angles)
        self.max_propers = int(8 * max_atoms if max_propers is None else max_propers)
        self.max_impropers = int(2 * max_atoms if max_impropers is None else max_impropers)
        # Unlike the valence families there is no default: the partial-Hessian
        # pair list exists only in stage 1, and a non-zero default would add a
        # key to caps() — the calibration fingerprint — for every batch that has
        # no such namespace.
        self.max_hessian_pairs = int(0 if max_hessian_pairs is None else max_hessian_pairs)
        if max_graphs is not None and int(max_graphs) < 2:
            raise ValueError(
                f"MoleculePadder: max_graphs={max_graphs} leaves no live slot beside the "
                "sink; need >= 2 (live graphs + one sink slot)."
            )
        self.max_graphs = None if max_graphs is None else int(max_graphs)

    @classmethod
    def from_counts(
        cls,
        counts: Mapping[str, Any],
        *,
        atoms: int,
        cutoff: float,
        slack: float = 1.10,
        graphs_slack: float = 1.25,
        round_to: int = 256,
        pad_atom_type: int = 1,
    ) -> MoleculePadder:
        """Caps sized from a dataset's per-molecule counts at atom capacity *atoms*.

        The atom dimension is the calibration knob (a power of two picked by
        the HBM sweep); every other cap follows the dataset's own density:
        ``cap_c = max(round_up(Σc / Σatoms × (atoms − N_GHOSTS) × slack),
        max_c)``. The ``max_c`` floor is what makes the largest single
        molecule packable on its own; the density term is what keeps a
        random batch from closing on a cheap column before the atom column
        is full. The graph-slot count is ``ceil(graphs_slack × (atoms −
        N_GHOSTS) / mean_atoms) + 1`` (one sink slot). Quantiles are never
        used: a cap below the maximum is a batch that cannot be packed. With a
        ``graphs`` vector the mean is per graph rather than per row, and the
        slot count never falls below one row's graphs plus the sink.

        Args:
            counts: Per-**row** count tables keyed by namespace — ``atoms``
                and ``pairs`` are required, ``bonds`` / ``angles`` /
                ``propers`` / ``impropers`` optional; a family absent here is
                neither padded nor budgeted. ``graphs`` is different in kind: it
                is not a padded axis but how many graphs each row contributes,
                and it is what sizes the graph-slot count. Omit it and a row is
                taken to be one graph, which is true of every per-molecule
                dataset and false of a record dataset (one THEMol torsion-scan
                row is 24 conformers). Each value is a ``(n,)`` integer
                tensor / array aligned with the dataset index.
            atoms: The padded atom dimension (ghost slots included). Must
                hold the largest molecule plus :data:`N_GHOSTS`.
            cutoff: Interaction cutoff, as for the constructor.
            slack: Head-room factor on every density-derived cap.
            graphs_slack: Head-room factor on the graph-slot count.
            round_to: Every density-derived cap is rounded up to a multiple
                of this.
            pad_atom_type: As for the constructor.

        Returns:
            A padder whose :meth:`budgets` keep every packed batch inside
            its caps.

        Raises:
            ValueError: A required table is missing, a table name is unknown,
                *atoms* cannot hold the largest molecule with its ghosts, or
                a table is empty / negative.
        """
        unknown = sorted(set(counts) - set(_TABLES))
        if unknown:
            raise ValueError(
                f"MoleculePadder.from_counts: unknown count tables {unknown}; "
                f"expected a subset of {list(_TABLES)}"
            )
        for required in ("atoms", "pairs"):
            if required not in counts:
                raise ValueError(
                    f"MoleculePadder.from_counts: counts must carry {required!r} — the "
                    f"{required} cap cannot be sized without per-molecule {required} counts"
                )
        if slack < 1.0 or graphs_slack < 1.0:
            raise ValueError("MoleculePadder.from_counts: slack factors must be >= 1.0")
        if round_to < 1:
            raise ValueError("MoleculePadder.from_counts: round_to must be >= 1")
        vectors = {name: _as_long_vector(name, values) for name, values in counts.items()}
        n_atoms = vectors["atoms"]
        n_graphs = vectors.get("graphs")
        max_atoms = int(atoms)
        atom_budget = max_atoms - N_GHOSTS
        largest = int(n_atoms.max())
        if atom_budget < largest:
            raise ValueError(
                f"MoleculePadder.from_counts: atoms={max_atoms} cannot hold the largest "
                f"molecule ({largest} atoms) plus {N_GHOSTS} ghost slots; need >= "
                f"{largest + N_GHOSTS}."
            )
        total_atoms = float(n_atoms.sum())
        caps: dict[str, int] = {}
        for name, vector in vectors.items():
            if name in ("atoms", "graphs"):
                continue
            density = float(vector.sum()) / total_atoms
            by_density = _round_up(density * atom_budget * slack, round_to)
            caps[name] = max(by_density, int(vector.max()))
        # A count row is one *dataset row*, and a row is not always one graph.
        # THEMol torsion-scan's row is a whole constrained profile: 24 conformers
        # of the same molecule, each its own graph in the batch. Dividing the atom
        # budget by the mean atoms *per row* then answers how many rows fit and
        # calls that the graph count, which is 24x short — a 16 384-atom batch got
        # max_graphs=30 and arrived carrying 529 (measured 2026-09-20). With a
        # graphs vector the mean is per graph, which is what the slot count counts.
        graphs_per_row = 1.0
        if n_graphs is not None:
            graphs_per_row = float(n_graphs.sum()) / float(n_graphs.numel())
        mean_atoms = total_atoms / (float(n_atoms.numel()) * graphs_per_row)
        max_graphs = int(math.ceil(graphs_slack * atom_budget / mean_atoms)) + 1
        if n_graphs is not None:
            # A row is indivisible, so one row's graphs must fit beside the sink.
            max_graphs = max(max_graphs, int(n_graphs.max()) + 1)
        return cls(
            max_atoms=max_atoms,
            max_pairs=caps["pairs"],
            cutoff=cutoff,
            pad_atom_type=pad_atom_type,
            max_bonds=caps.get("bonds", 0),
            max_angles=caps.get("angles", 0),
            max_propers=caps.get("propers", 0),
            max_impropers=caps.get("impropers", 0),
            max_hessian_pairs=caps.get("hessian_pairs", 0),
            max_graphs=max_graphs,
        )

    def valence_cap(self, namespace: str) -> int:
        """Configured count-axis cap for *namespace*, or 0 if unknown."""
        return int(getattr(self, f"max_{namespace}", 0))

    def _ghost_slots(self) -> int:
        """Atom slots the ghosts need: four with any padded valence family, else one."""
        return N_GHOSTS if any(self.valence_cap(ns) > 0 for ns in _VALENCE) else 1

    def caps(self) -> dict[str, int]:
        """The static shape, one entry per padded axis — the calibration fingerprint.

        Returns:
            ``{"atoms", "pairs", <padded valence families>, "graphs"?}`` →
            padded length. A valence family capped at ``0`` and an unset
            ``max_graphs`` are omitted.
        """
        out = {"atoms": self.max_atoms, "pairs": self.max_pairs}
        for ns in _VALENCE:
            cap = self.valence_cap(ns)
            if cap > 0:
                out[ns] = cap
        if self.max_graphs is not None:
            out["graphs"] = self.max_graphs
        return out

    def budgets(self) -> dict[str, int]:
        """Per-batch packer budgets that keep every batch inside :meth:`caps`.

        Atoms give up the ghost slots, graphs give up the sink slot, every
        other axis is its cap. Hand the result to
        :class:`~molix.data.sampler.TokenBudgetBatchSampler` (``budgets=``)
        together with the matching per-molecule counts.

        Returns:
            ``{table: budget}`` over the same keys as :meth:`caps`.
        """
        out = self.caps()
        out["atoms"] = self.max_atoms - self._ghost_slots()
        if self.max_graphs is not None:
            out["graphs"] = self.max_graphs - 1
        return out

    def check(self, counts: Mapping[str, Any]) -> None:
        """Refuse a dataset holding a molecule no batch of this shape can carry.

        Args:
            counts: Per-molecule count tables keyed by namespace (any subset
                of :meth:`budgets`' keys; ``graphs`` needs no table).

        Raises:
            ValueError: Some molecule alone exceeds a budget — naming the
                table, the offending count and the budget.
        """
        budgets = self.budgets()
        for name, values in counts.items():
            if name == "graphs":
                continue
            if name not in budgets:
                continue
            vector = _as_long_vector(name, values)
            largest = int(vector.max())
            if largest > budgets[name]:
                where = int(torch.argmax(vector))
                raise ValueError(
                    f"MoleculePadder.check: molecule {where} has {largest} {name} but the "
                    f"per-batch {name} budget is {budgets[name]} (cap {self.caps()[name]}); "
                    "it could never be packed. Raise the cap (from_counts never sets a cap "
                    "below the largest molecule) or drop the molecule explicitly."
                )

    def __call__(self, data: TensorDict | dict) -> TensorDict | dict:
        atoms = data["atoms"]
        pairs = data["pairs"]
        n = int(atoms.batch_size[0])
        e = int(pairs.batch_size[0])
        need_ghosts = N_GHOSTS if self._needs_valence_ghosts(data) else 1
        if n + need_ghosts > self.max_atoms:
            raise ValueError(
                f"MoleculePadder: batch has {n} atoms but max_atoms={self.max_atoms}; "
                f"raise max_atoms (need >= max real atoms + {need_ghosts} ghost slots)."
            )
        if e > self.max_pairs:
            raise ValueError(
                f"MoleculePadder: batch has {e} edges but max_pairs={self.max_pairs}; "
                "raise max_pairs."
            )
        sink = self._pad_graphs(data)

        dev = atoms["Z"].device
        n_pad = self.max_atoms - n
        e_pad = self.max_pairs - e
        ghost0 = n  # index of the first ghost atom; padding edges self-loop here

        # --- atoms: append ghost rows (Z = valid type, everything else 0) ---
        new_atoms: dict[str, torch.Tensor] = {}
        for k, t in atoms.items():
            pad = t.new_zeros((n_pad, *t.shape[1:]))
            if k == "Z":
                pad = pad + self.pad_atom_type
            elif k == "batch":
                # Ghosts belong to the sink graph slot when the graph count is
                # padded (dead by construction), else nominally to graph 0;
                # their energy is masked to 0 either way.
                pad = pad + sink
            # "pos" and atom targets (e.g. forces) pad to 0.
            new_atoms[k] = torch.cat([t, pad], dim=0)
        mask = torch.zeros(self.max_atoms, dtype=torch.bool, device=dev)
        mask[:n] = True
        new_atoms["mask"] = mask
        if n_pad >= N_GHOSTS:
            length = dead_pair_length(self.cutoff)
            pos = new_atoms["pos"]
            pos[ghost0 + 1] = pos.new_tensor([length, 0.0, 0.0])
            pos[ghost0 + 2] = pos.new_tensor([0.0, length, 0.0])
            pos[ghost0 + 3] = pos.new_tensor([0.0, 0.0, length])
        self._put(
            data,
            "atoms",
            TensorDict(new_atoms, batch_size=[self.max_atoms], device=atoms.device),
        )

        # --- edges: padding edges are dead (ghost0, ghost0) self-loops ---
        new_edges: dict[str, torch.Tensor] = {}
        for k, t in pairs.items():
            if k == "pair_index":
                pad = torch.full((e_pad, *t.shape[1:]), ghost0, dtype=t.dtype, device=dev)
            elif k in ("pair_diff", "shifts"):
                pad = dead_pair_shift(self.cutoff, dtype=t.dtype, device=dev).expand(e_pad, 3)
            elif k == "pair_dist":
                pad = t.new_full((e_pad,), dead_pair_length(self.cutoff))
            else:
                pad = t.new_zeros((e_pad, *t.shape[1:]))
            new_edges[k] = torch.cat([t, pad], dim=0)
        self._put(
            data,
            "pairs",
            TensorDict(new_edges, batch_size=[self.max_pairs], device=pairs.device),
        )

        ghosts = torch.arange(ghost0, ghost0 + N_GHOSTS, device=dev)
        for ns in _VALENCE:
            if ns not in data.keys():
                continue
            self._pad_valence(data, ns, ghosts)
        return data

    def _pad_graphs(self, data: TensorDict | dict) -> int:
        """Pad ``graphs`` to ``max_graphs`` slots; return the ghost atoms' graph index."""
        if self.max_graphs is None:
            return 0
        if "graphs" not in data.keys():
            raise ValueError(
                "MoleculePadder: max_graphs is set but the batch carries no graphs "
                "namespace; collate one (graphs.num_atoms) before padding."
            )
        graphs = data["graphs"]
        if not graphs.batch_size:
            raise ValueError(
                "MoleculePadder: max_graphs is set but graphs has no batch dimension "
                "(batch_size=[]); the graph count must be the graphs batch_size."
            )
        b = int(graphs.batch_size[0])
        if b + 1 > self.max_graphs:
            raise ValueError(
                f"MoleculePadder: batch has {b} graphs but max_graphs={self.max_graphs}; "
                "need >= live graphs + 1 (the ghost atoms' sink slot). Raise max_graphs "
                "or pack fewer molecules per batch."
            )
        g_pad = self.max_graphs - b
        new_graphs: dict[str, torch.Tensor] = {}
        for key, tensor in graphs.items():
            if key == "mask":
                continue
            # num_atoms pads to 0 — a dead slot holds no atoms; labels pad to 0.
            pad = tensor.new_zeros((g_pad, *tensor.shape[1:]))
            new_graphs[key] = torch.cat([tensor, pad], dim=0)
        mask = torch.zeros(self.max_graphs, dtype=torch.bool, device=graphs.device)
        mask[:b] = True
        new_graphs["mask"] = mask
        self._put(
            data,
            "graphs",
            TensorDict(new_graphs, batch_size=[self.max_graphs], device=graphs.device),
        )
        return self.max_graphs - 1

    def _needs_valence_ghosts(self, data: TensorDict | dict) -> bool:
        for ns in _VALENCE:
            cap = self.valence_cap(ns)
            if cap <= 0 or ns not in data.keys():
                continue
            if _live_count(data[ns]) < cap:
                return True
        return False

    def _pad_valence(self, data: TensorDict | dict, ns: str, ghosts: torch.Tensor) -> None:
        cap = self.valence_cap(ns)
        if cap <= 0:
            return
        block = data[ns]
        live = _live_count(block)
        if live > cap:
            raise ValueError(
                f"MoleculePadder: batch has {live} {ns} but max_{ns}={cap}; raise max_{ns}."
            )
        n_pad = cap - live
        new_block: dict[str, torch.Tensor] = {}
        for key, tensor in block.items():
            if key == "mask":
                continue
            new_block[key] = _pad_valence_tensor(key, tensor, n_pad, ghosts)
        mask = torch.zeros(cap, dtype=torch.bool, device=ghosts.device)
        if live:
            mask[:live] = True
        new_block["mask"] = mask
        batch_size = [] if "bond_index" in new_block else [cap]
        self._put(
            data,
            ns,
            TensorDict(new_block, batch_size=batch_size, device=block.device),
        )

    @staticmethod
    def _put(data: TensorDict | dict, key: str, value: TensorDict) -> None:
        """Replace a nested TensorDict whose ``batch_size`` changed.

        In-place ``data[key] = value`` keeps the parent's old batch size
        (GH200 Class-I: atoms ``[516]`` vs padded ``[16385]``).
        """
        if key in data.keys():
            del data[key]
        data[key] = value
