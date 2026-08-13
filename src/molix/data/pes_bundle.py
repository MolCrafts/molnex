"""One covalent topology stacked with many conformer coordinates.

``PesBundle`` is a data/view type: it does not encode, assign Class-I
parameters, or evaluate energy. Callers assign parameters once from
:meth:`PesBundle.topology_batch` and evaluate each frame from
:meth:`PesBundle.conformation_batch`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from tensordict import TensorDict

__all__ = ["PesBundle"]

_FORBIDDEN_ATOM_KEYS: tuple[str, ...] = ("pos", "vel", "forces")
_ATOM_CONFORMER_KEYS: tuple[str, ...] = ("pos", "vel", "forces", "q")
_GRAPH_CONFORMER_KEYS: tuple[str, ...] = ("energy", "cell", "hessian", "forces")
_COVALENT_NAMESPACES: tuple[str, ...] = ("bonds", "angles", "propers", "impropers")
_COVALENT_INDEX_KEYS: frozenset[str] = frozenset({"bond_index", "atomi", "atomj", "atomk", "atoml"})
_ALLOWED_TARGETS: frozenset[str] = frozenset({"energy", "forces", "hessian", "q"})
_TARGET_FROM_CONFORMER: tuple[tuple[str, tuple[str, str]], ...] = (
    ("energy", ("graphs", "energy")),
    ("forces", ("atoms", "forces")),
    ("hessian", ("graphs", "hessian")),
    ("q", ("atoms", "q")),
)


def _has(td: TensorDict, *path: str) -> bool:
    cur: object = td
    for key in path:
        if not isinstance(cur, TensorDict) or key not in cur.keys():
            return False
        cur = cur[key]
    return True


def _tensor(td: TensorDict, *path: str) -> torch.Tensor:
    value = td[path]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{'.'.join(path)} must be a tensor")
    return value


def _drop(td: TensorDict, *path: str) -> None:
    if not _has(td, *path):
        return
    parent: TensorDict = td
    for key in path[:-1]:
        nested = parent[key]
        if not isinstance(nested, TensorDict):
            return
        parent = nested
    del parent[path[-1]]


def _target_shape(key: str, n_conformers: int, n_atoms: int) -> tuple[int, ...]:
    if key == "energy":
        return (n_conformers,)
    if key == "forces":
        return (n_conformers, n_atoms, 3)
    if key == "hessian":
        return (n_conformers, 3 * n_atoms, 3 * n_atoms)
    if key == "q":
        return (n_conformers, n_atoms)
    raise ValueError(f"unknown target key {key!r}")


def _covalent_index_paths(td: TensorDict) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for namespace in _COVALENT_NAMESPACES:
        if not _has(td, namespace):
            continue
        block = td[namespace]
        if not isinstance(block, TensorDict):
            continue
        for key in block.keys():
            if key in _COVALENT_INDEX_KEYS:
                paths.append((namespace, str(key)))
    return paths


def _require_same_topology(first: TensorDict, other: TensorDict, index: int) -> None:
    if not _has(first, "atoms", "Z") or not _has(other, "atoms", "Z"):
        raise ValueError(f"conformer {index} atoms.Z does not match conformer 0")
    if not torch.equal(_tensor(first, "atoms", "Z"), _tensor(other, "atoms", "Z")):
        raise ValueError(f"conformer {index} atoms.Z does not match conformer 0")
    paths = set(_covalent_index_paths(first)) | set(_covalent_index_paths(other))
    for namespace, key in sorted(paths):
        if not _has(first, namespace, key) or not _has(other, namespace, key):
            raise ValueError(f"conformer {index} {namespace}.{key} does not match conformer 0")
        if not torch.equal(_tensor(first, namespace, key), _tensor(other, namespace, key)):
            raise ValueError(f"conformer {index} {namespace}.{key} does not match conformer 0")


def _strip_conformer_fields(topology: TensorDict) -> TensorDict:
    for key in _ATOM_CONFORMER_KEYS:
        _drop(topology, "atoms", key)
    if "edges" in topology.keys():
        del topology["edges"]
    for key in _GRAPH_CONFORMER_KEYS:
        _drop(topology, "graphs", key)
    return topology


def _stack_targets(conformers: Sequence[TensorDict]) -> TensorDict | None:
    data: dict[str, torch.Tensor] = {}
    for key, path in _TARGET_FROM_CONFORMER:
        if all(_has(conf, *path) for conf in conformers):
            data[key] = torch.stack([_tensor(conf, *path) for conf in conformers], dim=0)
    if not data:
        return None
    return TensorDict(data, batch_size=[len(conformers)])


class PesBundle:
    """Covalent graph plus stacked conformer frames for one molecule.

    Owns topology ``G`` (nuclear charges and covalent indices; optional
    graph charge ``Q`` in elementary charge ``e``) and coordinates
    ``R_c`` of shape ``(C, N, 3)`` in Å. Optional targets keep
    caller-owned energy units (forces as energy/Å, Hessian as
    energy/Å²; atomic charges ``q`` in ``e``). Relative energies are
    not pre-centered.

    Args:
        topology: Nested ``TensorDict`` (``batch_size=[]``) with
            required ``atoms.Z`` ``(N,)``. Must not contain
            ``atoms.pos`` / ``atoms.vel`` / ``atoms.forces`` or an
            ``edges`` namespace.
        pos: Conformer coordinates ``(C, N, 3)`` in Å, ``C >= 1``.
        group_id: Molecule-graph / isomer identity used for group
            splits.
        targets: Optional flat ``TensorDict`` with ``batch_size=[C]``
            and keys in ``energy`` / ``forces`` / ``hessian`` / ``q``.

    Reference:
        Wang et al., *Chem. Sci.* **2022**, 13, 12016–12033.
        https://doi.org/10.1039/D2SC02739A
        Takaba et al., arXiv:2307.07085 (2023).
    """

    def __init__(
        self,
        topology: TensorDict,
        pos: torch.Tensor,
        *,
        group_id: str | int,
        targets: TensorDict | None = None,
    ) -> None:
        owned = topology.clone()
        owned_pos = pos.clone()
        if not _has(owned, "atoms", "Z"):
            raise ValueError("topology must contain atoms.Z")
        nested = owned.keys(include_nested=True)
        for key in _FORBIDDEN_ATOM_KEYS:
            if ("atoms", key) in nested:
                raise ValueError(f"topology must not contain atoms.{key}")
        if "edges" in owned.keys():
            raise ValueError("topology must not contain the edges namespace")

        z = _tensor(owned, "atoms", "Z")
        n_atoms = int(z.shape[0])
        if owned_pos.ndim != 3 or owned_pos.shape[-1] != 3:
            raise ValueError(f"pos must have shape (C, N, 3), got {tuple(owned_pos.shape)}")
        n_conformers = int(owned_pos.shape[0])
        if n_conformers == 0:
            raise ValueError("pos must contain at least one conformer (C >= 1)")
        if int(owned_pos.shape[1]) != n_atoms:
            raise ValueError(
                f"pos atom count {int(owned_pos.shape[1])} does not match topology N={n_atoms}"
            )

        if not _has(owned, "atoms", "batch"):
            owned["atoms", "batch"] = torch.zeros(n_atoms, dtype=torch.long, device=z.device)

        owned_targets: TensorDict | None = None
        if targets is not None:
            owned_targets = targets.clone()
            for key in owned_targets.keys():
                name = str(key)
                if name not in _ALLOWED_TARGETS:
                    raise ValueError(f"unknown target key {name!r}")
                expected = _target_shape(name, n_conformers, n_atoms)
                got = tuple(owned_targets[name].shape)
                if got != expected:
                    raise ValueError(f"target {name!r} has shape {got}, expected {expected}")

        self._topology = owned
        self._pos = owned_pos
        self._group_id: str | int = group_id
        self._targets = owned_targets
        self._n_atoms = n_atoms
        self._n_conformers = n_conformers

    @classmethod
    def from_conformer_list(
        cls,
        conformers: Sequence[TensorDict],
        *,
        group_id: str | int,
    ) -> PesBundle:
        """Stack nested single-frame graphs that share one covalent topology.

        Each input ``TensorDict`` must already be nested and carry
        ``atoms.pos`` in Å. All frames must have equal ``atoms.Z`` and
        covalent index tensors. Conformer-varying fields are stripped
        from the stored topology; energy / force / Hessian / ``q``
        targets are stacked when every frame provides them (caller-owned
        energy units; ``q`` in ``e``).

        Args:
            conformers: Same-molecule nested frames, each with
                ``atoms.pos`` ``(N, 3)`` in Å.
            group_id: Molecule-graph / isomer identity.

        Returns:
            A bundle whose ``pos`` is the stacked Å frames.

        Raises:
            ValueError: Empty sequence, missing ``atoms.pos``, or a
                topology mismatch (message names the conformer index
                and key).
        """
        if not conformers:
            raise ValueError("conformers must be non-empty")
        first = conformers[0]
        if not _has(first, "atoms", "Z"):
            raise ValueError("conformer 0 missing atoms.Z")
        if not _has(first, "atoms", "pos"):
            raise ValueError("conformer 0 missing atoms.pos")
        n_atoms = int(_tensor(first, "atoms", "Z").shape[0])
        frames: list[torch.Tensor] = []
        for index, conf in enumerate(conformers):
            if index > 0:
                _require_same_topology(first, conf, index)
            if not _has(conf, "atoms", "pos"):
                raise ValueError(f"conformer {index} missing atoms.pos")
            pos_i = _tensor(conf, "atoms", "pos")
            if pos_i.shape != (n_atoms, 3):
                raise ValueError(
                    f"conformer {index} atoms.pos has shape {tuple(pos_i.shape)}, "
                    f"expected ({n_atoms}, 3)"
                )
            frames.append(pos_i)
        topology = _strip_conformer_fields(first.clone())
        return cls(
            topology,
            torch.stack(frames, dim=0),
            group_id=group_id,
            targets=_stack_targets(conformers),
        )

    @property
    def group_id(self) -> str | int:
        """Molecule-graph / isomer identity shared by every conformer."""
        return self._group_id

    @property
    def n_atoms(self) -> int:
        """Atom count ``N`` of the shared topology."""
        return self._n_atoms

    @property
    def n_conformers(self) -> int:
        """Number of stacked frames ``C``."""
        return self._n_conformers

    @property
    def pos(self) -> torch.Tensor:
        """Stacked conformer coordinates ``(C, N, 3)`` in Å."""
        return self._pos

    @property
    def topology(self) -> TensorDict:
        """Owned covalent topology (no ``atoms.pos``)."""
        return self._topology

    @property
    def targets(self) -> TensorDict | None:
        """Optional stacked targets, or ``None`` when the bundle has none."""
        return self._targets

    def topology_batch(self) -> TensorDict:
        """Clone the covalent topology with no coordinates.

        The returned ``TensorDict`` never contains ``pos`` (top-level or
        under ``atoms``). Optional ``graphs.total_charge`` stays in ``e``.
        Intended for a single parameter assignment on graph ``G``.

        Returns:
            A new topology ``TensorDict`` without Å coordinates.
        """
        return self._topology.clone()

    def conformation_batch(self, i: int) -> TensorDict:
        """Clone topology and write conformer ``i`` as ``atoms.pos``.

        ``atoms.pos`` is ``(N, 3)`` in Å. Present targets map to
        ``graphs.energy`` (caller-owned energy units), ``atoms.forces``
        (energy/Å), ``graphs.hessian`` (energy/Å², Cartesian
        ``x0,y0,z0,…``), and ``atoms.q`` (``e``).

        Args:
            i: Conformer index in ``[0, C)``.

        Returns:
            A new nested ``TensorDict`` for one frame.

        Raises:
            IndexError: If ``i`` is outside ``[0, C)``.
        """
        if i < 0 or i >= self._n_conformers:
            raise IndexError(f"conformation index {i} out of range for C={self._n_conformers}")
        batch = self._topology.clone()
        batch["atoms", "pos"] = self._pos[i]
        if self._targets is None:
            return batch
        if "energy" in self._targets.keys():
            if "graphs" not in batch.keys():
                batch["graphs"] = TensorDict({}, batch_size=[])
            batch["graphs", "energy"] = self._targets["energy"][i]
        if "hessian" in self._targets.keys():
            if "graphs" not in batch.keys():
                batch["graphs"] = TensorDict({}, batch_size=[])
            batch["graphs", "hessian"] = self._targets["hessian"][i]
        if "forces" in self._targets.keys():
            batch["atoms", "forces"] = self._targets["forces"][i]
        if "q" in self._targets.keys():
            batch["atoms", "q"] = self._targets["q"][i]
        return batch

    def conformer_group_ids(self) -> tuple[str | int, ...]:
        """Repeat ``group_id`` once per stacked frame.

        The tuple has length ``C`` so flattened ids can be passed to
        :func:`~molix.data.group_split.group_split_indices`. Physical
        coordinates remain Å; optional targets keep caller-owned energy
        units and charges in ``e``.

        Returns:
            ``(group_id,) * C``.
        """
        return (self._group_id,) * self._n_conformers

    def __len__(self) -> int:
        """Number of conformers ``C``."""
        return self._n_conformers

    def __iter__(self) -> Iterator[TensorDict]:
        """Yield :meth:`conformation_batch` for ``i = 0 … C-1``."""
        for i in range(self._n_conformers):
            yield self.conformation_batch(i)
