"""Molecular data types built on nested TensorDict.

Defines a composable type hierarchy for molecular graph data:

- **AtomData** (batch_size=[N]): per-atom tensors (Z, pos, batch)
- **EdgeData** (batch_size=[E]): per-edge tensors (edge_index, bond_diff, bond_dist)
- **GraphData** (batch_size=[B]): per-graph tensors (num_atoms, targets)
- **GraphBatch** (batch_size=[]): top-level container nesting atoms + edges [+ graphs]

Encoder outputs extend the base types via inheritance:

- **NodeRepAtoms** extends AtomData with ``node_features``
- **EdgeRepEdges** extends EdgeData with ``edge_features``

Example:
    >>> atoms = AtomData(
    ...     Z=torch.tensor([6, 1, 1]),
    ...     pos=torch.randn(3, 3),
    ...     batch=torch.tensor([0, 0, 0]),
    ...     batch_size=[3],
    ... )
    >>> edges = EdgeData(
    ...     edge_index=torch.tensor([[0, 1], [1, 0]]),
    ...     bond_diff=torch.randn(2, 3),
    ...     bond_dist=torch.rand(2),
    ...     batch_size=[2],
    ... )
    >>> batch = GraphBatch(atoms=atoms, edges=edges, batch_size=[])
    >>> batch["atoms", "Z"]  # tensor([6, 1, 1])
"""

from __future__ import annotations

from tensordict import TensorDict


def _validate_keys(td: TensorDict, required: frozenset[str], cls_name: str) -> None:
    """Check that all required keys are present in the TensorDict."""
    present = set(str(k) for k in td.keys())
    missing = required - present
    if missing:
        raise KeyError(f"{cls_name} missing required keys: {missing}")


# ---------------------------------------------------------------------------
# Atom-level (batch_size=[N_total])
# ---------------------------------------------------------------------------


class AtomData(TensorDict):
    """Per-atom tensors. ``batch_size=[N]``.

    Required keys:
        - ``Z``: Atomic numbers ``(N,)``
        - ``pos``: Atomic positions ``(N, 3)``
        - ``batch``: Graph membership ``(N,)``
    """

    _required_keys: frozenset[str] = frozenset({"Z", "pos", "batch"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _validate_keys(self, self._required_keys, type(self).__name__)


class NodeRepAtoms(AtomData):
    """AtomData extended with node-level encoder features.

    Additional required keys:
        - ``node_features``: Encoder output ``(N, [L,] D)``
    """

    _required_keys: frozenset[str] = AtomData._required_keys | {"node_features"}


# ---------------------------------------------------------------------------
# Edge-level (batch_size=[E_total])
# ---------------------------------------------------------------------------


class EdgeData(TensorDict):
    """Per-edge tensors. ``batch_size=[E]``.

    Required keys:
        - ``edge_index``: Source-target pairs ``(E, 2)``
        - ``bond_diff``: Edge displacement vectors ``(E, 3)``
        - ``bond_dist``: Edge distances ``(E,)``
    """

    _required_keys: frozenset[str] = frozenset({"edge_index", "bond_diff", "bond_dist"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _validate_keys(self, self._required_keys, type(self).__name__)


class EdgeRepEdges(EdgeData):
    """EdgeData extended with edge-level encoder features.

    Additional required keys:
        - ``edge_features``: Encoder output ``(E, [L,] D)``
    """

    _required_keys: frozenset[str] = EdgeData._required_keys | {"edge_features"}


# ---------------------------------------------------------------------------
# Graph-level (batch_size=[B])
# ---------------------------------------------------------------------------


class GraphData(TensorDict):
    """Per-graph tensors. ``batch_size=[B]``.

    Required keys:
        - ``num_atoms``: Atom counts per graph ``(B,)``

    Optional keys:
        - ``targets``: Nested TensorDict of target labels
    """

    _required_keys: frozenset[str] = frozenset({"num_atoms"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _validate_keys(self, self._required_keys, type(self).__name__)


# ---------------------------------------------------------------------------
# Top-level batch (batch_size=[])
# ---------------------------------------------------------------------------


class GraphBatch(TensorDict):
    """Top-level molecular graph batch. ``batch_size=[]``.

    Nests three levels with independent batch dimensions:
        - ``atoms``: AtomData ``(batch_size=[N])``
        - ``edges``: EdgeData ``(batch_size=[E])``
        - ``graphs``: GraphData ``(batch_size=[B])`` (optional)

    Required keys:
        - ``atoms``
        - ``edges``
    """

    _required_keys: frozenset[str] = frozenset({"atoms", "edges"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _validate_keys(self, self._required_keys, type(self).__name__)
