"""Declarative DAG pipeline with per-node caching.

A :class:`PipelineSpec` is a compiled, immutable DAG of :class:`Node`
objects. Each node wraps a task and declares which dict keys it reads and
writes. Edges are derived automatically from key dependencies. Every
cacheable node materialises its output to an independent
:class:`~molix.data.cache.PackedCache`, so changing a downstream task
never recomputes upstream nodes.

Typical usage::

    from molix.data import Pipeline, AtomicDress, NeighborList

    pipe = (
        Pipeline("qm9-u0")
        .add(NeighborList(cutoff=5.0), name="nlist")
        .add(AtomicDress(elements=[1, 6, 7, 8, 9]), name="dress")
        .build()
    )

    # DDP-aware per-node cache materialisation:
    dag = pipe.cache(source, base_dir=run_dir / "cache", fit_source=train_source)
    full = dag.dataset(mmap=True)
    train_ds, val_ds = full.split(sizes=(100_000, 30_000), seed=42)
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from molix.data.cache import PackedCache
from molix.data.task import BatchTask, DatasetTask, Runnable, SampleTask

__all__ = ["Node", "Edge", "DAGCache", "PipelineSpec", "Pipeline"]


# Reserved node name for the raw-source cache materialised when a pipeline has
# no cacheable transform nodes — a "no-op" pipeline still yields a dataset.
_SOURCE_NODE = "__source__"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """A single node in the pipeline DAG.

    Wraps a :class:`~molix.data.task.Task` (or plain callable) with
    optional key-level read/write declarations. When *reads* and *writes*
    are both empty, the node is treated as linearly dependent on the
    previous node in registration order.

    Each node is independently cacheable — its cache key is derived from
    its ``task_id`` chained onto the keys of all upstream nodes.
    """

    name: str
    task: Any  # SampleTask | DatasetTask | BatchTask | Callable
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()

    @property
    def task_id(self) -> str:
        """Deterministic identifier folded into the per-node cache key."""
        return getattr(self.task, "task_id", type(self.task).__qualname__)

    @property
    def is_cacheable(self) -> bool:
        """Only SampleTask and DatasetTask nodes are cached to disk."""
        return isinstance(self.task, (SampleTask, DatasetTask))

    @property
    def needs_fit(self) -> bool:
        """True for DatasetTask nodes that require a fit phase."""
        return isinstance(self.task, DatasetTask)

    def apply(self, data: dict) -> dict:
        """Dispatch *data* through the wrapped task.

        Accepts :class:`~molix.data.task.Runnable` (via ``execute``) or a
        plain callable.
        """
        task = self.task
        if isinstance(task, Runnable):
            return task.execute(data)
        if callable(task):
            return task(data)
        raise TypeError(f"Node {self.name!r} task is neither Runnable nor callable: {type(task)}")

    def cache_key(
        self,
        upstream_keys: tuple[str, ...],
        fit_source_id: str,
        extra: dict[str, str] | None,
    ) -> str:
        """12-hex SHA256 for this node given its upstream identity chain.

        The chain is transitive: node *N*'s key embeds node *N-1*'s key,
        which embeds node *N-2*'s key ... back to the source identity.
        Changing any upstream node or the source invalidates this node.
        """
        parts = [f"task={self.task_id}"]
        for uk in upstream_keys:
            parts.append(f"up={uk}")
        if fit_source_id:
            parts.append(f"fs={fit_source_id}")
        if extra:
            for k in sorted(extra):
                parts.append(f"{k}={extra[k]}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        return f"Node(name={self.name!r}, task={type(self.task).__name__})"


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """A directed data dependency between two nodes.

    Derived automatically from :attr:`Node.reads` and :attr:`Node.writes`
    during :meth:`Pipeline.build`.
    """

    from_node: str
    to_node: str
    keys: frozenset[str]

    def __repr__(self) -> str:
        return f"Edge({self.from_node} → {self.to_node}, keys={sorted(self.keys)})"


# ---------------------------------------------------------------------------
# DAGCache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DAGCache:
    """Per-node cache collection produced by :meth:`PipelineSpec.cache`.

    Maps each cacheable node name to its :class:`~molix.data.cache.PackedCache`
    file. The final node's cache is the authoritative pipeline output.
    """

    spec: "PipelineSpec"
    node_caches: dict[str, PackedCache]
    base_dir: Path

    @property
    def final(self) -> PackedCache:
        """The last cacheable node's cache — authoritative pipeline output."""
        if not self.node_caches:
            raise RuntimeError("DAGCache has no cacheable nodes.")
        names = [n.name for n in self.spec.prepare_nodes if n.name in self.node_caches]
        if names:
            return self.node_caches[names[-1]]
        # No transform nodes: a no-op pipeline caches the raw source verbatim.
        if _SOURCE_NODE in self.node_caches:
            return self.node_caches[_SOURCE_NODE]
        raise RuntimeError("DAGCache has no cacheable nodes.")

    def dataset(self, *, mmap: bool = True):
        """Create a dataset reading from the final node's cache.

        Args:
            mmap: If ``True`` return :class:`~molix.data.dataset.MmapDataset`,
                otherwise :class:`~molix.data.dataset.CachedDataset`.
        """
        from molix.data.dataset import CachedDataset, MmapDataset

        return MmapDataset(self.final.sink) if mmap else CachedDataset(self.final.sink)

    def node_dataset(self, node_name: str, *, mmap: bool = True):
        """Create a dataset reading from a specific node's cache.

        Useful for loading an intermediate pipeline stage (e.g. only up to
        the NeighborList node).

        Args:
            node_name: Name of the node whose cache to read.
            mmap: If ``True`` return :class:`~molix.data.dataset.MmapDataset`.
        """
        from molix.data.dataset import CachedDataset, MmapDataset

        if node_name not in self.node_caches:
            raise KeyError(
                f"Node {node_name!r} not found in DAGCache. "
                f"Available: {sorted(self.node_caches.keys())}"
            )
        sink = self.node_caches[node_name].sink
        return MmapDataset(sink) if mmap else CachedDataset(sink)

    def __repr__(self) -> str:
        nodes = sorted(self.node_caches.keys())
        return f"DAGCache(nodes={nodes}, base_dir={self.base_dir!s})"


# ---------------------------------------------------------------------------
# PipelineSpec — immutable compiled DAG
# ---------------------------------------------------------------------------


class PipelineSpec:
    """Compiled, immutable pipeline DAG.

    Holds the topologically sorted node list, derived edges, and all
    orchestration methods (fit, cache, run, transform). There is no
    separate execute / cache / DDP module — orchestration lives here.
    """

    __slots__ = ("name", "pipeline_id", "nodes", "edges", "_topo_order")

    def __init__(
        self,
        name: str,
        pipeline_id: str,
        nodes: tuple[Node, ...],
        edges: tuple[Edge, ...],
        topo_order: tuple[Node, ...],
    ) -> None:
        self.name = name
        self.pipeline_id = pipeline_id
        self.nodes = nodes
        self.edges = edges
        self._topo_order = topo_order

    # -- grouping ----------------------------------------------------------

    @property
    def prepare_nodes(self) -> tuple[Node, ...]:
        """Cacheable nodes in topological order (SampleTask / DatasetTask)."""
        return tuple(n for n in self._topo_order if n.is_cacheable)

    @property
    def batch_nodes(self) -> tuple[Node, ...]:
        """BatchTask nodes — post-collate, never cached."""
        return tuple(n for n in self._topo_order if isinstance(n.task, BatchTask))

    # -- introspection -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the DAG to a plain, JSON-friendly dict for introspection.

        Returns:
            A dict with ``name``, ``pipeline_id``, a ``nodes`` list (each
            with ``name`` / ``type`` / ``task_id`` / sorted ``reads`` /
            sorted ``writes`` / ``cacheable``) in topological order, and an
            ``edges`` list (each ``from`` / ``to`` / sorted ``keys``).
        """
        return {
            "name": self.name,
            "pipeline_id": self.pipeline_id,
            "nodes": [
                {
                    "name": n.name,
                    "type": type(n.task).__name__,
                    "task_id": n.task_id,
                    "reads": sorted(n.reads),
                    "writes": sorted(n.writes),
                    "cacheable": n.is_cacheable,
                }
                for n in self._topo_order
            ],
            "edges": [
                {"from": e.from_node, "to": e.to_node, "keys": sorted(e.keys)} for e in self.edges
            ],
        }

    def __repr__(self) -> str:
        names = ", ".join(n.name for n in self._topo_order)
        return f"PipelineSpec(name={self.name!r}, nodes=[{names}], id={self.pipeline_id})"

    # -- fit ---------------------------------------------------------------

    def fit(self, fit_source: Any) -> dict[str, dict[str, Any]]:
        """Fit every :class:`DatasetTask` node on *fit_source*.

        Applies upstream :class:`SampleTask` nodes between fits so each
        :class:`DatasetTask` sees the same transformed data it would
        during full execution.

        Args:
            fit_source: :class:`~molix.data.source.DataSource` used for
                fitting. Must support ``__len__`` and ``__getitem__``.

        Returns:
            ``{node_name: task.state_dict()}`` for every fitted node.
            Ready for :meth:`load_fit_states` or the *fit_states*
            parameter of :meth:`cache`.
        """
        samples = [fit_source[i] for i in range(len(fit_source))]
        states: dict[str, dict[str, Any]] = {}

        for node in self.prepare_nodes:
            if node.needs_fit:
                node.task.fit(samples)
                states[node.name] = node.task.state_dict()
            samples = [node.apply(s) for s in samples]

        return states

    def collect_fit_states(self) -> dict[str, dict[str, Any]]:
        """Return ``{node_name: task.state_dict()}`` for every fitted DatasetTask.

        Call after :meth:`run` or :meth:`cache` so fitted state is captured.
        """
        states: dict[str, dict[str, Any]] = {}
        for node in self.prepare_nodes:
            if node.needs_fit:
                states[node.name] = node.task.state_dict()
        return states

    def load_fit_states(self, states: dict[str, dict[str, Any]]) -> None:
        """Restore fitted state into each :class:`DatasetTask` node by name.

        Duplicate node names are checked at :meth:`Pipeline.build` time so
        routing is unambiguous.
        """
        by_name = {n.name: n for n in self.prepare_nodes}
        for name, state in states.items():
            node = by_name.get(name)
            if node is not None and node.needs_fit:
                node.task.load_state_dict(state)

    # -- cache -------------------------------------------------------------

    def _node_cache_keys(
        self,
        source: Any,
        fit_source: Any | None,
        extra: dict[str, str] | None,
    ) -> dict[str, str]:
        """Compute per-node cache keys without materialising data."""
        keys: dict[str, str] = {}
        upstream: list[str] = [source.source_id]
        fs_id = fit_source.source_id if fit_source is not None else ""

        for node in self.prepare_nodes:
            key = node.cache_key(tuple(upstream), fs_id, extra)
            keys[node.name] = key
            upstream.append(key)

        return keys

    def cache_key(
        self,
        source: Any,
        *,
        fit_source: Any | None = None,
        extra: dict[str, str] | None = None,
    ) -> str:
        """Full-pipeline cache identity — equivalent to the final node's key.

        Forwards to the last cacheable node's :meth:`Node.cache_key`.
        When the pipeline has no cacheable nodes, falls back to a hash of
        ``(pipeline_id, source_id)`` so empty pipelines still have stable
        identities.
        """
        keys = self._node_cache_keys(source, fit_source, extra)
        if not keys:
            parts = [f"pid={self.pipeline_id}", f"src={source.source_id}"]
            fs_id = fit_source.source_id if fit_source is not None else ""
            if fs_id:
                parts.append(f"fs={fs_id}")
            if extra:
                for k in sorted(extra):
                    parts.append(f"{k}={extra[k]}")
            return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
        last = [n.name for n in self.prepare_nodes][-1]
        return keys[last]

    def cache(
        self,
        source: Any,
        *,
        base_dir: str | Path,
        fit_source: Any | None = None,
        fit_states: dict[str, dict[str, Any]] | None = None,
        extra: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> DAGCache:
        """DDP-aware per-node cache materialisation.

        On rank 0: for each cacheable node in topological order, compute its
        cache key, check for an existing cache, and rebuild if missing. Each
        node writes an independent :class:`~molix.data.cache.PackedCache`.

        On other ranks: poll each node's sink file in order until all are
        ready.

        Args:
            source: Full :class:`~molix.data.source.DataSource` to
                materialise.
            base_dir: Directory holding cache files. Created on rank 0.
            fit_source: Source used for fitting :class:`DatasetTask` nodes.
                Mutually exclusive with *fit_states*.
            fit_states: Pre-fitted states from :meth:`fit`. When provided,
                fitting is skipped and these states are loaded instead.
            extra: Extra identity strings folded into each node's cache
                key (split sizes, seed, dtype, ...).
            overwrite: Force rebuild even if caches exist.

        Returns:
            :class:`DAGCache` mapping each node name to its cache file.
        """
        base_dir = Path(base_dir)
        prepare = self.prepare_nodes
        node_keys = self._node_cache_keys(source, fit_source, extra)

        if not prepare:
            return self._cache_source_only(source, base_dir, fit_source, extra, overwrite)

        # Resolve fit states: fit_source > fit_states > auto-fit on source
        if fit_states is not None:
            self.load_fit_states(fit_states)
        elif fit_source is not None:
            states = self.fit(fit_source)
            self.load_fit_states(states)
        elif any(n.needs_fit for n in prepare):
            # Auto-fit on the full source (same as old run() Case B)
            self.load_fit_states(self.fit(source))

        node_caches: dict[str, PackedCache] = {}

        if self._is_primary_rank():
            base_dir.mkdir(parents=True, exist_ok=True)
            upstream_data: Any = None  # PackedCache | list[dict]

            for i, node in enumerate(prepare):
                sink = base_dir / f"{self.name}-{node_keys[node.name]}.pt"
                packed = PackedCache(sink)

                if not overwrite and packed.is_ready():
                    node_caches[node.name] = packed
                    upstream_data = packed
                    continue

                # Rebuild: load upstream data
                if upstream_data is None:
                    samples = [source[j] for j in range(len(source))]
                elif isinstance(upstream_data, PackedCache):
                    payload = upstream_data.load(mmap=True)
                    n = int(payload["n_samples"])
                    samples = [PackedCache.unpack_sample(payload, j) for j in range(n)]
                else:
                    samples = upstream_data

                # Apply this node
                processed = [node.apply(s) for s in samples]

                # Build task_states for the save payload
                node_task_states: dict[str, dict[str, Any]] = {}
                if node.needs_fit:
                    node_task_states[node.name] = node.task.state_dict()

                packed.save(processed, task_states=node_task_states, overwrite=overwrite)
                node_caches[node.name] = packed
                upstream_data = packed
        else:
            for node in prepare:
                sink = base_dir / f"{self.name}-{node_keys[node.name]}.pt"
                PackedCache(sink).wait_until_ready()
                node_caches[node.name] = PackedCache(sink)

        return DAGCache(spec=self, node_caches=node_caches, base_dir=base_dir)

    def _cache_source_only(
        self,
        source: Any,
        base_dir: Path,
        fit_source: Any | None,
        extra: dict[str, str] | None,
        overwrite: bool,
    ) -> DAGCache:
        """Materialise the raw source when the pipeline has no transform nodes.

        A no-op pipeline still produces a usable dataset by caching the source
        samples verbatim under the reserved :data:`_SOURCE_NODE` key, identified
        by the empty-pipeline fallback :meth:`cache_key`. Rank-aware to match
        :meth:`cache`: rank 0 writes, other ranks wait.
        """
        key = self.cache_key(source, fit_source=fit_source, extra=extra)
        sink = base_dir / f"{self.name}-{key}.pt"
        packed = PackedCache(sink)
        if self._is_primary_rank():
            base_dir.mkdir(parents=True, exist_ok=True)
            if overwrite or not packed.is_ready():
                samples = [source[j] for j in range(len(source))]
                packed.save(samples, task_states={}, overwrite=overwrite)
        else:
            packed.wait_until_ready()
            packed = PackedCache(sink)
        return DAGCache(spec=self, node_caches={_SOURCE_NODE: packed}, base_dir=base_dir)

    # -- execution ---------------------------------------------------------

    def transform(self, sample: dict) -> dict:
        """Apply every prepare node to *sample* in order.

        Per-sample inference transform. Any :class:`DatasetTask` node must
        have been fitted already (e.g. via :meth:`fit` or a prior
        :meth:`cache`); the task will raise if called unfit.
        """
        for node in self.prepare_nodes:
            sample = node.apply(sample)
        return sample

    def run(
        self,
        source: Any,
        *,
        fit_source: Any | None = None,
        fit_states: dict[str, dict[str, Any]] | None = None,
    ) -> Iterator[dict]:
        """Iterate *source*, fit :class:`DatasetTask` nodes, yield processed samples.

        Pure in-memory; no disk IO.

        Args:
            source: Raw :class:`~molix.data.source.DataSource`.
            fit_source: Source used for fitting :class:`DatasetTask` nodes.
            fit_states: Pre-fitted states — skips the fit pass entirely.

        Yields:
            Processed sample dicts in *source* order.
        """
        if fit_states is not None:
            self.load_fit_states(fit_states)

        prepare = self.prepare_nodes
        has_dataset = any(n.needs_fit for n in prepare)

        if has_dataset and fit_source is not None and fit_states is None:
            # Fit on fit_source (upstream SampleTasks applied first)
            fit_samples = [fit_source[i] for i in range(len(fit_source))]
            for node in prepare:
                if node.needs_fit:
                    node.task.fit(fit_samples)
                fit_samples = [node.apply(s) for s in fit_samples]

        if has_dataset and fit_source is None and fit_states is None:
            # Fit on full source, interleaved
            buffered = [source[i] for i in range(len(source))]
            for node in prepare:
                if node.needs_fit:
                    node.task.fit(buffered)
                buffered = [node.apply(s) for s in buffered]
            yield from buffered
            return

        # Apply pass
        for i in range(len(source)):
            s = source[i]
            for node in prepare:
                s = node.apply(s)
            yield s

    # -- DDP helpers (private) -----------------------------------------------

    @staticmethod
    def _is_primary_rank() -> bool:
        """Return ``True`` when ``$RANK`` is ``"0"``, unset, or malformed.

        Drives rank-0-builds-others-wait in :meth:`cache`. Mirrors the
        launcher convention (``torchrun`` / ``torch.distributed``) and
        deliberately avoids calling ``torch.distributed.get_rank`` because
        the cache stage typically runs *before* the process group is
        initialised.
        """
        try:
            return int(os.environ.get("RANK", "0")) == 0
        except ValueError:
            return True


# ---------------------------------------------------------------------------
# Pipeline — builder DSL
# ---------------------------------------------------------------------------


class Pipeline:
    """Fluent builder for a :class:`PipelineSpec`.

    Three equivalent task-registration styles::

        Pipeline("p").add(NeighborList(cutoff=5.0))         # Task instance
        Pipeline("p").add(my_callable, name="normalize")     # bare callable
        Pipeline("p").node(Node("n", some_task))             # pre-built Node
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: list[Node] = []
        self._names: set[str] = set()

    def _register(self, node: Node) -> None:
        if node.name in self._names:
            raise ValueError(
                f"Node name {node.name!r} already registered in pipeline "
                f"{self.name!r}. Node names must be unique so fitted state "
                f"can be unambiguously routed."
            )
        self._names.add(node.name)
        self._entries.append(node)

    def add(
        self,
        task: Any,
        *,
        name: str | None = None,
        reads: frozenset[str] | set[str] | None = None,
        writes: frozenset[str] | set[str] | None = None,
    ) -> "Pipeline":
        """Add a Task instance or plain callable. Returns self for chaining.

        Args:
            task: A :class:`~molix.data.task.Task` subclass instance or a
                plain callable.
            name: Node name. Defaults to ``task.task_id`` or the type name.
            reads: Dict keys this node consumes (for dependency derivation).
            writes: Dict keys this node produces.
        """
        self._validate(task)
        entry_name = name if name is not None else getattr(task, "task_id", type(task).__qualname__)
        node = Node(
            name=entry_name,
            task=task,
            reads=frozenset(reads) if reads is not None else frozenset(),
            writes=frozenset(writes) if writes is not None else frozenset(),
        )
        self._register(node)
        return self

    def node(self, node: Node) -> "Pipeline":
        """Add a pre-built :class:`Node`. Returns self for chaining."""
        self._validate(node.task)
        self._register(node)
        return self

    def build(self) -> PipelineSpec:
        """Compile the registered nodes into an immutable :class:`PipelineSpec`.

        Derives :class:`Edge` objects from key dependencies when
        :attr:`Node.reads` / :attr:`Node.writes` are declared. When all
        nodes have empty reads/writes, edges reflect simple linear
        dependency (each node depends on the previous).

        Returns:
            An immutable :class:`PipelineSpec`.
        """
        nodes = tuple(self._entries)
        pipeline_id = self._compute_pipeline_id(self.name, nodes)
        edges = self._derive_edges(nodes)
        topo_order = self._topo_sort(nodes, edges)
        return PipelineSpec(self.name, pipeline_id, nodes, edges, topo_order)

    # -- private helpers ---------------------------------------------------

    @staticmethod
    def _validate(task: Any) -> None:
        if isinstance(task, (SampleTask, DatasetTask, BatchTask)):
            return
        if callable(task):
            return
        raise TypeError(
            f"Task must be a SampleTask/DatasetTask/BatchTask or callable, "
            f"got {type(task).__name__}"
        )

    @staticmethod
    def _compute_pipeline_id(name: str, nodes: tuple[Node, ...]) -> str:
        """Deterministic 16-hex id from pipeline name + node composition."""
        parts = [name]
        for n in nodes:
            parts.append(f"{n.name}:{n.task_id}:{type(n.task).__name__}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    @staticmethod
    def _derive_edges(nodes: tuple[Node, ...]) -> tuple[Edge, ...]:
        """Derive edges from key-level reads/writes declarations.

        When all nodes have empty reads and writes, edges reflect simple
        linear dependency (node *i* → node *i+1*).
        """
        edges: list[Edge] = []
        any_declared = any(n.reads or n.writes for n in nodes)

        if not any_declared:
            # Linear chain: each node depends on the previous
            for i in range(len(nodes) - 1):
                edges.append(Edge(nodes[i].name, nodes[i + 1].name, frozenset()))
            return tuple(edges)

        # Key-based dependency: for each pair (A before B), if A writes
        # something B reads, add an edge.
        for i, src in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                tgt = nodes[j]
                shared = src.writes & tgt.reads
                if shared:
                    edges.append(Edge(src.name, tgt.name, shared))
        return tuple(edges)

    @staticmethod
    def _topo_sort(nodes: tuple[Node, ...], edges: tuple[Edge, ...]) -> tuple[Node, ...]:
        """Topological sort of nodes by edge dependencies.

        When there are no key-declared edges, returns nodes in registration
        order (linear chain). Otherwise performs a standard Kahn topological
        sort.
        """
        if not edges:
            return nodes

        name_to_node = {n.name: n for n in nodes}
        in_degree: dict[str, int] = {n.name: 0 for n in nodes}
        adj: dict[str, list[str]] = {n.name: [] for n in nodes}

        for e in edges:
            adj[e.from_node].append(e.to_node)
            in_degree[e.to_node] += 1

        queue = [n for n in nodes if in_degree[n.name] == 0]
        result: list[Node] = []

        while queue:
            current = queue.pop(0)
            result.append(current)
            for neighbor_name in adj[current.name]:
                in_degree[neighbor_name] -= 1
                if in_degree[neighbor_name] == 0:
                    neighbor = name_to_node[neighbor_name]
                    # Insert after the dependent node's position for stability
                    queue.append(neighbor)

        if len(result) != len(nodes):
            missing = set(nodes) - set(result)
            raise ValueError(
                f"Cycle detected in pipeline {nodes[0].name if nodes else '?'}. "
                f"Unresolved nodes: {[n.name for n in missing]}"
            )

        return tuple(result)
