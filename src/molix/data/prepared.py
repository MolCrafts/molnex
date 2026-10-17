"""One dataset row, prepared once and concatenated on every later batch.

Collating molpy frames walks every column through the interpreter, and the
cutoff graph and the covalent relation are pure functions of a row that does
not change between epochs. The first read pays that work and stores the
columns. Later reads concatenate the stored columns.

Pair order is the per-row list, concatenated. One
:class:`~molix.neighbor.NeighborList` launch over a whole batch appends every
reverse edge after every forward edge, so the two orders differ when the list
is bidirectional. Both are the same cutoff graph: each edge still carries its
own endpoints, displacement and relation.
"""

from __future__ import annotations

import errno
import fcntl
import json
import mmap
import os
import struct
import time
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from molix.data.collate import OFFSET_TARGETS, collate_molecules, rebase_kind
from molix.data.collator import Prepare
from molix.data.dataset import BaseDataset
from molix.frame import TensorFrame, as_tensor_frame

_MAGIC = b"PRP2"
_U32 = struct.Struct("<I")


class _Column:
    """One leaf of a prepared row."""

    __slots__ = ("cat", "data", "name", "ns", "rebase")

    def __init__(self, ns: str, name: str, data: np.ndarray, cat: int, rebase: str) -> None:
        self.ns = ns
        self.name = name
        self.data = data
        self.cat = cat
        self.rebase = rebase

    @property
    def key(self) -> tuple[str, str]:
        """``(namespace, column)``."""
        return (self.ns, self.name)


def _align(n: int) -> int:
    """Next multiple of 8. Column bytes are read with ``frombuffer``."""
    return (n + 7) & ~7


def _as_record(sample: Any) -> tuple[Any, ...]:
    """The frames of one dataset row, in order."""
    if hasattr(sample, "keys") and "atoms" in sample:
        return (sample,)
    return tuple(sample)


def _shift(merged: np.ndarray, offsets: np.ndarray, counts: np.ndarray, axis: int) -> None:
    """Add each row's *offsets* to its slice of *merged*, in place."""
    if merged.size == 0:
        return
    shift = np.repeat(offsets, counts)
    if axis == 0:
        shape = (merged.shape[0],) + (1,) * (merged.ndim - 1)
    elif axis == 1:
        shape = (1, merged.shape[1]) + (1,) * (merged.ndim - 2)
    else:
        raise ValueError(f"a prepared column concatenates on axis 0 or 1, got {axis}")
    merged += shift.reshape(shape)


class BatchTaskLike(Protocol):
    """What a per-row or per-batch task is to this module: an id and ``execute``."""

    @property
    def task_id(self) -> str: ...

    def execute(self, batch: Any) -> Any: ...


class RowStore(Protocol):
    """Where packed rows are kept, by parent-dataset index (:class:`PreparedMmap`)."""

    def get(self, index: int) -> bytes | memoryview | None: ...

    def put(self, index: int, payload: bytes) -> None: ...


class PreparedRow:
    """The columns one dataset row contributes to a collated batch.

    Indices are local to the row. :meth:`assemble` shifts them into the batch.
    Built by :meth:`capture` from frames, or restored by :meth:`unpack`.

    Args:
        columns: Leaves, in the order the batch namespace should emit them.
        n_atoms: Atoms in the row.
        n_graphs: Graphs (frames) in the row.
        n_off: Rows of each namespace an ``hpair_`` column indexes.
        ns_rank: ``len(batch_size)`` of each namespace. ``0`` is the bond
            namespace, whose ``bond_index`` is ``(2, N)`` and is not a batch
            axis.
    """

    def __init__(
        self,
        columns: Sequence[_Column],
        *,
        n_atoms: int,
        n_graphs: int,
        n_off: dict[str, int],
        ns_rank: dict[str, int],
        keep: bytes | memoryview | None = None,
        family: _Family | None = None,
    ) -> None:
        self.columns = tuple(columns)
        self.n_atoms = int(n_atoms)
        self.n_graphs = int(n_graphs)
        self.n_off = dict(n_off)
        self.ns_rank = dict(ns_rank)
        # ``frombuffer`` views die with the bytes they read. Hold them.
        self._keep = keep
        # Rows restored together share one family object. Assemble then skips
        # rebuilding each row's column identity.
        self._family = family

    def signature(self) -> tuple[Any, ...]:
        """Column identity, ignoring the length along the concat axis.

        Two rows of one batch must agree on it. Lengths differ; schema does not.
        """
        frozen: list[tuple[Any, ...]] = []
        for column in self.columns:
            shape = list(column.data.shape)
            if shape:
                shape[column.cat] = -1
            frozen.append(
                (
                    column.ns,
                    column.name,
                    column.cat,
                    column.rebase,
                    column.data.dtype.str,
                    tuple(shape),
                )
            )
        return (
            tuple(frozen),
            tuple(sorted(self.n_off)),
            tuple(sorted(self.ns_rank.items())),
        )

    @classmethod
    def capture(
        cls, sample: Any, *, prepare: Prepare | None, tasks: Sequence[BatchTaskLike]
    ) -> PreparedRow:
        """Prepare one dataset row and keep the columns the batch concatenates.

        *prepare* is ``execute_record``, the same transform the frame collator
        runs. *tasks* are the per-row batch tasks (the neighbour list, the
        covalent relation) in the order the loader mounts them. Each task sees
        one row, so a task whose answer depends on the rest of the batch must
        not be passed here.

        Args:
            sample: A frame, or a sequence of frames that share one dataset row.
            prepare: Object with ``execute_record(frames) -> frames``, or
                ``None`` when *sample* is already prepared.
            tasks: Per-row tasks, each ``execute(batch) -> batch``.

        Returns:
            The row, with indices local to it.
        """
        record = _as_record(sample)
        if prepare is not None:
            record = tuple(prepare.execute_record(record))
        # One sample, so every frame of the record shares graphs.row.
        batch = collate_molecules([record])
        for task in tasks:
            batch = task.execute(batch)
        return cls.from_batch(batch)

    @classmethod
    def from_batch(cls, batch: TensorDictBase) -> PreparedRow:
        """Copy the leaves of a one-row batch out of the frame that built them.

        Args:
            batch: Output of :func:`~molix.data.collate.collate_molecules` on
                one dataset row, after the per-row tasks.

        Returns:
            A row that owns its arrays.
        """
        columns: list[_Column] = []
        ns_rank: dict[str, int] = {}
        for ns in batch.keys():
            if not isinstance(ns, str):
                continue
            block = batch.get(ns)
            if not isinstance(block, TensorDictBase):
                continue
            ns_rank[ns] = len(tuple(block.batch_size))
            for name in block.keys():
                if not isinstance(name, str):
                    continue
                tensor = block.get(name)
                if not torch.is_tensor(tensor):
                    continue
                rebase, cat = rebase_kind(ns, name)
                array = np.array(tensor.detach().cpu().numpy(), copy=True)
                columns.append(_Column(ns, name, array, cat, rebase))
        n_off = {}
        for target in OFFSET_TARGETS:
            block = batch.get(target)
            if isinstance(block, TensorDictBase) and block.batch_size:
                n_off[target] = int(block.batch_size[0])
            else:
                n_off[target] = 0
        graphs = batch.get("graphs")
        if isinstance(graphs, TensorDictBase) and graphs.batch_size:
            n_graphs = int(graphs.batch_size[0])
        else:
            n_graphs = 0
        return cls(
            columns,
            n_atoms=int(batch["atoms"].batch_size[0]),
            n_graphs=n_graphs,
            n_off=n_off,
            ns_rank=ns_rank,
        )

    def pack(self) -> bytes:
        """The bytes a row store keeps for this row.

        The schema (column names, dtypes, which axis grows) is identical for
        every row of one dataset, so it is one JSON object. The lengths that
        differ per row follow it as integers. :meth:`unpack_many` parses the
        schema once per batch.

        Returns:
            A ``PRP2`` payload. :meth:`unpack` is the inverse.
        """
        schema = _schema_json(self)
        keys = _schema_dict(self)["n_off"]
        ints = [self.n_atoms, self.n_graphs]
        ints.extend(int(self.n_off[key]) for key in keys)
        for column in self.columns:
            if column.data.shape:
                ints.append(int(column.data.shape[column.cat]))
        blob = bytearray(_MAGIC)
        blob += _U32.pack(len(schema))
        blob += schema
        blob += struct.pack("<" + "I" * len(ints), *ints)
        blob += b"\x00" * (_align(len(blob)) - len(blob))
        for column in self.columns:
            raw = np.ascontiguousarray(column.data).tobytes()
            blob += raw
            blob += b"\x00" * (_align(len(blob)) - len(blob))
        return bytes(blob)

    @classmethod
    def unpack(cls, payload: bytes | memoryview) -> PreparedRow:
        """The row :meth:`pack` stored.

        Args:
            payload: Bytes previously returned by :meth:`pack`, or a view of them.

        Returns:
            The row. Column arrays are views of *payload*.
        """
        return cls.unpack_many([payload])[0]

    @classmethod
    def unpack_many(cls, payloads: Sequence[bytes | memoryview]) -> list[PreparedRow]:
        """Rows stored by :meth:`pack`, parsing a repeated schema only once.

        Column arrays are views of each payload. A payload that is a memory
        view of a mapped file stays a view: the caller copies it in
        :meth:`assemble`.

        Args:
            payloads: One payload per row, in batch order.

        Returns:
            One row per payload. An empty input is an empty list.

        Raises:
            ValueError: A payload is not a prepared row, or a column runs past it.
        """
        if not payloads:
            return []
        schema, end = _schema_bytes(payloads[0])
        family = _Family.from_json(schema)
        rows: list[PreparedRow] = []
        for payload in payloads:
            got, got_end = _schema_bytes(payload)
            owner = family if got == schema else _Family.from_json(got)
            rows.append(_materialize(payload, got_end, owner))
        return rows

    @classmethod
    def assemble(cls, rows: Sequence[PreparedRow]) -> TensorFrame:
        """Concatenate *rows* into the batch the frame collator would have built.

        Args:
            rows: Prepared rows of one batch, in batch order. Every row must
                carry the same columns.

        Returns:
            The typed batch, before padding and before batch-level tasks.

        Raises:
            ValueError: *rows* is empty, or two rows differ in columns.
        """
        if not rows:
            raise ValueError("Cannot collate an empty frame list")
        family = rows[0]._family
        same = family is not None and all(row._family is family for row in rows)
        if not same:
            expected = rows[0].signature()
            for row in rows[1:]:
                if row.signature() != expected:
                    raise ValueError(
                        "prepared rows in one batch differ in columns; a batch is "
                        "heterogeneous in size, never in schema"
                    )
        n_rows = len(rows)
        atom_off = _exclusive_cumsum([row.n_atoms for row in rows])
        graph_off = _exclusive_cumsum([row.n_graphs for row in rows])
        named_off = {
            target: _exclusive_cumsum([row.n_off.get(target, 0) for row in rows])
            for target in OFFSET_TARGETS
        }
        built: dict[str, dict[str, torch.Tensor]] = {}
        template = rows[0]
        for index, column in enumerate(template.columns):
            parts = [row.columns[index].data for row in rows]
            if column.rebase == "row":
                counts = np.fromiter(
                    (part.shape[0] for part in parts), dtype=np.int64, count=n_rows
                )
                merged = np.repeat(np.arange(n_rows, dtype=parts[0].dtype), counts)
            else:
                merged = np.concatenate(parts, axis=column.cat)
                if column.rebase == "atom":
                    _shift(merged, atom_off, _counts(parts, column.cat), column.cat)
                elif column.rebase == "graph":
                    _shift(merged, graph_off, _counts(parts, column.cat), column.cat)
                elif column.rebase.startswith("off:"):
                    target = column.rebase.split(":", 1)[1]
                    _shift(merged, named_off[target], _counts(parts, column.cat), column.cat)
            built.setdefault(column.ns, {})[column.name] = torch.from_numpy(merged)
        out: dict[str, TensorDict] = {}
        for ns, cols in built.items():
            rank = template.ns_rank[ns]
            if rank == 1 and any(column.cat != 0 for column in template.columns if column.ns == ns):
                raise ValueError(f"{ns} concatenates on an axis other than its batch axis")
            batch_size = [] if rank == 0 else [int(next(iter(cols.values())).shape[0])]
            out[ns] = TensorDict(cols, batch_size=batch_size)
        return as_tensor_frame(TensorDict(out, batch_size=[]))


class _Spec:
    """One column of a prepared-row schema. Lengths along ``cat`` vary per row."""

    __slots__ = ("cat", "dtype", "itemsize", "name", "ns", "rank", "rebase", "tail", "tail_count")

    def __init__(self, spec: dict[str, Any]) -> None:
        self.ns = str(spec["ns"])
        self.name = str(spec["name"])
        self.dtype = np.dtype(spec["dtype"])
        self.itemsize = int(self.dtype.itemsize)
        self.cat = int(spec["cat"])
        self.rebase = str(spec["rebase"])
        self.rank = int(spec["rank"])
        self.tail = tuple(int(dim) for dim in spec["tail"])
        count = 1
        for dim in self.tail:
            count *= dim
        self.tail_count = count


class _Family:
    """The schema shared by every row of one dataset."""

    __slots__ = ("ints", "n_off_keys", "ns_rank", "schema_bytes", "specs")

    def __init__(
        self,
        schema: bytes,
        specs: tuple[_Spec, ...],
        n_off_keys: tuple[str, ...],
        ns_rank: dict[str, int],
    ) -> None:
        variable = sum(spec.rank > 0 for spec in specs)
        self.ints = struct.Struct("<" + "I" * (2 + len(n_off_keys) + variable))
        self.n_off_keys = n_off_keys
        self.ns_rank = ns_rank
        self.schema_bytes = schema
        self.specs = specs

    @classmethod
    def from_json(cls, schema: bytes) -> _Family:
        header = json.loads(schema)
        specs = tuple(_Spec(spec) for spec in header["columns"])
        keys = tuple(str(key) for key in header["n_off"])
        ns_rank = {str(key): int(value) for key, value in header["ns_rank"].items()}
        return cls(schema, specs, keys, ns_rank)


def _schema_dict(row: PreparedRow) -> dict[str, Any]:
    """Column identity of *row*, without the per-row lengths."""
    columns = []
    for column in row.columns:
        shape = column.data.shape
        tail = [int(dim) for axis, dim in enumerate(shape) if axis != column.cat]
        columns.append(
            {
                "cat": column.cat,
                "dtype": column.data.dtype.str,
                "name": column.name,
                "ns": column.ns,
                "rank": len(shape),
                "rebase": column.rebase,
                "tail": tail,
            }
        )
    return {"columns": columns, "n_off": sorted(row.n_off), "ns_rank": row.ns_rank}


def _schema_json(row: PreparedRow) -> bytes:
    """Canonical schema bytes. Equal rows of one dataset share them."""
    return json.dumps(_schema_dict(row), separators=(",", ":"), sort_keys=True).encode()


def _schema_bytes(payload: bytes | memoryview) -> tuple[bytes, int]:
    """The schema JSON of one payload, and the offset just past it."""
    if len(payload) < 8 or payload[: len(_MAGIC)] != _MAGIC:
        raise ValueError("prepared row does not start with its magic")
    schema_len = _U32.unpack_from(payload, len(_MAGIC))[0]
    end = 8 + schema_len
    if end > len(payload):
        raise ValueError("prepared row ends inside its header")
    return bytes(payload[8:end]), end


def _materialize(payload: bytes | memoryview, end: int, family: _Family) -> PreparedRow:
    """One row whose columns view *payload*."""
    if end + family.ints.size > len(payload):
        raise ValueError("prepared row ends inside its header")
    ints = family.ints.unpack_from(payload, end)
    at = _align(end + family.ints.size)
    n_off = {key: int(ints[2 + slot]) for slot, key in enumerate(family.n_off_keys)}
    cursor = 2 + len(family.n_off_keys)
    columns: list[_Column] = []
    for spec in family.specs:
        if spec.rank == 0:
            shape: tuple[int, ...] = ()
            count = 1
        else:
            length = int(ints[cursor])
            cursor += 1
            dims = list(spec.tail)
            dims.insert(spec.cat, length)
            shape = tuple(dims)
            count = length * spec.tail_count
        nbytes = count * spec.itemsize
        if at + nbytes > len(payload):
            raise ValueError("prepared row ends inside a column")
        array = np.frombuffer(payload, dtype=spec.dtype, count=count, offset=at).reshape(shape)
        at = _align(at + nbytes)
        columns.append(_Column(spec.ns, spec.name, array, spec.cat, spec.rebase))
    if at != len(payload):
        raise ValueError("prepared row has bytes after its last column")
    return PreparedRow(
        columns,
        n_atoms=int(ints[0]),
        n_graphs=int(ints[1]),
        n_off=n_off,
        ns_rank=family.ns_rank,
        keep=payload,
        family=family,
    )


class PreparedMmap:
    """Prepared-row payloads in one file, read back through a mapping.

    The index is ``(offset, length)`` per parent-dataset row. A missing row is
    ``offset < 0``. Readers map the data file and return a view of the slice;
    nothing copies the payload out of the mapping. The first write creates the
    index, and a later write of the same row leaves the first payload in place.

    Args:
        path: Directory that holds ``index.i64``, ``data.bin`` and ``lock``.
        n_rows: Length of the parent dataset. Cache keys are that dataset's
            indices, so a train split and a val split share this file.
    """

    def __init__(self, path: str | os.PathLike[str], n_rows: int) -> None:
        self.path = os.fspath(path)
        self.n_rows = int(n_rows)
        self._index: np.memmap | None = None
        self._data: mmap.mmap | None = None
        self._data_size = -1

    def get(self, index: int) -> memoryview | None:
        """The payload at *index*, or ``None`` when that row was never written.

        The view aliases the mapped file. It stays valid while this store does.
        """
        index = int(index)
        index_path = os.path.join(self.path, "index.i64")
        if not os.path.exists(index_path):
            return None
        table = self._map_index()
        offset = int(table[index, 0])
        length = int(table[index, 1])
        if offset < 0 or length < 0:
            return None
        mapped = self._map_data()
        if offset + length > len(mapped):
            mapped = self._map_data(force=True)
        return memoryview(mapped)[offset : offset + length]

    def put(self, index: int, payload: bytes) -> None:
        """Append *payload* and publish its slice. A second put of *index* does nothing."""
        index = int(index)
        if index < 0 or index >= self.n_rows:
            raise IndexError(f"index {index} is outside a prepared cache of length {self.n_rows}")
        lock = self._lock()
        try:
            self._create_index()
            table = self._map_index()
            if int(table[index, 0]) >= 0:
                return
            data_path = os.path.join(self.path, "data.bin")
            with open(data_path, "ab") as handle:
                handle.seek(0, os.SEEK_END)
                start = handle.tell()
                pad = (-start) & 7
                if pad:
                    handle.write(b"\x00" * pad)
                    start += pad
                handle.write(payload)
                handle.flush()
            table[index, 0] = start
            table[index, 1] = len(payload)
            table.flush()
            self._data_size = -1
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def close(self) -> None:
        """Drop the mapping. A later :meth:`get` maps the file again."""
        mapped = self._data
        self._data = None
        self._data_size = -1
        if mapped is not None:
            mapped.close()
        self._index = None

    def rows_present(self) -> int:
        """How many parent rows are stored; ``0`` before the first write."""
        if not os.path.exists(os.path.join(self.path, "index.i64")):
            return 0
        return int((self._map_index()[:, 0] >= 0).sum())

    def _lock(self):
        os.makedirs(self.path, exist_ok=True)
        handle = open(os.path.join(self.path, "lock"), "a+b")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def _create_index(self) -> None:
        path = os.path.join(self.path, "index.i64")
        nbytes = self.n_rows * 16
        if os.path.exists(path):
            found = os.path.getsize(path)
            if found != nbytes:
                raise ValueError(
                    f"prepared cache index at {path} has {found} bytes, expected {nbytes}"
                )
            return
        tmp = path + ".tmp"
        np.full((self.n_rows, 2), -1, dtype="<i8").tofile(tmp)
        os.replace(tmp, path)

    def _map_index(self) -> np.memmap:
        if self._index is None:
            self._index = np.memmap(
                os.path.join(self.path, "index.i64"),
                dtype="<i8",
                mode="r+",
                shape=(self.n_rows, 2),
            )
        return self._index

    def _map_data(self, *, force: bool = False) -> mmap.mmap:
        path = os.path.join(self.path, "data.bin")
        mapped: mmap.mmap | None = None
        size = -1
        # Several jobs append this file. A path stat can come back larger than
        # the size the fd we then open actually has, and mmap refuses a length
        # past the fd. Size and map come from the same fd, and a short file is
        # retried until the append is visible.
        for attempt in range(8):
            fd = os.open(path, os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                if size > 0:
                    if not force and self._data is not None and size == self._data_size:
                        return self._data
                    mapped = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
                    break
            except ValueError:
                mapped = None
            finally:
                os.close(fd)
            time.sleep(0.01 * (attempt + 1))
        if mapped is None:
            raise ValueError(f"prepared cache {path} could not be mapped (size {size})")
        previous = self._data
        self._data = mapped
        self._data_size = size
        if previous is not None:
            try:
                previous.close()
            except BufferError:
                # A get() view still aliases the shorter mapping. That view
                # keeps it alive; closing here is what the workers hit while
                # several processes append to the same data file.
                pass
        return mapped


def _counts(parts: Sequence[np.ndarray], axis: int) -> np.ndarray:
    """Length of each part along *axis*."""
    return np.fromiter((part.shape[axis] for part in parts), dtype=np.int64, count=len(parts))


def _exclusive_cumsum(lengths: Sequence[int]) -> np.ndarray:
    """Start offset of each row. The first entry is 0."""
    counts = np.asarray(lengths, dtype=np.int64)
    return np.cumsum(counts) - counts


class PreparedRows(BaseDataset):
    """A dataset whose rows are :class:`PreparedRow` after the first read.

    The loader's collate sees prepared rows and concatenates them. The wrapped
    dataset is read only for a row the store does not have. Cache keys are the
    parent dataset's indices, so a train split and a val split of one shard
    share the file. Table lengths (:meth:`count_rows`) are the wrapped
    dataset's, and the wrapped dataset itself is :attr:`dataset`.

    Args:
        dataset: The dataset the loader would have read. A subset view is
            unwrapped for the cache key and the fetch.
        prepare: ``execute_record`` transform, or ``None``.
        tasks: Per-row batch tasks, in loader order.
        store: ``get(index) -> bytes | None`` and ``put(index, payload)``.
            ``None`` recomputes every read and does not save it.

    Example::

        rows = PreparedRows(train, prepare=prepare, tasks=(nlist, relation), store=cache)
        batch = PreparedRow.assemble(rows.__getitems__(indices))
    """

    def __init__(
        self,
        dataset: BaseDataset,
        *,
        prepare: Prepare | None,
        tasks: Sequence[BatchTaskLike],
        store: RowStore | None = None,
    ) -> None:
        self._dataset = dataset
        self._prepare = prepare
        self._tasks = tuple(tasks)
        self._store = store

    @property
    def tasks(self) -> tuple[BatchTaskLike, ...]:
        """The tasks every row is captured with; the loader skips exactly these at collate."""
        return self._tasks

    @property
    def store(self) -> RowStore | None:
        """Where the packed rows are kept, or ``None`` when every read prepares afresh."""
        return self._store

    @property
    def dataset(self) -> BaseDataset:
        """The dataset the rows are prepared from."""
        return self._dataset

    def count_rows(self, table: str) -> torch.Tensor:
        """The wrapped dataset's per-row lengths of *table*.

        Read through the wrapped dataset, not through ``self[i]``: a prepared
        row is not a Frame, and a dataset that answers from a count sidecar
        keeps doing so.
        """
        return BaseDataset.nrows_of(self._dataset, table)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> PreparedRow:
        """The prepared row at *index* of this view.

        Args:
            index: Position in this view. A negative index counts from the end.

        Returns:
            The prepared row.

        Raises:
            TypeError: *index* is not an integer.
        """
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError(f"index must be an int, got {type(index).__name__}")
        return self.__getitems__([int(index)])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[PreparedRow]:
        """Prepared rows at *indices*, in that order.

        A missing store entry is built from the wrapped dataset and written
        before it is returned. A present entry does not touch the wrapped
        dataset.

        Args:
            indices: Positions in this view. Negative indices count from the end.

        Returns:
            One prepared row per index.

        Raises:
            IndexError: An index is outside this view.
        """
        local = self._local(indices)
        if not local:
            return []
        parent, source = self._parent(local)
        found: dict[int, PreparedRow] = {}
        missing: list[int] = []
        hits: list[tuple[int, bytes | memoryview]] = []
        store = self._store
        for slot, key in enumerate(parent):
            blob = None if store is None else store.get(key)
            if blob is None:
                missing.append(slot)
            else:
                hits.append((slot, blob))
        if hits:
            unpacked = PreparedRow.unpack_many([blob for _, blob in hits])
            for (slot, _), row in zip(hits, unpacked, strict=True):
                found[slot] = row
        if missing:
            keys = [parent[slot] for slot in missing]
            fetched = self._fetch(source, keys)
            for slot, key, sample in zip(missing, keys, fetched, strict=True):
                row = PreparedRow.capture(sample, prepare=self._prepare, tasks=self._tasks)
                if store is not None:
                    try:
                        store.put(key, row.pack())
                    except OSError as error:
                        if error.errno not in (errno.ENOSPC, errno.EIO, errno.EDQUOT):
                            raise
                        # The node scratch filled. Keep the row and stop writing;
                        # the next read of an uncached row rebuilds it.
                        self._store = None
                        store = None
                found[slot] = row
        return [found[slot] for slot in range(len(local))]

    def _local(self, indices: Sequence[int]) -> list[int]:
        n = len(self._dataset)
        local: list[int] = []
        for index in indices:
            item = int(index)
            if item < 0:
                item += n
            if item < 0 or item >= n:
                raise IndexError(f"index {index} is outside a dataset of length {n}")
            local.append(item)
        return local

    def _parent(self, local: Sequence[int]) -> tuple[list[int], Any]:
        """Shard indices and the dataset those indices address.

        A subset view says so through its public surface: ``indices`` and the
        dataset it selects from — ``source`` (molhub) or ``dataset`` (molix,
        torch). Anything else is addressed as it is.
        """
        dataset = self._dataset
        mapping = getattr(dataset, "indices", None)
        source = getattr(dataset, "source", None)
        if source is None:
            source = getattr(dataset, "dataset", None)
        if mapping is not None and source is not None and source is not dataset:
            return [int(mapping[item]) for item in local], source
        return list(local), dataset

    def _fetch(self, source: Any, parent: Sequence[int]) -> list[Any]:
        fetch = getattr(source, "__getitems__", None)
        if callable(fetch):
            return list(fetch(list(parent)))
        return [source[index] for index in parent]
