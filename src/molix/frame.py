"""The one container the model reads: :class:`TensorFrame`.

MolNex speaks exactly two data shapes. At rest a molecule is a molpy
``Frame`` (``mp.Frame``); in the model it is a ``TensorFrame`` (``mn.TensorFrame``)
— a :class:`~tensordict.TypedTensorDict` whose declared fields *are* the
post-collate batch schema. The names differ so a reader never has to guess
which side of :func:`~molix.data.collate.collate_molecules` they are on.

``TensorFrame`` adds no storage and no behaviour: it is a typed view over a
``TensorDictBase`` backend. :meth:`~tensordict.TypedTensorDict.from_tensordict`
wraps an existing batch **without copying** — the leaves keep sharing storage
with whatever produced them (a mmapped block column, a collate gather), so the
storage↔model seam stays copy-free.

:func:`as_tensor_frame` is the seam between "what a caller happens to hold" —
the plain collated batch, a view of it, or a nested mapping of the namespaces —
and the view every reader in molnex addresses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NotRequired, TypeVar

from tensordict import TensorDict, TensorDictBase, TypedTensorDict

__all__ = ["TensorFrame", "as_tensor_frame"]


T = TypeVar("T")


class TensorFrame(TypedTensorDict):
    """The post-collate batch, with its namespaces declared as fields.

    Fields mirror the schema in ``CLAUDE.md`` — ``atoms`` is the only namespace
    every batch has; the rest depend on what the frames carried and on which
    tasks have run:

    * ``atoms`` — ``batch_size=[N]``: ``Z``, ``pos``, ``batch`` graph
      membership, and any per-atom leaf the frames carried (``forces``,
      ``mass``, ``vel``, …). Encoders write ``node_features`` here.
    * ``pairs`` — ``batch_size=[E]``: ``pair_index`` ``(E, 2)``,
      ``pair_diff``, ``pair_dist``, optional ``shifts``. Written by
      :class:`~molix.neighbor.NeighborList`, not by collation, so it is
      empty until the graph is built. On the MD bind path the batch size is a
      fixed capacity with live edges in ``[0, num_pairs)``.
    * ``graphs`` — ``batch_size=[B]``: ``num_atoms``, optional ``cell``, and
      every per-graph target.
    * ``bonds`` — ``batch_size=[]``: the ``(2, N)`` COO ``bond_index`` and
      ``bond_types``. A cutoff neighbour is not a bond; the differing shape is
      the anti-alias guard.
    * ``angles`` / ``propers`` / ``impropers`` — ``batch_size=[N_terms]``:
      1-D ``atomi``…``atoml`` index columns (improper centre is ``atomi``).

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from molix.frame import TensorFrame
        >>> atoms = TensorDict(
        ...     Z=torch.tensor([1, 8]),
        ...     pos=torch.zeros(2, 3),
        ...     batch=torch.zeros(2, dtype=torch.long),
        ...     batch_size=[2],
        ... )
        >>> frame = TensorFrame(atoms=atoms, batch_size=[])
        >>> frame.atoms["Z"].tolist()
        [1, 8]

        Wrapping an existing batch is a view, not a copy:

        >>> batch = TensorDict(atoms=atoms, batch_size=[])
        >>> view = TensorFrame.from_tensordict(batch)
        >>> view["atoms", "pos"].data_ptr() == batch["atoms", "pos"].data_ptr()
        True
    """

    # ``NotRequired`` is how :class:`~tensordict.TypedTensorDict` declares an
    # optional field (it populates ``__optional_keys__`` from it). ``ty`` reads
    # the class through PEP 681 ``dataclass_transform`` instead, where PEP 655
    # ``NotRequired`` is not a legal field form — an upstream mismatch between
    # the two annotation protocols, not a schema error here.
    atoms: TensorDict
    pairs: NotRequired[TensorDict]  # ty: ignore[invalid-type-form]
    graphs: NotRequired[TensorDict]  # ty: ignore[invalid-type-form]
    bonds: NotRequired[TensorDict]  # ty: ignore[invalid-type-form]
    angles: NotRequired[TensorDict]  # ty: ignore[invalid-type-form]
    propers: NotRequired[TensorDict]  # ty: ignore[invalid-type-form]
    impropers: NotRequired[TensorDict]  # ty: ignore[invalid-type-form]


def as_tensor_frame(system: TensorDictBase | Mapping[str, Any]) -> TensorFrame:
    """The typed view of a post-collate batch — the one reader every system goes through.

    * a :class:`~molix.data.TensorFrame` is already the view and is returned
      unchanged — wrapping a view in a view would only cost an object;
    * any other ``TensorDictBase`` (the plain collated batch, a lazy stack, a
      differently-typed view) is wrapped by
      :meth:`~tensordict.TypedTensorDict.from_tensordict`, which shares every
      leaf's storage — the wrap is a *read*, so a write through the view lands
      in the caller's own batch;
    * a plain nested ``Mapping`` — the namespaces held in a dict — is the same
      system spelled differently, so it becomes a ``TensorDict`` first. The
      sub-namespaces are kept, not copied, so the leaves stay shared; the
      *container* is new, though, so a write through the view does not reach
      the caller's dict — a reader that writes back refuses a dict for that
      reason (:class:`~molix.data.charge.TotalCharge`);
    * anything else is refused. A reader answers ``None`` for a namespace it
      cannot find, so an unrecognised container would not raise on its way
      through — it would price an energy assembled from no topology and no
      pairs, which is the one failure mode that reaches a caller as a number.

    Module-level beside the view, because :class:`TensorFrame` adds no
    behaviour (CLAUDE.md § TensorDict Contract) and the conversion has no other
    owning type. molpot imports it from here — the dependency runs
    ``molix ← molrep ← molzoo / molpot`` — as do molix's own readers
    (:class:`~molix.data.charge.TotalCharge`, :mod:`molix.md.driver`,
    :class:`~molix.ff_export.LammpsWriter`): one spelling, no twin per package.

    ``check=False`` is deliberate: this is a *view*, not a validator. A batch
    with no ``atoms`` namespace is refused by the reader that needs it, which
    is a better message than a wrap-time complaint about a missing field, and
    it keeps a ``pos=``-only evaluation — the ``BasePotential`` contract —
    working on a system that carries nothing else.

    Args:
        system: The batch, as a ``TensorFrame``, any other ``TensorDictBase``,
            or a nested ``Mapping`` of the namespaces.

    Returns:
        The typed view over *system*'s own storage.

    Raises:
        TypeError: *system* is neither a ``TensorDictBase`` nor a ``Mapping``,
            naming what a system is.
    """
    if isinstance(system, TensorFrame):
        return system
    if isinstance(system, TensorDictBase):
        return TensorFrame.from_tensordict(system, check=False)
    if isinstance(system, Mapping):
        return TensorFrame.from_tensordict(TensorDict(dict(system), batch_size=[]), check=False)
    raise TypeError(
        f"a system is the post-collate batch — a TensorDict, a TensorFrame view of one, or a "
        f"nested mapping of its namespaces (atoms / edges / graphs / bonds / angles / propers "
        f"/ impropers) — but this is a {type(system).__name__}. Collate the frames with "
        "molix.data.collate_molecules and pass the batch it returns."
    )


def beside_labels(
    frame: TensorDictBase,
    moves: Mapping[tuple[str, str], tuple[str, str]],
    body: Callable[[], T],
) -> T:
    """Run *body*, which writes label columns, and leave its writes beside the labels.

    A model's ``energy`` writes ``graphs.energy`` and its derivative op
    ``atoms.forces`` — the columns a training batch holds its DFT labels on.
    *moves* maps each such column to where the prediction belongs
    (``ENERGY_KEY → ENERGY_PRED_KEY``): the label is taken before *body*,
    *body*'s write is copied to the target after it, and the label is put
    back. A column that had no label keeps *body*'s write where it was.

    Args:
        frame: The batch *body* writes into.
        moves: Written column → where the prediction goes.
        body: The forward to run; its return value is handed back.

    Returns:
        Whatever *body* returned.
    """
    labels = {key: frame.get(key, None) for key in moves}
    result = body()
    for key, target in moves.items():
        frame[target] = frame[key]
        label = labels[key]
        if label is not None:
            frame[key] = label
    return result
