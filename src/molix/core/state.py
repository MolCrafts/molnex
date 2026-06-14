"""Training state and result types for MolNex.

Scalars produced during training live inside the ``train`` / ``eval`` /
``performance`` / ``gpu`` sub-dicts of :class:`TrainState`. **Writes**
must address exactly one namespace at a time (``state["train"]["loss"]
= x``); slash-prefix and tuple-path writes are rejected at
``__setitem__`` so two hooks cannot silently share a flat namespace
and clobber each other.

**Reads** are ergonomic — :class:`TrainState` walks the nesting for
both slash-string (``state["eval/MAE"]``) and tuple-path
(``state[("eval", "MAE")]``) forms in addition to plain top-level
keys. Same goes for ``state.get(...)`` and ``key in state``. The
:func:`resolve` free function exists for use against arbitrary
``Mapping`` instances and accepts the same shapes.

See CLAUDE.md "State namespace contract" for the full rules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Stage(str, Enum):
    """Training stage enumeration."""

    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"
    PREDICT = "predict"


#: Sub-dict namespaces that :class:`TrainState` pre-creates and enforces
#: ownership for.
NAMESPACES: tuple[str, ...] = ("train", "eval", "performance", "gpu")


class TrainState(dict):
    """Training state container with namespace sub-dicts.

    Top-level keys: ``epoch``, ``global_step``, ``stage``,
    ``steps_since_last_eval``, ``best_metric``, plus the four sub-dict
    namespaces ``train``, ``eval``, ``performance``, ``gpu``. Hooks
    write scalars into the appropriate sub-dict.

    Reads accept three equivalent forms:

    * plain top-level key: ``state["epoch"]``
    * slash-string path:   ``state["eval/MAE"]`` → ``state["eval"]["MAE"]``
    * tuple path:          ``state[("eval", "MAE")]`` → same

    The same forms work for ``.get(key, default)`` and ``key in state``.
    Writes still go through nested dict access only — slash and tuple
    writes are rejected so namespace ownership stays explicit.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault("epoch", 0)
        self.setdefault("global_step", 0)
        self.setdefault("stage", Stage.TRAIN)
        self.setdefault("steps_since_last_eval", 0)
        for ns in NAMESPACES:
            if not isinstance(super().get(ns), dict):
                super().__setitem__(ns, {})

    @staticmethod
    def _path_segments(key: Any) -> tuple[str, ...] | None:
        """Return path segments if `key` is a multi-segment path, else None."""
        if isinstance(key, tuple):
            return key
        if isinstance(key, str) and "/" in key:
            return tuple(key.split("/"))
        return None

    def _walk(self, segments: tuple[str, ...]) -> tuple[bool, Any]:
        """Walk ``segments`` from self; return (found, value)."""
        node: Any = self
        for seg in segments:
            if not isinstance(node, Mapping) or seg not in node:
                return False, None
            node = node[seg]
        return True, node

    def __setitem__(self, key, value):
        """Reject slash / tuple writes and non-dict namespace replacements."""
        if isinstance(key, str) and "/" in key:
            ns, _, sub = key.partition("/")
            raise ValueError(
                f"TrainState does not accept slash-prefix keys for writes. "
                f"Write state[{ns!r}][{sub!r}] = ... instead of state[{key!r}] = .... "
                f"See CLAUDE.md 'State namespace contract'."
            )
        if isinstance(key, tuple):
            head = key[0] if key else "<empty>"
            tail = key[1:] if len(key) > 1 else ()
            raise ValueError(
                f"TrainState does not accept tuple-path writes. "
                f"Write state[{head!r}]{''.join(f'[{p!r}]' for p in tail)} = ... "
                f"instead of state[{key!r}] = .... "
                f"See CLAUDE.md 'State namespace contract'."
            )
        if key in NAMESPACES and not isinstance(value, dict):
            raise ValueError(
                f"TrainState[{key!r}] must be a dict (sub-namespace), got {type(value).__name__}."
            )
        super().__setitem__(key, value)

    def __getitem__(self, key):
        """Read a value, walking namespaces for slash / tuple paths.

        A plain top-level key (``"epoch"``) reads directly. A slash-string
        (``"eval/MAE"``) or tuple (``("eval", "MAE")``) walks the
        namespace nesting.

        Args:
            key: Top-level key, slash-string path, or tuple path.

        Returns:
            The resolved value.

        Raises:
            KeyError: ``key`` is absent, or any path segment is missing.
        """
        segments = self._path_segments(key)
        if segments is None:
            return super().__getitem__(key)
        found, value = self._walk(segments)
        if not found:
            raise KeyError(key)
        return value

    def __contains__(self, key) -> bool:
        """Membership test that understands slash / tuple paths.

        Args:
            key: Top-level key, slash-string path, or tuple path.

        Returns:
            ``True`` if the (possibly nested) key resolves, else ``False``.
        """
        segments = self._path_segments(key)
        if segments is None:
            return super().__contains__(key)
        found, _ = self._walk(segments)
        return found

    def get(self, key, default=None):
        """Read a value with a fallback, walking namespaces for paths.

        Args:
            key: Top-level key, slash-string path, or tuple path.
            default: Returned when the key / path does not resolve.

        Returns:
            The resolved value, or *default* on a miss.
        """
        segments = self._path_segments(key)
        if segments is None:
            return super().get(key, default)
        found, value = self._walk(segments)
        return value if found else default

    def increment_step(self) -> None:
        """Advance the ``global_step`` counter by one (in place)."""
        self["global_step"] = self["global_step"] + 1

    def increment_epoch(self) -> None:
        """Advance the ``epoch`` counter by one (in place)."""
        self["epoch"] = self["epoch"] + 1

    def set_stage(self, stage: Stage) -> None:
        """Set the current training :class:`Stage`.

        Args:
            stage: New stage (``TRAIN`` / ``EVAL`` / ``TEST`` / ``PREDICT``).
        """
        self["stage"] = stage

    @property
    def epoch(self) -> int:
        """Current epoch counter (top-level ``epoch`` key)."""
        return self["epoch"]

    @epoch.setter
    def epoch(self, value: int) -> None:
        """Set the ``epoch`` counter."""
        self["epoch"] = value

    @property
    def global_step(self) -> int:
        """Global optimizer-step counter (top-level ``global_step`` key)."""
        return self["global_step"]

    @global_step.setter
    def global_step(self, value: int) -> None:
        """Set the ``global_step`` counter."""
        self["global_step"] = value

    @property
    def stage(self) -> Stage:
        """Current training :class:`Stage` (top-level ``stage`` key)."""
        return self["stage"]

    @stage.setter
    def stage(self, value: Stage) -> None:
        """Set the current :class:`Stage`."""
        self["stage"] = value

    @property
    def steps_since_last_eval(self) -> int:
        """Steps elapsed since the last eval phase (``steps_since_last_eval``)."""
        return self["steps_since_last_eval"]

    @steps_since_last_eval.setter
    def steps_since_last_eval(self, value: int) -> None:
        """Set the ``steps_since_last_eval`` counter."""
        self["steps_since_last_eval"] = value

    @property
    def best_metric(self) -> float | None:
        """Best monitored metric seen so far, or ``None`` if unset.

        Owned by the checkpoint hook; read via :meth:`get` so it returns
        ``None`` before any value has been recorded.
        """
        return self.get("best_metric")

    @best_metric.setter
    def best_metric(self, value: float | None) -> None:
        """Set the ``best_metric`` scalar."""
        self["best_metric"] = value


# ---------------------------------------------------------------------------
# Path helpers (tuple-path → nested lookup / display string)
# ---------------------------------------------------------------------------


#: A path into :class:`TrainState`. A bare ``str`` names a top-level key
#: (``"epoch"``); a ``tuple`` walks the namespace hierarchy
#: (``("train", "loss")``). Used by :class:`molix.core.hooks.Log`,
#: :class:`CheckpointHook`, and the LR scheduler metric lookup.
Path = str | tuple[str, ...]


def resolve(state: Mapping[str, Any], path: Path, default: Any = None) -> Any:
    """Look up ``path`` in ``state``; return ``default`` if any segment misses.

    ``path`` may be a tuple of segments, a slash-separated string
    (``"eval/MAE"``), or a flat single-segment string (``"epoch"``).
    Mirrors :class:`TrainState`'s read behavior so the same call works
    against any ``Mapping`` (e.g. plain ``dict`` snapshots).
    """
    if isinstance(path, str):
        if "/" in path:
            parts: tuple[str, ...] = tuple(path.split("/"))
        else:
            return state.get(path, default)
    else:
        parts = tuple(path)
    node: Any = state
    for part in parts:
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def display(path: Path) -> str:
    """Render a path for user-facing display: ``("train","loss")`` → ``"train/loss"``."""
    if isinstance(path, str):
        return path
    return "/".join(path)


@dataclass
class StepResult:
    """Result from executing a training step."""

    loss: Any = None
    result: Any = None
    logs: Mapping[str, Any] = field(default_factory=dict)
