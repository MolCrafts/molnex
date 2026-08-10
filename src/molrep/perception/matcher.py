"""SmartsMatcher Protocol + Fake / molpy-backed implementations.

Matching returns LongTensor ``[arity, n_hits]`` (COO-style columns) to align
with valence topology conventions. Energy evaluation is out of scope.

molpy only — never bare ``molrs``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch

from molrep.perception.patterns import SymbolicPattern

__all__ = [
    "FakeSmartsMatcher",
    "MolpySmartsMatcher",
    "SmartsMatcher",
]


@runtime_checkable
class SmartsMatcher(Protocol):
    """Protocol for SMARTS/SMIRKS graph matching.

    Implementations return atom-index hits as a LongTensor shaped
    ``[arity, n_hits]`` (column per hit). Empty result is ``[arity, 0]``.
    """

    def match(self, mol: Any, pattern: SymbolicPattern) -> torch.Tensor:
        """Match ``pattern`` against ``mol``.

        Args:
            mol: Molecule graph accepted by the backend (opaque to callers
                that use :class:`FakeSmartsMatcher`; molpy ``Atomistic`` for
                :class:`MolpySmartsMatcher`).
            pattern: Validated symbolic pattern with arity.

        Returns:
            LongTensor of shape ``[pattern.arity, n_hits]``.
        """
        ...


class FakeSmartsMatcher:
    """Deterministic test double: ``pattern_str → matches`` table.

    No molpy required. Used in unit tests that do not need real chemistry.

    Args:
        hits: Mapping from pattern string to a LongTensor of shape
            ``[arity, K]``. Unknown patterns yield empty ``[arity, 0]``.

    Notes:
        Configured tensors must be integer dtype. When a configured tensor's
        leading dimension differs from ``pattern.arity``, :meth:`match` raises
        ``ValueError``.
    """

    def __init__(self, hits: Mapping[str, torch.Tensor] | None = None) -> None:
        self._hits: dict[str, torch.Tensor] = {}
        if hits:
            for key, tensor in hits.items():
                t = torch.as_tensor(tensor)
                if t.ndim != 2:
                    raise ValueError(
                        f"FakeSmartsMatcher hit for {key!r} must be 2-D "
                        f"[arity, K]; got shape {tuple(t.shape)}"
                    )
                if t.dtype not in (
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.long,
                ):
                    t = t.long()
                self._hits[key] = t.long().contiguous()

    def match(self, mol: Any, pattern: SymbolicPattern) -> torch.Tensor:
        """Return preconfigured hits for ``pattern.pattern``, or empty.

        Args:
            mol: Ignored (present for Protocol compatibility).
            pattern: Pattern whose string key is looked up.

        Returns:
            LongTensor ``[arity, K]``.

        Raises:
            ValueError: Configured hit arity disagrees with ``pattern.arity``.
        """
        del mol  # unused — deterministic table lookup
        hit = self._hits.get(pattern.pattern)
        if hit is None:
            return torch.empty((pattern.arity, 0), dtype=torch.long)
        if hit.shape[0] != pattern.arity:
            raise ValueError(
                f"configured hit arity {hit.shape[0]} != pattern.arity "
                f"{pattern.arity} for {pattern.pattern!r}"
            )
        return hit


class MolpySmartsMatcher:
    """SMARTS matching via molpy's public ``SmartsPattern`` surface.

    Soft-imports :class:`molpy.SmartsPattern` only (never bare ``molrs``).
    Hits are converted to LongTensor ``[arity, n_hits]``.

    Raises:
        ImportError: If ``molpy.SmartsPattern`` is unavailable at construction.
    """

    def __init__(self) -> None:
        # Soft import — keep the class importable even if SmartsPattern is
        # missing from a stripped molpy build; construction fails loudly.
        try:
            from molpy import SmartsPattern as _SmartsPattern
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "MolpySmartsMatcher requires molpy.SmartsPattern on the public "
                "API. Install molcrafts-molpy>=0.13; do not import molrs from "
                "molnex."
            ) from exc
        self._SmartsPattern = _SmartsPattern

    def match(self, mol: Any, pattern: SymbolicPattern) -> torch.Tensor:
        """Match ``pattern`` against a molpy molecule graph.

        Args:
            mol: Object accepted by ``SmartsPattern.find_matches`` (typically
                molpy ``Atomistic``).
            pattern: Symbolic pattern; ``pattern.pattern`` is compiled.

        Returns:
            LongTensor ``[arity, n_hits]``. Each column is the first
            ``arity`` atom indices of a SMARTS embedding.

        Raises:
            ValueError: A hit has fewer than ``arity`` atoms.
        """
        compiled = self._SmartsPattern(pattern.pattern)
        raw_hits = compiled.find_matches(mol)
        if not raw_hits:
            return torch.empty((pattern.arity, 0), dtype=torch.long)

        columns: list[list[int]] = []
        for hit in raw_hits:
            atoms = _hit_atom_list(hit)
            if len(atoms) < pattern.arity:
                raise ValueError(
                    f"SMARTS hit has {len(atoms)} atoms but pattern arity is "
                    f"{pattern.arity} ({pattern.pattern!r})"
                )
            columns.append([int(a) for a in atoms[: pattern.arity]])

        # Stack as [arity, K]
        return torch.tensor(columns, dtype=torch.long).T.contiguous()


def _hit_atom_list(hit: Any) -> list[int]:
    """Normalize a molpy/molrs SmartsMatch (or list) to atom indices."""
    if isinstance(hit, (list, tuple)):
        return [int(x) for x in hit]
    if hasattr(hit, "as_list"):
        return [int(x) for x in hit.as_list()]
    if hasattr(hit, "atoms"):
        return [int(x) for x in hit.atoms]
    raise TypeError(f"unsupported SMARTS hit type: {type(hit)!r}")
