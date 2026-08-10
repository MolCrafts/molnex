"""Map type confidences + chemical support → :class:`CoverageRegime`.

Accepts confidences produced by
:meth:`~molrep.heads.type.TypeHead.decode_with_confidence` (or any equivalent
``(type_ids, confidence)`` pair). This module does **not** reimplement
softmax-max.

Policy (defaults ``conf_in=0.8``, ``conf_near=0.5``):

* ``conf < conf_near`` → :attr:`~CoverageRegime.UNKNOWN`
* inside support, ``conf >= conf_in`` → :attr:`~CoverageRegime.IN_SUPPORT`
* inside support, ``conf_near <= conf < conf_in`` →
  :attr:`~CoverageRegime.NEAR_SUPPORT`
* outside support, ``conf >= conf_near`` →
  :attr:`~CoverageRegime.EXTRAPOLATING` (never ``IN_SUPPORT``)
* ``support is None`` → confidence-only (membership treated as in-support)

Reference:
    Spec: learnable-classical-ff-09-provenance
"""

from __future__ import annotations

import math

import torch

from molpot.heads.provenance.regime import CoverageRegime
from molrep.embedding.support import ChemicalSupportIndex

__all__ = ["SupportClassifier"]


class SupportClassifier:
    """Map confidences + support index → :class:`CoverageRegime` per row.

    Args:
        support: Optional :class:`~molrep.embedding.support.ChemicalSupportIndex`.
            When ``None``, classification is confidence-only (membership
            assumed in-support).
        conf_in: Minimum confidence for :attr:`~CoverageRegime.IN_SUPPORT`.
        conf_near: Minimum usable confidence; below →
            :attr:`~CoverageRegime.UNKNOWN`. Mid band
            ``[conf_near, conf_in)`` → :attr:`~CoverageRegime.NEAR_SUPPORT`
            when in support.
    """

    def __init__(
        self,
        support: ChemicalSupportIndex | None,
        *,
        conf_in: float = 0.8,
        conf_near: float = 0.5,
    ) -> None:
        if not 0.0 <= conf_near <= conf_in <= 1.0:
            raise ValueError(
                f"need 0 <= conf_near <= conf_in <= 1; got conf_near={conf_near}, conf_in={conf_in}"
            )
        self._support = support
        self.conf_in = float(conf_in)
        self.conf_near = float(conf_near)

    @property
    def support(self) -> ChemicalSupportIndex | None:
        """Attached chemical support index, if any."""
        return self._support

    def classify(
        self,
        type_ids: torch.Tensor,
        confidence: torch.Tensor,
        *,
        query: torch.Tensor | None = None,
    ) -> list[CoverageRegime]:
        """Classify each row into a coverage regime.

        Args:
            type_ids: Predicted discrete ids ``(m,)`` (long / int). Used for
                membership when ``query`` is omitted and a type-id bank is
                attached.
            confidence: Per-row confidence in ``[0, 1]``, shape ``(m,)``.
                Prefer values from
                :meth:`~molrep.heads.type.TypeHead.decode_with_confidence`.
            query: Optional continuous embeddings ``(m, dim)``. When given,
                membership uses L2 :meth:`~ChemicalSupportIndex.contains`
                instead of type-id lookup.

        Returns:
            List of :class:`CoverageRegime` of length ``m``.
        """
        if type_ids.ndim != 1:
            raise ValueError(f"type_ids must be 1-D; got {tuple(type_ids.shape)}")
        if confidence.shape != type_ids.shape:
            raise ValueError(
                f"confidence shape {tuple(confidence.shape)} != "
                f"type_ids shape {tuple(type_ids.shape)}"
            )

        m = int(type_ids.shape[0])
        if m == 0:
            return []

        in_support = self._membership(type_ids, query=query)
        regimes: list[CoverageRegime] = []
        for i in range(m):
            conf = float(confidence[i].item())
            inside = bool(in_support[i].item())
            regimes.append(self._regime_one(conf, inside))
        return regimes

    def _membership(
        self,
        type_ids: torch.Tensor,
        *,
        query: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._support is None:
            return torch.ones(type_ids.shape[0], dtype=torch.bool, device=type_ids.device)
        if query is not None:
            return self._support.contains(query)
        return self._support.contains_type_ids(type_ids)

    def _regime_one(self, conf: float, inside: bool) -> CoverageRegime:
        # NaN confidence is unusable.
        if math.isnan(conf) or conf < self.conf_near:
            return CoverageRegime.UNKNOWN
        if inside:
            if conf >= self.conf_in:
                return CoverageRegime.IN_SUPPORT
            return CoverageRegime.NEAR_SUPPORT
        # Outside support: usable confidence → extrapolating (never IN_SUPPORT).
        return CoverageRegime.EXTRAPOLATING
