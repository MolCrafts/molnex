"""Structured errors for force-field export."""

from __future__ import annotations

from molix.ff_export.cases import TranslationCase

__all__ = ["UnsupportedTermError"]


class UnsupportedTermError(ValueError):
    """Raised when an IR bag has no faithful backend translation.

    Attributes:
        term: IR term name (e.g. ``"improper_harmonic"``).
        reason: Human-readable explanation.
        case: Always :attr:`TranslationCase.UNSUPPORTED`.
    """

    def __init__(
        self,
        term: str,
        reason: str,
        *,
        case: TranslationCase = TranslationCase.UNSUPPORTED,
    ) -> None:
        self.term = term
        self.reason = reason
        self.case = case
        super().__init__(f"Unsupported IR term {term!r} ({case.name}): {reason}")
