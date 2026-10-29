"""Unit tests for molrep.encoders.ElementTable."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from molrep.encoders.element_table import ElementTable


class _Stub(ElementTable, nn.Module):
    """The smallest host: a module that registers a table, as every encoder does."""

    def __init__(self, atomic_numbers: list[int]) -> None:
        super().__init__()
        self.register_buffer(
            "z_table", torch.tensor(atomic_numbers, dtype=torch.long), persistent=False
        )


class TestElementTable:
    def test_validate_elements_names_the_unknown_numbers_and_the_table(self):
        with pytest.raises(ValueError, match=r"\[2, 7\].*\[1, 6, 8\]"):
            _Stub([1, 6, 8]).validate_elements(torch.tensor([1, 7, 2, 6]))

    def test_validate_elements_accepts_the_table(self):
        assert _Stub([8, 1]).validate_elements(torch.tensor([1, 8, 1])) is None

    def test_validate_once_spends_the_gate_on_the_first_eager_call(self):
        stub = _Stub([1, 6])
        with pytest.raises(ValueError, match=r"\[9\]"):
            stub._validate_once(torch.tensor([1, 9]))
        assert not stub._elements_validated
        stub._validate_once(torch.tensor([6, 1]))
        assert stub._elements_validated
        stub._validate_once(torch.tensor([9]))

    def test_validate_once_is_folded_out_under_tracing(self, monkeypatch):
        """``is_compiling()`` short-circuits before the host sync; the flag stays unspent."""
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        stub = _Stub([1, 6])
        stub._validate_once(torch.tensor([9]))
        assert not stub._elements_validated
