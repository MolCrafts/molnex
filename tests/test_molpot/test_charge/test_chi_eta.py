"""ChiEtaHead — mirrors ``src/molpot/charge/chi_eta.py``.

The head predicts the two per-atom parameters the charge equilibration reads:
electronegativity ``χ`` and hardness ``η``. ``η`` sits on the diagonal of the
system that is solved for ``q``, so its positivity is the head's contract, not
the solver's.
"""

from __future__ import annotations

import torch

from molpot.charge import ChiEtaHead


class TestChiEtaHead:
    def test_chi_eta_head_positive_eta(self):
        head = ChiEtaHead(feature_dim=8, hidden_dim=16)
        out = head(torch.randn(5, 8))
        assert out["chi"].shape == (5,)
        assert out["eta"].shape == (5,)
        assert (out["eta"] > 0).all()
