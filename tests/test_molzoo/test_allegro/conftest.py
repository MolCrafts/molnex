"""Shared fixtures for the Allegro product tests: the encoder's cluster, product sizes, weights."""

from __future__ import annotations

import os
from pathlib import Path

from tests.test_molrep.test_encoders.test_allegro.conftest import DTYPE, cluster
from tests.test_molrep.test_encoders.test_allegro.conftest import TINY as ENCODER_TINY

#: The encoder's sizes plus a small readout.
TINY = {**ENCODER_TINY, "readout_mlp_hidden_layers_width": 8}

#: Directory with the official packages and upstream goldens
#: (``allegro_small_default.nequip.zip`` + ``goldens.pt``, … — see
#: ``src/molzoo/specs/allegro.md`` §7.3).
WEIGHTS_DIR = Path(os.environ.get("MOLNEX_ALLEGRO_WEIGHTS_DIR", "/nonexistent"))

__all__ = ["DTYPE", "TINY", "WEIGHTS_DIR", "cluster"]
