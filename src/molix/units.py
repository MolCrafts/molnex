"""Physical constants and shared numerical conventions — the single source.

Every module needing k_B or the (amu, Å, fs) energy bridge imports from here.
Restating a constant locally is exactly the drift this module exists to
prevent: ``KB_EV_PER_K`` used to live in two verbatim copies (``molix.md`` and
``molix.quant``) that could only diverge silently.
"""

from __future__ import annotations

#: Boltzmann constant in eV/K (CODATA 2018).
KB_EV_PER_K: float = 8.617333262e-5

#: Energy-unit bridge: 1 amu·Å²/fs² = 103.6426965638 eV.
EV_PER_AMU_A2_FS2: float = 103.6426965638

#: k_B in the MD integrator's (amu, Å, fs) energy unit (amu·Å²/fs²), so
#: ``T = 2·KE / (dof · KB_AMU_A_FS)`` comes out in kelvin.
KB_AMU_A_FS: float = KB_EV_PER_K / EV_PER_AMU_A2_FS2

#: Padded ("dead") neighbour edges are displaced to ``factor · r_cut`` so every
#: cutoff envelope evaluates to exactly zero. Shared by
#: :class:`molix.md.neighbors.PeriodicNeighborList` and
#: :class:`molix.engine.static.StaticForward` so both padding paths silence
#: dead edges identically.
DEAD_EDGE_CUTOFF_FACTOR: float = 10.0
