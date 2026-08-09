"""Offline conversion: OMOL-cueq.model (pickled ScaleShiftMACE) -> a plain state_dict.

``OMOL-cueq.model`` is a full pickled ``mace.modules.models.ScaleShiftMACE``
object, so ``torch.load`` needs ``mace`` + ``e3nn`` importable — packages
MolNex forbids under ``src/``, ``tests/`` and in-repo ``scripts/``
(CLAUDE.md, "Allowed third-party surface"), and which are not installed in
the MolNex toolchain env. This script produces the offline twin of
``matpes_r2scan_cueq_state.pt``: a plain ``name -> tensor`` dump that
``torch.load(..., weights_only=True)`` reads with no third-party classes at
all. It needs neither ``mace`` nor ``e3nn``: :mod:`_stub_unpickler`
substitutes inert ``torch.nn.Module`` stubs for their classes while
unpickling, and cuEquivariance (installed) unpickles for real.

The derived ``omol_cueq_state.pt`` is a throwaway artifact — the source
checkpoint is downloadable (ACEsuit mace-foundations OMol family) and this
script regenerates the dump in one run, so only the code is kept, not the
419 MB output. The ``MOLNEX_MACE_WEIGHTS_DIR``-gated
``TestOfficialOMolWeights`` cases in
``tests/test_molzoo/test_mace/test_checkpoint.py`` skip cleanly while the
dump is absent.

Run once, next to the checkpoint (or pass explicit paths)::

    python scripts/omol_port/convert_omol_to_cueq_state.py \\
        [SOURCE.model] [TARGET_state.pt]

    # defaults: $MOLNEX_MACE_WEIGHTS_DIR/OMOL-cueq.model ->
    #           $MOLNEX_MACE_WEIGHTS_DIR/omol_cueq_state.pt

Verification performed at first conversion (2026-08-09):

* key dialect matches ``matpes_r2scan_cueq_state.pt`` (cueq 0.10.0: flat
  ``.f.m.graphs.*.graph.c*`` constants, ``(1, numel)`` linear weights);
* every one of the 104 ``nn.Parameter`` tensors of the molnex ``MACEOMol``
  finds a home through ``molzoo.mace.checkpoint.OMOL_REMAP`` (which raises
  on an unfilled learnable), with zero unexpected keys;
* ``radial_embedding.bessel_fn.bessel_weights`` sits 2.2120e-07 from the
  analytic ``n*pi/r_max`` init — the fingerprint recorded independently in
  ``src/molzoo/mace/checkpoint.py`` and ``src/molzoo/specs/mace_omol.md``.
"""

import os
import sys
from pathlib import Path

import _stub_unpickler
import torch

weights_dir = Path(os.environ.get("MOLNEX_MACE_WEIGHTS_DIR", "."))
source = Path(sys.argv[1]) if len(sys.argv) > 1 else weights_dir / "OMOL-cueq.model"
target = Path(sys.argv[2]) if len(sys.argv) > 2 else weights_dir / "omol_cueq_state.pt"

model = torch.load(source, map_location="cpu", weights_only=False, pickle_module=_stub_unpickler)
state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
torch.save(state, target)
print(f"wrote {target}: {len(state)} tensors")
