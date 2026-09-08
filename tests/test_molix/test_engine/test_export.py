"""export_for_lammps — the artifact set written next to the AOTI ``.pt2``.

The AOTInductor compile is the one third-party step here and is stubbed at its
own seam (:meth:`molix.export.Exporter.export_pretraced`): torch refuses that
compile on a CPU target, and it is not what these cases are about. Everything
around it runs unpatched — the exporter is really constructed, the eager
decomposability / force-consistency checks really run on the example system,
and the schema-v3 ``meta.json`` block and the manifest sibling are really
written.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from molix.engine import export as engine_export
from molix.export import Exporter
from tests.test_molix.test_engine.conftest import _FlatEFPotential

#: Model-manifest payload (``ModelManifest.to_dict()`` shape) carried verbatim.
_MANIFEST = {"schema_version": 1, "profile": "A0", "units": "real"}


def _stub_export_pretraced(
    self: Exporter,
    example_inputs: tuple[torch.Tensor, ...],
    path: str | Path,
    *,
    dynamic_shapes: object | None = None,
    inductor_configs: dict[str, object] | None = None,
) -> Path:
    """Stand in for the AOTI compile: write a placeholder and return its path.

    Same signature and return as the real method, so the export directory ends
    up with the file set the real compile would have left.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"stub .pt2: the AOTI compile is not under test here")
    return dest


class TestExportForLammps:
    def test_manifest_sibling_leaves_meta_json_untouched(self, tmp_path, monkeypatch):
        """The manifest lands beside the export; ``meta.json`` is byte-identical.

        The C++ meta reader validates ``doc["lammps"]`` only, so the manifest
        has to travel as its own file — and adding it must not perturb a single
        byte of the schema-v3 block the pair style parses.
        """
        monkeypatch.setattr(Exporter, "export_pretraced", _stub_export_pretraced)
        common = dict(
            species=[1, 6, 8],
            cutoff=4.0,
            adapter="flat",
            device="cpu",
            receptive_field_layers=2,
        )
        with_manifest = engine_export.export_for_lammps(
            _FlatEFPotential(), tmp_path / "with", manifest=_MANIFEST, **common
        )
        without_manifest = engine_export.export_for_lammps(
            _FlatEFPotential(), tmp_path / "without", **common
        )

        sibling = with_manifest / "model.manifest.json"
        assert json.loads(sibling.read_text()) == _MANIFEST
        assert not (without_manifest / "model.manifest.json").exists()
        assert (with_manifest / "model.meta.json").read_bytes() == (
            without_manifest / "model.meta.json"
        ).read_bytes()
