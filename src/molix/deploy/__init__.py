"""Engine-neutral deployment of molnex models.

* :mod:`molix.deploy.contract` — the site-energy calling convention every
  deployed model keeps (:class:`SiteEnergyModel`), the adapter that puts a
  molnex potential on it (:class:`SiteEnergyAdapter`) and its checks.
* :mod:`molix.deploy.metadata` — :class:`DeployMetadata`, the ``{name}.json``
  an engine reads beside the package.

A deployment composes three primitives; the format comes from
:mod:`molix.export`::

    from molix.deploy import DeployMetadata, SiteEnergyAdapter, example_inputs
    from molix.deploy.contract import DYNAMIC_SHAPES
    from molix.export import aoti

    model = SiteEnergyAdapter(potential).cuda()
    inputs = example_inputs([1, 6, 8], 6.0, torch.float32, "cuda")
    model.check(*inputs[:3])
    written = aoti.package(model, inputs, "out/model.pt2", dynamic_shapes=DYNAMIC_SHAPES)
    DeployMetadata(
        format="aoti", cutoff=6.0, species=(1, 6, 8), dtype="float32",
        receptive_field_layers=2,
        op_libraries=DeployMetadata.op_libraries_for(written.op_namespaces),
    ).write("out")
"""

from __future__ import annotations

from molix.deploy.contract import SiteEnergyAdapter, SiteEnergyModel, example_inputs, verify
from molix.deploy.metadata import DeployMetadata

__all__ = [
    "DeployMetadata",
    "SiteEnergyAdapter",
    "SiteEnergyModel",
    "example_inputs",
    "verify",
]
