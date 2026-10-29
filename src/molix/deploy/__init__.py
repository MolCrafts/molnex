"""Engine-neutral deployment of molnex models.

* :mod:`molix.deploy.contract` — the atom-energy calling convention every
  deployed model keeps (:class:`AtomEnergyModel`), the adapter that puts a
  molnex potential on it (:class:`AtomEnergyAdapter`) and its checks.
* :mod:`molix.deploy.metadata` — :class:`DeployMetadata`, the ``{name}.json``
  an engine reads beside the package.
* :mod:`molix.deploy.dressed` — :class:`DressedPotential`, a residual potential
  with its :class:`~molix.data.AtomicDress` added back onto the atom energies,
  so the package reports absolute energy.

A deployment composes three primitives; the format comes from
:mod:`molix.export`::

    from molix.deploy import DeployMetadata, AtomEnergyAdapter, example_inputs
    from molix.deploy.contract import DYNAMIC_SHAPES
    from molix.export import aoti

    model = AtomEnergyAdapter(potential).cuda()
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

from molix.deploy.contract import AtomEnergyAdapter, AtomEnergyModel, example_inputs, verify
from molix.deploy.dressed import DressedPotential
from molix.deploy.metadata import DeployMetadata

__all__ = [
    "DeployMetadata",
    "AtomEnergyAdapter",
    "AtomEnergyModel",
    "DressedPotential",
    "example_inputs",
    "verify",
]
