"""PiNet2 property products: the feature body plus a ``molpot`` response head.

:class:`PiNetDipole` (``molpot.heads.DipoleHead``) and
:class:`PiNetPolarizability` (``molpot.heads.ChargeResponseHead``) subclass
:class:`~molzoo.pinet.encoder.PiNetEncoder` exactly as
:class:`~molzoo.pinet.potential.PiNetPotential` does, so each is one flat
module. They pool the per-block tracks of
:meth:`~molzoo.pinet.encoder.PiNetEncoder.layer_features` over the layer axis
(``layer_reduction``) and write every head output as a batch column.

Reference:
    Li et al. "PiNN: Equivariant Neural Network Suite for Modeling
    Electrochemical Systems" J. Chem. Theory Comput. 21, 1382 (2025).
    https://doi.org/10.1021/acs.jctc.4c01570
"""

from __future__ import annotations

from typing import Literal

import torch
from tensordict import TensorDict

from molpot.heads import ChargeResponseHead, DipoleHead
from molzoo.pinet.encoder import PiNetEncoder
from molzoo.pinet.spec import PiNetSpec


class _PiNetProperty(PiNetEncoder):
    """Shared body of the property products: layer pooling and column writes."""

    #: Head key → namespace (``None`` = the batch root, for 0-dim values).
    _COLUMNS: dict[str, str | None] = {}

    def __init__(
        self,
        spec: PiNetSpec,
        layer_reduction: Literal["mean", "sum", "last"],
        *,
        dtype: torch.dtype,
    ) -> None:
        if layer_reduction not in ("mean", "sum", "last"):
            raise ValueError(f"layer_reduction must be mean, sum or last; got {layer_reduction!r}")
        super().__init__(spec, dtype=dtype)
        self.layer_reduction = layer_reduction

    def _pool(self, tracks: list[torch.Tensor]) -> torch.Tensor:
        """Reduce a per-block list ``[(R, ...)] * depth`` over the layer axis."""
        if self.layer_reduction == "last":
            return tracks[-1]
        stacked = torch.stack(tracks, dim=1)
        return stacked.mean(dim=1) if self.layer_reduction == "mean" else stacked.sum(dim=1)

    def _write(self, batch: TensorDict, values: dict[str, torch.Tensor]) -> TensorDict:
        for key, value in values.items():
            namespace = self._COLUMNS[key]
            if namespace is None:
                batch[key] = value
            else:
                batch[namespace, key] = value
        return batch


class PiNetDipole(_PiNetProperty):
    """PiNet2 features → :class:`molpot.heads.DipoleHead`.

    Args:
        spec: The encoder configuration (its head fields are ignored).
        hidden_dim: Head MLP width.
        variant: Dipole decomposition (``DipoleHead`` variants, e.g. ``"ac_ad"``).
        layer_reduction: How the per-block tracks are pooled.
        vector_dipole: Predict the dipole vector (else its norm).
        charge_neutrality: Project atomic charges onto ``graphs.total_charge``
            (``0`` when the batch has none).
        regularization: Report the bond-charge L2 penalty.
        dtype: Floating dtype every parameter and buffer is built at.
    """

    _COLUMNS = {
        "charge_sum_pre_proj": "graphs",
        "charge_sum_post_proj": "graphs",
        "atomic_charges": "atoms",
        "oxidation_charges": "atoms",
        "atomic_dipoles": "atoms",
        "bond_charges": "pairs",
        "bond_dipoles": "pairs",
        "bond_charge_l2": None,
        "molecular_dipole": "graphs",
        "dipole": "graphs",
    }

    def __init__(
        self,
        spec: PiNetSpec = PiNetSpec(),
        *,
        hidden_dim: int = 64,
        variant: str = "ac_ad",
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        vector_dipole: bool = True,
        charge_neutrality: bool = True,
        regularization: bool = True,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(spec, layer_reduction, dtype=dtype)
        c = self.feature_dim
        self.head = DipoleHead(
            node_scalar_dim=c,
            node_vector_dim=c,
            edge_scalar_dim=c * self.n_props,
            edge_vector_dim=c,
            hidden_dim=hidden_dim,
            variant=variant,
            vector_dipole=vector_dipole,
            charge_neutrality=charge_neutrality,
            regularization=regularization,
            dtype=self.dtype,
        )

    def forward(self, batch: TensorDict) -> TensorDict:
        """Write the dipole and its decomposition into ``batch`` in place."""
        layers, edge_index, vectors = self._batch_layers(batch)
        head = self.head
        uses_bonds = head.uses_bc
        keys = batch.keys(include_nested=True)
        return self._write(
            batch,
            head(
                pos=batch["atoms", "pos"],
                atom_batch=batch["atoms", "batch"],
                num_graphs=batch["graphs"].batch_size[0],
                node_scalars=self._pool([layer.p1 for layer in layers]),
                node_vectors=self._pool([layer.p3 for layer in layers]) if self.has_p3 else None,
                edge_scalars=self._pool([layer.block.i1 for layer in layers])
                if uses_bonds
                else None,
                edge_vectors=(
                    self._pool([layer.block.i3 for layer in layers])
                    if uses_bonds and self.has_p3
                    else None
                ),
                pair_index=edge_index if uses_bonds else None,
                pair_diff=vectors if uses_bonds else None,
                oxidation=(
                    batch["atoms", "oxidation"]
                    if head.uses_os and ("atoms", "oxidation") in keys
                    else None
                ),
                total_charge=(
                    batch["graphs", "total_charge"]
                    if head.uses_ac
                    and head.charge_neutrality
                    and ("graphs", "total_charge") in keys
                    else None
                ),
            ),
        )


class PiNetPolarizability(_PiNetProperty):
    """PiNet2 features → :class:`molpot.heads.ChargeResponseHead`.

    The head's element table is the encoder's ``atomic_numbers``.

    Args:
        spec: The encoder configuration (its head fields are ignored).
        variant: Response model (``"localchi"``, ``"local"``, ``"etainv"``,
            ``"eem"``, ``"acks2"``; ``"_iso"`` suffix for isotropic).
        iso: Isotropic polarizability.
        hidden_dim: Head MLP width.
        layer_reduction: How the per-block tracks are pooled.
        epsilon: Head regulariser.
        sigma: Per-element Gaussian charge widths.
        dtype: Floating dtype every parameter and buffer is built at.
    """

    _COLUMNS = {
        "alpha": "graphs",
        "chi": "graphs",
        "atom_mask": "graphs",
        "atom_diag": "atoms",
        "edge_response": "pairs",
        "eta_inv": "graphs",
        "eta": "graphs",
        "eta_e": "graphs",
        "chi_s": "graphs",
        "alpha_iso": "graphs",
    }

    def __init__(
        self,
        spec: PiNetSpec = PiNetSpec(),
        *,
        variant: str = "localchi",
        iso: bool = False,
        hidden_dim: int = 64,
        layer_reduction: Literal["mean", "sum", "last"] = "mean",
        epsilon: float = 0.01,
        sigma: dict[int, float] | None = None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(spec, layer_reduction, dtype=dtype)
        c = self.feature_dim
        self.head = ChargeResponseHead(
            node_scalar_dim=c,
            edge_scalar_dim=c * self.n_props,
            edge_vector_dim=c,
            atom_types=self._spec.atomic_numbers,
            variant=variant,
            iso=iso,
            hidden_dim=hidden_dim,
            epsilon=epsilon,
            sigma=sigma,
            dtype=self.dtype,
        )

    def forward(self, batch: TensorDict) -> TensorDict:
        """Write the polarizability and the response terms into ``batch`` in place."""
        layers, edge_index, vectors = self._batch_layers(batch)
        return self._write(
            batch,
            self.head(
                pos=batch["atoms", "pos"],
                Z=batch["atoms", "Z"],
                atom_batch=batch["atoms", "batch"],
                num_graphs=batch["graphs"].batch_size[0],
                pair_index=edge_index,
                pair_diff=vectors,
                node_scalars=self._pool([layer.p1 for layer in layers]),
                edge_scalars=self._pool([layer.block.i1 for layer in layers]),
                edge_vectors=(
                    self._pool([layer.block.i3 for layer in layers]) if self.has_p3 else None
                ),
            ),
        )
