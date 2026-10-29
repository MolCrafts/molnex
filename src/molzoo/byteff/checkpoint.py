"""ByteFF-26 checkpoint key table (``byteff2@v1.1.0`` names → molnex names).

Spec: ``.claude/specs/rondo-enc-01-graph-encoder.md`` §3. The remap machinery is
:class:`molzoo.checkpoint.CheckpointRemap`; :meth:`molzoo.byteff.potential.ByteFF.load`
builds it over :meth:`ByteFFKeys.table`. Longest prefix wins. The checkpoint's
torch_geometric ``MLP`` stores its linears as ``lins.<k>``; here an MLP that ends
in an activation is a :class:`~molrep.embedding.mlp.FFLayer` (linears at the
even indices of ``layers``) and one that ends in a linear is a
:class:`~molrep.embedding.mlp.DenseMLP` (``FFLayer`` then ``Linear``).

Reference:
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." Chem. Sci. 2025.
    https://arxiv.org/abs/2408.12817
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import ClassVar

__all__ = ["ByteFFKeys"]


class ByteFFKeys:
    """The rename table from a ByteFF-26 member's ``state_dict`` to a :class:`ByteFF`'s.

    Args:
        n_layers: Transformer layers of the encoder being loaded.
        head_layers: Linears in each bonded head's ``pre`` and ``post`` MLP.
        out_layers: Linears in each bonded parameter MLP.
        pair_layers: Linears of the ``LJEs`` MLPs, keyed ``lj_direct_layers``,
            ``bcc_layers``, ``bcc_out_layers`` (the
            :class:`~molzoo.rondo.spec.HeadsSpec` fields).
        graph: Prefix of the encoder in the target module (``"graph."`` on
            :class:`~molzoo.byteff.potential.ByteFF`, ``""`` for a bare encoder).
        heads: Prefix of the head slots (``"heads."`` on ByteFF).
        slots: The bonded head slots the target carries. The checkpoint's
            heads for any other slot are dropped (``()`` for a bare encoder).
    """

    #: ByteFF-26's bonded term → Rondo's head slot (``ByteFF.heads[<slot>]``).
    HEADS: ClassVar[dict[str, str]] = {
        "bond": "bonds",
        "angle": "angles",
        "proper": "propers",
        "improper": "impropers",
    }
    #: ByteFF-26's parameter → ``(head slot, parameter)`` in molnex.
    PARAMS: ClassVar[dict[str, tuple[str, str]]] = {
        "bond_k": ("bonds", "k"),
        "bond_r0": ("bonds", "r0"),
        "angle_k": ("angles", "k"),
        "angle_d0": ("angles", "theta0"),
        "proper_k": ("propers", "k"),
        "improper_k": ("impropers", "k"),
    }
    #: ByteFF-26's ``LJEs`` MLPs → ``(head slot, module)``: the slots of
    #: :class:`~molpot.classical.ByteFFLJ` and
    #: :class:`~molpot.classical.BondChargeCorrection`. Rondo's own pair heads
    #: (``lj``, ``charge``; spec D1, D2) are not in a checkpoint.
    PAIR: ClassVar[dict[str, tuple[str, str, str]]] = {
        "sigma_mlp": ("lj_direct", "sigma", "lj_direct_layers"),
        "epsilon_mlp": ("lj_direct", "epsilon", "lj_direct_layers"),
        "charge_pre_mlp": ("bcc", "pre", "bcc_layers"),
        "charge_out_mlp": ("bcc", "out", "bcc_out_layers"),
    }
    #: Every head slot a ByteFF-26 checkpoint can fill.
    SLOTS: ClassVar[tuple[str, ...]] = (*HEADS.values(), "lj_direct", "bcc")

    _GRAPH: ClassVar[str] = "graph_block."
    _BONDED: ClassVar[str] = "preff_block.pre_layers.PreMMBonded."
    _PAIR: ClassVar[str] = "preff_block.pre_layers.PreLJEs."
    _CONV: ClassVar[dict[str, str]] = {
        "lin_query": "query",
        "lin_key": "key",
        "lin_value": "value",
        "lin_proj": "proj",
        "norm1": "norm1",
        "norm2": "norm2",
        "lin_gate": "gate",
        "lin_edge": "edge_bias",
        "lin_proj_edge": "edge_proj",
        "norm1_edge": "edge_norm1",
        "norm2_edge": "edge_norm2",
    }

    def __init__(
        self,
        *,
        n_layers: int,
        head_layers: int,
        out_layers: int,
        pair_layers: Mapping[str, int],
        graph: str = "graph.",
        heads: str = "heads.",
        slots: Collection[str] = SLOTS,
    ) -> None:
        self.n_layers = n_layers
        self.head_layers = head_layers
        self.out_layers = out_layers
        self.pair_layers = dict(pair_layers)
        self.graph = graph
        self.heads = heads
        self.slots = tuple(slots)

    def table(self) -> dict[str, str | None]:
        """Official key prefix → Rondo name; ``None`` drops the entry."""
        table = self._encoder()
        table.update(self._bonded())
        table.update(self._pair())
        return table

    def _encoder(self) -> dict[str, str | None]:
        feature, graph = f"{self._GRAPH}feature_layer.", self.graph
        table: dict[str, str | None] = {
            f"{feature}atom_embedding": f"{graph}feature.element",
            f"{feature}connectivity_embedding": f"{graph}feature.degree",
            f"{feature}fm_chg_embedding": f"{graph}feature.formal_charge",
            f"{feature}ring_con_embedding": f"{graph}feature.ring_connectivity",
            f"{feature}min_ring_size_embedding": f"{graph}feature.min_ring_size",
            f"{feature}bond_ring_embedding": f"{graph}feature.in_ring",
            f"{feature}bond_order_embedding": f"{graph}feature.bond_order",
            f"{self._GRAPH}gnn_layer.edge_in_lin": f"{graph}edge_in",
            f"{self._GRAPH}gnn_layer.edge_out_lin": f"{graph}edge_out",
            f"{self._GRAPH}gnn_layer.gnn.lin": f"{graph}jk",
        }
        table.update(
            self._ending_in_activation(f"{feature}node_mlp", f"{graph}feature.node_mlp", 2)
        )
        table.update(
            self._ending_in_activation(f"{feature}edge_mlp", f"{graph}feature.edge_mlp", 2)
        )
        for layer in range(self.n_layers):
            official = f"{self._GRAPH}gnn_layer.gnn.convs.{layer}"
            ours = f"{graph}layers.{layer}"
            table.update({f"{official}.{a}": f"{ours}.{b}" for a, b in self._CONV.items()})
            table.update(self._ending_in_linear(f"{official}.ffn", f"{ours}.ffn", 2))
            table.update(self._ending_in_linear(f"{official}.ffn_edge", f"{ours}.edge_ffn", 2))
        return table

    def _bonded(self) -> dict[str, str | None]:
        table: dict[str, str | None] = {}
        for term, slot in self.HEADS.items():
            for part in ("pre", "post"):
                official = f"{self._BONDED}{part}_mlp.{term}"
                if slot in self.slots:
                    ours = f"{self.heads}{slot}.{part}"
                    table.update(self._ending_in_linear(official, ours, self.head_layers))
                else:
                    table[f"{official}."] = None
        for param, (slot, name) in self.PARAMS.items():
            official = f"{self._BONDED}out_mlp.{param}"
            if slot in self.slots:
                ours = f"{self.heads}{slot}.out.{name}"
                table.update(self._ending_in_linear(official, ours, self.out_layers))
            else:
                table[f"{official}."] = None
        return table

    def _pair(self) -> dict[str, str | None]:
        table: dict[str, str | None] = {}
        for official, (slot, name, layers) in self.PAIR.items():
            source = f"{self._PAIR}{official}"
            if slot in self.slots:
                ours = f"{self.heads}{slot}.{name}"
                table.update(self._ending_in_linear(source, ours, self.pair_layers[layers]))
            else:
                table[f"{source}."] = None
        return table

    @staticmethod
    def _ending_in_activation(official: str, ours: str, layers: int) -> dict[str, str | None]:
        """``lins.<k>`` of a PyG ``MLP(plain_last=False)`` → an ``FFLayer``."""
        return {f"{official}.lins.{k}": f"{ours}.layers.{2 * k}" for k in range(layers)}

    @staticmethod
    def _ending_in_linear(official: str, ours: str, layers: int) -> dict[str, str | None]:
        """``lins.<k>`` of a PyG ``MLP`` (``plain_last=True``) → a ``DenseMLP``."""
        table: dict[str, str | None] = {
            f"{official}.lins.{k}": f"{ours}.0.layers.{2 * k}" for k in range(layers - 1)
        }
        table[f"{official}.lins.{layers - 1}"] = f"{ours}.1"
        return table
