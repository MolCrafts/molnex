"""Numerics-neutrality smoke for chain step mace-subpackage-restructure-01.

Builds the eight MACE blocks that the step relocates — through their **new**
public import paths — on a fixed 4-node / 5-edge graph with deterministic
``linspace`` inputs, and asserts every block's output ``sum()`` / ``abs().max()``
and ``sorted(state_dict().keys())`` against literals captured on the pre-move
parent commit. A pure move must reproduce them bit-for-bit; any drift here means
the "verbatim relocation" claim is false.

Goldens
-------
Captured by running this script's construction sequence with the OLD import
paths (``molrep.interaction.density`` / ``molrep.interaction.product`` /
``molrep.readout.scalar`` / ``molrep.readout.product`` / ``molzoo.mace``) and
dumping JSON instead of asserting — same seeds, same inputs, same ``.double()``
calls. To re-capture, swap the imports below back and print the statistics.

    capture command : PYTHONPATH=src python capture_01_goldens.py  (old-path variant)
    parent commit   : cf60f99c23b8b322eed14d8c9a6a37e833482787
    torch           : 2.12.1+cpu
    date            : 2026-08-08
    device / dtype  : CPU, float64 (``config.set_precision("fp64")`` + ``.double()``)
    oracle          : self-baseline (molnex at cf60f99) — no third-party oracle

    goldens re-captured 2026-08-09 at e8d6595 + working-tree dtype/init fixes:
    ``_ScalarO3Linear`` N(0,1) init + ``config.ftype`` at construction (see
    commit message); previous values captured at cf60f99 (2026-08-08), which
    reproduces them bit-for-bit. Three blocks moved, each for one reason:

    * ``InteractionBlock`` — ``radial_mlp`` linears were built at the torch
      default fp32 and cast by ``.double()`` afterwards; they are now built at
      ``config.ftype`` directly, so ``kaiming_uniform_`` draws a different
      (fp64) stream from the same ``manual_seed(0)``.
    * ``NonLinearBiasReadout`` — ``_ScalarO3Linear`` now draws its weight from
      ``N(0, 1)`` instead of zero, so the untrained readout is no longer
      identically zero. The bias is still zero-initialised.
    * ``EmbeddingBlock`` — same fp32→fp64 init-stream shift in the node
      embedding. Only the ``node_features`` term of the composite checksum
      moved (−7.378921712535659 → 18.411832194168696); the deterministic
      ``edge_angular`` (5.0) and ``edge_radial`` (3.168506132872447) terms are
      bit-identical, and the weight std stays ≈1 (1.032 → 0.995), so the large
      swing in the *total* is cancellation bookkeeping, not a scale change.

    Unmoved, and re-verified: every block's ``sorted(state_dict().keys())``,
    and the five blocks whose parameters were already fp64 at construction.

Run:
    PYTHONPATH=src python regressions/mace-subpackage-restructure-01-blocks.py
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import torch

from molix import config

config.set_precision("fp64")

from molrep.embedding.mace import EmbeddingBlock  # noqa: E402
from molrep.embedding.node import DiscreteEmbeddingSpec  # noqa: E402
from molrep.interaction.mace.block import InteractionBlock  # noqa: E402
from molrep.interaction.mace.conv import ConvTP  # noqa: E402
from molrep.interaction.mace.density import (  # noqa: E402
    DensityInteraction,
    DensityResidualInteraction,
)
from molrep.readout.mace import (  # noqa: E402
    LinearReadout,
    NonLinearBiasReadout,
    NonLinearReadout,
    ProductHead,
)

RTOL = 1e-12
ATOL = 1e-15  # absolute floor so a golden that lands near zero stays comparable

N, E, F, NUM_RADIAL, L_MAX = 4, 5, 8, 5, 1
SH = "1x0e+1x1o"
TARGET = f"{F}x0e+{F}x1o"
HIDDEN_DIM = 32  # cue.Irreps("O3", irreps_from_l_max(1, 8)).dim = 8*1 + 8*3


class Golden(NamedTuple):
    """One block's captured signature: output statistics + state_dict key list."""

    total: float
    absmax: float
    keys: list[str]


GOLDENS: dict[str, Golden] = {
    "ConvTP": Golden(
        total=-3.8099331628376785,
        absmax=0.125,
        keys=[
            "cue_tp.f.m.graphs.0.graph.c0",
            "cue_tp.f.m.graphs.0.graph.c1",
        ],
    ),
    "InteractionBlock": Golden(
        total=0.07905200345001925,
        absmax=0.07585597226756544,
        keys=[
            "conv_tp.cue_tp.f.m.graphs.0.graph.c0",
            "conv_tp.cue_tp.f.m.graphs.0.graph.c1",
            "linear.f.m.graphs.0.graph.c0",
            "linear.f.m.graphs.0.graph.c1",
            "linear.weight",
            "node_linear.f.m.graphs.0.graph.c0",
            "node_linear.weight",
            "radial_mlp.mlp.0.bias",
            "radial_mlp.mlp.0.weight",
            "radial_mlp.mlp.2.bias",
            "radial_mlp.mlp.2.weight",
            "radial_mlp.mlp.4.bias",
            "radial_mlp.mlp.4.weight",
        ],
    ),
    "DensityInteraction": Golden(
        total=-0.08267766358519069,
        absmax=0.04609357899752259,
        keys=[
            "conv_tp.f.m.graphs.0.graph.c0",
            "conv_tp.f.m.graphs.0.graph.c1",
            "conv_tp_weights.layer0.weight",
            "conv_tp_weights.layer1.weight",
            "density_fn.layer0.weight",
            "linear.f.m.graphs.0.graph.c0",
            "linear.f.m.graphs.0.graph.c1",
            "linear.weight",
            "linear_up.f.m.graphs.0.graph.c0",
            "linear_up.weight",
            "skip_tp.f.m.graphs.0.graph.c0",
            "skip_tp.f.m.graphs.0.graph.c1",
            "skip_tp.weight",
        ],
    ),
    "DensityResidualInteraction": Golden(
        total=0.052798278602837645,
        absmax=0.09087984372039665,
        keys=[
            "conv_tp.f.m.graphs.0.graph.c0",
            "conv_tp.f.m.graphs.0.graph.c1",
            "conv_tp_weights.layer0.weight",
            "conv_tp_weights.layer1.weight",
            "density_fn.layer0.weight",
            "linear.f.m.graphs.0.graph.c0",
            "linear.f.m.graphs.0.graph.c1",
            "linear.weight",
            "linear_up.f.m.graphs.0.graph.c0",
            "linear_up.weight",
            "skip_tp.f.m.graphs.0.graph.c0",
            "skip_tp.weight",
        ],
    ),
    "LinearReadout": Golden(
        total=-0.017091029094867177,
        absmax=0.2238684806385588,
        keys=[
            "linear.f.m.graphs.0.graph.c0",
            "linear.weight",
        ],
    ),
    "NonLinearReadout": Golden(
        total=-0.1969420757337682,
        absmax=0.4829924729613344,
        keys=[
            "linear_1.f.m.graphs.0.graph.c0",
            "linear_1.weight",
            "linear_2.f.m.graphs.0.graph.c0",
            "linear_2.weight",
        ],
    ),
    "NonLinearBiasReadout": Golden(
        # _ScalarO3Linear draws its weight from N(0, 1) and zero-initialises its
        # bias, so the untrained OMOL-variant readout is non-trivial. It used to
        # be identically zero (zero weight *and* zero bias), which silently
        # zeroed every downstream force — that is what the init fix removed.
        total=-0.04351653087296714,
        absmax=0.04499837160305635,
        keys=[
            "linear_1.f.m.graphs.0.graph.c0",
            "linear_1.weight",
            "linear_2.bias",
            "linear_2.weight",
            "linear_mid.bias",
            "linear_mid.weight",
        ],
    ),
    "ProductHead": Golden(
        total=1.29665946085198,
        absmax=0.6573909647520318,
        keys=[
            "linear.bias",
            "linear.weight",
            "symmetric_contraction.symmetric_contraction.f.m.graphs.0.graph.c0",
            "symmetric_contraction.symmetric_contraction.f.m.graphs.1.graph.c0",
            "symmetric_contraction.symmetric_contraction.f.m.graphs.1.graph.c1",
            "symmetric_contraction.symmetric_contraction.f.m.graphs.1.graph.c2",
            "symmetric_contraction.symmetric_contraction.f.m.graphs.1.graph.c3",
            "symmetric_contraction.symmetric_contraction.projection",
            "symmetric_contraction.symmetric_contraction.weight",
        ],
    ),
    "EmbeddingBlock": Golden(
        total=26.580338327041144,
        absmax=2.854573509905571,
        keys=[
            "node_embedding.embedders.0.weight",
            "node_embedding.project.0.f.m.graphs.0.graph.c0",
            "node_embedding.project.0.weight",
        ],
    ),
}


def lin(*shape: int, lo: float = -0.5, hi: float = 0.5) -> torch.Tensor:
    """Deterministic ramp input — no RNG, so the goldens are reproducible."""
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(lo, hi, n, dtype=torch.float64).reshape(*shape)


class Checker:
    """Collects per-block deviations against the embedded goldens."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, module: torch.nn.Module, stats: tuple[float, float]) -> None:
        """Compare one block's (sum, absmax) and state_dict keys to its golden."""
        golden = GOLDENS[name]
        worst = 0.0
        for label, actual, expected in (
            ("sum", stats[0], golden.total),
            ("absmax", stats[1], golden.absmax),
        ):
            dev = abs(actual - expected)
            worst = max(worst, dev)
            if dev > RTOL * abs(expected) + ATOL:
                self.failures.append(
                    f"{name}.{label}: got {actual!r}, want {expected!r} (deviation {dev:.3e})"
                )
        keys = sorted(module.state_dict().keys())
        if keys != golden.keys:
            self.failures.append(
                f"{name}.state_dict keys drifted:\n  got  {keys}\n  want {golden.keys}"
            )
        print(f"  {name:<28} max deviation {worst:.3e}")

    def tensor(
        self,
        name: str,
        module: torch.nn.Module,
        result: torch.Tensor | tuple[torch.Tensor | None, ...],
    ) -> None:
        """Record a block whose output is a tensor (or a tuple whose first item is)."""
        t = result if isinstance(result, torch.Tensor) else result[0]
        assert isinstance(t, torch.Tensor)
        t = t.detach()
        self.check(name, module, (float(t.sum()), float(t.abs().max())))


def main() -> int:
    """Rebuild every relocated block through its new path and verify the goldens."""
    checker = Checker()

    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]]).t().contiguous()  # (E, 2)
    node_attrs = torch.zeros(N, 3, dtype=torch.float64)
    node_attrs[torch.arange(N), torch.tensor([0, 1, 2, 0])] = 1.0
    node_feats = lin(N, F)
    edge_attrs = lin(E, 4)
    edge_feats = lin(E, NUM_RADIAL)
    Z = torch.tensor([1, 6, 8, 1])
    edge_dist = torch.linspace(0.8, 2.5, E, dtype=torch.float64)
    edge_diff = lin(E, 3, lo=-1.0, hi=1.0)

    torch.manual_seed(0)
    conv = ConvTP(in_irreps=f"{F}x0e", out_irreps=TARGET, sh_irreps=SH).double()
    checker.tensor(
        "ConvTP",
        conv,
        conv(
            node_features=lin(N, F),
            edge_angular=edge_attrs,
            edge_index=edge_index,
            tp_weights=lin(E, conv.weight_numel),
        ),
    )

    torch.manual_seed(0)
    blk = InteractionBlock(
        num_features=F, num_bessel=NUM_RADIAL, l_max=L_MAX, avg_num_neighbors=2.0
    ).double()
    checker.tensor(
        "InteractionBlock",
        blk,
        blk(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        ),
    )

    common = dict(
        node_attrs_irreps="3x0e",
        edge_attrs_irreps=SH,
        edge_feats_irreps=f"{NUM_RADIAL}x0e",
        target_irreps=TARGET,
        radial_mlp=[8],
    )

    torch.manual_seed(0)
    di = DensityInteraction(node_feats_irreps=f"{F}x0e", edge_irreps=f"{F}x0e", **common).double()
    checker.tensor(
        "DensityInteraction",
        di,
        di(node_attrs, node_feats, edge_attrs, edge_feats, edge_index),
    )

    torch.manual_seed(0)
    dri = DensityResidualInteraction(
        node_feats_irreps=f"{F}x0e", edge_irreps=f"{F}x0e", hidden_irreps=f"{F}x0e", **common
    ).double()
    checker.tensor(
        "DensityResidualInteraction",
        dri,
        dri(node_attrs, node_feats, edge_attrs, edge_feats, edge_index),
    )

    torch.manual_seed(0)
    lr = LinearReadout(irreps_in=f"{F}x0e").double()
    checker.tensor("LinearReadout", lr, lr(node_feats))

    torch.manual_seed(0)
    nlr = NonLinearReadout(irreps_in=f"{F}x0e", mlp_dim=4).double()
    checker.tensor("NonLinearReadout", nlr, nlr(node_feats))

    torch.manual_seed(0)
    nlbr = NonLinearBiasReadout(irreps_in=f"{F}x0e", mlp_dim=4).double()
    checker.tensor("NonLinearBiasReadout", nlbr, nlbr(node_feats))

    torch.manual_seed(0)
    ph = ProductHead(
        hidden_dim=HIDDEN_DIM,
        out_dim=F,
        num_radial=NUM_RADIAL,
        l_max=L_MAX,
        max_body_order=2,
        num_species=9,
    ).double()
    checker.tensor("ProductHead", ph, ph(node_features=lin(N, HIDDEN_DIM), atom_types=Z))

    torch.manual_seed(0)
    emb = EmbeddingBlock(
        node_attr_specs=[DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=F)],
        num_features=F,
        r_max=5.0,
        num_bessel=NUM_RADIAL,
        l_max=L_MAX,
    ).double()
    nf, ea, ef = emb(Z=Z, edge_dist=edge_dist, edge_diff=edge_diff)
    checker.check(
        "EmbeddingBlock",
        emb,
        (
            float(nf.sum()) + float(ea.sum()) + float(ef.sum()),
            max(float(nf.abs().max()), float(ea.abs().max()), float(ef.abs().max())),
        ),
    )

    if checker.failures:
        print("\nFAILED — the relocation is not numerics-neutral:")
        for failure in checker.failures:
            print(f"  {failure}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
