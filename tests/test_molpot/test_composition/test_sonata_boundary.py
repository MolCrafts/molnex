"""Anti-double-counting boundary — ac-011.

Two assertions in one criterion:

1. **Runtime warning.** A user who hand-builds a parallel
   ``Sonata + Polarization`` composition (i.e. calls Sonata's
   electrostatic path AND ``Polarization.forward`` on the same atoms)
   must see a clear ``UserWarning`` mentioning ``double-count``. This
   surfaces the hazard at runtime even when the construction-time
   refusal in ``Sonata.__init__`` (which only catches
   ``Polarization`` passed as ``short_range_head``) does not fire.

2. **Documentation cross-reference.** ``Polarization`` must
   cross-reference ``Sonata`` in its docstring so a user landing on
   ``Polarization`` discovers the LES α-mode alternative.

If a regression silently lets a parallel composition run without
warning AND without cross-reference, the user's training pipeline will
double-count induction (Polarization's CG Thole solve plus any future
LES α-mode in the EwaldMultipoleEnergy path). The disagreement is
subtle and physics-dependent — the spec mandates surfacing the hazard
as a clear warning, not as numerical disagreement we hope the user
notices.

Per the sub-spec contract (``Out of scope`` block), a failing assertion
here is a real bug to file against ``sonata-01-composer`` (or against
the documentation deliverable) — this test does not modify production
code.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.config import config
from molpot import Polarization
from molpot.composition import Sonata, build_sonata
from molzoo import Allegro

# ---------------------------------------------------------------------------
# Module-local pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sonata_and_batch() -> tuple[Sonata, TensorDict]:
    orig_ftype = config["ftype"]
    config["ftype"] = torch.float64
    try:
        torch.manual_seed(0)
        encoder = Allegro(
            num_elements=10,
            num_scalar_features=16,
            num_tensor_features=4,
            r_max=5.0,
            num_bessel=8,
            num_layers=2,
            l_max=2,
            type_embed_dim=16,
            latent_mlp_depth=1,
            latent_mlp_width=32,
            avg_num_neighbors=12.0,
            expose_tensor_track=True,
        )
        sonata = build_sonata(
            encoder,
            sigma=1.0,
            dl=2.0,
            charge=True,
            dipole=True,
            quadrupole=True,
            constrain_total_charge=True,
            avg_num_neighbors=12.0,
        )
        sonata = sonata.double()
        sonata.eval()

        pos = torch.tensor(
            [
                [0.10, 0.20, 0.05],
                [1.55, 0.15, 0.10],
                [0.85, 1.30, -0.05],
                [-0.65, 1.25, 0.20],
            ],
            dtype=torch.float64,
        )
        Z = torch.tensor([1, 6, 8, 7], dtype=torch.long)
        edge_index = torch.tensor(
            [[i, j] for i in range(4) for j in range(4) if i != j], dtype=torch.long
        )
        bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
        bond_dist = bond_diff.norm(dim=-1)
        batch_idx = torch.zeros(4, dtype=torch.long)
        cell = 10.0 * torch.eye(3, dtype=torch.float64).unsqueeze(0)
        total_charge = torch.zeros(1, dtype=torch.float64)
        num_atoms = torch.tensor([4], dtype=torch.long)

        batch = TensorDict(
            atoms=TensorDict(Z=Z, pos=pos, batch=batch_idx, batch_size=[4]),
            edges=TensorDict(
                edge_index=edge_index,
                bond_diff=bond_diff,
                bond_dist=bond_dist,
                batch_size=[edge_index.shape[0]],
            ),
            graphs=TensorDict(
                num_atoms=num_atoms,
                total_charge=total_charge,
                cell=cell,
                batch_size=[1],
            ),
            batch_size=[],
        )
        yield sonata, batch
    finally:
        config["ftype"] = orig_ftype


# ---------------------------------------------------------------------------
# 10 — Anti-double-counting warning + cross-reference (ac-011)
# ---------------------------------------------------------------------------


def test_anti_double_counting_warning(sonata_and_batch) -> None:
    """A user hand-composing ``Sonata + Polarization`` in parallel on the
    same data must see a ``UserWarning`` mentioning ``double-count``.

    The warning hook lives on ``Polarization.forward`` (per the
    sub-spec design): when Polarization is invoked on data that has
    already been routed through the LES electrostatic path (detected
    via the presence of an ``energy_es`` / ``pot_es`` signal on the
    upstream forward), Polarization warns rather than silently
    contributing a second induced-response energy.
    """
    sonata, batch = sonata_and_batch

    sonata_out = sonata(batch.clone())
    charges = sonata_out["atomic_charges"].detach()

    polarization = Polarization()
    pos = batch["atoms", "pos"]
    atom_batch = batch["atoms", "batch"]
    edge_index = batch["edges", "edge_index"]
    # Synthetic isotropic polarizabilities (~ atomic-scale).
    alpha = torch.full((4,), 1.0, dtype=torch.float64)

    with pytest.warns(UserWarning, match="double-count"):
        polarization(
            pos=pos,
            charge=charges,
            alpha=alpha,
            batch=atom_batch,
            edge_index=edge_index,
        )


def test_polarization_docstring_cross_references_sonata() -> None:
    """``Polarization``'s class docstring must mention ``Sonata`` so the
    user discovers the LES α-mode composer line as an alternative."""
    assert Polarization.__doc__ is not None
    assert "Sonata" in Polarization.__doc__, (
        "Polarization.__doc__ does not reference Sonata. "
        "The sub-spec requires the cross-reference so a user landing on "
        "Polarization discovers the LES α-mode composer line. Add a "
        "`See also: :class:`molpot.composition.Sonata`` (or equivalent) "
        "to Polarization's class docstring."
    )
