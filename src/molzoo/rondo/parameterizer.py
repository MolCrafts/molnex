"""The Rondo recipe: chemical perception plus one head per Class-I style.

Spec: ``src/molzoo/specs/rondo.md`` §3.2-§3.5, §4.
"""

from __future__ import annotations

from molpot.charge import (
    AtomicChargeModel,
    ChiEtaHead,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
)
from molpot.composition.heads import LJParameterHead
from molpot.composition.mm_heads import (
    AngleParamHead,
    BondParamHead,
    ImproperParamHead,
    ProperTorsionParamHead,
)
from molpot.forcefield import Parameterizer
from molrep.chem.encoder import ChemEncoder

from .forcefield import RondoForceField
from .spec import RondoSpec

__all__ = ["Rondo"]


class Rondo(Parameterizer):
    """Predict every Class-I parameter as a continuous function of the graph.

    A :class:`~molpot.forcefield.Parameterizer` whose encoder is
    :class:`~molrep.chem.encoder.ChemEncoder` and whose template is
    :class:`~molzoo.rondo.forcefield.RondoForceField`. Calling it returns the
    filled field; :meth:`~molpot.forcefield.Parameterizer.tables` returns the
    same parameters as plain tensors and is the hot path.

    The encoder reads **topology only** — no coordinate enters any predicted
    parameter. That is what makes the deployed force exactly
    ``-dE/dpos`` rather than a partial derivative that silently drops a
    parameter-response term.

    Charges are not read off the graph: a :class:`ChiEtaHead` predicts
    electronegativity and hardness, and
    :class:`~molpot.charge.ElectronegativityEqualization` solves the KKT
    system whose Lagrange multiplier *enforces* the molecule's total charge.
    Conservation is a constraint of the solve, not a correction applied after
    one.

    Args:
        spec: Configuration. When omitted, keyword arguments populate a new
            :class:`RondoSpec`.
        **kwargs: Forwarded to :class:`RondoSpec` when ``spec`` is not given.

    Raises:
        TypeError: Both ``spec`` and keyword overrides were passed.

    Example::

        from molzoo.rondo import Rondo, RondoSpec

        model = Rondo(RondoSpec(hidden=64, n_blocks=4))
        field = model(batch)                  # filled RondoForceField
        tables = model.tables(batch)          # the same, as tensors
    """

    def __init__(self, spec: RondoSpec | None = None, **kwargs) -> None:
        if spec is None:
            spec = RondoSpec(**kwargs)
        elif kwargs:
            raise TypeError("Rondo accepts either a RondoSpec or keyword overrides, not both")
        width = spec.hidden
        encoder = ChemEncoder(
            atom_dim=width,
            bond_dim=width,
            angle_dim=width,
            proper_dim=width,
            improper_dim=width,
            hidden_dim=width,
            num_elements=spec.num_elements,
            num_bond_types=spec.num_bond_types,
            n_blocks=spec.n_blocks,
            local_attention=spec.local_attention,
        )
        super().__init__(
            encoder,
            RondoForceField(
                cutoff=spec.cutoff,
                scale_14_charge=spec.scale_14_charge,
                scale_14_lj=spec.scale_14_lj,
                mixing=spec.mixing,
            ),
            bond_head=BondParamHead(feature_dim=width, hidden_dim=width),
            angle_head=AngleParamHead(feature_dim=width, hidden_dim=width),
            proper_head=ProperTorsionParamHead(
                feature_dim=width,
                hidden_dim=width,
                n_terms=spec.n_torsion_terms,
            ),
            improper_head=ImproperParamHead(
                feature_dim=width,
                hidden_dim=width,
                include_periodic=True,
                include_harmonic=False,
            ),
            atom_head=LJParameterHead(feature_dim=width, hidden_dim=width),
            charge_model=AtomicChargeModel(
                ChiEtaHead(feature_dim=width, hidden_dim=width),
                ElectronegativityEqualization(
                    DiagonalHardnessKernel(),
                    max_atoms=spec.eem_max_atoms,
                    jitter=1e-4,
                ),
            ),
        )
        self.config = spec
