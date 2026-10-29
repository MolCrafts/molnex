"""One LAMMPS style, as a module: encoder features in, that style's parameters out.

The class names are the molrs style names (:class:`molpot.potentials.bond.BondHarmonic`
and its siblings). Each term owns the head that predicts the style's open
parameters and the conversion of those parameters into the potential's
convention. A style that is absent is not priced and has no parameters.

The bonded terms are the published model's heads (reference
``model/ff_layers/mm_bonded.py``): each reads the atom and bond features of its
own atoms and bonds, in every order that names the same term, and the
parameters come out of :class:`~molpot.classical.heads.TanhWindow`. Two
conventions differ from molrs and are converted here, once: the published model's bond and
angle energies are ``½k(x − x0)²`` where LAMMPS ``harmonic`` is ``K(x − x0)²``,
so ``K = k / 2``; the published model's improper ``k(1 + cos(2φ − π))`` is cvff with
``d = −1``, ``n = 2``. the published model's hypervalent patch (P or S with five or more
bonds, ``mm_bonded.py:patch_bonded_terms``) is applied in the published model's units,
before the conversion. The pair terms are Rondo's own (spec departures D1, D2).

Spec: ``.claude/specs/rondo-enc-01-graph-encoder.md``.

References:
    Zheng, T. et al. https://arxiv.org/abs/2408.12817
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping
from typing import ClassVar

import torch
from tensordict import TensorDict

from molpot.charge import (
    AtomicChargeModel,
    ChiEtaHead,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
)
from molpot.classical.heads import LJHead, SymmetricHead, TanhWindow
from molrep.chem.bond_table import BondTable
from molrep.chem.equivalence import Equivalence
from molrep.embedding.mlp import DenseMLP
from molrep.encoders.features import EncoderFeatures

__all__ = [
    "TERMS",
    "AngleHarmonic",
    "BondChargeCorrection",
    "BondHarmonic",
    "LJCutDirect",
    "DihedralPeriodic",
    "Hypervalent",
    "ImproperCvff",
    "LJCut",
    "PairCoulCut",
    "Term",
]

Tables = Mapping[str, Mapping[str, torch.Tensor]]

#: the published model's hypervalent patch values, in the published model's ``½k`` units
#: (``mm_bonded.py:patch_bonded_terms``): a bond touching the atom, an angle centred on it.
_HYPERVALENT_BOND_K = 1000.0
_HYPERVALENT_ANGLE_K = 278.44


class Term(abc.ABC):
    """A LAMMPS style a model can carry, omit, or gain after the encoder is trained.

    ``style`` is the force-field key (``"bond:harmonic"``). ``head_key`` is the
    state-dict slot (``heads.bonds``). ``namespace`` is where :meth:`predict`
    writes its columns.
    """

    style: ClassVar[str]
    head_key: ClassVar[str]
    namespace: ClassVar[str]

    @abc.abstractmethod
    def predict(
        self, features: EncoderFeatures, frame: TensorDict, potential: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """This batch's parameters, one row per term or atom, in the potential's convention."""

    @abc.abstractmethod
    def potential_params(self, tables: Tables) -> dict[str, torch.Tensor]:
        """The tensors the style's potential reads, from :meth:`predict`'s table."""


class Hypervalent(torch.nn.Module):
    """The atoms the published model patches: phosphorus or sulfur with five or more live bonds.

    ``mm_bonded.py:patch_bonded_terms`` replaces every bonded parameter that
    touches such an atom with a constant; the degree is counted from
    ``bonds.bond_index`` (dead padded bonds excluded by ``bonds.mask``).
    """

    def forward(self, frame: TensorDict) -> torch.Tensor:
        """``(N,)`` bool, one entry per atom."""
        bonds = frame["bonds"]
        index = bonds["bond_index"].long()
        live = bonds["mask"].long() if "mask" in bonds.keys() else torch.ones_like(index[0])
        Z = frame["atoms", "Z"].long()
        degree = torch.zeros_like(Z).index_add(0, index[0], live).index_add(0, index[1], live)
        return (degree >= 5) & ((Z == 15) | (Z == 16))


class _Bonded(Term, SymmetricHead):
    """A bonded style priced by one of the published model's heads.

    Subclasses declare :attr:`arity`, :attr:`window_constants` and the atom order of
    every copy; the bond features of a copy are those of consecutive atoms
    (or, for an improper, of the centre and each leg).
    """

    #: Atoms in one term.
    arity: ClassVar[int]
    #: Bonds in one copy.
    bonds_per_copy: ClassVar[int]
    #: ``(std, mean, lo, hi)`` of each predicted parameter, in the published
    #: model's units (``½k``, degrees).
    window_constants: ClassVar[dict[str, tuple[float, float, float, float]]]
    #: Width of each parameter's output.
    widths: ClassVar[dict[str, int]]

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        *,
        hidden_dim: int,
        width: int,
        layers: int,
        out_hidden_dim: int,
        out_layers: int,
        tanh_output: float,
        dtype: torch.dtype,
    ) -> None:
        copy_atoms = self.bonds_per_copy + 1
        SymmetricHead.__init__(
            self,
            copy_atoms * atom_dim + self.bonds_per_copy * bond_dim,
            self.widths,
            hidden_dim=hidden_dim,
            width=width,
            layers=layers,
            out_hidden_dim=out_hidden_dim,
            out_layers=out_layers,
            dtype=dtype,
        )
        self.windows = torch.nn.ModuleDict(
            {
                name: TanhWindow(*constants, tanh_output=tanh_output)
                for name, constants in self.window_constants.items()
            }
        )
        self.hypervalent = Hypervalent()

    @staticmethod
    def _bond_rows(frame: TensorDict) -> BondTable:
        index = frame["bonds", "bond_index"].long()
        return BondTable(index[0], index[1], int(frame["atoms"].batch_size[0]))

    def _values(
        self, features: EncoderFeatures, frame: TensorDict, potential: torch.nn.Module
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The term's atom index ``(arity, M)`` and its parameters in the published units."""
        index = potential.index(frame).long()
        raw = self.raw(self.copies(features, frame, index))
        return index, {name: self.windows[name](value) for name, value in raw.items()}

    @abc.abstractmethod
    def copies(
        self, features: EncoderFeatures, frame: TensorDict, index: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Every ordering of one term's atom and bond features, ``(M, copy_dim)`` each."""


class BondHarmonic(_Bonded):
    """``bond_style harmonic`` — ``K`` and ``r0`` from the published model's bond head."""

    style = "bond:harmonic"
    head_key = "bonds"
    namespace = "bonds"
    arity = 2
    bonds_per_copy = 1
    widths = {"k": 1, "r0": 1}
    window_constants = {"k": (200.0, 700.0, 80.0, 4000.0), "r0": (0.19, 1.3, 0.5, 5.0)}

    def copies(
        self, features: EncoderFeatures, frame: TensorDict, index: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        del frame
        x, e = features.atom, features.bond
        return (
            torch.cat((x[index[0]], e, x[index[1]]), dim=-1),
            torch.cat((x[index[1]], e, x[index[0]]), dim=-1),
        )

    def predict(
        self, features: EncoderFeatures, frame: TensorDict, potential: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        index, value = self._values(features, frame, potential)
        special = self.hypervalent(frame)
        patched = (special[index[0]] | special[index[1]]).unsqueeze(-1)
        k = torch.where(patched, torch.full_like(value["k"], _HYPERVALENT_BOND_K), value["k"])
        r0 = torch.where(patched, torch.zeros_like(value["r0"]), value["r0"])
        return {"k": 0.5 * k[:, 0], "r0": r0[:, 0]}

    def potential_params(self, tables: Tables) -> dict[str, torch.Tensor]:
        table = tables["bonds"]
        return {"k": table["k"], "r0": table["r0"]}


class AngleHarmonic(_Bonded):
    """``angle_style harmonic`` — ``K`` and ``theta0`` (degrees) from the published angle head."""

    style = "angle:harmonic"
    head_key = "angles"
    namespace = "angles"
    arity = 3
    bonds_per_copy = 2
    widths = {"k": 1, "theta0": 1}
    window_constants = {"k": (60.0, 130.0, 40.0, 1000.0), "theta0": (10.0, 120.0, 20.0, 180.0)}

    def copies(
        self, features: EncoderFeatures, frame: TensorDict, index: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        x, e = features.atom, features.bond
        table = self._bond_rows(frame)
        i, j, k = index
        e_ij, e_jk = e[table.rows(i, j)[0]], e[table.rows(j, k)[0]]
        return (
            torch.cat((x[i], e_ij, x[j], e_jk, x[k]), dim=-1),
            torch.cat((x[k], e_jk, x[j], e_ij, x[i]), dim=-1),
        )

    def predict(
        self, features: EncoderFeatures, frame: TensorDict, potential: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        index, value = self._values(features, frame, potential)
        patched = self.hypervalent(frame)[index[1]].unsqueeze(-1)
        k = torch.where(patched, torch.full_like(value["k"], _HYPERVALENT_ANGLE_K), value["k"])
        theta0 = torch.where(patched, torch.zeros_like(value["theta0"]), value["theta0"])
        return {"k": 0.5 * k[:, 0], "theta0": theta0[:, 0]}

    def potential_params(self, tables: Tables) -> dict[str, torch.Tensor]:
        table = tables["angles"]
        return {"k": table["k"], "theta0": table["theta0"]}


class DihedralPeriodic(_Bonded):
    """``dihedral_style periodic`` — four signed ``k_n``, phases 0° (odd ``n``) and 180° (even).

    ``k`` is zero through a linear centre (``propers.linear``, the published model's SMARTS
    mask) and on any torsion containing a hypervalent atom.
    """

    style = "dihedral:periodic"
    head_key = "propers"
    namespace = "propers"
    arity = 4
    bonds_per_copy = 3
    widths = {"k": 4}
    window_constants = {"k": (1.2, 0.4, -20.0, 20.0)}

    def __init__(self, atom_dim: int, bond_dim: int, **settings) -> None:
        super().__init__(atom_dim, bond_dim, **settings)
        periodicity = torch.arange(1, 5, dtype=settings["dtype"])
        self.register_buffer("periodicity", periodicity, persistent=False)
        self.register_buffer("phase", 180.0 * ((periodicity + 1.0) % 2.0), persistent=False)

    def copies(
        self, features: EncoderFeatures, frame: TensorDict, index: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        x, e = features.atom, features.bond
        table = self._bond_rows(frame)
        i, j, k, l = index  # noqa: E741
        e_ij, e_jk, e_kl = (e[table.rows(a, b)[0]] for a, b in ((i, j), (j, k), (k, l)))
        return (
            torch.cat((x[i], e_ij, x[j], e_jk, x[k], e_kl, x[l]), dim=-1),
            torch.cat((x[l], e_kl, x[k], e_jk, x[j], e_ij, x[i]), dim=-1),
        )

    def predict(
        self, features: EncoderFeatures, frame: TensorDict, potential: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        propers = frame["propers"]
        if "linear" not in propers.keys():
            raise KeyError(
                "dihedral:periodic zeroes torsions through a linear centre from propers.linear; "
                "the batch has no such column"
            )
        index, value = self._values(features, frame, potential)
        special = self.hypervalent(frame)[index].any(dim=0)
        keep = ~(special | propers["linear"].bool())
        return {"k": value["k"] * keep.unsqueeze(-1).to(value["k"].dtype)}

    def potential_params(self, tables: Tables) -> dict[str, torch.Tensor]:
        k = tables["propers"]["k"]
        rows = k.shape[0]
        return {
            "k": k,
            "periodicity": self.periodicity.to(k.dtype).expand(rows, -1),
            "phase": self.phase.to(k.dtype).expand(rows, -1),
        }


class ImproperCvff(_Bonded):
    """``improper_style cvff`` — ``K[1 − cos 2φ]`` on one row per centre, centre first.

    The amplitude comes from the published model's improper head over the three legs
    ``[x_c, e_c·leg, x_leg]``, so it does not depend on the order of the legs.
    The dihedral itself is taken over the row as stored (centre, then the legs
    in ascending order), as the reference does. A batch whose stored order is
    another engine's (AMBER puts the centre third) names the chemical centre
    and legs in ``impropers.centre`` / ``leg1`` / ``leg2`` / ``leg3``; the
    features are assembled from those, the energy from the stored order.

    Only a carbon or nitrogen centre carries the term (the published model's
    ``get_impropers``); molpy emits a row for every three-connected atom, since
    which centres are planar enough is force-field data, so a boron, sulfonium
    or sulfoxide centre's ``k`` is zero here.
    """

    style = "improper:cvff"
    head_key = "impropers"
    namespace = "impropers"
    arity = 4
    bonds_per_copy = 1
    widths = {"k": 1}
    window_constants = {"k": (3.7, 4.3, 0.0, 20.0)}

    def __init__(self, atom_dim: int, bond_dim: int, **settings) -> None:
        super().__init__(atom_dim, bond_dim, **settings)
        dtype = settings["dtype"]
        self.register_buffer("sign", torch.tensor(-1.0, dtype=dtype), persistent=False)
        self.register_buffer("periodicity", torch.tensor(2.0, dtype=dtype), persistent=False)

    def copies(
        self, features: EncoderFeatures, frame: TensorDict, index: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        x, e = features.atom, features.bond
        table = self._bond_rows(frame)
        centre, legs = self._centre_and_legs(frame, index)
        return tuple(
            torch.cat((x[centre], e[table.rows(centre, leg)[0]], x[leg]), dim=-1) for leg in legs
        )

    @staticmethod
    def _centre_and_legs(
        frame: TensorDict, index: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """The chemical centre and legs of every row."""
        impropers = frame["impropers"]
        if "centre" in impropers.keys():
            # The stored order is the energy's (AMBER puts the centre third);
            # the chemical centre and its legs are named explicitly.
            centre = impropers["centre"].long()
            return centre, tuple(impropers[name].long() for name in ("leg1", "leg2", "leg3"))
        return index[0], tuple(index[1:])

    def predict(
        self, features: EncoderFeatures, frame: TensorDict, potential: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        index, value = self._values(features, frame, potential)
        special = self.hypervalent(frame)[index].any(dim=0)
        centre_z = frame["atoms", "Z"][self._centre_and_legs(frame, index)[0]]
        keep = ~special & ((centre_z == 6) | (centre_z == 7))
        return {"k": value["k"][:, 0] * keep.to(value["k"].dtype)}

    def potential_params(self, tables: Tables) -> dict[str, torch.Tensor]:
        k = tables["impropers"]["k"]
        return {
            "k": k,
            "sign": self.sign.to(k.dtype).expand(k.shape),
            "periodicity": self.periodicity.to(k.dtype).expand(k.shape),
        }


class LJCut(Term, LJHead):
    """``pair_style lj/cut`` — σ and ε around the atom's GAFF2 well (spec D2)."""

    style = "pair:lj/cut"
    head_key = "lj"
    namespace = "atoms"

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        sigma_gain: float,
        epsilon_gain: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            feature_dim,
            hidden_dim,
            sigma_gain=sigma_gain,
            epsilon_gain=epsilon_gain,
            dtype=dtype,
        )

    def predict(
        self, features: EncoderFeatures, frame: TensorDict, potential: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        del potential
        atoms = frame["atoms"]
        for column in ("gaff2_sigma", "gaff2_epsilon"):
            if column not in atoms.keys():
                raise KeyError(
                    f"pair lj/cut is priced around atoms.{column}; the batch has no such column"
                )
        return dict(
            self(
                features.atom,
                sigma_ref=atoms["gaff2_sigma"].to(features.atom.dtype),
                epsilon_ref=atoms["gaff2_epsilon"].to(features.atom.dtype),
            )
        )

    def potential_params(self, tables: Tables) -> dict[str, torch.Tensor]:
        table = tables["atoms"]
        return {"epsilon": table["epsilon"], "sigma": table["sigma"]}


class PairCoulCut(Term, AtomicChargeModel):
    """``pair_style coul/cut`` — partial charges from electronegativity equalisation (spec D1)."""

    style = "pair:coul/cut"
    head_key = "charge"
    namespace = "atoms"

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        max_atoms: int,
        jitter: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            ChiEtaHead(feature_dim=feature_dim, hidden_dim=hidden_dim, dtype=dtype),
            ElectronegativityEqualization(
                DiagonalHardnessKernel(), max_atoms=max_atoms, jitter=jitter
            ),
        )

    def predict(
        self, features: EncoderFeatures, frame: TensorDict, potential: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        del potential
        return {
            "charge": self.charges(
                features.atom, frame, total_charge=frame["graphs", "total_charge"]
            )
        }

    def potential_params(self, tables: Tables) -> dict[str, torch.Tensor]:
        return {"charge": tables["atoms"]["charge"]}


class LJCutDirect(Term, torch.nn.Module):
    """``pair_style lj/cut`` — the published σ and ε (``mm_nonbonded.py:PreLJEs``).

    ``σ = t·tanh(mlp_σ(x)) + t`` and ``ε = t·tanh(mlp_ε(x)) + t`` per atom, with
    ``t`` the head's ``tanh_output``. The published model prices them with Lorentz–Berthelot
    mixing (arithmetic σ, geometric ε) and no cutoff; the force field declares
    that. Rondo's own LJ head is :class:`LJCut` (spec D2); this one serves
    :class:`~molzoo.byteff.potential.ByteFF`.

    Args:
        atom_dim: Atom feature width.
        hidden_dim: Hidden width of each MLP.
        layers: Linears in each MLP.
        tanh_output: ``t``.
        dtype: Floating dtype of every parameter.
    """

    style = "pair:lj/cut"
    head_key = "lj_direct"
    namespace = "atoms"

    def __init__(
        self, atom_dim: int, *, hidden_dim: int, layers: int, tanh_output: float, dtype: torch.dtype
    ) -> None:
        torch.nn.Module.__init__(self)
        self.sigma = DenseMLP(atom_dim, hidden_dim, 1, layers, dtype=dtype)
        self.epsilon = DenseMLP(atom_dim, hidden_dim, 1, layers, dtype=dtype)
        self.window = TanhWindow(1.0, tanh_output, 0.0, 2.0 * tanh_output, tanh_output=tanh_output)

    def predict(
        self, features: EncoderFeatures, frame: TensorDict, potential: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        del frame, potential
        return {
            "sigma": self.window(self.sigma(features.atom))[:, 0],
            "epsilon": self.window(self.epsilon(features.atom))[:, 0],
        }

    def potential_params(self, tables: Tables) -> dict[str, torch.Tensor]:
        table = tables["atoms"]
        return {"epsilon": table["epsilon"], "sigma": table["sigma"]}


class BondChargeCorrection(Term, torch.nn.Module):
    """``pair_style coul/cut`` — bond charge corrections (reference ``PreLJEs._bcc_charge``).

    Each bond moves charge ``b = t·tanh(out(pre([x_i, e, x_j]) − pre([x_j, e, x_i])))``
    from ``j`` to ``i``; ``b`` is antisymmetric in the bond's direction by
    construction, so the total charge is the total formal charge exactly. The
    starting charges are the formal charges averaged over equivalence classes.
    Rondo's own charge head is :class:`PairCoulCut` (EEM, spec D1).

    Args:
        atom_dim: Atom feature width.
        bond_dim: Bond feature width.
        hidden_dim: Hidden width of ``pre``.
        width: Output width of ``pre``.
        layers: Linears in ``pre``.
        out_hidden_dim: Hidden width of ``out``.
        out_layers: Linears in ``out`` (``tanh`` between them, no biases).
        tanh_output: ``t``.
        dtype: Floating dtype of every parameter.
    """

    style = "pair:coul/cut"
    head_key = "bcc"
    namespace = "atoms"

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        *,
        hidden_dim: int,
        width: int,
        layers: int,
        out_hidden_dim: int,
        out_layers: int,
        tanh_output: float,
        dtype: torch.dtype,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.pre = DenseMLP(2 * atom_dim + bond_dim, hidden_dim, width, layers, dtype=dtype)
        self.out = DenseMLP(
            width, out_hidden_dim, 1, out_layers, activation="tanh", bias=False, dtype=dtype
        )
        self.tanh_output = float(tanh_output)
        self.equivalence = Equivalence()

    def predict(
        self, features: EncoderFeatures, frame: TensorDict, potential: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        del potential
        bonds = frame["bonds"]
        index = bonds["bond_index"].long()
        i, j = index[0], index[1]
        x, e = features.atom, features.bond
        forward = self.pre(torch.cat((x[i], e, x[j]), dim=-1))
        backward = self.pre(torch.cat((x[j], e, x[i]), dim=-1))
        transfer = self.tanh_output * torch.tanh(self.out(forward - backward)[:, 0])
        if "mask" in bonds.keys():
            transfer = transfer * bonds["mask"].to(transfer.dtype)
        atoms = frame["atoms"]
        formal = atoms["formal_charge"].to(x.dtype).unsqueeze(-1)
        start = self.equivalence(formal, atoms["equi_index"].long())[:, 0]
        return {"charge": start.index_add(0, i, transfer).index_add(0, j, -transfer)}

    def potential_params(self, tables: Tables) -> dict[str, torch.Tensor]:
        return {"charge": tables["atoms"]["charge"]}


TERMS: dict[str, Callable[..., Term]] = {
    BondHarmonic.style: BondHarmonic,
    AngleHarmonic.style: AngleHarmonic,
    DihedralPeriodic.style: DihedralPeriodic,
    ImproperCvff.style: ImproperCvff,
    LJCut.style: LJCut,
    PairCoulCut.style: PairCoulCut,
}
