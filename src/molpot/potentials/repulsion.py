"""Ziegler-Biersack-Littmark screened-nuclear-repulsion pair term.

Short-range repulsion that keeps a learned potential physical when two nuclei
approach closer than any training configuration ever did. MACE's foundation
models add it to the interaction energy so high-temperature MD and structure
search cannot fall into an unphysical attractive well at small ``r``.

Reference:
    Ziegler, Biersack, Littmark. "The Stopping and Range of Ions in Solids"
    Pergamon, 1985.
    Batatia et al. "A foundation model for atomistic materials chemistry"
    (MACE-MP-0), § pair repulsion. https://arxiv.org/abs/2401.00096
    LAMMPS ``pair_style zbl`` (``pair_zbl_const.h``) — the form
    :class:`UniversalZBL` follows, as nequip's ``ZBL`` does.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molix import config
from molix.F.scatter import scatter_sum_compile_safe as _scatter_sum
from molrep.embedding.covalent import covalent_radii
from molrep.embedding.cutoff import PolynomialCutoff

# Universal ZBL screening function coefficients: phi(x) = sum_k c_k exp(-b_k x).
_SCREENING_C = (0.1818, 0.5099, 0.2802, 0.02817)
_SCREENING_B = (3.2, 0.9423, 0.4028, 0.2016)
# e^2 / (4 pi eps_0) in eV*Angstrom.
_COULOMB_CONSTANT = 14.3996
# Bohr radius in Angstrom, the length unit of the ZBL screening length.
_BOHR = 0.529


class ZBLRepulsion(nn.Module):
    """Per-atom ZBL repulsion energy with a polynomial cutoff envelope.

    For each edge ``(i, j)``:

    .. math::

        a_{ij} &= \\frac{a_p \\cdot a_0}{Z_i^{a_e} + Z_j^{a_e}} \\\\
        V_{ij} &= \\frac{k\\,Z_i Z_j}{r_{ij}}
                  \\sum_k c_k e^{-b_k r_{ij} / a_{ij}} \\; u(r_{ij}; R_i + R_j)

    where ``u`` is the polynomial envelope (:meth:`PolynomialCutoff.envelope`)
    over the pair's summed covalent radii. Each edge contributes ``V_ij / 2`` to
    its target atom, so a bidirectional edge list double-counts exactly back to
    the pair energy.

    Args:
        exponent: Polynomial-envelope exponent ``p``.
        a_exp: Exponent of ``Z`` in the screening length.
        a_prefactor: Prefactor of the screening length.
        max_z: Largest atomic number in the covalent-radius table.
        trainable: If ``True``, ``a_exp`` / ``a_prefactor`` become learnable.
    """

    def __init__(
        self,
        *,
        exponent: int = 6,
        a_exp: float = 0.300,
        a_prefactor: float = 0.4543,
        max_z: int = 118,
        trainable: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        ftype = config.ftype if dtype is None else dtype  # transitional (2c)
        self.exponent = int(exponent)

        self.register_buffer("c", torch.tensor(_SCREENING_C, dtype=ftype))
        self.c: torch.Tensor
        self.register_buffer("covalent_radii", covalent_radii(max_z, dtype=ftype))
        self.covalent_radii: torch.Tensor

        for name, value in (("a_exp", a_exp), ("a_prefactor", a_prefactor)):
            tensor = torch.tensor(float(value), dtype=ftype)
            if trainable:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)
        self.a_exp: torch.Tensor
        self.a_prefactor: torch.Tensor

    def forward(
        self,
        r: torch.Tensor,
        Z: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-atom ZBL repulsion energy.

        Args:
            r: Edge distances ``(E,)``.
            Z: Atomic numbers ``(N,)``.
            pair_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target
                (the repo-wide edge convention).

        Returns:
            Per-atom energy ``(N,)`` in eV.
        """
        z = Z.long()
        source, target = pair_index[:, 0], pair_index[:, 1]
        z_source = z[source].to(r.dtype)
        z_target = z[target].to(r.dtype)

        screening_length = (
            self.a_prefactor
            * _BOHR
            / (torch.pow(z_source, self.a_exp) + torch.pow(z_target, self.a_exp))
        )
        x = r / screening_length
        phi = sum(self.c[k] * torch.exp(-_SCREENING_B[k] * x) for k in range(len(_SCREENING_B)))

        radii = self.covalent_radii
        pair_cutoff = radii[z[source]] + radii[z[target]]
        envelope = PolynomialCutoff.envelope(r, pair_cutoff, self.exponent)

        v_edges = 0.5 * (_COULOMB_CONSTANT * z_source * z_target) / r * phi * envelope
        return _scatter_sum(v_edges, target, Z.shape[0])


class UniversalZBL(nn.Module):
    """LAMMPS ``pair_zbl`` universal screened repulsion with a polynomial envelope.

    For each directed pair ``(i → j)``:

    .. math::

        x &= \\frac{(Z_i^{0.23} + Z_j^{0.23})\\, r}{a_0}, \\quad
        \\phi(x) = \\sum_k c_k e^{d_k x} \\\\
        V_{ij} &= \\tfrac12 k_e \\frac{Z_i Z_j}{r} \\phi(x)\\, u_p(r / r_{max})

    summed onto the **source** atom ``i``; a bidirectional list counts each pair
    once in total. Constants are LAMMPS ``pair_zbl_const.h`` (``a_0 = 0.46850``
    Å) and ``k_e = 14.399645`` eV·Å (LAMMPS ``metal``); the envelope is cut at
    the model's ``r_max``, not at covalent radii — the difference from
    :class:`ZBLRepulsion` (MACE's form).

    Args:
        r_max: Envelope radius in Å.
        exponent: Polynomial-envelope exponent ``p``.

    Reference:
        ``nequip/nn/pair_potential.py::ZBL`` (nequip 0.19).
    """

    _C = (0.02817, 0.28022, 0.50986, 0.18175)
    _D = (-0.20162, -0.40290, -0.94229, -3.19980)

    def __init__(
        self, *, r_max: float, exponent: int = 6, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.cutoff = PolynomialCutoff(r_cut=r_max, exponent=exponent, dtype=dtype)
        self.register_buffer(
            "_qqr2exesquare", torch.tensor(14.399645 * 0.5, dtype=torch.float64), persistent=False
        )

    def forward(self, r: torch.Tensor, Z: torch.Tensor, pair_index: torch.Tensor) -> torch.Tensor:
        """Per-atom repulsion energy.

        Args:
            r: Pair distances ``(E,)`` in Å.
            Z: Atomic numbers ``(N,)``.
            pair_index: ``(E, 2)``, ``[:, 0]`` source.

        Returns:
            ``(N,)`` in eV, in ``r``'s dtype.
        """
        source, target = pair_index[:, 0], pair_index[:, 1]
        z_i = Z[source].to(r.dtype)
        z_j = Z[target].to(r.dtype)
        x = (z_i.pow(0.23) + z_j.pow(0.23)) * r / 0.46850
        phi = sum(c * torch.exp(d * x) for c, d in zip(self._C, self._D))
        v = (self._qqr2exesquare * (z_i * z_j / r) * phi).to(r.dtype) * self.cutoff(r)
        return _scatter_sum(v, source, Z.shape[0])
