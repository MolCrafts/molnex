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
    ) -> None:
        super().__init__()
        ftype = config.ftype
        self.exponent = int(exponent)

        self.register_buffer("c", torch.tensor(_SCREENING_C, dtype=ftype))
        self.c: torch.Tensor
        self.register_buffer("covalent_radii", covalent_radii(max_z))
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
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-atom ZBL repulsion energy.

        Args:
            r: Edge distances ``(E,)``.
            Z: Atomic numbers ``(N,)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target
                (the repo-wide edge convention).

        Returns:
            Per-atom energy ``(N,)`` in eV.
        """
        z = Z.long()
        source, target = edge_index[:, 0], edge_index[:, 1]
        z_source = z[source].to(r.dtype)
        z_target = z[target].to(r.dtype)

        screening_length = (
            self.a_prefactor
            * _BOHR
            / (torch.pow(z_source, self.a_exp) + torch.pow(z_target, self.a_exp))
        )
        x = r / screening_length
        phi = sum(
            self.c[k] * torch.exp(-_SCREENING_B[k] * x) for k in range(len(_SCREENING_B))
        )

        radii = self.covalent_radii
        pair_cutoff = radii[z[source]] + radii[z[target]]
        envelope = PolynomialCutoff.envelope(r, pair_cutoff, self.exponent)

        v_edges = 0.5 * (_COULOMB_CONSTANT * z_source * z_target) / r * phi * envelope
        return _scatter_sum(v_edges, target, Z.shape[0])
