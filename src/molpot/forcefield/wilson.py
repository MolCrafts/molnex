"""Wilson B-matrix (``B = ∂q/∂x``) for Class-I internal coordinates.

*Internal coordinates* ``charge`` are the chemically meaningful shape variables of
a molecule. This module builds the two that a **Class-I force field** uses --
the classical functional form of the AMBER / OPLS / CHARMM family, in which
the energy is a sum of harmonic bond and angle terms, periodic torsions,
fixed atomic charges and Lennard-Jones pairs. The two coordinates handled
here are the bond length ``r`` between two bonded atoms (a *stretch*) and the
angle ``θ`` between two bonds sharing an atom (a *bend*).

Stack the ``N`` atom positions into one flat Cartesian vector
``x ∈ R^{3N}``, atom-major / xyz-minor, so entry ``3a + c`` is component
``c ∈ {x, y, z}`` of atom ``a`` -- exactly ``pos.reshape(-1)``. The Wilson
B-matrix is the Jacobian that converts a small Cartesian displacement into
the resulting change of every internal coordinate::

    δq = B δx        B[m, 3a + c] = ∂q_m / ∂x_{a,c}        B is (n_q, 3N)

``n_q`` is the number of internal coordinates (one row each), ``3N`` the
number of Cartesian components (one column each). Row ``m`` is built from the
*s-vectors* of term ``m``: the s-vector ``s_a = ∂q_m/∂x_a`` is Wilson's name
for the 3-component gradient of that one internal coordinate with respect to
the position of atom ``a``. It is non-zero only for the two atoms of a
stretch or the three atoms of a bend, so every row is sparse by construction.

Row units. A stretch row is **dimensionless**: its entries are direction
cosines, i.e. the components of the unit vector along the bond, each the
cosine of the angle between the bond and one Cartesian axis (Å/Å). A bend row
carries **rad Å⁻¹**, numerically the same as **Å⁻¹** because the radian is a
ratio of two lengths and hence dimensionless. Positions are Å throughout.
A Hessian projected with these rows therefore carries kcal mol⁻¹ Å⁻² on
stretch rows and kcal mol⁻¹ rad⁻² on bend rows, which is what the consumer
:class:`molix.core.losses.ProjectedInternalHessianLoss` documents.

Rows are produced analytically (closed-form s-vectors), not by autograd or by
finite differences: exact to machine precision, one pass, and independent of
``pos.requires_grad``.

**No Seminario projection.** ``B`` is supplied at the geometry it is handed;
no minimum-geometry projection is performed anywhere in this module. The
*Cartesian Hessian* ``H_x`` is the matrix of second energy derivatives
``∂²E/∂x∂x`` (kcal mol⁻¹ Å⁻²), and ``H_AB`` is its 3×3 sub-block coupling
atoms ``A`` and ``B``. The Seminario recipe reads a bond force constant off
that sub-block as ``k = ûᵀ(−H_AB)û``, with ``û`` the unit vector along the
bond (the original paper sums the eigenvalue contributions of ``−H_AB``
along ``û``). That identity holds *only* at a stationary point -- a geometry
where every force vanishes -- so off a minimum it silently absorbs the
residual gradient into the force constant. This builder therefore never
eigen-decomposes a Hessian sub-block and never assumes a minimum; projecting
a Hessian into internal coordinates stays with the consumer
(:class:`molix.core.losses.ProjectedInternalHessianLoss`).

Reference:
    Wilson, E. B.; Decius, J. C.; Cross, P. C. *Molecular Vibrations: The
    Theory of Infrared and Raman Vibrational Spectra*, McGraw-Hill, 1955,
    Ch. 4 (s-vector formulas for stretch and bend).
    Wilson, E. B. "A Method of Obtaining the Expanded Secular Equation for
    the Vibration Frequency of a Molecule" *J. Chem. Phys.* **9**, 76 (1941).
    https://doi.org/10.1063/1.1750829
    Seminario, J. M. "Calculation of Intramolecular Force Fields from
    Second-Derivative Tensors" *Int. J. Quantum Chem.* **60**, 1271 (1996)
    -- cited for the minimum-geometry restriction this module does not rely
    on. https://doi.org/10.1002/(SICI)1097-461X(1996)60:7<1271::AID-QUA8>3.0.CO;2-W
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["WilsonBMatrix"]


class WilsonBMatrix:
    """Analytic Wilson ``B = ∂q/∂x`` rows for bond stretches and bend angles.

    Each row holds the s-vectors of one internal coordinate -- the per-atom
    gradients ``s_a = ∂q/∂x_a`` defined in the module docstring -- laid out in
    the flat ``3N`` Cartesian ordering ``pos.reshape(-1)``.

    Stateless builder: construct it directly, then ask for the rows you need.
    Not an :class:`torch.nn.Module` -- it owns no parameters and no buffers,
    only the two tolerances below.

    Index conventions are the ones the Class-I potentials already use, so a
    caller is unlikely to mis-wire them. Both index tensors use the
    **coordinate-list (COO)** layout: one row per slot in the term, one column
    per term. ``bond_index`` is COO ``(2, n_bonds)`` (row 0 = atom ``i``,
    row 1 = atom ``j``); ``angle_index`` is COO ``(3, n_angles)`` whose row 1
    is the *apex* -- the central atom ``j`` shared by both legs ``i–j`` and
    ``k–j`` of the bend, matching
    :class:`~molpot.potentials.angles.harmonic.AngleHarmonic`.

    That COO shape is also the anti-alias guard against ``pair_index``. A
    *bond* is fixed covalent topology; an *edge* is a geometric neighbour pair
    within a cutoff radius, recomputed from positions every step, and is
    stored transposed as ``(E, 2)`` so that a mis-wired tensor is caught by
    its shape instead of quietly yielding a wrong matrix. A tensor whose
    leading dimension is not 2 (bonds) or 3 (angles) is rejected loudly.

    Row units:
        Stretch rows are **dimensionless** (direction cosines, Å/Å); bend rows
        are **rad Å⁻¹**, numerically **Å⁻¹** since the radian is dimensionless.

    Args:
        sin_tol: Smallest ``|sin θ|`` accepted for a bend row, dimensionless.
            Below it the bend coordinate itself is undefined (linear angle)
            and the s-vectors diverge, so the build raises instead of clamping.
        length_tol: Smallest bond or angle-leg length (Å) accepted for a row.

    Attributes:
        sin_tol: The validated ``|sin θ|`` floor, dimensionless.
        length_tol: The validated length floor (Å).

    Raises:
        ValueError: If ``sin_tol`` or ``length_tol`` is not positive.

    Example:
        >>> import torch
        >>> from molpot import WilsonBMatrix
        >>> pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # Å
        >>> bond_index = torch.tensor([[0], [1]])  # one bond, atoms 0 and 1
        >>> WilsonBMatrix().stretch_rows(pos, bond_index)
        tensor([[-1.,  0.,  0.,  1.,  0.,  0.]])

    Reference:
        Wilson, Decius & Cross, *Molecular Vibrations*, McGraw-Hill, 1955,
        Ch. 4; Wilson, *J. Chem. Phys.* **9**, 76 (1941).
        https://doi.org/10.1063/1.1750829
    """

    def __init__(self, *, sin_tol: float = 1e-6, length_tol: float = 1e-9) -> None:
        if sin_tol <= 0.0:
            raise ValueError(f"sin_tol must be positive, got {sin_tol}")
        if length_tol <= 0.0:
            raise ValueError(f"length_tol must be positive, got {length_tol}")
        self.sin_tol = float(sin_tol)
        self.length_tol = float(length_tol)

    def stretch_rows(self, pos: Tensor, bond_index: Tensor) -> Tensor:
        """Wilson rows for bond stretches ``q = r_ij = ‖x_j − x_i‖``.

        ``x_i`` and ``x_j`` are the positions (Å) of the two bonded atoms and
        ``r_ij`` their separation. Moving either atom along the bond changes
        ``r_ij`` at unit rate; moving it perpendicular to the bond does not
        change ``r_ij`` to first order. So with the bond unit vector
        ``û = (x_j − x_i) / r_ij`` the only non-zero s-vectors are::

            ∂r/∂x_i = −û        ∂r/∂x_j = +û

        Every other atom contributes a zero 3-vector to the row.

        Args:
            pos: Cartesian positions ``(N, 3)`` in Å, ``N`` atoms.
            bond_index: Covalent bonds in coordinate-list (COO) layout
                ``(2, n_bonds)``, integer atom ids -- row 0 is ``i``, row 1 is
                ``j``. Not a geometric ``pair_index`` ``(E, 2)``.

        Returns:
            Rows ``(n_bonds, 3N)`` in input order, dtype and device of
            ``pos``. Entries are **dimensionless** direction cosines, i.e.
            components of ``±û``.

        Raises:
            ValueError: If ``bond_index`` is not COO ``(2, n_bonds)``, or if
                any bond is shorter than ``length_tol`` (the message names the
                offending term index).
        """
        self._check_coo(bond_index, 2, "bond_index")
        n_atoms = pos.shape[0]
        n_bonds = int(bond_index.shape[1])
        blocks = pos.new_zeros((n_bonds, n_atoms, 3))
        if n_bonds == 0:
            return blocks.reshape(n_bonds, 3 * n_atoms)

        atom_i, atom_j = bond_index[0], bond_index[1]
        delta = pos[atom_j] - pos[atom_i]
        length = torch.linalg.vector_norm(delta, dim=-1)
        degenerate = length < self.length_tol
        if bool(degenerate.any()):
            term = int(torch.nonzero(degenerate)[0, 0])
            raise ValueError(
                f"degenerate bond term {term}: ‖x_j − x_i‖ = {float(length[term]):.3e} Å is "
                f"below length_tol = {self.length_tol:.3e} Å. The stretch coordinate r and "
                "its Wilson s-vectors are undefined for coincident atoms."
            )
        unit = delta / length.unsqueeze(-1)

        term_ids = torch.arange(n_bonds, device=pos.device)
        blocks[term_ids, atom_i] = -unit
        blocks[term_ids, atom_j] = unit
        return blocks.reshape(n_bonds, 3 * n_atoms)

    def bend_rows(self, pos: Tensor, angle_index: Tensor) -> Tensor:
        """Wilson rows for bend angles ``q = θ_ijk`` with apex atom ``j``.

        ``θ_ijk`` is the angle at the *apex* (central) atom ``j`` between its
        two legs to atoms ``i`` and ``k``, in radians. Writing the two leg
        unit vectors as ``u = (x_i − x_j) / r_ij`` and
        ``v = (x_k − x_j) / r_kj``, with leg lengths ``r_ij``, ``r_kj`` in Å,
        the angle follows from ``cos θ = u·v`` and ``sin θ = ‖u × v‖``.
        Differentiating that relation gives the s-vectors::

            ∂θ/∂x_i = (cos θ · u − v) / (r_ij · sin θ)
            ∂θ/∂x_k = (cos θ · v − u) / (r_kj · sin θ)
            ∂θ/∂x_j = −( ∂θ/∂x_i + ∂θ/∂x_k )

        The apex s-vector is fixed by translation invariance: shifting the
        whole molecule cannot change ``θ``, so the three s-vectors must sum to
        zero. Each end s-vector is perpendicular to its own leg and scales as
        ``1/r``, which is where the Å⁻¹ of the row units comes from.

        Args:
            pos: Cartesian positions ``(N, 3)`` in Å, ``N`` atoms.
            angle_index: Angles in coordinate-list (COO) layout
                ``(3, n_angles)``, integer atom ids -- row 0 is ``i``, row 1
                is the apex ``j`` (matching
                :class:`~molpot.potentials.angles.harmonic.AngleHarmonic`),
                row 2 is ``k``.

        Returns:
            Rows ``(n_angles, 3N)`` in input order, dtype and device of
            ``pos``. Entries carry **rad Å⁻¹**, numerically **Å⁻¹**.

        Raises:
            ValueError: If ``angle_index`` is not COO ``(3, n_angles)``, if
                either leg is shorter than ``length_tol``, or if
                ``|sin θ| < sin_tol`` (near-linear bend). Degenerate cases
                name the offending row index instead of being clamped.
        """
        self._check_coo(angle_index, 3, "angle_index")
        n_atoms = pos.shape[0]
        n_angles = int(angle_index.shape[1])
        blocks = pos.new_zeros((n_angles, n_atoms, 3))
        if n_angles == 0:
            return blocks.reshape(n_angles, 3 * n_atoms)

        atom_i, atom_j, atom_k = angle_index[0], angle_index[1], angle_index[2]
        leg_ij = pos[atom_i] - pos[atom_j]
        leg_kj = pos[atom_k] - pos[atom_j]
        r_ij = torch.linalg.vector_norm(leg_ij, dim=-1)
        r_kj = torch.linalg.vector_norm(leg_kj, dim=-1)
        short = (r_ij < self.length_tol) | (r_kj < self.length_tol)
        if bool(short.any()):
            row = int(torch.nonzero(short)[0, 0])
            raise ValueError(
                f"degenerate bend term {row}: leg lengths r_ij = {float(r_ij[row]):.3e} Å, "
                f"r_kj = {float(r_kj[row]):.3e} Å; one is below length_tol = "
                f"{self.length_tol:.3e} Å, so the angle θ is undefined."
            )

        u = leg_ij / r_ij.unsqueeze(-1)
        v = leg_kj / r_kj.unsqueeze(-1)
        cos_theta = (u * v).sum(dim=-1)
        sin_theta = torch.linalg.vector_norm(torch.linalg.cross(u, v, dim=-1), dim=-1)
        linear = sin_theta < self.sin_tol
        if bool(linear.any()):
            row = int(torch.nonzero(linear)[0, 0])
            raise ValueError(
                f"degenerate bend term {row}: |sin θ| = {float(sin_theta[row]):.3e} is below "
                f"sin_tol = {self.sin_tol:.3e}. The angle is linear to within tolerance and "
                "its Wilson s-vectors diverge as sin θ → 0; clamping would emit a large wrong "
                "row and poison the projected force constants."
            )

        s_i = (cos_theta.unsqueeze(-1) * u - v) / (r_ij * sin_theta).unsqueeze(-1)
        s_k = (cos_theta.unsqueeze(-1) * v - u) / (r_kj * sin_theta).unsqueeze(-1)
        s_j = -(s_i + s_k)

        term_ids = torch.arange(n_angles, device=pos.device)
        blocks[term_ids, atom_i] = s_i
        blocks[term_ids, atom_j] = s_j
        blocks[term_ids, atom_k] = s_k
        return blocks.reshape(n_angles, 3 * n_atoms)

    def graph_blocks(
        self,
        pos: Tensor,
        atom_batch: Tensor,
        *,
        bond_index: Tensor | None = None,
        angle_index: Tensor | None = None,
    ) -> list[Tensor]:
        """One independent ``B`` block per graph of a packed batch.

        A *packed batch* concatenates several molecules into one flat atom
        list; ``atom_batch[a]`` says which molecule (graph) atom ``a`` belongs
        to. Terms are split by that vector and their global atom ids are
        rebased to graph-local ids, so block ``g`` is only ``3 N_g`` wide
        (``N_g`` = atoms in graph ``g``) -- cross-graph coupling is
        structurally impossible rather than merely unlikely. Within a block
        the rows are *all bond rows in input order, then all angle rows in
        input order*, so a caller's force-constant vector (``k_r…`` then
        ``k_θ…``) aligns by construction.

        Args:
            pos: Packed Cartesian positions ``(N, 3)`` in Å, all graphs
                concatenated.
            atom_batch: Graph membership ``(N,)``, non-decreasing, so each
                graph owns one contiguous run of atoms (the packed collate
                ``repeat_interleave`` layout).
            bond_index: Optional bonds in coordinate-list (COO) layout
                ``(2, n_bonds)``, ids global to the packed batch.
            angle_index: Optional angles in COO layout ``(3, n_angles)``, ids
                global to the packed batch, row 1 the apex (central) atom.

        Returns:
            One tensor per graph in graph-id order, ``int(atom_batch[-1]) + 1``
            of them; block ``g`` has shape ``(n_q_g, 3 N_g)`` (``n_q_g`` =
            bonds plus angles of graph ``g``) with the dtype and device of
            ``pos``. Bond rows are dimensionless and angle rows carry Å⁻¹, as
            for the single-molecule builders. A graph carrying no terms yields
            ``(0, 3 N_g)``; an empty ``pos`` yields an empty list.

        Raises:
            ValueError: If ``atom_batch`` does not match ``pos``, is not
                non-decreasing, if an index tensor has the wrong COO shape, or
                if a term's atoms span two graphs (the message names the
                offending term index). The per-graph builds also raise on
                degenerate geometry exactly as :meth:`stretch_rows` and
                :meth:`bend_rows` do, with graph-local term indices.
        """
        if atom_batch.ndim != 1 or atom_batch.shape[0] != pos.shape[0]:
            raise ValueError(
                f"atom_batch must be (N,) matching pos (N, 3); got "
                f"{tuple(atom_batch.shape)} for pos {tuple(pos.shape)}."
            )
        if bond_index is not None:
            self._check_coo(bond_index, 2, "bond_index")
        if angle_index is not None:
            self._check_coo(angle_index, 3, "angle_index")
        if pos.shape[0] == 0:
            return []
        if bool((atom_batch[1:] < atom_batch[:-1]).any()):
            raise ValueError(
                "atom_batch must be non-decreasing (packed collate layout); got a graph id "
                "that decreases, so graphs are not contiguous blocks of atoms."
            )

        num_graphs = int(atom_batch[-1].item()) + 1
        counts = torch.bincount(atom_batch, minlength=num_graphs)
        offsets = torch.cumsum(counts, dim=0) - counts

        bond_graph = self._term_graph(atom_batch, bond_index, "bond")
        angle_graph = self._term_graph(atom_batch, angle_index, "angle")

        blocks: list[Tensor] = []
        for graph in range(num_graphs):
            start = int(offsets[graph].item())
            n_local = int(counts[graph].item())
            local_pos = pos[start : start + n_local]
            rows: list[Tensor] = []
            if bond_index is not None and bond_graph is not None:
                selected = bond_index[:, bond_graph == graph] - start
                rows.append(self.stretch_rows(local_pos, selected))
            if angle_index is not None and angle_graph is not None:
                selected = angle_index[:, angle_graph == graph] - start
                rows.append(self.bend_rows(local_pos, selected))
            if not rows:
                rows.append(local_pos.new_zeros((0, 3 * n_local)))
            blocks.append(torch.cat(rows, dim=0))
        return blocks

    @staticmethod
    def _check_coo(index: Tensor, arity: int, name: str) -> None:
        """Raise unless ``index`` is COO ``(arity, n_terms)`` -- 2 bonds, 3 angles.

        Mirrors :class:`~molpot.potentials.bonds.harmonic.BondHarmonic`'s
        guard: the covalent ``bond_index (2, n_bonds)`` is deliberately a
        different shape from the geometric neighbour graph ``pair_index
        (E, 2)`` so a cutoff pair can never be silently used as a bond.
        ``name`` only appears in the error message.
        """
        if index.ndim != 2 or index.shape[0] != arity:
            raise ValueError(
                f"{name} must be COO [{arity}, n_terms] covalent topology; got shape "
                f"{tuple(index.shape)}. A neighbour-graph pair_index [E, 2] is not a bond "
                "or angle list — see the bond_index vs pair_index distinction in CLAUDE.md."
            )

    @staticmethod
    def _term_graph(atom_batch: Tensor, index: Tensor | None, kind: str) -> Tensor | None:
        """Graph id ``(n_terms,)`` of each term, requiring one graph per term.

        Returns ``None`` when ``index`` is ``None`` (that term kind was not
        supplied), and raises ``ValueError`` naming the offending term when a
        term's atoms belong to more than one graph. ``kind`` (``"bond"`` /
        ``"angle"``) only appears in that message.
        """
        if index is None:
            return None
        if index.shape[1] == 0:
            return torch.zeros(0, dtype=atom_batch.dtype, device=atom_batch.device)
        graphs = atom_batch[index]
        spanning = (graphs != graphs[0]).any(dim=0)
        if bool(spanning.any()):
            term = int(torch.nonzero(spanning)[0, 0])
            raise ValueError(
                f"{kind} term {term} spans more than one graph: atoms "
                f"{index[:, term].tolist()} map to graph ids {graphs[:, term].tolist()}. "
                "Every term's atoms must belong to a single graph of the packed batch."
            )
        return graphs[0]
