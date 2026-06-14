"""Pure electrostatic kernels — math functions decoupled from any head.

Modules in this package implement the *math* of the electrostatic
interaction (real-space pair kernels, reciprocal-space scalar kernels,
self-correction constants, per-graph realspace / reciprocal compute
paths) as plain :class:`torch.nn.Module` units. They take tensor inputs
and return tensor outputs — no notion of "head", no induced-response
inlining, no per-graph dispatch beyond what a single sample needs.

Heads (e.g. :class:`molpot.heads.BondChargeHead`,
:class:`molpot.heads.DipoleHead`) emit ``q_i``, ``\\mu_i``, ``Q_i`` and
*call into* these kernels; they do not contain electrostatic math.
"""

from molpot.potentials.elec.kernels.multipole_ewald import MultipoleEwaldKernel

__all__ = ["MultipoleEwaldKernel"]
