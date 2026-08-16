"""Re-export :class:`molpy.ForceField` as the only MD force-field type.

``from molix.md import ForceField`` and ``from molpy import ForceField`` are
the same object. Evaluation is ``ff.to_potentials()`` then ``calc_energy`` /
``calc_forces`` on a working ``TensorDict``, wired by :class:`~molix.md.driver.MD`.
"""

from molpy import ForceField

__all__ = ["ForceField"]
