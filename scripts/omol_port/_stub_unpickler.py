"""Offline-only: unpickle a mace/e3nn archive with nn.Module-shaped stubs.

Stubbing the mace/e3nn classes as ``torch.nn.Module`` subclasses lets the real
``nn.Module.state_dict()`` machinery run over the restored tree, so every
*installed* module (cuequivariance's) contributes through its own
``_save_to_state_dict``. Nothing from ``mace`` or ``e3nn`` is imported —
their classes are replaced by inert stubs at unpickle time, which is what
lets this live in-repo under the "no mace-torch / e3nn" third-party rule.
"""

import pickle
from pickle import *  # noqa: F401,F403

import torch

_ALLOWED = (
    "torch",
    "collections",
    "builtins",
    "__builtin__",
    "_codecs",
    "numpy",
    "cuequivariance",
    "cuequivariance_torch",
)


class _ModStub(torch.nn.Module):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self._stub_args = args

    def __setitem__(self, *args):
        pass

    def append(self, *args):
        pass


_MADE: dict[tuple[str, str], type] = {}


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in _ALLOWED:
            return super().find_class(module, name)
        key = (module, name)
        if key not in _MADE:
            _MADE[key] = type(name, (_ModStub,), {"__module__": module})
        return _MADE[key]


def load(f, **kw):
    return Unpickler(f, **kw).load()
