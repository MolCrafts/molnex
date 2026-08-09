"""Fake-quantization infrastructure (precision/quantization sweeps).

Infrastructure — lives in ``molix`` (the base layer), not ``molzoo`` (models
only). Operates on generic ``nn.Module`` / ``state_dict``, so it knows nothing
about any specific architecture.

Object model (strategy pattern, no free functions):

- :class:`QuantScheme` — abstract fake-quant strategy; concrete subclasses
  (:class:`FP16Scheme`, :class:`BF16Scheme`, :class:`Int8Scheme`,
  :class:`Int8PerChannelScheme`, :class:`Int4Scheme`, :class:`Int4PerChannelScheme`)
  self-register under a string name and implement :meth:`QuantScheme.quantize`.
- :class:`FakeQuantize` — an ``nn.Module`` weight *parametrization*. On access it
  returns the quantized weight with a straight-through estimator, so the same
  object serves PTQ (eval) and QAT (gradients flow as identity).
- :class:`Quantizer` — the non-invasive switch. ``Quantizer("int8").apply(model)``
  registers the parametrization on every float weight in place; the model's own
  code is never edited. PTQ on a checkpoint is :meth:`Quantizer.quantize_state_dict`.

Compute *dtype* (fp32/fp64/mixed) is the orthogonal axis and stays with
:meth:`molix.config.MolnexConfig.set_precision`. Quantization here perturbs
weight *values* while compute stays at ``config["ftype"]``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from molix.schema import FORCES_KEY
from molix.units import KB_EV_PER_K


class QuantScheme(ABC):
    """Abstract fake-quantization strategy (quantize-then-dequantize to float).

    Concrete subclasses set a class-level :attr:`name` and are auto-registered,
    so :meth:`from_name` resolves the string keys used across sweeps.
    """

    name: str = ""
    _registry: dict[str, type[QuantScheme]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            QuantScheme._registry[cls.name] = cls

    @classmethod
    def from_name(cls, name: str) -> QuantScheme:
        """Instantiate the scheme for ``name`` (e.g. ``"int8"``, ``"int4_pc"``).

        Registered schemes resolve first. Unregistered symmetric-integer schemes of
        the form ``"int<N>"`` / ``"int<N>_pc"`` (any bit width ``N ≥ 2``) are built on
        the fly from :class:`IntScheme`, so a bit-width sweep needs no extra classes.
        """
        try:
            return cls._registry[name]()
        except KeyError:
            pass
        m = re.fullmatch(r"int(\d+)(_pc)?", name)
        if m and int(m.group(1)) >= 2:
            scheme = IntScheme()
            scheme.n_bits = int(m.group(1))
            scheme.per_channel = m.group(2) is not None
            scheme.name = name
            return scheme
        valid = ", ".join(sorted(cls._registry))
        raise ValueError(f"unknown scheme {name!r}; valid schemes: {valid}") from None

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """All registered scheme names."""
        return tuple(sorted(cls._registry))

    def quantize(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``t`` with quantization error injected (non-float tensors pass through)."""
        if not torch.is_floating_point(t):
            return t
        return self._quantize(t)

    @abstractmethod
    def _quantize(self, t: torch.Tensor) -> torch.Tensor:
        """Inject quantization error into a float tensor (same dtype/shape)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class FP16Scheme(QuantScheme):
    """Round to IEEE float16 mantissa, keep the original dtype."""

    name = "fp16"

    def _quantize(self, t: torch.Tensor) -> torch.Tensor:
        return t.half().to(t.dtype)


class BF16Scheme(QuantScheme):
    """Round to bfloat16 mantissa, keep the original dtype."""

    name = "bf16"

    def _quantize(self, t: torch.Tensor) -> torch.Tensor:
        return t.bfloat16().to(t.dtype)


class IntScheme(QuantScheme):
    """Symmetric integer fake-quant on a ``[-qmax, qmax]`` grid.

    Subclasses set :attr:`n_bits` and :attr:`per_channel` (per-channel uses a
    separate scale per output row, i.e. dim 0).
    """

    n_bits: int = 8
    per_channel: bool = False

    def _quantize(self, t: torch.Tensor) -> torch.Tensor:
        qmax = 2 ** (self.n_bits - 1) - 1  # int8 -> 127, int4 -> 7
        if self.per_channel and t.dim() >= 1:
            dims = tuple(range(1, t.dim()))
            amax = t.abs().amax(dim=dims, keepdim=True) if dims else t.abs()
        else:
            amax = t.abs().max()
        scale = (amax / qmax).clamp(min=1e-12)
        q = torch.clamp(torch.round(t / scale), -qmax, qmax)
        return q * scale


class Int8Scheme(IntScheme):
    name = "int8"
    n_bits = 8


class Int8PerChannelScheme(IntScheme):
    name = "int8_pc"
    n_bits = 8
    per_channel = True


class Int4Scheme(IntScheme):
    name = "int4"
    n_bits = 4


class Int4PerChannelScheme(IntScheme):
    name = "int4_pc"
    n_bits = 4
    per_channel = True


class FakeQuantize(nn.Module):
    """Weight parametrization that fake-quantizes on access (straight-through).

    Registered onto a module's ``weight`` via :class:`Quantizer`. The forward
    returns the quantized value but routes gradients straight through to the
    underlying full-precision weight (STE), so a quantized model both runs
    (PTQ / eval) and trains (QAT) without touching the model definition.
    """

    def __init__(self, scheme: QuantScheme):
        super().__init__()
        self.scheme = scheme

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if not weight.is_floating_point():
            return weight
        q = self.scheme.quantize(weight)
        return weight + (q - weight).detach()  # value = q, gradient = identity


class Quantizer:
    """Non-invasive fake-quant switch for a whole model.

    Examples:
        >>> q = Quantizer("int8")
        >>> q.apply(model)                      # QAT/eval: weights quantized on access
        >>> q.remove(model)                     # back to full precision
        >>> sd_q = Quantizer("int4").quantize_state_dict(model.state_dict())  # PTQ

    ``Quantizer(None)`` is the full-precision baseline (every method is a no-op),
    so a sweep is just ``Quantizer(scheme)`` over ``[None, *QuantScheme.names()]``.
    """

    #: weight-like attributes quantized on each module
    target_attrs: tuple[str, ...] = ("weight",)

    def __init__(self, scheme: QuantScheme | str | None):
        if scheme is None or isinstance(scheme, QuantScheme):
            self.scheme = scheme
        else:
            self.scheme = QuantScheme.from_name(scheme)

    def _quantizable(self, module: nn.Module, attr: str) -> bool:
        w = getattr(module, attr, None)
        return (
            isinstance(w, torch.Tensor)
            and not isinstance(w, nn.UninitializedParameter)
            and w.is_floating_point()
        )

    def apply(self, model: nn.Module) -> nn.Module:
        """Register the fake-quant parametrization on every float weight, in place."""
        if self.scheme is None:
            return model
        for module in model.modules():
            for attr in self.target_attrs:
                if self._quantizable(module, attr):
                    parametrize.register_parametrization(module, attr, FakeQuantize(self.scheme))
        return model

    def remove(self, model: nn.Module) -> nn.Module:
        """Strip any fake-quant parametrizations, restoring full-precision weights."""
        for module in model.modules():
            for attr in self.target_attrs:
                if parametrize.is_parametrized(module, attr):
                    parametrize.remove_parametrizations(module, attr, leave_parametrized=False)
        return model

    def quantize_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """PTQ: a copy of ``state_dict`` with every float tensor fake-quantized."""
        if self.scheme is None:
            return dict(state_dict)
        return {
            k: (self.scheme.quantize(v) if torch.is_floating_point(v) else v)
            for k, v in state_dict.items()
        }

    @staticmethod
    def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Drop the ``_orig_mod.`` prefix ``torch.compile`` adds to checkpoint keys."""
        prefix = "_orig_mod."
        return {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state_dict.items()}

    def __repr__(self) -> str:
        return f"Quantizer({self.scheme!r})"


class ForceDelta:
    """Force residual ``ΔF = F_quant − F_ref`` on a single configuration.

    The static-ensemble probe of the quantization-as-thermal-noise study: how a
    fake-quantized model's forces deviate from the full-precision reference on the
    same geometry. :meth:`summary` reports the residual's moments (criteria a/b:
    unbiased + Gaussian); the RMS feeds :class:`EffectiveTemperature`.
    """

    def __init__(self, delta_f: torch.Tensor):
        self.delta_f = delta_f

    @classmethod
    def between(cls, model_ref: nn.Module, model_quant: nn.Module, batch: object) -> ForceDelta:
        """ΔF from a reference vs quantized model on a fresh clone of ``batch`` each."""
        # Potentials fix force derivation at construction (monomorphic forward,
        # since b85d12f); both models must already be built with it.
        f_ref = model_ref(batch.clone())[FORCES_KEY].detach()
        f_quant = model_quant(batch.clone())[FORCES_KEY].detach()
        return cls(f_quant - f_ref)

    def summary(self) -> dict[str, float]:
        """Moments of ΔF (float64): bias, std, RMS (eV/Å), skew, excess kurtosis, n."""
        x = self.delta_f.detach().to(torch.float64).flatten()
        n = int(x.numel())
        mean = x.mean()
        std = x.std(unbiased=False)
        rms = x.pow(2).mean().sqrt()
        if float(std) > 0.0:
            z = (x - mean) / std
            skew = z.pow(3).mean()
            exkurt = z.pow(4).mean() - 3.0
        else:
            skew = torch.zeros((), dtype=torch.float64)
            exkurt = torch.zeros((), dtype=torch.float64)
        return {
            "F_bias": float(mean),
            "F_std": float(std),
            "F_rms": float(rms),
            "F_skew": float(skew),
            "F_exkurt": float(exkurt),
            "n": n,
        }


class EffectiveTemperature:
    """Eq8 effective-temperature artifact of quantization force noise.

    ``k_B·T_eff = ⟨|ΔF|²⟩·Δt / (2·γ·m·d)`` (``d = 3N`` total DOF), scaling as
    ``1/γ``. The system constants (Δt, γ, m, d) are fixed per instance; the force
    noise ⟨|ΔF|²⟩ is supplied per call. :meth:`ratio` divides by ``k_B·T_target``
    for the dimensionless reported quantity.
    """

    #: Boltzmann constant in eV/K (single source: :mod:`molix.units`).
    KB_EV_PER_K: float = KB_EV_PER_K

    def __init__(self, *, dt: float, gamma: float, mass: float, dof: int):
        self.dt = dt
        self.gamma = gamma
        self.mass = mass
        self.dof = dof

    def energy(self, f_rms_sq: float) -> float:
        """``k_B·T_eff`` (energy units) from the mean squared force residual ⟨|ΔF|²⟩."""
        return f_rms_sq * self.dt / (2.0 * self.gamma * self.mass * self.dof)

    def ratio(self, f_rms_sq: float, t_target: float) -> float:
        """Dimensionless ``T_eff / T_target`` (scales as ``1/γ``)."""
        return self.energy(f_rms_sq) / (self.KB_EV_PER_K * t_target)
