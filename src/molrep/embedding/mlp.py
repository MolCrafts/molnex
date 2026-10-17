"""Scalar MLPs shared by the encoders and heads.

- :class:`ScalarLinearLayer` / :class:`ScalarMLPFunction` — variance-preserving
  bias-free MLP, port of ``nequip.nn.mlp`` (Allegro end to end).
- :class:`MomentNormalizedMLP` — moment-normalised SiLU (``normalize2mom``)
  MLP as in upstream MACE's radial MLP.
- :class:`FFLayer` — plain ``Dense`` stack with an optional activation and
  bias, port of PiNN's ``FFLayer`` (PiNet PP / II / PI / out layers).

Living in :mod:`molrep` lets any encoder (``molzoo``) or head (``molpot``)
compose them without a cross-package dependency. Sharing the nequip init
across encoder *and* readout is what keeps forward activation variance
constant through an Allegro graph.

Reference:
    ``nequip/nn/mlp.py::ScalarLinearLayer`` / ``ScalarMLPFunction``
    ``pinn/networks/pinet.py::FFLayer`` (Teoroo-CMC/PiNN)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import ClassVar, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from molix import config
from molix.nn.activation import activation_from_name


class ScalarLinearLayer(nn.Module):
    """Linear layer with ``α``-scaled weights and reference's variance-preserving init.

    Mirrors ``nequip.nn.mlp.ScalarLinearLayer``: weights are drawn from
    ``U(-√3, √3)`` (so per-element variance is 1) and rescaled by a
    constant ``α`` at every forward, where the caller computes
    ``α = gain / √fan_in``. This keeps the underlying weights themselves
    unit-variance (good for optimisation) while preserving forward
    activation variance after multiplication by α.

    The weight lives in a contained, bias-free ``nn.Linear`` (``self.linear``,
    weight ``(out, in)``). The forward is ``linear(x) * α + b``; the bias
    stays on this module, unscaled, under its old key. Upstream nequip stores
    the weight as ``(in, out)`` under ``<name>.weight``:
    :attr:`upstream_linear_layout` declares that, and
    :meth:`molzoo.checkpoint.CheckpointRemap.load` transposes it onto
    ``<name>.linear.weight``.

    No bias by default, matching the reference's ``bias=False`` everywhere.
    """

    #: Upstream keeps this layer's weight as ``(in, out)`` under ``.weight``.
    upstream_linear_layout: ClassVar[str] = "in_out"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
        bias: bool = False,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=dtype), persistent=False)
        # ``skip_init``: no kaiming draw. The reference's unit-variance draw is
        # taken in upstream's ``(in, out)`` shape and transposed in, so a seeded
        # build is bit for bit the model the old direct tensor gave.
        self.linear = nn.utils.skip_init(
            nn.Linear, in_features, out_features, bias=False, dtype=dtype
        )
        with torch.no_grad():
            draw = torch.empty((in_features, out_features), dtype=dtype)
            self.linear.weight.copy_(nn.init.uniform_(draw, -math.sqrt(3), math.sqrt(3)).t())
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x) * self.alpha
        if self.bias is None:
            return y
        return y + self.bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, alpha={self.alpha.item():.6f}"
        )


class ScalarMLPFunction(nn.Module):
    """Forward-variance-preserving MLP. Mirrors ``nequip.nn.mlp.ScalarMLPFunction``.

    Per-element weight std is ``1`` (uniform init); each Linear's effective
    weight is multiplied by ``α = gain / √fan_in`` at forward time. With
    ``forward_weight_init=True`` (the reference default), the gain is
    ``√2`` for hidden layers (matches ReLU/SiLU forward variance) and
    ``1`` for the first layer.

    Args:
        input_dim: Input feature dim.
        output_dim: Output feature dim.
        hidden_layers_depth: Number of hidden layers (``0`` → bare linear).
        hidden_layers_width: Hidden width. Required if depth > 0; ignored
            at depth 0.
        nonlinearity: Activation between hidden layers — a name or module
            class (:func:`molix.nn.activation_from_name`), or ``None`` for a
            deep linear stack.
        bias: Whether to include bias on every linear (reference: ``False``).
        dtype: Floating dtype every layer is built at (the owning module's).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers_depth: int = 0,
        hidden_layers_width: Optional[int] = None,
        nonlinearity: str | type[nn.Module] | None = nn.SiLU,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_layers_depth != 0:
            assert (
                hidden_layers_depth > 0
                and hidden_layers_width is not None
                and (hidden_layers_width > 0)
            ), "hidden_layers_width must be a positive int when depth > 0"
        nonlinearity = activation_from_name(nonlinearity)
        hidden_dims = [hidden_layers_width] * hidden_layers_depth if hidden_layers_depth > 0 else []
        self.dims = [input_dim, *hidden_dims, output_dim]
        self.num_layers = len(self.dims) - 1
        assert self.num_layers >= 1
        self.is_nonlinear = (nonlinearity is not None) and (self.num_layers > 1)

        layers: list[nn.Module] = []
        for layer_idx, (h_in, h_out) in enumerate(zip(self.dims, self.dims[1:])):
            gain = 1.0 if (nonlinearity is None or layer_idx == 0) else math.sqrt(2.0)
            layers.append(
                ScalarLinearLayer(
                    in_features=h_in,
                    out_features=h_out,
                    alpha=gain / math.sqrt(h_in),
                    bias=bias,
                    dtype=dtype,
                )
            )
            if layer_idx != self.num_layers - 1 and nonlinearity is not None:
                layers.append(nonlinearity())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def normalize2mom(act: Callable[[torch.Tensor], torch.Tensor]) -> float:
    """Return the constant that normalises ``act`` to unit second moment.

    ``c`` such that ``E[(c·act(z))²] = 1`` for ``z ~ N(0, 1)``, estimated on a
    fixed one-million-sample draw (seed 0) so the value is reproducible across
    processes and matches the constant upstream MACE uses.

    Args:
        act: Elementwise activation.

    Returns:
        The scaling constant.
    """
    gen = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.float64)
    return act(z).pow(2).mean().pow(-0.5).item()


class MomentNormalizedMLP(nn.Module):
    """Scalar MLP with ``1/√fan_in`` weight scaling and moment-normalised SiLU.

    Moment-normalised SiLU (``normalize2mom``) MLP as in upstream MACE, which
    uses it for its radial weight generator and edge-density head. Each
    layer computes ``x @ (W / √fan_in)``; every layer but the last then applies
    ``c·SiLU`` with ``c`` from :func:`normalize2mom`. There are no biases.

    Sub-modules are named ``layer0 … layerN``; each holds its weight in a
    contained ``nn.Linear`` as ``layerN.linear.weight`` ``(out, in)``, and the
    checkpoint loader transposes upstream's ``layerN.weight`` ``(in, out)``
    onto it (``_Layer.upstream_linear_layout``).

    Args:
        channels: Widths ``[in, hidden…, out]`` (at least two entries).
        dtype: Floating dtype the layers are built at (the owning module's).

    Reference:
        Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
        Networks for Fast and Accurate Force Fields" NeurIPS 2022.
        https://arxiv.org/abs/2206.07697
    """

    class _Layer(nn.Module):
        """One ``linear(x) / √fan_in`` layer, the weight in a contained ``nn.Linear``.

        Upstream MACE (e3nn ``FullyConnectedNet``) stores the weight as
        ``(in, out)`` under ``layerN.weight``; :attr:`upstream_linear_layout`
        declares that and the loader transposes it onto ``layerN.linear.weight``.
        """

        #: Upstream keeps this layer's weight as ``(in, out)`` under ``.weight``.
        upstream_linear_layout: ClassVar[str] = "in_out"

        def __init__(self, in_features: int, out_features: int, *, dtype: torch.dtype) -> None:
            super().__init__()
            # ``skip_init``: no kaiming draw. Upstream's N(0, 1) draw is taken in
            # its ``(in, out)`` shape and transposed in, so a seeded build is bit
            # for bit the model the old direct tensor gave.
            self.linear = nn.utils.skip_init(
                nn.Linear, in_features, out_features, bias=False, dtype=dtype
            )
            with torch.no_grad():
                self.linear.weight.copy_(torch.randn(in_features, out_features, dtype=dtype).t())
            # A tensor buffer is visible to Dynamo's checkpoint higher-order
            # op as an explicit module input.  Keeping this as a Python float
            # makes ``self.weight * self.alpha`` a tracked free variable in a
            # checkpointed interaction and breaks ``fullgraph=True`` tracing.
            # It is derived metadata, not checkpoint state, just like
            # :class:`MLP`'s per-layer scaling above.
            self.register_buffer(
                "alpha",
                # Store the derived value before any precision cast.  A
                # zero-dimensional float64 buffer follows PyTorch's scalar
                # promotion rules for float32 inputs and preserves the old
                # Python-float accuracy when the module is converted to
                # float64.
                torch.tensor(1.0 / math.sqrt(in_features), dtype=torch.float64),
                persistent=False,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x) * self.alpha

    def __init__(self, channels: list[int], *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        if len(channels) < 2:
            raise ValueError(f"channels must have at least [in, out], got {channels}")
        self.channels = list(channels)
        # See ``_Layer.alpha``: checkpoint tracing needs this derived scalar as
        # an explicit, non-persistent module tensor too.
        self.register_buffer(
            "_act_cst",
            torch.tensor(normalize2mom(F.silu), dtype=torch.float64),
            persistent=False,
        )
        for i, (h_in, h_out) in enumerate(zip(channels, channels[1:])):
            self.add_module(f"layer{i}", self._Layer(h_in, h_out, dtype=dtype))
        self._num_layers = len(channels) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP.

        Args:
            x: Input ``(..., channels[0])``.

        Returns:
            ``(..., channels[-1])``.
        """
        for i in range(self._num_layers):
            x = getattr(self, f"layer{i}")(x)
            if i < self._num_layers - 1:
                x = F.silu(x) * self._act_cst
        return x

    def first_layer_parts(self, widths: Sequence[int]) -> tuple[torch.Tensor, ...]:
        """Layer 0's weight, split by a partition of its input channels.

        Lets a caller evaluate ``layer0`` in pieces instead of handing this
        module one wide tensor: ``cat(parts, -1) @ W`` and ``Σ part_i @ W_i``
        are the same sum. That is what keeps a concatenated ``(E, in)`` edge
        tensor from being materialised at all.

        Args:
            widths: Input-channel widths, summing to ``channels[0]``.

        Returns:
            One ``(width_i, channels[1])`` weight per part — a transposed view
            of the contained linear's ``(out, in)`` weight — with the
            ``1/√fan_in`` scaling already folded in, so ``part_i @ w_i`` is
            that part's contribution.
        """
        layer = self.layer0
        if sum(widths) != self.channels[0]:
            raise ValueError(f"widths {list(widths)} do not partition {self.channels[0]} inputs")
        return tuple((layer.linear.weight * layer.alpha).t().split(list(widths), dim=0))

    def first_layer_bias(self) -> torch.Tensor | None:
        """``None`` — this MLP is bias-free, as upstream MACE's radial MLP is."""
        return None

    def forward_from_first(self, hidden: torch.Tensor) -> torch.Tensor:
        """Continue after layer 0, given its output.

        Args:
            hidden: ``layer0``'s pre-activation output ``(..., channels[1])``.

        Returns:
            ``(..., channels[-1])`` — identical to :meth:`forward` on the input
            that produced ``hidden``.
        """
        x = hidden
        for i in range(1, self._num_layers):
            x = F.silu(x) * self._act_cst
            x = getattr(self, f"layer{i}")(x)
        return x


class FFLayer(nn.Module):
    """Feed-forward ``Dense`` stack on the last axis (PiNN ``FFLayer``).

    One ``nn.Linear`` per entry of ``n_nodes``, each followed by the
    activation. Every linear is fully specified at construction (no
    ``LazyLinear``), so the module is export / ``torch.compile`` / functorch
    safe without a warm-up pass. An empty ``n_nodes`` is the identity.

    Args:
        n_nodes: Output width of each layer.
        in_dim: Input width.
        activation: Name or module class (:func:`molix.nn.activation_from_name`);
            ``None`` is linear — the Keras ``Dense`` default PiNN relies on.
        use_bias: Give every linear a bias.
        dtype: Floating dtype every linear is built at (the owning module's).

    Raises:
        ValueError: If ``in_dim`` is not positive.
    """

    def __init__(
        self,
        n_nodes: Sequence[int],
        *,
        in_dim: int,
        activation: str | type[nn.Module] | None = None,
        use_bias: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        if int(in_dim) <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}.")
        act_cls = activation_from_name(activation)
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for width in n_nodes:
            layers.append(nn.Linear(prev, int(width), bias=use_bias, dtype=dtype))
            if act_cls is not None:
                layers.append(act_cls())
            prev = int(width)
        self.layers = nn.Sequential(*layers)
        self.in_dim = int(in_dim)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stack to ``x`` ``(..., in_dim)`` → ``(..., output_dim)``."""
        return self.layers(x)
