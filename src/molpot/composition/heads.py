"""Parameter prediction heads for potential composition."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from molix import config


class LJParameterHead(nn.Module):
    """Predict per-atom Lennard-Jones parameters from node features.

    Maps node-level features to positive epsilon (well depth) and
    sigma (zero-crossing distance) via a small MLP whose outputs are
    sigmoid-bounded to ``[min, max]`` so ``(σ/r)¹²`` can never overflow.

    Args:
        feature_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        min_epsilon: Minimum epsilon floor.
        max_epsilon: Maximum epsilon ceiling.
        min_sigma: Minimum sigma floor.
        max_sigma: Maximum sigma ceiling.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        min_epsilon: float = 1e-4,
        max_epsilon: float = 1.0,
        min_sigma: float = 1e-4,
        max_sigma: float = 5.0,
    ):
        super().__init__()
        if max_epsilon <= min_epsilon:
            raise ValueError(f"max_epsilon {max_epsilon} must exceed min_epsilon {min_epsilon}")
        if max_sigma <= min_sigma:
            raise ValueError(f"max_sigma {max_sigma} must exceed min_sigma {min_sigma}")
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2, dtype=config.ftype),
        )
        self._init_zero_energy()

    def _init_zero_energy(self) -> None:
        """Start the head at zero LJ contribution.

        A Class-I head predicts an additive MM energy on top of a frozen
        per-element dressing, so ``E_MM`` should start at ~0. The default
        ``sigmoid(0) = 0.5`` init (``epsilon ≈ 0.5 kcal/mol``,
        ``sigma ≈ 2.5 Å``) places a ``(sigma/r)**12`` wall of ~1e5 kcal/mol
        on close nonbonded contacts (``r < sigma``) and explodes the initial
        energy/force scale and gradient norm. Zeroing the final layer and
        biasing both outputs to their floors makes ``epsilon ≈ 0`` and
        ``sigma ≈ 0`` at init, so the initial LJ energy and forces are ~0 and
        the dressing alone matches the total-energy scale.
        """
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        # sigmoid(-8) ≈ 3.4e-4 → epsilon ≈ min_epsilon, sigma ≈ min_sigma.
        last.bias.data[0] = -8.0
        last.bias.data[1] = -8.0

    def forward(self, node_features: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Predict LJ parameters from node features.

        Args:
            node_features: Per-node features ``(N, D)``.
            **kwargs: Ignored; accepted for a uniform parameter-head signature
                (e.g. MultiHead forwarding ``batch`` / ``Z``).

        Returns:
            Dict with ``"epsilon"`` ``(N,)`` and ``"sigma"`` ``(N,)``.
        """
        raw = self.mlp(node_features)
        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * torch.sigmoid(
            raw[:, 0]
        )
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * torch.sigmoid(raw[:, 1])
        return {"epsilon": epsilon, "sigma": sigma}


class RepulsionParameterHead(nn.Module):
    """Predict per-atom repulsion parameters (eps_rep, lam_rep).

    Args:
        feature_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        min_eps: Minimum floor for eps_rep.
        min_lam: Minimum floor for lam_rep.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        min_eps: float = 1e-4,
        min_lam: float = 1e-4,
    ):
        super().__init__()
        self.min_eps = min_eps
        self.min_lam = min_lam
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2, dtype=config.ftype),
        )

    def forward(self, node_features: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Predict per-atom repulsion parameters from node features.

        Args:
            node_features: Per-atom features ``(N, feature_dim)``.
            **kwargs: Ignored; accepted for a uniform parameter-head signature.

        Returns:
            Dict with ``eps_rep`` ``(N,)`` and ``lam_rep`` ``(N,)``, each
            ``softplus``-mapped to stay above its configured floor.
        """
        raw = self.mlp(node_features)
        eps_rep = F.softplus(raw[:, 0]) + self.min_eps
        lam_rep = F.softplus(raw[:, 1]) + self.min_lam
        return {"eps_rep": eps_rep, "lam_rep": lam_rep}


class ChargeTransferParameterHead(nn.Module):
    """Predict per-atom charge-transfer parameters (eps_ct, lam_ct).

    Args:
        feature_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        min_eps: Minimum floor for eps_ct.
        min_lam: Minimum floor for lam_ct.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        min_eps: float = 1e-4,
        min_lam: float = 1e-4,
    ):
        super().__init__()
        self.min_eps = min_eps
        self.min_lam = min_lam
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2, dtype=config.ftype),
        )

    def forward(self, node_features: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Predict per-atom charge-transfer parameters from node features.

        Args:
            node_features: Per-atom features ``(N, feature_dim)``.
            **kwargs: Ignored; accepted for a uniform parameter-head signature.

        Returns:
            Dict with ``eps_ct`` ``(N,)`` and ``lam_ct`` ``(N,)``, each
            ``softplus``-mapped to stay above its configured floor.
        """
        raw = self.mlp(node_features)
        eps_ct = F.softplus(raw[:, 0]) + self.min_eps
        lam_ct = F.softplus(raw[:, 1]) + self.min_lam
        return {"eps_ct": eps_ct, "lam_ct": lam_ct}


class ChargeHead(nn.Module):
    """Predict per-atom partial charges with per-molecule charge conservation.

    Predicts raw charges and shifts them so each molecule sums to ``total_charge``.

    Args:
        feature_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        total_charge: Target total charge per molecule.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        total_charge: float = 0.0,
    ):
        super().__init__()
        self.total_charge = total_charge
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=config.ftype),
        )

    def forward(
        self, node_features: torch.Tensor, *, batch: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Predict charges with per-molecule neutrality constraint.

        Args:
            node_features: Per-node features ``(N, D)``.
            batch: Graph index per atom ``(N,)``.

        Returns:
            Dict with ``"charge"`` ``(N,)``.
        """
        raw_charge = self.mlp(node_features).squeeze(-1)  # (N,)

        # Per-molecule charge conservation: shift so sum = total_charge
        num_graphs = int(batch.max().item()) + 1
        # Sum of raw charges per molecule
        charge_sum = torch.zeros(num_graphs, dtype=raw_charge.dtype, device=raw_charge.device)
        charge_sum.index_add_(0, batch, raw_charge)
        # Count atoms per molecule
        atom_count = torch.zeros(num_graphs, dtype=raw_charge.dtype, device=raw_charge.device)
        atom_count.index_add_(0, batch, torch.ones_like(raw_charge))
        # Shift per atom
        shift = (self.total_charge - charge_sum) / atom_count.clamp(min=1)
        charge = raw_charge + shift[batch]

        return {"charge": charge}


class TSScalingHead(nn.Module):
    """Tkatchenko-Scheffler volume-scaling head.

    Predicts effective C6, polarizability (alpha), and r_star by scaling
    free-atom reference values with a learned volume ratio.

    Reference values are stored as buffers (indexed by atomic number Z).

    Args:
        feature_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        c6_free: Free-atom C6 reference ``(num_elements,)``.
        alpha_free: Free-atom polarizability reference ``(num_elements,)``.
        r_star_free: Free-atom vdW radius reference ``(num_elements,)``.
    """

    def __init__(
        self,
        feature_dim: int,
        c6_free: torch.Tensor,
        alpha_free: torch.Tensor,
        r_star_free: torch.Tensor,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.register_buffer("c6_free", c6_free)
        self.register_buffer("alpha_free", alpha_free)
        self.register_buffer("r_star_free", r_star_free)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, dtype=config.ftype),
        )

    def forward(
        self, node_features: torch.Tensor, *, Z: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Predict TS-scaled dispersion parameters.

        Args:
            node_features: Per-node features ``(N, D)``.
            Z: Atomic numbers ``(N,)``.

        Returns:
            Dict with ``"c6"`` ``(N,)``, ``"alpha"`` ``(N,)``, ``"r_star"`` ``(N,)``.
        """
        # Volume ratio: softplus ensures positivity
        volume_ratio = F.softplus(self.mlp(node_features).squeeze(-1))  # (N,)

        c6_ref = self.c6_free[Z]
        alpha_ref = self.alpha_free[Z]
        r_star_ref = self.r_star_free[Z]

        # TS scaling: C6 ~ V^2, alpha ~ V, r_star ~ V^(1/3)
        c6 = c6_ref * volume_ratio.pow(2)
        alpha = alpha_ref * volume_ratio
        r_star = r_star_ref * volume_ratio.pow(1.0 / 3.0)

        return {"c6": c6, "alpha": alpha, "r_star": r_star}
