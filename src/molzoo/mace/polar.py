"""MACE-POLAR isolated-molecule energy (local + Fukui + field + Hartree).

:class:`MacePolar` specialises :class:`~molzoo.mace.potential.MacePotential`.
Energy per graph, in eV::

    E = E0 + scale·ε_local + shift  +  E_electron  +  E_Hartree

``total_spin`` is ASE multiplicity ``M = 2S+1``.

Reference:
    Batatia et al. "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model
    for Molecular Chemistry" arXiv:2602.19411.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from molix.F.scatter import scatter_sum_compile_safe as _scatter_sum
from molpot.potentials.gto.features import GTOFieldFeatures
from molpot.potentials.gto.fukui import FukuiEquilibration
from molpot.potentials.gto.hartree import GTOHartreeEnergy
from molrep.interaction.polar.field import (
    AgnosticEmbeddedOneBodyVariableUpdate,
    OneBodyMLPFieldReadout,
)
from molrep.interaction.polar.mixer import MultiLayerFeatureMixer
from molrep.interaction.polar.source import EnvironmentDependentSpinSourceBlock
from molrep.interaction.product import sh_irreps_from_l_max
from molrep.readout.mace import NonLinearBiasReadout
from molzoo.mace.checkpoint import polar_keys
from molzoo.mace.encoder import _irreps
from molzoo.mace.geometry import edge_lengths, edge_vectors
from molzoo.mace.potential import MacePotential, _energy_table, _parse_irreps
from molzoo.mace.spec import MACEPolarSpec


def _expand_field_norms(norms: list[float], n_widths: int, l_max: int) -> list[float]:
    expanded: list[float] = []
    for ell in range(l_max + 1):
        for j in range(n_widths):
            expanded.extend([norms[ell * n_widths + j]] * (2 * ell + 1))
    return expanded


class MacePolar(MacePotential):
    """MACE-POLAR-1. Flat constructor; spec is assembled and validated inside."""

    _keys = polar_keys
    _unexpected = "return"

    def __init__(
        self,
        *,
        atomic_numbers: Sequence[int],
        atomic_energies: Sequence[float] | torch.Tensor,
        r_max: float = 6.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        l_max: int = 3,
        num_features: int = 512,
        max_hidden_l: int = 1,
        num_interactions: int = 2,
        correlation: int = 3,
        mlp_dim: int = 16,
        edge_channels: int = 128,
        radial_mlp: Sequence[int] = (64, 64, 64),
        scale: float = 1.0,
        shift: float = 0.0,
        num_recursion_steps: int = 2,
        multipole_lmax: int = 1,
        multipole_sigma: float = 1.5,
        field_lmax: int = 1,
        field_widths: Sequence[float] = (1.5, 3.0),
        field_norms: Sequence[float] = (20.0, 20.0, 0.5, 0.5),
        add_local_electron_energy: bool = True,
        kspace_cutoff_factor: float = 1.0,
        compute_forces: bool = True,
        use_fallback: bool = False,
    ) -> None:
        spec = MACEPolarSpec(
            atomic_numbers=list(atomic_numbers),
            atomic_energies=_energy_table(atomic_energies),
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            l_max=l_max,
            num_features=num_features,
            max_hidden_l=max_hidden_l,
            num_interactions=num_interactions,
            correlation=correlation,
            mlp_dim=mlp_dim,
            edge_channels=edge_channels,
            radial_mlp=list(radial_mlp),
            scale=scale,
            shift=shift,
            num_recursion_steps=num_recursion_steps,
            multipole_lmax=multipole_lmax,
            multipole_sigma=multipole_sigma,
            field_lmax=field_lmax,
            field_widths=list(field_widths),
            field_norms=list(field_norms),
            add_local_electron_energy=add_local_electron_energy,
            kspace_cutoff_factor=kspace_cutoff_factor,
            use_fallback=use_fallback,
        )
        super().__init__(spec, compute_forces=compute_forces, use_fallback=use_fallback)
        hidden = _irreps(spec.max_hidden_l, spec.num_features)
        n_el = len(spec.atomic_numbers)
        y = sh_irreps_from_l_max(spec.multipole_lmax).replace(" ", "")
        charges = f"{y}+{y}"
        n_w = len(spec.field_widths)
        # Official Polar keeps two unsimplified spin copies
        # (``2x0e+2x1o+2x0e+2x1o``), not ``4x0e+4x1o``. ``cat(feat_α, feat_β)``
        # of two ``ir_mul`` field features is exactly that layout.
        field = _irreps(spec.field_lmax, n_w)
        potential = f"{field}+{field}"
        node_attrs = f"{n_el}x0e"

        self.lr_source_maps = nn.ModuleList(
            [
                EnvironmentDependentSpinSourceBlock(irreps_in=hidden, l_max=spec.multipole_lmax)
                for _ in range(spec.num_interactions)
            ]
        )
        self.layer_feature_mixer = MultiLayerFeatureMixer(
            irreps=hidden, num_layers=spec.num_interactions
        )
        self.fukui_source_map = NonLinearBiasReadout(
            irreps_in=hidden, mlp_dim=spec.mlp_dim, out_mul=2
        )
        self.fukui = FukuiEquilibration()
        self.field_dependent_charges_maps = nn.ModuleList(
            [
                AgnosticEmbeddedOneBodyVariableUpdate(
                    node_attrs_irreps=node_attrs,
                    node_feats_irreps=hidden,
                    potential_irreps=potential,
                    charges_irreps=charges,
                )
                for _ in range(spec.num_recursion_steps)
            ]
        )
        self.local_electron_energy = OneBodyMLPFieldReadout(
            node_feats_irreps=hidden,
            potential_irreps=potential,
            charges_irreps=charges,
        )
        self.electric_potential_descriptor = GTOFieldFeatures(
            density_sigma=spec.multipole_sigma,
            density_lmax=spec.multipole_lmax,
            feature_lmax=spec.field_lmax,
            feature_widths=list(spec.field_widths),
            kspace_cutoff_factor=spec.kspace_cutoff_factor,
        )
        self.coulomb_energy = GTOHartreeEnergy(
            sigma=spec.multipole_sigma,
            l_max=spec.multipole_lmax,
            include_self_interaction=True,
            kspace_cutoff_factor=spec.kspace_cutoff_factor,
        )
        expanded = _expand_field_norms(list(spec.field_norms), n_w, spec.field_lmax)
        self.register_buffer(
            "field_feature_norms",
            torch.tensor(expanded, dtype=torch.float64),
            persistent=True,
        )
        self.add_local_electron_energy = spec.add_local_electron_energy
        self.num_recursion_steps = spec.num_recursion_steps
        self._y_dim = (spec.multipole_lmax + 1) ** 2

    def energy_core(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        *,
        edge_diff: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        total_spin: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-graph Polar energy ``(B,)`` in eV.

        ``cell`` selects the official k-space Hartree / field path (single
        system). Isolated molecules omit it and keep the real-space pair sum.
        """
        vectors = edge_vectors(positions, edge_index, edge_diff=edge_diff)
        lengths = edge_lengths(vectors)
        node_attrs = self.node_attrs(Z, positions.dtype)
        edge_feats, cutoff = self.radial_features(lengths, Z, edge_index)
        # Official PolarMACE feeds e3nn SH the yzx permutation of the
        # displacement (``_permute_to_e3nn_convention``). Scalar local
        # energy is invariant; GTO dipoles are not.
        edge_attrs = self.angular_features(vectors[:, [1, 2, 0]])
        node_feats = self.initial_node_features(node_attrs)
        e0 = _scatter_sum(self.atomic_energies(Z), batch, num_graphs)

        per_layer = self.layer_features(
            node_feats=node_feats,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=cutoff,
        )
        node_es = node_feats.new_zeros(node_feats.shape[0])
        if self.readouts is None:
            raise RuntimeError("MacePolar requires per-layer readouts")
        for readout, layer in zip(self.readouts, per_layer, strict=True):
            node_es = node_es + readout(layer).squeeze(-1)
        inter_e = _scatter_sum(self.scale_shift(node_es), batch, num_graphs)

        density = per_layer[0].new_zeros(per_layer[0].shape[0], 2, self._y_dim)
        for src, layer in zip(self.lr_source_maps, per_layer, strict=True):
            density = density + src(layer)
        mixed = self.layer_feature_mixer(torch.stack(per_layer, dim=0))

        if total_charge is None:
            total_charge = positions.new_zeros(num_graphs)
        if total_spin is None:
            total_spin = positions.new_ones(num_graphs)
        density = self.fukui(
            density, self.fukui_source_map(per_layer[-1]), batch, total_charge, total_spin
        )
        field_independent = density.clone()
        norms = self.field_feature_norms.to(dtype=density.dtype, device=density.device)
        potential_features = density.new_zeros(density.shape[0], 2 * int(norms.numel()))
        for update in self.field_dependent_charges_maps:
            feat_a = self.electric_potential_descriptor(
                density[:, 0, :], positions, batch, cell=cell
            )
            feat_b = self.electric_potential_descriptor(
                density[:, 1, :], positions, batch, cell=cell
            )
            potential_features = torch.cat([feat_a / norms, feat_b / norms], dim=-1)
            out = update(
                node_attrs,
                mixed,
                potential_features,
                density.reshape(density.shape[0], -1),
            )
            density = density + out[:, :-2].reshape(density.shape)
            density = self.fukui(density, out[:, -2:], batch, total_charge, total_spin)

        le = self.local_electron_energy(
            per_layer[-1],
            potential_features,
            field_independent.reshape(density.shape[0], -1),
            density.reshape(density.shape[0], -1),
        )
        le_total = _scatter_sum(le, batch, num_graphs)
        if not self.add_local_electron_energy:
            le_total = torch.zeros_like(le_total)
        hartree = self.coulomb_energy(density.sum(dim=1), positions, batch, cell=cell)
        return e0 + inter_e + le_total + hartree

    @classmethod
    def _from_official(cls, official: dict[str, Any], **kwargs: Any) -> MacePolar:
        hidden = official.get("hidden_irreps")
        mlp = official.get("MLP_irreps")
        num_features = official.get("num_features")
        max_hidden_l = official.get("max_hidden_l")
        mlp_dim = official.get("mlp_dim")
        if hidden is not None:
            num_features, max_hidden_l = _parse_irreps(str(hidden), "hidden_irreps")
        if mlp is not None:
            mlp_dim, _ = _parse_irreps(str(mlp), "MLP_irreps")
        ctor = dict(
            atomic_numbers=official["atomic_numbers"],
            atomic_energies=official["atomic_energies"],
            r_max=official.get("r_max"),
            num_bessel=official.get("num_bessel"),
            num_polynomial_cutoff=official.get("num_polynomial_cutoff"),
            l_max=official.get("max_ell", official.get("l_max")),
            num_features=num_features,
            max_hidden_l=max_hidden_l,
            num_interactions=official.get("num_interactions"),
            correlation=official.get("correlation"),
            mlp_dim=mlp_dim,
            edge_channels=official.get("edge_channels"),
            radial_mlp=official.get("radial_MLP", official.get("radial_mlp")),
            scale=official.get("atomic_inter_scale", official.get("scale")),
            shift=official.get("atomic_inter_shift", official.get("shift")),
            num_recursion_steps=official.get("num_recursion_steps"),
            multipole_lmax=official.get("multipole_lmax"),
            multipole_sigma=official.get("multipole_sigma"),
            field_lmax=official.get("field_lmax"),
            field_widths=official.get("field_widths"),
            field_norms=official.get("field_norms"),
            add_local_electron_energy=official.get("add_local_electron_energy"),
            kspace_cutoff_factor=official.get("kspace_cutoff_factor"),
        )
        ctor.update(kwargs)
        return cls(**{k: v for k, v in ctor.items() if v is not None})
