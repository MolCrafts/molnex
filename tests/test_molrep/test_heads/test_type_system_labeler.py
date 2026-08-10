"""Tests for TypeSystemLabeler and MultiTypeHead (learnable-classical-ff-06)."""

from __future__ import annotations

import torch

from molrep.condensation import (
    Condenser,
    InteractionClass,
    TypeSystem,
    bond_default_criterion,
)
from molrep.heads import Labeler, MultiTypeHead, TypeHead, TypeSystemLabeler


class TestTypeSystemLabeler:
    def test_labeler_protocol_surface(self):
        ts = TypeSystem.from_prototypes(
            InteractionClass.BOND,
            [{"k": 300.0, "r0": 1.09}, {"k": 500.0, "r0": 1.5}],
            criterion=bond_default_criterion(),
        )
        labeler = TypeSystemLabeler(ts)
        assert isinstance(labeler, Labeler)
        assert labeler.num_types == 2
        assert set(labeler.type_map.keys()) == {0, 1}
        assert all(isinstance(v, str) for v in labeler.type_map.values())

    def test_label_params_in_range(self):
        ts = TypeSystem.from_prototypes(
            InteractionClass.BOND,
            [{"k": 300.0, "r0": 1.09}, {"k": 500.0, "r0": 1.5}],
            criterion=bond_default_criterion(),
        )
        labeler = TypeSystemLabeler(ts)
        params = {
            "k": torch.tensor([300.0, 500.0]),
            "r0": torch.tensor([1.09, 1.5]),
        }
        ids = labeler.label(params)
        assert ids.dtype == torch.long
        assert ids.tolist() == [0, 1]
        assert all(0 <= int(i) < labeler.num_types for i in ids)

    def test_label_from_condenser_assignment(self):
        result = Condenser().merge(
            [{"k": torch.tensor([300.0, 300.0]), "r0": torch.tensor([1.09, 1.09])}],
            interaction=InteractionClass.BOND,
            criterion=bond_default_criterion(),
        )
        labeler = TypeSystemLabeler(result.type_system)
        ids = labeler.label(result.assignment.type_ids)
        assert ids.tolist() == [0, 0]


class TestTypeHeadMultiSystem:
    def test_existing_type_head_api(self):
        head = TypeHead(hidden_dim=4, num_types=5)
        out = head(torch.ones(3, 4))
        assert out.shape == torch.Size([3, 5])
        idx = head.decode(out)
        assert idx.shape == torch.Size([3])
        idx2, conf = head.decode_with_confidence(out)
        assert idx2.shape == conf.shape == torch.Size([3])

    def test_from_type_system(self):
        ts = TypeSystem.from_prototypes(
            InteractionClass.BOND,
            [{"k": 1.0, "r0": 1.0}, {"k": 2.0, "r0": 1.1}],
        )
        head = TypeHead.from_type_system(8, ts)
        assert head.num_types == 2
        logits = head(torch.randn(4, 8))
        assert logits.shape == (4, 2)

    def test_multi_type_head_additive(self):
        bond_ts = TypeSystem.from_prototypes(
            InteractionClass.BOND,
            [{"k": 1.0, "r0": 1.0}],
        )
        angle_ts = TypeSystem.from_prototypes(
            InteractionClass.ANGLE,
            [{"k": 10.0, "theta0": 1.9}, {"k": 20.0, "theta0": 2.0}],
        )
        multi = MultiTypeHead.from_type_systems(
            4,
            {"bond": bond_ts, "angle": angle_ts},
        )
        assert multi.num_types == {"bond": 1, "angle": 2}
        logits = multi(
            {
                "bond": torch.randn(2, 4),
                "angle": torch.randn(3, 4),
            }
        )
        assert logits["bond"].shape == (2, 1)
        assert logits["angle"].shape == (3, 2)
        decoded = multi.decode(logits)
        assert decoded["bond"].shape == (2,)
