"""Unit tests for source-aware Class-I losses."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from molpy import Block, Frame
from scripts.mm_param_learning.class_i_fit import (
    _FIT_SCHEMA,
    ClassIFit,
    _index_omol_source,
    _require_group,
    energy_mse,
    fit_atomic_reference,
    label_sample,
    split_records,
    val_omol_mae,
)
from scripts.mm_param_learning.espaloma_mm_protocol_train import enumerate_impropers

from molix.batch.collate import collate_molecules


class TestOfficialSourceCache:
    def test_same_role_reuses_loaded_instance(self, monkeypatch):
        """A second load for one role must not construct a second LMDB env."""
        import scripts.mm_param_learning.class_i_fit as mod

        monkeypatch.setitem(mod._OFFICIAL_SOURCE_CACHE, "4m", None)

        class FakeOMol25:
            instances = 0

            def __init__(self, *, role):
                self.role = role

            def load(self):
                FakeOMol25.instances += 1
                return f"loaded-{self.role}"

        monkeypatch.setattr(
            "molhub.dataset.omol25.OMol25",
            lambda *, role, max_rows=None, hub=None: FakeOMol25(role=role),
        )
        first = mod._official_source("4m")
        second = mod._official_source("4m")
        assert first == second == "loaded-4m"
        assert FakeOMol25.instances == 1



class TestEnergyMse:
    def test_energy_mse_is_nonzero_for_one_geometry(self):
        pred = torch.tensor([10.0])
        ref = torch.tensor([4.0])
        assert float(energy_mse(pred, ref)) == 36.0


class TestFitAtomicReference:
    def test_recovers_per_element_table(self) -> None:
        def _mol(name: str, z: list[int], energy: float) -> Frame:
            n = len(z)
            atoms = Block()
            atoms["number"] = np.array(z, dtype=np.int64)
            atoms["x"] = np.zeros(n, dtype=np.float64)
            atoms["y"] = np.zeros(n, dtype=np.float64)
            atoms["z"] = np.zeros(n, dtype=np.float64)
            frame = Frame()
            frame["atoms"] = atoms
            frame.meta["energy"] = energy
            frame.meta["molecule_id"] = name
            return frame

        # ε[H]=2, ε[O]=5 → H2 = 4, water = 9 (H counts differ, so the system is full rank)
        h2 = _mol("h2", [1, 1], 4.0)
        water = _mol("water", [8, 1, 1], 9.0)
        table = fit_atomic_reference([(h2,), (water,)])
        assert torch.allclose(table(torch.tensor([1, 8])), torch.tensor([2.0, 5.0]), atol=1e-5)


class TestRequireGroup:
    def test_missing_identity_raises(self) -> None:
        frame = Frame()
        frame["atoms"] = Block()
        frame.meta["energy"] = 0.0
        with pytest.raises(ValueError, match="molecule_id"):
            _require_group(frame)

    def test_reads_molecule_id(self) -> None:
        frame = Frame()
        frame["atoms"] = Block()
        frame.meta["molecule_id"] = "mol-7"
        assert _require_group(frame) == "mol-7"


class TestLabelSample:
    def test_water_and_peroxide_collate(self) -> None:
        def _geom(
            name: str,
            z: list[int],
            xyz: list[list[float]],
            bonds: list[tuple[int, int]],
        ) -> Frame:
            n = len(z)
            atoms = Block()
            atoms["element"] = np.array(["O" if v == 8 else "H" for v in z], dtype="U2")
            atoms["number"] = np.array(z, dtype=np.int64)
            pos = np.array(xyz, dtype=np.float64)
            atoms["x"], atoms["y"], atoms["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
            atoms["fx"] = np.zeros(n, dtype=np.float64)
            atoms["fy"] = np.zeros(n, dtype=np.float64)
            atoms["fz"] = np.zeros(n, dtype=np.float64)
            blk = Block()
            blk["atomi"] = np.array([a for a, _ in bonds], dtype=np.int64)
            blk["atomj"] = np.array([b for _, b in bonds], dtype=np.int64)
            frame = Frame()
            frame["atoms"] = atoms
            frame["bonds"] = blk
            frame.meta["energy"] = 0.0
            frame.meta["molecule_id"] = name
            return frame

        water = _geom(
            "water",
            [8, 1, 1],
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            [(0, 1), (0, 2)],
        )
        hooh = _geom(
            "hooh",
            [8, 8, 1, 1],
            [[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [-0.5, 0.9, 0.0], [1.95, -0.9, 0.0]],
            [(0, 1), (0, 2), (1, 3)],
        )
        samples = [label_sample(water, group=0.0), label_sample(hooh, group=1.0)]
        assert samples[0]["propers"]["atomi"].numel() == 0
        assert samples[1]["propers"]["atomi"].numel() > 0
        assert samples[0]["impropers"]["atomi"].numel() == 0
        assert samples[1]["impropers"]["atomi"].numel() == 0
        batch = collate_molecules(samples, _FIT_SCHEMA)
        n_propers = int(samples[1]["propers"]["atomi"].shape[0])
        assert int(batch["propers", "atomi"].shape[0]) == n_propers
        assert int(batch["impropers", "atomi"].shape[0]) == 0

    def test_bondless_frame_collates_with_water(self) -> None:
        water = Frame()
        atoms = Block()
        atoms["element"] = np.array(["O", "H", "H"], dtype="U2")
        atoms["number"] = np.array([8, 1, 1], dtype=np.int64)
        atoms["x"] = np.array([0.0, 0.96, -0.24], dtype=np.float64)
        atoms["y"] = np.array([0.0, 0.0, 0.93], dtype=np.float64)
        atoms["z"] = np.zeros(3, dtype=np.float64)
        bonds = Block()
        bonds["atomi"] = np.array([0, 0], dtype=np.int64)
        bonds["atomj"] = np.array([1, 2], dtype=np.int64)
        water["atoms"] = atoms
        water["bonds"] = bonds
        water.meta["energy"] = 0.0
        water.meta["molecule_id"] = "w"
        lone = Frame()
        one = Block()
        one["element"] = np.array(["Ne"], dtype="U2")
        one["number"] = np.array([10], dtype=np.int64)
        one["x"] = np.array([0.0], dtype=np.float64)
        one["y"] = np.array([0.0], dtype=np.float64)
        one["z"] = np.array([0.0], dtype=np.float64)
        lone["atoms"] = one
        lone.meta["energy"] = 0.0
        lone.meta["molecule_id"] = "ne"
        samples = [label_sample(water, group=0.0), label_sample(lone, group=1.0)]
        assert "angles" in samples[1]
        assert samples[1]["angles"]["atomi"].numel() == 0
        batch = collate_molecules(samples, _FIT_SCHEMA)
        assert int(batch["angles", "atomi"].shape[0]) == int(samples[0]["angles"]["atomi"].shape[0])


class TestEnumerateImpropers:
    def test_trivalent_center_emits_one_row(self):
        atomi = np.array([0, 0, 0], dtype=np.int64)
        atomj = np.array([1, 2, 3], dtype=np.int64)
        ii, jj, kk, ll = enumerate_impropers(atomi, atomj, 4)
        assert ii.tolist() == [0]
        assert sorted((int(jj[0]), int(kk[0]), int(ll[0]))) == [1, 2, 3]

    def test_divalent_center_emits_nothing(self):
        atomi = np.array([0, 0], dtype=np.int64)
        atomj = np.array([1, 2], dtype=np.int64)
        ii, _, _, _ = enumerate_impropers(atomi, atomj, 3)
        assert ii.size == 0


class TestForceAwareEval:
    def test_no_grad_still_returns_forces(self) -> None:
        torch.manual_seed(0)
        atoms = Block()
        atoms["element"] = np.array(["O", "H", "H"], dtype="U2")
        atoms["number"] = np.array([8, 1, 1], dtype=np.int64)
        atoms["x"] = np.array([0.0, 0.96, -0.24], dtype=np.float64)
        atoms["y"] = np.array([0.0, 0.0, 0.93], dtype=np.float64)
        atoms["z"] = np.zeros(3, dtype=np.float64)
        atoms["fx"] = np.array([0.1, -0.05, -0.05], dtype=np.float64)
        atoms["fy"] = np.zeros(3, dtype=np.float64)
        atoms["fz"] = np.zeros(3, dtype=np.float64)
        bonds = Block()
        bonds["atomi"] = np.array([0, 0], dtype=np.int64)
        bonds["atomj"] = np.array([1, 2], dtype=np.int64)
        frame = Frame()
        frame["atoms"] = atoms
        frame["bonds"] = bonds
        frame.meta["energy"] = -76.0
        frame.meta["molecule_id"] = "w"
        fit = ClassIFit(
            [(frame,)],
            [(frame,)],
            hidden=16,
            n_blocks=1,
            n_torsion_terms=2,
            device="cpu",
            epochs=1,
            energy_scope="full",
        )
        sample = label_sample(frame, group=0.0)
        batch = collate_molecules([sample], _FIT_SCHEMA)
        model = fit.model()
        with torch.no_grad():
            out = model(batch)
        assert "forces" in out
        assert out["forces"].shape == (3, 3)
        metric = val_omol_mae(out, batch)
        assert float(metric) > 0.0


class TestRunWritesMolRecWal:
    def _water(self, molecule_id: str) -> tuple:
        atoms = Block()
        atoms["number"] = np.array([8, 1, 1], dtype=np.int64)
        atoms["x"] = np.array([0.0, 0.96, -0.24], dtype=np.float64)
        atoms["y"] = np.array([0.0, 0.0, 0.93], dtype=np.float64)
        atoms["z"] = np.zeros(3, dtype=np.float64)
        atoms["fx"] = np.zeros(3, dtype=np.float64)
        atoms["fy"] = np.zeros(3, dtype=np.float64)
        atoms["fz"] = np.zeros(3, dtype=np.float64)
        bonds = Block()
        bonds["atomi"] = np.array([0, 0], dtype=np.int64)
        bonds["atomj"] = np.array([1, 2], dtype=np.int64)
        frame = Frame()
        frame["atoms"] = atoms
        frame["bonds"] = bonds
        frame.meta["energy"] = -76.0
        frame.meta["molecule_id"] = molecule_id
        return (frame,)

    def test_run_writes_metrics_and_status(self, tmp_path):
        import json

        torch.manual_seed(0)
        train = [self._water("w0"), self._water("w1")]
        val = [self._water("v0")]
        fit = ClassIFit(
            train,
            val,
            hidden=16,
            n_blocks=1,
            n_torsion_terms=2,
            device="cpu",
            epochs=1,
            energy_scope="full",
            record_root=tmp_path,
        )
        fit.run(max_steps=1)

        assert (tmp_path / "metrics" / "metrics.jsonl").exists()
        assert (tmp_path / "status" / "status.json").exists()
        status = json.loads((tmp_path / "status" / "status.json").read_text())
        assert status["state"] == "succeeded"


class TestSplitCoverage:
    def test_overlap_raises(self):
        frame = Frame()
        frame["atoms"] = Block()
        frame["atoms"]["number"] = [6]
        frame["atoms"]["x"] = [0.0]
        frame["atoms"]["y"] = [0.0]
        frame["atoms"]["z"] = [0.0]
        frame.meta["molecule_id"] = "same"
        rec = (frame,)
        from scripts.mm_param_learning.class_i_fit import split_coverage

        with pytest.raises(ValueError, match="overlap"):
            split_coverage([rec], [rec])

    def test_manifest_has_no_overlap(self, tmp_path):
        from scripts.mm_param_learning.class_i_fit import split_coverage, write_split_manifest

        records = []
        for i in range(10):
            frame = Frame()
            atoms = Block()
            atoms["number"] = [6, 1]
            atoms["x"] = [0.0, 1.0]
            atoms["y"] = [0.0, 0.0]
            atoms["z"] = [0.0, 0.0]
            frame["atoms"] = atoms
            frame.meta["molecule_id"] = f"m{i}"
            frame.meta["energy"] = 0.0
            records.append((frame,))
        train, val = split_records(records, train_frac=0.8, seed=2666)
        path = write_split_manifest(tmp_path / "split.json", train, val)
        cov = split_coverage(train, val)
        assert cov["overlap"] == 0
        assert cov["n_train_molecules"] == 8
        assert cov["n_val_molecules"] == 2
        assert path.is_file()


class TestSplitRecords:
    def test_eight_two_no_overlap(self) -> None:
        records = []
        for i in range(10):
            frame = Frame()
            frame["atoms"] = Block()
            frame.meta["molecule_id"] = f"m{i}"
            frame.meta["energy"] = 0.0
            records.append((frame,))
        train, val = split_records(records, train_frac=0.8, seed=2666)
        assert len(train) == 8
        assert len(val) == 2
        train_ids = {_require_group(rec[0]) for rec in train}
        val_ids = {_require_group(rec[0]) for rec in val}
        assert train_ids.isdisjoint(val_ids)
        again, _ = split_records(records, train_frac=0.8, seed=2666)
        assert [_require_group(rec[0]) for rec in again] == [
            _require_group(rec[0]) for rec in train
        ]


class TestIndexOmolSource:
    def test_keeps_small_drops_huge(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("MOLHUB_HOME", str(tmp_path))

        def _rec(n: int, energy: float) -> tuple:
            atoms = Block()
            atoms["number"] = np.ones(n, dtype=np.int64)
            atoms["element"] = np.array(["H"] * n, dtype="U2")
            frame = Frame()
            frame["atoms"] = atoms
            frame.meta["energy"] = energy
            return (frame,)

        class _Src:
            source_id = "fake-omol"
            root = tmp_path / "fake"

            def __len__(self) -> int:
                return 3

            def __getitem__(self, idx: int):
                return [_rec(2, 4.0), _rec(8, 16.0), _rec(100, 200.0)][idx]

        keep, counts, e0 = _index_omol_source(_Src(), max_atoms=32)
        assert keep == [0, 1]
        assert counts == [2, 8]
        assert e0.shape == (95,)
        again, counts2, _ = _index_omol_source(_Src(), max_atoms=32)
        assert again == keep
        assert counts2 == counts
