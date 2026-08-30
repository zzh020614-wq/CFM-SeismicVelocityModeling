"""Metrics, normalisation and configuration handling (all torch-free)."""
from __future__ import annotations

import unittest

import numpy as np

from tdcfm.evaluation.metrics import (
    aggregate_by_class,
    corr,
    dix_rms_consistency,
    mae,
    rmse,
    ssim,
    well_consistency,
)
from tdcfm.training.config import TrainCfg, active_cond_channels, apply_overrides, cfg_from_ckpt
from tdcfm.training.normalize import normalize_sample, unit_to_vel, vel_to_unit


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.a = rng.uniform(1500, 6000, size=(32, 32))

    def test_identical_fields_are_perfect(self):
        self.assertEqual(mae(self.a, self.a), 0.0)
        self.assertEqual(rmse(self.a, self.a), 0.0)
        self.assertAlmostEqual(corr(self.a, self.a), 1.0, places=6)
        self.assertAlmostEqual(ssim(self.a, self.a), 1.0, places=6)
        self.assertEqual(dix_rms_consistency(self.a, self.a), 0.0)

    def test_constant_offset_is_recovered_exactly(self):
        b = self.a + 100.0
        self.assertAlmostEqual(mae(self.a, b), 100.0, places=6)
        self.assertAlmostEqual(rmse(self.a, b), 100.0, places=6)
        self.assertAlmostEqual(corr(self.a, b), 1.0, places=6)

    def test_ssim_degrades_with_noise(self):
        rng = np.random.default_rng(1)
        noisy = self.a + rng.normal(0, 500, self.a.shape)
        self.assertLess(ssim(self.a, noisy), ssim(self.a, self.a + 1.0))

    def test_well_consistency_only_looks_at_masked_pixels(self):
        b = self.a.copy()
        mask = np.zeros_like(self.a)
        mask[:, 3] = 1.0
        b[:, 3] += 50.0                     # error at the well
        b[:, 10] += 5000.0                  # huge error away from the well
        self.assertAlmostEqual(well_consistency(b, self.a, mask), 50.0, places=6)

    def test_well_consistency_is_nan_without_wells(self):
        self.assertTrue(np.isnan(
            well_consistency(self.a, self.a, np.zeros_like(self.a))))

    def test_aggregate_by_class_splits_and_averages(self):
        records = [
            {"class_id": 0, "mae": 10.0}, {"class_id": 0, "mae": 20.0},
            {"class_id": 4, "mae": 60.0},
        ]
        out = aggregate_by_class(records, {0: "near_horizontal", 4: "salt"})
        self.assertAlmostEqual(out["overall"]["mae"], 30.0)
        self.assertAlmostEqual(out["per_class"][0]["mae"], 15.0)
        self.assertEqual(out["per_class"][0]["n"], 2)
        self.assertEqual(out["per_class"][4]["name"], "salt")

    def test_aggregate_ignores_nan_entries(self):
        records = [{"class_id": 0, "well": float("nan")}, {"class_id": 0, "well": 8.0}]
        out = aggregate_by_class(records, {0: "near_horizontal"})
        self.assertAlmostEqual(out["overall"]["well"], 8.0)


class TestNormalisation(unittest.TestCase):
    def test_velocity_round_trip(self):
        v = np.linspace(1500, 6000, 64)
        np.testing.assert_allclose(unit_to_vel(vel_to_unit(v, 1500, 6000), 1500, 6000),
                                   v, rtol=1e-6)

    def test_bounds_map_to_minus_one_and_one(self):
        self.assertAlmostEqual(float(vel_to_unit(np.array(1500.0), 1500, 6000)), -1.0)
        self.assertAlmostEqual(float(vel_to_unit(np.array(6000.0), 1500, 6000)), 1.0)

    def test_out_of_range_velocities_are_clipped(self):
        self.assertAlmostEqual(float(vel_to_unit(np.array(9000.0), 1500, 6000)), 1.0)

    def test_normalize_sample_shapes_and_ranges(self):
        rng = np.random.default_rng(0)
        target = rng.uniform(1500, 6000, size=(16, 16)).astype(np.float32)
        cond = np.stack([
            rng.uniform(1500, 6000, size=(16, 16)),      # rms
            np.zeros((16, 16)),                          # well values
            np.zeros((16, 16)),                          # well mask
            rng.normal(0, 0.1, size=(16, 16)),           # imaging
        ]).astype(np.float32)
        cond[1, :, 4] = target[:, 4]
        cond[2, :, 4] = 1.0

        t, c = normalize_sample(target, cond, 1500.0, 6000.0, well_interp=False)
        self.assertEqual(t.shape, (1, 16, 16))
        self.assertEqual(c.shape, (4, 16, 16))
        self.assertTrue((t >= -1).all() and (t <= 1).all())
        self.assertTrue((np.abs(c[3]) <= 1.0 + 1e-6).all())
        self.assertEqual(set(np.unique(c[2])), {0.0, 1.0})   # mask stays binary


class TestTrainCfg(unittest.TestCase):
    def test_overrides_cast_to_the_declared_type(self):
        cfg = apply_overrides(TrainCfg(), ["batch_size=8", "lr=1e-3", "use_imaging=False"])
        self.assertEqual(cfg.batch_size, 8)
        self.assertAlmostEqual(cfg.lr, 1e-3)
        self.assertIs(cfg.use_imaging, False)

    def test_boolean_false_is_not_read_as_true(self):
        """A non-empty string is truthy, so booleans need explicit handling."""
        for word in ("False", "false", "0", "no", "off"):
            self.assertIs(apply_overrides(TrainCfg(), [f"random_flip={word}"]).random_flip,
                          False)
        for word in ("True", "true", "1", "yes", "on"):
            self.assertIs(apply_overrides(TrainCfg(), [f"random_flip={word}"]).random_flip,
                          True)

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(KeyError):
            apply_overrides(TrainCfg(), ["nonexistent_field=1"])
        with self.assertRaises(ValueError):
            apply_overrides(TrainCfg(), ["batch_size"])

    def test_dropping_imaging_shrinks_the_input(self):
        self.assertEqual(active_cond_channels(TrainCfg()), 4)
        self.assertEqual(active_cond_channels(TrainCfg(use_imaging=False)), 3)

    def test_cfg_from_ckpt_ignores_unknown_keys(self):
        cfg = cfg_from_ckpt({"batch_size": 12, "some_removed_field": 5})
        self.assertEqual(cfg.batch_size, 12)

    def test_cfg_round_trips_through_a_dict(self):
        cfg = TrainCfg(n_wells=4, well_width=5, use_imaging=False)
        self.assertEqual(cfg_from_ckpt(cfg.to_dict()), cfg)


if __name__ == "__main__":
    unittest.main()
