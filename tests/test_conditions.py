"""Self-consistency of the derived conditions and of the shard round trip."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from cfm.conditions.config import COND_NAMES, CondCfg
from cfm.conditions.derive import derive_conditions, stack_cond
from cfm.conditions.dix import dix_forward
from cfm.conditions.seismic import reflectivity, ricker
from cfm.shards import ConditionShardWriter, VelocityShardWriter, read_velocity_shard
from cfm.timeconv.convert import depth_to_time


def _layered_depth_model(nz=48, nx=96, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(1700, 5200, size=(nz, nx)), axis=0).astype(np.float32)


class TestDeriveConditions(unittest.TestCase):
    def setUp(self):
        self.v_depth = _layered_depth_model()
        self.tau_axis = np.linspace(0.0, 0.5, 48)
        self.cfg = CondCfg()
        self.out = derive_conditions(self.v_depth, 16.0, self.tau_axis,
                                     np.random.default_rng(20260630), self.cfg)

    def test_target_equals_the_depth_to_time_conversion(self):
        """The label is the time-domain truth, nothing else."""
        expected = depth_to_time(self.v_depth, 16.0, self.tau_axis)
        np.testing.assert_array_equal(self.out["target"], expected)

    def test_channel_order_matches_cond_names(self):
        cond = stack_cond(self.out)
        self.assertEqual(cond.shape, (len(COND_NAMES),) + self.out["target"].shape)
        for i, name in enumerate(COND_NAMES):
            np.testing.assert_array_equal(cond[i], self.out[name])

    def test_rms_is_a_smoothed_dix_forward_model(self):
        """Smoothing must move the RMS field but not change its overall level."""
        raw = dix_forward(self.out["target"])
        self.assertFalse(np.allclose(self.out["rms"], raw))
        self.assertLess(abs(float(self.out["rms"].mean() - raw.mean())), 60.0)
        self.assertLess(float(self.out["rms"].std()), float(raw.std()))

    def test_conditions_are_finite_and_physical(self):
        for name in ("target", "rms", "well_val"):
            self.assertTrue(np.isfinite(self.out[name]).all(), name)
        self.assertTrue(np.isfinite(self.out["imaging"]).all())
        self.assertLessEqual(float(self.out["target"].max()), self.cfg.v_max)
        self.assertGreaterEqual(float(self.out["target"].min()), self.cfg.v_min)
        self.assertEqual(set(np.unique(self.out["well_mask"])), {0.0, 1.0})

    def test_derivation_is_deterministic_in_the_sample_seed(self):
        again = derive_conditions(self.v_depth, 16.0, self.tau_axis,
                                  np.random.default_rng(20260630), self.cfg)
        for name in COND_NAMES:
            np.testing.assert_array_equal(self.out[name], again[name])
        other = derive_conditions(self.v_depth, 16.0, self.tau_axis,
                                  np.random.default_rng(20260631), self.cfg)
        self.assertFalse(np.array_equal(self.out["imaging"], other["imaging"]))

    def test_stage3_wells_are_single_columns_in_range(self):
        """Stage 3 stores 3-8 single-column wells; the dataloader replaces them."""
        mask = self.out["well_mask"]
        cols = np.where((mask > 0.5).any(axis=0))[0]
        self.assertTrue(3 <= len(cols) <= 8)
        np.testing.assert_array_equal(np.sort(cols), self.out["well_cols"])
        sel = mask > 0.5
        np.testing.assert_allclose(self.out["well_val"][sel], self.out["target"][sel])


class TestRmsSmoothingComposition(unittest.TestCase):
    """The dataloader adds smoothing on top of the stage-3 baseline instead of
    regenerating the dataset. That is only valid because Gaussians compose as
    ``sigma_total^2 = sigma_base^2 + sigma_extra^2``; this test checks the claim
    numerically on a real RMS field.
    """

    def test_two_stage_smoothing_matches_one_pass(self):
        from scipy.ndimage import gaussian_filter

        # Use the production grid: the comparison has to stay clear of the
        # boundaries, and sigma_tau = 18 reaches roughly 3*sigma into the field.
        target = depth_to_time(_layered_depth_model(nz=256, nx=256),
                               16.0, np.linspace(0.0, 3.0, 256))
        raw = dix_forward(target)

        base = np.array([9.0, 4.0])
        total = np.array([18.0, 8.0])
        extra = np.sqrt(total ** 2 - base ** 2)

        one_pass = gaussian_filter(raw, tuple(total), mode="nearest")
        two_pass = gaussian_filter(gaussian_filter(raw, tuple(base), mode="nearest"),
                                   tuple(extra), mode="nearest")
        # Only the boundary handling differs, so compare away from the edges.
        interior = (slice(60, -60), slice(30, -30))
        residual = float(np.abs(one_pass[interior] - two_pass[interior]).max())
        self.assertLess(residual / float(raw[interior].mean()), 1e-3,
                        f"composition residual {residual:.2f} m/s is too large")


class TestSeismic(unittest.TestCase):
    def test_ricker_is_symmetric_and_zero_mean(self):
        w = ricker(25.0, 0.002)
        self.assertEqual(len(w) % 2, 1)
        np.testing.assert_allclose(w, w[::-1], atol=1e-12)
        self.assertLess(abs(float(w.mean())), 0.05)
        self.assertAlmostEqual(float(w.max()), float(w[len(w) // 2]))

    def test_reflectivity_sign_follows_the_velocity_contrast(self):
        v = np.array([[2000.0], [3000.0], [2000.0]], np.float32)
        r = reflectivity(v)
        self.assertGreater(r[0, 0], 0)          # velocity increase
        self.assertLess(r[1, 0], 0)             # velocity decrease
        self.assertEqual(r[2, 0], 0)            # last sample has no interface below


class TestShardRoundTrip(unittest.TestCase):
    def test_velocity_shards_read_back_identically(self):
        v = _layered_depth_model(nz=8, nx=8)
        with tempfile.TemporaryDirectory() as d:
            w = VelocityShardWriter(d, shard_size=2, fmt="npz")
            for i in range(3):
                w.add(v + i, class_id=i % 5, seed=1000 + i, gidx=i)
            shards = w.close()
            self.assertEqual([s["n"] for s in shards], [2, 1])
            vv, cc, ss = read_velocity_shard(os.path.join(d, shards[0]["file"]))
            np.testing.assert_allclose(vv[0], v)
            np.testing.assert_array_equal(cc, [0, 1])
            np.testing.assert_array_equal(ss, [1000, 1001])

    def test_condition_shards_preserve_target_and_cond(self):
        target = _layered_depth_model(nz=8, nx=8)
        cond = np.stack([target, target * 0, target * 0 + 1, target * 0.5])
        with tempfile.TemporaryDirectory() as d:
            w = ConditionShardWriter(d, shard_size=4, fmt="npz", cond_names=COND_NAMES)
            w.add(target, cond, class_id=2, seed=7, gidx=0)
            shards = w.close()
            data = np.load(os.path.join(d, shards[0]["file"]))
            np.testing.assert_allclose(data["target"][0], target)
            np.testing.assert_allclose(data["cond"][0], cond)
            self.assertEqual(int(data["class_id"][0]), 2)
            self.assertEqual(int(data["seed"][0]), 7)


if __name__ == "__main__":
    unittest.main()
