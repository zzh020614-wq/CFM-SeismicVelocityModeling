"""Depth-to-time conversion and the closed loop back to depth."""
from __future__ import annotations

import unittest

import numpy as np

from cfm.timeconv.convert import (
    column_tau,
    column_tau_max,
    depth_to_time,
    make_tau_axis,
    scan_tmax,
    time_to_depth,
)


class TestDepthToTime(unittest.TestCase):
    def setUp(self):
        self.dz = 16.0
        rng = np.random.default_rng(0)
        # Velocity increasing with depth, as in a compacting sequence.
        self.v = np.sort(rng.uniform(1600, 5500, size=(64, 24)), axis=0).astype(np.float32)

    def test_tau_is_strictly_increasing_per_trace(self):
        """v > 0 makes tau monotonic, so the conversion cannot reorder layers."""
        for x in range(self.v.shape[1]):
            tau = column_tau(self.v[:, x], self.dz)
            self.assertTrue((np.diff(tau) > 0).all())

    def test_constant_velocity_maps_exactly(self):
        """A constant-velocity model must stay constant in time."""
        v = np.full((32, 4), 2500.0, np.float32)
        tau_axis = make_tau_axis(2.0 * 32 * 16.0 / 2500.0, 32)
        out = depth_to_time(v, 16.0, tau_axis)
        np.testing.assert_allclose(out, 2500.0, rtol=1e-5)

    def test_output_shape_and_range(self):
        tau_axis = make_tau_axis(float(column_tau_max(self.v, self.dz).min()), 128)
        out = depth_to_time(self.v, self.dz, tau_axis)
        self.assertEqual(out.shape, (128, self.v.shape[1]))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), float(self.v.min()) - 1e-3)
        self.assertLessEqual(float(out.max()), float(self.v.max()) + 1e-3)

    def test_beyond_tau_max_holds_the_deepest_velocity(self):
        """Past the true end of a trace, np.interp holds the last value."""
        t_max = float(column_tau_max(self.v, self.dz).max()) * 1.5
        tau_axis = make_tau_axis(t_max, 200)
        out = depth_to_time(self.v, self.dz, tau_axis)
        np.testing.assert_allclose(out[-1], self.v[-1], rtol=1e-5)

    def test_scan_tmax_returns_the_requested_percentile(self):
        t_max, all_tau = scan_tmax([self.v, self.v * 1.1], self.dz, percentile=90.0)
        self.assertEqual(all_tau.size, 2 * self.v.shape[1])
        self.assertAlmostEqual(t_max, float(np.percentile(all_tau, 90.0)))


class TestClosedLoop(unittest.TestCase):
    def test_round_trip_recovers_a_constant_model(self):
        """time_to_depth inverts depth_to_time for a constant-velocity model."""
        nz, nx, dz, v0 = 64, 8, 16.0, 3000.0
        v = np.full((nz, nx), v0, np.float32)
        tau_axis = make_tau_axis(2.0 * nz * dz / v0, nz)
        back = time_to_depth(depth_to_time(v, dz, tau_axis), tau_axis, dz, nz)
        np.testing.assert_allclose(back, v0, rtol=1e-4)

    def test_round_trip_stays_within_a_sample_of_the_layered_model(self):
        """A gradient model survives the round trip to within the discretisation offset.

        Both cumulative sums use the right-hand endpoint, which leaves a fixed
        sub-sample bias that does not vanish as ``n_t`` grows. The meaningful
        bound is therefore the velocity change over a couple of depth samples,
        not an absolute number of m/s.
        """
        nz, nx, dz = 128, 6, 16.0
        v = np.repeat(np.linspace(1800, 4800, nz, dtype=np.float32)[:, None], nx, axis=1)
        t_max = float(column_tau_max(v, dz).max())
        tau_axis = make_tau_axis(t_max, 256)
        back = time_to_depth(depth_to_time(v, dz, tau_axis), tau_axis, dz, nz)
        per_sample_contrast = (4800.0 - 1800.0) / (nz - 1)
        self.assertLess(float(np.abs(back[2:-2] - v[2:-2]).mean()),
                        2.0 * per_sample_contrast)


if __name__ == "__main__":
    unittest.main()
