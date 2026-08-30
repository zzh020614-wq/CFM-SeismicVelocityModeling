"""Unit tests for the Dix forward and inverse transforms.

Only the unsmoothed chain is checked::

    target -> dix_forward -> rms -> dix_inverse -> recovered target

The heavy RMS smoothing applied in the real pipeline destroys information on
purpose and is deliberately outside the scope of an exact round-trip test.
"""
from __future__ import annotations

import unittest

import numpy as np

from cfm.conditions.dix import dix_forward, dix_inverse


class TestDix(unittest.TestCase):
    def setUp(self):
        # (n_t, nx): rows are samples on the common time axis, columns are traces.
        # The velocities must vary; a constant model would also pass under an
        # off-by-one discretisation error.
        self.target = np.array(
            [
                [2000.0, 1800.0],
                [4000.0, 2600.0],
                [2500.0, 5000.0],
                [5200.0, 3100.0],
            ],
            dtype=np.float32,
        )
        self.tau_axis = np.linspace(0.0, 0.012, self.target.shape[0])

    def test_forward_matches_manual_second_sample(self):
        """The RMS accumulates along tau: sample 1 is the RMS of the first two layers."""
        rms = dix_forward(self.target)
        expected = np.sqrt((2000.0 ** 2 + 4000.0 ** 2) / 2.0)
        np.testing.assert_allclose(rms[1, 0], expected, rtol=1e-6, atol=1e-3)

    def test_forward_inverse_round_trip_recovers_target(self):
        """The unsmoothed RMS inverts back to the multi-trace, varying target."""
        recovered = dix_inverse(dix_forward(self.target), self.tau_axis)
        np.testing.assert_allclose(recovered, self.target, rtol=1e-5, atol=1e-2)

    def test_output_shape_dtype_and_values_are_valid(self):
        """The result must be storable under the stage-3 conventions."""
        recovered = dix_inverse(dix_forward(self.target), self.tau_axis)
        self.assertEqual(recovered.shape, self.target.shape)
        self.assertEqual(recovered.dtype, np.float32)
        self.assertTrue(np.isfinite(recovered).all())

    def test_inverse_clips_unphysical_differences(self):
        """A violently varying RMS must not produce NaNs or out-of-range velocities."""
        rms = np.array(
            [
                [3000.0, 3000.0],
                [1500.0, 6000.0],
                [1500.0, 6000.0],
            ],
            dtype=np.float32,
        )
        tau_axis = np.linspace(0.0, 0.008, rms.shape[0])
        interval = dix_inverse(rms, tau_axis)
        self.assertTrue(np.isfinite(interval).all())
        self.assertGreaterEqual(float(interval.min()), 1500.0)
        self.assertLessEqual(float(interval.max()), 6000.0)

    def test_inverse_rejects_tau_axis_length_mismatch(self):
        """A time axis of the wrong length must fail loudly rather than broadcast."""
        rms = dix_forward(self.target)
        with self.assertRaises(ValueError):
            dix_inverse(rms, self.tau_axis[:-1])


if __name__ == "__main__":
    unittest.main()
