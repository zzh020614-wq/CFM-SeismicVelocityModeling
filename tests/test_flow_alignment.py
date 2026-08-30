"""Batch alignment of target, spatial conditions, class label and seed under OT-CFM.

A fake OT matcher forces both a permutation and a repeated target index, so the
test needs neither torch, POT nor a GPU. This is the failure mode that silently
trains every sample against another sample's conditions.
"""
from __future__ import annotations

import unittest

import numpy as np

from tdcfm.training.flow import _sample_otcfm_batch, sample_flow_batch


class _FakeIndices(np.ndarray):
    """Adds the ``Tensor.to`` interface the code expects onto a NumPy index array."""

    def to(self, device=None, dtype=None):
        del device
        return np.asarray(self, dtype=dtype).view(_FakeIndices)


def _indices(values):
    return np.asarray(values, dtype=np.int64).view(_FakeIndices)


class _FakeTensor(np.ndarray):
    """Adds the minimal tensor interface used by the code under test."""

    @property
    def device(self):
        return "cpu"

    def new_tensor(self, values):
        return np.asarray(values, dtype=self.dtype).view(_FakeIndices)


def _tensor(values, dtype=np.float32):
    return np.asarray(values, dtype=dtype).view(_FakeTensor)


class _FakeOTFlowMatcher:
    def __init__(self, target_order):
        self.target_order = np.asarray(target_order, dtype=np.int64)

    def guided_sample_location_and_conditional_flow(self, x0, x1, y0=None, y1=None):
        del x0, y0
        order = self.target_order
        t = np.zeros(len(order), dtype=np.float32)
        xt = x1[order]
        ut = np.zeros_like(xt)
        return t, xt, ut, None, y1[order].view(_FakeIndices)


class _FakePlainFlowMatcher:
    def sample_location_and_conditional_flow(self, x0, x1):
        t = np.zeros(x1.shape[0], dtype=np.float32)
        return t, x1, x1 - x0


class TestFlowAlignment(unittest.TestCase):
    def setUp(self):
        # Each sample tags its target, condition and label with the same id, so
        # identity can be checked after reordering.
        self.x0 = _tensor(np.zeros((3, 1, 1, 1)))
        self.x1 = _tensor(np.arange(3).reshape(3, 1, 1, 1))
        self.cond = _tensor(
            np.broadcast_to(np.arange(3).reshape(3, 1, 1, 1), (3, 4, 1, 1)).copy())
        self.y = _tensor(np.arange(3), dtype=np.int64)
        self.seed = np.array([100, 101, 102], dtype=np.int64)

    def test_ot_reorders_cond_and_y_with_target(self):
        """A permuted, repeating target index must carry cond, y and seed with it."""
        fm = _FakeOTFlowMatcher(target_order=[2, 0, 2])
        _, xt, _, cond_aligned, y_aligned, sampled = _sample_otcfm_batch(
            fm, self.x0, self.x1, self.cond, self.y, _indices([0, 1, 2]))

        target_ids = xt[:, 0, 0, 0].astype(np.int64)
        cond_ids = cond_aligned[:, 0, 0, 0].astype(np.int64)
        seed_aligned = self.seed[sampled]
        np.testing.assert_array_equal(target_ids, [2, 0, 2])
        np.testing.assert_array_equal(cond_ids, target_ids)
        np.testing.assert_array_equal(y_aligned, target_ids)
        np.testing.assert_array_equal(seed_aligned, [102, 100, 102])

    def test_non_ot_keeps_original_condition_order(self):
        """The plain CFM branch does not resample x1, so nothing may be reordered."""
        fm = _FakePlainFlowMatcher()
        _, xt, _, cond_aligned, y_aligned, sampled = sample_flow_batch(
            fm, "icfm", self.x0, self.x1, self.cond, self.y)

        np.testing.assert_array_equal(xt, self.x1)
        np.testing.assert_array_equal(cond_aligned, self.cond)
        np.testing.assert_array_equal(y_aligned, self.y)
        np.testing.assert_array_equal(sampled, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
