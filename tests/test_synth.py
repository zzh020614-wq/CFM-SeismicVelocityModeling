"""Determinism and physical validity of the depth-domain model generator.

Reproducibility is the property the whole dataset rests on: the split is
leak-free only because ``sample_seed = seed_base + global_index`` determines
every random parameter of a sample.
"""
from __future__ import annotations

import unittest

import numpy as np

from cfm.synth.config import CLASS_TABLE, N_CLASSES, SynthCfg
from cfm.synth.generate import build_assignment
from cfm.synth.generator import ModelGenerator


def _small_cfg() -> SynthCfg:
    """A coarse grid, so the tests stay fast while exercising the same code."""
    cfg = SynthCfg()
    cfg.grid.nz, cfg.grid.nx = 64, 64
    return cfg


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.cfg = _small_cfg()
        self.gen = ModelGenerator(self.cfg)

    def test_same_seed_gives_identical_models(self):
        for class_id in CLASS_TABLE:
            a = self.gen.generate(class_id, 20260630)["v"]
            b = self.gen.generate(class_id, 20260630)["v"]
            np.testing.assert_array_equal(a, b, err_msg=f"class {class_id} is not deterministic")

    def test_different_seeds_give_different_models(self):
        for class_id in CLASS_TABLE:
            a = self.gen.generate(class_id, 1)["v"]
            b = self.gen.generate(class_id, 2)["v"]
            self.assertFalse(np.array_equal(a, b), f"class {class_id} ignores its seed")

    def test_shape_dtype_and_physical_bounds(self):
        g = self.cfg.grid
        for class_id in CLASS_TABLE:
            out = self.gen.generate(class_id, 100 + class_id)
            v = out["v"]
            self.assertEqual(v.shape, (g.nz, g.nx))
            self.assertEqual(v.dtype, np.float32)
            self.assertTrue(np.isfinite(v).all())
            self.assertGreaterEqual(float(v.min()), g.v_min)
            self.assertLessEqual(float(v.max()), g.v_max)
            self.assertEqual(out["class_id"], class_id)

    def test_salt_class_contains_a_high_velocity_body(self):
        """Class 4 must actually emplace salt, at least for most seeds."""
        hits = 0
        for seed in range(20):
            v = self.gen.generate(4, seed)["v"]
            if np.isclose(v, self.cfg.salt.salt_v).sum() > 20:
                hits += 1
        self.assertGreater(hits, 15, "class 4 rarely produced a salt body")

    def test_fault_class_has_sharper_lateral_steps_than_the_flat_class(self):
        """The fault operator runs after the blur, so its planes stay sharp."""
        def max_lateral_step(class_id):
            return max(float(np.abs(np.diff(self.gen.generate(class_id, s)["v"], axis=1)).max())
                       for s in range(5))

        self.assertGreater(max_lateral_step(3), max_lateral_step(0))


class TestClassAssignment(unittest.TestCase):
    def test_round_robin_is_balanced_and_shard_mixed(self):
        assign = build_assignment(103, list(range(N_CLASSES)))
        counts = np.bincount(assign, minlength=N_CLASSES)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        # Consecutive samples differ in class, so every shard is class-mixed.
        self.assertEqual(assign[:N_CLASSES], list(range(N_CLASSES)))

    def test_class_subset_is_respected(self):
        assign = build_assignment(20, [0, 3, 4])
        self.assertEqual(set(assign), {0, 3, 4})


if __name__ == "__main__":
    unittest.main()
