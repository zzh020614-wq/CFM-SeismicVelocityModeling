"""Well placement and extraction.

The key property is that the well condition written by stage 3 and the one the
dataloader rebuilds from the label are the same object. If they ever diverge,
the network trains on one well set and is evaluated against another.
"""
from __future__ import annotations

import unittest

import numpy as np

from cfm.conditions.wells import build_wells, interp_well_horizontal, well_columns, well_rng


class TestWellColumns(unittest.TestCase):
    def test_columns_are_inside_the_section_and_do_not_touch(self):
        """Widened well bands must stay in bounds and keep at least one column apart."""
        for width in (1, 2, 3, 5):
            cols = well_columns(256, 10, width, well_rng(7))
            left = (width - 1) // 2
            right = width - 1 - left
            self.assertEqual(len(cols), 10)
            self.assertGreaterEqual(int(cols.min()) - left, 0)
            self.assertLessEqual(int(cols.max()) + right, 255)
            # Sorted and separated by more than the band width, so no two bands merge.
            gaps = np.diff(np.sort(cols))
            self.assertTrue((gaps >= width + 1).all(),
                            f"width={width} produced touching bands: {cols}")

    def test_band_width_is_exact_for_even_widths(self):
        """An even width must not silently become width+1 columns."""
        target = np.zeros((16, 64), np.float32)
        _, mask = build_wells(target, n_wells=4, width=2, seed=3)
        runs = _column_runs(mask[0] > 0.5)
        self.assertEqual(runs, [2, 2, 2, 2])

    def test_too_many_wells_raises(self):
        """Asking for more wells than fit must fail loudly, not silently overlap."""
        with self.assertRaises(ValueError):
            well_columns(32, 10, 5, well_rng(0))

    def test_placement_is_seed_deterministic_and_sample_dependent(self):
        a = well_columns(256, 10, 3, well_rng(42))
        b = well_columns(256, 10, 3, well_rng(42))
        c = well_columns(256, 10, 3, well_rng(43))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))


class TestBuildWells(unittest.TestCase):
    def test_values_are_the_true_column_not_a_replicated_centre(self):
        """Inside a widened band every column carries its own true value."""
        rng = np.random.default_rng(0)
        target = rng.uniform(1500, 6000, size=(32, 128)).astype(np.float32)
        val, mask = build_wells(target, n_wells=4, width=3, seed=11)
        sel = mask > 0.5
        np.testing.assert_allclose(val[sel], target[sel])
        self.assertEqual(int(sel[0].sum()), 12)          # 4 wells x 3 columns
        # Outside the wells nothing is revealed.
        self.assertTrue((val[~sel] == 0).all())

    def test_rebuilt_wells_hold_true_values_of_the_label(self):
        """The dataloader replaces the stage-3 wells; the values must still be true.

        Depth-to-time conversion is vertical and per trace, so the time-domain
        log at trace c is exactly column c of the label. Rebuilding therefore
        stays consistent with the label no matter which columns are chosen.
        """
        from cfm.conditions.config import CondCfg
        from cfm.conditions.derive import derive_conditions, stack_cond

        rng = np.random.default_rng(1)
        v_depth = np.sort(rng.uniform(1600, 5200, size=(48, 96)), axis=0).astype(np.float32)
        tau_axis = np.linspace(0.0, 0.4, 48)
        seed = 20260630

        out = derive_conditions(v_depth, 16.0, tau_axis,
                                np.random.default_rng(seed), CondCfg())
        stored = stack_cond(out)
        target = out["target"]

        # What stage 3 wrote: single columns, values taken from the label.
        stored_sel = stored[2] > 0.5
        np.testing.assert_allclose(stored[1][stored_sel], target[stored_sel])

        # What the dataloader feeds instead: 4 wells x 3 columns, same property.
        val, mask = build_wells(target, n_wells=4, width=3, seed=seed)
        sel = mask > 0.5
        np.testing.assert_allclose(val[sel], target[sel])
        self.assertEqual(int(sel[0].sum()), 12)


class TestWellInterpolation(unittest.TestCase):
    def test_interpolation_preserves_the_well_values(self):
        target = np.tile(np.linspace(2000, 5000, 32, dtype=np.float32)[:, None], (1, 64))
        val, mask = build_wells(target, n_wells=4, width=1, seed=5)
        filled = interp_well_horizontal(val, mask)
        sel = mask > 0.5
        np.testing.assert_allclose(filled[sel], val[sel], rtol=1e-6)
        self.assertTrue((filled > 0).all(), "interpolation must leave no empty columns")

    def test_single_well_is_returned_unchanged(self):
        target = np.ones((8, 32), np.float32) * 3000.0
        val, mask = build_wells(target, n_wells=1, width=1, seed=0)
        np.testing.assert_array_equal(interp_well_horizontal(val, mask), val)


def _column_runs(row: np.ndarray) -> list[int]:
    """Lengths of the consecutive True runs in a 1-D boolean row."""
    runs, current = [], 0
    for flag in row:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


if __name__ == "__main__":
    unittest.main()
