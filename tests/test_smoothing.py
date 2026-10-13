import unittest

from smoothing import OneEuroFilter, PointOneEuroFilter


class OneEuroFilterTests(unittest.TestCase):
    def test_first_value_initializes_without_offset(self):
        filter_ = OneEuroFilter()
        self.assertEqual(filter_.filter(0.25, 100), 0.25)

    def test_smooths_a_step_and_adapts_to_speed(self):
        fixed = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        adaptive = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        fixed.filter(0.0, 0)
        adaptive.filter(0.0, 0)

        fixed_value = fixed.filter(1.0, 10)
        adaptive_value = adaptive.filter(1.0, 10)

        self.assertGreater(fixed_value, 0.0)
        self.assertLess(fixed_value, 1.0)
        self.assertGreater(adaptive_value, fixed_value)

    def test_timestamps_must_increase(self):
        filter_ = OneEuroFilter()
        filter_.filter(1.0, 100)
        with self.assertRaises(ValueError):
            filter_.filter(2.0, 100)
        with self.assertRaises(ValueError):
            filter_.filter(2.0, 99)

    def test_reset_and_large_gap_reinitialize(self):
        filter_ = OneEuroFilter(max_gap_ms=100)
        filter_.filter(0.0, 0)
        self.assertEqual(filter_.filter(1.0, 101), 1.0)
        filter_.filter(0.0, 110)
        filter_.reset()
        self.assertEqual(filter_.filter(0.75, 1), 0.75)

    def test_point_filter_handles_both_coordinates(self):
        filter_ = PointOneEuroFilter()
        self.assertEqual(filter_.filter((0.2, 0.8), 10), (0.2, 0.8))
        x, y = filter_.filter((0.4, 0.6), 20)
        self.assertGreater(x, 0.2)
        self.assertLess(y, 0.8)


if __name__ == "__main__":
    unittest.main()
