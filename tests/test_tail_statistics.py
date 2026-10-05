"""Offline regression tests for the actual empirical upper-tail functions.

Run with Python -B. No raw production records are accessed or files written.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from protocol import tail_mean
from recorded_band_decomposition import top_tail_indices
from tail_statistics import upper_tail_count
from public_statistics_audit import independent_upper_tail_count


class TailStatisticsTests(unittest.TestCase):
    def assert_tail(self, n: int, expected_k: int) -> None:
        values = np.arange(n, dtype=float)
        self.assertEqual(upper_tail_count(n), expected_k)
        indices = top_tail_indices(values)
        self.assertEqual(len(indices), expected_k)
        np.testing.assert_array_equal(indices, np.arange(n - expected_k, n))
        if n:
            self.assertEqual(tail_mean(values), float(values[-expected_k:].mean()))
        else:
            self.assertTrue(np.isnan(tail_mean(values)))

    def test_exact_boundaries(self) -> None:
        for n, expected in [(20, 1), (40, 2), (560, 28), (100, 5), (2000, 100)]:
            with self.subTest(n=n):
                self.assert_tail(n, expected)

    def test_nonmultiples_and_short_samples(self) -> None:
        for n, expected in [(0, 0), (1, 1), (19, 1), (21, 2), (39, 2), (41, 3), (559, 28), (561, 29)]:
            with self.subTest(n=n):
                self.assert_tail(n, expected)

    def test_ties_keep_deterministic_cardinality(self) -> None:
        values = np.ones(40)
        np.testing.assert_array_equal(top_tail_indices(values), [38, 39])
        self.assertEqual(tail_mean(values), 1.0)
        values[-4:] = 2.0
        np.testing.assert_array_equal(top_tail_indices(values), [38, 39])
        self.assertEqual(tail_mean(values), 2.0)

    def test_zeros_do_not_expand_tail(self) -> None:
        values = np.zeros(20)
        np.testing.assert_array_equal(top_tail_indices(values), [19])
        self.assertEqual(tail_mean(values), 0.0)
        values[-1] = 10.0
        self.assertEqual(tail_mean(values), 10.0)

    def test_mean_counts_only_finite_observations(self) -> None:
        values = np.r_[np.arange(20, dtype=float), np.nan, np.inf, -np.inf]
        self.assertEqual(tail_mean(values), 19.0)
        self.assertTrue(np.isnan(tail_mean([np.nan, np.inf])))

    def test_supported_quantiles(self) -> None:
        values = np.arange(20, dtype=float)
        self.assertEqual(tail_mean(values, 0.0), 9.5)
        np.testing.assert_array_equal(top_tail_indices(values, 0.0), np.arange(20))
        self.assertEqual(tail_mean(values, 0.5), 14.5)
        self.assertEqual(upper_tail_count(560, Decimal("0.95")), 28)
        self.assertEqual(upper_tail_count(20, "0.95"), 1)

    def test_invalid_quantiles_are_rejected_even_for_empty_data(self) -> None:
        for q in [-0.01, 1.0, 1.01, np.nan, np.inf, -np.inf, "invalid"]:
            for call in [lambda: upper_tail_count(0, q),
                         lambda: tail_mean([], q),
                         lambda: top_tail_indices(np.array([]), q)]:
                with self.subTest(q=q), self.assertRaises(ValueError):
                    call()

    def test_invalid_counts(self) -> None:
        with self.assertRaises(ValueError):
            upper_tail_count(-1)
        for n in [1.5, True, "20"]:
            with self.subTest(n=n), self.assertRaises(TypeError):
                upper_tail_count(n)

    def test_independent_rational_recalculation(self) -> None:
        for q in [0.0, 0.1, 0.25, 0.5, 0.95, 0.99, 0.999]:
            for n in range(2001):
                self.assertEqual(upper_tail_count(n, q), independent_upper_tail_count(n, q))


if __name__ == "__main__":
    unittest.main(verbosity=2)
