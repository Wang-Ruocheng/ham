import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("histogram_collector", Path(__file__).with_name("collect_histograms.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HistogramTests(unittest.TestCase):
    def test_up_and_checkerboard(self):
        np.testing.assert_array_equal(module.histograms(np.ones((2, 6, 6))), [0, 0, 0, 0, 72])
        board = 2 * (np.indices((6, 6)).sum(axis=0) % 2) - 1
        np.testing.assert_array_equal(module.histograms(board[None]), [36, 0, 0, 0, 0])

    def test_bruteforce_and_nll(self):
        spins = np.random.default_rng(913).choice([-1, 1], size=(3, 6, 6))
        products = []
        for s in spins:
            for i in range(6):
                for j in range(6):
                    field = s[(i+1) % 6, j] + s[(i-1) % 6, j] + s[i, (j+1) % 6] + s[i, (j-1) % 6]
                    products.append(s[i, j] * field)
        counts = module.histograms(spins)
        direct_counts = [products.count(t) for t in [-4, -2, 0, 2, 4]]
        np.testing.assert_array_equal(counts, direct_counts)
        np.testing.assert_array_equal(counts, module.histograms(-spins))
        for coupling in (-1.3, 0., .42, 1.7):
            expected = np.logaddexp(0, -2 * coupling * np.asarray(products)).mean()
            actual = (counts * np.logaddexp(0, -2 * coupling * np.arange(-4, 5, 2))).sum() / counts.sum()
            self.assertAlmostEqual(actual, expected, places=14)

    def test_invalid_inputs_rejected(self):
        for value in (np.zeros((2, 3, 3)), np.ones((2, 3)), np.full((2, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                module.histograms(value)


if __name__ == "__main__":
    unittest.main()
