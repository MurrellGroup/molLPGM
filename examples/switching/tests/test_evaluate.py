import unittest
import numpy as np
from evaluate import compare


class DistributionMetrics(unittest.TestCase):
    def test_identical_empirical_distribution(self):
        values = np.array([-3.0, -1.0, -1.0, 0.0, 2.0, 4.0])
        result = compare(values[::-1], values)
        self.assertEqual(result["wasserstein_1"], 0.0)
        self.assertEqual(result["ks_distance"], 0.0)

    def test_translation_has_known_wasserstein_distance(self):
        reference = np.linspace(-1.0, 1.0, 1001)
        result = compare(reference + 3.0, reference)
        self.assertAlmostEqual(result["wasserstein_1"], 3.0)
        self.assertEqual(result["ks_distance"], 1.0)

    def test_nonfinite_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            compare([0.0, float("nan")], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
