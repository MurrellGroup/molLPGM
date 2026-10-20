"""Guard the link between the included weights, source quantiles and figure paths."""

from pathlib import Path
from statistics import NormalDist
import unittest

import numpy as np
import torch

from model import MarginalModel, integrate


class ReferenceTests(unittest.TestCase):
    def test_trained_weights_reproduce_displayed_paths(self):
        torch.set_num_threads(2)
        reference = Path(__file__).resolve().parents[1] / "reference"
        model = MarginalModel().eval()
        model.load_state_dict(
            torch.load(reference / "model.pt", weights_only=True, map_location="cpu")[
                "ema"
            ]
        )
        with np.load(reference / "switching-01.npz") as saved:
            source = np.array(
                [NormalDist().inv_cdf(q) for q in saved["source_quantiles"]],
                dtype=np.float32,
            )
            # Allow normal float32 variation across CPU kernels.
            actual = integrate(model, torch.from_numpy(source), saved["time"])
            np.testing.assert_allclose(actual, saved["generated"], atol=2e-5, rtol=2e-5)
            self.assertEqual(len(saved["endpoint_samples"]), 100000)
            self.assertTrue(np.all(np.diff(actual, axis=1) > 0))


if __name__ == "__main__":
    unittest.main()
