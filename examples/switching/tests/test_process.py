import unittest
import numpy as np
from process import Process, advance, next_jump, observations, paths, target


class ProcessTests(unittest.TestCase):
    def test_hazard_inversion(self):
        t = np.array([0.0, 0.2, 0.8])
        e = np.array([0.1, 0.2, 0.02])
        s = next_jump(t, np.ones(3), e)
        np.testing.assert_allclose(
            6 * ((1 - t) * (s - t) - (s - t) ** 2 / 2), e, rtol=1e-12
        )
        s = next_jump(t, -np.ones(3), e)
        np.testing.assert_allclose(6 * np.log((1 - t) / (1 - s)), e, rtol=1e-12)

    def test_endpoint_and_piecewise_solution(self):
        t = np.linspace(0, 1, 101)
        x, z, y, vertices = paths(t, 100, np.random.default_rng(9))
        np.testing.assert_allclose(x[-1], y, atol=1e-15)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.all(z[-1] == 1))
        for v, endpoint in zip(vertices, y):
            midpoint = (v[:-1, 0] + v[1:, 0]) / 2
            value = advance(v[:-1, 1], endpoint, v[:-1, 2], v[:-1, 0], midpoint)
            np.testing.assert_allclose(value, (v[:-1, 1] + v[1:, 1]) / 2, atol=1e-12)

    def test_observation_simulator_matches_independent_path_simulator(self):
        # Independent algorithms and seeds must give the same marginal moments.
        n = 50000
        obs, _ = observations(n, np.random.default_rng(18), times=0.5)
        x, z, y, _ = paths(np.array([0.0, 0.5, 1.0]), n, np.random.default_rng(19))
        self.assertLess(abs(obs[:, 0].mean() - x[1].mean()), 0.015)
        self.assertLess(abs(obs[:, 0].std() - x[1].std()), 0.015)
        self.assertLess(abs(obs[:, 2].mean() - (z[1] * y).mean()), 0.015)


if __name__ == "__main__":
    unittest.main()
