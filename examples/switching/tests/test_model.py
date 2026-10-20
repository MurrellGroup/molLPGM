import unittest
import numpy as np
import torch
from model import MarginalModel, integrate


class LinearVelocity(torch.nn.Module):
    def forward(self, xt):
        return -0.7 * xt[:, 0] + 0.3


class IntegratorTests(unittest.TestCase):
    def test_zero_learned_velocity_does_not_transport_to_target(self):
        model = MarginalModel().eval()
        self.assertEqual(model.net[0].in_features, 2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        x0 = torch.tensor([-2.0, 0.1, 3.0])
        actual = integrate(model, x0, np.linspace(0, 1, 17))
        np.testing.assert_array_equal(actual, np.broadcast_to(x0.numpy(), actual.shape))

    def test_linear_ode_against_analytic_solution(self):
        x0 = torch.tensor([-2.0, 0.1, 3.0], dtype=torch.float64)
        times = np.linspace(0, 1, 65)
        actual = integrate(LinearVelocity(), x0, times)
        exact = 0.3 / 0.7 + (x0.numpy()[None, :] - 0.3 / 0.7) * np.exp(
            -0.7 * times[:, None]
        )
        np.testing.assert_allclose(actual, exact, atol=1e-9, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
